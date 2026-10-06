# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The thresholding kernel: its result type, its table, its configuration and its refusals."""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import DefinitionError, Rejected, Unresolved
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.artifacts.abi import Bus
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import ThresholdTable
from kernels.helpers import FULL_DSP48E2, THRESHOLD_TABLE, controlled, threshold_base

AUTO_MEMORY: dict[str, object] = {"ram_style": "auto", "ultra_stages": 0}


def threshold(
    *,
    table: ThresholdTable = THRESHOLD_TABLE,
    pe: int | None = 1,
    bias: int = -1,
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT5",
    memory: dict[str, object] | None = None,
    axilite: bool = False,
    deep: bool = False,
) -> ThresholdingAxiKernel:
    """PE is committed as a choice; ``None`` leaves it open (a table without channels has none)."""
    base = threshold_base(
        table=table,
        bias=bias,
        input_dtype=input_dtype,
        threshold_dtype=threshold_dtype,
    )
    factors: dict[str, object] = {} if pe is None else {"pe": pe}
    if pe is not None:
        # The memories, like PE, range over the table (its thresholds' stages).
        factors |= AUTO_MEMORY if memory is None else memory
    if axilite:
        # Runtime-writable thresholds present their bus through a control node.
        facts = dict(
            input_dtype=resolve_qonnx_datatype_name(input_dtype),
            threshold_dtype=resolve_qonnx_datatype_name(threshold_dtype),
            thresholds=table,
            bias=bias,
            platform=FULL_DSP48E2,
        )
        return controlled(
            ThresholdingAxiKernel, facts, use_axilite=True, deep_pipeline=deep, **factors
        )
    report = base.try_with_choices(use_axilite=axilite, deep_pipeline=deep, **factors)
    assert report.accepted
    return report.instance


def test_threshold_states_its_result_type_initial_table_and_configuration() -> None:
    point = threshold()
    requirements = point.module
    assert point.result_dtype.name == "INT3"
    assert (
        dict(requirements.parameters)["THRESHOLDS"]
        == "'{'{'{5'h1e, 5'h0, 5'h3}, '{5'h1f, 5'h1, 5'h4}}}"
    )
    assert threshold(bias=0).result_dtype.name == "UINT2"
    assert threshold(bias=-4).result_dtype.name == "INT3"
    narrow_negative = threshold(bias=-5)
    assert narrow_negative.result_dtype.name == "INT33"
    assert isinstance(
        narrow_negative.inspect(ThresholdingAxiKernel.module).accepted_result, Rejected
    )
    enabled = threshold(axilite=True, deep=True).module
    assert dict(enabled.parameters)["USE_AXILITE"] == 1
    assert dict(enabled.parameters)["DEEP_PIPELINE"] == 1
    assert dict(threshold(pe=2).module.parameters)["PE"] == 2
    config = next(
        port for port in requirements.abi.pins if isinstance(port, Bus) and port.name == "s_axilite"
    )
    assert {
        signal.width for signal in config.signals if signal.logical in ("awaddr", "araddr")
    } == {5}


def test_threshold_sets_present_a_selector_bus_and_refuse_axilite_addressing() -> None:
    table: ThresholdTable = (((-2, 0, 3), (-1, 1, 4)), ((-3, 0, 5), (-2, 0, 6)))
    point = threshold(table=table)
    requirements = point.module
    assert dict(requirements.parameters)["SETS"] == 2
    selector = next(
        port
        for port in requirements.abi.pins
        if isinstance(port, Bus) and port.name == "s_axis_set"
    )
    assert next(signal.width for signal in selector.signals if signal.logical == "tdata") == 8
    assert isinstance(
        threshold(table=table, axilite=True).inspect(ThresholdingAxiKernel.module).accepted_result,
        Rejected,
    )


def test_threshold_partial_dtype_query_does_not_adopt_implementation_decisions() -> None:
    base = threshold_base()
    assert base.result_dtype.name == "INT3"
    assert isinstance(base.query(ThresholdingAxiKernel.use_axilite), Unresolved)
    assert isinstance(base.query(ThresholdingAxiKernel.deep_pipeline), Unresolved)
    assert isinstance(base.inspect(ThresholdingAxiKernel.module).accepted_result, Unresolved)


def test_threshold_rejects_unsupported_profiles_and_malformed_tables() -> None:
    profiles = (
        threshold(table=(), pe=None),
        threshold(table=(((2, 1),),)),
        threshold(table=(((0, 20),),)),
        threshold(table=(((0,), (0, 1)),)),
        threshold(threshold_dtype="UINT5"),
        threshold(bias=1 << 31),
        threshold(bias=-10),
        threshold(memory={"ram_style": "distributed", "block_stages": 2, "ultra_stages": 1}),
    )
    assert all(
        isinstance(point.inspect(ThresholdingAxiKernel.module).accepted_result, Rejected)
        for point in profiles
    )
    with pytest.raises(DefinitionError, match="threshold table"):
        threshold(table=cast(ThresholdTable, (([-2, 0, 3],),)))
    # PE is a divisor of the table's channels: another is refused where it is committed.
    for pe in (0, 3, 4):
        assert not threshold_base().try_with_choices(pe=pe).accepted
