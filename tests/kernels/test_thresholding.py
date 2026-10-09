# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The thresholding kernel: its result type, its table, its configuration and its refusals."""

from __future__ import annotations

from typing import cast

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Available, DefinitionError, Rejected, Unresolved, design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Bus
from finn.kernels.channels import Channel
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import ThresholdTable
from kernels.helpers import FULL_DSP48E2, THRESHOLD_TABLE, Root, controlled, threshold_base

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
    """PE is committed as a choice; ``None`` leaves it open (a table without rows has none)."""
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
    # The table is THRESHOLDS_FILE, named by its contents: per row fold (C / PE = 2),
    # the row's thresholds padded to a power of two, WT = 5 bits each.
    initial = point.thresholds_file
    assert initial.data == b"1e\n00\n03\n00\n1f\n01\n04\n00\n"
    assert initial.path.startswith("thresholds_") and initial.path.endswith(".dat")
    assert dict(requirements.parameters)["THRESHOLDS_FILE"] == f'"{initial.path}"'
    assert dict(requirements.parameters)["THRESHOLDS"] == "'{default: '0}"
    assert initial in requirements.data
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
    # Flat, PE is any the RTL takes with the table's two rows (one a multiple of the
    # other); another is refused where it is committed.
    for pe in (0, 3, 5):
        assert not threshold_base().try_with_choices(pe=pe).accepted
    assert threshold_base().try_with_choices(pe=4).accepted


def placed_on(channels: int, table: ThresholdTable) -> ThresholdingAxiKernel:
    """The kernel between two boundary channels of three rows of ``channels``, PE open."""
    shape = (3, channels)

    class Placed(Root):
        x = Channel(
            tensor=Tensor(shape, ScalarEncoding(DataType["INT8"])), port="x", platform=FULL_DSP48E2
        )
        y = Channel(
            tensor=Tensor(shape, ScalarEncoding(DataType["INT3"])), port="y", platform=FULL_DSP48E2
        )
        activate = ThresholdingAxiKernel(
            input_dtype=DataType["INT8"],
            threshold_dtype=DataType["INT5"],
            thresholds=table,
            bias=-1,
            input_channel=x,
            output_channel=y,
            platform=FULL_DSP48E2,
        )

    kernel: ThresholdingAxiKernel = design_space(Placed()).activate
    return kernel


def test_a_placed_table_has_one_row_or_a_row_a_channel() -> None:
    """Placed, PE ranges over the input's channels, whatever the table's rows; a table of
    another number of rows is refused by name."""
    row = THRESHOLD_TABLE[0][0]
    for table in (((row,),), (tuple(row for _ in range(6)),)):
        point = placed_on(6, table)
        assert point.field(ThresholdingAxiKernel.pe).candidates() == Available((1, 2, 3, 6))
        chosen = point.try_with_choices(pe=3, use_axilite=False, deep_pipeline=False, **AUTO_MEMORY)
        assert chosen.accepted
        assert dict(chosen.instance.module.parameters)["C"] == len(table[0])
    other = placed_on(6, ((row, row),)).try_with_choices(pe=1).instance
    found = other.query(ThresholdingAxiKernel.schedule)
    assert isinstance(found, Rejected)
    assert {finding.code for finding in found.findings} == {"threshold-rows"}


def test_runtime_writable_thresholds_declare_the_writes_of_their_table() -> None:
    """Each threshold at (channel fold, lane, threshold) as thresholding_axi decodes it, its
    word masked to the threshold's bits; presented on the control bus with it."""
    table = (((-2, 0, 3), (-1, 1, 4), (0, 2, 5)),)  # three rows, N = 3
    point = threshold(table=table, pe=1, axilite=True)
    # PE 1: three folds of one lane; two bits of threshold, no lane bit, two of fold.
    expected = []
    for channel, row in enumerate(table[0]):
        for index, value in enumerate(row):
            expected.append((((channel << 2) | index) << 2, value & 0x1F))
    assert point.register_map.writes == tuple(expected)
    assert point.control_bus.registers == point.register_map
    shared = threshold(table=(((-2, 0, 3),),), pe=2, axilite=True)  # C = 1: one row, lane 0
    assert shared.register_map.writes == ((0, 0x1E), (4, 0), (8, 3))
    assert threshold(pe=1).control_bus.bus is None  # held, not presented: nothing to write


def test_a_threshold_wider_than_a_word_is_written_in_words_low_first() -> None:
    point = threshold(
        table=(((-(1 << 35), 1 << 34),),),
        pe=1,
        input_dtype="INT8",
        threshold_dtype="INT40",
        axilite=True,
    )
    mask = (1 << 40) - 1
    low, high = (-(1 << 35)) & mask, 1 << 34
    # Two words a threshold: the word select below the threshold index; the last commits.
    assert point.register_map.writes == (
        (0, low & 0xFFFFFFFF),
        (4, low >> 32),
        (8, high & 0xFFFFFFFF),
        (12, high >> 32),
    )


def _clog2(value: int) -> int:
    return (value - 1).bit_length()


@pytest.mark.parametrize(
    "sets,rows,count,pe",
    [(1, 1, 3, 4), (1, 6, 3, 2), (1, 4, 4, 4), (2, 6, 5, 2), (3, 2, 7, 1), (2, 2, 1, 4)],
)
def test_the_thresholds_file_holds_each_threshold_where_the_rtl_reads_it(
    sets: int, rows: int, count: int, pe: int
) -> None:
    """``thresholding.sv`` reads memory word ``a`` of stage ``s`` and PE lane ``p`` at a
    file address (``genInitFile``); without a file it takes ``THRESHOLDS[set][c][t]``
    (``genInitParam``). Both, transcribed here, agree for every threshold the RTL loads."""
    table = tuple(
        tuple(tuple(16 * g + 4 * r + t - 30 for t in range(count)) for r in range(rows))
        for g in range(sets)
    )
    point = threshold(table=table, pe=pe, threshold_dtype="INT8")
    words = [int(word, 16) for word in point.thresholds_file.data.split()]
    folds, lanes = (1, rows) if pe >= rows else (rows // pe, pe)
    group = 2 ** _clog2(folds) if sets > 1 else folds
    assert len(words) == sets * group * 2 ** _clog2(lanes) * 2 ** _clog2(count)
    stages = count.bit_length()
    for stage in range(stages):
        below = stages - 1 - stage
        for lane in range(pe):
            for address in range(group * 2**stage):
                upper, index = address % 2**stage, address // 2**stage
                position = upper * 2 ** (below + 1) + 2**below - 1
                fold, chosen = index % 2 ** _clog2(folds), index // 2 ** _clog2(folds)
                if position >= count or fold >= folds:
                    continue  # never loaded from the table: the RTL masks it
                read = (
                    index * 2 ** _clog2(lanes) * 2 ** _clog2(count)
                    + (lane % lanes) * 2 ** _clog2(count)
                    + position
                )
                expected = table[chosen][fold * lanes + lane % lanes][position]
                assert words[read] == expected & 0xFF


def test_a_table_over_a_million_bits_is_no_parameter() -> None:
    """Vivado stops on a parameter over 10**6 bits (issue thresholds-parameter-limit):
    the table is the file's, and no parameter grows with it."""
    rows, count = 512, 255
    table = (tuple(tuple(range(-128, 127)) for _ in range(rows)),)
    point = threshold(table=table, pe=8, input_dtype="INT8", threshold_dtype="INT8")
    assert rows * count * 8 > 10**6
    parameters = dict(point.module.parameters)
    assert max(len(str(value)) for value in parameters.values()) < 40
    assert len(point.thresholds_file.data.split()) == (rows // 8) * 8 * 256
