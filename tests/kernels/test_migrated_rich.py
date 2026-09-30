# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public API checks for traversal, threshold tables and HLS source views."""

from __future__ import annotations

from typing import cast

import pytest

from finn.kernels.artifacts.abi import Bus, Signal
from finn.kernels.artifacts.hls import HlsSourceRequirements, render_hls_sources
from finn.kernels.datatypes.semantics import IntegerVector, ThresholdTable
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.memstream_hls import MemStreamHlsKernel
from finn.kernels.resources import template_root
from finn.core.space import (
    DefinitionError,
    Param,
    Rejected,
    Space,
    Unresolved,
    design_space,
)
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import finnlib_root

TABLE: ThresholdTable = (((-2, 0, 3), (-1, 1, 4)),)


def generator(
    *,
    bits: int = 13,
    frame: int = 6,
    dims: IntegerVector = (3, 6),
    strides: IntegerVector = (0, 1),
) -> InputGeneratorKernel:
    return design_space(
        InputGeneratorKernel(word_bits=bits, frame_words=frame, dims=dims, strides=strides)
    ).with_choices(ram_style="auto")


def threshold_base(
    *,
    table: ThresholdTable = TABLE,
    bias: int = -1,
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT5",
    bram: int = 0,
    uram: int = 0,
) -> ThresholdingAxiKernel:
    return design_space(
        ThresholdingAxiKernel(
            input_dtype=resolve_qonnx_datatype_name(input_dtype),
            threshold_dtype=resolve_qonnx_datatype_name(threshold_dtype),
            thresholds=table,
            bias=bias,
            depth_trigger_bram=bram,
            depth_trigger_uram=uram,
        )
    )


def threshold(
    *,
    table: ThresholdTable = TABLE,
    pe: int | None = 1,
    bias: int = -1,
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT5",
    bram: int = 0,
    uram: int = 0,
    axilite: bool = False,
    deep: bool = False,
) -> ThresholdingAxiKernel:
    """PE is committed as a choice; ``None`` leaves it open (a table without channels has none)."""
    base = threshold_base(
        table=table,
        bias=bias,
        input_dtype=input_dtype,
        threshold_dtype=threshold_dtype,
        bram=bram,
        uram=uram,
    )
    factors = {} if pe is None else {"pe": pe}
    report = base.try_with_choices(use_axilite=axilite, deep_pipeline=deep, **factors)
    assert report.accepted
    return report.instance


def memstream(dtype: str = "INT9", depth: int = 3) -> MemStreamHlsKernel:
    return design_space(
        MemStreamHlsKernel(element_dtype=resolve_qonnx_datatype_name(dtype), depth=depth)
    )


def test_generator_preserves_zero_stride_replay_and_multibit_native_markers() -> None:
    point = generator()
    requirements = point.build_requirements
    assert requirements.parameters == (
        ("COEFS", "'{0, 1}"),
        ("D", 2),
        ("DATA_WIDTH", 13),
        ("DIMS", "'{3, 6}"),
        ("FM_SIZE", 6),
        ("RAM_STYLE", '"auto"'),
    )
    widths = {port.name: port.width for port in requirements.abi.ports if isinstance(port, Signal)}
    assert widths["idat"] == widths["odat"] == 13
    assert widths["olst"] == 2
    assert all(isinstance(port, Signal) for port in requirements.abi.ports)
    ranked = generator(frame=56, dims=(3, 4, 2, 3), strides=(16, 1, 16, 2))
    ports = ranked.build_requirements.abi.ports
    assert (
        next(port.width for port in ports if isinstance(port, Signal) and port.name == "olst") == 4
    )


@pytest.mark.parametrize(
    ("dims", "strides"),
    (((), ()), ((3, 6), (1,)), ((2, 6), (1, 1)), ((3, 6), (-1, 1)), ((0, 6), (0, 1))),
)
def test_generator_refuses_invalid_loop_geometry(
    dims: IntegerVector, strides: IntegerVector
) -> None:
    assert isinstance(
        generator(dims=dims, strides=strides)
        .inspect(InputGeneratorKernel.build_requirements)
        .accepted_result,
        Rejected,
    )


@pytest.mark.parametrize("bad", ([3, 6], (3, True), (3, [6])))
def test_generator_requires_exact_immutable_integer_vectors(bad: object) -> None:
    # A bad literal is refused at the node call.
    with pytest.raises(DefinitionError, match="integer vector"):
        generator(dims=cast(IntegerVector, bad))


def test_threshold_output_initialization_and_configuration_profiles_are_preserved() -> None:
    point = threshold()
    requirements = point.build_requirements
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
        narrow_negative.inspect(ThresholdingAxiKernel.build_requirements).accepted_result, Rejected
    )
    enabled = threshold(axilite=True, deep=True).build_requirements
    assert dict(enabled.parameters)["USE_AXILITE"] == 1
    assert dict(enabled.parameters)["DEEP_PIPELINE"] == 1
    assert dict(threshold(pe=2).build_requirements.parameters)["PE"] == 2
    config = next(
        port
        for port in requirements.abi.ports
        if isinstance(port, Bus) and port.name == "s_axilite"
    )
    assert {
        signal.width for signal in config.signals if signal.logical in ("awaddr", "araddr")
    } == {5}


def test_threshold_multiple_sets_keep_selector_bus_and_refuse_axilite_addressing() -> None:
    table: ThresholdTable = (((-2, 0, 3), (-1, 1, 4)), ((-3, 0, 5), (-2, 0, 6)))
    point = threshold(table=table)
    requirements = point.build_requirements
    assert dict(requirements.parameters)["SETS"] == 2
    selector = next(
        port
        for port in requirements.abi.ports
        if isinstance(port, Bus) and port.name == "s_axis_set"
    )
    assert next(signal.width for signal in selector.signals if signal.logical == "tdata") == 8
    assert isinstance(
        threshold(table=table, axilite=True)
        .inspect(ThresholdingAxiKernel.build_requirements)
        .accepted_result,
        Rejected,
    )


def test_threshold_partial_dtype_query_does_not_adopt_implementation_decisions() -> None:
    base = threshold_base()
    assert base.result_dtype.name == "INT3"
    assert isinstance(base.query(ThresholdingAxiKernel.use_axilite), Unresolved)
    assert isinstance(base.query(ThresholdingAxiKernel.deep_pipeline), Unresolved)
    assert isinstance(
        base.inspect(ThresholdingAxiKernel.build_requirements).accepted_result, Unresolved
    )


def test_threshold_rejects_existing_unsupported_profiles_and_malformed_tables() -> None:
    profiles = (
        threshold(table=(), pe=None),
        threshold(table=(((2, 1),),)),
        threshold(table=(((0, 20),),)),
        threshold(table=(((0,), (0, 1)),)),
        threshold(threshold_dtype="UINT5"),
        threshold(bias=1 << 31),
        threshold(bias=-10),
        threshold(bram=-1),
    )
    assert all(
        isinstance(
            point.inspect(ThresholdingAxiKernel.build_requirements).accepted_result, Rejected
        )
        for point in profiles
    )
    with pytest.raises(DefinitionError, match="threshold table"):
        threshold(table=cast(ThresholdTable, (([-2, 0, 3],),)))
    # PE is a divisor of the table's channels: another is refused where it is committed.
    for pe in (0, 3, 4):
        assert not threshold_base().try_with_choices(pe=pe).accepted


@pytest.mark.parametrize(
    ("dtype", "cpp"),
    (
        ("INT9", "ap_int<9>"),
        ("UINT3", "ap_uint<3>"),
        ("BINARY", "ap_uint<1>"),
        ("FLOAT32", "float"),
    ),
)
def test_hls_view_preserves_cpp_types_interfaces_and_header_closure(dtype: str, cpp: str) -> None:
    point = memstream(dtype)
    requirements = point.sources
    assert isinstance(requirements, HlsSourceRequirements)
    assert point.cpp_type == cpp
    assert not hasattr(requirements, "abi")
    assert [(p.name, p.cpp_type, p.shape, p.mode) for p in requirements.interfaces] == [
        ("mem", cpp, (3,), "s_axilite"),
        ("dst", cpp, (), "axis"),
    ]
    rendered = render_hls_sources(
        requirements,
        roots={"finnlib": finnlib_root()},
        template_roots=(template_root(),),
    )
    assert [name for name, _ in rendered] == [
        "hls/util/util.hpp",
        "hls/infra/memstream.hpp",
        "memstream_hls.cpp",
    ]
    top = dict(rendered)["memstream_hls.cpp"].decode()
    assert f"using element_t = {cpp};" in top
    assert "(&mem)[3]" in top


@pytest.mark.parametrize(("dtype", "depth"), (("BIPOLAR", 3), ("INT1025", 3), ("INT9", 1)))
def test_hls_native_type_and_depth_limits_remain_explicit_refusals(dtype: str, depth: int) -> None:
    assert isinstance(
        memstream(dtype, depth).inspect(MemStreamHlsKernel.sources).accepted_result,
        Rejected,
    )


def test_rich_roots_require_parameters_and_optional_parent_depth_permits_narrow_hls_type() -> None:
    for family in (InputGeneratorKernel, ThresholdingAxiKernel, MemStreamHlsKernel):
        with pytest.raises(DefinitionError, match="is not supplied"):
            design_space(family())

    # Replaces an inline exposed Param child binding: the optional depth is the
    # parent's own formal, bound to the child by name.
    class OptionalMemory(Space):
        depth: int = Param(required=False)
        memory = MemStreamHlsKernel(element_dtype=resolve_qonnx_datatype_name("INT9"), depth=depth)

    point = design_space(OptionalMemory())
    assert point.memory.cpp_type == "ap_int<9>"
    assert isinstance(point.memory.inspect(MemStreamHlsKernel.sources).accepted_result, Unresolved)
