# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical dotp accepts caller-owned geometry and accumulation requirements."""

from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    UNSUPPLIED,
    Available,
    DefinitionError,
    Param,
    Rejected,
    Space,
    Unresolved,
    configure,
    inspection,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.artifacts.abi import Clock, Data, Derived as DerivedClock
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
import finn.kernels.dotp as dotp_axi
from kernels import helpers
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.target import DspBlock
from finn.kernels.base import Kernel
from finn.kernels.physical.layout import UnusedBitPolicy
from finn.kernels.resources import resource_root


def parameters(**updates):
    result = dict(
        pe=2,
        simd=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        result_dtype=DataType["INT9"],
        target_dsp=DspBlock.DSP58,
        segment_length=0,
        compute_pumping=False,
    )
    result.update(updates)
    return result


def point_for(facts, **choices):
    """Configure a dotp node from its formals, then commit choices by stable key."""
    return helpers.point_for(DotpAxiKernel, facts, **choices)


def kernel(**updates):
    facts = parameters(**updates)
    pumping = facts.pop("compute_pumping")
    return point_for(facts, compute_pumping=pumping)


class _PartialDotp(Space):
    pe: Param[int] = Param(int, default=UNSUPPLIED)
    simd: Param[int] = Param(int, default=UNSUPPLIED)
    activation_dtype: Param[QONNXDataType] = Param(
        QONNX_DATATYPE_VALUE_SEMANTICS, default=UNSUPPLIED
    )
    weights_dtype: Param[QONNXDataType] = Param(QONNX_DATATYPE_VALUE_SEMANTICS, default=UNSUPPLIED)
    result_dtype: Param[QONNXDataType] = Param(QONNX_DATATYPE_VALUE_SEMANTICS, default=UNSUPPLIED)
    target_dsp: Param[DspBlock] = Param(DspBlock, default=UNSUPPLIED)
    segment_length: Param[int] = Param(int, default=UNSUPPLIED)
    # Placed outside any stream: its optional stream formals stay unsupplied.
    component = DotpAxiKernel(
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        segment_length=segment_length,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        result_dtype=result_dtype,
    )


def partial(facts):
    return configure(_PartialDotp(**facts)).component


def test_component_groups_its_interfaces_and_keeps_one_root_physical_output():
    assert DotpAxiKernel.__bases__ == (Kernel,)
    assert not hasattr(dotp_axi, "dotp_axi_requirements")
    point = kernel()
    assert tuple(item.key for item in point.capabilities() if item.scope == "") == (
        "activation_port",
        "build_requirements",
        "interfaces",
        "ports",
        "result_port",
        "weights_port",
    )
    assert point.interfaces() == tuple(
        port.stream() for port in (point.activation, point.weights, point.result)
    )
    point.build_requirements()
    inputs = {item.key for item in inspection.members(point) if item.kind == "param"}
    assert inputs == {
        "pe",
        "simd",
        "activation_dtype",
        "weights_dtype",
        "result_dtype",
        "target_dsp",
        "segment_length",
    }
    # Optional reference inputs, supplied with Stream nodes by a parent that places
    # dotp between streams; alone, each is an unsupplied presence.
    streams = {item.key: item.kind for item in inspection.members(point)}
    assert {name: streams[name] for name in ("activation_stream", "result_stream")} == {
        "activation_stream": "present",
        "result_stream": "present",
    }
    # Operand dtypes are kernel facts; scalars and ports bind to them.
    assert point.activation.dtype == point.activation_type.dtype == point.activation_dtype
    for name in ("contract", "region", "logical", "binding", "realization", "reference"):
        assert not hasattr(DotpAxiKernel, name)


@pytest.mark.parametrize("target", tuple(DspBlock))
@pytest.mark.parametrize("pumping", (False, True))
def test_assessed_view_preserves_geometry_and_clocks(target, pumping):
    settings = parameters(target_dsp=target, compute_pumping=pumping)
    point = kernel(**settings)
    requirements = point.build_requirements()
    rtl = dict(requirements.parameters)
    assert rtl["ACCU_WIDTH"] == 9
    assert rtl["PE"] == 2 and rtl["SIMD"] == 4
    assert rtl["NARROW_WEIGHTS"] == 0
    assert rtl["SIGNED_ACTIVATIONS"] == 1
    ports = {port.name: port for port in requirements.abi.ports}
    assert ports["ap_clk2x"].role == (Clock(DerivedClock("ap_clk", 2)) if pumping else Data())
    assert bool(requirements.abi.clock_alignments) is pumping
    assert ports["ap_rst_n"].role.synchronous_to == (
        ("ap_clk", "ap_clk2x") if pumping else ("ap_clk",)
    )
    for stream, width in ((point.activation, 16), (point.weights, 24), (point.result, 24)):
        assert stream.stream().bus(clock="ap_clk", reset="ap_rst_n") == ports[stream.name]
        assert (
            next(signal.width for signal in ports[stream.name].signals if signal.logical == "tdata")
            == width
        )


def test_physical_framing_has_no_workload_period_and_only_activation_has_last():
    point = kernel(pe=3, simd=5)
    point.build_requirements()
    assert point.activation.lanes == 5
    assert point.weights.lanes == 15
    assert point.result.lanes == 3
    assert point.activation.last
    assert not point.weights.last and not point.result.last
    # Weight field p*SIMD+s is low-field-first, matching the native RTL array.
    assert [field.bit_offset for field in point.weights.payload.fields] == list(range(0, 45, 3))


@pytest.mark.parametrize("segment", (0, 2))
def test_segment_length_is_explicit_and_reaches_rtl(segment):
    requirements = kernel(simd=7, segment_length=segment, compute_pumping=True).build_requirements()
    assert dict(requirements.abi.parameters)["SEGMENTLEN"] == str(segment)


def test_dsp48_segment_length_is_ignored_by_rtl_but_preserved_as_supplied():
    requirements = kernel(target_dsp=DspBlock.DSP48E2, segment_length=100).build_requirements()
    assert dict(requirements.parameters)["SEGMENTLEN"] == 100


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"pe": 0}, "dotp-geometry"),
        ({"simd": -1}, "dotp-geometry"),
        ({"weights_dtype": DataType["UINT3"]}, "dotp-weight-type"),
        ({"weights_dtype": DataType["TERNARY"]}, "dotp-weight-type"),
        ({"activation_dtype": DataType["BINARY"]}, "dotp-activation-width"),
        ({"activation_dtype": DataType["BIPOLAR"]}, "dotp-activation-type"),
        ({"activation_dtype": DataType["FLOAT32"]}, "dotp-activation-type"),
        ({"activation_dtype": DataType["INT32"]}, "dotp-activation-width"),
        ({"activation_dtype": DataType["UINT24"]}, "dotp-activation-width"),
        (
            {"activation_dtype": DataType["UINT18"], "target_dsp": DspBlock.DSP48E2},
            "dotp-activation-width",
        ),
        (
            {"activation_dtype": DataType["UINT9"], "weights_dtype": DataType["INT8"]},
            "dotp-activation-width",
        ),
        ({"weights_dtype": DataType["INT27"]}, "dotp-weight-type"),
        ({"result_dtype": DataType["UINT9"]}, "dotp-result-type"),
        ({"result_dtype": DataType["FLOAT32"]}, "dotp-result-type"),
        ({"result_dtype": DataType["INT59"]}, "dotp-accumulator-width"),
        (
            {"result_dtype": DataType["INT49"], "target_dsp": DspBlock.DSP48E2},
            "dotp-accumulator-width",
        ),
        ({"segment_length": -1}, "dotp-segment"),
        ({"segment_length": 3}, "dotp-segment"),
        ({"simd": 1, "compute_pumping": True}, "dotp-pumping"),
    ],
)
def test_physical_view_preserves_support_refusals(updates, code):
    physical = kernel(**updates).build_requirements.inspect()
    refused = physical.accepted_result
    assert isinstance(refused, Rejected), refused
    assert code in {
        finding.code
        for answer in physical.constraints.results.values()
        if isinstance(answer, Rejected)
        for finding in answer.findings
    }


@pytest.mark.parametrize(
    "updates,error",
    [
        # A mistyped formal is refused at the node call; a mistyped choice at commit.
        ({"pe": True}, DefinitionError),
        ({"simd": 2.5}, DefinitionError),
        ({"segment_length": True}, DefinitionError),
        ({"compute_pumping": 1}, ValueError),
        ({"target_dsp": "DSP58"}, DefinitionError),
        ({"activation_dtype": "INT3"}, DefinitionError),
        ({"result_dtype": None}, DefinitionError),
    ],
)
def test_space_rejects_mistyped_values_at_binding(updates, error):
    with pytest.raises(error):
        kernel(**updates)


def test_constraints_gate_acceptance_without_revalidating_raw_codegen():
    point = kernel(segment_length=-1)
    physical = point.build_requirements.inspect()
    assert isinstance(physical.output_result, Available)
    assert point.query(DotpAxiKernel.codegen) == physical.output_result
    assert dict(physical.output_result.value.parameters)["SEGMENTLEN"] == -1
    refused = physical.accepted_result
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dotp-segment"}


@pytest.mark.parametrize("updates,stream", [({"simd": 0}, "activation"), ({"pe": 0}, "result")])
def test_invalid_native_interface_is_a_rejection_not_a_callback_failure(updates, stream):
    point = kernel(**updates)
    answer = getattr(point, stream).stream.query()
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"interface-lanes"}
    assert isinstance(point.build_requirements.inspect().accepted_result, Rejected)


def test_physical_constraints_can_report_before_other_inputs_resolve():
    point = partial(
        dict(
            target_dsp=DspBlock.DSP58,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["UINT3"],
        ),
    )
    refusal = point.inspect(DotpAxiKernel.input_types_supported).result
    assert isinstance(refusal, Rejected)
    assert {finding.code for finding in refusal.findings} == {"dotp-weight-type"}


def test_result_dtype_and_pumping_are_required_without_any_workload_dimensions():
    facts = parameters()
    facts.pop("result_dtype")
    facts.pop("compute_pumping")
    point = partial(facts)
    assert isinstance(point.build_requirements.inspect().accepted_result, Unresolved)
    assert point.activation.element_bits == 3
    assert point.weights.lanes == 8
    facts["result_dtype"] = DataType["INT12"]
    point = point_for(facts)
    assert isinstance(point.build_requirements.inspect().accepted_result, Unresolved)


@pytest.mark.parametrize("missing", tuple(parameters()))
def test_required_physical_facts_reject_omission_and_decision_remains_unresolved(missing):
    facts = parameters()
    facts.pop(missing)
    choices = {}
    if "compute_pumping" in facts:
        choices["compute_pumping"] = facts.pop("compute_pumping")
    if missing == "compute_pumping":
        point = point_for(facts)
        assert isinstance(point.build_requirements.inspect().accepted_result, Unresolved)
    else:
        # A bare call is legal; the missing formal is refused when configure() prepares it.
        with pytest.raises(DefinitionError, match=f"^{missing} is not supplied"):
            point_for(facts, **choices)


@pytest.mark.parametrize("bits", (4, 9, 12, 58))
def test_caller_selects_accumulator_capacity_and_owns_the_reduction_bound(bits):
    point = kernel(result_dtype=DataType[f"INT{bits}"])
    assert point.result.dtype == DataType[f"INT{bits}"]
    assert dict(point.build_requirements().parameters)["ACCU_WIDTH"] == bits


@pytest.mark.parametrize(
    "target,activation,weight",
    [
        (DspBlock.DSP48E1, "INT18", "INT24"),
        (DspBlock.DSP48E2, "UINT17", "INT26"),
        (DspBlock.DSP58, "UINT23", "INT26"),
        (DspBlock.DSP58, "INT9", "INT8"),
        (DspBlock.DSP58, "UINT8", "INT8"),
    ],
)
def test_supported_signed_and_unsigned_dsp_boundaries(target, activation, weight):
    requirements = kernel(
        target_dsp=target,
        activation_dtype=DataType[activation],
        weights_dtype=DataType[weight],
        result_dtype=DataType["INT48"],
    ).build_requirements()
    assert dict(requirements.parameters)["SIGNED_ACTIVATIONS"] == int(DataType[activation].signed())


def test_sources_materialize_from_the_assessed_requirements(tmp_path):
    requirements = kernel(compute_pumping=True).build_requirements()
    store = ArtifactStore(tmp_path / "store")
    finnlib = Path(__file__).resolve().parents[2] / "deps" / "finnlib"
    if not (finnlib / "rtl/dotp_axi.sv").is_file():
        pytest.skip("FinnLib sources are unavailable")
    prepared = prepare_module_build(
        requirements,
        roots={"finnlib": finnlib, "kernels": resource_root()},
        template_roots=(),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    upstream = {
        "rtl/add_multi_pkg.sv",
        "rtl/add_multi.sv",
        "rtl/dotp_8sx9_dsp58.sv",
        "rtl/dotp.sv",
        "rtl/dotp_axi.sv",
    }
    expected = upstream
    emitted = {Path(name).name: Path(materialized.directory) / name for name in materialized.files}
    assert set(emitted) == {Path(name).name for name in expected}
    for path in upstream:
        assert emitted[Path(path).name].read_bytes() == (finnlib / path).read_bytes()
    wrapper = next(
        source for source in requirements.contributions if source.path == "rtl/dotp_axi.sv"
    )
    assert wrapper.root == "finnlib"
    assert wrapper.provides == ("module:dotp_axi",)
    assert wrapper.requires == ("module:dotp", "module:dotp_8sx9_dsp58")


def test_subbyte_result_padding_has_no_zero_fill_promise():
    point = kernel(
        pe=1,
        simd=1,
        activation_dtype=DataType["INT2"],
        weights_dtype=DataType["INT2"],
        result_dtype=DataType["INT4"],
    )
    point.build_requirements()
    assert point.result.payload_bits == 4 and point.result.carrier_bits == 8
    assert point.result.payload.unused[0].policy is UnusedBitPolicy.UNSPECIFIED
    assert point.activation.payload.unused[0].policy is UnusedBitPolicy.IGNORE_ON_RECEIVE
    field = point.result.payload.fields[0]
    mask = (1 << field.bit_width) - 1
    assert (0xFF >> field.bit_offset) & mask == (0x0F >> field.bit_offset) & mask
