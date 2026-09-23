# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical dotp accepts caller-owned geometry and accumulation requirements."""

from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from kernels._next_helpers import assess, point_for, value
from finn.kernels.space._next import (
    Decided,
    Rejected,
    Unresolved,
    Param,
    Space,
    Subspace,
    inspection,
)
from finn.kernels.space._next.errors import RequestError
from finn.kernels.datatypes._next_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.artifacts.abi import Clock, Data, Derived as DerivedClock
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
import finn.kernels.dotp as dotp_axi
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.target import DspBlock
from finn.kernels._next_base import Kernel
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


def kernel(**updates):
    facts = parameters(**updates)
    pumping = facts.pop("compute_pumping")
    return point_for(DotpAxiKernel, _supplied(facts), compute_pumping=pumping)


def _supplied(facts):
    nested = {
        "activation_dtype": "activation.dtype",
        "weights_dtype": "weights.dtype",
        "result_dtype": "result.dtype",
    }
    return {nested.get(name, name): value for name, value in facts.items()}


class _PartialDotp(Space):
    pe = Param(int, required=False)
    simd = Param(int, required=False)
    activation_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    weights_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    result_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    target_dsp = Param(DspBlock, required=False)
    segment_length = Param(int, required=False)
    component = Subspace(
        DotpAxiKernel,
        pe=pe,
        simd=simd,
        target_dsp=target_dsp,
        segment_length=segment_length,
        bindings={
            DotpAxiKernel.activation.dtype: activation_dtype,
            DotpAxiKernel.weights.dtype: weights_dtype,
            DotpAxiKernel.result.dtype: result_dtype,
        },
    )


def partial(facts):
    return point_for(_PartialDotp, facts).component


def test_component_groups_its_interfaces_and_keeps_one_root_physical_output():
    assert DotpAxiKernel.__bases__ == (Kernel,)
    assert not hasattr(dotp_axi, "dotp_axi_requirements")
    point = kernel()
    assert tuple(item.key for item in point.capabilities() if item.scope == "") == ("physical",)
    value(point.physical().accepted_answer)
    inputs = {item.key for item in inspection.members(point) if item.kind == "param"}
    assert inputs == {
        "pe",
        "simd",
        "activation.dtype",
        "weights.dtype",
        "result.dtype",
        "target_dsp",
        "segment_length",
    }
    assert not hasattr(DotpAxiKernel, "activation_dtype")
    for name in ("contract", "region", "logical", "binding", "realization", "reference"):
        assert not hasattr(DotpAxiKernel, name)


@pytest.mark.parametrize("target", tuple(DspBlock))
@pytest.mark.parametrize("pumping", (False, True))
def test_assessed_view_preserves_geometry_and_clocks(target, pumping):
    settings = parameters(target_dsp=target, compute_pumping=pumping)
    point = kernel(**settings)
    requirements = value(point.physical().accepted_answer)
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
        assert stream.bus(clock="ap_clk", reset="ap_rst_n") == ports[stream.name]
        assert (
            next(signal.width for signal in ports[stream.name].signals if signal.logical == "tdata")
            == width
        )


def test_physical_framing_has_no_workload_period_and_only_activation_has_last():
    point = kernel(pe=3, simd=5)
    value(point.physical().accepted_answer)
    assert point.activation.elements_per_beat == 5
    assert point.weights.elements_per_beat == 15
    assert point.result.elements_per_beat == 3
    assert point.activation.last
    assert not point.weights.last and not point.result.last
    # Weight field p*SIMD+s is low-field-first, matching the native RTL array.
    assert [field.bit_offset for field in point.weights.payload.fields] == list(range(0, 45, 3))


@pytest.mark.parametrize("segment", (0, 2))
def test_segment_length_is_explicit_and_reaches_rtl(segment):
    requirements = value(
        kernel(simd=7, segment_length=segment, compute_pumping=True).physical().accepted_answer
    )
    assert dict(requirements.abi.parameters)["SEGMENTLEN"] == str(segment)


def test_dsp48_segment_length_is_ignored_by_rtl_but_preserved_as_supplied():
    requirements = value(
        kernel(target_dsp=DspBlock.DSP48E2, segment_length=100).physical().accepted_answer
    )
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
    physical = kernel(**updates).physical()
    refused = physical.accepted_answer
    assert isinstance(refused, Rejected), refused
    assert code in {
        finding.code
        for answer in physical.constraints.answers.values()
        if isinstance(answer, Rejected)
        for finding in answer.findings
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"pe": True},
        {"simd": 2.5},
        {"segment_length": True},
        {"compute_pumping": 1},
        {"target_dsp": "DSP58"},
        {"activation_dtype": "INT3"},
        {"result_dtype": None},
    ],
)
def test_space_rejects_mistyped_values_at_binding(updates):
    with pytest.raises(RequestError):
        kernel(**updates)


def test_constraints_gate_acceptance_without_revalidating_raw_codegen():
    point = kernel(weights_dtype=DataType["UINT3"], segment_length=-1)
    physical = point.physical()
    assert isinstance(physical.output_answer, Decided)
    assert physical.output_answer.value == value(point.answer(DotpAxiKernel.codegen))
    assert dict(physical.output_answer.value.parameters)["SEGMENTLEN"] == -1
    refused = physical.accepted_answer
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dotp-weight-type", "dotp-segment"}


@pytest.mark.parametrize("updates,stream", [({"simd": 0}, "activation"), ({"pe": 0}, "result")])
def test_invalid_native_interface_is_a_rejection_not_a_callback_failure(updates, stream):
    point = kernel(**updates)
    answer = point.answer(getattr(DotpAxiKernel, stream).stream)
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"dotp-interface"}
    assert isinstance(point.physical().accepted_answer, Rejected)


def test_physical_constraints_can_report_before_other_inputs_resolve():
    point = partial(
        dict(
            target_dsp=DspBlock.DSP58,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["UINT3"],
        ),
    )
    refusal = assess(point, DotpAxiKernel.input_types_supported)
    assert isinstance(refusal, Rejected)
    assert {finding.code for finding in refusal.findings} == {"dotp-weight-type"}


def test_result_dtype_and_pumping_are_required_without_any_workload_dimensions():
    facts = parameters()
    facts.pop("result_dtype")
    facts.pop("compute_pumping")
    point = partial(facts)
    assert isinstance(point.physical().accepted_answer, Unresolved)
    assert point.activation.element_bits == 3
    assert point.weights.elements_per_beat == 8
    facts["result_dtype"] = DataType["INT12"]
    point = point_for(DotpAxiKernel, _supplied(facts))
    assert isinstance(point.physical().accepted_answer, Unresolved)


@pytest.mark.parametrize("missing", tuple(parameters()))
def test_required_physical_facts_reject_omission_and_decision_remains_unresolved(missing):
    facts = parameters()
    facts.pop(missing)
    choices = {}
    if "compute_pumping" in facts:
        choices["compute_pumping"] = facts.pop("compute_pumping")
    if missing == "compute_pumping":
        point = point_for(DotpAxiKernel, _supplied(facts))
        assert isinstance(point.physical().accepted_answer, Unresolved)
    else:
        with pytest.raises(RequestError, match="missing required parameters"):
            point_for(DotpAxiKernel, _supplied(facts), **choices)


@pytest.mark.parametrize("bits", (4, 9, 12, 58))
def test_caller_selects_accumulator_capacity_and_owns_the_reduction_bound(bits):
    point = kernel(result_dtype=DataType[f"INT{bits}"])
    assert point.result.dtype == DataType[f"INT{bits}"]
    assert dict(value(point.physical().accepted_answer).parameters)["ACCU_WIDTH"] == bits


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
    requirements = value(
        kernel(
            target_dsp=target,
            activation_dtype=DataType[activation],
            weights_dtype=DataType[weight],
            result_dtype=DataType["INT48"],
        )
        .physical()
        .accepted_answer
    )
    assert dict(requirements.parameters)["SIGNED_ACTIVATIONS"] == int(DataType[activation].signed())


def test_sources_materialize_from_the_assessed_requirements(tmp_path):
    requirements = value(kernel(compute_pumping=True).physical().accepted_answer)
    store = ArtifactStore(tmp_path / "store")
    finnlib = Path(__file__).resolve().parents[2] / "deps" / "finnlib"
    if not (finnlib / "rtl/linalg/dotp_axi.sv").is_file():
        pytest.skip("FinnLib sources are unavailable")
    prepared = prepare_module_build(
        requirements,
        roots={"finnlib": finnlib, "kernels": resource_root()},
        template_roots=(),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    upstream = {
        "rtl/arith/add_multi_pkg.sv",
        "rtl/arith/add_multi.sv",
        "rtl/linalg/dotp_8sx9_dsp58.sv",
        "rtl/linalg/dotp.sv",
    }
    local = "dotp_axi.sv"
    expected = upstream | {local}
    emitted = {Path(name).name: Path(materialized.directory) / name for name in materialized.files}
    assert set(emitted) == {Path(name).name for name in expected}
    for path in upstream:
        assert emitted[Path(path).name].read_bytes() == (finnlib / path).read_bytes()
    assert emitted["dotp_axi.sv"].read_bytes() == (resource_root() / local).read_bytes()
    wrapper = next(source for source in requirements.contributions if source.path == local)
    assert wrapper.root == "kernels"
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
    value(point.physical().accepted_answer)
    assert point.result.payload_bits == 4 and point.result.carrier_bits == 8
    assert point.result.payload.unused[0].policy is UnusedBitPolicy.UNSPECIFIED
    assert point.activation.payload.unused[0].policy is UnusedBitPolicy.IGNORE_ON_RECEIVE
    field = point.result.payload.fields[0]
    mask = (1 << field.bit_width) - 1
    assert (0xFF >> field.bit_offset) & mask == (0x0F >> field.bit_offset) & mask
