# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical dotp on the Kernel protocol: three ports, its own folding factors, one admission.

Each compute core is its own kernel over the shared ``DotpAxiKernel``
declaration; most cases exercise the packed core, which every DSP target has.
A core sits between three streams (``helpers.placed_dotp``): its elements and
extents come from their tensors, and PE, SIMD and pumping are its Decisions.
"""

from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    DefinitionError,
    Rejected,
    Unresolved,
    inspection,
)
from finn.dataflow.gemm import Form
from finn.kernels.artifacts.abi import Clock, Data, Derived as DerivedClock
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.base import Kernel
from finn.kernels.dotp import DotpAxiKernel, Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.physical.layout import UnusedBitPolicy
from finn.kernels.port import AxiStreamPort
from finn.kernels.resources import resource_root
from finn.kernels.target import DspBlock
from kernels import helpers


def parameters(**updates):
    result = dict(
        pe=2,
        simd=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        result_dtype=DataType["INT9"],
        target_dsp=DspBlock.DSP58,
        target_period_ns=5.0,
        compute_pumping=False,
    )
    result.update(updates)
    return result


def kernel(family=PackedDotpKernel, **updates):
    return helpers.placed_dotp(family, **parameters(**updates))


def codes(result):
    assert isinstance(result, Rejected), result
    return {finding.code for finding in result.findings}


def test_a_core_declares_ports_folding_factors_and_facts_and_the_base_derives_the_module():
    assert DotpAxiKernel.__bases__ == (Kernel,)
    assert PackedDotpKernel.__bases__ == Int8Dsp58DotpKernel.__bases__ == (DotpAxiKernel,)
    point = kernel()
    assert [item.key for item in point.capabilities() if item.scope == "compute"] == [
        "compute.build_requirements",
        "compute.tieoffs",
    ]
    for port in (point.x, point.w, point.y):
        assert isinstance(port, AxiStreamPort)
    found = {item.key: item.kind for item in inspection.members(point)}
    # Facts, supplied by the placing parent (constants of this placement).
    facts = {key for key, kind in found.items() if kind == "const" and key.count(".") == 1}
    assert facts == {
        "compute.form",
        "compute.target_dsp",
        "compute.target_period_ns",
        "compute.reshape_activations",
        "compute.result_dtype",
        "compute.narrow_weights",
        "compute.x_stream",
        "compute.w_stream",
        "compute.y_stream",
    }
    # The core's own choices.
    assert {item.key for item in inspection.decisions(point)} == {
        "compute.pe",
        "compute.simd",
        "compute.compute_pumping",
    }
    for name in ("activation_dtype", "activation_type", "activation", "iteration", "contraction"):
        assert not hasattr(DotpAxiKernel, name)


@pytest.mark.parametrize("target", tuple(DspBlock))
@pytest.mark.parametrize("pumping", (False, True))
def test_assessed_view_preserves_geometry_and_clocks(target, pumping):
    point = kernel(target_dsp=target, compute_pumping=pumping)
    requirements = point.build_requirements
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
    for port, width in ((point.x, 16), (point.w, 24), (point.y, 24)):
        assert port.pins == (ports[port.name],)
        tdata = next(signal for signal in ports[port.name].signals if signal.logical == "tdata")
        assert tdata.width == width
    assert point.tieoffs.inputs == (() if pumping else (("ap_clk2x", 0),))


def test_physical_framing_follows_the_schedule_and_only_activation_has_last():
    point = kernel(pe=3, simd=5)
    _ = point.build_requirements
    assert point.x.axis.elements_per_beat == 5
    assert point.w.axis.elements_per_beat == 15
    assert point.y.axis.elements_per_beat == 3
    assert point.x.axis.last and not point.w.axis.last and not point.y.axis.last
    # Weight lane p*SIMD+s sits lane zero lowest, matching the native RTL array.
    assert [lane.bit_offset for lane in point.w.axis.payload.lanes] == list(range(0, 45, 3))


def test_the_schedule_splits_n_by_pe_and_k_by_simd():
    point = kernel(pe=2, simd=2, outputs=4, reduction=6, rows=3)
    assert (point.rows, point.outputs, point.reduction) == (3, 4, 6)
    x, w, y = (port.presented.form for port in (point.x, point.w, point.y))
    assert (x.beats, x.lanes) == (3 * 2 * 3, 2)  # each row replayed per output fold
    assert (w.beats, w.lanes) == (3 * 2 * 3, 4)
    assert (y.beats, y.lanes) == (3 * 2, 2)
    assert [rule.beats for rule in point.x.presented.markers] == [3]


@pytest.mark.parametrize(
    "period,pumping,segment",
    [
        # SIMD 7: a chain of three DSP58s, or two when pumped (six products each).
        (5.0, False, 3),  # 7 DSPs fit 5 ns; the whole chain is one segment
        (1.5, False, 2),  # 0.741 + 0.605 ns: two DSPs per segment
        (1.0, False, 1),
        (5.0, True, 2),  # pumped: against 2.5 ns, still the whole chain
        (2.0, True, 1),
    ],
)
def test_segment_length_follows_the_target_period(period, pumping, segment):
    point = kernel(simd=7, target_period_ns=period, compute_pumping=pumping)
    assert point.segment_length == segment
    assert dict(point.build_requirements.abi.parameters)["SEGMENTLEN"] == str(segment)


def test_dsp48_carries_the_segment_length_the_rtl_ignores():
    requirements = kernel(target_dsp=DspBlock.DSP48E2, simd=7).build_requirements
    assert dict(requirements.parameters)["SEGMENTLEN"] == 3


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"weights_dtype": DataType["UINT3"]}, "dtype-family"),
        ({"weights_dtype": DataType["TERNARY"]}, "dtype-family"),
        ({"activation_dtype": DataType["BINARY"]}, "dtype-minimum-bits"),
        ({"activation_dtype": DataType["BIPOLAR"]}, "dtype-family"),
        ({"activation_dtype": DataType["FLOAT32"]}, "dtype-family"),
        ({"activation_dtype": DataType["INT32"]}, "dotp-activation-width"),
        ({"activation_dtype": DataType["UINT24"]}, "dotp-activation-width"),
        (
            {"activation_dtype": DataType["UINT18"], "target_dsp": DspBlock.DSP48E2},
            "dotp-activation-width",
        ),
        ({"weights_dtype": DataType["INT27"]}, "dotp-weight-width"),
        ({"result_dtype": DataType["UINT9"]}, "dtype-family"),
        ({"result_dtype": DataType["FLOAT32"]}, "dtype-family"),
        ({"result_dtype": DataType["INT59"]}, "dotp-accumulator-width"),
        (
            {"result_dtype": DataType["INT49"], "target_dsp": DspBlock.DSP48E2},
            "dotp-accumulator-width",
        ),
        ({"target_period_ns": 0.7}, "dotp-clock-period"),
        ({"target_period_ns": 1.4, "compute_pumping": True}, "dotp-clock-period"),
        ({"simd": 1, "compute_pumping": True}, "dotp-pumping"),
    ],
)
def test_physical_view_reports_each_refusal_once(updates, code):
    # An encoding a port refuses is that port's finding; the admission group
    # adds only the DSP's own bounds, so no node reports a cause twice.
    physical = kernel(**updates).inspect(DotpAxiKernel.build_requirements)
    refused = physical.accepted_result
    assert isinstance(refused, Rejected), refused
    found = [(finding.owner, finding.code) for finding in refused.findings]
    assert code in {code for _, code in found}
    assert len(found) == len(set(found)), found


@pytest.mark.parametrize(
    "updates,error",
    [
        # A mistyped fact is refused at the node call; a mistyped folding factor at commit.
        ({"target_period_ns": "5"}, DefinitionError),
        ({"target_dsp": "DSP58"}, DefinitionError),
        ({"form": "dense"}, DefinitionError),
        ({"pe": True}, ValueError),
        ({"simd": 2.5}, ValueError),
        ({"compute_pumping": 1}, ValueError),
    ],
)
def test_space_rejects_mistyped_values_at_binding(updates, error):
    with pytest.raises(error):
        kernel(**updates)


@pytest.mark.parametrize("factor", ("pe", "simd"))
def test_a_folding_factor_must_divide_its_extent(factor):
    extents = {"outputs": 6, "reduction": 6}
    with pytest.raises(ValueError, match=f"compute.{factor}"):
        kernel(**{**extents, factor: 4})


def test_constraints_gate_acceptance_without_revalidating_raw_codegen():
    point = kernel(simd=1, compute_pumping=True)
    physical = point.inspect(DotpAxiKernel.build_requirements)
    assert isinstance(physical.output_result, Available)
    assert point.query(DotpAxiKernel.codegen) == physical.output_result
    assert dict(physical.output_result.value.parameters)["PUMPED_COMPUTE"] == 1
    assert codes(physical.accepted_result) == {"dotp-pumping"}


def test_the_core_refuses_before_its_folding_factors_are_chosen():
    point = kernel(weights_dtype=DataType["INT27"], pe=None, simd=None, compute_pumping=None)
    assert codes(point.inspect(DotpAxiKernel.core_supported).result) == {"dotp-weight-width"}
    # The module waits on the folding factors; the refusal is already known.
    module = point.inspect(DotpAxiKernel.build_requirements).accepted_result
    assert not isinstance(module, Available)
    # Folding factors left open leave an admissible core unresolved, not refused.
    open_factors = kernel(pe=None, simd=None, compute_pumping=None)
    assert isinstance(open_factors.inspect(DotpAxiKernel.core_supported).result, Available)
    assert isinstance(
        open_factors.inspect(DotpAxiKernel.build_requirements).accepted_result, Unresolved
    )
    assert open_factors.x.element.bits == 3


@pytest.mark.parametrize("missing", ("target_dsp", "target_period_ns"))
def test_required_physical_facts_reject_omission(missing):
    facts = parameters()
    facts.pop(missing)
    # A bare call is legal; the missing formal is refused when design_space() prepares it.
    with pytest.raises(DefinitionError, match=f"compute.{missing} is not supplied"):
        helpers.placed_dotp(PackedDotpKernel, **facts)


@pytest.mark.parametrize("bits", (4, 9, 12, 58))
def test_the_results_stream_selects_accumulator_capacity(bits):
    point = kernel(result_dtype=DataType[f"INT{bits}"])
    assert point.y.element.dtype == DataType[f"INT{bits}"]
    assert dict(point.build_requirements.parameters)["ACCU_WIDTH"] == bits


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
    ).build_requirements
    assert dict(requirements.parameters)["SIGNED_ACTIVATIONS"] == int(DataType[activation].signed())


def test_sources_materialize_from_the_assessed_requirements(tmp_path):
    requirements = kernel(compute_pumping=True).build_requirements
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
        "rtl/linalg/dotp.sv",
        "rtl/linalg/dotp_axi.sv",
    }
    emitted = {Path(name).name: Path(materialized.directory) / name for name in materialized.files}
    assert set(emitted) == {Path(name).name for name in upstream}
    for path in upstream:
        assert emitted[Path(path).name].read_bytes() == (finnlib / path).read_bytes()
    wrapper = next(
        source for source in requirements.contributions if source.path == "rtl/linalg/dotp_axi.sv"
    )
    assert wrapper.root == "finnlib"
    assert wrapper.provides == ("module:dotp_axi",)
    assert wrapper.requires == ("module:dotp",)
    int8 = kernel(Int8Dsp58DotpKernel).build_requirements
    assert [source.path for source in int8.contributions] == [
        "rtl/linalg/dotp_8sx9_dsp58.sv",
        "rtl/linalg/dotp_axi.sv",
    ]
    assert int8.contributions[1].requires == ("module:dotp_8sx9_dsp58",)


def test_subbyte_result_padding_has_no_zero_fill_promise():
    point = kernel(
        pe=1,
        simd=1,
        activation_dtype=DataType["INT2"],
        weights_dtype=DataType["INT2"],
        result_dtype=DataType["INT4"],
    )
    _ = point.build_requirements
    result, activation = point.y.axis, point.x.axis
    assert result.payload_bits == 4 and result.carrier_bits == 8
    assert result.payload.unused[0].policy is UnusedBitPolicy.UNSPECIFIED
    assert activation.payload.unused[0].policy is UnusedBitPolicy.IGNORE_ON_RECEIVE
    lane = result.payload.lanes[0]
    mask = (1 << lane.bit_width) - 1
    assert (0xFF >> lane.bit_offset) & mask == (0x0F >> lane.bit_offset) & mask


def test_each_core_kernel_names_its_core_and_the_shared_base_places_none():
    for family, core in ((PackedDotpKernel, "dotp"), (Int8Dsp58DotpKernel, "dotp_8sx9_dsp58")):
        requirements = kernel(family).build_requirements
        assert requirements.implementation_id == family.id
        assert dict(requirements.parameters)["CORE"] == f'"{core}"'
    refused = kernel(DotpAxiKernel).inspect(DotpAxiKernel.build_requirements).accepted_result
    assert codes(refused) == {"dotp-core"}


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"target_dsp": DspBlock.DSP48E2}, "dotp-target"),
        (
            {"activation_dtype": DataType["UINT9"], "weights_dtype": DataType["INT8"]},
            "dotp-activation-width",
        ),
        ({"activation_dtype": DataType["INT10"]}, "dotp-activation-width"),
        ({"weights_dtype": DataType["INT9"]}, "dotp-weight-width"),
    ],
)
def test_the_int8_core_takes_signed_nine_by_eight_products_on_dsp58_only(updates, code):
    refused = kernel(Int8Dsp58DotpKernel, **updates).query(DotpAxiKernel.build_requirements)
    assert code in codes(refused)
    for activation in ("INT9", "UINT8"):
        accepted = kernel(
            Int8Dsp58DotpKernel,
            activation_dtype=DataType[activation],
            weights_dtype=DataType["INT8"],
            result_dtype=DataType["INT32"],
        )
        assert isinstance(accepted.query(DotpAxiKernel.build_requirements), Available)


def test_depthwise_activations_carry_pe_channels_of_simd_and_only_int8_reads_them():
    point = kernel(Int8Dsp58DotpKernel, form=Form.DEPTHWISE, pe=3, simd=2)
    rtl = dict(point.build_requirements.parameters)
    assert rtl["ACTIVATION_BROADCASTING"] == 0
    lanes = [port.axis.elements_per_beat for port in (point.x, point.w, point.y)]
    assert lanes == [6, 6, 3]
    refused = kernel(form=Form.DEPTHWISE).query(DotpAxiKernel.build_requirements)
    assert codes(refused) == {"dotp-form"}
