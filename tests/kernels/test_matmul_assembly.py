# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-only MatMulKernel construction, packing, precision, and portable builds."""

from pathlib import Path

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.core.space import Available, Rejected, Unresolved
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import prepare_module_build, render_module_sources
from finn.kernels.artifacts.store import ArtifactStore
from kernels.helpers import point_for
from finn.kernels.matmul import MatMulKernel, WeightDelivery, exact_result_dtype, matmul_assembly
from finn.kernels.dotp import DotpAxiKernel, PackedDotpKernel
from finn.core.space import Decision, View, constraint, reject
from finn.kernels.target import DspBlock
from finn.kernels.physical.structure import ConstantBits, PhysicalPin, PinSlice
from finn.kernels.resources import resource_root, template_root


ROOT = Path(__file__).resolve().parents[2]
FACTS = dict(
    rows=2,
    reduction=4,
    outputs=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    target_dsp=DspBlock.DSP48E2,
    target_period_ns=5.0,
)


def assembly(**changes):
    values = dict(
        rows=3,
        reduction=4,
        outputs=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        pe=2,
        simd=2,
        target_dsp=DspBlock.DSP48E2,
    )
    values.update(changes)
    return matmul_assembly(**values)


def test_external_construction_owns_replay_and_exact_precision():
    built = assembly()
    assert built.result_dtype == DataType["INT8"]
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (6, 12, 6)
    assert built.initializer == ()
    assert [item.instance_id for item in built.structure.instances] == [
        "u_replay_buffer",
        "u_compute_packed",
    ]
    replay, dotp = (dict(item.requirements.parameters) for item in built.structure.instances)
    assert replay == {"LEN": 2, "REP": 2, "W": 6}
    assert dotp["ACCU_WIDTH"] == 8
    assert dotp["NARROW_WEIGHTS"] == 0
    assert {port.name for port in built.structure.top_abi.ports if isinstance(port, Bus)} == {
        "in0_V",
        "in1_V",
        "out0_V",
    }
    assert any(
        wire.destination.pin == PhysicalPin("u_compute_packed", "s_axis_input_tlast")
        and wire.source == PinSlice(PhysicalPin("u_replay_buffer", "olast"), 0, 1)
        for wire in built.structure.wires
    )


def test_cyclic_image_has_output_then_reduction_then_pe_simd_order():
    weights = [[-4, -3, -2, -1], [0, 1, 2, 3], [3, 2, 1, 0], [-1, -2, -3, -4]]
    built = assembly(weight_delivery=WeightDelivery.CYCLIC, weights=weights)
    # Hand-packed INT3 fields: p0/s0, p0/s1, p1/s0, p1/s1, low first.
    assert built.initializer == (0x22C, 0x6BE, 0xDD3, 0x941)
    assert "in1_V" not in {port.name for port in built.structure.top_abi.ports}
    cyclic = dict(built.structure.instances[2].requirements.parameters)
    assert cyclic == {
        "DEPTH": 4,
        "W": 12,
        "INIT_DATA": "48'h941dd36be22c",
        "ROM_STYLE": '"auto"',
    }
    assert built.weight_beats == 12


def test_input_padding_is_ignored_and_child_padding_is_zero():
    built = assembly(reduction=6, outputs=3, pe=1)
    assert built.structure.ignored_top_input_bits == (
        PinSlice(PhysicalPin(None, "in0_V_tdata"), 6, 2),
        PinSlice(PhysicalPin(None, "in1_V_tdata"), 6, 2),
    )
    zeros = {
        wire.destination: wire.source
        for wire in built.structure.wires
        if isinstance(wire.source, ConstantBits)
    }
    assert zeros == {
        PinSlice(PhysicalPin("u_compute_packed", "s_axis_input_tdata"), 6, 2): ConstantBits(2, 0),
        PinSlice(PhysicalPin("u_compute_packed", "s_axis_weights_tdata"), 6, 2): ConstantBits(2, 0),
        # Unpumped, no domain drives dotp's 2x clock input: it is tied low.
        PinSlice(PhysicalPin("u_compute_packed", "ap_clk2x"), 0, 1): ConstantBits(1, 0),
    }
    # INT8 is exact for six INT3 products, so use width=2 to observe output padding.
    padded = assembly(reduction=2, outputs=3, pe=1)
    assert padded.result_dtype == DataType["INT7"]
    assert any(
        wire.destination == PinSlice(PhysicalPin(None, "out0_V_tdata"), 7, 1)
        and wire.source == PinSlice(PhysicalPin("u_compute_packed", "m_axis_output_tdata"), 7, 1)
        for wire in padded.structure.wires
    )


@pytest.mark.parametrize(
    "activation,weight,length,bits",
    [
        ("INT2", "INT2", 1, 4),
        ("INT3", "INT3", 4, 8),
        ("UINT3", "INT3", 3, 8),
        ("INT8", "INT8", 8, 19),
    ],
)
def test_precision_covers_full_ranges_and_is_minimal(activation, weight, length, bits):
    a, w = DataType[activation], DataType[weight]
    dtype = exact_result_dtype(length, a, w)
    endpoints = [int(x) * int(y) * length for x in (a.min(), a.max()) for y in (w.min(), w.max())]
    assert dtype.bitwidth() == bits
    assert int(dtype.min()) <= min(endpoints) <= max(endpoints) <= int(dtype.max())
    assert min(endpoints) < -(1 << (bits - 2)) or max(endpoints) >= (1 << (bits - 2))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"pe": 3}, "domain-membership"),
        ({"simd": 3}, "domain-membership"),
        ({"rows": 0}, "positive"),
        # Both the replay's native length and dotp's accumulator refuse this width.
        ({"reduction": 1 << 48}, "replay-geometry|dotp-accumulator-width"),
        ({"weight_delivery": "external"}, "WeightDelivery"),
        ({"weight_delivery": WeightDelivery.CYCLIC}, "requires weights"),
        ({"weights": [[0] * 4] * 4}, "no initializer"),
        ({"weight_delivery": WeightDelivery.CYCLIC, "weights": [[0]]}, "shape"),
        ({"weight_delivery": WeightDelivery.CYCLIC, "weights": [[4] * 4] * 4}, "admitted"),
        ({"weight_delivery": WeightDelivery.CYCLIC, "weights": [[0.5] * 4] * 4}, "integer"),
    ],
)
def test_invalid_configuration_fails_during_construction(changes, match):
    with pytest.raises(ValueError, match=match):
        assembly(**changes)


def test_space_selects_folding_and_constructs_without_a_logical_contract():
    base = point_for(
        MatMulKernel,
        FACTS,
        pe=2,
        delivery="external",
        compute="packed",
        replay="buffer",
        **{"weight_stream.transport": "direct"},
    )
    point = base.with_choices(simd=2)
    assert isinstance(
        point.compute.inspect(DotpAxiKernel.build_requirements).accepted_result, Unresolved
    )
    assert isinstance(point.inspect(MatMulKernel.structure).accepted_result, Unresolved)
    point = point.with_choices(compute_pumping=False)
    assert point.result_type == DataType["INT8"]
    assert point.inspect(MatMulKernel.dimensions_supported).result == Available(True)
    assert point.compute.pe == point.pe
    assert point.compute.result.dtype == point.result_type
    _ = point.compute.build_requirements
    assert point.folding.result_beats == 4
    assert point.structure.requirements == point.build_requirements
    assert not hasattr(MatMulKernel, "contract")
    refused = base.with_choices(simd=1, compute_pumping=True)
    assert isinstance(
        refused.compute.inspect(DotpAxiKernel.build_requirements).accepted_result, Rejected
    )
    rejected = refused.query(MatMulKernel.structure)
    assert isinstance(rejected, Rejected)
    assert "dotp-pumping" in {finding.code for finding in rejected.findings}


@pytest.mark.parametrize("delivery", tuple(WeightDelivery))
def test_build_is_complete_and_initializer_changes_identity(tmp_path, delivery):
    options = dict(weight_delivery=delivery)
    if delivery is WeightDelivery.CYCLIC:
        options["weights"] = [[0] * 4] * 4
    built = assembly(**options)
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        built.requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    rendered = render_module_sources(prepared, store)
    wrapper = dict(rendered.contents)[prepared.abi.entry_point + ".sv"].decode()
    assert ".ACCU_WIDTH(8)" in wrapper
    assert ".olast(n__u_replay_buffer__olast)" in wrapper
    assert prepared.slots == ()
    if delivery is WeightDelivery.CYCLIC:
        assert ".INIT_DATA(48'h0)" in wrapper
        changed = assembly(weight_delivery=delivery, weights=[[1] * 4] * 4)
        changed_prepared = prepare_module_build(
            changed.requirements,
            roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
            template_roots=(template_root(),),
            blobs=store,
        )
        assert prepared.name != changed_prepared.name


def test_matmul_honors_the_child_physical_view_not_just_its_raw_module(monkeypatch):
    class RestrictedDotp(PackedDotpKernel):
        @constraint
        def view_only_rule(self) -> bool | Rejected:
            return reject(
                "test-view-only", "this physical View refuses the selected implementation"
            )

        build_requirements = View(
            PackedDotpKernel.codegen, requires=(PackedDotpKernel.support, view_only_rule)
        )

    class RestrictedMatMul(MatMulKernel):
        # A narrower compute Decision whose one candidate is the restricted core.
        compute = Decision(
            values={
                "packed": RestrictedDotp(
                    activation_dtype=MatMulKernel.activation_dtype,
                    weights_dtype=MatMulKernel.weights_dtype,
                    result_dtype=MatMulKernel.result_type,
                    pe=MatMulKernel.pe,
                    simd=MatMulKernel.simd,
                    target_dsp=MatMulKernel.target_dsp,
                    target_period_ns=MatMulKernel.target_period_ns,
                    compute_pumping=MatMulKernel.compute_pumping,
                    # References to MatMulKernel's stream nodes, which RestrictedMatMul inherits.
                    activation_stream=MatMulKernel.replayed,
                    weights_stream=MatMulKernel.weight_stream,
                    result_stream=MatMulKernel.results,
                )
            }
        )

    point = point_for(
        RestrictedMatMul,
        FACTS,
        pe=2,
        simd=2,
        delivery="external",
        compute="packed",
        compute_pumping=False,
        replay="buffer",
        **{"weight_stream.transport": "direct"},
    )
    assert isinstance(point.compute.query(DotpAxiKernel.codegen), Available)
    assert isinstance(
        point.compute.inspect(DotpAxiKernel.build_requirements).accepted_result, Rejected
    )
    refused = point.query(MatMulKernel.structure)
    assert isinstance(refused, Rejected)
    assert "test-view-only" in {finding.code for finding in refused.findings}
    # Substitute a fully authored family to exercise the convenience entry
    # point through the same accepted-view path, without mutating declarations.
    monkeypatch.setattr("finn.kernels.matmul.MatMulKernel", RestrictedMatMul)
    with pytest.raises(ValueError, match="test-view-only"):
        assembly()
