# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-only MatMulKernel construction, packing, precision, and portable builds."""

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.core.space import Available, Rejected, Unresolved, inspection, selections
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import emit_module
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from kernels.helpers import finnlib_root, FULL_DSP48E2, labels, matmul_point, placed
from finn.kernels.matmul import MatMulKernel, exact_result_dtype
from kernels.helpers import WeightDelivery, matmul_assembly
from finn.kernels.dotp import DotpAxiKernel, PackedDotpKernel
from finn.core.space import Decision, View, constraint, reject


FACTS = dict(
    m=2,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
    target_period_ns=5.0,
)


def assembly(**changes):
    values = dict(
        m=3,
        k=4,
        n=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        pe=2,
        simd=2,
        platform=FULL_DSP48E2,
    )
    values.update(changes)
    return matmul_assembly(**values)


def test_external_construction_owns_replay_and_exact_precision():
    built = assembly()
    assert built.result_dtype == DataType["INT8"]
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (6, 12, 6)
    assert built.initializer == ()
    # The activation stream's adapter replays each row once per output fold: the
    # root's stream, the real edge into the core.
    assert labels(built.module) == ["x.adapter.input_gen.input_gen", "matmul.compute.packed"]
    replay, dotp = (dict(leaf.parameters) for _, leaf in built.module.fragment.instances)
    assert (replay["FM_SIZE"], replay["DIMS"], replay["COEFS"]) == (2, "'{2, 2}", "'{0, 1}")
    assert dotp["ACCU_WIDTH"] == 8
    assert dotp["NARROW_WEIGHTS"] == 0
    assert {port.name for port in built.module.pins.ports if isinstance(port, Bus)} == {
        "in0_V",
        "in1_V",
        "out0_V",
    }
    assert any(
        link.sink.instance == "matmul.compute.packed"
        and link.source.instance == "x.adapter.input_gen.input_gen"
        and link.markers == (("olst", 1, "s_axis_input_tlast", None),)
        for link in built.module.fragment.links
    )


def test_stored_image_has_output_then_reduction_then_pe_simd_order():
    # Written by output, stored (k, n).
    by_output = [[-4, -3, -2, -1], [0, 1, 2, 3], [3, 2, 1, 0], [-1, -2, -3, -4]]
    weights = [list(column) for column in zip(*by_output)]
    built = assembly(weight_delivery=WeightDelivery.MEMSTREAM, weights=weights)
    # Hand-packed INT3 lanes: p0/s0, p0/s1, p1/s0, p1/s1, low first.
    assert built.initializer == (0x22C, 0x6BE, 0xDD3, 0x941)
    assert "in1_V" not in {port.name for port in built.module.pins.ports}
    memory = dict(placed(built.module, "w.source.memstream").parameters)
    assert {name: memory[name] for name in ("DEPTH", "WIDTH", "SETS", "RAM_STYLE")} == {
        "DEPTH": 4,
        "WIDTH": 12,
        "SETS": 1,
        "RAM_STYLE": '"auto"',
    }
    assert built.weight_beats == 12


def netlist_text(module, directory):
    emitted = emit_module(module, directory, roots={"finnlib": finnlib_root()})
    return (emitted.directory / (emitted.entry_point + ".sv")).read_text()


def test_input_padding_is_ignored_and_child_padding_is_zero(tmp_path):
    built = assembly(k=6, n=3, pe=1)
    # Six payload bits in 8-bit words: the top inputs' padding is read by nothing, the
    # core's padding is driven zero.
    into = [link for link in built.module.fragment.links if link.source.instance is None]
    assert {(link.source.data, link.payload_bits, link.sink.data_bits) for link in into} == {
        ("in0_V_tdata", 6, 6),
        ("in1_V_tdata", 6, 8),
    }
    text = netlist_text(built.module, tmp_path / "a")
    assert "assign n__u_matmul_compute_packed__s_axis_weights_tdata[7:6] = 2'h0;" in text
    assert "assign n__u_matmul_compute_packed__s_axis_input_tdata[7:6] = 2'h0;" in text
    assert "in0_V_tdata[7:6]" not in text and "in1_V_tdata[7:6]" not in text
    # Unpumped, no domain drives dotp's 2x clock input: it is held low.
    assert placed(built.module, "matmul.compute.packed").held.inputs == (("ap_clk2x", 0),)
    assert "assign n__u_matmul_compute_packed__ap_clk2x = 1'h0;" in text
    # INT8 is exact for six INT3 products, so use width=2 to observe output padding.
    padded = assembly(k=2, n=3, pe=1)
    assert padded.result_dtype == DataType["INT7"]
    assert (
        "assign out0_V_tdata[7] = n__u_matmul_compute_packed__m_axis_output_tdata[7];"
        in netlist_text(padded.module, tmp_path / "b")
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
        ({"m": 0}, "matmul-extents"),
        # dotp's accumulator refuses this width.
        ({"k": 1 << 48}, "dotp-accumulator-width"),
        ({"weight_delivery": "external"}, "WeightDelivery"),
        ({"weight_delivery": WeightDelivery.MEMSTREAM}, "requires weights"),
        ({"weights": [[0] * 4] * 4}, "no initializer"),
        ({"weight_delivery": WeightDelivery.MEMSTREAM, "weights": [[0]]}, "shape"),
        ({"weight_delivery": WeightDelivery.MEMSTREAM, "weights": [[4] * 4] * 4}, "admitted"),
        ({"weight_delivery": WeightDelivery.MEMSTREAM, "weights": [[0.5] * 4] * 4}, "integer"),
    ],
)
def test_invalid_configuration_fails_during_construction(changes, match):
    with pytest.raises(ValueError, match=match):
        assembly(**changes)


def test_the_core_is_forced_and_owns_its_folding_factors():
    base = commit(matmul_point(**FACTS), {"w.transport": "direct"})
    # On DSP48E2 only the packed core admits the configuration, before any folding
    # factor: it is forced, never committed.
    assert ("matmul.compute", "packed") in {
        (item.key, item.value) for item in inspection.forced(base)
    }
    assert selections.capture(base).keys == ("w.transport",)
    point = commit(
        base,
        {"matmul.compute.packed.pe": 2, "matmul.compute.packed.simd": 2},
    )
    # The activation stream's adapter applies once its plan is known, which the
    # folding decides; the stream is the root's.
    point = commit(
        point,
        {"x.adapter": "input_gen", "x.adapter.input_gen.input_gen.ram_style": "auto"},
    )
    compute = point.matmul.compute
    assert isinstance(compute.inspect(DotpAxiKernel.module).accepted_result, Unresolved)
    assert isinstance(point.inspect(Kernel.module).accepted_result, Unresolved)
    point = commit(point, {"matmul.compute.packed.compute_pumping": False})
    matmul = point.matmul
    assert matmul.result_type == DataType["INT8"]
    assert matmul.inspect(MatMulKernel.admission).result == Available(True)
    assert matmul.compute.y.element.dtype == matmul.result_type
    _ = matmul.compute.module
    assert matmul.compute.y.presented.form.beats == 4
    # The MatMul contributes its children's netlist; the root's module adds the streams.
    # Without them, MatMul alone is no complete module: its core's inputs are the root's.
    assert [label for label, _ in matmul.netlist.instances] == ["compute.packed"]
    alone = matmul.query(MatMulKernel.module)
    assert isinstance(alone, Rejected) and "nothing drives" in alone.findings[0].message
    assert labels(point.module) == ["x.adapter.input_gen.input_gen", "matmul.compute.packed"]
    assert not hasattr(MatMulKernel, "contract") and not hasattr(MatMulKernel, "pe")
    refused = commit(
        base,
        {
            "matmul.compute.packed.pe": 2,
            "matmul.compute.packed.simd": 1,
            "matmul.compute.packed.compute_pumping": True,
            "x.adapter": "input_gen",
            "x.adapter.input_gen.input_gen.ram_style": "auto",
        },
    )
    # No core admits it: the core is no longer forced but refused, naming each reason.
    answer = refused.matmul.query(MatMulKernel.compute)
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["decision-no-viable-case"]
    assert "dotp-pumping" in answer.findings[0].message
    # Committed on purpose, the core is placed and refuses the configuration itself.
    chosen = commit(refused, {"matmul.compute": "packed"})
    assert isinstance(chosen.matmul.compute.inspect(DotpAxiKernel.module).accepted_result, Rejected)
    rejected = chosen.query(Kernel.module)
    assert isinstance(rejected, Rejected)
    assert "dotp-pumping" in {finding.code for finding in rejected.findings}


@pytest.mark.parametrize("delivery", tuple(WeightDelivery))
def test_build_is_complete_and_initializer_changes_identity(tmp_path, delivery):
    options = dict(weight_delivery=delivery)
    if delivery is not WeightDelivery.EXTERNAL:
        options["weights"] = [[0] * 4] * 4
    built = assembly(**options)
    roots = {"finnlib": finnlib_root()}
    emitted = emit_module(built.module, tmp_path / "a", roots=roots)
    wrapper = (emitted.directory / (emitted.entry_point + ".sv")).read_text()
    assert ".ACCU_WIDTH(8)" in wrapper
    assert ".olst(n__u_x_adapter_input_gen_input_gen__olst)" in wrapper
    if delivery is WeightDelivery.MEMSTREAM:
        assert '.INIT_FILE("memstream_' in wrapper
    if delivery is not WeightDelivery.EXTERNAL:
        # The image is part of the identity: its content-named INIT_FILE.
        changed = assembly(weight_delivery=delivery, weights=[[1] * 4] * 4)
        renamed = emit_module(changed.module, tmp_path / "b", roots=roots)
        assert emitted.entry_point != renamed.entry_point


def test_matmul_honors_the_child_physical_view_not_just_its_raw_module(monkeypatch):
    class RestrictedDotp(PackedDotpKernel):
        @constraint
        def view_only_rule(self) -> bool | Rejected:
            return reject(
                "test-view-only", "this physical View refuses the selected implementation"
            )

        module = View(PackedDotpKernel.built, requires=(PackedDotpKernel.admission, view_only_rule))

    class RestrictedMatMul(MatMulKernel):
        # A compute Decision whose one candidate is the restricted core, on the
        # streams MatMulKernel is supplied, which RestrictedMatMul inherits.
        compute = Decision(
            {"packed": RestrictedDotp},
            form=MatMulKernel.datapath,
            target_period_ns=MatMulKernel.target_period_ns,
            platform=MatMulKernel.platform,
            reshape_activations=MatMulKernel.dense_view,
            result_dtype=MatMulKernel.result_type,
            x_stream=MatMulKernel.x_stream,
            w_stream=MatMulKernel.w_stream,
            y_stream=MatMulKernel.y_stream,
        )

    # Substitute a fully authored family, without mutating declarations.
    monkeypatch.setattr("kernels.helpers.MatMulKernel", RestrictedMatMul)
    point = commit(
        matmul_point(**FACTS),
        {
            "matmul.compute": "packed",
            "matmul.compute.packed.pe": 2,
            "matmul.compute.packed.simd": 2,
            "matmul.compute.packed.compute_pumping": False,
            "x.adapter": "input_gen",
            "x.adapter.input_gen.input_gen.ram_style": "auto",
            "w.transport": "direct",
        },
    )
    compute = point.matmul.compute
    assert isinstance(compute.query(DotpAxiKernel.codegen), Available)
    assert isinstance(compute.inspect(DotpAxiKernel.module).accepted_result, Rejected)
    refused = point.query(Kernel.module)
    assert isinstance(refused, Rejected)
    assert "test-view-only" in {finding.code for finding in refused.findings}
    # The convenience entry point takes the same accepted-view path.
    with pytest.raises(ValueError, match="test-view-only"):
        assembly()
