# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMulKernel construction, packing, precision, and portable builds."""

import pytest
from qonnx.core.datatype import DataType
from support.emitted_netlist import read
from support.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    WeightDelivery,
    built,
    choose,
    exact_result_dtype,
    keyed,
    labels,
    matmul_assembly,
    matmul_point,
    module_of,
    placed,
)
from support.netlist import Const, Drive, Pin
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    Rejected,
    Select,
    Unresolved,
    constraint,
    design_space,
    inspection,
    reject,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import emit_module
from finn.kernels.configure import commit
from finn.kernels.dotp import DotpAxiKernel, PackedDotpKernel
from finn.kernels.library import library_module
from finn.kernels.matmul import MatMulKernel

FACTS = dict(
    m=2,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)


def int4(shape: tuple[int, ...]) -> Tensor:
    return Tensor(shape, ScalarEncoding(DataType["INT4"]))


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
    # The activation channel's adapter replays each row once per output fold: the
    # root's channel, the real edge into the core.
    assert labels(built.module) == ["x.adapter.input_gen.input_gen", "matmul.compute.packed"]
    replay, dotp = (dict(leaf.parameters) for _, leaf in built.module.fragment.instances)
    assert (replay["FM_SIZE"], replay["DIMS"], replay["COEFS"]) == (2, "'{2, 2}", "'{0, 1}")
    assert dotp["ACCU_WIDTH"] == 8
    assert dotp["NARROW_WEIGHTS"] == 0
    assert {port.name for port in built.module.abi.pins if isinstance(port, Bus)} == {
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
    assert "in1_V" not in {port.name for port in built.module.abi.pins}
    memory = dict(placed(built.module, "w.source.memstream").parameters)
    assert {name: memory[name] for name in ("DEPTH", "WIDTH", "SETS", "RAM_STYLE")} == {
        "DEPTH": 4,
        "WIDTH": 12,
        "SETS": 1,
        "RAM_STYLE": '"auto"',
    }
    assert built.weight_beats == 12


def netlist_read(module, directory):
    """The emitted top of ``module`` as ports, instances and drives."""
    emitted = emit_module(module, directory, roots={"finnlib": finnlib_root()})
    return read(module, (emitted.directory / (emitted.entry_point + ".sv")).read_text())


def test_input_padding_is_ignored_and_child_padding_is_zero(tmp_path):
    built = assembly(k=6, n=3, pe=1)
    # Six payload bits in 8-bit words: the top inputs' padding is read by nothing, the
    # core's padding is driven zero.
    into = [link for link in built.module.fragment.links if link.source.instance is None]
    assert {(link.source.data, link.payload_bits, link.sink.data_bits) for link in into} == {
        ("in0_V_tdata", 6, 6),
        ("in1_V_tdata", 6, 8),
    }
    wired = netlist_read(built.module, tmp_path / "a")
    compute = "u_matmul_compute_packed"
    assert {
        Drive(Pin(compute, "s_axis_weights_tdata", (7, 6)), Const(2, 0)),
        Drive(Pin(compute, "s_axis_input_tdata", (7, 6)), Const(2, 0)),
    } <= set(wired.drives)
    # ... and the padding in the top inputs is read by nothing.
    assert {
        drive.source.bits
        for drive in wired.drives
        if isinstance(drive.source, Pin)
        and drive.source.instance is None
        and drive.source.port in ("in0_V_tdata", "in1_V_tdata")
    } == {(5, 0)}
    # Unpumped, no domain drives dotp's 2x clock input: it is held low.
    assert placed(built.module, "matmul.compute.packed").held.inputs == (("ap_clk2x", 0),)
    assert Drive(Pin(compute, "ap_clk2x"), Const(1, 0)) in set(wired.drives)
    # INT8 is exact for six INT3 products, so use width=2 to observe output padding.
    padded = assembly(k=2, n=3, pe=1)
    assert padded.result_dtype == DataType["INT7"]
    assert Drive(
        Pin(None, "out0_V_tdata", (7, 7)), Pin(compute, "m_axis_output_tdata", (7, 7))
    ) in set(netlist_read(padded.module, tmp_path / "b").drives)


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
        ({"m": 0}, "positive extent"),  # its extents are its tensors', positive
        ({"weight_delivery": "external"}, "WeightDelivery"),
        ({"weight_delivery": WeightDelivery.MEMSTREAM}, "requires weights"),
        ({"weights": [[0] * 4] * 4}, "no initializer"),
        # Weights of another shape: MatMul refuses them on its facts, before any choice.
        ({"weight_delivery": WeightDelivery.MEMSTREAM, "weights": [[0]]}, "matmul-tensor"),
        ({"weight_delivery": WeightDelivery.MEMSTREAM, "weights": [[4] * 4] * 4}, "admitted"),
    ],
)
def test_invalid_configuration_fails_during_construction(changes, match):
    with pytest.raises(ValueError, match=match):
        assembly(**changes)


def test_a_folding_factor_outside_its_domain_is_refused_by_its_own_finding_alone():
    """PE 3 divides neither extent, and that is also why no core is viable once it is
    committed. It is refused with its own finding (the selector's beside it); SIMD and
    pumping, committed with it, whose own values hold, are not refused through the
    selector."""
    trial = matmul_point(**FACTS).trial(
        {
            ("matmul", "pe"): 3,
            ("matmul", "simd"): 2,
            ("matmul", "compute.packed.compute_pumping"): False,
        },
        check=False,
    )
    assert list(trial.refused) == [("matmul", "pe")]
    found = {(item.code, item.owner) for item in trial.refused[("matmul", "pe")]}
    assert ("domain-membership", "pe") in found


def test_a_folding_factor_under_an_open_core_is_refused_by_its_domain_or_set() -> None:
    """On DSP58 two cores are possible, so ``compute`` stays open. PE is MatMul's, checked
    against its domain whatever the core: 3 divides no extent of 4, and is refused at its
    own commit by its own finding; 2 is set, waiting on nothing (its domain reads N from the
    facts alone, and every core takes it). Both readings (other facts: twice the rows)."""
    point = design_space(
        MatMulKernel(platform=FULL_DSP58, x_tensor=int4((8, 4)), w_tensor=int4((4, 4)))
    )
    assert isinstance(point.query(MatMulKernel.compute), Unresolved)
    pe = {item.key: item.reference for item in inspection.decisions(point)}["pe"]
    for reading in (
        lambda change: point.try_with_choices(change),
        lambda change: point.try_with_facts({MatMulKernel.x_tensor: int4((16, 4))}, change),
    ):
        three = reading({pe: 3})
        assert not three.accepted
        (outcome,) = (item for item in three.outcomes if item.status == "refused")
        assert outcome.owner == "pe"
        assert [item.code for item in outcome.result.findings] == ["domain-membership"]
        two = reading({pe: 2})
        assert two.accepted
        state = dict(inspection.states(two.instance))["pe"]
        assert state == inspection.Set(2)


#: Known weights (K, N) = (4, 4) over [-3, 3]: INT3 values, which INT2 does not hold.
KNOWN = ((1, -2, 3, 0), (2, 1, -1, 1), (0, 3, -2, 1), (1, 1, 1, -3))
PACKED = {"compute": "packed", "pe": 2}
FOLDED = {**PACKED, "simd": 2, "compute.packed.compute_pumping": False}


def validity_of(point):
    """The point's validity status, and each piece that is not true with its codes."""
    found = inspection.validity(point)
    return found.status, [
        (piece.owner, {finding.code for finding in piece.result.findings}) for piece in found.pieces
    ]


@pytest.mark.parametrize(
    ("weights", "choices", "status", "pieces"),
    (
        ("INT4", FOLDED, "valid", []),
        ("streamed", FOLDED, "valid", []),
        # MatMul's rule on the tensor it carries, refused on its facts alone.
        ("INT2", {}, "refused", [("carried", {"dtype-storage"})]),
        (
            "INT4",
            PACKED,
            "open",
            [
                ("compute.packed.pumping_has_lanes", {"decision-unassigned"}),
                ("compute.packed.simd_fits", {"decision-unassigned"}),
                ("compute.packed.stream_widths_supported", {"decision-unassigned"}),
            ],
        ),
    ),
)
def test_matmul_s_rules_and_its_cores_read_the_values_its_views_present(
    weights, choices, status, pieces
):
    """``carried`` reads the tensors MatMul requires and carries as values, and its cores
    read the carried and stored weights as values through their formals (``w_tensor``,
    ``result_range``), never MatMul's views: valid, refused and open each answer,
    with no dependency cycle. The views present the same values."""
    if weights == "streamed":
        facts = dict(w_tensor=int4((4, 4)))
    else:
        facts = dict(w_contents=KNOWN, weights_dtype=DataType[weights])
    point = design_space(MatMulKernel(platform=FULL_DSP48E2, x_tensor=int4((2, 4)), **facts))
    point = commit(point, choices) if choices else point
    assert validity_of(point) == (status, pieces)
    if status != "refused":
        assert point.w_carried == point.carried_weights == point.compute.w_tensor
        assert point.activation_tensor == point.required_activations
        assert point.weight_tensor == point.required_weights
    if weights == "INT4" and status != "refused":
        assert point.w_stored == point.stored_weights
        # Its columns over INT4 activations [-8, 7] (``column_range``): the stored values.
        assert point.result_range == (-54, 52)


def test_the_selected_core_s_own_rule_refuses_the_commit_that_breaks_it() -> None:
    """The selected core's own rule: pumping with one SIMD lane, committed together, is
    refused by it, at its origin, nothing standing."""
    facts = dict(w_contents=KNOWN, weights_dtype=DataType["INT4"])
    point = design_space(MatMulKernel(platform=FULL_DSP48E2, x_tensor=int4((2, 4)), **facts))
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    choices = {**PACKED, "simd": 1, "compute.packed.compute_pumping": True}
    result = point.try_with_choices({handles[key]: value for key, value in choices.items()})
    assert not result.accepted
    assert [
        (rule.rule, rule.standing, {finding.code for finding in rule.result.findings})
        for rule in result.broken
    ] == [("compute.packed.pumping_has_lanes", (), {"dotp-pumping"})]


def test_weights_that_are_not_integers_are_unrecognized_as_the_weights_value() -> None:
    with refused("literal-unrecognized", error=DefinitionError, message="integer tensor"):
        assembly(weight_delivery=WeightDelivery.MEMSTREAM, weights=[[0.5] * 4] * 4)


def test_the_core_is_one_possible_case_and_is_given_the_folding():
    base = matmul_point(**FACTS)
    # On DSP48E2 only the packed core is possible, before any folding factor (the INT8
    # core is a DSP58 mode, refused by its own rule): ``compute`` is open until set, as
    # conversion's policy sets it, never forced. Its channels wait on its contracts, which the
    # folding decides.
    ((_, compute),) = base.viable([("matmul", "compute")])
    assert compute.cases == ("packed",)
    assert base.chosen() == {} and base.unbound("x") is not None
    point = choose(
        base,
        {
            "matmul.compute": "packed",
            "matmul.pe": 2,
            "matmul.simd": 2,
        },
    )
    point = choose(
        point, {"x.transport": "direct", "w.transport": "direct", "y.transport": "direct"}
    )
    # The activation channel's adapter applies once its plan is known, which the
    # folding decides; the channel is the network's.
    point = choose(
        point,
        {"x.adapter": "input_gen", "x.adapter.input_gen.input_gen.ram_style": "auto"},
    )
    compute = point.member("matmul").compute
    assert isinstance(compute.inspect(DotpAxiKernel.netlist).accepted_result, Unresolved)
    assert isinstance(module_of(point, "matmul"), Unresolved)
    point = choose(
        point,
        {"matmul.compute.packed.compute_pumping": False, "matmul.compute.packed.reducer": "tree"},
    )
    matmul = point.member("matmul")
    assert matmul.result_type == DataType["INT8"]
    assert inspection.validity(matmul).status == "valid"
    assert matmul.compute.y_element.dtype == matmul.result_type
    _ = library_module(matmul.compute)
    assert matmul.compute.y_stream.sequence.form.beats == 4
    # The MatMul contributes its children's netlist; the network's module adds the
    # channels.
    assert [label for label, _ in matmul.netlist.instances] == ["compute.packed"]
    module = built(point, "matmul")
    assert labels(module) == ["x.adapter.input_gen.input_gen", "matmul.compute.packed"]
    assert not hasattr(MatMulKernel, "contract")
    # The folding is MatMul's, the role's, which every core is given.
    assert isinstance(MatMulKernel.pe, Decision) and isinstance(MatMulKernel.simd, Decision)
    assert matmul.compute.pe == matmul.pe == 2
    # Pumping with one SIMD lane is not possible: with SIMD 1 only False is, and the
    # change committing True is refused by the core's rule.
    one = choose(
        base,
        {
            "matmul.compute": "packed",
            "matmul.pe": 2,
            "matmul.simd": 1,
        },
    )
    ((_, pumping),) = one.viable([("matmul", "compute.packed.compute_pumping")])
    assert pumping.cases == (False,)
    pumped = one.trial(keyed({"matmul.compute.packed.compute_pumping": True}))
    assert not pumped.accepted
    found = [finding for why in pumped.refused.values() for finding in why]
    assert "dotp-pumping" in {f.code for f in found} | {c.code for f in found for c in f.causes}


@pytest.mark.parametrize("delivery", tuple(WeightDelivery))
def test_build_is_complete_and_initializer_changes_identity(tmp_path, delivery):
    options = dict(weight_delivery=delivery)
    if delivery is not WeightDelivery.EXTERNAL:
        options["weights"] = [[0] * 4] * 4
    built = assembly(**options)
    roots = {"finnlib": finnlib_root()}
    emitted = emit_module(built.module, tmp_path / "a", roots=roots)
    wrapper = (emitted.directory / (emitted.entry_point + ".sv")).read_text()
    # Unknown weights: the datatypes' range, INT8; known zero weights: [0, 0], whose
    # smallest encoding, BINARY, the packed core accumulates in (FinnLib's ACCU_WIDTH >= 1).
    assert f".ACCU_WIDTH({1 if 'weights' in options else 8})" in wrapper
    assert ".olst(n__u_x_adapter_input_gen_input_gen__olst)" in wrapper
    if delivery is WeightDelivery.MEMSTREAM:
        assert '.INIT_FILE("memstream_' in wrapper
    if delivery is not WeightDelivery.EXTERNAL:
        # The image is part of the identity: its content-named INIT_FILE.
        changed = assembly(weight_delivery=delivery, weights=[[1] * 4] * 4)
        renamed = emit_module(changed.module, tmp_path / "b", roots=roots)
        assert emitted.entry_point != renamed.entry_point


def test_matmul_honors_its_cores_rules_not_just_its_raw_module(monkeypatch):
    class RestrictedDotp(PackedDotpKernel):
        @constraint
        def view_only_rule(self) -> bool | Rejected:
            return reject("test-view-only", "this rule refuses the selected implementation")

    class RestrictedMatMul(MatMulKernel):
        # A compute Decision whose one candidate is the restricted core, on the
        # tensors MatMulKernel is given, which RestrictedMatMul inherits.
        compute = Select(
            packed=RestrictedDotp,
            form=MatMulKernel.datapath,
            platform=MatMulKernel.platform,
            reshape_activations=MatMulKernel.dense_view,
            result_range=MatMulKernel.result_range,
            x_tensor=MatMulKernel.x_tensor,
            w_tensor=MatMulKernel.carried_weights,
            columns=MatMulKernel.n,
            widest_reduction=MatMulKernel.widest_reduction,
            pe=MatMulKernel.pe,
            simd=MatMulKernel.simd,
        )

    # Substitute a fully authored Space class, without mutating declarations.
    monkeypatch.setattr("support.helpers.MatMulKernel", RestrictedMatMul)
    base = matmul_point(**FACTS)
    # The core's rule counts toward MatMul: its one case is not possible, refused by the
    # rule at its origin, and the change selecting it is refused, naming the rule.
    ((_, compute),) = base.viable([("matmul", "compute")])
    assert compute.cases == ()
    ((value, rules),) = [
        (item.value, [rule.rule for rule in item.broken])
        for item in compute.values or ()
        if isinstance(item, inspection.Refused)
    ]
    assert (value, rules) == ("packed", ["compute.packed.view_only_rule"])
    trial = base.trial(
        keyed(
            {
                "matmul.compute": "packed",
                "matmul.pe": 2,
                "matmul.simd": 2,
                "matmul.compute.packed.compute_pumping": False,
                "matmul.compute.packed.reducer": "tree",
            }
        )
    )
    assert not trial.accepted
    found = [finding for why in trial.refused.values() for finding in why]
    assert "test-view-only" in {f.code for f in found} | {c.code for f in found for c in f.causes}
    # The convenience entry point sets the core by its policy: no value can be set, the
    # rule named.
    with pytest.raises(ValueError, match="'packed' breaks compute.packed.view_only_rule"):
        assembly()
