# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One MatMulKernel space for dense and depthwise forms.

The form is a fact of the operation; what follows from it is derived: whether
activations are broadcast, whether rows are replayed, the activation traversal.
A depthwise operation is realized natively (only the INT8 DSP58 core reads one
channel per lane) or densely, with block-diagonal weights on any core; the
dense realization needs the weights, so a weight memory. Weights are
stored (k, n): window by channel.
"""

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    WeightDelivery,
    built,
    choose,
    keyed,
    labels,
    matmul_assembly,
    matmul_point,
    module_of,
    placed,
    with_adapter_memories,
    with_direct_transports,
)

from finn.core.space import Available, Unresolved, inspection
from finn.dataflow.gemm import Form
from finn.dataflow.steps import Step
from finn.dataflow.traversal import Traversal
from finn.kernels.channels import Channel
from finn.kernels.matmul import MatMulKernel

FACTS = dict(
    m=2,
    k=9,  # the window
    n=4,  # the channels
    form=Form.DEPTHWISE,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    platform=FULL_DSP58,
)


def point(core="int8_dsp58", **facts):
    facts = {**FACTS, **facts}
    choices = {
        "w.transport": "direct",
        "matmul.compute": core,
        "matmul.pe": 2,
        "matmul.simd": 3,
        f"matmul.compute.{core}.compute_pumping": False,
        **({"matmul.compute.packed.reducer": "tree"} if core == "packed" else {}),
    }
    native = "native" if facts["form"] is Form.DEPTHWISE else None
    return with_direct_transports(
        with_adapter_memories(choose(matmul_point(realization=native, **facts), choices))
    )


def parameters(module, label):
    return dict(placed(module, label).parameters)


def test_the_realization_comes_before_the_cores_choices():
    """The cores' facts read the realization (the datapath it gives them): in the flow
    order it comes before their choices, and, declared first, before the folding and
    ``compute`` too. The folding, MatMul's, is declared before ``compute``: conversion
    folds first, then picks the core."""
    keys = [item.key for item in inspection.decisions(MatMulKernel)]
    assert keys[:4] == ["realization", "pe", "simd", "compute"]
    assert all(key.startswith("compute.") for key in keys[4:])


def test_depthwise_rows_pass_once_with_a_frame_per_window():
    configured = point()
    x = configured.member("x")
    # Nothing is replayed: the channel only closes each window's frame.
    assert x.steps.kinds == (Step.MARKERS,)
    boundary = x.boundary
    # Rows, then channel folds, then window folds; lane s * PE + p is window
    # position s of channel p.
    channel_tile = Traversal.over(
        (2, 9, 4), ((0, 2, 1), (2, 2, 2), (1, 3, 3)), ((1, 3, 1), (2, 2, 1))
    )
    assert boundary.transport.name == "in0_V" and boundary.form == channel_tile
    assert x.sides.source.sequence == boundary.sequence
    framed = x.sides.sink.sequence
    assert framed.form == boundary.form
    assert [level.beats for level in framed.markers] == [3]
    module = built(configured, "matmul")
    compute = parameters(module, "matmul.compute.int8_dsp58")
    # An input_gen passing three-beat frames in order, closing each.
    markers = parameters(module, "x.adapter.input_gen.input_gen")
    assert (markers["FM_SIZE"], markers["DIMS"], markers["COEFS"]) == (3, "'{3}", "'{1}")
    assert compute["ACTIVATION_BROADCASTING"] == 0
    assert compute["CORE"] == '"dotp_8sx9_dsp58"'
    widths = {
        port.name: next(item.width for item in port.signals if item.logical == "tdata")
        for port in module.abi.pins
        if port.name.endswith("_V")
    }
    # PE channels of SIMD window positions per activation beat; the result is exact.
    assert widths == {"in0_V": 24, "in1_V": 24, "out0_V": 24}
    (framing,) = [
        link.markers
        for link in module.fragment.links
        if link.sink.instance == "matmul.compute.int8_dsp58" and link.markers
    ]
    assert framing == (("olst", 0, "s_axis_input_tlast", None),)


def test_a_dense_form_replays_each_row_per_output_fold():
    dense = point(form=Form.DENSE)
    assert dense.member("x").steps.kinds == (Step.REORDER, Step.MARKERS)
    compute = parameters(built(dense, "matmul"), "matmul.compute.int8_dsp58")
    assert compute["ACTIVATION_BROADCASTING"] == 1


def test_only_the_int8_dsp58_core_reads_a_depthwise_form():
    # The packed core refuses the form: selecting it is refused, naming the core's rule,
    # the native realization standing in its way.
    with pytest.raises(ValueError, match="compute.packed.core_supported: dotp-form"):
        point("packed")
    facts = {**FACTS, "pe": 2, "simd": 3}
    # External weights exclude the dense realization; natively, one core fits. On
    # DSP48E2 neither is possible, and the realization's policy (first valid) says why.
    assert matmul_assembly(**facts).module is not None
    with pytest.raises(
        ValueError,
        match="realization: choice-no-valid-value: .*'native' breaks native_readable; "
        "'dense' breaks carried, realization_supported",
    ):
        matmul_assembly(**{**facts, "platform": FULL_DSP48E2})


@pytest.mark.parametrize(
    "platform,activation,offered",
    [
        (FULL_DSP58, "INT4", ("native", "dense")),
        (FULL_DSP48E2, "INT4", ("dense",)),  # no core reads it natively on DSP48E2
        (FULL_DSP58, "INT12", ("dense",)),  # wider than the INT8 core's lanes
    ],
)
def test_native_is_offered_only_where_a_core_reads_the_depthwise_operands(
    platform, activation, offered
):
    # The realization states its need where its case is declared, so a native case
    # whose core would refuse it is not possible (it would leave compute no case).
    facts = {**FACTS, "platform": platform, "activation_dtype": DataType[activation]}
    weights = tuple(tuple((r + c) % 5 - 2 for c in range(4)) for r in range(9))
    space = matmul_point(**facts, weights=weights)
    ((_, realization),) = space.viable([("matmul", "realization")])
    assert realization.cases == offered
    if offered == ("dense",):
        refused = {
            item.value: [
                (space.pair("matmul", rule.rule)[1], rule.standing) for rule in item.broken
            ]
            for item in realization.values or ()
            if isinstance(item, inspection.Refused)
        }
        assert refused == {"native": [("native_readable", ())]}
        # Committed, native is refused by the same rule (a commit that breaks a rule is).
        trial = space.trial(keyed({"matmul.realization": "native"}))
        assert not trial.accepted
        assert [(f.code, f.owner) for why in trial.refused.values() for f in why] == [
            ("matmul-native", "native_readable")
        ]


def test_depthwise_cyclic_weights_are_the_channel_tile():
    weights = [[(c + k) % 16 - 8 for c in range(4)] for k in range(9)]
    built = matmul_assembly(
        **FACTS,
        pe=2,
        simd=3,
        realization="native",
        weight_delivery=WeightDelivery.MEMSTREAM,
        weights=weights,
    )
    assert "in1_V" not in {port.name for port in built.module.abi.pins}
    # Channel folds, then window folds; PE channels of SIMD taps a beat, SIMD fastest.
    assert len(built.initializer) == 2 * 3
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (12, 12, 4)


def test_one_network_carries_the_weights_of_whichever_realization_is_committed():
    # The weight channel carries MatMul's view of its weights (``w_carried``), which
    # follows the realization: open, the channel's tensor waits on it, MatMul's.
    weights = tuple(tuple((c + k) % 7 - 3 for c in range(4)) for k in range(9))
    point = matmul_point(**FACTS, weights=weights)
    pending = point.member("w").query(Channel.tensor)
    assert isinstance(pending, Unresolved)
    assert {finding.owner for finding in pending.findings} == {"matmul.realization"}
    for realization, shape, core in (
        ("native", (9, 4), "int8_dsp58"),
        ("dense", (36, 4), "packed"),
    ):
        committed = choose(point, {"matmul.realization": realization})
        assert committed.member("matmul").w_carried.shape == shape
        # The realization and the kernel's choices commit in one batch, then the
        # channels'.
        folded = choose(
            point,
            {
                "matmul.realization": realization,
                "matmul.compute": core,
                "matmul.pe": 2,
                "matmul.simd": 3,
                f"matmul.compute.{core}.compute_pumping": False,
                **({"matmul.compute.packed.reducer": "tree"} if core == "packed" else {}),
            },
        )
        # Whether the weight channel has a value is the weights' presence, a fact: its
        # source is placed by it, no choice.
        assert folded.member("w").valued and folded.member("w").tensor.shape == shape
        assert ("w", "source") not in folded.chosen()
        configured = choose(
            folded,
            {
                "w.transport": "direct",
                "w.source.ram_style": "auto",
                "w.source.pumped_memory": False,
            },
        )
        module = module_of(with_direct_transports(with_adapter_memories(configured)), "matmul")
        assert isinstance(module, Available), module
        assert f"matmul.compute.{core}" in labels(module.value)


# By channel, then stored (k, n): window by channel.
BY_CHANNEL = ((1, -2, 3, -4), (2, 3, -1, 0), (-3, 1, 2, 1))
WEIGHTS = tuple(zip(*BY_CHANNEL))
DENSE = dict(
    m=2,
    k=4,
    n=3,
    form=Form.DEPTHWISE,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    pe=3,
    simd=4,
    weight_delivery=WeightDelivery.MEMSTREAM,
    weights=WEIGHTS,
)


def test_a_dense_realization_reads_window_by_channel_rows_against_block_diagonal_weights():
    # On DSP48E2 only the dense realization computes it: the packed core.
    built = matmul_assembly(platform=FULL_DSP48E2, **DENSE)
    assert labels(built.module) == [
        "x.adapter.input_gen.input_gen",
        "w.source.memstream",
        "matmul.compute.packed",
    ]
    compute = parameters(built.module, "matmul.compute.packed")
    assert compute["ACTIVATION_BROADCASTING"] == 1 and compute["SIMD"] == 4
    # Rows of 4 x 3 = 12 activations in three SIMD beats, replayed once (PE = C).
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (6, 6, 2)
    # W'[k * 3 + c', c] = W[k, c] where c' = c: each weight beat is one tile of W'.
    blocks = [
        [BY_CHANNEL[c][k] if other == c else 0 for k in range(4) for other in range(3)]
        for c in range(3)
    ]
    # Their owner states the INT4-typed weights in [-4, 3] as INT3: three bits a weight.
    mask = 0x7
    expected = tuple(
        sum((blocks[p][start + s] & mask) << (3 * (p * 4 + s)) for p in range(3) for s in range(4))
        for start in range(0, 12, 4)
    )
    assert built.initializer == expected
    # The result precision is the operation's: each channel's window of 4 weights over
    # INT4 activations, not 12 (zeros add nothing): [-74, 76].
    assert built.result_dtype == DataType["INT8"]


def test_the_dense_realization_needs_known_weights_and_either_may_be_chosen_on_dsp58():
    external = {**DENSE, "weight_delivery": WeightDelivery.EXTERNAL, "weights": None}
    with pytest.raises(ValueError, match="matmul-realization"):
        matmul_assembly(platform=FULL_DSP48E2, realization="dense", **external)
    # Left out on DSP58, where both are possible, the realization takes its policy, first
    # valid in the flow order: native, and the INT8 core that reads it.
    default = matmul_assembly(platform=FULL_DSP58, **DENSE)
    assert default.network.chosen()[("matmul", "realization")] == "native"
    assert "matmul.compute.int8_dsp58" in labels(default.module)
    for realization, core in (("native", "int8_dsp58"), ("dense", "packed")):
        built = matmul_assembly(platform=FULL_DSP58, realization=realization, core=core, **DENSE)
        assert f"matmul.compute.{core}" in labels(built.module)


@pytest.mark.parametrize(
    "weights,delivery,narrow",
    [
        # Stated as INT3, whose minimum -4 they hold: not narrow.
        (WEIGHTS, WeightDelivery.MEMSTREAM, 0),
        # Stated as INT3, without its -4: narrow.
        (((-3, 0, 0), (0, 3, 0), (0,) * 3, (0,) * 3), WeightDelivery.MEMSTREAM, 1),
        (((-8, 0, 0), (0,) * 3, (0,) * 3, (0,) * 3), WeightDelivery.MEMSTREAM, 0),
        (None, WeightDelivery.EXTERNAL, 0),  # weights at run time promise nothing
    ],
)
def test_narrow_weights_follow_known_weights(weights, delivery, narrow):
    facts = {**DENSE, "form": Form.DENSE, "n": 3, "pe": 3}
    built = matmul_assembly(
        platform=FULL_DSP48E2, **{**facts, "weights": weights, "weight_delivery": delivery}
    )
    assert parameters(built.module, "matmul.compute.packed")["NARROW_WEIGHTS"] == narrow


@pytest.mark.parametrize("realization", ["native", "dense"])
@pytest.mark.parametrize("core", ["packed", "int8_dsp58"])
def test_a_cores_folding_factors_are_listed_from_facts_and_simd_fits_its_datapath(
    realization, core
):
    """PE's and SIMD's domains, MatMul's, read the operation's facts alone, whichever
    realization: PE the channels' divisors, SIMD the divisors of the widest K (window times
    channels). A native datapath's K is the window's, so the SIMDs that divide the widest K
    and not the window's are listed and refused by the core's fit (``dotp-simd``). The
    packed core refuses a native depthwise operation: selecting it is refused, and the
    factors are listed with the core still open, each possible as far as anything
    decides."""
    weights = tuple(tuple((r + c) % 5 - 2 for c in range(4)) for r in range(9))
    network = matmul_point(realization=realization, **FACTS, weights=weights)
    selected = network.trial(keyed({"matmul.compute": core}))
    assert selected.accepted == (realization == "dense" or core == "int8_dsp58")
    if not selected.accepted:
        assert [(f.code, f.owner) for why in selected.refused.values() for f in why] == [
            ("dotp-form", "compute.packed.core_supported")
        ]
    kernel = (selected.network if selected.accepted else network).member("matmul")
    pe, simd = inspection.viable(
        kernel,
        [MatMulKernel.pe, MatMulKernel.simd],
    )
    widest = (1, 2, 3, 4, 6, 9, 12, 18, 36)
    assert [item.value for item in pe.values or ()] == [1, 2, 4] and pe.cases == (1, 2, 4)
    assert [item.value for item in simd.values or ()] == list(widest)
    if realization == "dense" or not selected.accepted:
        assert simd.cases == widest
    else:
        assert simd.cases == (1, 3, 9)
        refused = {
            item.value: {
                (rule.rule, finding.code)
                for rule in item.broken
                for finding in rule.result.findings
            }
            for item in simd.values or ()
            if isinstance(item, inspection.Refused)
        }
        assert set(refused) == {2, 4, 6, 12, 18, 36}
        assert all(
            found == {("matmul.compute.int8_dsp58.simd_fits", "dotp-simd")}
            for found in refused.values()
        )
