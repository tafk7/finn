# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel with a known value carries a ``source``: MatMul's weights on its weight channel.

The MatMul sits in a network (``support.helpers.matmul_point``) that gives it its
weights: the weight channel carries what MatMul states for it, its ``contents`` and
its tensor over their range. Known weights give the channel a value, so its
``source`` applies (``memstream``, its one candidate, set first valid as the network
builds the channel: the interim channel rule) and stores them; unknown weights leave the
channel without a source, and it is the boundary ``in1_V``. The channel's contracts are
MatMul's, known once its folding is: its choices are made after the kernel's. These
tests cover the source, laziness, the memory's own choices, their replay on another
network, the platform's capabilities, the invariant that a channel with a value has its
source as its only producer, and the weight channel's FIFO.
"""

from dataclasses import replace

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.ranges import TensorRange
from support.helpers import (
    FULL_DSP48E2,
    WeightDelivery,
    built,
    choose,
    codes,
    keyed,
    labels,
    matmul_assembly,
    matmul_point,
    module_of,
    placed,
    with_adapter_memories,
    with_direct_transports,
)

from finn.core.space import (
    Available,
    Inapplicable,
    Rejected,
    Unresolved,
    inspection,
)
from finn.core.space.errors import ConfigurationError
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.module import Composed
from finn.kernels.base import NETLIST
from finn.kernels.channels import Channel
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.matmul import KnownWeights, MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Fact, Interface, Network, NetworkError, wire
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.capabilities import Platform

STORED_INSTANCE = "w.source.memstream"
ADAPTER_INSTANCE = "x.adapter.input_gen.input_gen"
# Written by output, stored (k, n).
BY_OUTPUT = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
WEIGHTS = tuple(zip(*BY_OUTPUT))
FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)
"""The harness's MatMul (``matmul_point``, ``matmul_assembly``)."""
KNOWN = dict(platform=FULL_DSP48E2)
"""MatMul's facts beside its tensors and its weights."""
INT3, INT8 = ScalarEncoding(DataType["INT3"]), ScalarEncoding(DataType["INT8"])
# The result of WEIGHTS' columns over INT3 activations (K7): [-30, 40].
INT7 = ScalarEncoding(DataType["INT7"])
FOLDING = {
    "matmul.compute": "packed",
    "matmul.compute.packed.compute_pumping": False,
    "matmul.compute.packed.reducer": "tree",
}


def base(**facts):
    return matmul_point(**{**FACTS, **facts})


def folded(point, *, pe=2, simd=2):
    """``point`` with MatMul's core and folding committed: its channels' contracts known."""
    return choose(point, {**FOLDING, "matmul.pe": pe, "matmul.simd": simd})


def handle(point, key):
    """The choice ``key`` of ``point``, by name, listed or not (its model's)."""
    every = inspection.decisions(inspection.model(point))
    return next(item.reference for item in every if item.key == key)


def configured(point, *, style=None, pe=2, simd=2):
    choices = {"w.transport": "direct"}
    if style is not None:
        choices |= {
            "w.source.ram_style": style,
            "w.source.pumped_memory": False,
        }
    # The activation channel's adapter is forced by the folding; its memory is the flow's.
    return with_direct_transports(
        with_adapter_memories(choose(folded(point, pe=pe, simd=simd), choices))
    )


def stored(platform: Platform) -> Network:
    """MatMul with known weights, its channels on ``platform``: the weight channel
    carries what MatMul states for it."""
    return Network(
        {
            "x": wire(
                tensor=Tensor((3, 4), INT3),
                port="in0_V",
                consumer=Interface("matmul", "x_contract"),
                platform=platform,
            ),
            "w": wire(
                tensor=Fact("matmul", "carried_weights"),
                contents=Fact("matmul", "stored_weights"),
                consumer=Interface("matmul", "w_contract"),
                platform=platform,
            ),
            "y": wire(
                tensor=Tensor((3, 4), INT7),
                port="out0_V",
                producer=Interface("matmul", "y_contract"),
                platform=platform,
            ),
            "matmul": MatMulKernel(
                **{**KNOWN, "platform": platform},
                x_tensor=Tensor((3, 4), INT3),
                weights=KnownWeights(IntegerTensorValue.of(WEIGHTS), DataType["INT3"]),
            ),
        }
    )


def instance_parameters(point, instance):
    return dict(placed(built(point, "matmul"), instance).parameters)


def owners(result):
    return {finding.owner for finding in result.findings}


def test_known_weights_give_the_weight_channel_its_source():
    external = configured(base())
    stored_ = configured(base(weights=WEIGHTS), style="block")
    compute = "matmul.compute.packed"
    for point, ports, instances in (
        (external, {"in0_V", "in1_V", "out0_V"}, [ADAPTER_INSTANCE, compute]),
        (stored_, {"in0_V", "out0_V"}, [ADAPTER_INSTANCE, STORED_INSTANCE, compute]),
    ):
        module = built(point, "matmul")
        assert isinstance(module, Composed)
        assert {p.name for p in module.abi.pins if isinstance(p, Bus)} == ports
        assert labels(module) == instances
    # Known weights: the channel has a value, so its source is placed: no choice, nothing
    # stored for it.
    w = stored_.member("w")
    assert w.valued and ("w", "source") not in stored_.chosen(("w",))
    assert isinstance(w.source, MemStreamKernel)
    assert w.source.image == (0x22C, 0x6BE, 0xDD3, 0x941)
    # The weight channel's producer is its source, a leaf below the channel at the label
    # the channel states for it.
    assert w.sides.source_owner == "source.memstream"
    assert instance_parameters(stored_, STORED_INSTANCE)["RAM_STYLE"] == '"block"'
    # Unknown weights: no value, no source; the channel is the boundary in1_V.
    unvalued = external.member("w")
    assert not unvalued.valued
    assert isinstance(unvalued.query(Channel.source), Inapplicable)
    assert unvalued.sides.source_owner is None
    # MatMul's netlist is its cores' alone.
    matmul = stored_.member("matmul")
    assert [item.node for item in matmul.netlists] == ["compute.packed"]


def test_an_unvalued_channel_never_demands_its_source():
    w = configured(base()).member("w")
    evidence = inspection.explain(w, Channel.netlist)
    visited = {node.declaration.key for node in evidence.nodes}
    # Whether the channel has a value is read (the weights' presence), never the value.
    assert {"valued", "streams", "sides"} <= visited
    # Nothing of the source runs: the channel's members are summed (its netlist), so its
    # source's guard is read, and is False; everything below it is inapplicable.
    reached = {
        node.declaration.key: node.result
        for node in evidence.nodes
        if node.declaration.key.startswith("source.")
    }
    assert reached.pop("source.$guard", Available(False)) == Available(False)
    assert all(isinstance(result, Inapplicable) for result in reached.values())


def test_the_channel_owns_its_source_and_its_choices():
    records = {item.key: item for item in inspection.decisions(Channel)}
    # The source is placed where the channel has a value: no choice of its own.
    assert "source" not in records
    local = records["source.ram_style"]
    # The choice belongs to the memory kernel the channel places as its source.
    assert local.scope == "source" and not local.selector
    assert not any(key[1].startswith("memory") for key in base(weights=WEIGHTS).decisions())
    # Its choices apply as soon as the source is set: the explicit styles, LUTRAM first
    # for a table of 64 bits, and ``auto`` last.
    chosen = folded(base(weights=WEIGHTS)).member("w").source
    assert chosen.field(MemStreamKernel.ram_style).candidates() == Available(
        ("distributed", "block", "ultra", "auto")
    )
    # Without a value the source is absent, and so are its choices: one set is accepted
    # and kept, dormant, out of play.
    unvalued = folded(base()).trial(keyed({"w.source.ram_style": "block"}))
    assert unvalued.accepted
    states = dict(unvalued.network.states(("w",)))
    assert states[("w", "source.ram_style")] == inspection.Dormant("block")


def test_a_channel_without_contents_lists_none_of_its_sources_choices():
    # SZ113: without contents the channel's source is absent under the facts alone (its
    # guard, ``valued``, reads the facts only), so the source's choices are in none of the
    # channel's lists: its decisions, its states, the network's states and open choices.
    # Asked by name they still answer, and set by name one is accepted, dormant, and is then
    # listed: a stored choice is never pruned (SZ119 (5)).
    source = {"source.ram_style", "source.pumped_memory"}
    network = configured(base())
    w = network.member("w")
    assert not w.valued
    assert source <= {item.key for item in inspection.decisions(Channel)}
    assert source <= {item.key for item in inspection.decisions(inspection.model(w))}
    assert not source & {item.key for item in inspection.decisions(w)}
    assert not source & {key for key, _ in inspection.states(w)}
    assert not source & {key for (_, key), _ in network.states(("w",))}
    assert not source & {key for _, key in network.undecided()}
    (asked,) = inspection.viable(w, [handle(w, "source.ram_style")])
    assert asked.key == "source.ram_style" and asked.values is not None
    stored = w.with_choices({handle(w, "source.ram_style"): "block"})
    states = dict(inspection.states(stored))
    assert states["source.ram_style"] == inspection.Dormant("block")
    assert "source.pumped_memory" not in states
    assert "source.ram_style" in {item.key for item in inspection.decisions(stored)}


def test_the_stored_memory_takes_its_choices_with_its_channel_and_refuses_bad_weights():
    # The memory's choices are set, first valid, as the network builds the weight channel
    # (the interim channel rule): LUTRAM for a table of 64 bits, unpumped.
    built_with = configured(base(weights=WEIGHTS))
    assert built_with.chosen(("w",)) == {
        ("w", "source.ram_style"): "distributed",
        ("w", "source.pumped_memory"): False,
        ("w", "transport"): "direct",
    }
    assert isinstance(module_of(built_with, "matmul"), Available)
    # Weights outside their type are refused where MatMul states their range on the
    # weight channel's tensor; the channel, which carries it, is refused with it.
    outside = base(weights=((4,) * 4,) * 4)
    refused = outside.member("matmul").query(MatMulKernel.carried_weights)
    assert isinstance(refused, Rejected)
    assert codes(refused) == {"dtype-storage"}
    assert owners(refused) == {"carried_weights"}
    assert codes(outside.query("w", NETLIST)) == {"dtype-storage"}
    # The view presenting it is a product: refused with MatMul, whose cores read the
    # refused value.
    presented = outside.member("matmul").query(MatMulKernel.w_carried)
    assert codes(presented) == {"dtype-storage"}
    # A shape error is refused by MatMul: its K is its activations', not the weights'.
    wrong = base(weights=((0,),)).member("matmul").query(MatMulKernel.carried)
    assert isinstance(wrong, Rejected) and codes(wrong) == {"matmul-tensor"}


def test_the_result_range_follows_what_the_weight_channel_carries():
    """With a value on the weight channel, each output column's dot products over the
    activation datatype, unioned over columns; without one, the datatypes' (K7). The
    core states the range on what it produces."""
    stored_, external = base(weights=WEIGHTS).member("matmul"), base().member("matmul")
    assert stored_.result_range == (-30, 40)
    assert stored_.result_type == DataType["INT7"]
    assert external.result_range == (-48, 64)
    assert external.result_type == DataType["INT8"]
    core = configured(base(weights=WEIGHTS), style="block").member("matmul").compute
    assert core.y_element == ScalarEncoding(DataType["INT7"], TensorRange(-30, 40))
    assert core.parameters["ACCU_WIDTH"] == 7


@pytest.mark.parametrize(
    "weights,value_range,result",
    [
        (((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), (-4, 3), "INT3"),  # identity
        (((0,) * 4,) * 4, (0, 0), "BINARY"),  # zeros
    ],
)
def test_a_narrow_result_range_is_the_result_and_the_accumulator(weights, value_range, result):
    """The result is the columns' range in its smallest encoding, narrower than one INT3 x
    INT3 product (3 + 3 - 1 bits), and the packed core accumulates in it: FinnLib's
    packed dotp computes the sum modulo 2**ACCU_WIDTH, exact where the range bounds it."""
    point = configured(base(weights=weights), style="block")
    matmul = point.member("matmul")
    assert matmul.result_range == value_range
    assert matmul.result_type == DataType[result]
    assert matmul.compute.y_element == ScalarEncoding(DataType[result], TensorRange(*value_range))
    assert matmul.compute.parameters["ACCU_WIDTH"] == DataType[result].bitwidth()
    assert isinstance(module_of(point, "matmul"), Available)


def test_the_cores_do_not_wait_on_the_source():
    point = choose(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    point = choose(
        point,
        {
            "matmul.pe": 4,
            "matmul.simd": 2,
            "matmul.compute.packed.compute_pumping": True,
            "matmul.compute.packed.reducer": "tree",
        },
    )
    # The weight channel, built once the folding is known, takes its memory's choices
    # (the interim channel rule), so the module is complete.
    assert isinstance(module_of(point, "matmul"), Available)
    # The packed core does not wait for the memory: its weights' range is the one their
    # owner states on the channel.
    compute = point.member("matmul").compute
    assert isinstance(compute.inspect(DotpAxiKernel.netlist).accepted_result, Available)
    assert compute.parameters["NARROW_WEIGHTS"] == 0  # the weights hold -4
    assert inspection.validity(compute).status == "valid"
    folding = choose(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    folding = choose(folding, {"matmul.simd": 4})
    # Folding refusals come before a PE or a memory style is chosen.
    trial = folding.trial(keyed({"matmul.pe": 3}))
    assert "domain-membership" in {
        finding.code for why in trial.refused.values() for finding in why
    }


def test_the_sources_choices_round_trip_through_another_network():
    point = configured(base(weights=WEIGHTS), style="block")
    saved = point.chosen()
    # Every choice is stored: the source's and the adapter's memory too, set where the
    # network builds their channels (the interim channel rule).
    assert sorted(f"{owner}.{key}" for owner, key in saved) == [
        "matmul.compute",
        "matmul.compute.packed.compute_pumping",
        "matmul.compute.packed.reducer",
        "matmul.pe",
        "matmul.simd",
        "w.source.pumped_memory",
        "w.source.ram_style",
        "w.transport",
        "x.adapter_ram_style",
        "x.transport",
        "y.transport",
    ]
    replayed = base(weights=WEIGHTS).with_choices(saved)
    assert built(replayed, "matmul") == built(point, "matmul")
    # Replay under different facts: the same choices, a new image.
    other = base(weights=tuple(row[::-1] for row in WEIGHTS)).with_choices(saved)
    assert other.member("w").source.image != point.member("w").source.image
    # Without known weights the channel has no source: its choices are kept, dormant.
    unvalued = base().with_choices(saved)
    states = dict(unvalued.states(("w",)))
    assert states[("w", "source.ram_style")] == inspection.Dormant("block")
    # Facts that invalidate a saved folding refuse replay atomically: five outputs.
    wider = tuple((*row, 0) for row in WEIGHTS)
    with pytest.raises(NetworkError, match="matmul.pe"):
        base(weights=wider, n=5).with_choices(saved)


def test_several_weight_sets_without_known_weights_leave_the_set_channel_unused():
    # No value, no source: nothing consumes the set index, which refuses itself.
    point = configured(base(weight_sets=2))
    answer = module_of(point, "matmul")
    assert isinstance(answer, Rejected)
    assert "channel-unused" in codes(answer)


def test_a_source_whose_rule_is_false_refuses_the_change_that_places_it():
    # Several sets need the set channel; a memory without one refuses itself.
    members = {
        **{name: stored(FULL_DSP48E2).wiring(name) for name in ("x", "y")},
        "w": wire(
            tensor=Fact("matmul", "carried_weights"),
            contents=Fact("matmul", "stored_weights"),
            sets=2,
            consumer=Interface("matmul", "w_contract"),
            platform=FULL_DSP48E2,
        ),
        "matmul": MatMulKernel(
            **KNOWN,
            x_tensor=Tensor((3, 4), INT3),
            weights=KnownWeights(IntegerTensorValue.of((WEIGHTS, WEIGHTS)), DataType["INT3"]),
        ),
    }
    folding = {**FOLDING, "matmul.pe": 2, "matmul.simd": 2}
    # The folding gives the weight channel its value, so its source is placed, and the
    # memory's own rule is false: the channel would newly read refused, so the fold is
    # refused by that rule, the channel's own refusal. A strategy can be stopped by a
    # channel it does not decide, where forcing accepted the fold and left the channel
    # silently refused.
    # The memory's output order, which the channel's sides read, refuses the same way: so
    # do the channel's rules and guards that read its sides.
    memory = {
        ("memstream-set-channel", "source.selected"),
        ("memstream-set-channel", "source.output_sequence"),
    }
    trial = Network(members).trial(keyed(folding))
    assert set(trial.refused) == {("w", "")}
    assert {
        (why.code, why.owner) for why in trial.refused[("w", "")] if why.code != "guard-refused"
    } == memory
    # Bound with the folding stored, the channel is refused by the memory's rule.
    w = Network(members, stored=keyed(folding)).member("w")
    validity = inspection.validity(w)
    assert validity.status == "refused"
    assert "source.selected" in {piece.owner for piece in validity.pieces}
    assert {
        (finding.code, finding.owner)
        for piece in validity.pieces
        for finding in piece.result.findings
        if finding.code != "guard-refused"
    } == memory


@pytest.mark.parametrize(
    ("platform", "refused"),
    (
        (FULL_DSP48E2, set()),
        (replace(FULL_DSP48E2, uram=False), {"uram-absent"}),
        (replace(FULL_DSP48E2, uram_init=False), {"uram-init"}),
    ),
)
def test_the_platform_narrows_the_source_memory(platform, refused):
    point = folded(stored(platform))
    trial = point.trial(keyed({"w.source.ram_style": "ultra"}))
    assert trial.accepted == (not refused)
    if refused:
        # The style's domain holds it, and the memory's rule refuses it on the platform:
        # the change is refused, naming the rule.
        ((rule, why),) = trial.refused.items()
        assert rule[0] == "w" and rule[1].startswith("source.")
        assert {finding.code for finding in why} == refused
    # The platform does not narrow pumping: the doubled clock is the shell's, which
    # its root supplies.
    ((_, pumping),) = point.viable([("w", "source.pumped_memory")])
    assert pumping.cases == (False, True)


def test_a_channel_with_a_value_has_its_source_as_its_only_producer():
    x_tensor, w_tensor, y_tensor = (
        Tensor((3, 4), INT3),
        Tensor((4, 4), INT3),
        Tensor((3, 4), INT8),
    )
    produced = Network(
        {
            "x": wire(
                tensor=x_tensor,
                port="in0_V",
                consumer=Interface("matmul", "x_contract"),
                platform=FULL_DSP48E2,
            ),
            "w": wire(
                tensor=w_tensor,
                port="in1_V",
                consumer=Interface("matmul", "w_contract"),
                platform=FULL_DSP48E2,
            ),
            "y": wire(
                tensor=y_tensor,
                contents=IntegerTensorValue.of(((0,) * 4,) * 3),
                producer=Interface("matmul", "y_contract"),
                platform=FULL_DSP48E2,
            ),
            "matmul": MatMulKernel(platform=FULL_DSP48E2, x_tensor=x_tensor, weights=w_tensor),
        }
    )
    answer = folded(produced).member("y").query(Channel.sides)
    assert isinstance(answer, Rejected) and codes(answer) == {"channel-users"}
    assert "only producer" in answer.findings[0].message


@pytest.mark.parametrize("delivery", tuple(WeightDelivery))
def test_public_adapter_matches_the_space_path(delivery):
    known = delivery is not WeightDelivery.EXTERNAL
    adapted = matmul_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=delivery,
        weights=[list(row) for row in WEIGHTS] if known else None,
    )
    point = configured(base(weights=WEIGHTS) if known else base(), style="auto" if known else None)
    assert adapted.module == built(point, "matmul")


@pytest.mark.parametrize("style", ("auto", "distributed", "block", "ultra"))
def test_ram_style_reaches_the_stored_memory(style):
    assembled = matmul_assembly(
        **{**FACTS, "pe": 2, "simd": 2},
        weight_delivery=WeightDelivery.MEMSTREAM,
        weights=WEIGHTS,
        ram_style=style,
    )
    assert ("RAM_STYLE", f'"{style}"') in placed(assembled.module, STORED_INSTANCE).parameters


def test_adapter_rejects_an_unknown_ram_style():
    with pytest.raises(ValueError, match="domain"):
        matmul_assembly(
            **{**FACTS, "pe": 2, "simd": 2},
            weight_delivery=WeightDelivery.MEMSTREAM,
            weights=WEIGHTS,
            ram_style="lutram",
        )


# -- the weight channel's transport slot --------------------------------------------------

FIFO = "w.transport.fifo.buffer"
FIFO_DEPTH = "w.transport.fifo.depth"
FIFO_RAM_STYLE = "w.transport.fifo.buffer.ram_style"


def buffered(point, depth=None):
    choices = {"w.transport": "fifo", FIFO_RAM_STYLE: "auto"}
    if depth is not None:
        choices[FIFO_DEPTH] = depth
    return choose(point, choices)


def test_a_buffered_channel_places_a_fifo_between_its_producer_and_consumer():
    for weights in (None, WEIGHTS):
        point = configured(
            base(weights=weights) if weights else base(), style=("auto" if weights else None)
        )
        module = built(buffered(point, depth=32), "matmul")
        assert FIFO in labels(module)
        hops = {(link.source.instance, link.sink.instance) for link in module.fragment.links}
        producer = STORED_INSTANCE if weights else None
        assert (producer, FIFO) in hops
        assert (FIFO, "matmul.compute.packed") in hops
        assert dict(placed(module, FIFO).parameters)["DEPTH"] == 32


def test_fifo_depth_is_owned_by_the_channel_and_only_demanded_when_selected():
    point = configured(base())
    w = point.member("w")
    records = {item.key: item for item in inspection.decisions(w)}
    assert records["transport.fifo.depth"].scope == "transport.fifo"
    depth = handle(w, "transport.fifo.depth")
    assert isinstance(w.field(depth).query(), Inapplicable)
    undecided = buffered(point)
    assert isinstance(module_of(undecided, "matmul"), Unresolved)
    with pytest.raises(ConfigurationError):
        w.with_choices(
            {
                handle(w, "transport"): "fifo",
                handle(w, "transport.fifo.buffer.ram_style"): "auto",
                depth: 1,
            }
        )
    deep = buffered(point, depth=64)
    saved = deep.chosen()
    assert ("w", "transport.fifo.depth") in saved
    assert built(base().with_choices(saved), "matmul") == built(deep, "matmul")
    # In the channel's space, removing the FIFO keeps its depth and memory style,
    # dormant: out of play, and back with the FIFO.
    fifo = deep.member("w")
    direct = fifo.try_with_choices({handle(fifo, "transport"): "direct"})
    assert direct.accepted and set(direct.dormant) == {
        "transport.fifo.depth",
        "transport.fifo.buffer.ram_style",
    }
    assert direct.instance.netlist == w.netlist
    # In the network the same: nothing is dropped, and the module is the direct one.
    trial = deep.trial(keyed({"w.transport": "direct"}))
    assert trial.accepted and not trial.rederived
    assert ("w", "transport.fifo.depth") in trial.network.chosen()
    assert built(trial.network, "matmul") == built(point, "matmul")


def test_adapter_places_the_weight_fifo_on_request():
    assembled = matmul_assembly(**{**FACTS, "pe": 2, "simd": 2}, weight_fifo_depth=16)
    assert FIFO in labels(assembled.module)
