# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel with a known value carries a ``source``: MatMul's weights on its weight channel.

The MatMul sits in a network (``kernels.helpers.matmul_point``) that gives it its
weights: the weight channel carries what MatMul states for it, its ``contents`` and
its tensor over their range. Known weights give the channel a value, so its
``source`` applies (``memstream``, its one candidate, forced) and stores them;
unknown weights leave the channel without a source, and it is the boundary ``in1_V``.
The channel's ends are MatMul's, known once its folding is: its choices are committed
after the kernel's. These tests cover the forced source, laziness, the memory's own
choices, their replay on another network, the platform's capabilities, the invariant
that a channel with a value has its source as its only producer, and the weight
channel's FIFO.
"""

from dataclasses import replace

import pytest
from qonnx.core.datatype import DataType

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
from finn.kernels.configure import commit
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Fact, Interface, Network, NetworkError, wire
from finn.platform.capabilities import Platform
from kernels.helpers import (
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
    """``point`` with MatMul's core and folding committed: its channels' ends known."""
    return choose(
        point, {**FOLDING, "matmul.compute.packed.pe": pe, "matmul.compute.packed.simd": simd}
    )


def handle(point, key):
    return next(item.reference for item in inspection.decisions(point) if item.key == key)


def configured(point, *, style=None, pe=2, simd=2):
    choices = {"w.transport": "direct"}
    if style is not None:
        choices |= {
            "w.source.memstream.ram_style": style,
            "w.source.memstream.pumped_memory": False,
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
                consumer=Interface("matmul", "x_end"),
                platform=platform,
            ),
            "w": wire(
                tensor=Fact("matmul", "w_carried"),
                contents=Fact("matmul", "w_stored"),
                consumer=Interface("matmul", "w_end"),
                platform=platform,
            ),
            "y": wire(
                tensor=Tensor((3, 4), INT7),
                port="out0_V",
                producer=Interface("matmul", "y_end"),
                platform=platform,
            ),
            "matmul": MatMulKernel(
                **{**FACTS, "platform": platform},
                x_tensor=Tensor((3, 4), INT3),
                w_contents=WEIGHTS,
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
    # Known weights: the channel has a value, and its one source is forced, never committed.
    w = stored_.member("w")
    assert w.valued and stored_.forced(("w",))[("w", "source")] == "memstream"
    state = w.field(handle(w, "source")).state
    assert isinstance(state, Available) and state.value.status == "unassigned"
    assert isinstance(w.source, MemStreamKernel)
    assert w.source.image == (0x22C, 0x6BE, 0xDD3, 0x941)
    # The weight channel's producer is its source, a leaf below the channel.
    assert w.endpoints.source_owner == "source.memstream"
    assert instance_parameters(stored_, STORED_INSTANCE)["RAM_STYLE"] == '"block"'
    # Unknown weights: no value, no source; the channel is the boundary in1_V.
    unvalued = external.member("w")
    assert not unvalued.valued
    assert isinstance(unvalued.query(Channel.source), Inapplicable)
    assert unvalued.endpoints.source_owner is None
    # MatMul's netlist is its cores' alone, and it derives the same either way.
    matmul = stored_.member("matmul")
    assert [item.node for item in matmul.netlists] == ["compute.packed"]
    assert external.member("matmul").producer_identity() == matmul.producer_identity()


def test_an_unvalued_channel_never_demands_its_source():
    w = configured(base()).member("w")
    evidence = inspection.explain(w, Channel.netlist)
    visited = {node.declaration.key for node in evidence.nodes}
    # Whether the channel has a value is read (the weights' presence), never the value.
    assert {"valued", "ends", "endpoints"} <= visited
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
    assert records["source"].selector and records["source"].cases == ("memstream",)
    local = records["source.memstream.ram_style"]
    # The choice belongs to the memory kernel the channel places as its source.
    assert local.scope == "source.memstream" and not local.selector
    assert not any(key[1].startswith("memory") for key in base(weights=WEIGHTS).decisions())
    # Its choices apply as soon as the source is forced: the explicit styles, LUTRAM first
    # for a table of 64 bits, and ``auto`` last.
    chosen = folded(base(weights=WEIGHTS)).member("w").source
    assert chosen.field(MemStreamKernel.ram_style).candidates() == Available(
        ("distributed", "block", "ultra", "auto")
    )
    # Without a value the source is inapplicable, and so are its choices.
    unvalued = folded(base()).trial(keyed({"w.source.memstream.ram_style": "block"}))
    assert unvalued.refused and all(
        finding.code == "inapplicable" for why in unvalued.refused.values() for finding in why
    )


def test_the_stored_memory_needs_its_own_choices_and_refuses_bad_weights():
    uncommitted = configured(base(weights=WEIGHTS))
    assert isinstance(module_of(uncommitted, "matmul"), Unresolved)
    # Weights outside their type are refused where MatMul states their range on the
    # weight channel's tensor; the channel, which carries it, is refused with it.
    outside = base(weights=((4,) * 4,) * 4)
    refused = outside.member("matmul").query(MatMulKernel.w_carried)
    assert isinstance(refused, Rejected)
    assert codes(refused) == {"dtype-storage"}
    assert owners(refused) == {"w_carried"}
    assert codes(outside.query("w", NETLIST)) == {"dtype-storage"}
    # A shape error is refused by the memory that packs them.
    wrong = module_of(configured(base(weights=((0,),)), style="auto"), "matmul")
    assert isinstance(wrong, Rejected) and "shape" in wrong.findings[0].message


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
    assert core.y.element == ScalarEncoding(DataType["INT7"], (-30, 40))
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
    assert matmul.compute.y.element == ScalarEncoding(DataType[result], value_range)
    assert matmul.compute.parameters["ACCU_WIDTH"] == DataType[result].bitwidth()
    assert isinstance(module_of(point, "matmul"), Available)


def test_the_cores_do_not_wait_on_the_source():
    point = choose(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    point = choose(
        point,
        {
            "matmul.compute.packed.pe": 4,
            "matmul.compute.packed.simd": 2,
            "matmul.compute.packed.compute_pumping": True,
            "matmul.compute.packed.reducer": "tree",
        },
    )
    assert isinstance(module_of(point, "matmul"), Unresolved)
    # The packed core does not wait for the memory: its weights' range is the one their
    # owner states on the channel.
    compute = point.member("matmul").compute
    assert isinstance(compute.inspect(DotpAxiKernel.module).accepted_result, Available)
    assert compute.parameters["NARROW_WEIGHTS"] == 0  # the weights hold -4
    assert compute.inspect(DotpAxiKernel.admission).result == Available(True)
    folding = choose(base(weights=WEIGHTS), {"matmul.compute": "packed"})
    folding = choose(folding, {"matmul.compute.packed.simd": 4})
    # Folding refusals come before a PE or a memory style is chosen.
    trial = folding.trial(keyed({"matmul.compute.packed.pe": 3}))
    assert "domain-membership" in {
        finding.code for why in trial.refused.values() for finding in why
    }


def test_the_sources_choices_round_trip_through_another_network():
    point = configured(base(weights=WEIGHTS), style="block")
    saved = point.chosen()
    # Only choices made on purpose: the forced source and adapter selector are not saved.
    assert sorted(f"{owner}.{key}" for owner, key in saved) == [
        "matmul.compute",
        "matmul.compute.packed.compute_pumping",
        "matmul.compute.packed.pe",
        "matmul.compute.packed.reducer",
        "matmul.compute.packed.simd",
        "w.source.memstream.pumped_memory",
        "w.source.memstream.ram_style",
        "w.transport",
        "x.adapter.input_gen.input_gen.ram_style",
        "x.transport",
        "y.transport",
    ]
    replayed = base(weights=WEIGHTS).with_choices(saved)
    assert built(replayed, "matmul") == built(point, "matmul")
    # Replay under different facts: the same choices, a new image.
    other = base(weights=tuple(row[::-1] for row in WEIGHTS)).with_choices(saved)
    assert other.member("w").source.image != point.member("w").source.image
    # Without known weights the channel has no source: its choices are stale, refused.
    with pytest.raises(NetworkError, match="w.source.memstream.ram_style: .*inapplicable"):
        base().with_choices(saved)
    # Facts that invalidate a saved folding refuse replay atomically.
    with pytest.raises(NetworkError, match="matmul.compute.packed.pe"):
        base(weights=WEIGHTS, n=5).with_choices(saved)


def test_several_weight_sets_without_known_weights_leave_the_set_channel_unused():
    # No value, no source: nothing consumes the set index, which refuses itself.
    point = configured(base(weight_sets=2))
    answer = module_of(point, "matmul")
    assert isinstance(answer, Rejected)
    assert "channel-unused" in codes(answer)


def test_a_non_viable_source_is_refused_and_committed_is_refused_by_its_candidate():
    # Several sets need the set channel; a memory without one refuses itself.
    network = Network(
        {
            **{name: stored(FULL_DSP48E2).wiring(name) for name in ("x", "y")},
            "w": wire(
                tensor=Fact("matmul", "w_carried"),
                contents=Fact("matmul", "w_stored"),
                sets=2,
                consumer=Interface("matmul", "w_end"),
                platform=FULL_DSP48E2,
            ),
            "matmul": MatMulKernel(
                **FACTS,
                x_tensor=Tensor((3, 4), INT3),
                w_contents=(WEIGHTS, WEIGHTS),
            ),
        }
    )
    w = folded(network).member("w")
    answer = w.query(Channel.source)
    assert isinstance(answer, Rejected) and codes(answer) == {"decision-no-viable-case"}
    assert "memstream-set-channel" in answer.findings[0].message
    # Committed on purpose, the case is accepted (committing never checks a kernel
    # case's admission), and the candidate then refuses the configuration.
    chosen = commit(w, {"source": "memstream"})
    refusal = inspection.admission(chosen.source)
    assert isinstance(refusal, Rejected) and "memstream-set-channel" in codes(refusal)


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
    trial = point.trial(keyed({"w.source.memstream.ram_style": "ultra"}))
    assert trial.accepted == (not refused)
    if refused:
        (why,) = trial.refused.values()
        assert {finding.code for finding in why} == refused
    # The platform does not narrow pumping: the doubled clock is the shell's, which
    # its root admits.
    assert ("w", "source.memstream.pumped_memory") not in point.forced()


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
                consumer=Interface("matmul", "x_end"),
                platform=FULL_DSP48E2,
            ),
            "w": wire(
                tensor=w_tensor,
                port="in1_V",
                consumer=Interface("matmul", "w_end"),
                platform=FULL_DSP48E2,
            ),
            "y": wire(
                tensor=y_tensor,
                contents=((0,) * 4,) * 3,
                producer=Interface("matmul", "y_end"),
                platform=FULL_DSP48E2,
            ),
            "matmul": MatMulKernel(**FACTS, x_tensor=x_tensor, w_tensor=w_tensor),
        }
    )
    answer = folded(produced).member("y").query(Channel.endpoints)
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
FIFO_DEPTH = "w.transport.fifo.buffer.depth"
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
    assert records["transport.fifo.buffer.depth"].scope == "transport.fifo.buffer"
    depth = handle(w, "transport.fifo.buffer.depth")
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
    assert ("w", "transport.fifo.buffer.depth") in saved
    assert built(base().with_choices(saved), "matmul") == built(deep, "matmul")
    # In the channel's space, removing the FIFO requires clearing its stale depth and
    # memory style atomically.
    fifo = deep.member("w")
    stale = fifo.try_with_choices({handle(fifo, "transport"): "direct"})
    assert not stale.accepted
    direct = fifo.with_choices(
        fifo.field(handle(fifo, "transport")).change("direct"),
        fifo.field(handle(fifo, "transport.fifo.buffer.depth")).clear(),
        fifo.field(handle(fifo, "transport.fifo.buffer.ram_style")).clear(),
    )
    assert direct.netlist == w.netlist
    # In the network, the change drops them, named.
    trial = deep.trial(keyed({"w.transport": "direct"}))
    assert set(trial.dropped) == {
        ("w", "transport.fifo.buffer.depth"),
        ("w", "transport.fifo.buffer.ram_style"),
    }
    assert built(trial.network, "matmul") == built(point, "matmul")


def test_adapter_places_the_weight_fifo_on_request():
    assembled = matmul_assembly(**{**FACTS, "pe": 2, "simd": 2}, weight_fifo_depth=16)
    assert FIFO in labels(assembled.module)
