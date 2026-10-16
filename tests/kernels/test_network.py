# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The network: kernels and channels by name, each its own space, and the wiring.

On the Chain (``support.chain``): its members ranked kernels first, then each channel
after the members it reads; a channel bound once its ends are known, and once per set
of facts; a change reaching one kernel and the channels that read it; what replay drops
named, what the change names refused; its module, one composed netlist, each member under
its node; its stored choices replayed as each member binds; a kernel given bound; and a
change that keeps a network's class.
"""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support import chain
from support.helpers import FULL_DSP48E2, built, choose, keyed, labels, matmul_point

from finn.core.space import Space, Unresolved, design_space
from finn.kernels.base import CYCLES
from finn.kernels.network import Interface, Network, NetworkError, Wire, wire

FOLDING = {
    f"{layer}.compute.packed.{key}": value
    for layer in ("first", "second")
    for key, value in (("pe", 2), ("simd", 2), ("compute_pumping", False), ("reducer", "tree"))
}


def test_members_are_ranked_kernels_first_then_each_channel_after_what_it_reads() -> None:
    network = chain.network()
    assert network.members == chain.MEMBERS
    assert network.kernels == ("first", "activate", "second")
    # The module places them as declared: channels, then kernels.
    assert network.order == ("x", "w1", "hidden", "levels", "w2", "y", *network.kernels)
    # A channel that reads another's interface (the set port of the weights' source)
    # follows it, whatever the order declared.
    weights = ((1, 0), (0, 1))
    sets = matmul_point(
        m=2,
        k=2,
        n=2,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        platform=FULL_DSP48E2,
        weights=(weights, weights),
        weight_sets=2,
    )
    assert sets.order.index("set") < sets.order.index("w")
    assert sets.members.index("w") < sets.members.index("set")


def test_a_channel_waits_with_its_choices_until_its_ends_are_known() -> None:
    network = chain.network()
    held = network.unbound("x")
    assert held is not None and held.member == "first" and isinstance(held.result, Unresolved)
    # Its cycles wait on what its end waits on: the kernel's folding.
    assert network.waits("x", network.query("x", CYCLES)) == (("first", "compute.packed.pe"),)
    with pytest.raises(NetworkError, match="x is not bound"):
        network.member("x")
    folded = choose(network, FOLDING)
    assert folded.unbound("x") is None and folded.member("x").consumer.label == (
        "first.compute.packed.x"
    )


def test_a_channel_is_bound_once_per_set_of_facts() -> None:
    one, two = choose(chain.network(), FOLDING), choose(chain.network(), FOLDING)
    assert one.facts("hidden") == two.facts("hidden")
    assert one.member("hidden") is two.member("hidden")


def test_a_change_reaches_its_kernel_and_the_channels_that_read_it() -> None:
    folded = with_transports(choose(chain.network(), FOLDING))
    changed = choose(folded, {"first.compute.packed.pe": 1})
    unchanged = ("activate", "second", "levels", "w2", "y")
    assert all(changed.member(name) is folded.member(name) for name in unchanged)
    assert all(
        changed.member(name) is not folded.member(name) for name in ("first", "x", "w1", "hidden")
    )
    # Its stored choices replayed on the channels it rebound.
    assert changed.chosen(("hidden",)) == folded.chosen(("hidden",))


def with_transports(network: Network) -> Network:
    return choose(network, {f"{name}.transport": "direct" for name in network.channels})


def test_a_trial_names_what_replay_drops_and_refuses_what_the_change_names() -> None:
    folded = with_transports(choose(chain.network(), FOLDING))
    fifo = choose(folded, {"hidden.transport": "fifo", "hidden.transport.fifo.buffer.depth": 16})
    trial = fifo.trial(keyed({"hidden.transport": "direct"}))
    assert trial.accepted and set(trial.dropped) == {("hidden", "transport.fifo.buffer.depth")}
    assert ("hidden", "transport.fifo.buffer.depth") not in trial.network.chosen()
    refused = fifo.trial(keyed({"first.compute.packed.pe": 3}))
    assert not refused.accepted and refused.network is fifo
    assert list(refused.refused) == [("first", "compute.packed.pe")]
    with pytest.raises(NetworkError, match="hidden.transport.fifo.buffer.depth"):
        fifo.with_choices(keyed({"hidden.transport": "direct"}))


def test_its_module_is_one_netlist_in_the_order_declared() -> None:
    module = built(chain.chain(), chain.STEM)
    assert module.stem == "finn_chain"
    assert labels(module) == [
        "x.adapter.input_gen.input_gen",
        "w1.source.memstream",
        "levels.adapter.input_gen.input_gen",
        "w2.source.memstream",
        "first.compute.packed",
        "activate",
        "second.compute.packed",
    ]


def test_a_network_reads_only_its_own_members() -> None:
    with pytest.raises(NetworkError, match="no members"):
        Network({"x": wire(tensor=chain.X_TENSOR, consumer=Interface("k", "x_end"))})


def test_stored_choices_are_replayed_as_each_member_binds() -> None:
    """A network given its members' stored choices replays each as the member binds: a
    kernel's on its point, a channel's once its ends are known (until then it holds
    them); what replay refuses is dropped, with why."""
    stored = chain.chain().chosen()
    replayed = Network(chain.members(), stored=stored)
    assert replayed.chosen() == stored and not replayed.dropped
    stale = Network(chain.members(), stored={**stored, ("first", "compute.packed.pe"): 3})
    assert list(stale.dropped) == [("first", "compute.packed.pe")]
    assert {f.code for f in stale.dropped[("first", "compute.packed.pe")]} == {"domain-membership"}
    waiting = Network(chain.members(), stored={("x", "transport"): "direct"})
    held = waiting.unbound("x")
    assert held is not None and held.stored == {"transport": "direct"}
    assert waiting.chosen()[("x", "transport")] == "direct" and not waiting.dropped
    with pytest.raises(NetworkError, match="nobody.pe: no member of the network stores it"):
        Network(chain.members(), stored={("nobody", "pe"): 1})


def test_a_kernel_given_bound_is_used_as_it_is() -> None:
    """A kernel's point, bound by its owner's cache, is a member as it is."""
    declared = chain.members()
    point = design_space(declared["first"])  # type: ignore[type-var]
    network = Network({**declared, "first": point})
    assert network.member("first") is point


def test_each_member_sits_under_its_node_in_the_module() -> None:
    """``nodes`` names each member's node in the module, the prefix of its instances and
    of the ends it presents to its channels; two members of one node are refused."""
    stored = chain.chain().chosen()
    renamed = Network(chain.members(), stored=stored, nodes={"first": "mm_first"})
    module = built(renamed, chain.STEM)
    assert "mm_first.compute.packed" in labels(module)
    assert renamed.member("x").consumer.label == "mm_first.compute.packed.x"
    with pytest.raises(NetworkError, match="members share a node in the module: x: x, first"):
        Network(chain.members(), nodes={"first": "x"})


class Labelled(Network):
    """A network that keeps a label beside its members, and what it read of them."""

    __slots__ = ("label", "reads")

    def __init__(self, members: dict[str, Space | Wire], label: str) -> None:
        super().__init__(members)
        self.label, self.reads = label, 0

    def _changed(self) -> None:
        self.reads = 0


def test_a_change_keeps_the_networks_class_and_what_it_holds() -> None:
    labelled = Labelled(chain.members(), "the chain")
    labelled.reads = 3
    changed = labelled.trial(keyed(FOLDING)).network
    assert isinstance(changed, Labelled) and changed is not labelled
    assert (changed.label, changed.reads) == ("the chain", 0)
    assert labelled.reads == 3
