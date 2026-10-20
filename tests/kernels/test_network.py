# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The network: kernels and channels by name, each its own space, and the wiring.

On the Chain (``support.chain``): its members ranked kernels first, then each channel
after the members it reads; a channel bound once its ends are known, and once per set
of facts; a change reaching one kernel and the channels that read it; what the change
names refused, by its value's domain or by the rule it would break; its module, one
composed netlist, each member under its node, refused where its pins cannot be formed;
its stored choices set as each member binds, as one change (a value outside its domain
dropped, a rule they break refusing the network); the interim channel rule where a channel
is rebuilt from new facts; a kernel given bound; and a change that keeps a network's
class.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from qonnx.core.datatype import DataType
from support import chain
from support.helpers import FULL_DSP48E2, built, choose, keyed, labels, matmul_point, module_of

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    Decision,
    Rejected,
    Space,
    Unresolved,
    derived,
    design_space,
    inspection,
)
from finn.kernels.base import CYCLES
from finn.kernels.members import reasons
from finn.kernels.network import Fact, Interface, Network, NetworkError, Wire, wire
from finn.platform.capabilities import Platform
from finn.platform.shells import bus_counts

FOLDING = {
    f"{layer}.{key}": value
    for layer in ("first", "second")
    for key, value in (
        ("compute", "packed"),
        ("compute.packed.pe", 2),
        ("compute.packed.simd", 2),
        ("compute.packed.compute_pumping", False),
        ("compute.packed.reducer", "tree"),
    )
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
    # Its cycles wait on what its end waits on: the kernel's core, open (its folding's
    # presence waits on it).
    assert network.waits("x", network.query("x", CYCLES)) == (("first", "compute"),)
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
    # Its stored choices kept on the channels it rebuilt from their new facts, and the
    # output adapter those facts bring into play at its first valid value (the interim
    # channel rule): PE 1 against the thresholding's 2 lanes.
    assert changed.chosen(("hidden",)) == {
        **folded.chosen(("hidden",)),
        ("hidden", "output_adapter"): "vpc",
    }


def with_transports(network: Network) -> Network:
    return choose(network, {f"{name}.transport": "direct" for name in network.channels})


def test_a_trial_refuses_what_the_change_names_and_keeps_what_it_puts_out_of_play() -> None:
    folded = with_transports(choose(chain.network(), FOLDING))
    fifo = choose(folded, {"hidden.transport": "fifo", "hidden.transport.fifo.buffer.depth": 16})
    # The FIFO's depth, under a case no longer selected, is kept, dormant: nothing drops.
    trial = fifo.trial(keyed({"hidden.transport": "direct"}))
    assert trial.accepted and not trial.rederived
    states = dict(trial.network.states(("hidden",)))
    assert states[("hidden", "transport.fifo.buffer.depth")] == inspection.Dormant(16)
    assert fifo.with_choices(keyed({"hidden.transport": "direct"})).chosen(("hidden",)) == {
        ("hidden", "transport"): "direct",
        ("hidden", "transport.fifo.buffer.depth"): 16,
    }
    # A value outside its domain refuses the change, by its key.
    refused = fifo.trial(keyed({"first.compute.packed.pe": 3}))
    assert not refused.accepted and refused.network is fifo
    assert list(refused.refused) == [("first", "compute.packed.pe")]
    assert [f.code for f in refused.refused[("first", "compute.packed.pe")]] == [
        "domain-membership"
    ]
    # A rule it would break refuses it, by the rule's key, with the stored choices
    # standing in its way: pumping needs lanes to split.
    pumped = choose(folded, {"first.compute.packed.compute_pumping": True})
    broken = pumped.trial(keyed({"first.compute.packed.simd": 1}))
    rule = ("first", "compute.packed.pumping_has_lanes")
    assert list(broken.refused) == [rule]
    assert [f.code for f in broken.refused[rule]] == ["dotp-pumping"]
    assert broken.standing == {rule: (("first", "compute.packed.compute_pumping"),)}
    with pytest.raises(NetworkError, match="pumping_has_lanes.*standing: first.compute.packed"):
        pumped.with_choices(keyed({"first.compute.packed.simd": 1}))


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


def test_stored_choices_are_set_as_each_member_binds() -> None:
    """A network given its members' stored choices sets them as each member binds, as one
    change: a kernel's on its point, a channel's once its ends are known (until then it
    holds them). A value outside its domain is dropped, with why, and left open; nothing
    is filled in."""
    stored = chain.chain().chosen()
    replayed = Network(chain.members(), stored=stored)
    assert replayed.chosen() == stored and not replayed.dropped
    stale = Network(chain.members(), stored={**stored, ("first", "compute.packed.pe"): 3})
    assert list(stale.dropped) == [("first", "compute.packed.pe")]
    # Its own domain refuses 3: no other piece stands beside it.
    assert {(f.code, f.owner) for f in stale.dropped[("first", "compute.packed.pe")]} == {
        ("domain-membership", "compute.packed.pe"),
    }
    assert ("first", "compute.packed.pe") in stale.undecided()
    waiting = Network(chain.members(), stored={("x", "transport"): "direct"})
    held = waiting.unbound("x")
    assert held is not None and held.stored == {"transport": "direct"}
    assert waiting.chosen()[("x", "transport")] == "direct" and not waiting.dropped
    with pytest.raises(NetworkError, match="nobody.pe: no member of the network stores it"):
        Network(chain.members(), stored={("nobody", "pe"): 1})


class Board(Space):
    """A member that states the platform of the Chain's hidden channel: with UltraRAM or
    without, by its own choice."""

    uram: bool = Decision(values=(True, False))

    @derived
    def platform(self) -> Platform:
        return replace(FULL_DSP48E2, uram=self.uram, uram_init=self.uram)


def boarded() -> dict[str, Space | Wire]:
    """The Chain's members, its hidden channel's platform read from a ``Board``."""
    members = chain.members()
    hidden = cast(Wire, members["hidden"])
    return {
        "board": Board(),
        **members,
        "hidden": Wire({**hidden.facts, "platform": Fact("board", "platform")}),
    }


ULTRA_FIFO = {
    "board.uram": True,
    "hidden.transport": "fifo",
    "hidden.transport.fifo.buffer.depth": 16,
    "hidden.transport.fifo.buffer.ram_style": "ultra",
}


def test_a_stored_set_that_breaks_a_rule_refuses_the_network_by_name() -> None:
    """Stored choices are set together as the member binds: a rule they break refuses the
    network, naming it once, its findings carried about the member (a FIFO in UltraRAM on
    a platform without it)."""
    stored = keyed({**ULTRA_FIFO, "board.uram": False})
    with pytest.raises(
        NetworkError, match="hidden: its stored choices are refused: .*uram"
    ) as caught:
        Network(boarded(), stored={**chain.chain().chosen(), **stored})
    rule = "transport.fifo.buffer.uram_present"
    assert str(caught.value).count(rule) == 1
    assert [(f.code, f.owner, dict(f.details)) for f in caught.value.findings] == [
        ("uram-absent", rule, {"tensor": "hidden"})
    ]


def test_reasons_name_each_key_once_with_the_choices_standing_in_its_way() -> None:
    """A refusal's text: each key once, in order, a finding it owns by code and message,
    another's whole, and the stored choices standing in its way where any do."""
    owned = Finding(FindingKind.REJECTION, "dotp-pumping", "rule", "pumping requires SIMD >= 2")
    other = Finding(FindingKind.REJECTION, "uram-absent", "elsewhere", "no UltraRAM")
    found = {"rule": (owned, other), "a": (other,)}
    assert reasons(found, {"rule": ("simd", "pumping")}) == (
        "a: elsewhere: uram-absent: no UltraRAM; rule: dotp-pumping: pumping requires "
        "SIMD >= 2; elsewhere: uram-absent: no UltraRAM (standing: simd, pumping)"
    )


def test_a_channel_rebuilt_from_new_facts_rederives_only_what_they_refuse() -> None:
    """Interim (the ends pass replaces it): a channel rebuilt from new facts keeps every
    stored choice the new facts accept; one they refuse is derived again, first valid, and
    named with its old value, its new one and why. A change of the channel's own choices
    alone takes nothing, and a choice a change left open stays open."""
    network = choose(Network(boarded()), {**{"activate.pe": chain.PE}, **FOLDING})
    network = choose(network, ULTRA_FIFO)
    trial = network.trial(keyed({"board.uram": False}))
    assert trial.accepted
    assert list(trial.rederived) == [("hidden", "transport.fifo.buffer.ram_style")]
    (finding,) = trial.rederived[("hidden", "transport.fifo.buffer.ram_style")]
    assert finding.code == "choice-rederived"
    assert dict(finding.details) == {"value": "ultra", "rederived": "auto"}
    assert [cause.code for cause in finding.causes] == ["uram-absent"]
    assert trial.network.chosen(("hidden",)) == {
        **network.chosen(("hidden",)),
        ("hidden", "transport.fifo.buffer.ram_style"): "auto",
    }
    # A transport a change left open stays open when its channel is rebuilt.
    reopened = network.trial({}, (("hidden", "transport"),)).network
    assert dict(reopened.states(("hidden",)))[("hidden", "transport")] == inspection.Open()
    rebuilt = reopened.trial(keyed({"board.uram": False}))
    assert rebuilt.accepted and not rebuilt.rederived
    assert ("hidden", "transport") in rebuilt.network.undecided()


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


def test_a_top_whose_pins_cannot_be_formed_is_refused_as_its_netlist() -> None:
    """Two boundary channels at one port: the netlist is made, and the shell counts it,
    but the top's pins name that port twice, so its module is refused (``kernel-netlist``),
    not raised."""
    members = chain.members()
    members["y"] = Wire({**members["y"].facts, "port": "s_axis_0"})
    clashing = Network(members, stored=chain.chain().chosen())
    netlist = clashing.netlist()
    assert isinstance(netlist, Available)
    assert bus_counts(*netlist.value) == (0, 0, False)
    module = module_of(clashing, chain.STEM)
    assert isinstance(module, Rejected)
    assert [(f.code, f.message) for f in module.findings] == [
        ("kernel-netlist", "an ABI names one pin twice")
    ]


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
