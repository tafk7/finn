# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition declared by hand: kernels and channels as one design space.

On the Chain (``support.chain``): its members ranked kernels first, then each channel
after the members it reads; a channel's sides waiting on its kernels' folding; a change
keeping the channels' stored choices; what the change names refused, by its value's
domain or by the rule it would break, with the stored choices standing in its way; its
module, one composed netlist, each member under its node (its name, ``\\W`` as ``_``),
refused where its pins cannot be formed; its stored choices bound, as assignment (a value
outside its domain dropped as stale, a rule they break binding and read refused); and a
change that keeps a partition's class.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support import chain
from support.helpers import (
    FULL_DSP48E2,
    built,
    by_hand,
    choose,
    keyed,
    labels,
    matmul_point,
    module_of,
)

from finn.core.space import Available, Decision, Rejected, Space, derived, inspection
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.kernels.artifacts.module import Fragment, Link, LinkEnd
from finn.kernels.base import Timed
from finn.kernels.composer import (
    ChannelDeclaration,
    Facts,
    PartitionDeclaration,
    Port,
    Stated,
    channel,
)
from finn.kernels.partition import Partition, partition, top_markers
from finn.kernels.sides import Side
from finn.platform.capabilities import Platform

FOLDING = {
    f"{layer}.{key}": value
    for layer in ("first", "second")
    for key, value in (
        ("compute", "packed"),
        ("pe", 2),
        ("simd", 2),
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
    # A channel that reads another's port (the set port of the weights' source)
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


def test_a_channel_s_sides_wait_on_its_kernels_folding() -> None:
    network = chain.network()
    assert network.validity("x").status == "open"
    # Its cycles wait on what its consumer's side waits on: the kernel's core, open (its
    # folding's presence waits on it).
    assert network.waits("x", network.query("x", Timed.cycles)) == (("first", "compute"),)
    folded = choose(network, FOLDING)
    assert folded.member("x").consumer.label == "first.compute.packed.x"


def with_transports(network: Partition) -> Partition:
    return choose(network, {f"{name}.transport": "direct" for name in network.channels})


def test_a_change_keeps_the_channels_stored_choices_and_places_what_its_facts_need() -> None:
    folded = with_transports(choose(chain.network(), FOLDING))
    changed = choose(folded, {"first.pe": 1})
    # The stored choices of the channels its kernel's sides reach kept, and the output
    # adapter those sides bring into play placed, no choice: PE 1 against the
    # thresholding's 2 lanes.
    assert changed.chosen(("hidden",)) == folded.chosen(("hidden",))
    assert changed.member("hidden").output_adapting
    assert not folded.member("hidden").output_adapting


def test_a_trial_refuses_what_the_change_names_and_keeps_what_it_puts_out_of_play() -> None:
    folded = with_transports(choose(chain.network(), FOLDING))
    fifo = choose(folded, {"hidden.transport": "fifo", "hidden.transport.fifo.depth": 16})
    # The FIFO's depth, under a case no longer selected, is kept, dormant: nothing drops.
    trial = fifo.trial(keyed({"hidden.transport": "direct"}))
    assert trial.accepted
    states = dict(trial.network.states(("hidden",)))
    assert states[("hidden", "transport.fifo.depth")] == inspection.Dormant(16)
    assert fifo.with_choices(keyed({"hidden.transport": "direct"})).chosen(("hidden",)) == {
        ("hidden", "transport"): "direct",
        ("hidden", "transport.fifo.depth"): 16,
    }
    # A value outside its domain refuses the change, by its key.
    refused = fifo.trial(keyed({"first.pe": 3}))
    assert not refused.accepted and refused.network is fifo
    assert list(refused.refused) == [("first", "pe")]
    assert [f.code for f in refused.refused[("first", "pe")]] == ["domain-membership"]
    # A rule it would break refuses it, by the rule's key, with the stored choices
    # standing in its way: pumping needs lanes to split.
    pumped = choose(folded, {"first.compute.packed.compute_pumping": True})
    broken = pumped.trial(keyed({"first.simd": 1}))
    rule = ("first", "compute.packed.pumping_has_lanes")
    assert list(broken.refused) == [rule]
    assert [f.code for f in broken.refused[rule]] == ["dotp-pumping"]
    assert broken.standing == {rule: (("first", "compute.packed.compute_pumping"),)}
    with pytest.raises(ValueError, match="pumping_has_lanes.*standing: first.compute.packed"):
        pumped.with_choices(keyed({"first.simd": 1}))


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


def test_stored_choices_are_bound_as_assignment() -> None:
    """A partition given its members' stored choices binds them on its design space, as
    assignment (the engine's ``bind``): a value outside its domain, or a key no member
    owns, is dropped, with why, and left open; a channel's whose sides wait on an open
    kernel choice is held, stored; nothing is filled in."""
    stored = chain.chain().chosen()
    replayed = by_hand(chain.members(), stored)
    assert replayed.chosen() == stored and not replayed.dropped
    stale = by_hand(chain.members(), {**stored, ("first", "pe"): 3})
    assert list(stale.dropped) == [("first", "pe")]
    # Its own domain refuses 3: no other piece stands beside it.
    assert {(f.code, f.owner) for f in stale.dropped[("first", "pe")]} == {
        ("domain-membership", "pe"),
    }
    assert ("first", "pe") in stale.undecided()
    waiting = by_hand(chain.members(), {("x", "transport"): "direct"})
    assert waiting.chosen()[("x", "transport")] == "direct" and not waiting.dropped
    nobody = by_hand(chain.members(), {("nobody", "pe"): 1})
    assert [(f.code, f.owner) for f in nobody.dropped[("nobody", "pe")]] == [
        ("not-a-decision", "nobody.pe")
    ]


def test_a_stored_key_no_member_declares_binds_and_is_named_stale_as_not_a_decision() -> None:
    """Bind is assignment: a stored key of no member (``nobody.pe``) does not refuse the
    partition. It is dropped as stale, named by its dotted key with ``not-a-decision``,
    owned by the key; every other stored choice binds as it was."""
    stored = chain.chain().chosen()
    bound = by_hand(chain.members(), {**stored, ("nobody", "pe"): 1})
    assert bound.chosen() == stored
    assert list(bound.dropped) == [("nobody", "pe")]
    assert list(bound.stale) == ["nobody.pe"]
    ((found,),) = bound.stale.values()
    assert (found.code, found.owner, found.message) == (
        "not-a-decision",
        "nobody.pe",
        "no choice of the partition",
    )
    assert not by_hand(chain.members(), stored).stale


class Board(Space):
    """A member that states the platform of the Chain's hidden channel: with UltraRAM or
    without, by its own choice."""

    uram: bool = Decision(domain=(True, False))

    @derived
    def platform(self) -> Platform:
        return replace(FULL_DSP48E2, uram=self.uram, uram_init=self.uram)


def boarded() -> dict[str, Facts | ChannelDeclaration]:
    """The Chain's members, its hidden channel's platform read from a ``Board``."""
    members = chain.members()
    hidden = members["hidden"]
    assert isinstance(hidden, ChannelDeclaration)
    return {
        "board": Facts(Board, {}),
        **members,
        "hidden": ChannelDeclaration(
            hidden.producer,
            hidden.consumer,
            {**hidden.facts, "platform": Stated("board", "platform")},
        ),
    }


ULTRA_FIFO = {
    "board.uram": True,
    "hidden.transport": "fifo",
    "hidden.transport.fifo.depth": 16,
    "hidden.transport.fifo.buffer.ram_style": "ultra",
}


def test_a_stored_set_that_breaks_a_rule_binds_and_reads_refused_by_name() -> None:
    """Stored choices are bound together: a rule they break leaves the member reading
    refused, its validity naming the rule (a FIFO in UltraRAM on a platform without it);
    nothing is dropped."""
    stored = keyed({**ULTRA_FIFO, "board.uram": False})
    bound = by_hand(boarded(), {**chain.chain().chosen(), **stored})
    assert not bound.dropped
    validity = bound.validity("hidden")
    assert validity.status == "refused" and isinstance(validity.result, Rejected)
    rule = "transport.fifo.buffer.uram_present"
    assert [(f.code, f.owner) for f in validity.result.findings] == [("uram-absent", rule)]


def test_a_change_a_stored_channel_choice_stands_in_the_way_of_is_refused_naming_it() -> None:
    """A change that makes a channel's rule false, a stored choice of the channel standing
    in its way, is refused, naming the rule and the choice (the seam's fold rule,
    ``finn.kernels.explore.Seam.set``, clears it and sets it again). Cleared in the same
    change, it is left open; a choice a change left open stays open."""
    network = choose(by_hand(boarded()), {"activate.pe": chain.PE, **FOLDING})
    network = choose(network, ULTRA_FIFO)
    style = ("hidden", "transport.fifo.buffer.ram_style")
    trial = network.trial(keyed({"board.uram": False}))
    rule = ("hidden", "transport.fifo.buffer.uram_present")
    assert not trial.accepted and list(trial.refused) == [rule]
    assert [finding.code for finding in trial.refused[rule]] == ["uram-absent"]
    assert trial.standing == {rule: (style,)}
    cleared = network.trial(keyed({"board.uram": False}), (style,))
    assert cleared.accepted
    assert dict(cleared.network.states(("hidden",)))[style] == inspection.Open()
    assert cleared.network.chosen(("hidden",)) == {
        key: value for key, value in network.chosen(("hidden",)).items() if key != style
    }
    # A transport a change left open stays open when its channel's platform changes.
    reopened = network.trial({}, (("hidden", "transport"),)).network
    assert dict(reopened.states(("hidden",)))[("hidden", "transport")] == inspection.Open()
    changed = reopened.trial(keyed({"board.uram": False}))
    assert changed.accepted
    assert ("hidden", "transport") in changed.network.undecided()


def renamed(members: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    """``members`` with member ``old`` named ``new``, and each channel's reads of it."""

    def side(each: Any) -> Any:
        if isinstance(each, (Port, Stated)) and each.member == old:
            return replace(each, member=new)
        return each

    found: dict[str, Any] = {}
    for name, member in members.items():
        if isinstance(member, ChannelDeclaration):
            facts = {fact: side(value) for fact, value in member.facts.items()}
            member = ChannelDeclaration(side(member.producer), side(member.consumer), facts)
        found[new if name == old else name] = member
    return found


def test_each_member_sits_under_its_node_in_the_module() -> None:
    """Each member's node in the module is its name with ``\\W`` as ``_``: the prefix of
    its instances and of the sides it gives its channels."""
    members = renamed(chain.members(), "first", "mm.first")
    stored = {
        ("mm.first" if owner == "first" else owner, key): value
        for (owner, key), value in chain.chain().chosen().items()
    }
    named = by_hand(members, stored)
    assert named.node("mm.first") == "mm_first"
    assert "mm_first.compute.packed" in labels(built(named, chain.STEM))
    assert named.member("x").consumer.label == "mm_first.compute.packed.x"


def test_a_top_whose_pins_cannot_be_formed_is_refused_as_its_netlist() -> None:
    """Two boundary channels at one port: the netlist is made, but the top's pins name that
    port twice, so its module is refused (``kernel-netlist``), not raised."""
    members = chain.members()
    members["y"] = channel(
        Port("second", "y"),
        Side.boundary("s_axis_0"),
        tensor=chain.Y_TENSOR,
        platform=FULL_DSP48E2,
    )
    clashing = by_hand(members, chain.chain().chosen())
    netlist = clashing.netlist()
    assert isinstance(netlist, Available)
    module = module_of(clashing, chain.STEM)
    assert isinstance(module, Rejected)
    assert [(f.code, f.message) for f in module.findings] == [
        ("kernel-netlist", "an ABI names one pin twice")
    ]


def marked(bus: Bus) -> Bus:
    """``bus`` with a marker pin (AXIS ``tlast``), named after it."""
    last = Member("tlast", f"{bus.name}_last")
    return Bus(
        bus.name,
        bus.protocol,
        (*bus.signals, last),
        bus.endpoint,
        bus.role,
        bus.associated_clock,
        bus.associated_reset,
    )


def test_marker_pins_the_boundary_leaves_unruled_refuse_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No target's boundary carries a marker pin (a channel presents its AXIS port with
    none), so a hand-built partition's is given them: an input's marker no link consumes
    and an output's no link rules are the module's own refusals, ``channel-top-marker``,
    each naming its pins. The netlist is answered as it was (the check is the module's)."""
    read = Partition.netlist

    def netlist(network: Partition) -> object:
        answer = read(network)
        assert isinstance(answer, Available)
        fragment, buses = answer.value
        return Available((fragment, tuple(marked(bus) for bus in buses)))

    network = by_hand(chain.members(), chain.chain().chosen())
    assert isinstance(network.netlist(), Available)
    monkeypatch.setattr(Partition, "netlist", netlist)
    found = module_of(network, chain.STEM)
    assert isinstance(found, Rejected)
    assert [(f.code, f.message) for f in found.findings] == [
        (
            "channel-top-marker",
            "top input markers ['s_axis_0_last'] would be left unconsumed",
        ),
        (
            "channel-top-marker",
            "top output markers ['m_axis_0_last'] lack rules",
        ),
    ]


def test_a_marker_pin_a_link_takes_or_drives_is_not_left_unruled() -> None:
    """An input marker is consumed where a link takes it as its source pin, an output
    marker ruled where a link drives it; a bus with no marker pin leaves none."""
    inside = LinkEnd("k", "d", 8, "v", "r")
    wired = Fragment(
        (),
        (
            Link(
                LinkEnd(None, "in_data", 8, "in_valid", "in_ready"),
                inside,
                8,
                (0,),
                (("in_last", None, "k_last", None),),
            ),
            Link(
                inside,
                LinkEnd(None, "out_data", 8, "out_valid", "out_ready"),
                8,
                (0,),
                ((None, None, "out_last", None),),
            ),
        ),
        (),
    )
    inbound = marked(Bus("in", StandardProtocol.AXIS, (Member("tdata", "in_data", 8),)))
    outbound = marked(
        Bus("out", StandardProtocol.AXIS, (Member("tdata", "out_data", 8),), Endpoint.INITIATOR)
    )
    assert (inbound.signals[-1].physical, outbound.signals[-1].physical) == ("in_last", "out_last")
    assert top_markers(wired, (inbound, outbound)) == ()
    bare = Bus("in", StandardProtocol.AXIS, (Member("tdata", "in_data", 8),))
    assert top_markers(Fragment(), (bare,)) == ()
    missing = top_markers(Fragment(), (inbound, outbound))
    assert [f.message for f in missing] == [
        "top input markers ['in_last'] would be left unconsumed",
        "top output markers ['out_last'] lack rules",
    ]


class Labelled(Partition):
    """A partition that keeps a label beside its design space."""

    __slots__ = ("label",)

    def __init__(self, *args: Any, label: str, **given: Any) -> None:
        super().__init__(*args, **given)
        self.label = label


def test_a_change_keeps_the_partitions_class_and_what_it_holds() -> None:
    members = chain.members()
    declared = PartitionDeclaration(
        {name: each for name, each in members.items() if isinstance(each, Facts)},
        {name: each for name, each in members.items() if isinstance(each, ChannelDeclaration)},
    )
    labelled = partition(declared, kind=partial(Labelled, label="the chain"))
    changed = labelled.trial(keyed(FOLDING)).network
    assert isinstance(changed, Labelled) and changed is not labelled
    assert changed.label == "the chain"
