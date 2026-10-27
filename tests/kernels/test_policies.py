# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A choice's policy (``finn.kernels.policies``): fixed, first valid (in listed order or in
a ranking view's), or required; stated by the owner of a design space, never by the engine
or the kernel, keyed by class-level Decision references, checked where it is stated
(``stated``), read through one reader (``policy``) and applied once, in the flow order
(``defaulted``). A channel's are its own (``Channel.policies``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import pytest
from support.refusals import refused

from finn.core.reporting.findings import Finding
from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Rejected,
    Select,
    Space,
    bind,
    constraint,
    design_space,
    domain,
    inspection,
    reject,
    view,
)
from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.policies import (
    FIRST_VALID,
    Alone,
    Defaulted,
    FirstValid,
    Fixed,
    Policy,
    Required,
    Unstated,
    defaulted,
    policy,
    ranked,
    stated,
)
from finn.kernels.transpose import TransposeKernel
from finn.kernels.utilization import BLOCK_FIRST_BITS, memory_preference, memory_styles


class Narrow(Space):
    width: int = Param()
    lanes: int = Decision(values=(4, 2, 1))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.lanes > self.width:
            return reject("toy-lanes", f"{self.lanes} lanes are wider than {self.width}")
        return True


class Wide(Space):
    width: int = Param()
    lanes: int = Decision(values=(8, 16))


class Toy(Space):
    width: int = Param()
    core: Narrow | Wide = Select(narrow=Narrow, wide=Wide, width=width)
    depth: int = Decision(domain=domain(accepts=lambda *, candidate: candidate > 0))
    style: str = Decision(values=("small", "large"))

    @constraint
    def style_fits(self) -> bool | Rejected:
        if self.style == "small" and self.width > 4:
            return reject("toy-style", f"small holds 4, not {self.width}")
        return True


def owner(root: type[Space], policies: Mapping[object, Policy]) -> Any:
    """An owner stating ``policies`` for ``root``, checked as a KernelOp's are."""
    checked = stated("Owner", root, policies)

    class Owner:
        policies: ClassVar[Mapping[object, Policy]] = checked

    return Owner


TOY = owner(Toy, {Toy.depth: Required()})
"""Toy's owner: its depth, known by membership only, required; every other choice first
valid."""


def states(point: Any) -> dict[str, object]:
    return dict(inspection.states(point))


def applied(
    point: Any, who: Any, *, kept: set[str] | frozenset[str] = frozenset()
) -> Defaulted[Any]:
    """``point`` with ``who``'s policies applied once, as conversion does it (the kernel alone)."""
    return defaulted(point, Alone(who), kept=kept)


def done(found: Defaulted[Any]) -> Any:
    assert isinstance(found, Defaulted), found
    assert found.fallbacks == () and found.unset == () and found.refused == {}
    return found.point


def test_each_choice_in_play_takes_its_first_valid_value_in_the_flow_order() -> None:
    """The selector's first case first, then the choices of that case alone; each value
    the first its rules do not refuse; the required choice left open, the other case's
    choices absent."""
    point = done(applied(design_space(Toy(width=2)), TOY))
    assert states(point) == {
        "core": inspection.Set("narrow"),
        "core.narrow.lanes": inspection.Set(2),  # 4 is wider than 2
        "core.wide.lanes": inspection.Absent(),
        "depth": inspection.Open(),
        "style": inspection.Set("small"),
    }
    assert states(done(applied(design_space(Toy(width=8)), TOY)))["style"] == (
        inspection.Set("large")
    )


class Memory(Space):
    bits: int = Param()
    ram_style: str = Decision(domain=memory_styles())

    @view
    def ram_preference(self) -> tuple[str, ...]:
        found: int = self.bits
        return memory_preference(found)


def test_first_valid_is_in_listed_order_unless_its_policy_names_a_ranking() -> None:
    """A memory's style lists LUTRAM first (``inspection.viable``). Unlisted, first valid
    takes it whatever the size; with a ranking view named, block RAM from
    ``BLOCK_FIRST_BITS``, LUTRAM below."""
    large = design_space(Memory(bits=BLOCK_FIRST_BITS))
    (listed,) = inspection.viable(large, [Memory.ram_style])
    assert listed.cases[:2] == ("distributed", "block")
    assert states(done(applied(large, Unstated)))["ram_style"] == inspection.Set("distributed")
    ranking = owner(Memory, {Memory.ram_style: FirstValid(order=Memory.ram_preference)})
    assert states(done(applied(large, ranking)))["ram_style"] == inspection.Set("block")
    small = done(applied(design_space(Memory(bits=BLOCK_FIRST_BITS - 1)), ranking))
    assert states(small)["ram_style"] == inspection.Set("distributed")


class Refusing(Space):
    """Independent rule and local ranking refusals, with distinct triggers."""

    bits: int = Param()
    ram_style: str = Decision(domain=memory_styles())

    @constraint
    def holds(self) -> bool | Rejected:
        return reject("toy-memory", "refused") if self.bits < 0 else True

    @view
    def ram_preference(self) -> tuple[str, ...] | Rejected:
        found: int = self.bits
        if found < -1:
            return reject("ranking-unavailable", "ranking cannot characterize this input")
        return memory_preference(found)


def test_a_ranking_that_cannot_be_read_falls_back_to_listed_order_named() -> None:
    """An own-refused ranking uses the named fallback, keeping only its local cause."""
    ranking = owner(Refusing, {Refusing.ram_style: FirstValid(order=Refusing.ram_preference)})
    point = design_space(Refusing(bits=-2))
    (item,) = (each for each in inspection.decisions(point) if each.key == "ram_style")
    (listed,) = inspection.viable(point, [item.reference])
    values, fallback = ranked(point, ranking, item, listed.values or ())
    assert values == listed.values
    assert fallback is not None
    assert (fallback.code, fallback.owner) == ("choice-order-unknown", "ram_style")
    assert fallback.message == (
        "first valid in listed order: its order is not known "
        "(Refusing.ram_preference: it is refused)"
    )
    assert dict(fallback.details) == {
        "policy": "first valid",
        "order": "Refusing.ram_preference",
    }
    assert {cause.code for cause in fallback.causes} == {"ranking-unavailable"}
    readable = design_space(Refusing(bits=BLOCK_FIRST_BITS))
    values, fallback = ranked(readable, ranking, item, listed.values or ())
    assert fallback is None and values[0].value == "block"


def test_a_choice_already_set_is_kept() -> None:
    held = design_space(Toy(width=2)).with_choices({Toy.style: "large"})
    assert states(done(applied(held, TOY)))["style"] == inspection.Set("large")


def test_a_choice_named_kept_is_left_open() -> None:
    """``kept`` names open choices the caller leaves open (the channel's policies keep a choice
    a change cleared): every other one takes its policy."""
    point = done(applied(design_space(Toy(width=2)), TOY, kept={"style"}))
    assert states(point)["style"] == inspection.Open()
    assert states(point)["core"] == inspection.Set("narrow")


def test_a_choice_no_value_can_be_set_for_is_left_open_named_and_the_rest_are_set() -> None:
    """No lane count fits a width of 0: the choice is left open and the pass goes on (the
    other choices are set), and the result carries its one finding, owned by the choice's
    key, its policy and each value's rule named, those rules' findings its causes. (It
    stopped at the first such choice, and the result was that finding alone.)"""
    found = applied(design_space(Toy(width=0)), TOY)
    assert found.refused == {} and found.fallbacks == ()
    (finding,) = found.unset
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "core.narrow.lanes")
    assert dict(finding.details) == {"policy": "first valid"}
    assert "4 breaks core.narrow.fits; 2 breaks core.narrow.fits" in finding.message
    assert {cause.code for cause in finding.causes} == {"toy-lanes"}
    assert states(found.point) == {
        "core": inspection.Set("narrow"),
        "core.narrow.lanes": inspection.Open(),
        "core.wide.lanes": inspection.Absent(),
        "depth": inspection.Open(),
        "style": inspection.Set("small"),
    }


# -- the pass over a root of members --------------------------------------------------------


class Source(Space):
    lanes: int = Decision(values=(4, 2, 1))


class Sink(Space):
    width: int = Param()
    style: str = Decision(values=("a", "b"))
    pad: int = Decision(values=(1, 2))

    @constraint
    def holds(self) -> bool | Rejected:
        if self.style == "a" and self.width > 4:
            return reject("toy-style", f"a holds 4, not {self.width}")
        return True


class Gauge(Space):
    """A member with no choice, refused where its width is past four."""

    width: int = Param()

    @constraint
    def holds(self) -> bool | Rejected:
        return reject("toy-gauge", f"{self.width} is past four") if self.width > 4 else True


class Pair(Space):
    """Two members, the sink declared before the source, and a rule of the root that
    reads a choice of each: style ``a`` takes two lanes at most. A third, a gauge, has no
    choice."""

    width: int = Param()
    sink = Sink(width=width)
    source = Source()
    gauge = Gauge(width=width)

    @constraint
    def fits(self) -> bool | Rejected:
        if self.sink.style == "a" and self.source.lanes > 2:
            return reject("toy-pair", "style a takes at most two lanes")
        return True


class Members:
    """A root's members as ``Placed``: a key's first segment is its member, each member has
    its owner and is refused where its own Space is (``inspection.validity``); it records
    the choices the root held at each refusal it read."""

    def __init__(self, owners: Mapping[str, Any]) -> None:
        self._owners = owners
        self.read: list[tuple[str, dict[str, object]]] = []

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(self._owners)

    def member(self, key: str) -> str:
        return key.split(".", maxsplit=1)[0]

    def owner(self, member: str) -> Any:
        return self._owners[member]

    def refusal(self, point: Any, member: str) -> tuple[Finding, ...]:
        self.read.append((member, states(point)))
        result = inspection.validity(getattr(point, member)).result
        return result.findings if isinstance(result, Rejected) else ()


def placed(*members: str, owners: Mapping[str, Any] | None = None) -> Members:
    """``members``, each its owner's in ``owners`` (``Unstated`` where none)."""
    return Members({name: (owners or {}).get(name, Unstated) for name in members})


def test_the_pass_takes_the_choices_in_the_root_s_flow_order_not_its_members() -> None:
    """The root's flow order interleaves its members: ``sink.style`` is first here, though
    the sink is the second member. A choice whose acceptance reads another member's is set
    while that one is still open (the rule waits, and accepts ``a``), so the source is
    left two lanes. Member by member, the source first, it would have taken four, and
    the sink ``b``."""
    point = design_space(Pair(width=2))
    assert [key for key, _ in inspection.states(point)] == [
        "sink.style",
        "sink.pad",
        "source.lanes",
    ]
    done = defaulted(point, placed("source", "sink"))
    assert done.unset == () and done.refused == {} and done.fallbacks == ()
    assert states(done.point) == {
        "sink.style": inspection.Set("a"),
        "sink.pad": inspection.Set(1),
        "source.lanes": inspection.Set(2),  # 4 breaks ``fits`` with ``a`` set
    }


def test_a_member_s_refusal_is_read_when_the_pass_reaches_its_first_choice() -> None:
    """Not before the pass: the sink's is read first, with every choice open, the source's
    when the pass reaches ``source.lanes``, with the sink's choices set by then."""
    members = placed("source", "sink")
    defaulted(design_space(Pair(width=2)), members)
    assert [(name, held["sink.style"], held["source.lanes"]) for name, held in members.read] == [
        ("sink", inspection.Open(), inspection.Open()),
        ("source", inspection.Set("a"), inspection.Open()),
    ]


def test_a_member_refused_on_its_own_is_skipped_whole_and_the_others_are_set() -> None:
    """A sink wider than style ``a`` holds, with ``a`` stored: it is refused by its own rule
    (``toy-style``), none of its choices set (its ``pad`` stays open), reported beside the
    point; the source, set by what the root holds, takes four lanes."""
    stored = bind(design_space(Pair(width=8)), {Pair.sink.style: "a"}).instance
    done = defaulted(stored, placed("source", "sink"))
    assert list(done.refused) == ["sink"]
    assert {finding.code for finding in done.refused["sink"]} == {"toy-style"}
    assert done.unset == ()
    assert states(done.point) == {
        "sink.style": inspection.Set("a"),
        "sink.pad": inspection.Open(),
        "source.lanes": inspection.Set(2),  # the root's rule: ``a`` takes two lanes at most
    }


def test_a_member_with_no_choice_is_still_read_and_reported_refused() -> None:
    """Conversion refuses a kernel refused on its own facts that has nothing to set: its
    refusal is read, at the end of the pass, where no choice of it was reached."""
    done = defaulted(design_space(Pair(width=8)), placed("gauge"))
    assert list(done.refused) == ["gauge"]
    assert {finding.code for finding in done.refused["gauge"]} == {"toy-gauge"}
    assert defaulted(design_space(Pair(width=2)), placed("gauge")).refused == {}


def test_each_choice_takes_the_policy_of_its_member_s_owner() -> None:
    """One class, ``Source``, placed twice, each member under its own owner: the left's
    fixes its lanes, the right's states nothing (first valid, four lanes)."""

    class Pairing(Space):
        left = Source()
        right = Source()

    fixing = owner(Pairing, {Source.lanes: Fixed(1)})
    done = defaulted(
        design_space(Pairing()),
        placed("left", "right", owners={"left": fixing}),
    )
    assert states(done.point) == {
        "left.lanes": inspection.Set(1),
        "right.lanes": inspection.Set(4),
    }


def test_readable_ranking_is_independent_of_a_refused_model() -> None:
    ranking = owner(Refusing, {Refusing.ram_style: FirstValid(order=Refusing.ram_preference)})
    point = design_space(Refusing(bits=-1))
    assert inspection.validity(point).status == "refused"
    assert point.ram_preference == memory_preference(-1)
    (item,) = (each for each in inspection.decisions(point) if each.key == "ram_style")
    (listed,) = inspection.viable(point, [item.reference])
    values, fallback = ranked(point, ranking, item, listed.values or ())
    assert values == listed.values
    assert fallback is None


class Keeper(Space):
    """A memory member whose ranking view refuses while the memory is refused: a channel's
    view reading its kernel's table, where the kernel is refused."""

    bits: int = Param()
    memory = Refusing(bits=bits)


class Unchecked(Members):
    """Members none of which is refused on its own: the pass reads no refusal."""

    def refusal(self, point: Any, member: str) -> tuple[Finding, ...]:
        return ()


def test_a_ranking_the_pass_cannot_read_is_named_beside_the_point() -> None:
    """The choice is still set, in listed order, and the pass reports the fallback
    (``choice-order-unknown``, owned by the choice, naming the view): as a channel's view
    reads the table of a kernel the pass skipped as refused, and the channel is not."""
    ranking = owner(Keeper, {Refusing.ram_style: FirstValid(order=Refusing.ram_preference)})
    done = defaulted(design_space(Keeper(bits=-2)), Unchecked({"memory": ranking}))
    assert states(done.point)["memory.ram_style"] == inspection.Set("distributed")
    (fallback,) = done.fallbacks
    assert (fallback.code, fallback.owner) == ("choice-order-unknown", "memory.ram_style")
    assert dict(fallback.details)["order"] == "Refusing.ram_preference"


def test_a_member_outside_the_pass_is_left_as_it_is() -> None:
    """The pass covers the members it is given: another's choices are left open."""
    done = defaulted(design_space(Pair(width=2)), placed("source"))
    assert states(done.point)["source.lanes"] == inspection.Set(4)
    assert states(done.point)["sink.style"] == inspection.Open()


class Fifo(Space):
    depth: int = Decision(domain=domain(accepts=lambda *, candidate: candidate > 0))


class Link(Space):
    """A channel's shape: a transport that is direct or a FIFO whose depth a strategy sizes,
    and a word too wide to join directly."""

    width: int = Param()
    transport: Fifo | None = Select(direct=None, fifo=Fifo)

    @constraint
    def joins(self) -> bool | Rejected:
        if self.width > 4 and self.transport is None:
            return reject("toy-direct", "a wide word is not joined directly")
        return True


def test_the_transport_is_direct_where_it_is_valid_else_a_fifo_its_depth_left_open() -> None:
    """First valid in listed order: direct where a rule does not refuse it; where one does,
    the FIFO, its depth (required: a strategy sizes it) open and not reported as unset."""
    link = owner(Link, {Fifo.depth: Required()})
    narrow = defaulted(design_space(Link(width=2)), Alone(link))
    assert states(narrow.point)["transport"] == inspection.Set("direct")
    wide = defaulted(design_space(Link(width=8)), Alone(link))
    assert wide.unset == () and wide.refused == {}
    assert states(wide.point) == {
        "transport": inspection.Set("fifo"),
        "transport.fifo.depth": inspection.Open(),
    }


class Stuck(Space):
    """Two choices none of whose values fits the limit, and a third that is free."""

    limit: int = Param()
    first: int = Decision(values=(1, 2))
    second: int = Decision(values=(1, 2))
    free: int = Decision(values=(1, 2))

    @constraint
    def first_fits(self) -> bool | Rejected:
        return reject("toy-first", "past the limit") if self.first > self.limit else True

    @constraint
    def second_fits(self) -> bool | Rejected:
        return reject("toy-second", "past the limit") if self.second > self.limit else True


def test_each_choice_no_value_can_be_set_for_is_reported_and_left_open() -> None:
    """Both of them, in the flow order (conversion reported the first only); the free
    choice between and after them is set."""
    done = applied(design_space(Stuck(limit=0)), owner(Stuck, {}))
    assert [(finding.code, finding.owner) for finding in done.unset] == [
        ("choice-no-valid-value", "first"),
        ("choice-no-valid-value", "second"),
    ]
    assert states(done.point) == {
        "first": inspection.Open(),
        "second": inspection.Open(),
        "free": inspection.Set(1),
    }


SMALL = owner(Toy, {Toy.depth: Required(), Toy.style: Fixed("small"), Toy.core: Fixed("wide")})
ONE_LANE = owner(Toy, {Toy.depth: Fixed(4), Narrow.lanes: Fixed(1)})


def test_a_fixed_value_is_set_in_place_of_the_first_valid_one() -> None:
    """A selector's case and a value alike; a nested class's Decision wherever the class
    is placed; a choice known by membership only, fixed."""
    point = done(applied(design_space(Toy(width=2)), SMALL))
    assert states(point) == {
        "core": inspection.Set("wide"),
        "core.narrow.lanes": inspection.Absent(),
        "core.wide.lanes": inspection.Set(8),
        "depth": inspection.Open(),
        "style": inspection.Set("small"),
    }
    point = done(applied(design_space(Toy(width=4)), ONE_LANE))
    assert states(point)["core.narrow.lanes"] == inspection.Set(1)
    assert states(point)["depth"] == inspection.Set(4)


def test_a_fixed_value_its_rules_refuse_names_the_value_and_the_rule() -> None:
    found = applied(design_space(Toy(width=8)), SMALL)
    (finding,) = found.unset
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "style")
    assert dict(finding.details) == {"policy": "fixed"}
    assert (
        finding.message == "no value can be set (fixed): 'small' is refused: it breaks style_fits"
    )
    assert {cause.code for cause in finding.causes} == {"toy-style"}


def test_one_reader_gives_each_decision_its_owner_s_policy_or_first_valid() -> None:
    """``policy``: the policy stated for the Decision's class-level reference, wherever it
    is placed; first valid, in listed order, where none is."""
    found = {item.key: policy(ONE_LANE, item) for item in inspection.decisions(Toy)}
    assert found == {
        "core": FIRST_VALID,
        "core.narrow.lanes": Fixed(1),
        "core.wide.lanes": FIRST_VALID,
        "depth": Fixed(4),
        "style": FIRST_VALID,
    }
    assert all(policy(Unstated, item) == FIRST_VALID for item in inspection.decisions(Toy))


class Other(Space):
    lanes: int = Decision(values=(1, 2))

    @view
    def ranking(self) -> tuple[int, ...]:
        return (2, 1)


@pytest.mark.parametrize(
    ("policies", "code", "message"),
    [
        (
            {Toy.depth: Required(), Other.lanes: Fixed(1)},
            "default-policy-undeclared",
            "Other.lanes",
        ),
        ({Toy.depth: Required(), Toy.style: "small"}, "default-policy-invalid", "Fixed"),
        ({Toy.depth: Required(), Toy.style: Fixed("medium")}, "default-policy-value", "domain"),
        ({Toy.depth: Required(), Toy.style: Fixed(3)}, "default-policy-value", "type"),
        ({Toy.depth: Required(), Toy.core: Fixed("huge")}, "default-policy-value", "cases"),
        ({Toy.depth: Fixed(0)}, "default-policy-value", "domain"),
        (
            {Toy.depth: Required(), Narrow.lanes: FirstValid(order=Other.ranking)},
            "default-policy-order",
            "Other.ranking",
        ),
        ({}, "default-policy-unlisted", "Toy.depth is known by membership only"),
        ({Toy.depth: FirstValid()}, "default-policy-unlisted", "Required()"),
    ],
)
def test_policies_are_checked_where_they_are_stated(
    policies: Mapping[object, object], code: str, message: str
) -> None:
    """Each key a Decision of a class the root reaches; each value a policy; a fixed
    value one the Decision can take, as far as is known without facts; an order a view of
    the Decision's class; a choice known by membership only required or fixed."""
    with refused(code, error=DefinitionError, message=message):
        stated("Owner", Toy, policies)


def test_a_channel_states_its_own_policies() -> None:
    """A FIFO's depth required, a memory's style first valid in its ranking (a source's,
    its kernel's; the input adapter's one style, the channel's), and every other channel
    choice first valid, in listed order; checked against ``Channel`` when the module is
    imported."""
    assert dict(Channel.policies) == {
        ChannelFifo.depth: Required(),
        MemStreamKernel.ram_style: FirstValid(order=MemStreamKernel.ram_preference),
        Channel.adapter_ram_style: FirstValid(order=Channel.adapter_ram_preference),
    }
    found = {item.key: policy(Channel, item) for item in inspection.decisions(Channel)}
    assert {key for key, each in found.items() if each == Required()} == {"transport.fifo.depth"}
    ranked_keys = {
        key
        for key, each in found.items()
        if isinstance(each, FirstValid) and each.order is not None
    }
    assert ranked_keys == {"source.ram_style", "adapter_ram_style"}


def test_no_kernel_states_a_policy_but_a_ranking_it_reads_out() -> None:
    """A kernel holds hardware knowledge only: the memories' rankings are views (a domain
    ranks nothing; an input generator's style is given, and its channel ranks it), and a
    kernel has no policies of its own."""
    for kernel in (MemStreamKernel, TransposeKernel):
        assert not hasattr(kernel, "defaults") and not hasattr(kernel, "policies")
    assert not hasattr(Channel, "defaults")
    for ranking in (
        MemStreamKernel.ram_preference,
        TransposeKernel.ram_preference,
        Channel.adapter_ram_preference,
    ):
        assert callable(cast(Any, ranking).function)
    assert not hasattr(InputGeneratorKernel, "ram_preference")
