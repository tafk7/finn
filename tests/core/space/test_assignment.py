# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change is assignment, refused where it would make a rule newly false: the hand cases.

A change that sets a new value is refused when the configuration it would publish holds a
false piece of validity through an origin it was not false through before: a constraint
false on its own values, a derived value or a view's body refusing on its own values, a
choice whose listed domain lists no value. The refusal names the origin (never a guard,
never a rule false only through a refused input) and the stored choices standing in its
way: those the origin reads that the change does not name, never what decides whether
the origin, or what it reads, exists. A change of clears alone is never refused.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from support.space_cores import built, commit

from finn.core.space import (
    ConfigurationError,
    Decision,
    Domain,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    divisors_of,
    domain,
    inspection,
    reject,
    view,
)
from finn.core.space._running.evaluate import evaluation
from finn.core.space._running.questions import pairs
from finn.core.space.edits import ConfigurationResult
from finn.core.space.inspection import Absent, Dormant, Open, Possible, Refused, Set, Waiting
from finn.core.space.occurrence import ask, held


def handle(point: Any, key: str) -> Any:
    """The reference of the choice keyed ``key``."""
    return {item.key: item.reference for item in inspection.decisions(point)}[key]


def keyed(point: Any, choices: dict[str, object]) -> ConfigurationResult[Any]:
    """``point`` tried with ``choices`` set, by key."""
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    report = point.try_with_choices({owned[key]: value for key, value in choices.items()})
    return cast(ConfigurationResult[Any], report)


def broken(report: ConfigurationResult[Any]) -> list[tuple[str, tuple[str, ...]]]:
    """A refused change's broken rules and the choices standing in each one's way."""
    return [(rule.rule, rule.standing) for rule in report.broken]


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved)), result
    return {finding.code for finding in result.findings}


def held_pairs(point: Space) -> set[tuple[str, str]]:
    """Each false piece of the root's validity with each origin it is false through, by
    the piece's key and the origin's owner."""
    ds = held(point).ds
    found = ask(ds.model, lambda session: pairs(evaluation(ds, session)))
    return {(piece, origin.owner) for (piece, _), origins in found.items() for origin in origins}


# -- the canon's two examples -------------------------------------------------------------


class Tiles(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))

    @derived
    def cycles(self) -> int:
        return self.extent // self.lanes

    @constraint
    def within_budget(self) -> bool:
        return self.cycles <= 4


class Pair(Space):
    extent: int = Param()
    left = Tiles(extent=extent)
    right = Tiles(extent=extent)

    @constraint
    def right_not_slower(self) -> bool:
        return self.right.cycles <= self.left.cycles

    @view
    def timing(self) -> tuple[int, int]:
        return self.left.cycles, self.right.cycles


def test_the_canons_pair_a_change_alone_is_refused_and_both_at_once_accepted() -> None:
    base = design_space(Pair(extent=12))
    assert inspection.validity(base).status == "open"  # nothing is set
    point = base.with_choices({Pair.left.lanes: 3, Pair.right.lanes: 3})
    assert inspection.validity(point).status == "valid"
    assert point.timing == (4, 4)
    report = point.try_with_choices({Pair.left.lanes: 6})
    assert not report.accepted and report.instance is point
    # right_not_slower: right (4 cycles) would be slower than left (2); right stands.
    assert broken(report) == [("right_not_slower", ("right.lanes",))]
    with pytest.raises(ConfigurationError, match="right_not_slower") as caught:
        point.with_choices({Pair.left.lanes: 6})
    assert caught.value.report == report
    faster = point.with_choices({Pair.left.lanes: 6, Pair.right.lanes: 6})  # both at once
    assert faster.timing == (2, 2)
    # Left's own budget: 1 and 2 lanes make 12 and 6 cycles, nothing standing.
    assert broken(point.try_with_choices({Pair.left.lanes: 2})) == [("left.within_budget", ())]


class Memory(Space):
    uram: bool = Param()
    style: str = Decision(values=("block", "ultra"))

    @constraint
    def ultra_needs_uram(self) -> bool | Rejected:
        if self.style == "ultra" and not self.uram:
            return reject("uram-absent", "no UltraRAM")
        return True


def test_the_canons_memory_a_case_the_facts_refuse_is_refused_naming_its_rule() -> None:
    bare = design_space(Memory(uram=False))
    report = bare.try_with_choices(style="ultra")
    assert not report.accepted
    assert broken(report) == [("ultra_needs_uram", ())]  # the facts alone break it
    assert codes(report.broken[0].result) == {"uram-absent"}
    assert bare.with_choices(style="block").style == "block"
    assert design_space(Memory(uram=True)).with_choices(style="ultra").style == "ultra"


# -- blame at the origin ------------------------------------------------------------------


class Pumped(Space):
    simd: int = Decision(values=(1, 2, 4))
    pumping: bool = Decision(values=(False, True))

    @derived
    def lanes(self) -> int | Rejected:
        if self.pumping and self.simd < 2:
            return reject("pumping-lanes", "pumping needs SIMD of 2 or more")
        return self.simd * (2 if self.pumping else 1)

    @constraint
    def fits(self) -> bool:
        return self.lanes <= 8

    @constraint
    def some(self) -> bool:
        return self.lanes >= 1

    @view
    def throughput(self) -> int:
        return self.lanes


def test_a_refusal_through_a_refused_input_is_blamed_at_its_origin_once() -> None:
    point = design_space(Pumped()).with_choices(simd=1)
    report = point.try_with_choices(pumping=True)
    assert not report.accepted
    # ``fits`` and ``some`` are false only through ``lanes``: one origin, SIMD standing.
    assert broken(report) == [("lanes", ("simd",))]
    assert codes(report.broken[0].result) == {"pumping-lanes"}
    assert point.with_choices(simd=2, pumping=True).throughput == 4


class Packed(Space):
    simd: int = Decision(values=(1, 2))
    pumping: bool = Decision(values=(False, True))

    @constraint
    def pumping_has_lanes(self) -> bool:
        return not self.pumping or self.simd >= 2


class Plain(Space):
    pass


class Core(Space):
    compute: Packed | Plain = Select(packed=Packed, plain=Plain)


def test_a_rule_in_a_case_never_names_its_selector_standing() -> None:
    point = commit(design_space(Core()), {"compute": "packed", "compute.packed.simd": 1})
    report = keyed(point, {"compute.packed.pumping": True})
    assert broken(report) == [("compute.packed.pumping_has_lanes", ("compute.packed.simd",))]


class Sum(Space):
    a: int = Decision(values=tuple(range(11)))
    b: int = Decision(values=tuple(range(11)))
    c: int = Decision(values=tuple(range(11)))

    @constraint
    def total(self) -> bool:
        return self.a + self.b + self.c <= 10


def test_a_new_value_is_refused_with_every_stored_choice_it_reads_standing() -> None:
    point = design_space(Sum()).with_choices(b=4, c=4)
    assert broken(point.try_with_choices(a=3)) == [("total", ("b", "c"))]
    assert point.with_choices(a=2).a == 2
    # Set together, nothing stands; one moved beside it, only the other.
    assert broken(point.try_with_choices(a=3, b=5)) == [("total", ("c",))]
    assert point.with_choices(a=3, b=3).a == 3


class Gate(Space):
    c: int = Decision(values=(1, 2))


class Gated(Space):
    s: str = Decision(values=("a", "b"))

    @derived
    def is_b(self) -> bool:
        return self.s == "b"

    g = Gate(when=is_b)

    @constraint
    def r(self) -> bool:
        return self.s != "b" or self.g.c <= 1


def test_a_choice_the_rule_reads_as_a_value_stands_though_it_decides_what_else_exists() -> None:
    # ``r`` reads ``s`` as a value, and reads ``g.c``, which exists because ``s`` is "b":
    # ``s`` stands. Only a read made for existence alone does not.
    point = design_space(Gated()).with_choices(s="b")
    assert broken(keyed(point, {"g.c": 2})) == [("r", ("s",))]
    (c,) = inspection.viable(point, [handle(point, "g.c")])
    assert [
        (item.value, [(rule.rule, rule.standing) for rule in item.broken])
        for item in c.values or ()
        if isinstance(item, Refused)
    ] == [(2, [("r", ("s",))])]
    assert point.with_choices({handle(point, "g.c"): 1}).g.c == 1


class Alternative(Space):
    p: int = Decision(values=(1, 2))


class Selected(Space):
    s: Alternative | Plain = Select(a=Alternative, b=Plain)

    @constraint
    def q(self) -> bool:
        return not isinstance(self.s, Alternative) or self.s.p <= 1


def test_a_selector_read_only_to_find_its_candidate_does_not_stand() -> None:
    # ``q`` reads ``s`` only to find the candidate whose ``p`` it reads: nothing stands.
    point = commit(design_space(Selected()), {"s": "a"})
    assert broken(keyed(point, {"s.a.p": 2})) == [("q", ())]


# -- a rule already false, and a refusing value nothing rules on --------------------------


class Facts(Space):
    w: int = Param()
    x: int = Decision(values=(1, 2))
    y: int = Decision(values=(1, 2))

    @derived
    def d(self) -> int | Rejected:
        return reject("negative", "a negative width") if self.w < 0 else self.w

    @derived
    def unread(self) -> int | Rejected:
        return reject("never-read", "nothing rules on this") if self.y == 2 else self.y

    @constraint
    def r(self) -> bool:
        return self.d + self.x > 0


def test_a_rule_already_false_through_the_facts_refuses_no_change() -> None:
    base = design_space(Facts(w=-1))
    assert inspection.validity(base).status == "refused"
    assert held_pairs(base) == {("r", "d")}
    # Setting x adds no pair: r was false through d before, and still is.
    point = base.with_choices(x=1)
    assert held_pairs(point) == {("r", "d")}
    # A refusing derived value nothing rules on refuses no change.
    assert isinstance(point.with_choices(y=2).query(Facts.unread), Rejected)


# -- switching into a case: the rule's origin inside or outside the case ------------------


class Inner(Space):
    depth: int = Param()

    @constraint
    def deep_enough(self) -> bool:
        return self.depth >= 2


class Shallow(Space):
    pass


class Switch(Space):
    w: int = Param()
    c: int = Decision(values=(1, 2))

    @derived
    def depth(self) -> int | Rejected:
        if self.w < 0:
            return reject("negative", "a negative width")
        return reject("c-two", "c of 2 leaves no depth") if self.c == 2 else self.w

    mode: Inner | Shallow = Select(inner=Inner(depth=depth), shallow=Shallow)


def test_switching_into_a_case_whose_rule_is_false_outside_it_is_refused() -> None:
    # Over the facts alone: nothing stands.
    facts = commit(design_space(Switch(w=-1)), {"c": 1, "mode": "shallow"})
    report = facts.try_with_choices(mode="inner")
    assert broken(report) == [("depth", ())]
    # Over a stored choice: it stands. Before the switch nothing rules on ``depth``.
    stored = commit(design_space(Switch(w=4)), {"c": 2, "mode": "shallow"})
    assert isinstance(stored.query(Switch.depth), Rejected)
    assert broken(stored.try_with_choices(mode="inner")) == [("depth", ("c",))]
    # The rule inside the case, false on its own values, likewise: it reads ``c`` through
    # ``depth``, so ``c`` stands.
    low = commit(design_space(Switch(w=1)), {"c": 1, "mode": "shallow"})
    assert broken(low.try_with_choices(mode="inner")) == [("mode.inner.deep_enough", ("c",))]
    assert commit(design_space(Switch(w=4)), {"c": 1, "mode": "inner"}).c == 1


# -- a guard is never an origin -----------------------------------------------------------


class Guarded(Space):
    size: int = Decision(values=(-1, 8))
    scale: int = Decision(values=(1, 2))

    @derived
    def big(self) -> bool | Rejected:
        if self.size < 0:
            return reject("size-negative", "a size is never negative")
        return self.size * self.scale > 4

    @constraint(when=big)
    def limit(self) -> bool:
        return self.size < 100


def test_a_guard_refused_through_its_input_names_the_inputs_origin() -> None:
    point = design_space(Guarded()).with_choices(scale=1)
    report = point.try_with_choices(size=-1)
    # ``big`` refuses before it reads ``scale``: what it read decides what stands (a
    # method records its reads as it makes them).
    assert broken(report) == [("big", ())]
    # Built directly: the rule under the refused guard does not apply, and the guard's
    # piece stands for it, once.
    refused = built(design_space(Guarded()), {"size": -1, "scale": 1})
    assert isinstance(refused.query(Guarded.limit), Inapplicable)
    validity = inspection.validity(refused)
    assert [(piece.owner, codes(piece.result)) for piece in validity.pieces] == [
        ("limit", {"guard-refused", "size-negative"})
    ]


# -- a view's body refusing on its own values ---------------------------------------------


class Stage(Space):
    lanes: int = Decision(values=(1, 2, 3))

    @view
    def beats(self) -> int | Rejected:
        return (
            reject("odd-lanes", "three lanes beat unevenly") if self.lanes == 3 else 8 // self.lanes
        )


class Line(Space):
    stage = Stage()

    @constraint
    def paced(self) -> bool:
        return self.stage.beats >= 4


def test_a_views_body_refusing_on_its_own_values_is_an_origin() -> None:
    point = design_space(Line())
    assert broken(point.try_with_choices({Line.stage.lanes: 3})) == [("stage.beats", ())]
    assert point.with_choices({Line.stage.lanes: 2}).stage.beats == 4


# -- clears, and what a published configuration keeps ------------------------------------


class Child(Space):
    k: int = Decision(values=(1, 2))

    @constraint
    def own(self) -> bool:
        return self.k == 1

    @view
    def out(self) -> int:
        return 10


class Parent(Space):
    limit: int = Param()
    other: int = Decision(values=(1, 2))
    child = Child()

    @constraint
    def under(self) -> bool:
        return self.child.out <= self.limit


def test_a_change_of_clears_alone_is_never_refused() -> None:
    # Built directly with the child refused, its view refuses with it, and the parent's
    # rule is false only through that view: no origin of its own.
    masked = built(design_space(Parent(limit=5)), {"child.k": 2, "other": 1})
    assert held_pairs(masked) == {("child.own", "child.own")}
    # Clearing ``k`` reopens the child: the view answers, and ``under`` is false at its own
    # origin. A clear only removes information: accepted, and the configuration reads
    # refused, naming the rule it revealed.
    cleared = masked.with_choices(masked.child.field(Child.k).clear())
    assert held_pairs(cleared) == {("under", "under")}
    # A commit repeating the value stored sets nothing: still clears alone.
    again = masked.try_with_choices(masked.child.field(Child.k).clear(), other=1)
    assert again.accepted
    # A new value beside the clear is judged: the revealed rule refuses it.
    moved = masked.try_with_choices(masked.child.field(Child.k).clear(), other=2)
    assert broken(moved) == [("under", ())]


class Budget(Space):
    w: int = Param()
    first = Tiles(extent=12)
    second = Tiles(extent=12)

    @constraint
    def facts_fail(self) -> bool:
        return self.w > 0

    @view
    def total(self) -> int:
        return self.first.cycles + self.second.cycles


def test_from_a_bind_accepted_changes_add_no_pair_and_bind_again_unrefused() -> None:
    base = design_space(Budget(w=0))  # refused by the facts from the start
    start = held_pairs(base)
    assert start == {("facts_fail", "facts_fail")}
    point = base
    for change in (
        {Budget.first.lanes: 3},
        {Budget.second.lanes: 4},
        {Budget.first.lanes: 6, Budget.second.lanes: 6},
    ):
        point = point.with_choices(change)
        assert held_pairs(point) == start  # an accepted change that sets adds no pair
    assert not point.try_with_choices({Budget.first.lanes: 1}).accepted  # 12 cycles
    cleared = point.with_choices(point.first.field(Tiles.lanes).clear())
    assert held_pairs(cleared) <= start  # a clear-only change adds none either
    # What the walk published binds again, as one change from nothing, unrefused.
    for published in (point, cleared):
        stored = {
            item.reference: published.field(item.reference).get()
            for item in inspection.decisions(published)
            if not isinstance(published.query(item.reference), Unresolved)
        }
        assert design_space(Budget(w=0)).try_with_choices(stored).accepted


def test_an_addition_never_moves_a_view_down() -> None:
    point = design_space(Pair(extent=12)).with_choices({Pair.right.lanes: 3})
    # The Space is open and its view waits on left; no addition leaves it refused.
    for lanes in (1, 2, 3, 4, 6, 12):
        report = point.try_with_choices({Pair.left.lanes: lanes})
        if report.accepted:
            assert inspection.validity(report.instance).status != "refused"
            assert report.instance.timing == (12 // lanes, 4)
        else:
            assert report.broken


# -- what a change puts dormant, brings into play and opens ------------------------------


class Store(Space):
    ultra: bool = Param()
    style: str = Decision(values=("block", "ultra"))
    depth: int = Decision(values=(1, 2))

    @constraint
    def has_ultra(self) -> bool | Rejected:
        if self.style == "ultra" and not self.ultra:
            return reject("ultra-absent", "the platform has no UltraRAM")
        return True


class Wire(Space):
    taps: int = Decision(values=(1, 2))


class Sourced(Space):
    ultra: bool = Param()
    source: Store | Wire = Select(memory=Store(ultra=ultra), wire=Wire)


def test_a_selector_switched_away_and_back_keeps_its_cases_choices_dormant() -> None:
    point = commit(
        design_space(Sourced(ultra=False)),
        {"source": "memory", "source.memory.style": "block", "source.memory.depth": 2},
    )
    away = point.try_with_choices({Sourced.source: "wire"})
    assert away.accepted
    assert away.dormant == ("source.memory.style", "source.memory.depth")
    assert away.in_play == () and away.opened == ("source.wire.taps",)
    wire = away.instance
    assert isinstance(wire.query(handle(wire, "source.memory.style")), Inapplicable)
    back = wire.try_with_choices({Sourced.source: "memory"})
    assert back.accepted and back.dormant == ()
    assert back.in_play == ("source.memory.style", "source.memory.depth")
    assert back.instance.source.style == "block"


def test_a_switch_back_is_refused_by_a_restored_choice_breaking_its_rule() -> None:
    wire = commit(design_space(Sourced(ultra=False)), {"source": "wire"})
    # Set while its case is not selected: dormant, and no rule of an absent case counts.
    dormant = keyed(wire, {"source.memory.style": "ultra"})
    assert dormant.accepted and dormant.dormant == () and dormant.in_play == ()
    report = dormant.instance.try_with_choices({Sourced.source: "memory"})
    assert broken(report) == [("source.memory.has_ultra", ("source.memory.style",))]
    # Set again in the same change, the switch is accepted.
    fixed = keyed(dormant.instance, {"source": "memory", "source.memory.style": "block"})
    assert fixed.accepted and fixed.in_play == ("source.memory.style",)
    assert fixed.opened == ("source.memory.depth",) and fixed.dormant == ()


def test_a_choice_under_an_open_selector_waits_and_clearing_the_selector_reports_nothing() -> None:
    base = design_space(Sourced(ultra=True))
    style = handle(base, "source.memory.style")
    waiting = base.try_with_choices({style: "ultra"})
    assert waiting.accepted and (waiting.dormant, waiting.in_play) == ((), ())
    assert isinstance(waiting.instance.query(style), Unresolved)
    chosen = waiting.instance.try_with_choices({Sourced.source: "memory"})
    assert chosen.in_play == ("source.memory.style",)
    assert chosen.opened == ("source.memory.depth",)
    # Clearing the selector leaves its cases' stored choices waiting: neither list.
    cleared = chosen.instance.try_with_choices(chosen.instance.field(Sourced.source).clear())
    assert cleared.accepted
    assert (cleared.dormant, cleared.in_play, cleared.opened) == ((), (), ())
    assert isinstance(cleared.instance.query(style), Unresolved)


class Optional(Space):
    enabled: bool = Decision(values=(False, True))
    lanes: int = Decision(values=(1, 2), when=enabled)


def test_a_when_guard_puts_its_choices_dormant_and_back_into_play() -> None:
    point = design_space(Optional()).with_choices(enabled=True, lanes=2)
    off = point.try_with_choices(enabled=False)
    assert off.dormant == ("lanes",) and off.in_play == ()
    on = off.instance.try_with_choices(enabled=True)
    assert on.in_play == ("lanes",) and on.instance.lanes == 2
    unset = design_space(Optional()).with_choices(enabled=False)
    assert unset.try_with_choices(enabled=True).opened == ("lanes",)


class Leaf(Space):
    lanes: int = Decision(values=(1, 2))

    @constraint
    def two(self) -> bool:
        return self.lanes == 2


class Bare(Space):
    pass


class Middle(Space):
    inner: Leaf | Bare = Select(leaf=Leaf, bare=Bare)


class Outer(Space):
    outer: Middle | Bare = Select(middle=Middle, bare=Bare)


def test_presence_is_the_and_over_the_whole_path_above_a_choice() -> None:
    # The outer selector is open; the inner one stores another case: the leaf is absent.
    point = commit(design_space(Outer()), {"outer.middle.inner": "bare"})
    lanes = handle(point, "outer.middle.inner.leaf.lanes")
    assert isinstance(point.query(lanes), Inapplicable)  # not waiting on ``outer``
    # Its rule does not apply: nothing waits on it.
    assert "outer.middle.inner.leaf.two" not in [
        piece.owner for piece in inspection.validity(point).pieces
    ]
    stored = commit(design_space(Outer()), {"outer.middle.inner.leaf.lanes": 1})
    away = keyed(stored, {"outer.middle.inner": "bare"})
    assert away.dormant == ("outer.middle.inner.leaf.lanes",)
    assert isinstance(away.instance.query(lanes), Inapplicable)


class Behind(Space):
    x: int = Decision(values=(1,))


class Threshold(Space):
    """``g`` applies where ``k`` is at least 3: a condition computed in this case, so it
    carries the case's guard."""

    k: int = Param()

    @derived
    def big(self) -> bool:
        return self.k >= 3

    g: Behind = Behind(when=big)


class Over(Space):
    k: int = Param()
    s: Threshold | Bare = Select(a=Threshold(k=k), b=Bare())


class OverChoice(Space):
    k: int = Decision(values=(0, 5))
    s: Threshold | Bare = Select(a=Threshold(k=k), b=Bare())


def test_a_false_when_beneath_an_open_selector_makes_its_scope_absent() -> None:
    """Presence is the AND over the path: the ``when`` reads false whatever the selector
    above it will hold, so its choice is absent, not open waiting on the selector."""
    states = dict(inspection.states(design_space(Over(k=0))))
    assert states == {"s": Open(), "s.a.g.x": Absent()}
    assert dict(inspection.states(design_space(Over(k=5))))["s.a.g.x"] == Open(("s",))


def test_turning_a_when_off_beneath_an_open_selector_puts_its_choice_dormant() -> None:
    point = commit(design_space(OverChoice()), {"k": 5, "s.a.g.x": 1})
    assert dict(inspection.states(point))["s.a.g.x"] == Waiting(1, ("s",))
    k = handle(point, "k")
    (possible,) = inspection.viable(point, [k])
    assert possible.values == (
        Possible(0, dormant=("s.a.g.x",)),
        Possible(5),
    )
    off = keyed(point, {"k": 0})
    assert off.accepted and off.dormant == ("s.a.g.x",)
    assert dict(inspection.states(off.instance))["s.a.g.x"] == Dormant(1)
    # Back on, it waits on the selector again: not present, so not in play.
    on = keyed(off.instance, {"k": 5})
    assert (on.dormant, on.in_play) == ((), ())
    assert dict(inspection.states(on.instance))["s.a.g.x"] == Waiting(1, ("s",))


class OwnGate(Space):
    c: int = Decision(values=(0, 5))

    @derived
    def big(self) -> bool:
        return self.c >= 3

    g: Behind = Behind(when=big)


class Switched(Space):
    c: bool = Decision(values=(False, True))
    g: Behind = Behind(when=c)


class OverOwn(Space):
    s: OwnGate | Switched | Bare = Select(a=OwnGate(), w=Switched(), b=Bare())


@pytest.mark.parametrize(("case", "stored"), [("a", 0), ("w", False)])
def test_a_when_reading_a_choice_of_its_own_open_case_waits_on_the_selector(
    case: str, stored: object
) -> None:
    """The condition reads a choice whose own presence waits on the selector, directly
    or through a derived value: the scope stays open, whatever the stored value."""
    point = commit(design_space(OverOwn()), {f"s.{case}.c": stored})
    states = dict(inspection.states(point))
    assert states[f"s.{case}.c"] == Waiting(stored, ("s",))
    assert states[f"s.{case}.g.x"] == Open(("s",))


# -- facts changes: a new design space, never refused --------------------------------------


class Sized(Space):
    extent: int = Param()
    uram: bool = Param()
    lanes: int = Decision(domain=divisors_of(extent))
    style: str = Decision(values=("block", "ultra"))

    @constraint
    def ultra_needs_uram(self) -> bool | Rejected:
        if self.style == "ultra" and not self.uram:
            return reject("uram-absent", "no UltraRAM")
        return True


def test_a_facts_change_drops_a_stored_value_left_outside_its_domain() -> None:
    """4 divides 12, not 10: under the new facts it is not a value of the choice at all,
    so it is dropped, with its check's findings, and the choice is open."""
    point = design_space(Sized(extent=12, uram=True)).with_choices(lanes=4)
    report = point.try_with_facts({Sized.extent: 10})
    assert report.accepted and report.broken == ()
    assert [(o.owner, o.status, o.requested) for o in report.outcomes] == [
        ("extent", "changed", True),
        ("lanes", "dropped", False),
    ]
    assert codes(report.outcomes[1].result) == {"domain-membership"}
    assert dict(inspection.states(report.instance))["lanes"] == Open()
    assert point.lanes == 4  # the receiver still holds it
    # Set or cleared in the same change, it is not dropped: the batch's own.
    assert point.with_facts({Sized.extent: 10}, {Sized.lanes: 5}).lanes == 5
    cleared = point.try_with_facts({Sized.extent: 10}, point.field(Sized.lanes).clear())
    assert [(o.owner, o.status) for o in cleared.outcomes] == [
        ("extent", "changed"),
        ("lanes", "removed"),
    ]
    assert isinstance(cleared.instance.query(Sized.lanes), Unresolved)


def test_a_facts_change_breaking_a_rule_is_accepted_and_names_the_choices_in_its_way() -> None:
    """``style`` is inside its domain under the new facts; the rule now excludes the point,
    so the design space reads refused, and the report names the rule with ``style``."""
    point = design_space(Sized(extent=12, uram=True)).with_choices(style="ultra")
    report = point.try_with_facts({Sized.uram: False})
    assert report.accepted and report.instance.style == "ultra"
    assert broken(report) == [("ultra_needs_uram", ("style",))]
    assert codes(report.broken[0].result) == {"uram-absent"}
    validity = inspection.validity(report.instance)
    assert validity.status == "refused"
    assert [piece.owner for piece in validity.pieces] == ["ultra_needs_uram"]
    # A clear of the choice standing is accepted, and the rule waits on it again.
    cleared = report.instance.try_with_choices(report.instance.field(Sized.style).clear())
    assert cleared.accepted
    assert inspection.validity(cleared.instance).status == "open"
    # Cleared in the same change, nothing is newly false.
    both = point.try_with_facts({Sized.uram: False}, point.field(Sized.style).clear())
    assert both.accepted and both.broken == ()
    # A fact set to the value it has sets nothing: the receiver itself.
    assert point.with_facts({Sized.uram: True}) is point


def test_a_facts_change_emptying_a_present_choices_domain_is_accepted_and_reads_refused() -> None:
    """Section 16.24 settles reading 9 the other way: the domain's piece is its own origin,
    with nothing standing in its way, reported; the stored value is dropped."""
    base = design_space(Sized(extent=12, uram=True))
    report = base.try_with_facts({Sized.extent: 0})
    assert report.accepted and broken(report) == [("lanes", ())]
    assert codes(report.broken[0].result) == {"domain-empty"}
    assert [piece.owner for piece in inspection.validity(report.instance).pieces] == ["lanes"]
    stored = base.with_choices(lanes=4).try_with_facts({Sized.extent: 0})
    assert stored.accepted and broken(stored) == [("lanes", ())]
    assert [(o.owner, o.status) for o in stored.outcomes if not o.requested] == [
        ("lanes", "dropped")
    ]
    # Created with such facts, a design space reads refused alike.
    empty = design_space(Sized(extent=0, uram=True))
    assert [piece.owner for piece in inspection.validity(empty).pieces] == ["lanes"]
    # A facts change keeping that piece through the same origin reports nothing.
    kept = empty.try_with_facts({Sized.uram: False})
    assert kept.accepted and kept.instance.uram is False and kept.broken == ()


class Spanned(Space):
    wide: bool = Param()
    width: int = Param(default=8)

    @derived(when=wide)
    def span(self) -> int:
        return self.width

    lanes: int = Decision(domain=divisors_of(span))


def test_a_facts_change_leaving_a_domains_input_absent_drops_the_value_and_reads_refused() -> None:
    """Section 16.24 with 16.17-18: the input does not exist for the new facts, so the
    stored value is dropped (``domain-input-absent``) and the present choice's list cannot
    be made, at its own origin, nothing standing."""
    wide = design_space(Spanned(wide=True))
    report = wide.try_with_facts({Spanned.wide: False})
    assert report.accepted and broken(report) == [("lanes", ())]
    stored = wide.with_choices(lanes=4)
    report = stored.try_with_facts({Spanned.wide: False})
    assert report.accepted and broken(report) == [("lanes", ())]
    (dropped,) = [o for o in report.outcomes if not o.requested]
    assert dropped.owner == "lanes" and dropped.status == "dropped"
    assert codes(dropped.result) == {"domain-input-absent"}
    validity = inspection.validity(report.instance)
    assert validity.status == "refused"
    assert {codes(piece.result).pop() for piece in validity.pieces} == {"domain-input-absent"}


# -- which values are possible, and every choice's state ---------------------------------


def test_the_canons_examples_possible_values() -> None:
    point = design_space(Pair(extent=12)).with_choices({Pair.left.lanes: 3, Pair.right.lanes: 3})
    (left,) = inspection.viable(point, [Pair.left.lanes])
    assert left.key == "left.lanes" and left.cases == (3,)
    # 1, 2: left's own budget; 4, 6, 12: right (3) stands in the way.
    assert [
        (item.value, [(rule.rule, rule.standing) for rule in item.broken])
        for item in left.values or ()
        if isinstance(item, Refused)
    ] == [
        (1, [("left.within_budget", ())]),
        (2, [("left.within_budget", ())]),
        (4, [("right_not_slower", ("right.lanes",))]),
        (6, [("right_not_slower", ("right.lanes",))]),
        (12, [("right_not_slower", ("right.lanes",))]),
    ]
    (style,) = inspection.viable(design_space(Memory(uram=False)), [Memory.style])
    assert style.cases == ("block",)  # "ultra" is refused by the facts alone


def test_a_possible_value_names_what_it_would_move_and_publishes_nothing() -> None:
    point = commit(
        design_space(Sourced(ultra=False)),
        {"source": "memory", "source.memory.style": "block", "source.memory.depth": 2},
    )
    (source,) = inspection.viable(point, [Sourced.source])
    assert source.values == (
        Possible("memory"),  # the value stored: setting it changes nothing
        Possible("wire", ("source.memory.style", "source.memory.depth"), (), ("source.wire.taps",)),
    )
    assert point.source.depth == 2  # nothing published


def test_values_under_an_open_selector() -> None:
    # Under the open selector, the style is listed as if its case were selected; each
    # value waits with it, nothing refuses.
    base = design_space(Sourced(ultra=False))
    (style,) = inspection.viable(base, [handle(base, "source.memory.style")])
    assert style.cases == ("block", "ultra")


def domain_of_positive() -> Domain[int]:
    """Any positive value: known by membership only."""
    return domain(accepts=lambda *, candidate: candidate > 0)


class States(Space):
    extent: int = Param()
    source: Store | Wire = Select(memory=Store(ultra=True), wire=Wire)
    enabled: bool = Decision(values=(False, True))
    lanes: int = Decision(domain=divisors_of(extent))
    depth: int = Decision(domain=domain_of_positive(), when=enabled)


def test_every_choice_has_one_of_five_states_in_rank_order() -> None:
    base = design_space(States(extent=12))
    point = keyed(
        base, {"source.memory.style": "block", "enabled": False, "depth": 8, "lanes": 4}
    ).instance
    found = dict(inspection.states(point))
    assert found == {
        "source": Open(()),
        "source.memory.style": Waiting("block", ("source",)),
        "source.memory.depth": Open(("source",)),
        "source.wire.taps": Open(("source",)),
        "enabled": Set(False),
        "lanes": Set(4),
        "depth": Dormant(8),
    }
    keys = [key for key, _ in inspection.states(point)]
    assert keys.index("source") < keys.index("source.memory.style")
    assert keys.index("enabled") < keys.index("depth")
    wire = keyed(point, {"source": "wire"}).instance
    states = dict(inspection.states(wire))
    assert states["source.memory.style"] == Dormant("block")
    assert states["source.memory.depth"] == Absent()
    assert states["source.wire.taps"] == Open(())
    on = dict(inspection.states(wire.with_choices(enabled=True)))
    assert on["depth"] == Set(8)
    cleared = wire.with_choices(wire.field(States.depth).clear(), enabled=True)
    assert dict(inspection.states(cleared))["depth"] == Open(())
    (depth,) = inspection.viable(cleared, [States.depth])
    assert depth.values is None  # known by membership only
    assert [item.key for item in inspection.decisions(cleared) if not item.listed] == ["depth"]
