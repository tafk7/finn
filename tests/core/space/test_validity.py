# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One kind of rule: a constraint owned by a Space, and the Space's validity.

A Space is **valid** when every piece that counts is true, **refused** when one is
false, and **open** otherwise (Kleene's AND: false beats unknown, unknown beats true).
The pieces are its constraints (every one, while the Space is present), its present
children's validity, every present choice whose domain lists no value, and a guard
refused through a refused input. An open choice is no piece: the constraints that read
it wait on it. A stored value's type check is no piece either: it was made at
assignment. A case not selected contributes nothing. ``inspection.validity`` answers it,
with the pieces that are not true and their owners.

A view is a product: it refuses when its output or its Space is refused, answers while
its Space is open, and an inactive view answers ``Inapplicable`` without reading its
Space's validity. A constraint reads values; one that reads a view of its own Space, or
of an ancestor, closes a dependency cycle. An export may name a derived value, which a
holder reads as the value whatever the exporting Space's validity. A value is possible
unless a rule it reaches refuses it (``inspection.viable``), and a value set is checked
against the choice's own domain, over the facts. A state describes the commitment, a
read the answer: every stored choice is committed with its stored value, whether it
reads as its value, waits, or is dormant.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from support.refusals import refused
from support.space_cores import built, commit

from core.space._possible_support import broken, possible, refusals
from finn.core.space import (
    Available,
    Decision,
    DecisionState,
    EvaluationError,
    Inapplicable,
    Members,
    Param,
    Rejected,
    Space,
    Unresolved,
    View,
    ViewKey,
    constraint,
    derived,
    design_space,
    divisors_of,
    domain,
    inspection,
    occurrence,
    reject,
    selected,
    view,
)
from finn.core.space.inspection import Piece, Validity


def owners(validity: Validity) -> list[str]:
    return [piece.owner for piece in validity.pieces]


def codes(piece: Piece) -> set[str]:
    return {finding.code for finding in piece.result.findings}


# -- the pieces --------------------------------------------------------------------------


class Core(Space):
    """A core whose constraints read its own choices and a formal."""

    width: int = Param()
    lanes: int = Decision(values=(1, 2, 4))

    @constraint
    def divides(self) -> bool | Rejected:
        if self.width % self.lanes:
            return reject("lanes-divide", "the lanes do not divide the width")
        return True

    @derived
    def off(self) -> bool:
        return False

    @constraint(when=off)
    def never_applies(self) -> bool | Rejected:
        return reject("inactive", "an inactive constraint counts as true")


class Engine(Space):
    width: int = Param()
    core = Core(width=width)
    wanted: int = Decision(values=(1, 2))

    @constraint
    def small(self) -> bool | Rejected:
        if self.wanted > 1:
            return reject("too-many", "one is enough")
        return True


def test_a_space_is_valid_when_every_piece_is_true() -> None:
    point = commit(design_space(Engine(width=8)), {"core.lanes": 2, "wanted": 1})
    validity = inspection.validity(point)
    assert validity == Validity("valid", (), Available(True))


def test_a_constraint_that_waits_leaves_the_space_open_naming_what_it_waits_on() -> None:
    point = design_space(Engine(width=8))
    # Nothing is filled in: ``small`` waits on ``wanted`` as ``core.divides`` on its lanes.
    assert isinstance(point.query(Engine.wanted), Unresolved)
    validity = inspection.validity(point)
    assert validity.status == "open" and isinstance(validity.result, Unresolved)
    assert owners(validity) == ["core.divides", "small"]
    assert all(codes(piece) == {"decision-unassigned"} for piece in validity.pieces)
    assert [{f.owner for f in piece.result.findings} for piece in validity.pieces] == [
        {"core.lanes"},
        {"wanted"},
    ]


def test_a_false_piece_refuses_whatever_else_waits() -> None:
    # Setting a value a constraint refuses on its own values is refused, naming it: no
    # stored choice stands in its way.
    report = design_space(Engine(width=8)).try_with_choices({Engine.wanted: 2})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert outcome.result == Available(True) and outcome.status == "checked"
    assert broken(report) == {"small": set()}
    # The same choice stored in a configuration built directly reads refused.
    validity = inspection.validity(built(design_space(Engine(width=8)), {"wanted": 2}))
    assert validity.status == "refused"
    assert owners(validity) == ["small"]  # core.divides still waits: false beats unknown
    assert codes(validity.pieces[0]) == {"too-many"}


def test_a_refused_child_refuses_its_parent_naming_the_childs_piece() -> None:
    report = design_space(Engine(width=6)).try_with_choices(
        {Engine.core.lanes: 4, Engine.wanted: 1}
    )
    assert not report.accepted and broken(report) == {"core.divides": set()}
    point = built(design_space(Engine(width=6)), {"core.lanes": 4, "wanted": 1})
    validity = inspection.validity(point)
    assert validity.status == "refused"
    assert owners(validity) == ["core.divides"]
    assert inspection.validity(point.core) == Validity(
        "refused", validity.pieces, validity.pieces[0].result
    )


def test_every_constraint_of_a_present_space_counts_and_an_inactive_one_is_true() -> None:
    # ``never_applies`` is inactive: it counts as true. ``divides`` belongs to no group,
    # view or requirement: it counts all the same.
    class Lone(Space):
        width: int = Param()

        @constraint
        def odd(self) -> bool | Rejected:
            return True if self.width % 2 else reject("even", "an odd width only")

    assert inspection.validity(design_space(Lone(width=3))).status == "valid"
    assert owners(inspection.validity(design_space(Lone(width=4)))) == ["odd"]


class Starved(Space):
    """``lanes`` has no possible case: its constraint refuses every one."""

    width: int = Param()
    lanes: int = Decision(values=(2, 4))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.lanes > self.width:
            return reject("too-wide", "more lanes than width")
        return True


def test_an_open_decision_whose_values_a_constraint_refuses_leaves_the_space_open() -> None:
    # No piece says the choice has no possible value: the constraint waits on it.
    for width in (1, 2):
        point = design_space(Starved(width=width))
        validity = inspection.validity(point)
        assert validity.status == "open" and owners(validity) == ["fits"]
        assert codes(validity.pieces[0]) == {"decision-unassigned"}
        assert isinstance(point.query(Starved.lanes), Unresolved)


def test_a_value_every_constraint_refuses_is_refused_where_asked() -> None:
    narrow = design_space(Starved(width=1))
    assert possible(narrow, Starved.lanes) == ()
    assert refusals(narrow, Starved.lanes) == {2: {"fits": set()}, 4: {"fits": set()}}
    wide = design_space(Starved(width=2))
    assert possible(wide, Starved.lanes) == (2,)
    assert refusals(wide, Starved.lanes) == {4: {"fits": set()}}


class Refusing(Space):
    width: int = Param()

    @constraint
    def never(self) -> bool | Rejected:
        return reject("never", "this candidate is always refused")


class Plain(Space):
    width: int = Param()


class Selector(Space):
    width: int = Param()
    compute: Refusing | Plain = Decision({"refusing": Refusing, "plain": Plain}, width=width)


def test_a_case_not_selected_contributes_nothing_and_the_selected_one_counts() -> None:
    # While the selector is open, the refusing candidate's constraint waits on it.
    point = design_space(Selector(width=4))
    validity = inspection.validity(point)
    assert validity.status == "open" and owners(validity) == ["compute.refusing.never"]
    # Switching to the refusing candidate makes its rule false on the facts alone.
    report = point.try_with_choices({Selector.compute: "refusing"})
    assert not report.accepted and broken(report) == {"compute.refusing.never": set()}
    selected = built(point, {"compute": "refusing"})
    assert owners(inspection.validity(selected)) == ["compute.refusing.never"]
    plain = commit(point, {"compute": "plain"})
    assert inspection.validity(plain).status == "valid"
    candidate = inspection.candidate(plain, Selector.compute, "refusing")
    assert candidate is not None
    absent = inspection.validity(candidate)
    assert absent == Validity("valid", (), Inapplicable())


class Either(Space):
    width: int = Param()
    compute: Plain | Core = Decision({"plain": Plain, "core": Core}, width=width)


def test_the_pieces_of_a_placed_space_whose_presence_waits_are_unknown_each_under_its_guard() -> (
    None
):
    # A placed Space is not one piece: each constraint inside is unknown, waiting on the
    # selector, whatever its body would answer (``never_applies`` too: its own guard is not
    # read while the Space's presence waits). ``plain`` holds no piece: it adds nothing.
    validity = inspection.validity(design_space(Either(width=8)))
    assert validity.status == "open"
    assert owners(validity) == ["compute.core.divides", "compute.core.never_applies"]
    for piece in validity.pieces:
        assert {(f.code, f.owner) for f in piece.result.findings} == {
            ("decision-unassigned", "compute")
        }


def test_a_placed_space_with_no_pieces_whose_presence_waits_contributes_nothing() -> None:
    class Bare(Space):
        width: int = Param()
        lanes: int = Decision(values=(1, 2))

        @view
        def doubled(self) -> int:
            return 2 * self.lanes

    class Open(Space):
        width: int = Param()
        compute: Plain | Bare = Decision({"plain": Plain, "bare": Bare}, width=width)

    # An open choice and a view are no pieces: the configuration is valid while the
    # selector is open. A choice stored inside waits on the selector, and is no piece
    # either: its value was checked when it was set.
    point = design_space(Open(width=8))
    assert inspection.validity(point) == Validity("valid", (), Available(True))
    kept = commit(point, {"compute.bare.lanes": 2})
    assert isinstance(kept.query(Open.compute["bare"].lanes), Unresolved)  # type: ignore[index]
    assert inspection.validity(kept) == Validity("valid", (), Available(True))


# -- views are products ----------------------------------------------------------------


class Product(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2, 3))

    @constraint
    def divides(self) -> bool | Rejected:
        if self.width % self.lanes:
            return reject("lanes-divide", "the lanes do not divide the width")
        return True

    @view
    def cycles(self) -> int:
        return self.width // 1


def test_a_view_answers_while_its_space_is_open_and_refuses_with_it() -> None:
    open_ = design_space(Product(width=4))
    assert inspection.validity(open_).status == "open"
    assert open_.cycles == 4
    assessment = open_.inspect(Product.cycles)
    assert isinstance(assessment.validity, Unresolved) and assessment.readiness.ready is None
    report = open_.try_with_choices(lanes=3)  # ``divides`` false on its own values
    assert not report.accepted and broken(report) == {"divides": set()}
    refused = built(open_, {"lanes": 3})
    answer = refused.query(Product.cycles)
    assert isinstance(answer, Rejected)
    assert [(f.owner, f.code) for f in answer.findings] == [("divides", "lanes-divide")]
    assert refused.inspect(Product.cycles).output_result == Available(4)


ACCESS = ViewKey("access", int)


class Port(Space):
    """A port: its ``access`` is a product only while it binds, and its constraint reads
    the tensor its kernel gives it."""

    binds: bool = Param()
    tensor: int = Param()

    @constraint
    def admitted(self) -> bool | Rejected:
        return True if self.tensor > 0 else reject("no-tensor", "the port carries nothing")

    @view(when=binds)
    def access(self) -> int:
        return 4

    exports = {ACCESS: access}


class Leaf(Space):
    """A leaf whose output port's tensor follows the extents its ports' accesses give: a
    loop through the output port's validity, cut by its guard (an initiator does not bind)."""

    x = Port(binds=True, tensor=1)
    y = Port(binds=False)
    accesses = Members(ACCESS)

    @derived
    def extents(self) -> int:
        return sum(member.value for member in self.accesses)

    y.tensor = extents


def test_an_inactive_view_answers_inapplicable_and_never_reads_its_spaces_validity() -> None:
    point = design_space(Leaf())
    assert inspection.validity(point).status == "valid"
    assert point.extents == 4
    current = occurrence.state(point)
    access = current.model.resolve(point.y._scope, Port.access)
    entry = current.world.entry(access)
    assert isinstance(entry.result, Inapplicable)
    validity = current.linked.scopes[point.y._scope].validity
    read = [node for node, _ in entry.trace]
    assert read == [current.linked.nodes[access].guard]
    assert validity not in read
    assessment = point.y.inspect(Port.access)
    assert assessment.validity == Inapplicable()


def test_the_same_loop_with_the_guard_holding_is_a_dependency_cycle() -> None:
    class Bound(Space):
        x = Port(binds=True, tensor=1)
        y = Port(binds=True)
        accesses = Members(ACCESS)

        @derived
        def extents(self) -> int:
            return sum(member.value for member in self.accesses)

        y.tensor = extents

    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        inspection.validity(design_space(Bound()))
    path = str(caught.value)
    for step in ("y.$validity", "y.admitted", "extents", "y.access"):
        assert step in path, path


# -- a rule that reads a view: a dependency cycle --------------------------------------


def test_a_constraint_reading_a_view_of_its_own_space_is_a_cycle_naming_its_path() -> None:
    class Storage(Space):
        depth: int = Decision(values=(2, 4))

        @view
        def storage(self) -> int:
            return 8 * self.depth

        @constraint
        def capacity_supported(self) -> bool:
            return self.storage <= 32

    # The change that sets ``depth`` reads the root's validity, and meets the cycle there.
    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        design_space(Storage()).with_choices(depth=2)
    assert caught.value.role == "dependency cycle"
    assert caught.value.detail == (
        "$validity (validity) -> capacity_supported (constraint) -> storage (view) -> $validity"
    )


def test_a_childs_constraint_reading_its_parents_view_through_a_formal_is_a_cycle() -> None:
    class Child(Space):
        width: int = Param()

        @constraint
        def narrow(self) -> bool:
            return self.width <= 8

    class Parent(Space):
        lanes: int = Decision(values=(1, 2))

        @view
        def carried(self) -> int:
            return 4 * self.lanes

        child = Child(width=cast(int, carried))

    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        design_space(Parent()).with_choices(lanes=1)
    assert "carried" in str(caught.value) and "child.narrow" in str(caught.value)

    # Reading a derived value instead, which the view presents, is no cycle.
    class Rewritten(Space):
        lanes: int = Decision(values=(1, 2))

        @derived
        def carried_value(self) -> int:
            return 4 * self.lanes

        carried = View(carried_value)
        child = Child(width=carried_value)

    fixed = design_space(Rewritten()).with_choices(lanes=1)
    assert inspection.validity(fixed).status == "valid" and fixed.carried == 4


# -- an export may name a derived value --------------------------------------------------


WIDTH = ViewKey("width", int)
COST = ViewKey("cost", int)


class Stage(Space):
    """Refused past 4 lanes; exports its width as a value and its cost as a product."""

    lanes: int = Decision(values=(2, 8))

    @derived
    def width(self) -> int:
        return 2 * self.lanes

    @constraint
    def fits(self) -> bool | Rejected:
        return True if self.lanes <= 4 else reject("too-wide", "at most 4 lanes")

    @view
    def cost(self) -> int:
        return 10 * self.lanes

    exports = {WIDTH: width, COST: cost}


class Holder(Space):
    stage = Stage()
    widths = Members(WIDTH)
    costs = Members(COST)

    @derived
    def total_width(self) -> int:
        return sum(member.value for member in self.widths)

    @derived
    def total_cost(self) -> int:
        return sum(member.value for member in self.costs)


def test_a_value_export_answers_while_its_space_is_refused_and_a_view_export_refuses() -> None:
    report = design_space(Holder()).try_with_choices({Holder.stage.lanes: 8})
    assert not report.accepted and broken(report) == {"stage.fits": set()}
    point = built(design_space(Holder()), {"stage.lanes": 8})
    assert owners(inspection.validity(point.stage)) == ["stage.fits"]
    assert point.total_width == 16
    refused = point.query(Holder.total_cost)
    assert isinstance(refused, Rejected)
    assert {f.code for f in refused.findings} == {"too-wide"}
    fine = commit(design_space(Holder()), {"stage.lanes": 2})
    assert fine.total_width == 4 and fine.total_cost == 20


# -- possible cases read the constraints -------------------------------------------------


class Pumped(Space):
    simd: int = Decision(values=(1, 2, 4))
    pumping: bool = Decision(values=(False, True))
    style: str = Decision(values=("a", "b"))

    @derived
    def pumpable(self) -> bool:
        return self.simd >= 2

    @constraint
    def pumps(self) -> bool | Rejected:
        if self.pumping and not self.pumpable:
            return reject("pumping", "pumping needs SIMD of 2 or more")
        return True


def test_a_value_case_a_constraint_refuses_is_not_possible() -> None:
    point = commit(design_space(Pumped()), {"simd": 1})
    assert possible(point, Pumped.pumping) == (False,)
    assert refusals(point, Pumped.pumping) == {True: {"pumps": {"simd"}}}
    # While SIMD is open, the constraint waits for it: every value is possible.
    open_ = design_space(Pumped())
    assert possible(open_, Pumped.simd) == (1, 2, 4)
    assert possible(open_, Pumped.pumping) == (False, True)
    assert possible(open_, Pumped.style) == ("a", "b")


def test_a_change_to_a_value_a_constraint_refuses_is_refused_naming_what_stands() -> None:
    report = commit(design_space(Pumped()), {"simd": 1}).try_with_choices({Pumped.pumping: True})
    assert not report.accepted
    assert broken(report) == {"pumps": {"simd"}}


class Narrowing(Space):
    width: int = Param()

    @constraint
    def even(self) -> bool | Rejected:
        return True if self.width % 2 == 0 else reject("odd", "an even width only")


class Feeding(Space):
    width: int = Decision(values=(3, 4))
    child = Narrowing(width=width)


def test_a_value_case_is_tried_against_the_constraints_of_the_spaces_it_feeds() -> None:
    point = design_space(Feeding())
    assert isinstance(point.query(Feeding.width), Unresolved)
    assert possible(point, Feeding.width) == (4,)
    assert refusals(point, Feeding.width) == {3: {"child.even": set()}}


class Twin(Space):
    factor: int = Decision(values=(1, 2, 3))


class Pair(Space):
    first = Twin()
    second = Twin()

    @constraint
    def agree(self) -> bool | Rejected:
        if self.first.factor != self.second.factor:
            return reject("disagree", "the factors differ")
        return True


def test_an_ancestors_constraint_reading_a_decision_rules_out_its_cases() -> None:
    point = commit(design_space(Pair()), {"first.factor": 2})
    assert possible(point, Pair.second.factor) == (2,)
    assert refusals(point, Pair.second.factor) == {
        1: {"agree": {"first.factor"}},
        3: {"agree": {"first.factor"}},
    }
    # Open, the constraint reads both and refuses no value of either.
    assert possible(design_space(Pair()), Pair.second.factor) == (1, 2, 3)


class Source(Space):
    width: int = Decision(values=(3, 4))


class Wired(Space):
    source = Source()
    sink = Narrowing(width=source.width)


def test_a_siblings_constraint_reading_a_decision_through_a_formal_rules_out_its_cases() -> None:
    point = design_space(Wired())
    assert possible(point, Wired.source.width) == (4,)
    assert refusals(point, Wired.source.width) == {3: {"sink.even": set()}}


class Capped(Space):
    """A constraint that reads another choice before the Decision below it."""

    gate: int = Decision(values=(1, 2))
    inner = Twin()

    @constraint
    def capped(self) -> bool | Rejected:
        if self.gate == 2 and self.inner.factor > 1:
            return reject("capped", "the factor is capped at 1 behind the gate")
        return True


def test_a_value_is_refused_by_what_a_constraint_waited_on_before_reading_it() -> None:
    base = design_space(Capped())
    # While the gate is open the constraint stops there and does not read the factor.
    assert possible(base, Capped.inner.factor) == (1, 2, 3)
    report = base.try_with_choices({Capped.gate: 2})
    assert report.accepted
    point = report.instance
    assert possible(point, Capped.inner.factor) == (1,)
    assert refusals(point, Capped.inner.factor) == {
        2: {"capped": {"gate"}},
        3: {"capped": {"gate"}},
    }


class Small(Space):
    pass


class Large(Space):
    pass


class Chooser(Space):
    """The owner's constraint reads its own selector, then a formal and another choice."""

    strict: bool = Param()
    other: int = Decision(values=(1, 2))
    core: Small | Large = Decision({"small": Small, "large": Large})
    which = selected(core)

    @constraint
    def allowed(self) -> bool | Rejected:
        if self.which == "small" and self.strict and self.other == 2:
            return reject("small-ruled-out", "the small core is ruled out")
        return True


def test_a_selectors_case_is_tried_against_its_owners_constraint_reading_the_selector() -> None:
    point = design_space(Chooser(strict=True))
    # While the other choice is open, the constraint waits for it: both cores are possible.
    assert possible(point, Chooser.other) == (1, 2)
    assert possible(point, Chooser.core) == ("small", "large")
    report = point.try_with_choices({Chooser.other: 2})
    assert report.accepted
    assert possible(report.instance, Chooser.core) == ("large",)
    assert refusals(report.instance, Chooser.core) == {"small": {"allowed": {"other"}}}


def test_a_selectors_values_are_found_again_under_other_facts_its_owners_constraint_read() -> None:
    point = commit(design_space(Chooser(strict=False)), {"other": 2})
    assert possible(point, Chooser.core) == ("small", "large")
    report = point.try_with_facts({Chooser.strict: True})
    assert report.accepted
    assert possible(report.instance, Chooser.core) == ("large",)
    assert refusals(report.instance, Chooser.core) == refusals(
        commit(design_space(Chooser(strict=True)), {"other": 2}), Chooser.core
    )


class Storage(Space):
    style: str = Decision(values=("lut", "bram"))


class Queue(Space):
    storage = Storage()


class Wire(Space):
    pass


class Channel(Space):
    """A channel's constraint reads its own transport selector, and a choice two Spaces
    below it, inside the selector's candidate."""

    depth: int = Param()
    transport: Queue | Wire = Decision({"queue": Queue, "wire": Wire})

    @constraint
    def compatible(self) -> bool | Rejected:
        transport = self.transport
        if isinstance(transport, Wire):
            return True if self.depth == 0 else reject("wire-depth", "a wire holds no data")
        if self.depth > 32 and transport.storage.style == "lut":
            return reject("lut-depth", "a deep queue needs block RAM")
        return True


def test_an_ancestors_constraint_refuses_its_own_selector_and_a_choice_two_spaces_below() -> None:
    assert possible(design_space(Channel(depth=0)), Channel.transport) == ("queue", "wire")
    shallow = design_space(Channel(depth=8))
    assert refusals(shallow, Channel.transport) == {"wire": {"compatible": set()}}
    deep = design_space(Channel(depth=64))
    # The queue is possible: setting it opens the storage style, on which the rule waits.
    assert possible(deep, Channel.transport) == ("queue",)
    # Set while the transport is open, the style waits: both values are possible.
    style = Channel.transport["queue"].storage.style  # type: ignore[index]
    assert possible(deep, style) == ("lut", "bram")
    stored = commit(deep, {"transport.queue.storage.style": "lut"})
    assert possible(stored, Channel.transport) == ()
    assert refusals(stored, Channel.transport) == {
        "queue": {"compatible": {"transport.queue.storage.style"}},
        "wire": {"compatible": set()},
    }
    validity = inspection.validity(stored)
    assert validity.status == "open" and owners(validity) == ["compatible"]


# -- domains: a listed domain lists all its values ---------------------------------------


class Listed(Space):
    """``x`` lists 1, 2 and 3 up to ``limit`` and its membership rejects 2; ``y`` lists 1 and
    2 and rejects 2; ``lanes`` lists the divisors of ``limit``. ``small`` reads ``other``
    and ``spare``, and no domain reads either."""

    limit: int = Param()
    other: int = Param()
    x: int = Decision(
        domain=domain(
            accepts=lambda *, candidate, limit: candidate != 2 and candidate <= limit,
            candidates=lambda *, limit: tuple(value for value in (1, 2, 3) if value <= limit),
            limit=limit,
        )
    )
    y: int = Decision(
        domain=domain(accepts=lambda *, candidate: candidate != 2, candidates=lambda: (1, 2))
    )
    lanes: int = Decision(domain=divisors_of(limit))
    spare: int = Decision(values=(1, 2))

    @constraint
    def small(self) -> bool | Rejected:
        return True if self.spare <= self.other else reject("big", "spare past other")


def test_a_choices_type_check_reads_only_its_domains_inputs() -> None:
    point = design_space(Listed(limit=4, other=1)).with_choices(x=1)
    current = occurrence.state(point)
    index = occurrence.decision_index(point, Listed.x)
    entry = current.world.entry(index)
    assert entry.result == Available(1)
    keys = {current.linked.nodes[item].key for item in current.evaluator.bits.nodes(entry.reads)}
    assert keys == {"x", "limit"}


def test_a_value_the_domains_membership_rejects_is_refused_at_the_change() -> None:
    point = design_space(Listed(limit=4, other=2))
    # Nothing is filled in for ``y``, though one of its listed values passes membership.
    assert isinstance(point.query(Listed.y), Unresolved)
    report = point.try_with_choices({Listed.x: 2})
    assert not report.accepted
    refused = report.outcomes[0].result
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"domain-membership"}


def test_a_listed_value_the_domains_membership_rejects_is_not_listed() -> None:
    point = design_space(Listed(limit=4, other=2))
    assert possible(point, Listed.x) == (1, 3)
    assert possible(point, Listed.y) == (1,)


class Empty(Space):
    z: int = Decision(values=())


def test_an_empty_list_refuses_the_configuration_and_the_choice_stays_open() -> None:
    point = design_space(Empty())
    validity = inspection.validity(point)
    assert validity.status == "refused" and owners(validity) == ["z"]
    assert codes(validity.pieces[0]) == {"domain-empty"}
    (finding,) = validity.pieces[0].result.findings
    assert "lists no value" in finding.message
    # The refusal is the piece's: a read of the choice answers it is open.
    read = point.query(Empty.z)
    assert isinstance(read, Unresolved)
    assert {finding.code for finding in read.findings} == {"decision-unassigned"}


def test_a_domain_whose_computed_list_is_empty_refuses_its_decision() -> None:
    # The divisors of 0 are none: ``lanes`` is refused, and so is ``x`` (nothing up to 0).
    validity = inspection.validity(design_space(Listed(limit=0, other=2)))
    assert validity.status == "refused" and owners(validity) == ["lanes", "x"]
    assert all(codes(piece) == {"domain-empty"} for piece in validity.pieces)


def test_an_empty_list_under_a_selector_still_open_is_an_unknown_piece() -> None:
    class Hollow(Space):
        width: int = Param()
        z: int = Decision(values=())

    class Open(Space):
        width: int = Param()
        compute: Plain | Hollow = Decision({"plain": Plain, "hollow": Hollow}, width=width)

    # Its presence waits: the piece waits on the selector, as every piece of its Space.
    validity = inspection.validity(design_space(Open(width=8)))
    assert validity.status == "open" and owners(validity) == ["compute.hollow.z"]
    assert {(f.code, f.owner) for f in validity.pieces[0].result.findings} == {
        ("decision-unassigned", "compute")
    }


# -- a domain's input refusing on facts refuses the assignment ---------------------------


class Lanes(Space):
    """``width`` refuses an odd ``w``; ``lanes`` lists the divisors of ``width`` among 1, 2,
    4 and 8."""

    w: int = Param()

    @derived
    def width(self) -> int | Rejected:
        return reject("bad-width", "an odd width") if self.w % 2 else self.w

    lanes: int = Decision(
        domain=domain(
            accepts=lambda *, candidate, width: width % candidate == 0,
            candidates=lambda *, width: tuple(d for d in (1, 2, 4, 8) if width % d == 0),
            width=width,
        )
    )


def refused_through_width(answer: object) -> bool:
    return isinstance(answer, Rejected) and {(f.code, f.owner) for f in answer.findings} == {
        ("bad-width", "width")
    }


def test_a_value_whose_domain_input_is_refused_is_refused_naming_the_input() -> None:
    report = design_space(Lanes(w=7)).try_with_choices({Lanes.lanes: 1})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert outcome.status == "refused"
    assert refused_through_width(outcome.result)


def test_new_facts_that_refuse_a_stored_choices_domain_input_are_refused() -> None:
    kept = commit(design_space(Lanes(w=8)), {"lanes": 4})
    report = kept.try_with_facts({Lanes.w: 7})
    assert not report.accepted and report.instance is kept
    (reached,) = [outcome for outcome in report.outcomes if not outcome.requested]
    assert reached.owner == "lanes" and reached.status == "refused"
    assert refused_through_width(reached.result)
    # Cleared in the same change, the facts still refuse it: the open choice's list
    # cannot be made, false through ``width`` on the facts alone.
    cleared = kept.try_with_facts({Lanes.w: 7}, kept.field(Lanes.lanes).clear())
    assert not cleared.accepted and broken(cleared) == {"width": set()}


def test_only_a_domain_that_answers_false_refuses_a_commit() -> None:
    report = design_space(Lanes(w=8)).try_with_choices({Lanes.lanes: 3})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert isinstance(outcome.result, Rejected)
    assert {finding.code for finding in outcome.result.findings} == {"domain-membership"}


# -- every kept commitment reads as committed ----------------------------------------------


class Store(Space):
    """A memory whose rule needs UltraRAM for ``ultra``: a rule of the Space's, not the
    choice's domain, so the choice's own check holds whatever the platform has."""

    ultra: bool = Param()
    style: str = Decision(values=("block", "ultra"))

    @constraint
    def has_ultra(self) -> bool | Rejected:
        if self.style == "ultra" and not self.ultra:
            return reject("ultra-absent", "the platform has no UltraRAM")
        return True


class Sourced(Space):
    """The memory as its source's one case: nothing selects it but a change."""

    ultra: bool = Param()
    source: Store = Decision({"memory": Store}, ultra=ultra)


def committed_choices(point: Space) -> dict[str, object]:
    """Every Decision whose state is committed, by key, with its stored value: what saving
    a configuration writes."""
    found: dict[str, object] = {}
    for item in inspection.decisions(point):
        state = point.field(item.reference).state
        if isinstance(state, Available) and state.value.status == "committed":
            found[item.key] = state.value.value
    return found


def reads(point: Space) -> dict[str, object]:
    """What a read of each Decision answers, by key."""
    return {item.key: point.query(item.reference) for item in inspection.decisions(point)}


def kept(built: Any, choices: dict[str, object], key: str, value: object) -> object:
    """``choices`` committed on ``built`` keep ``key`` committed with ``value``; saved and
    committed again on a fresh binding of the same facts, every state and read is the same.
    The read of ``key``, for the kind to check."""
    point = commit(design_space(built()), choices)
    reference = {item.key: item.reference for item in inspection.decisions(point)}[key]
    assert point.field(reference).state == Available(DecisionState(key, "committed", value))
    assert committed_choices(point)[key] == value
    again = commit(design_space(built()), committed_choices(point))
    assert committed_choices(again) == committed_choices(point)
    assert reads(again) == reads(point)
    assert inspection.validity(again) == inspection.validity(point)
    return point.query(reference)


def test_a_checked_choice_is_committed_and_reads_as_its_value() -> None:
    assert kept(lambda: Lanes(w=8), {"lanes": 4}, "lanes", 4) == Available(4)


def test_a_choice_stored_under_an_open_selector_is_committed_and_waits_on_it() -> None:
    for ultra in (False, True):
        read = kept(
            lambda: Sourced(ultra=ultra),
            {"source.memory.style": "ultra"},
            "source.memory.style",
            "ultra",
        )
        assert isinstance(read, Unresolved)
        assert {(f.code, f.owner) for f in read.findings} == {("decision-unassigned", "source")}


def test_a_switch_bringing_a_stored_choice_into_play_is_refused_by_the_rule_it_breaks() -> None:
    point = commit(design_space(Sourced(ultra=False)), {"source.memory.style": "ultra"})
    report = point.try_with_choices({Sourced.source: "memory"})
    assert not report.accepted
    assert broken(report) == {"source.memory.has_ultra": {"source.memory.style"}}


# -- a domain input that exists only under another choice: a definition error ------------


class Banks(Space):
    """A choice whose domain reads ``width``, a value that applies only while ``wide``
    holds."""

    wide: bool = Decision(values=(False, True))

    @derived(when=wide)
    def width(self) -> int:
        return 8

    lanes: int = Decision(
        domain=domain(
            accepts=lambda *, candidate, width: width % candidate == 0,
            candidates=lambda *, width: tuple(d for d in (1, 2, 4, 8) if width % d == 0),
            width=width,
        )
    )


def test_a_domain_input_that_another_choices_guard_decides_is_a_definition_error() -> None:
    """``width`` exists while ``wide`` holds, a choice: what the domain reads would then
    depend on a choice, which is a constraint's to say."""
    with refused("domain-reads-choice", owner="lanes", message="reads the choice wide"):
        design_space(Banks())


class Spanned(Space):
    @derived
    def span(self) -> int:
        return 8


class Shallow(Space):
    pass


class Banked(Space):
    """A choice whose domain reads ``span``, a value of the selector's ``spanned`` case."""

    mode: Spanned | Shallow = Decision({"spanned": Spanned, "shallow": Shallow})
    lanes: int = Decision(
        domain=domain(
            accepts=lambda *, candidate, span: span % candidate == 0,
            candidates=lambda *, span: tuple(d for d in (1, 2, 4, 8) if span % d == 0),
            span=cast(Any, mode)["spanned"].span,
        )
    )


def test_a_domain_input_of_another_case_is_a_definition_error() -> None:
    with refused("domain-reads-choice", owner="lanes", message="reads the choice mode"):
        design_space(Banked())
