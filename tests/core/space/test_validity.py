# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One kind of rule: a constraint owned by a Space, and the Space's validity.

A Space is **valid** when every piece that counts is true, **refused** when one is
false, and **undecided** otherwise (Kleene's AND: false beats unknown, unknown beats
true). The pieces are its constraints (every one, while the Space is present), its
present children's validity, its stored choices' own checks (an unchecked one is
unknown), and every open Decision with no possible case. A case not selected
contributes nothing. ``inspection.validity`` answers it, with the pieces that are not
true and their owners.

A view is a product: it refuses when its output or its Space is refused, answers while
its Space is undecided, and an inactive view answers ``Inapplicable`` without reading
its Space's validity. A constraint reads values; one that reads a view of its own Space,
or of an ancestor, closes a dependency cycle. An export may name a derived value, which
a holder reads as the value whatever the exporting Space's validity. A value case is
possible unless a constraint that reads its Decision refuses it, and a commit is checked
against the choice's own domain only. A state describes the commitment, a read the answer:
every kept choice is committed with its stored value, whether it reads as its value,
unresolved while unchecked, or refused through a refused guard or a refused input.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from support.space_cores import commit, forced

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
    _forcing,
    _runtime,
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

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")


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


def test_a_constraint_that_waits_leaves_the_space_undecided_naming_what_it_waits_on() -> None:
    point = design_space(Engine(width=8))
    # ``small`` refuses 2, so ``wanted`` is forced to 1 and ``small`` holds.
    assert forced(point) == {"wanted": 1}
    validity = inspection.validity(point)
    assert validity.status == "undecided" and isinstance(validity.result, Unresolved)
    assert owners(validity) == ["core.divides"]
    assert codes(validity.pieces[0]) == {"decision-unassigned"}
    assert {f.owner for f in validity.pieces[0].result.findings} == {"core.lanes"}


def test_a_false_piece_refuses_whatever_else_waits() -> None:
    # Committing a value a constraint refuses is accepted: the configuration reads refused.
    report = design_space(Engine(width=8)).try_with_choices({Engine.wanted: 2})
    assert report.accepted
    (outcome,) = report.outcomes
    assert outcome.result == Available(True)
    validity = inspection.validity(report.instance)
    assert validity.status == "refused"
    assert owners(validity) == ["small"]  # core.divides still waits: false beats unknown
    assert codes(validity.pieces[0]) == {"too-many"}


def test_a_refused_child_refuses_its_parent_naming_the_childs_piece() -> None:
    point = commit(design_space(Engine(width=6)), {"core.lanes": 4, "wanted": 1})
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


class Bounded(Space):
    """``size`` is checked against ``limit``: it is unchecked until ``limit`` is known."""

    limit: int = Decision(values=(3, 9))
    size: int = Decision(
        domain=domain(accepts=lambda *, candidate, limit: candidate <= limit, limit=limit)
    )


def test_an_unchecked_choice_is_an_unknown_piece() -> None:
    point = design_space(Bounded()).with_choices(size=8)
    assert inspection.unchecked(point) == (inspection.Unchecked("size", 8, ("limit",)),)
    validity = inspection.validity(point)
    assert validity.status == "undecided"
    assert owners(validity) == ["size"]
    assert codes(validity.pieces[0]) == {"decision-unchecked"}
    assert inspection.validity(point.with_choices(limit=9)).status == "valid"


class Starved(Space):
    """``lanes`` has no possible case: its constraint refuses every one."""

    width: int = Param()
    lanes: int = Decision(values=(2, 4))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.lanes > self.width:
            return reject("too-wide", "more lanes than width")
        return True


def test_an_open_decision_with_no_possible_case_refuses_the_space() -> None:
    point = design_space(Starved(width=1))
    validity = inspection.validity(point)
    assert validity.status == "refused"
    # The Decision is refused, and so is the constraint that reads it (its input refused).
    assert owners(validity) == ["fits", "lanes"]
    assert all(codes(piece) == {"decision-no-viable-case"} for piece in validity.pieces)
    (finding,) = validity.pieces[1].result.findings
    assert "too-wide" in finding.message
    # With a possible case, the open Decision is no piece: forced or still to choose.
    assert inspection.validity(design_space(Starved(width=2))).status == "valid"
    assert forced(design_space(Starved(width=2))) == {"lanes": 2}


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
    # The refusing candidate is not possible (its validity refuses), so plain is forced.
    point = design_space(Selector(width=4))
    assert forced(point) == {"compute": "plain"}
    assert inspection.validity(point).status == "valid"
    selected = commit(point, {"compute": "refusing"})
    assert owners(inspection.validity(selected)) == ["compute.refusing.never"]
    candidate = inspection.candidate(point, Selector.compute, "refusing")
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
    assert validity.status == "undecided"
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
    # selector is open. A choice stored inside is one, unknown until the selector settles.
    point = design_space(Open(width=8))
    assert inspection.validity(point) == Validity("valid", (), Available(True))
    kept = commit(point, {"compute.bare.lanes": 2})
    assert owners(inspection.validity(kept)) == ["compute.bare.lanes"]


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


def test_a_view_answers_while_its_space_is_undecided_and_refuses_with_it() -> None:
    open_ = design_space(Product(width=4))
    assert inspection.validity(open_).status == "undecided"
    assert open_.cycles == 4
    assessment = open_.inspect(Product.cycles)
    assert isinstance(assessment.validity, Unresolved) and assessment.readiness.ready is None
    refused = open_.with_choices(lanes=3)  # a commit is a check of the domain only
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
    entry = _runtime.evaluate(current, access)
    assert isinstance(entry.result, Inapplicable)
    validity = current.linked.scopes[point.y._scope].validity
    assert entry.dependencies == (current.linked.nodes[access].guard,)
    assert validity not in entry.dependencies
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

    point = design_space(Storage()).with_choices(depth=2)
    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        _ = point.storage
    assert caught.value.role == "dependency cycle"
    assert caught.value.detail == (
        "storage (value) -> $validity (value) -> capacity_supported (constraint) -> storage"
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

    point = design_space(Parent()).with_choices(lanes=1)
    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        inspection.validity(point)
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
    point = commit(design_space(Holder()), {"stage.lanes": 8})
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
    assert forced(point) == {"pumping": False}
    (item,) = inspection.forced(point)
    assert set(item.refused) == {"True"} and "pumps" in item.refused["True"]
    assert "pumping" in item.refused["True"]
    # While SIMD is open, the constraint waits for it: both cases are possible.
    assert {item.key: item.cases for item in inspection.viable(design_space(Pumped()))} == {
        "simd": (1, 2, 4),
        "pumping": (False, True),
        "style": ("a", "b"),
    }
    # Committing the refused case is accepted, and the configuration reads refused.
    accepted = point.try_with_choices({Pumped.pumping: True})
    assert accepted.accepted
    assert owners(inspection.validity(accepted.instance)) == ["pumps"]


def test_a_value_decision_no_constraint_reads_costs_no_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trials: list[str] = []
    extended = _forcing._extended

    def counted(base: Any, assignments: Any, added: int) -> Any:
        trials.append(base.linked.nodes[added].key)
        return extended(base, assignments, added)

    monkeypatch.setattr(_forcing, "_extended", counted)
    forced(commit(design_space(Pumped()), {"simd": 1}))
    assert "style" not in trials and "pumping" in trials


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
    assert forced(point) == {"width": 4}
    (item,) = inspection.forced(point)
    assert "odd" in item.refused["3"]


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
    assert forced(point) == {"second.factor": 2}
    (item,) = inspection.forced(point)
    assert {case: "disagree" in why for case, why in item.refused.items()} == {"1": True, "3": True}
    # Open, the constraint reads both and refuses no case of either.
    assert forced(design_space(Pair())) == {}


class Source(Space):
    width: int = Decision(values=(3, 4))


class Wired(Space):
    source = Source()
    sink = Narrowing(width=source.width)


def test_a_siblings_constraint_reading_a_decision_through_a_formal_rules_out_its_cases() -> None:
    point = design_space(Wired())
    assert forced(point) == {"source.width": 4}
    (item,) = inspection.forced(point)
    assert "odd" in item.refused["3"]


class Capped(Space):
    """A constraint that reads another choice before the Decision below it."""

    gate: int = Decision(values=(1, 2))
    inner = Twin()

    @constraint
    def capped(self) -> bool | Rejected:
        if self.gate == 2 and self.inner.factor > 1:
            return reject("capped", "the factor is capped at 1 behind the gate")
        return True


def test_a_decisions_verdict_reads_what_a_constraint_waited_on_before_reading_it() -> None:
    base = design_space(Capped())
    # While the gate is open the constraint stops there and does not read the factor, which
    # stays open; its verdict still read the gate, so committing it reaches the verdict.
    assert {item.key: item.cases for item in inspection.viable(base)}["inner.factor"] == (1, 2, 3)
    report = base.try_with_choices({Capped.gate: 2})
    assert report.accepted
    assert forced(report.instance) == {"inner.factor": 1}
    assert forced(commit(design_space(Capped()), {"gate": 2})) == {"inner.factor": 1}


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
    assert {item.key: item.cases for item in inspection.viable(point)} == {
        "other": (1, 2),
        "core": ("small", "large"),
    }
    # A change commits the other choice: the selector's verdict read it, through the
    # constraint, so the successor finds it again and forces the large core.
    report = point.try_with_choices({Chooser.other: 2})
    assert report.accepted
    assert forced(report.instance) == {"core": "large"}
    (item,) = inspection.forced(report.instance)
    assert "small-ruled-out" in item.refused["small"]
    assert forced(commit(design_space(Chooser(strict=True)), {"other": 2})) == {"core": "large"}


def test_a_selectors_verdict_is_found_again_under_other_facts_its_owners_constraint_read() -> None:
    point = commit(design_space(Chooser(strict=False)), {"other": 2})
    assert forced(point) == {}
    report = point.try_with_facts({Chooser.strict: True})
    assert report.accepted
    assert forced(report.instance) == {"core": "large"}
    assert forced(report.instance) == forced(
        commit(design_space(Chooser(strict=True)), {"other": 2})
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


def test_an_ancestors_constraint_forces_its_own_selector_and_a_choice_two_spaces_below() -> None:
    assert forced(design_space(Channel(depth=0))) == {}
    shallow = design_space(Channel(depth=8))
    assert forced(shallow) == {"transport": "queue"}
    deep = design_space(Channel(depth=64))
    assert forced(deep) == {"transport": "queue", "transport.queue.storage.style": "bram"}
    reasons = {item.key: item.refused for item in inspection.forced(deep)}
    assert "wire-depth" in reasons["transport"]["wire"]
    assert "lut-depth" in reasons["transport.queue.storage.style"]["'lut'"]
    # Under other facts, the choice below the selector is found again.
    report = shallow.try_with_facts({Channel.depth: 64})
    assert report.accepted and forced(report.instance) == forced(deep)
    # A stored choice there that the constraint refuses: the commit is accepted (its own
    # check holds), and the selector's trial reads it, so the queue is not possible either:
    # the transport has no possible case, and the configuration is refused naming it.
    stored = commit(deep, {"transport.queue.storage.style": "lut"})
    assert forced(stored) == {}
    (transport,) = inspection.viable(stored)
    assert transport.key == "transport" and transport.cases == ()
    assert "lut-depth" in transport.refused["queue"] and "wire-depth" in transport.refused["wire"]
    validity = inspection.validity(stored)
    assert validity.status == "refused" and {"compatible", "transport"} <= set(owners(validity))
    # Shallow, the same stored choice leaves the queue possible, and forced.
    assert forced(commit(shallow, {"transport.queue.storage.style": "lut"})) == {
        "transport": "queue"
    }


class Choosing(Space):
    width: int = Param()
    compute: Plain | Core = Decision({"plain": Plain, "core": Core}, width=width)


def test_a_decision_refused_on_forcings_copies_but_through_its_guard_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A selector's verdict reads its candidate's validity on copies that do not force, and
    # rests on the guards there, not on the Decisions: a Decision's piece is false there
    # only through its guard. An engine that broke that would raise, not answer.
    assert forced(design_space(Choosing(width=4))) == {}
    original = _runtime._forced

    def refusing(snapshot: _runtime.Snapshot, index: int) -> Any:
        if not snapshot.forcing and snapshot.linked.nodes[index].key == "compute.core.lanes":
            return reject("broken", "refused on a copy that does not force")
        return original(snapshot, index)

    monkeypatch.setattr(_runtime, "_forced", refusing)
    with pytest.raises(EvaluationError, match="its guard not"):
        forced(design_space(Choosing(width=4)))


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


def test_a_domains_enumeration_and_membership_read_only_its_own_inputs() -> None:
    point = design_space(Listed(limit=4, other=1))
    current = occurrence.state(point)
    nodes = current.linked.nodes
    index = occurrence.decision_index(point, Listed.x)

    def keys(read: frozenset[int]) -> set[str]:
        return {nodes[item].key for item in read}

    for entry in (
        _runtime.enumeration(current, index),
        _runtime.membership(current, index, 1),
        _runtime.listed_members(current, index, (1, 2, 3)),
    ):
        assert keys(entry.decisions) == set() and keys(entry.facts) == {"limit"}


def test_a_listed_value_the_domains_membership_rejects_is_no_case() -> None:
    point = design_space(Listed(limit=4, other=2))
    viable = {item.key: item for item in inspection.viable(point)}
    assert viable["x"].cases == (1, 3)
    assert set(viable["x"].refused) == {"2"} and "domain-membership" in viable["x"].refused["2"]
    # The one listed value its membership accepts is forced, the other named.
    assert forced(point) == {"y": 1}
    (item,) = inspection.forced(point)
    assert set(item.refused) == {"2"} and "domain-membership" in item.refused["2"]
    # Committing the rejected value is refused at the commit: the type check.
    report = point.try_with_choices({Listed.x: 2})
    assert not report.accepted
    refused = report.outcomes[0].result
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"domain-membership"}


class Empty(Space):
    z: int = Decision(values=())


def test_an_empty_list_refuses_its_decision_and_the_configuration() -> None:
    point = design_space(Empty())
    assert {item.key: item.cases for item in inspection.viable(point)} == {"z": ()}
    validity = inspection.validity(point)
    assert validity.status == "refused" and owners(validity) == ["z"]
    assert codes(validity.pieces[0]) == {"decision-no-viable-case"}
    (finding,) = validity.pieces[0].result.findings
    assert "lists no value" in finding.message
    assert isinstance(point.query(Empty.z), Rejected)


def test_a_domain_whose_computed_list_is_empty_refuses_its_decision() -> None:
    # The divisors of 0 are none: ``lanes`` is refused, and so is ``x`` (nothing up to 0).
    validity = inspection.validity(design_space(Listed(limit=0, other=2)))
    assert validity.status == "refused" and owners(validity) == ["lanes", "x"]
    assert all(codes(piece) == {"decision-no-viable-case"} for piece in validity.pieces)


def test_an_empty_list_under_a_selector_still_open_is_no_piece() -> None:
    class Hollow(Space):
        width: int = Param()
        z: int = Decision(values=())

    class Open(Space):
        width: int = Param()
        compute: Plain | Hollow = Decision({"plain": Plain, "hollow": Hollow}, width=width)

    # Its presence waits: it is neither forced nor refused for having no case.
    point = design_space(Open(width=8))
    assert inspection.validity(point).status == "valid"
    assert forced(point) == {}


# -- a refusal elsewhere never refuses a commit ------------------------------------------


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


def test_a_commit_whose_domain_input_is_refused_is_kept_and_reads_refused_through_it() -> None:
    report = design_space(Lanes(w=7)).try_with_choices({Lanes.lanes: 1})
    assert report.accepted
    (outcome,) = report.outcomes
    assert outcome.status == "changed" and not outcome.unchecked
    assert refused_through_width(outcome.result)
    point = report.instance
    # Kept, never read as a value: refused through the input, not waiting on anything.
    assert inspection.unchecked(point) == ()
    assert refused_through_width(point.query(Lanes.lanes))
    assert point.field(Lanes.lanes).state == Available(DecisionState("lanes", "committed", 1))
    validity = inspection.validity(point)
    assert validity.status == "refused" and owners(validity) == ["lanes"]
    assert refused_through_width(validity.pieces[0].result)


def test_new_facts_that_refuse_a_kept_choices_domain_input_are_accepted() -> None:
    kept = commit(design_space(Lanes(w=8)), {"lanes": 4})
    report = kept.try_with_facts({Lanes.w: 7})
    assert report.accepted
    (reached,) = [outcome for outcome in report.outcomes if not outcome.requested]
    assert reached.owner == "lanes" and refused_through_width(reached.result)
    assert refused_through_width(report.instance.query(Lanes.lanes))
    # Facts that make the input whole again make the choice readable.
    again = report.instance.with_facts({Lanes.w: 8})
    assert again.lanes == 4 and inspection.validity(again).status == "valid"


def test_only_a_domain_that_answers_false_refuses_a_commit() -> None:
    report = design_space(Lanes(w=8)).try_with_choices({Lanes.lanes: 3})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert isinstance(outcome.result, Rejected)
    assert {finding.code for finding in outcome.result.findings} == {"domain-membership"}


# -- every kept commitment reads as committed ----------------------------------------------


class Beats(Space):
    """A beat whose domain reads ``size``, a choice: committed while ``size`` is open, the
    beat is kept unchecked."""

    size: int = Decision(values=(4, 8))
    beat: int = Decision(domain=divisors_of(size))


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
    """The memory as its source's one case: ``ultra`` stored without UltraRAM leaves the
    source no possible case, which refuses the stored style's guard."""

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


def test_an_unchecked_choice_is_committed_and_reads_unresolved() -> None:
    read = kept(Beats, {"beat": 2}, "beat", 2)
    assert isinstance(read, Unresolved)
    assert {(f.code, f.owner) for f in read.findings} == {("decision-unchecked", "beat")}


def test_a_choice_admitted_under_a_refused_guard_is_committed_and_reads_refused_through_it() -> (
    None
):
    read = kept(
        lambda: Sourced(ultra=False),
        {"source.memory.style": "ultra"},
        "source.memory.style",
        "ultra",
    )
    assert isinstance(read, Rejected)
    assert {(f.code, f.owner) for f in read.findings} == {("decision-no-viable-case", "source")}
    # The rule is the cause, the choice's own check holding: with UltraRAM it reads as its value.
    assert kept(
        lambda: Sourced(ultra=True),
        {"source.memory.style": "ultra"},
        "source.memory.style",
        "ultra",
    ) == Available("ultra")


def test_a_choice_kept_through_a_refused_input_is_committed_and_reads_refused_through_it() -> None:
    assert refused_through_width(kept(lambda: Lanes(w=7), {"lanes": 1}, "lanes", 1))


# -- a domain input that does not apply ----------------------------------------------------


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


def test_a_commit_whose_domain_input_does_not_apply_is_kept_unchecked_waiting_on_its_guard() -> (
    None
):
    """The choice waits on what decides that its input applies (``wide``, the input's guard
    reads it), not on the input."""
    narrow = commit(design_space(Banks()), {"wide": False})
    assert isinstance(narrow.query(Banks.width), Inapplicable)
    report = narrow.try_with_choices({Banks.lanes: 3})
    assert report.accepted
    (outcome,) = report.outcomes
    assert outcome.status == "changed" and outcome.unchecked
    assert isinstance(outcome.result, Unresolved)
    assert [(f.code, f.owner, f.details) for f in outcome.result.findings] == [
        ("input-inapplicable", "wide", (("input", "width"),))
    ]
    point = report.instance
    assert isinstance(point.query(Banks.lanes), Unresolved)
    assert inspection.unchecked(point) == (inspection.Unchecked("lanes", 3, ("wide",)),)
    assert inspection.validity(point).status == "undecided"
    assert point.field(Banks.lanes).state == Available(DecisionState("lanes", "committed", 3))
    # The input applying, the choice is checked: a value its domain admits reads as itself,
    # one it does not refuses the change.
    assert point.try_with_choices({Banks.wide: True, Banks.lanes: 4}).instance.lanes == 4
    refused = point.try_with_choices({Banks.wide: True})
    assert not refused.accepted
    assert [(o.owner, o.status) for o in refused.outcomes if o.status == "refused"] == [
        ("lanes", "refused")
    ]


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


def test_a_commit_whose_domain_input_another_case_leaves_out_waits_on_the_selector() -> None:
    shallow = commit(design_space(Banked()), {"mode": "shallow"})
    report = shallow.try_with_choices({Banked.lanes: 3})
    assert report.accepted
    (outcome,) = report.outcomes
    assert outcome.unchecked and isinstance(outcome.result, Unresolved)
    assert [(f.code, f.owner, f.details) for f in outcome.result.findings] == [
        ("input-inapplicable", "mode", (("input", "mode.spanned.span"),))
    ]
    point = report.instance
    assert inspection.unchecked(point) == (inspection.Unchecked("lanes", 3, ("mode",)),)
    # The case that holds the input selected, the choice is checked, and 3 does not divide 8.
    assert not point.try_with_choices({Banked.mode: "spanned"}).accepted
    assert point.try_with_choices({Banked.mode: "spanned", Banked.lanes: 2}).instance.lanes == 2
