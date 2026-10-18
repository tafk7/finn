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
against the choice's own domain only.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from support.space_cores import commit, forced

from finn.core.space import (
    Available,
    Decision,
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
    domain,
    inspection,
    occurrence,
    reject,
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


def test_a_placed_space_whose_presence_waits_is_an_unknown_piece() -> None:
    class Either(Space):
        width: int = Param()
        compute: Plain | Core = Decision({"plain": Plain, "core": Core}, width=width)

    validity = inspection.validity(design_space(Either(width=8)))
    assert validity.status == "undecided"
    assert owners(validity) == ["compute.core", "compute.plain"]


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


@pytest.mark.xfail(
    strict=True,
    reason="a value case is tried against the constraints of its Decision's Space and of the "
    "Spaces it places; an ancestor's constraint reading it is not tried yet",
)
def test_an_ancestors_constraint_reading_a_decision_rules_out_its_cases() -> None:
    point = commit(design_space(Pair()), {"first.factor": 2})
    assert forced(point) == {"second.factor": 2}
