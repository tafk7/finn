# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A view reads as its accepted value, on declarations and on configurations.

On a configuration ``point.kitchen.cost`` is the accepted value (a read of a
view that is not accepted raises ``ValueUnavailableError`` carrying its
result): refused with its Space, and answering while its Space is open. On a
declaration ``kitchen.cost`` is a reference to that value, read by another view and
by nothing else: a view is read out, so a formal is supplied by a derived value
(``kitchen.price``), never by a view. Assessment and query are explicit calls on the
configuration: ``point.inspect(Hall.total)`` and ``point.query(Hall.total)``.
"""

from __future__ import annotations

from typing import cast

import pytest
from support.refusals import refused
from support.space_cores import built

import finn.core.space as space
from finn.core.space import (
    Available,
    BoundValue,
    Decision,
    DefinitionError,
    Inapplicable,
    Param,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    ValueUnavailableError,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.core.space.declarations import View


class Room(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @constraint
    def small_enough(self) -> bool | Rejected:
        return True if self.area <= 30 else reject("too-large", f"area {self.area} exceeds 30")

    @derived
    def price(self) -> int:
        return self.area * self.finish

    cost = view(price)


class Hall(Space):
    want_garage: bool = Decision(values=(False, True))
    kitchen = Room(area=12)
    dining = Room(area=16)
    garage = Room(area=30, when=want_garage)
    hall = Room()
    hall.area = kitchen.price  # a derived value supplies a formal
    annex = Room(area=dining.price)  # at the call, too
    checked = view(dining.area)

    @view
    def doubled(self) -> int:
        return self.kitchen.cost * 2  # a view reads a view

    @view
    def total(self) -> int:
        return self.kitchen.cost + self.dining.cost


def chosen(kitchen: int = 2, dining: int = 1) -> Hall:
    base = design_space(Hall())
    return base.with_choices({Hall.kitchen.finish: kitchen, Hall.dining.finish: dining})


def too_large(hall: int) -> Hall:
    """The kitchen at finish 3 makes the hall 36 large, over its 30: the change is refused
    by the hall's rule, false on its own values; built directly, the configuration reads
    refused."""
    report = design_space(Hall()).try_with_choices(
        {Hall.kitchen.finish: 3, Hall.dining.finish: 1, Hall.hall.finish: hall}
    )
    assert not report.accepted
    assert [(rule.rule, rule.standing) for rule in report.broken] == [("hall.small_enough", ())]
    choices = {"kitchen.finish": 3, "dining.finish": 1, "hall.finish": hall}
    return cast(Hall, built(design_space(Hall()), choices))


def test_a_view_reads_as_its_accepted_value_on_a_configuration() -> None:
    point = chosen()
    assert point.kitchen.cost == 24 and point.dining.cost == 16
    assert point.total == 40
    assert point.doubled == 48  # the kitchen's cost, 12 * 2 = 24, doubled
    assert point.checked == 16
    # The read, the query and the assessment agree.
    assert point.query(Hall.total) == Available(40)
    assert point.inspect(Hall.total).accepted_result == Available(40)
    with pytest.raises(TypeError, match="not callable"):
        point.total()  # type: ignore[operator]


def test_an_unaccepted_view_read_raises_carrying_its_result() -> None:
    base = design_space(Hall())
    with pytest.raises(ValueUnavailableError) as unresolved:
        _ = base.kitchen.cost
    assert isinstance(unresolved.value.result, Unresolved)
    assert unresolved.value.result == base.query(Hall.kitchen.cost)
    assert {finding.owner for finding in unresolved.value.result.findings} == {"kitchen.finish"}
    # A refused Space: the read of its view carries the refusal.
    refused = too_large(hall=1)
    with pytest.raises(ValueUnavailableError) as rejected:
        _ = refused.hall.cost
    assert isinstance(rejected.value.result, Rejected)
    assert {finding.code for finding in rejected.value.result.findings} == {"too-large"}
    assert rejected.value.result == refused.inspect(Hall.hall.cost).accepted_result
    # An absent node's view is inapplicable.
    without = base.with_choices(want_garage=False)
    with pytest.raises(ValueUnavailableError) as absent:
        _ = without.garage.cost
    assert isinstance(absent.value.result, Inapplicable)


def test_a_derived_value_read_inside_a_computation_blocks_like_any_read() -> None:
    calls: list[str] = []

    class Reader(Space):
        kitchen = Room(area=12)
        dining = Room(area=16)

        @derived
        def sum_of_costs(self) -> int:
            calls.append("started")
            value = self.kitchen.price + self.dining.price
            calls.append("finished")
            return value

        @view
        def total(self) -> int:
            return self.sum_of_costs

    base = design_space(Reader())
    assert isinstance(base.query(Reader.sum_of_costs), Unresolved)
    assert calls == ["started"]  # halted at the first unavailable read
    # The Space is valid (an open choice is no piece): the view waits on its output alone.
    assessment = base.inspect(Reader.total)
    assert assessment.validity == Available(True)
    assert isinstance(assessment.output_result, Unresolved)
    assert assessment.accepted_result == assessment.output_result
    half = base.with_choices({Reader.kitchen.finish: 2})
    assert isinstance(half.inspect(Reader.total).accepted_result, Unresolved)
    point = half.with_choices({Reader.dining.finish: 1})
    assert point.total == 40
    assert calls[-1] == "finished"


def test_a_derived_value_supplies_a_formal_by_assignment_and_at_the_call() -> None:
    base = design_space(Hall())
    assert isinstance(base.hall.query(Room.area), Unresolved)  # waits for kitchen.price
    point = chosen()
    assert point.hall.area == 24  # hall.area = kitchen.price
    assert point.annex.area == 16  # Room(area=dining.price)
    # The formal answers where the hall is refused; its view is refused with it.
    big = too_large(hall=2)
    assert big.hall.area == 36 and isinstance(big.query(Hall.hall.cost), Rejected)


def test_a_view_refuses_with_its_space_and_answers_while_it_is_open_in_both_forms() -> None:
    point = chosen()
    # The garage's presence waits on its Decision: its constraint is unknown under it, the
    # hall is open, and its views answer.
    validity = inspection.validity(point)
    assert validity.status == "open"
    assert [piece.owner for piece in validity.pieces] == ["garage.small_enough"]
    assert point.checked == 16 and point.total == 40
    assert point.inspect(Hall.checked).validity == validity.result
    # A child refused refuses the hall, and every view of it, in both forms.
    refused = too_large(hall=1)
    validity = inspection.validity(refused)
    assert validity.status == "refused"
    assert [piece.owner for piece in validity.pieces] == ["hall.small_enough"]
    for view_ in (Hall.total, Hall.checked):
        answer = refused.query(view_)
        assert isinstance(answer, Rejected)
        assert {finding.code for finding in answer.findings} == {"too-large"}
    # The children that are valid still answer theirs.
    assert refused.kitchen.cost == 36 and refused.dining.cost == 16

    class Strict(Space):
        kitchen = Room(area=50)
        dining = Room(area=10)
        checked = view(dining.area)

    strict = design_space(Strict()).with_choices(
        {Strict.kitchen.finish: 1, Strict.dining.finish: 1}
    )
    assert isinstance(strict.inspect(Strict.checked).validity, Rejected)
    with pytest.raises(ValueUnavailableError) as caught:
        _ = strict.checked
    assert isinstance(caught.value.result, Rejected)


def test_inspect_and_query_are_explicit_calls_on_the_configuration() -> None:
    point = chosen()
    # A view of a child: on the child's configuration, or through a path.
    on_child = point.kitchen.inspect(Room.cost)
    assert on_child == point.inspect(Hall.kitchen.cost)
    assert on_child.accepted_result == Available(24)
    assert point.kitchen.query(Room.cost) == point.query(Hall.kitchen.cost) == Available(24)
    assert point.inspect(Hall.kitchen.small_enough).result == Available(True)
    with pytest.raises(RequestError, match="kitchen.area is not a view or a constraint"):
        point.inspect(Hall.kitchen.area)
    # A view binds as a value accessor, like a derived value.
    field = point.field(Hall.total)
    assert type(field) is BoundValue
    assert field.get() == 40 and field.query() == Available(40)


def test_a_views_own_name_in_its_class_body_is_its_declaration() -> None:
    # In its own body a view is the declaration (View[T]): it may be wrapped, but it
    # supplies no formal: a view is read out, so a binding to it is refused. Statically
    # it is not the T a formal takes.
    class Plaque(Space):
        area: int = Param()

    class Wing(Space):
        kitchen = Room(area=12)

        @view
        def doubled(self) -> int:
            return 2 * self.kitchen.cost

        checked = view(doubled)

    point = design_space(Wing()).with_choices({Wing.kitchen.finish: 1})
    assert point.checked == 24

    class Fed(Space):
        kitchen = Room(area=12)

        @view
        def doubled(self) -> int:
            return 2 * self.kitchen.cost

        annex = Plaque(area=cast(int, doubled))

    with refused("view-read-in", error=DefinitionError):
        design_space(Fed())


def test_the_callable_view_surface_is_gone() -> None:
    assert not hasattr(space, "BoundView")
    assert not hasattr(space, "accepted")
    assert "BoundView" not in space.__all__ and "accepted" not in space.__all__
    assert not hasattr(design_space(Hall()), "view")
    assert isinstance(Hall.total, View)  # class access is the declaration
