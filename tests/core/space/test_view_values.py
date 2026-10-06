# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A view reads as its accepted value, on declarations and on configurations.

On a configuration ``point.kitchen.cost`` is the accepted value (a read of a
view that is not accepted raises ``ValueUnavailableError`` carrying its
result). On a declaration ``kitchen.cost`` is a reference to that value: it
supplies a formal, feeds ``Present`` and joins ``requires=``. Assessment and
query are explicit calls on the configuration: ``point.inspect(Hall.total)``
and ``point.query(Hall.total)``.
"""

from __future__ import annotations

from typing import cast

import pytest

import finn.core.space as space
from finn.core.space import (
    Available,
    BoundValue,
    Decision,
    DefinitionError,
    Inapplicable,
    Param,
    Present,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    ValueUnavailableError,
    View,
    constraint,
    derived,
    design_space,
    reject,
    view,
)


class Room(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @constraint
    def small_enough(self) -> bool | Rejected:
        return True if self.area <= 30 else reject("too-large", f"area {self.area} exceeds 30")

    @view(requires=(small_enough,))
    def cost(self) -> int:
        return self.area * self.finish


class Hall(Space):
    want_garage: bool = Decision(values=(False, True))
    kitchen = Room(area=12)
    dining = Room(area=16)
    garage = Room(area=30, when=want_garage)
    hall = Room()
    hall.area = kitchen.cost  # a view's accepted value supplies a formal
    annex = Room(area=dining.cost)  # at the call, too
    porch = Room(area=Present(garage.cost))  # whichever is present
    doubled = View(kitchen.cost * 2)  # an expression over a view reference
    checked = View(dining.area, requires=(kitchen.cost, dining.small_enough))

    @view(requires=(kitchen.cost, dining.cost))
    def total(self) -> int:
        return self.kitchen.cost + self.dining.cost


def chosen(kitchen: int = 2, dining: int = 1) -> Hall:
    base = design_space(Hall())
    return base.with_choices({Hall.kitchen.finish: kitchen, Hall.dining.finish: dining})


def test_a_view_reads_as_its_accepted_value_on_a_configuration() -> None:
    point = chosen()
    assert point.kitchen.cost == 24 and point.dining.cost == 16
    assert point.total == 40
    assert point.doubled == 48
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
    # A refused obligation: the read carries the refusal.
    refused = chosen(kitchen=3).with_choices({Hall.hall.finish: 1})
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


def test_a_view_read_inside_a_computation_blocks_like_any_read() -> None:
    calls: list[str] = []

    class Reader(Space):
        kitchen = Room(area=12)
        dining = Room(area=16)

        @derived
        def sum_of_costs(self) -> int:
            calls.append("started")
            value = self.kitchen.cost + self.dining.cost
            calls.append("finished")
            return value

        @view(requires=(kitchen.cost, dining.cost))
        def total(self) -> int:
            return self.sum_of_costs

    base = design_space(Reader())
    assert isinstance(base.query(Reader.sum_of_costs), Unresolved)
    assert calls == ["started"]  # halted at the first unavailable view read
    # The per-view obligation results are unchanged: each view counts by its acceptance.
    results = base.inspect(Reader.total).constraints.results
    assert set(results) == {"kitchen.cost", "dining.cost"}
    assert all(isinstance(answer, Unresolved) for answer in results.values())
    half = base.with_choices({Reader.kitchen.finish: 2})
    results = half.inspect(Reader.total).constraints.results
    assert results["kitchen.cost"] == Available(True)  # its acceptance, not its value
    assert isinstance(results["dining.cost"], Unresolved)
    point = half.with_choices({Reader.dining.finish: 1})
    assert point.total == 40
    assert calls[-1] == "finished"


def test_a_view_reference_supplies_a_formal_by_assignment_and_at_the_call() -> None:
    base = design_space(Hall())
    assert isinstance(base.hall.query(Room.area), Unresolved)  # waits for kitchen.cost
    point = chosen()
    assert point.hall.area == 24  # hall.area = kitchen.cost
    assert point.annex.area == 16  # Room(area=dining.cost)
    # A formal supplied by a refused view is refused with the view's refusal.
    big = chosen(kitchen=3).with_choices({Hall.hall.finish: 2})
    assert big.hall.area == 36 and isinstance(big.query(Hall.hall.cost), Rejected)
    # Present over a guarded view: absent until the garage is wanted.
    assert isinstance(point.with_choices(want_garage=False).query(Hall.porch.area), Unresolved)
    garage = point.with_choices({Hall.want_garage: True, Hall.garage.finish: 1})
    assert garage.porch.area == 30


def test_view_references_are_obligations_in_both_view_forms() -> None:
    point = chosen()
    assessment = point.inspect(Hall.checked)
    assert assessment.constraints.results == {
        "kitchen.cost": Available(True),
        "dining.small_enough": Available(True),
    }
    refused = chosen(kitchen=3).with_choices({Hall.hall.finish: 1})
    # hall.cost refuses, but the obligations of total are kitchen and dining only.
    assert refused.total == 36 + 16
    total = refused.inspect(Hall.total)
    assert set(total.constraints.results) == {"kitchen.cost", "dining.cost"}

    class Strict(Space):
        kitchen = Room(area=50)
        dining = Room(area=10)
        checked = View(dining.area, requires=(kitchen.cost,))

    strict = design_space(Strict()).with_choices(
        {Strict.kitchen.finish: 1, Strict.dining.finish: 1}
    )
    results = strict.inspect(Strict.checked).constraints.results
    assert isinstance(results["kitchen.cost"], Rejected)
    with pytest.raises(ValueUnavailableError) as caught:
        _ = strict.checked
    assert isinstance(caught.value.result, Rejected)


def test_only_views_and_constraints_are_obligations() -> None:
    class Wrong(Space):
        kitchen = Room(area=12)
        checked = View(kitchen.area, requires=(kitchen.area,))

    with pytest.raises(DefinitionError, match="a view may require only constraints"):
        design_space(Wrong())


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
    # In its own body a view is the declaration (View[T]): it may be required,
    # exported or wrapped, and at runtime it supplies a formal like any
    # reference. Statically it is not the T a formal takes.
    class Wing(Space):
        kitchen = Room(area=12)

        @view
        def doubled(self) -> int:
            return 2 * self.kitchen.cost

        annex = Room(area=cast(int, doubled))
        checked = View(kitchen.area, requires=(doubled,))

    point = design_space(Wing()).with_choices({Wing.kitchen.finish: 1})
    assert point.annex.area == 24
    assert point.inspect(Wing.checked).constraints.results == {"doubled": Available(True)}


def test_the_callable_view_surface_is_gone() -> None:
    assert not hasattr(space, "BoundView")
    assert not hasattr(space, "accepted")
    assert "BoundView" not in space.__all__ and "accepted" not in space.__all__
    assert not hasattr(design_space(Hall()), "view")
    assert isinstance(Hall.total, View)  # class access is the declaration
