# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The body that declares a node supplies its formals, once; the rest is its Space class's.

A formal is supplied at the call or by assignment on the declaration. A Decision,
a Decision over nodes, a child node and behaviour belong to the Space class: no
body supplies them, and a subclass changes them. A node below the declaration is
supplied where it is declared, never through a path. A root supplies plain
values; structure belongs to a Space class. None of this is about hardware.
"""

from __future__ import annotations

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    Members,
    Param,
    Rejected,
    Space,
    ViewKey,
    composite,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)

COST = ViewKey("cost", int)


class Room(Space):
    area: int = Param(default=12)
    finish: int = Decision(values=(1, 2, 3))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.area > 20:
            return reject("too-large", f"area {self.area} exceeds 20")
        return True

    @view(requires=(fits,))
    def cost(self) -> int:
        return self.area * self.finish

    exports = {COST: cost}


class Wing(Space):
    kitchen = Room(area=14)
    study = Room()
    costs = Members(COST)


def test_a_formal_is_supplied_at_the_call_or_by_assignment() -> None:
    class Hall(Space):
        kitchen = Room(area=14)
        hall = Room()
        hall.area = kitchen.area

    point = design_space(Hall()).with_choices({Hall.kitchen.finish: 1, Hall.hall.finish: 2})
    assert (point.kitchen.area, point.hall.area) == (14, 14)
    assert point.hall.cost == 28
    # A formal nothing supplies keeps its declared default.
    assert design_space(Wing()).study.area == 12


def test_one_body_supplying_a_formal_twice_is_a_definition_error() -> None:
    with refused(
        "formal-assigned-twice", error=DefinitionError, message="area is already assigned at"
    ) as caught:

        class AtTheCallAndAfter(Space):
            room = Room(area=3)
            room.area = 4

    assert "assigned at test_formal_bindings.py:" in str(caught.error)


def test_a_node_below_a_declaration_is_not_assigned_through_a_path() -> None:
    with refused(
        "assignment-not-own-formal",
        error=DefinitionError,
        message="a node below it is supplied where it is declared",
    ):

        class Through(Space):
            wing = Wing()
            wing.kitchen.area = 16


@pytest.mark.parametrize(
    ("what", "supply"),
    [
        ("Decision", lambda: Room(finish=2)),
        ("Decision", lambda: Room(finish=Decision(values=(1, 2)))),
        ("child node", lambda: Wing(kitchen=Room(area=9))),  # type: ignore[call-arg]
    ],
)
def test_a_decision_or_a_child_node_belongs_to_the_space_class(what: str, supply: object) -> None:
    with refused(
        "member-belongs-to-class",
        error=DefinitionError,
        message=f"is a {what}: it belongs to the Space class",
    ):
        supply()  # type: ignore[operator]


def test_a_decision_over_nodes_belongs_to_the_space_class() -> None:
    class Plant(Space):
        heating: Room | None = Decision({"room": Room}, optional=True)

    with refused("member-belongs-to-class", error=DefinitionError, message=r"Decision over nodes"):
        Plant(heating="room")  # type: ignore[arg-type]
    plant = Plant()
    with refused("member-belongs-to-class", error=DefinitionError, message=r"Decision over nodes"):
        plant.heating = "room"  # type: ignore[assignment]


@pytest.mark.parametrize("member", ["fits", "cost", "costs"])
def test_behaviour_is_not_assignable(member: str) -> None:
    target = Wing() if member == "costs" else Room()
    with refused("member-belongs-to-class", error=DefinitionError):
        setattr(target, member, 1)

    class Budget(Space):
        claims = Members(COST)

        @derived
        def claimed(self) -> int:
            return len(self.claims)

    for name in ("claims", "claimed"):
        with refused("member-belongs-to-class", error=DefinitionError):
            setattr(Budget(), name, 1)


def test_a_subclass_changes_what_belongs_to_the_space_class() -> None:
    class Plain(Room):
        finish: int = Decision(values=(1,))

    class Large(Wing):
        kitchen = Plain(area=19)

    point = design_space(Large())
    assert point.kitchen.area == 19 and point.kitchen.finish == 1
    assert [item.key for item in inspection.decisions(point)] == [
        "kitchen.finish",
        "study.finish",
    ]


def test_a_root_that_supplies_structure_is_refused_with_a_pointer() -> None:
    class Sized(Space):
        size: int = Param()

    with refused(
        "root-structural-formal",
        error=DefinitionError,
        message="write a subclass, or build one with composite",
    ):
        design_space(Sized(size=Decision(values=(1, 2))))
    with refused("root-structural-formal", error=DefinitionError, message=r"\bsize\b"):
        design_space(Sized(size=Room.area))
    # The pointer's way: the structure in a Space class built from data.
    room = Room(area=3)
    sized = composite("Measured", {"room": room, "sized": Sized(size=room.area)})
    assert getattr(design_space(sized()), "sized").size == 3
