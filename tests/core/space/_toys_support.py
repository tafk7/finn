# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The space tests' house toy, shared by its own tests and the collapse tests.

The house (``test_house``): rooms, a structural heating choice, a relation of two rooms
and a budget over every member's cost. Nothing here is about hardware. Each part states its cost
as a derived value on a common base, ``Costed``, which the house's ``Members`` reads: the budget
is a rule, and a rule reads no view.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Members,
    Param,
    Rejected,
    Select,
    Space,
    constraint,
    derived,
    reject,
    view,
)


class Costed(Space):
    """What the house's budget counts: a part states its ``cost``. The base's own is the one
    a part stating none would answer, a refusal, so a part placed without its cost is
    never taken to cost nothing."""

    @derived
    def cost(self) -> int | Rejected:
        return reject("costed-cost", f"{type(self).__qualname__} states no cost")


class Room(Costed):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @derived
    def cost(self) -> int:
        return self.area * self.finish


class Boiler(Costed):
    kw: int = Param()

    @derived
    def cost(self) -> int:
        return 30 + self.kw


class HeatPump(Costed):
    kw: int = Param()
    cop: int = Decision(values=(3, 4))

    @derived
    def cost(self) -> int:
        return 10 * self.cop + self.kw


class Thermostat(Costed):
    kw: int = Param()

    @derived
    def cost(self) -> int:
        return 2 if self.kw < 10 else 5


class House(Space):
    budget: int = Param()
    want_garage: bool = Decision(values=(False, True))
    hall = Room()  # its area is supplied by the assignment below
    kitchen = Room(area=12)
    dining = Room(area=16)
    garage = Room(area=20, when=want_garage)
    heating: Boiler | HeatPump = Select(boiler=Boiler(kw=24), heat_pump=HeatPump(kw=8))
    thermostat = Thermostat(kw=heating.kw)
    hall.area = kitchen.area  # an edge declared after its nodes
    costs = Members(Costed.cost)

    @constraint
    def matched(self) -> bool | Rejected:
        """A relation of two children, the parent's: their finishes agree."""
        kitchen, dining = self.kitchen.finish, self.dining.finish
        if kitchen != dining:
            return reject("mismatch", f"kitchen.finish={kitchen}, dining.finish={dining}")
        return True

    @constraint
    def within_budget(self) -> bool | Rejected:
        spent = {member.node: member.value for member in self.costs}
        if sum(spent.values()) > self.budget:
            return reject("over-budget", f"{sum(spent.values())} exceeds {self.budget}")
        return True

    @view
    def total(self) -> int:
        """Refused with the house: its relation and its budget are the house's constraints."""
        return sum(member.value for member in self.costs)
