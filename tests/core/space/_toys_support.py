# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The space tests' house toy, shared by its own tests and the collapse tests.

The house (``test_house``): rooms, a structural heating choice, a relation of two rooms
and a budget over every member's cost. Nothing here is about hardware.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Members,
    Param,
    Rejected,
    Space,
    ViewKey,
    constraint,
    reject,
    view,
)

COST = ViewKey("cost", int)


class Room(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @view
    def cost(self) -> int:
        return self.area * self.finish

    exports = {COST: cost}


class Boiler(Space):
    kw: int = Param()

    @view
    def cost(self) -> int:
        return 30 + self.kw

    exports = {COST: cost}


class HeatPump(Space):
    kw: int = Param()
    cop: int = Decision(values=(3, 4))

    @view
    def cost(self) -> int:
        return 10 * self.cop + self.kw

    exports = {COST: cost}


class Thermostat(Space):
    kw: int = Param()

    @view
    def cost(self) -> int:
        return 2 if self.kw < 10 else 5

    exports = {COST: cost}


class House(Space):
    budget: int = Param()
    want_garage: bool = Decision(values=(False, True))
    hall = Room()  # its area is supplied by the assignment below
    kitchen = Room(area=12)
    dining = Room(area=16)
    garage = Room(area=20, when=want_garage)
    # A class attribute may name a candidate: a typed handle, not a placement.
    heat_pump = HeatPump(kw=8)
    heating: Boiler | HeatPump = Decision({"boiler": Boiler(kw=24), "heat_pump": heat_pump})
    thermostat = Thermostat(kw=heating.kw)
    hall.area = kitchen.area  # an edge declared after its nodes
    costs = Members(COST)

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

    @view(requires=(costs, within_budget, matched))
    def total(self) -> int:
        return sum(member.value for member in self.costs)
