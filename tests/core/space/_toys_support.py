# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The space tests' two toys, shared by their own tests and the collapse tests.

The house (``test_house``): rooms, a structural heating choice, a relation node
and a budget over every member's cost. The company (``test_references``):
departments sharing one budget through a reference input, which sees them as
its users. Nothing here is about hardware.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    LocatedParam,
    Members,
    Param,
    Rejected,
    Space,
    Users,
    ViewKey,
    constraint,
    reject,
    view,
)

COST = ViewKey("cost", int)


SPEND = ViewKey("spend", int)


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


class Match(Space):
    """A relation node: two located values must agree."""

    a: LocatedParam[int] = LocatedParam()
    b: LocatedParam[int] = LocatedParam()

    @constraint
    def same(self) -> bool | Rejected:
        a, b = self.a, self.b
        if a.value != b.value:
            return reject(
                "mismatch", f"{a.node}.{a.member}={a.value}, {b.node}.{b.member}={b.value}"
            )
        return True

    @view(requires=(same,))
    def agreed(self) -> int:
        return self.a.value


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
    matched = Match(a=kitchen.finish, b=dining.finish)
    costs = Members(COST)

    @constraint
    def within_budget(self) -> bool | Rejected:
        spent = {member.node: member.value for member in self.costs}
        if sum(spent.values()) > self.budget:
            return reject("over-budget", f"{sum(spent.values())} exceeds {self.budget}")
        return True

    @view(requires=(costs, within_budget, matched.agreed))
    def total(self) -> int:
        return sum(member.value for member in self.costs)


class Budget(Space):
    """Knows nothing about departments: it sees whoever references it."""

    limit: int = Param()
    rate: int = Param()
    claims = Users(SPEND)

    @constraint
    def covered(self) -> bool | Rejected:
        total = sum(claim.value for claim in self.claims)
        if total > self.limit:
            named = ", ".join(f"{c.node}.{c.member}={c.value}" for c in self.claims)
            return reject("over-budget", f"{named} exceed {self.limit}")
        return True

    @view(requires=(claims, covered))
    def remaining(self) -> int:
        return self.limit - sum(claim.value for claim in self.claims)


class Department(Space):
    budget: Budget = Param()
    staff: int = Decision(values=(1, 2, 3))

    @view
    def spend(self) -> int:
        # The referenced node's configuration: a real value, read in a method.
        return self.staff * self.budget.rate

    exports = {SPEND: spend}


class Company(Space):
    limit: int = Param()
    open_lab: bool = Decision(values=(False, True))
    shared = Budget(limit=limit, rate=10)
    sales = Department(budget=shared)
    research = Department()
    research.budget = shared  # a reference input may be assigned like any formal
    lab = Department(budget=shared, when=open_lab)
