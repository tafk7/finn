# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Per-input exports: a node presents one view through each input that references.

``exports = {SPEND: {budget: budget_spend, reserve: reserve_spend}}`` maps a key
to one view per reference input. ``Users(SPEND)`` on a budget then sees only the
view presented through the input that references it, so a refusal of one view
reaches only its own budget. ``Members(SPEND)`` sees every entry, located by the
input's name. Nothing here is about hardware.
"""

from __future__ import annotations

import pytest

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Located,
    Members,
    Param,
    Rejected,
    Space,
    Users,
    ViewKey,
    constraint,
    default_semantics,
    design_space,
    reject,
    view,
)
from finn.core.space.collection import collect_space

SPEND = ViewKey("spend", int)


class Budget(Space):
    limit: int = Param()
    claims = Users(SPEND)

    @constraint
    def covered(self) -> bool | Rejected:
        if sum(claim.value for claim in self.claims) > self.limit:
            return reject("over-budget", f"claims exceed {self.limit}")
        return True

    @view(requires=(claims, covered))
    def remaining(self) -> int:
        return self.limit - sum(claim.value for claim in self.claims)


class Department(Space):
    """Draws on two budgets; each sees only what is drawn from it."""

    budget: Budget = Param()
    reserve: Budget = Param()
    staff: int = Decision(values=(1, 2, 3))

    @view
    def budget_spend(self) -> int:
        return 10 * self.staff

    @view(semantics=default_semantics(int))
    def reserve_spend(self) -> int | Rejected:
        if self.staff == 3:
            return reject("reserve-closed", "the reserve funds at most two staff")
        return self.staff

    exports = {SPEND: {budget: budget_spend, reserve: reserve_spend}}


class Company(Space):
    operating = Budget(limit=100)
    rainy_day = Budget(limit=5)
    sales = Department(budget=operating, reserve=rainy_day)
    spends = Members(SPEND)


def staffed(staff: int) -> Company:
    return design_space(Company()).with_choices({Company.sales.staff: staff})


def test_each_referenced_node_sees_only_the_view_presented_through_its_input() -> None:
    point = staffed(2)
    assert point.operating.claims == (Located("sales", "budget", 20),)
    assert point.rainy_day.claims == (Located("sales", "reserve", 2),)
    assert point.operating.remaining == 80
    assert point.rainy_day.remaining == 3


def test_a_refused_view_reaches_only_its_own_input() -> None:
    point = staffed(3)
    refused = point.rainy_day.query(Budget.remaining)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"reserve-closed"}
    # The operating budget is untouched by the reserve's refusal.
    assert point.operating.query(Budget.remaining) == Available(70)


def test_members_lists_each_input_entry_located_by_the_input() -> None:
    assert staffed(1).spends == (
        Located("sales", "budget", 10),
        Located("sales", "reserve", 1),
    )


def test_a_per_input_export_maps_reference_inputs_to_views() -> None:
    class NotAnInput(Space):
        budget: Budget = Param()
        amount: int = Param()

        @view
        def spend(self) -> int:
            return self.amount

        exports = {SPEND: {amount: spend}}

    class NotAView(Space):
        budget: Budget = Param()
        amount: int = Param()

        exports = {SPEND: {budget: amount}}  # type: ignore[dict-item]

    with pytest.raises(DefinitionError, match="amount is not a reference input"):
        collect_space(NotAnInput)
    with pytest.raises(DefinitionError, match="export spend for budget has the wrong kind"):
        collect_space(NotAView)
