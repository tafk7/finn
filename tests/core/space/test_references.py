# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Reference inputs and ``Users``: several nodes share one node, which sees them.

``budget: Budget = Param()`` is a reference input. Supplied with a
node placed beside it, it references that node: several departments share one
budget, and inside their methods ``self.budget`` is the budget's configuration.
Supplied with a fresh node, it places that node there. ``Users(SPEND)`` in the
budget is the mirror of ``Members``: every present node whose input references
the budget, located by its name and input. Nothing here is about hardware.
The company is declared in ``_toys_support``, which the collapse tests share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pytest
from core.space._toys_support import SPEND, Budget, Company, Department

from finn.core.space import (
    Available,
    Const,
    Decision,
    DefinitionError,
    Inapplicable,
    Located,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueSemantics,
    ValueUnavailableError,
    composite,
    default_semantics,
    derived,
    design_space,
    inspection,
    view,
)
from finn.core.space.errors import RequestError


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.code for finding in result.findings}


def staffed(point: Company, **staff: int) -> Company:
    return point.with_choices({getattr(Company, name).staff: n for name, n in staff.items()})


def test_several_nodes_reference_one_node_which_sees_them_as_users() -> None:
    point = staffed(design_space(Company(limit=100)).with_choices(open_lab=True), sales=2, lab=1)
    point = staffed(point, research=3)
    assert point.shared.claims == (
        Located("sales", "budget", 20),
        Located("research", "budget", 30),
        Located("lab", "budget", 10),
    )
    assert point.shared.remaining == 40
    # Inside a method, and from the driver, the input reads as the referenced node.
    assert isinstance(point.sales.budget, Budget) and point.sales.budget.rate == 10
    # Place-once holds: one budget scope, referenced three times; none is copied.
    scopes = [scope.name for scope in inspection.model(point).linked.scopes]
    assert scopes == ["", "shared", "sales", "research", "lab"]


def test_a_guarded_user_drops_out_of_users() -> None:
    closed = staffed(
        design_space(Company(limit=100)).with_choices(open_lab=False), sales=1, research=1
    )
    assert [claim.node for claim in closed.shared.claims] == ["sales", "research"]
    # As an obligation each user counts separately; the absent one is inapplicable.
    results = closed.shared.inspect(Budget.remaining).constraints.results
    assert isinstance(results["lab.spend"], Inapplicable)
    assert isinstance(results["sales.spend"], Available)
    assert closed.shared.remaining == 80


def test_the_shared_node_refuses_with_located_names() -> None:
    point = staffed(
        design_space(Company(limit=40)).with_choices(open_lab=False), sales=3, research=2
    )
    refused = point.shared.query(Budget.remaining)
    assert codes(refused) == {"over-budget"}
    assert isinstance(refused, Rejected)
    assert refused.findings[0].owner == "shared.covered"
    assert refused.findings[0].message == "sales.budget=30, research.budget=20 exceed 40"
    # A user still undecided leaves the relation unresolved, not refused.
    pending = design_space(Company(limit=40)).with_choices(open_lab=False)
    assert isinstance(pending.shared.query(Budget.remaining), Unresolved)


class Outsourced(Space):
    budget: Budget = Param()
    fee = Const(25)

    @view
    def spend(self) -> int:
        return self.fee

    exports = {SPEND: spend}


class Firm(Space):
    shared = Budget(limit=100, rate=10)
    sales = Department(budget=shared)
    inhouse = Department(budget=shared)  # a candidate handle that references the budget
    support: Department | Outsourced | None = Decision(
        {"inhouse": inhouse, "outsourced": Outsourced(budget=shared)}, optional=True
    )


def test_a_choice_candidate_references_a_shared_node() -> None:
    base = design_space(Firm()).with_choices({Firm.sales.staff: 1})
    outsourced = base.with_choices(support="outsourced")
    assert outsourced.shared.claims == (
        Located("sales", "budget", 10),
        Located("support.outsourced", "budget", 25),
    )
    inhouse = base.with_choices({Firm.support: "inhouse", Firm.inhouse.staff: 2})
    assert [(c.node, c.value) for c in inhouse.shared.claims] == [
        ("sales", 10),
        ("support.inhouse", 20),
    ]
    # An inactive candidate is not a user; None places nothing.
    assert [c.node for c in base.with_choices(support="none").shared.claims] == ["sales"]
    # Until the choice is made, the users are unresolved: a candidate may yet be one.
    assert isinstance(base.shared.query(Budget.claims), Unresolved)


def test_referencing_an_absent_node_reads_inapplicable() -> None:
    class Maybe(Space):
        funded: bool = Decision(values=(False, True))
        budget = Budget(limit=10, rate=2, when=funded)
        team = Department(budget=budget)

    point = design_space(Maybe()).with_choices({Maybe.team.staff: 2})
    assert point.with_choices(funded=True).team.spend == 4
    unfunded = point.with_choices(funded=False)
    assert isinstance(unfunded.team.query(Department.spend), Inapplicable)
    assert isinstance(unfunded.team.query(Department.budget), Inapplicable)


def test_a_fresh_node_supplied_to_a_reference_input_is_placed_there() -> None:
    class Solo(Space):
        team = Department(budget=Budget(limit=5, rate=1))

    point = design_space(Solo()).with_choices({Solo.team.staff: 3})
    # Placed at the input: its keys live below the node that placed it.
    assert point.team.budget.remaining == 2
    assert point.team.budget.claims == (Located(None, "budget", 3),)


def test_a_composite_forwards_a_reference_input_to_its_children() -> None:
    class Division(Space):
        budget: Budget = Param()
        team = Department(budget=budget)  # forwards the division's own input

        @view
        def spend(self) -> int:
            return self.team.spend

        exports = {SPEND: spend}

    class Group(Space):
        shared = Budget(limit=100, rate=5)
        division = Division(budget=shared)

    point = design_space(Group()).with_choices({Group.division.team.staff: 2})
    assert point.division.team.budget.rate == 5
    # A user is the node whose input names the budget: the division, not its team.
    assert point.shared.claims == (Located("division", "budget", 10),)


def test_references_resolve_where_they_are_written() -> None:
    elsewhere = Budget(limit=1, rate=1)

    class Other(Space):
        held = elsewhere

    class Stray(Space):
        team = Department(budget=elsewhere)

    with pytest.raises(DefinitionError, match="is not placed in <root>"):
        design_space(Stray())
    assert isinstance(Other.held, Budget)

    class Holder(Space):
        inner = Budget(limit=1, rate=1)

    with pytest.raises(DefinitionError, match=r"\.inner reaches into another node"):

        class Deep(Space):
            holder = Holder()
            team = Department(budget=holder.inner)  # typed Budget: refused at the call

    loose = Budget(limit=1, rate=1)
    first, second = Department(budget=loose), Department(budget=loose)
    with pytest.raises(DefinitionError, match="placed by none"):
        design_space(composite("Loose", {"first": first, "second": second})())
    with pytest.raises(DefinitionError, match="expected a Budget node"):
        Department(budget=Department())  # type: ignore[arg-type]


def test_an_unsupplied_reference_input_is_reported_when_prepared() -> None:
    class Orphan(Space):
        team = Department()

    with pytest.raises(DefinitionError, match=r"team\.budget is not supplied"):
        design_space(Orphan())


def test_a_reference_input_queries_as_the_referenced_node_and_has_a_typed_presence() -> None:
    point = staffed(design_space(Company(limit=100)), sales=1)
    # The query of a reference input answers the referenced node's configuration.
    answer = point.sales.query(Department.budget)
    assert isinstance(answer, Available) and isinstance(answer.value, Budget)
    assert answer.value.limit == 100
    assert point.sales.present(Department.budget) is True
    # The node it references is absent: the input is inapplicable, and not present.
    closed = point.with_choices(open_lab=False)
    assert isinstance(closed.query(Company.lab), Inapplicable)
    assert closed.present(Company.lab) is False
    assert closed.lab.present(Department.budget) is True  # the budget itself is present
    # Undecided presence reads like any value: it raises.
    with pytest.raises(ValueUnavailableError):
        point.present(Company.lab)
    assert isinstance(point.query(Company.lab), Unresolved)
    opened = point.with_choices(open_lab=True)
    lab = opened.query(Company.lab)
    assert isinstance(lab, Available) and isinstance(lab.value, Department)
    assert opened.present(Company.lab) is True

    class Optional(Space):
        budget: Budget = Param(required=False)

        @view
        def funded(self) -> bool:
            return self.present(Optional.budget)  # presence read inside a method

    unsupplied = design_space(Optional())
    assert isinstance(unsupplied.query(Optional.budget), Unresolved)
    assert unsupplied.present(Optional.budget) is False
    assert unsupplied.funded is False


def test_a_value_input_is_present_when_it_is_supplied() -> None:
    class Rated(Space):
        rate: int = Param(required=False)

        @derived
        def charged(self) -> int:
            # Presence read inside a method: an omitted rate does not halt it.
            return self.rate if self.present(Rated.rate) else 0

    class Office(Space):
        rate: int = Param(required=False)
        forwarded = Rated(rate=rate)  # bound to the enclosing formal
        literal = Rated(rate=3)
        unbound = Rated()  # an optional formal nobody binds
        level: int = Decision(values=(1, 2))
        chosen = Rated(rate=level)

    omitted = design_space(Office())
    assert omitted.present(Office.rate) is False
    assert omitted.forwarded.present(Rated.rate) is False
    assert omitted.forwarded.charged == 0
    assert omitted.literal.present(Rated.rate) is True and omitted.literal.charged == 3
    assert omitted.unbound.present(Rated.rate) is False and omitted.unbound.charged == 0
    supplied = design_space(Office(rate=5))
    assert supplied.present(Office.rate) is True
    assert supplied.forwarded.present(Rated.rate) is True and supplied.forwarded.charged == 5
    assert design_space(Rated(rate=2)).charged == 2
    # Presence before value: a source that applies is present while its value is
    # undecided, and the undecided value halts the method that reads it.
    assert omitted.chosen.present(Rated.rate) is True
    assert codes(omitted.chosen.query(Rated.charged)) == {"decision-unassigned"}
    assert omitted.with_choices(level=2).chosen.charged == 2
    # Anything else is neither a node nor a value input.
    with pytest.raises(RequestError, match="or a value input"):
        omitted.literal.present(Rated.charged)


def test_a_read_through_a_reference_input_is_typed_in_the_class_body() -> None:
    @dataclass(frozen=True)
    class Rate:
        per_head: int
        heads: int

        @property
        def total(self) -> int:
            return self.per_head * self.heads

    class Tariff(Space):
        rate: Rate = Param()

    class Ledger(Space):
        amount: int = Param()

    class Office(Space):
        tariff: Tariff = Param()
        ledger = Ledger(amount=tariff.rate.total)  # an attribute of the referenced value

    class Firm(Space):
        tariff = Tariff(rate=Rate(10, 3))
        office = Office(tariff=tariff)

    assert design_space(Firm()).office.ledger.amount == 30
    with pytest.raises(AttributeError):
        getattr(Office.tariff.rate, "missing")


PAIR: ValueSemantics[tuple[int, int]] = ValueSemantics(
    tuple[int, int],
    "pair",
    lambda value: type(value) is tuple and len(value) == 2,
    lambda left, right: left == right,
    lambda value: value,
)


@dataclass(frozen=True)
class Split:
    head: int
    tail: int
    both: Annotated[tuple[int, int], PAIR]


def test_an_attribute_of_a_derived_value_is_read_in_the_class_body() -> None:
    class Ledger(Space):
        amount: int = Param()

    class Office(Space):
        total: int = Param()

        @derived(semantics=default_semantics(Split))
        def split(self) -> Split:
            head = self.total // 3
            return Split(head, self.total - head, (head, self.total - head))

        ledger = Ledger(amount=split.tail)  # an attribute of this body's own derived value

    class Paired(Space):
        pair: tuple[int, int] = Param(semantics=PAIR)

    class Branch(Office):
        second = Ledger(amount=Office.split.head)  # inherited, read the same way
        paired = Paired(pair=Office.split.both)  # semantics named by the annotation

    point = design_space(Branch(total=30))
    assert point.ledger.amount == 20 and point.second.amount == 10
    assert point.paired.pair == (10, 20)
    with pytest.raises(AttributeError):
        getattr(Office.split, "missing")
