# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Value inputs: whether one is supplied, and the attributes of a value read in a class body.

A formal takes a value. ``present(formal)`` reads whether it is supplied without
evaluating it; an attribute of a formal's value, or of a derived value of the same
body, is read in the class body as a typed projection. Nothing here is about hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pytest
from support.refusals import refused

from core.space._results_support import codes
from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Space,
    ValueSemantics,
    default_semantics,
    derived,
    design_space,
)
from finn.core.space.errors import RequestError


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


def test_an_attribute_of_a_value_input_is_read_in_the_class_body() -> None:
    @dataclass(frozen=True)
    class Rate:
        per_head: int
        heads: int

        @property
        def total(self) -> int:
            return self.per_head * self.heads

    class Ledger(Space):
        amount: int = Param()

    class Office(Space):
        rate: Rate = Param()
        ledger = Ledger(amount=rate.total)  # an attribute of the formal's value

    assert design_space(Office(rate=Rate(10, 3))).ledger.amount == 30
    with pytest.raises(AttributeError):
        getattr(Office.rate, "missing")


def test_a_formal_takes_a_value_never_a_node() -> None:
    class Budget(Space):
        limit: int = Param()

    with refused("formal-takes-node", error=DefinitionError):

        class Department(Space):
            budget: Budget = Param()


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
