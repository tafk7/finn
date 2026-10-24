# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Value inputs: an optional one left out, and the attributes of a value read in a class body.

A formal takes a value, always known: an optional one left out is ``None``, which its
readers test. An attribute of a formal's value, or of a derived value of the same body, is
read in the class body as a typed projection. Nothing here is about hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from support.refusals import refused

from core.space._results_support import codes
from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Space,
    derived,
    design_space,
)
from finn.core.space.errors import RequestError


def test_an_optional_value_input_left_out_is_none() -> None:
    class Rated(Space):
        rate: int | None = Param(default=None)

        @derived
        def charged(self) -> int:
            # An optional input is known: left out, it is None, and a method tests it.
            return 0 if self.rate is None else self.rate

    class Office(Space):
        rate: int | None = Param(default=None)
        forwarded = Rated(rate=rate)  # bound to the enclosing formal
        literal = Rated(rate=3)
        unbound = Rated()  # an optional formal nobody binds: its default
        level: int = Decision(values=(1, 2))
        chosen = Rated(rate=level)

    omitted = design_space(Office())
    assert omitted.rate is None
    assert omitted.forwarded.rate is None and omitted.forwarded.charged == 0
    assert omitted.literal.charged == 3
    assert omitted.unbound.rate is None and omitted.unbound.charged == 0
    supplied = design_space(Office(rate=5))
    assert supplied.forwarded.charged == 5
    assert design_space(Rated(rate=2)).charged == 2
    # A source that waits is read as its value: the undecided value halts the method.
    assert codes(omitted.chosen.query(Rated.charged)) == {"decision-unassigned"}
    assert omitted.with_choices(level=2).chosen.charged == 2
    # present() takes a node: a value input is no node, and anything else neither.
    with pytest.raises(RequestError, match="is not None"):
        omitted.literal.present(Rated.rate)
    with pytest.raises(RequestError, match="takes a node"):
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

    with refused("formal-takes-node", error=DefinitionError, message=r"Budget node"):

        class Department(Space):
            budget: Budget = Param()


@dataclass(frozen=True)
class Split:
    head: int
    tail: int
    both: tuple[int, int]


def test_an_attribute_of_a_derived_value_is_read_in_the_class_body() -> None:
    class Ledger(Space):
        amount: int = Param()

    class Office(Space):
        total: int = Param()

        @derived
        def split(self) -> Split:
            head = self.total // 3
            return Split(head, self.total - head, (head, self.total - head))

        ledger = Ledger(amount=split.tail)  # an attribute of this body's own derived value

    class Paired(Space):
        pair: tuple[int, int] = Param()

    class Branch(Office):
        second = Ledger(amount=Office.split.head)  # inherited, read the same way
        paired = Paired(pair=Office.split.both)  # typed by the field's annotation

    point = design_space(Branch(total=30))
    assert point.ledger.amount == 20 and point.second.amount == 10
    assert point.paired.pair == (10, 20)
    with pytest.raises(AttributeError):
        getattr(Office.split, "missing")
