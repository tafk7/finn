# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``Members(Base.member)``: the present children deriving from a base, each read at one
member of it.

Its entries are the placed children (in member order, a choice's cases in case order)
whose class is ``Base`` or derives from it, each at the member's name, so a subclass's
override is the one read; an inapplicable entry is left out. The member is a view or a
derived value of ``Base``, spelled ``Base.member`` outside it and ``Members(member)`` in
its own body.
"""

from __future__ import annotations

from typing import Any

from support.refusals import refused

from finn.core.space import (
    DefinitionError,
    Located,
    Members,
    Param,
    Select,
    Space,
    constraint,
    derived,
    design_space,
    view,
)
from finn.core.space.declarations import Constraint


class Priced(Space):
    """A base: a view ``price``, and a derived value ``weight``."""

    size: int = Param(default=2)

    @view
    def price(self) -> int:
        return 10 * self.size

    @derived
    def weight(self) -> int:
        return self.size

    # In its own body, once the members are declared.
    prices = Members(price)
    weights = Members(weight)


class Cheap(Priced):
    @view
    def price(self) -> int:  # an override: the one read
        return self.size


class Other(Space):
    """Not a ``Priced``: it declares a ``price`` of its own and contributes nothing."""

    @view
    def price(self) -> int:
        return 999


def test_entries_in_member_order_each_named_by_its_node_and_member() -> None:
    class Shelf(Space):
        first = Priced(size=3)
        second = Priced(size=5)
        prices = Members(Priced.price)
        weights = Members(Priced.weight)

    point = design_space(Shelf())
    assert point.prices == (Located("first", "price", 30), Located("second", "price", 50))
    assert point.weights == (Located("first", "weight", 3), Located("second", "weight", 5))


def test_the_in_body_spelling_and_the_base_spelling_give_the_same_entries() -> None:
    """``Priced`` spells ``prices`` in its own body; a subclass placing children inherits it."""

    class Shelf(Priced):
        first = Priced(size=3)
        second = Cheap(size=5)
        outside = Members(Priced.price)

    point = design_space(Shelf())
    expected = (Located("first", "price", 30), Located("second", "price", 5))
    assert point.prices == expected
    assert point.outside == expected
    assert point.weights == (Located("first", "weight", 3), Located("second", "weight", 5))
    # Placing none, the base's own body has no entry.
    assert design_space(Priced()).prices == ()


def test_a_child_not_deriving_the_base_contributes_nothing() -> None:
    class Shelf(Space):
        first = Priced(size=3)
        stranger = Other()
        second = Priced(size=4)
        prices = Members(Priced.price)

    point = design_space(Shelf())
    assert [entry.node for entry in point.prices] == ["first", "second"]


def test_a_subclass_override_is_the_member_read() -> None:
    class Shelf(Space):
        plain = Priced(size=3)
        cheap = Cheap(size=3)
        prices = Members(Priced.price)

    point = design_space(Shelf())
    assert point.prices == (Located("plain", "price", 30), Located("cheap", "price", 3))


def test_a_choice_places_its_candidates_in_case_order() -> None:
    class Shelf(Space):
        pick: Priced | Cheap = Select(plain=Priced(size=3), cheap=Cheap(size=3))
        prices = Members(Priced.price)

    point = design_space(Shelf())
    assert point.with_choices(pick="cheap").prices == (Located("pick.cheap", "price", 3),)
    assert point.with_choices(pick="plain").prices == (Located("pick.plain", "price", 30),)


def test_an_inapplicable_entry_is_skipped() -> None:
    class Quiet(Priced):
        """Its price is guarded off."""

        loud: bool = Param(default=False)

        @view(when=loud)
        def price(self) -> int:
            return 1

    class Shelf(Space):
        first = Priced(size=3)
        quiet = Quiet()
        loud = Quiet(loud=True)
        prices = Members(Priced.price)

    point = design_space(Shelf())
    assert point.prices == (Located("first", "price", 30), Located("loud", "price", 1))


def test_a_member_of_the_wrong_kind_is_refused() -> None:
    with refused("members-member-invalid", error=DefinitionError):
        Members(Param())
    with refused("members-member-invalid", error=DefinitionError):
        Members(Priced.size)

    @constraint
    def holds(self: Any) -> bool:
        return True

    assert isinstance(holds, Constraint)
    with refused("members-member-invalid", error=DefinitionError):
        Members(holds)
    with refused("members-member-invalid", error=DefinitionError):
        Members(3)


def test_a_member_no_class_declares_is_refused_when_the_class_is_compiled() -> None:
    class Shelf(Space):
        first = Priced()
        prices = Members(view(Priced.price))

    with refused("members-member-invalid", error=DefinitionError):
        design_space(Shelf())
