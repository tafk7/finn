# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A reference is typed as its value, so its value-like misuse must fail at runtime.

Inside a class body ``kitchen.area`` is statically an ``int`` but is really a
declaration reference. Each value-like use raises ``ReferenceUseError`` naming
the reference, where it was written, and what to do instead. Integer arithmetic on a
reference and an attribute read of a referenced value are not value-like uses a derived
value cannot state: each is a ``DefinitionError`` (``reference-expression``,
``reference-projection``) that says to declare a derived value, and a reference to a
node's member or a child's path is read as before.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    ReferenceUseError,
    Select,
    Space,
    derived,
    design_space,
    view,
)


class Room(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2))

    @view
    def cost(self) -> int:
        return self.area * self.finish


class Boiler(Space):
    kw: int = Param()


def kitchen_area() -> object:
    return Room(area=12).area


MISUSES: dict[str, tuple[Callable[[object], object], str]] = {
    "truthiness": (lambda ref: bool(ref), "a truth value"),
    "if": (lambda ref: 1 if ref else 0, "a truth value"),
    "equality": (lambda ref: ref == 12, "equality comparison with a value"),
    "inequality": (lambda ref: ref != 12, "equality comparison with a value"),
    "less-than": (lambda ref: ref < 3, "ordering comparison"),  # type: ignore[operator]
    "greater-equal": (lambda ref: ref >= 3, "ordering comparison"),  # type: ignore[operator]
    "reflected ordering": (lambda ref: 3 < ref, "ordering comparison"),  # type: ignore[operator]
    "len": (lambda ref: len(ref), "a length"),  # type: ignore[arg-type]
    "iteration": (lambda ref: list(ref), "iteration"),  # type: ignore[call-overload]
    "membership": (lambda ref: 3 in ref, "membership testing"),  # type: ignore[operator]
    "indexing": (lambda ref: ref[0], "indexing"),  # type: ignore[index]
    "str": (lambda ref: str(ref), "str()"),
    "format": (lambda ref: format(ref, "d"), "formatting"),
    "f-string": (lambda ref: f"{ref}", "formatting"),
    "int": (lambda ref: int(ref), "int()"),  # type: ignore[call-overload]
    "float": (lambda ref: float(ref), "float()"),  # type: ignore[arg-type]
    "round": (lambda ref: round(ref), "round()"),  # type: ignore[call-overload]
    "floor": (lambda ref: math.floor(ref), "floor()"),  # type: ignore[call-overload]
    "abs": (lambda ref: abs(ref), "abs()"),  # type: ignore[arg-type]
    "range": (lambda ref: range(ref), "an index"),  # type: ignore[call-overload]
    "max": (lambda ref: max(ref, 3), "ordering comparison"),  # type: ignore[call-overload]
    "min": (lambda ref: min(ref, 3), "ordering comparison"),  # type: ignore[call-overload]
    "sorted": (lambda ref: sorted([ref, 3]), "ordering comparison"),  # type: ignore[type-var]
    "call": (lambda ref: ref(), "calling"),  # type: ignore[operator]
}


@pytest.mark.parametrize("name", sorted(MISUSES))
def test_each_value_like_use_of_a_reference_fails_loudly_and_specifically(name: str) -> None:
    misuse, what = MISUSES[name]
    with pytest.raises(ReferenceUseError) as caught:
        misuse(kitchen_area())
    message = str(caught.value)
    assert f"is a declaration reference, not a value: {what}" in message
    assert "Room node" in message and ".area" in message
    assert "test_reference_misuse.py:" in message  # where the reference was written
    assert "Compute with it in a @derived or @view method instead" in message


def kitchen_cost() -> object:
    return Room(area=12).cost


@pytest.mark.parametrize("name", sorted(MISUSES))
def test_a_view_reference_is_refused_like_any_other_reference(name: str) -> None:
    # A view reads as its accepted value, so a reference to it is typed as that
    # value and every value-like use is refused exactly as for a formal.
    misuse, what = MISUSES[name]
    with pytest.raises(ReferenceUseError) as caught:
        misuse(kitchen_cost())
    message = str(caught.value)
    assert f"is a declaration reference, not a value: {what}" in message
    assert "Room node" in message and ".cost" in message
    assert "test_reference_misuse.py:" in message


def test_a_view_reference_has_no_assessment_accessors() -> None:
    # Assessment is point.inspect(Room.cost); the reference itself is only a value.
    cost = kitchen_cost()
    for name in ("inspect", "query", "get"):
        with pytest.raises(AttributeError):
            getattr(cost, name)


def test_misuse_in_a_class_body_names_the_node_by_its_declaration() -> None:
    # Inside the body the node is not named yet: its declaration line identifies it.
    with pytest.raises(ReferenceUseError, match=r"<Room node \(declared at .*\)>\.area .* truth"):

        class House(Space):
            kitchen = Room(area=12)
            if kitchen.area:
                bigger = Room(area=20)

    with pytest.raises(ReferenceUseError, match=r"selector over cases \['boiler'\].*\.kw"):

        class Heated(Space):
            heating: Boiler = Select(boiler=Boiler(kw=3))
            size = max(heating.kw, 10)


def test_references_are_keys() -> None:
    class House(Space):
        kitchen = Room(area=12)
        dining = Room(area=16)

    # Stable hashing and equality by (placement path, member): usable as keys.
    assert House.kitchen.finish == House.kitchen.finish
    assert House.kitchen.finish != House.dining.finish
    assert hash(House.kitchen.finish) == hash(House.kitchen.finish)
    assert len({House.kitchen.finish: 1, House.dining.finish: 2}) == 2
    assert {House.kitchen.finish: 2}[House.kitchen.finish] == 2
    # Equality is not overloaded into an expression.
    assert isinstance(House.kitchen.finish == House.kitchen.finish, bool)


OPERATORS: dict[str, tuple[Callable[[Any], object], str]] = {
    "add": (lambda ref: ref + 1, "+"),
    "reflected add": (lambda ref: 1 + ref, "+"),
    "sub": (lambda ref: ref - 1, "-"),
    "reflected sub": (lambda ref: 1 - ref, "-"),
    "mul": (lambda ref: ref * 2, "*"),
    "reflected mul": (lambda ref: 2 * ref, "*"),
    "floordiv": (lambda ref: ref // 2, "//"),
    "reflected floordiv": (lambda ref: 2 // ref, "//"),
    "mod": (lambda ref: ref % 2, "%"),
    "reflected mod": (lambda ref: 2 % ref, "%"),
    "negation": (lambda ref: -ref, "unary -"),
    "two references": (lambda ref: ref + ref, "+"),
}


def one(function: Callable[..., object]) -> Callable[[], object]:
    return lambda: derived(function)


def value_two() -> int:
    return 2


REFERENCES: dict[str, tuple[Callable[[], object], str]] = {
    "a node's member": (kitchen_area, "Room node"),
    "a view of a node": (kitchen_cost, "Room node"),
    "a formal": (lambda: Param(), "declared at "),
    "a derived value": (one(value_two), "declared at "),
    "a Derived": (lambda: derived(value_two), "declared at "),
    "a decision": (lambda: Decision(values=(1, 2)), "declared at "),
}


@pytest.mark.parametrize("reference", sorted(REFERENCES))
@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_arithmetic_on_a_reference_is_refused_by_name(operator: str, reference: str) -> None:
    compute, symbol = OPERATORS[operator]
    make, named = REFERENCES[reference]
    message = rf"arithmetic \({re.escape(symbol)}\) on a reference.*declare a derived value"
    with refused("reference-expression", error=DefinitionError, message=message) as caught:
        compute(make())
    assert named in caught.finding.message


def test_arithmetic_in_a_class_body_names_the_reference_as_written() -> None:
    with refused(
        "reference-expression", error=DefinitionError, message=r"^width \(declared at "
    ) as caught:

        class Wing(Space):
            width: int = Param()
            doubled = width * 2

    assert "derived value" in caught.finding.message
    assert str(dict(caught.finding.details)["declaration"]).startswith("width (declared at ")

    with refused("reference-expression", error=DefinitionError, message=r"^size \(declared at "):

        class Hall(Space):
            @derived
            def size(self) -> int:
                return 3

            wide = 1 + size


def test_a_declared_derived_value_computes_what_arithmetic_on_a_reference_did() -> None:
    class House(Space):
        kitchen = Room(area=12)

        @derived(area=kitchen.area)
        def doubled(*, area: int) -> int:
            return area * 2

    assert design_space(House()).doubled == 24


@dataclass(frozen=True)
class Rate:
    per_head: int
    heads: int

    @property
    def total(self) -> int:
        return self.per_head * self.heads


class Priced(Space):
    rate: Rate = Param()

    @derived
    def quoted(self) -> Rate:
        return self.rate


def test_an_attribute_read_on_a_formal_is_refused_by_name() -> None:
    message = r"^rate \(declared at .*\): reading \.total .*declare a derived value"
    with refused("reference-projection", error=DefinitionError, message=message):

        class Office(Space):
            rate: Rate = Param()
            total = rate.total


def test_an_attribute_read_on_a_derived_value_is_refused_by_name() -> None:
    message = r"^quoted \(declared at .*\): reading \.per_head .*declare a derived value"
    with refused("reference-projection", error=DefinitionError, message=message):

        class Office(Space):
            @derived
            def quoted(self) -> Rate:
                return Rate(1, 2)

            per_head = quoted.per_head


def test_an_attribute_read_on_a_nodes_member_is_refused_by_name() -> None:
    with refused("reference-projection", error=DefinitionError, message="Priced node.*rate") as c:

        class Office(Space):
            priced = Priced(rate=Rate(1, 2))
            total = priced.rate.total

    assert "derived value" in c.finding.message
    with refused("reference-projection", error=DefinitionError, message="Priced node.*quoted"):

        class Annex(Space):
            priced = Priced(rate=Rate(1, 2))
            heads = priced.quoted.heads


def test_an_attribute_no_annotation_names_is_still_an_attribute_error() -> None:
    class Office(Space):
        rate: Rate = Param()

        @derived
        def quoted(self) -> Rate:
            return Rate(1, 2)

        priced = Priced(rate=Rate(1, 2))
        missing = [
            hasattr(rate, "missing"),
            hasattr(quoted, "missing"),
            hasattr(priced.rate, "missing"),
            hasattr(rate, "__wrapped__"),
        ]

    assert Office.missing == [False] * 4  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        getattr(Office.rate, "missing")


def test_what_the_engine_reads_off_a_reference_is_not_refused() -> None:
    class Office(Space):
        rate: Rate = Param()

        @derived
        def quoted(self) -> Rate:
            return Rate(1, 2)

        priced = Priced(rate=Rate(1, 2))

    for reference in (Office.rate, Office.priced.rate):
        assert reference.semantics is not None  # type: ignore[attr-defined]
    assert Office.quoted.function is not None  # type: ignore[attr-defined]
    assert Office.rate.name == "rate" and Office.rate.owner is Office  # type: ignore[attr-defined]


def test_a_path_through_children_and_selected_members_is_read_as_before() -> None:
    class Leaf(Space):
        kw: int = Param()

    class Mid(Space):
        leaf = Leaf(kw=4)

    class Plant(Space):
        mid = Mid()
        heating: Leaf = Select(boiler=Leaf(kw=3))
        feed: int = Param()
        sink = Leaf(kw=mid.leaf.kw)  # a path through two children
        burner = Leaf(kw=heating.kw)  # a selected member
        into = Leaf(kw=feed)

    point = design_space(Plant(feed=7))
    assert (point.sink.kw, point.into.kw) == (4, 7)
    assert Plant.mid.leaf.kw == Plant.mid.leaf.kw


def test_a_view_reference_is_not_callable_in_a_class_body() -> None:
    with pytest.raises(ReferenceUseError, match="calling"):

        class House(Space):
            kitchen = Room(area=12)
            total = kitchen.cost() + 1  # type: ignore[operator]


def test_a_declaration_is_not_a_configuration() -> None:
    node = Room(area=3)
    with pytest.raises(Exception, match="node declaration, not a configuration"):
        node.query(Room.area)
    assert repr(node).startswith("<Room node")
    assert "test_reference_misuse.py:" in repr(node)
