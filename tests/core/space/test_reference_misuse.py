# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A reference is typed as its value, so its value-like misuse must fail at runtime.

Inside a class body ``kitchen.area`` is statically an ``int`` but is really a
declaration reference. Each value-like use raises ``ReferenceUseError`` naming
the reference, where it was written, and what to do instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from finn.core.space import (
    Decision,
    Param,
    ReferenceUseError,
    Space,
    ViewKey,
    design_space,
    inspection,
    view,
)
from finn.core.space.expressions import Expr


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


def test_misuse_in_a_class_body_names_the_node_by_its_declaration() -> None:
    # Inside the body the node is not named yet: its declaration line identifies it.
    with pytest.raises(ReferenceUseError, match=r"<Room node \(declared at .*\)>\.area .* truth"):

        class House(Space):
            kitchen = Room(area=12)
            if kitchen.area:
                bigger = Room(area=20)

    with pytest.raises(ReferenceUseError, match=r"Decision over \['boiler'\].*\.kw"):

        class Heated(Space):
            heating: Boiler = Decision(values={"boiler": Boiler(kw=3)})
            size = max(heating.kw, 10)


def test_references_are_keys_integer_arithmetic_builds_expressions() -> None:
    class House(Space):
        kitchen = Room(area=12)
        dining = Room(area=16)
        doubled = kitchen.area * 2

    # Stable hashing and equality by (placement path, member): usable as keys.
    assert House.kitchen.finish == House.kitchen.finish
    assert House.kitchen.finish != House.dining.finish
    assert hash(House.kitchen.finish) == hash(House.kitchen.finish)
    assert len({House.kitchen.finish: 1, House.dining.finish: 2}) == 2
    assert {House.kitchen.finish: 2}[House.kitchen.finish] == 2
    # Equality is not overloaded into an expression.
    assert isinstance(House.kitchen.finish == House.kitchen.finish, bool)
    assert isinstance(vars(House)["doubled"], Expr)
    assert design_space(House()).doubled == 24
    info = inspection.reference(House.kitchen.finish)
    assert (info.path, info.member) == (("kitchen",), "finish")


def test_a_view_reference_is_not_callable_in_a_class_body() -> None:
    with pytest.raises(ReferenceUseError, match="calling"):

        class House(Space):
            kitchen = Room(area=12)
            total = kitchen.cost() + 1


def test_a_declaration_is_not_a_configuration() -> None:
    node = Room(area=3)
    with pytest.raises(Exception, match="node declaration, not a configuration"):
        node.query(Room.area)
    assert repr(node).startswith("<Room node")
    assert "test_reference_misuse.py:" in repr(node)
    assert ViewKey("cost", int).name == "cost"
