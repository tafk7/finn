# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A ``T | Rejected`` output takes ``T``'s semantics.

A union of one value type and the engine's own result markers (``Rejected``,
``Inapplicable``, ``Unresolved``) infers the value type's default semantics, as
``QueryResult[T]`` names ``T``. An explicit ``semantics=`` still overrides and
is checked against ``T``. A Protocol and a union of two value types (or with
``None``) still need ``semantics=``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pytest

from finn.core.space import (
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueSemantics,
    View,
    constraint,
    default_semantics,
    derived,
    design_space,
    reject,
)
from finn.core.space.collection import collect_space
from finn.core.space.errors import DefinitionError


@dataclass(frozen=True)
class Point:
    x: int


@dataclass(frozen=True)
class Other:
    y: int


class Named(Protocol):
    @property
    def name(self) -> str: ...


def semantics_of(space_type: type[Space], name: str) -> ValueSemantics[object]:
    effective = collect_space(space_type)
    return effective.semantics[effective.members[name]]


class Marked(Space):
    n: int = Param()

    @derived
    def point(self) -> Point | Rejected:
        return reject("odd", f"{self.n} is odd") if self.n % 2 else Point(self.n)

    @derived
    def other(self) -> Other | Inapplicable | Rejected:
        return reject("no-other", "never built")

    @derived
    def pair(self) -> tuple[int, ...] | Unresolved:
        return (self.n, self.n)

    @constraint
    def even(self) -> bool | Rejected:
        return True


def test_a_marked_union_infers_the_value_type() -> None:
    assert semantics_of(Marked, "point").type_token is Point
    assert semantics_of(Marked, "other").type_token is Other
    assert semantics_of(Marked, "pair").type_token is tuple
    assert semantics_of(Marked, "even").type_token is bool
    assert design_space(Marked(n=4)).point == Point(4)
    refused = design_space(Marked(n=3)).query(Marked.point)
    assert isinstance(refused, Rejected) and refused.findings[0].code == "odd"


POINT: ValueSemantics[Point] = ValueSemantics(
    Point, "point", lambda value: type(value) is Point, lambda a, b: a == b, lambda value: value
)


def test_explicit_semantics_still_override() -> None:
    class Explicit(Space):
        @derived(semantics=POINT)
        def point(self) -> Point | Rejected:
            return Point(1)

    assert semantics_of(Explicit, "point") is POINT


def test_explicit_semantics_on_a_marked_union_are_checked_against_the_value_type() -> None:
    class Mismatched(Space):
        # The typed decorator refuses it statically as well.
        @derived(semantics=default_semantics(Other))  # type: ignore[arg-type]
        def point(self) -> Point | Rejected:
            return Point(1)

    with pytest.raises(DefinitionError, match="Point is incompatible with Other"):
        collect_space(Mismatched)


def test_a_protocol_still_needs_semantics() -> None:
    class Implicit(Space):
        @derived
        def named(self) -> Named | Rejected:
            raise AssertionError("never run")

    with pytest.raises(DefinitionError, match="Protocol outputs require explicit semantics="):
        collect_space(Implicit)


def test_a_union_of_values_still_needs_semantics() -> None:
    class Optional(Space):
        @derived
        def point(self) -> Point | None:
            return None

    class Either(Space):
        @derived
        def value(self) -> int | str | Rejected:
            return 1

    for space_type in (Optional, Either):
        with pytest.raises(DefinitionError, match="needs explicit semantics="):
            collect_space(space_type)


class Projected(Space):
    """A derived value typed only by its annotation projects in the class body."""

    n: int = Param()

    @derived
    def point(self) -> Point | Rejected:
        return Point(self.n)

    x = View(point.x)


def test_a_marked_output_projects_by_its_value_type() -> None:
    assert design_space(Projected(n=3)).x == 3
    with pytest.raises(AttributeError):
        Projected.point.missing  # type: ignore[attr-defined]  # Point annotates none
