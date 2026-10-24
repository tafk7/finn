# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A ``T | Rejected`` output is of type ``T``.

A union of value types and the engine's own result markers (``Rejected``,
``Inapplicable``, ``Unresolved``) is checked by the value types less the markers, as
``QueryResult[T]`` names ``T``: one class, ``T | None``, or a union of several classes.
A derived value states no ``semantics=``: its return annotation is its type. A Protocol
is no value type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Protocol

import pytest
from support.refusals import refused

from finn.core.space import (
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    View,
    constraint,
    derived,
    design_space,
    reject,
    view,
)
from finn.core.space.collection import collect_space
from finn.core.space.errors import DefinitionError
from finn.core.space.semantics import ValueSemantics, default_semantics


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


def test_a_derived_value_states_no_semantics() -> None:
    with refused("derived-semantics", error=DefinitionError, message="semantics= is gone"):

        class Explicit(Space):
            @derived(semantics=default_semantics(Point))
            def point(self) -> Point | Rejected:
                return Point(1)


def test_a_view_states_no_semantics() -> None:
    """A view's check is its return annotation's, or its source's for ``View(source)``."""
    with refused("view-semantics", error=DefinitionError, message="semantics= is gone"):

        class Explicit(Space):
            @view(semantics=default_semantics(Point))
            def point(self) -> Point | Rejected:
                return Point(1)

    with refused("view-semantics", error=DefinitionError, message="semantics= is gone"):

        class Guarded(Space):
            enabled: bool = Param()

            @view(semantics=default_semantics(Point), when=enabled)
            def point(self) -> Point | Rejected:
                return Point(1)

    class Stated(Space):
        n: int = Param()

        @view
        def point(self) -> Point | Rejected:
            return Point(self.n)

        again = View(point)
        plain = View(n)

    assert semantics_of(Stated, "point").type_token is Point
    assert semantics_of(Stated, "again").type_token is Point
    assert semantics_of(Stated, "plain").type_token is int


def test_a_value_semantics_in_annotated_metadata_is_refused() -> None:
    """The annotation's type is the check: a ``ValueSemantics`` in ``Annotated[...]`` is no
    longer taken or dropped silently."""

    with refused(
        "annotation-semantics",
        error=DefinitionError,
        message="semantics= is gone",
    ):

        class Stated(Space):
            point: Annotated[Point, default_semantics(Point)] = Param()

    class Plain(Space):
        point: Annotated[Point, "units"] = Param()

    assert semantics_of(Plain, "point").type_token is Point


def test_a_protocol_is_no_value_type() -> None:
    class Implicit(Space):
        @derived
        def named(self) -> Named | Rejected:
            raise AssertionError("never run")

    with refused("annotation-not-value-type", error=DefinitionError, message=r"no value type"):
        collect_space(Implicit)


def test_a_union_of_values_is_checked_by_each() -> None:
    class Optional(Space):
        @derived
        def point(self) -> Point | None:
            return None

    class Either(Space):
        @derived
        def value(self) -> int | str | Rejected:
            return 1

    optional = semantics_of(Optional, "point")
    assert optional.type_token is Point and optional.accepts(None)
    either = semantics_of(Either, "value")
    assert either.type_token == frozenset({int, str})
    assert either.accepts(1) and either.accepts("1") and not either.accepts(1.0)
    assert design_space(Optional()).point is None and design_space(Either()).value == 1


class Projected(Space):
    """A derived value typed only by its annotation: another derived value reads its attribute."""

    n: int = Param()

    @derived
    def point(self) -> Point | Rejected:
        return Point(self.n)

    @derived(point=point)
    def x_of_point(*, point: Point) -> int:
        return point.x

    x = View(x_of_point)


def test_a_marked_output_is_read_by_a_derived_value_through_its_value_type() -> None:
    assert design_space(Projected(n=3)).x == 3
    with pytest.raises(AttributeError):
        Projected.point.missing  # type: ignore[attr-defined]  # Point annotates none
