# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed integer/index expressions for local-contract authoring.

These are declaration syntax, not a new normalized map algebra or evaluator.
Each generated value uses the existing Space dependency machinery.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from finn.dataflow.model.logical._contract_support import (
    condition_node,
    install_members,
    property_node,
)
from finn.kernels.space.declarations import (
    AuthoringError,
    Constraint,
    Derived,
    ValueSource,
    reject,
    semantics_for,
)

IntegerValue: TypeAlias = "int | ValueSource[int] | IntegerExpression"


@dataclass(frozen=True, init=False)
class IntegerExpression:
    """An integer literal, a shared integer source, or addition/multiplication."""

    _node: int | ValueSource[int] | tuple[str, IntegerExpression, IntegerExpression]

    def __init__(self, value: IntegerValue) -> None:
        if isinstance(value, IntegerExpression):
            node = value._node
        elif isinstance(value, ValueSource):
            if value.value_semantics.type_token is not int:
                raise AuthoringError("integer expressions require integer ValueSources")
            node = value
        elif type(value) is int:
            node = value
        else:
            raise AuthoringError("integer expression requires an integer or integer source")
        object.__setattr__(self, "_node", node)

    def _binary(self, operation: str, other: IntegerValue) -> IntegerExpression:
        result = object.__new__(IntegerExpression)
        object.__setattr__(result, "_node", (operation, self, IntegerExpression(other)))
        return result

    def __add__(self, other: IntegerValue) -> IntegerExpression:
        return self._binary("+", other)

    __radd__ = __add__

    def __mul__(self, other: IntegerValue) -> IntegerExpression:
        return self._binary("*", other)

    __rmul__ = __mul__

    def __sub__(self, other: IntegerValue) -> IntegerExpression:
        return self + IntegerExpression(other) * -1

    def __rsub__(self, other: IntegerValue) -> IntegerExpression:
        return IntegerExpression(other) - self

    @property
    def sources(self) -> tuple[ValueSource[int], ...]:
        node = self._node
        if isinstance(node, ValueSource):
            return (node,)
        if isinstance(node, tuple):
            return tuple(dict.fromkeys((*node[1].sources, *node[2].sources)))
        return ()

    def evaluate(self, values: Mapping[ValueSource[int], int]) -> int:
        node = self._node
        if isinstance(node, ValueSource):
            return values[node]
        if isinstance(node, tuple):
            operation, left, right = node
            a, b = left.evaluate(values), right.evaluate(values)
            return a + b if operation == "+" else a * b
        return node

    def as_source(self) -> tuple[ValueSource[int], bool]:
        """Return a source and whether this expression created a new owned node."""
        if isinstance(self._node, ValueSource):
            return self._node, False
        sources = self.sources

        def evaluate(values: Mapping[str, object]) -> int:
            return self.evaluate(
                {source: cast("int", values[f"v{j}"]) for j, source in enumerate(sources)}
            )

        return property_node(
            int,
            tuple((f"v{j}", source) for j, source in enumerate(sources)),
            evaluate,
        ), True


def integer(value: IntegerValue) -> IntegerExpression:
    return IntegerExpression(value)


@dataclass(frozen=True, eq=False, init=False)
class Index:
    """One scoped index identity and its parameterized extent."""

    name: str
    extent: IntegerExpression
    value: ValueSource[int]
    condition: Constraint
    owns_value: bool

    def __init__(self, name: str, extent: IntegerValue) -> None:
        if not name.isidentifier() or not name.isascii():
            raise AuthoringError("index name must be an ASCII identifier")
        expression = integer(extent)
        value, owned = expression.as_source()

        def positive(values: Mapping[str, object]) -> object:
            actual = cast("int", values["extent"])
            if actual <= 0:
                return reject("contract-index-extent", f"index {name} needs a positive extent")
            return True

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "extent", expression)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "owns_value", owned)
        object.__setattr__(self, "condition", condition_node((("extent", value),), positive))

    def expression(self) -> Subscript:
        return Subscript(integer(0), ((self, integer(1)),))

    def __add__(self, other: SubscriptValue) -> Subscript:
        return self.expression() + other

    __radd__ = __add__

    def __sub__(self, other: SubscriptValue) -> Subscript:
        return self.expression() - other

    def __mul__(self, coefficient: IntegerValue) -> Subscript:
        return self.expression() * coefficient

    __rmul__ = __mul__


SubscriptValue: TypeAlias = "IntegerValue | Index | Subscript"


@dataclass(frozen=True)
class Subscript:
    """A tensor coordinate affine in index identities, with shared scalar sources."""

    constant: IntegerExpression
    terms: tuple[tuple[Index, IntegerExpression], ...] = ()

    def __add__(self, other: SubscriptValue) -> Subscript:
        rhs = subscript(other)
        terms = dict(self.terms)
        for axis, coefficient in rhs.terms:
            terms[axis] = terms[axis] + coefficient if axis in terms else coefficient
        return Subscript(self.constant + rhs.constant, tuple(terms.items()))

    __radd__ = __add__

    def __sub__(self, other: SubscriptValue) -> Subscript:
        return self + subscript(other) * -1

    def __mul__(self, coefficient: IntegerValue) -> Subscript:
        return Subscript(
            self.constant * coefficient,
            tuple((axis, value * coefficient) for axis, value in self.terms),
        )

    __rmul__ = __mul__


def subscript(value: SubscriptValue) -> Subscript:
    if isinstance(value, Subscript):
        return value
    if isinstance(value, Index):
        return value.expression()
    return Subscript(integer(value))


@dataclass(frozen=True)
class Schedule:
    """One rectangular local work schedule; fixed coordinates select a scope."""

    axes: tuple[Index, ...]
    fixed: tuple[tuple[Index, IntegerExpression], ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.axes)) != len(self.axes) or len({x.name for x in self.axes}) != len(
            self.axes
        ):
            raise AuthoringError("schedule index identities and names must be unique")
        if len({axis for axis, _ in self.fixed}) != len(self.fixed) or any(
            axis not in self.axes for axis, _ in self.fixed
        ):
            raise AuthoringError("fixed coordinates must name distinct axes of this schedule")

    def fix(self, **values: IntegerValue) -> Schedule:
        by_name = {axis.name: axis for axis in self.axes}
        if any(name not in by_name for name in values):
            raise AuthoringError("fixed coordinate names an index outside this schedule")
        fixed = dict(self.fixed)
        fixed.update((by_name[name], integer(value)) for name, value in values.items())
        return Schedule(self.axes, tuple(fixed.items()))


@dataclass(frozen=True, eq=False, init=False)
class ExactQuotient(Derived[int]):
    """One exact-division declaration installs its independently assessable condition."""

    condition: Constraint = field(init=False)

    def __init__(self, numerator: ValueSource[int], denominator: ValueSource[int]) -> None:
        for value in (numerator, denominator):
            if value.value_semantics.type_token is not int:
                raise AuthoringError("exact division requires integer sources")

        def valid(*, numerator: int, denominator: int) -> object:
            if numerator <= 0 or denominator <= 0 or numerator % denominator:
                return reject(
                    "contract-exact-division",
                    "exact division needs positive divisible integers",
                    values={"numerator": numerator, "denominator": denominator},
                )
            return True

        def divide(*, numerator: int, denominator: int) -> object:
            condition = valid(numerator=numerator, denominator=denominator)
            return numerator // denominator if condition is True else condition

        dependencies = (("numerator", numerator), ("denominator", denominator))
        Derived.__init__(self, semantics_for(int), None, dependencies, divide)
        object.__setattr__(self, "condition", Constraint(dependencies, valid))

    def __set_name__(self, owner: type[object], name: str) -> None:
        install_members(owner, ((f"{name}_exact", self.condition),))


__all__ = ["ExactQuotient", "Index", "IntegerExpression", "Schedule", "Subscript", "integer"]
