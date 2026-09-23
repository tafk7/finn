# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private, typed adapters from declaration aggregates to ordinary Space nodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, Signature
from typing import TypeVar

from finn.dataflow._engine import ValueSemantics
from finn.dataflow.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Derived,
    Readiness,
    Space,
    ValueSource,
    semantics_for,
)

T = TypeVar("T")
Evaluation = Callable[[Mapping[str, object]], object]
Dependencies = tuple[tuple[str, ValueSource[object]], ...]


@dataclass(frozen=True)
class NamedEvaluation:
    """Enforce the exact generated keyword contract reported to Space."""

    names: tuple[str, ...]
    function: Evaluation = field(compare=False, repr=False)

    @property
    def __signature__(self) -> Signature:
        return Signature(tuple(Parameter(name, Parameter.KEYWORD_ONLY) for name in self.names))

    def __call__(self, **values: object) -> object:
        self.__signature__.bind(**values)
        return self.function(values)


def property_node(
    value_type: type[T] | ValueSemantics[T],
    dependencies: Dependencies,
    evaluate: Evaluation,
) -> Derived[T]:
    return Derived(
        semantics_for(value_type),
        None,
        dependencies,
        NamedEvaluation(tuple(name for name, _ in dependencies), evaluate),
    )


def condition_node(dependencies: Dependencies, evaluate: Evaluation) -> Constraint:
    return Constraint(
        dependencies,
        NamedEvaluation(tuple(name for name, _ in dependencies), evaluate),
    )


def install_members(owner: type[object], members: Sequence[tuple[str, object]]) -> None:
    if not issubclass(owner, Space):
        raise AuthoringError("contract declarations must be owned by a Space")
    names = tuple(name for name, _ in members)
    if len(names) != len(set(names)):
        raise AuthoringError("a contract aggregate generated one member name twice")
    for name in names:
        if any(name in base.__dict__ for base in owner.__mro__):
            raise AuthoringError(f"{owner.__name__}.{name} conflicts with an existing declaration")
    existing = {
        id(value)
        for base in owner.__mro__
        for value in base.__dict__.values()
        if isinstance(value, (ValueSource, Constraint, ConstraintGroup, Readiness))
    }
    generated: set[int] = set()
    for name, value in members:
        if isinstance(value, (ValueSource, Constraint, ConstraintGroup, Readiness)):
            if id(value) in existing or id(value) in generated:
                raise AuthoringError(
                    f"{owner.__name__}.{name} would duplicate declaration ownership"
                )
            generated.add(id(value))
    for name, value in members:
        setattr(owner, name, value)
