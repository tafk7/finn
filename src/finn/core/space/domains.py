# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Decision membership and optional enumeration with explicit dependencies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from math import isqrt
from typing import Generic, TypeVar, cast

from .errors import DefinitionError, EvaluationError
from .results import (
    QueryResult,
    Available,
    Inapplicable,
    Rejected,
    Unresolved,
    owned_result,
    reject,
)
from .semantics import ValueSemantics, default_semantics

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Domain(Generic[T]):
    """A membership callback and optional enumeration over declared inputs.

    Dependency objects are author declarations until compilation links them.
    Neither constructing nor binding a domain invokes its callbacks. Enumeration
    is advisory: every assignment goes through membership independently.
    """

    dependencies: tuple[tuple[str, object], ...]
    accepts: Callable[..., bool | QueryResult[bool]]
    candidates: Callable[..., Iterable[T] | QueryResult[Iterable[T]]] | None = None
    value_semantics: ValueSemantics[T] | None = None
    _finite_values: tuple[T, ...] | None = None

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.dependencies)
        if len(set(names)) != len(names):
            raise DefinitionError("domain dependency names must be unique")
        if any(not name.isidentifier() or name == "candidate" for name in names):
            raise DefinitionError(
                "domain dependency names must be identifiers other than candidate"
            )
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if not callable(self.accepts):
            raise DefinitionError("domain membership must be callable")
        if self.candidates is not None and not callable(self.candidates):
            raise DefinitionError("domain enumeration must be callable")

    def with_semantics(self, semantics: ValueSemantics[T]) -> Domain[T]:
        """Bind finite values to their Decision's declared snapshot and equality."""

        if (
            self.value_semantics is not None
            and self.value_semantics.type_token is not semantics.type_token
        ):
            raise DefinitionError("domain and decision have incompatible value semantics")
        if self._finite_values is not None:
            return finite(self._finite_values, semantics)
        return replace(self, value_semantics=semantics)

    def membership(
        self,
        candidate: T,
        dependency_values: Mapping[str, object],
        *,
        semantics: ValueSemantics[T],
        owner: str,
    ) -> QueryResult[bool]:
        """Evaluate membership and preserve explicit semantic refusal/nonvalues."""

        try:
            if not semantics.accepts(candidate):
                raise TypeError(f"expected candidate of nominal type {semantics.name}")
            if self._finite_values is not None:
                result: bool | QueryResult[bool] = any(
                    semantics.values_equal(candidate, allowed) for allowed in self._finite_values
                )
            else:
                result = self.accepts(candidate=candidate, **dependency_values)
            if type(result) is bool:
                result = Available(result)
            if not isinstance(result, (Available, Inapplicable, Rejected, Unresolved)):
                raise TypeError("domain membership must return bool or QueryResult[bool]")
            if isinstance(result, Available):
                if type(result.value) is not bool:
                    raise TypeError("domain membership must return bool or QueryResult[bool]")
                if result.value is False:
                    return reject(
                        "domain-membership", "candidate is outside the domain", owner=owner
                    )
            return owned_result(result, owner)
        except Exception as cause:
            raise EvaluationError(owner, "domain membership", str(cause)) from cause

    def enumerate(
        self,
        dependency_values: Mapping[str, object],
        *,
        semantics: ValueSemantics[T],
        owner: str,
    ) -> QueryResult[tuple[T, ...]]:
        """Read the optional candidate provider without adopting any value."""

        if self.candidates is None:
            return Inapplicable()
        try:
            result = self.candidates(**dependency_values)
            if isinstance(result, (Inapplicable, Rejected, Unresolved)):
                return owned_result(result, owner)
            values = result.value if isinstance(result, Available) else result
            return Available(tuple(semantics.freeze(value) for value in values))
        except Exception as cause:
            raise EvaluationError(owner, "domain enumeration", str(cause)) from cause


def finite(values: Iterable[T], semantics: ValueSemantics[T] | None = None) -> Domain[T]:
    """Finite membership uses declared equality and supports unhashable values.

    A Decision binds this domain to its semantics before use. Passing semantics
    here also makes direct domain use safe for mutable definition values.
    """

    ordered = tuple(values)
    if semantics is not None:
        ordered = tuple(semantics.freeze(value) for value in ordered)

    def accepts(*, candidate: T) -> bool:
        if semantics is None:
            raise DefinitionError(
                "a finite domain must be bound to value semantics before evaluation"
            )
        return any(semantics.values_equal(candidate, allowed) for allowed in ordered)

    def candidates() -> tuple[T, ...]:
        # The public enumeration method snapshots each result. The callback is
        # declaration data consumed only through that evaluation boundary.
        return ordered

    return Domain((), accepts, candidates, semantics, ordered)


def domain(
    *,
    accepts: Callable[..., bool | QueryResult[bool]],
    candidates: Callable[..., Iterable[T] | QueryResult[Iterable[T]]] | None = None,
    semantics: ValueSemantics[T] | None = None,
    **dependencies: object,
) -> Domain[T]:
    return Domain(tuple(dependencies.items()), accepts, candidates, semantics)


def divisors_of(extent: object) -> Domain[int]:
    """Positive divisors of a declared positive integer extent."""

    def accepts(*, candidate: int, extent: int) -> bool:
        return type(extent) is int and extent > 0 and candidate > 0 and extent % candidate == 0

    def candidates(*, extent: int) -> tuple[int, ...]:
        if type(extent) is not int:
            raise TypeError("divisors_of requires an integer extent")
        if extent <= 0:
            return ()
        low: list[int] = []
        high: list[int] = []
        for value in range(1, isqrt(extent) + 1):
            if extent % value == 0:
                low.append(value)
                quotient = extent // value
                if quotient != value:
                    high.append(quotient)
        return tuple(low + list(reversed(high)))

    return domain(
        accepts=accepts,
        candidates=cast(Callable[..., Iterable[int] | QueryResult[Iterable[int]]], candidates),
        semantics=default_semantics(int),
        extent=extent,
    )


__all__ = ["Domain", "divisors_of", "domain", "finite"]
