# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Decision membership and optional enumeration with explicit dependencies.

A domain's cases may state requirements (``requires``, ``requiring``): facts a
case needs, each refused with its own named finding. Membership is the base's
and then every applicable requirement; enumeration stays the declared cases,
so the domain does not move with the facts. A case whose requirement fails is
not viable, which forcing reads (``finn.core.space._forcing``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from math import isqrt
from typing import Generic, TypeVar, cast

from .errors import DefinitionError, EvaluationError, ValueUnavailableError
from .results import (
    Available,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
    owned_result,
    reject,
)
from .semantics import ValueSemantics, default_semantics, unrecognized

T = TypeVar("T")


def _contains(values: tuple[T, ...], candidate: T, semantics: ValueSemantics[T]) -> bool:
    """Finite membership compares detached operands, including unhashable values."""
    return any(
        semantics.values_equal(semantics.freeze(candidate), semantics.freeze(allowed))
        for allowed in values
    )


@dataclass(frozen=True, slots=True)
class Requirement:
    """A fact some cases of a value Decision need: ``fact`` (a reference, truthy when
    it holds) for the cases it applies to (``cases``: values, a predicate over the
    case, or None for every case), refused as ``code: message``."""

    fact: object
    code: str
    message: str
    cases: tuple[object, ...] | Callable[[object], bool] | None = None

    def applies(self, candidate: object) -> bool:
        if self.cases is None:
            return True
        if callable(self.cases):
            return bool(self.cases(candidate))
        return candidate in self.cases


def requirement_argument(index: int) -> str:
    """The linked argument name of a domain's ``index``-th requirement's fact (never an
    identifier, so never a dependency's name)."""
    return f"requires[{index}]"


@dataclass(frozen=True, slots=True)
class Domain(Generic[T]):
    """A membership callback and optional enumeration over declared inputs.

    Dependency objects are author declarations until compilation links them.
    Neither constructing nor binding a domain invokes its callbacks. Enumeration
    is advisory: every assignment goes through membership independently.
    ``requirements`` refuse the cases they apply to when their fact does not
    hold; their facts are linked beside the dependencies.
    """

    dependencies: tuple[tuple[str, object], ...]
    accepts: Callable[..., bool | QueryResult[bool]]
    candidates: Callable[..., Iterable[T] | QueryResult[Iterable[T]]] | None = None
    value_semantics: ValueSemantics[T] | None = None
    _finite_values: tuple[T, ...] | None = None
    requirements: tuple[Requirement, ...] = ()

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
            return replace(finite(self._finite_values, semantics), requirements=self.requirements)
        return replace(self, value_semantics=semantics)

    def _own(self, dependency_values: Mapping[str, object]) -> dict[str, object]:
        """The callbacks' arguments: the dependencies, without the requirements' facts."""
        return {name: dependency_values[name] for name, _ in self.dependencies}

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
                raise TypeError(unrecognized(semantics))
            if self._finite_values is not None:
                result: bool | QueryResult[bool] = _contains(
                    self._finite_values, candidate, semantics
                )
            else:
                result = self.accepts(candidate=candidate, **self._own(dependency_values))
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
                for index, item in enumerate(self.requirements):
                    if (
                        item.applies(candidate)
                        and not dependency_values[requirement_argument(index)]
                    ):
                        return reject(item.code, f"{candidate!r}: {item.message}", owner=owner)
            return owned_result(result, owner)
        except (EvaluationError, ValueUnavailableError):
            raise
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
            result = self.candidates(**self._own(dependency_values))
            if isinstance(result, (Inapplicable, Rejected, Unresolved)):
                return owned_result(result, owner)
            values = result.value if isinstance(result, Available) else result
            return Available(tuple(semantics.freeze(value) for value in values))
        except (EvaluationError, ValueUnavailableError):
            raise
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
        return _contains(ordered, candidate, semantics)

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


def requires(
    fact: object,
    finding: str,
    *,
    cases: tuple[object, ...] | Callable[[object], bool] | None = None,
) -> Requirement:
    """A requirement of a value Decision's cases: ``fact`` must hold, or the case is
    refused with ``finding``, written ``"code: message"``
    (``requires(platform.uram, "uram-absent: the platform has no UltraRAM",
    cases=("ultra",))``)."""
    code, _, message = finding.partition(":")
    if not code.strip() or not message.strip():
        raise DefinitionError("a requirement's finding is written 'code: message'")
    return Requirement(fact, code.strip(), message.strip(), cases)


def requiring(
    base: Iterable[T] | Domain[T],
    *requirements: Requirement,
    semantics: ValueSemantics[T] | None = None,
) -> Domain[T]:
    """``base`` (a domain, or values) whose cases state ``requirements``.

    ``Decision(values=..., requires=(...))`` writes the same.
    """
    domain_ = base if isinstance(base, Domain) else finite(base, semantics)
    return replace(domain_, requirements=(*domain_.requirements, *requirements))


__all__ = [
    "Domain",
    "Requirement",
    "divisors_of",
    "domain",
    "finite",
    "requirement_argument",
    "requires",
    "requiring",
]
