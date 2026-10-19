# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Decision membership, optional enumeration and an optional order, over declared inputs.

A domain states a Decision's values and nothing else: its membership and its
enumeration read facts only (the root's facts, values that read only facts, formals
bound to those), and assigning a value checks it against the domain alone (its type
check). A listed domain lists all its values: each listed value is checked by
membership too (one it rejects is not listed), and an empty list means none. A rule
linking a value to anything else (``ultra`` needs UltraRAM, two choices go together) is
a constraint of the Space that owns the choice.

Each of the domain's callables takes the inputs its signature names, of those the domain
declares: ``accepts(*, candidate, ...)``, ``candidates(*, ...)`` and ``order(values,
*, ...)``. A domain's **order** ranks its enumerated values and checks nothing: it may
read other values (a choice, the size a memory holds), and while one it reads is open the
ranking waits. It returns a permutation of the values it is given, which is checked. The
ranking is information for a policy (conversion's first valid value, a strategy); validity
and refusals never read it.

A domain is **ordered** when its cases have an order by construction, which its
enumeration follows: the divisors of an extent ascending, an integer range. A
domain over listed values is unordered unless declared ordered
(``Decision(values=..., ordered=True)``). That too is information for a policy (the DSE
seam steps an ordered choice); validity never depends on it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
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

_OWNER = "space.domains"

T = TypeVar("T")


def _contains(values: tuple[T, ...], candidate: T, semantics: ValueSemantics[T]) -> bool:
    """Finite membership compares detached operands, including unhashable values."""
    return any(
        semantics.values_equal(semantics.freeze(candidate), semantics.freeze(allowed))
        for allowed in values
    )


def _named(
    function: Callable[..., object], declared: tuple[str, ...], skip: tuple[str, ...]
) -> tuple[str, ...]:
    """The inputs ``function``'s signature names (every declared one where it takes
    ``**``keywords), less ``skip``. Compilation checks that each is declared."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return declared
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
        return declared
    return tuple(
        parameter.name
        for parameter in parameters
        if parameter.kind in (parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in skip
    )


@dataclass(frozen=True, slots=True)
class Domain(Generic[T]):
    """A membership callback, an optional enumeration and an optional order over declared
    inputs.

    Dependency objects are author declarations until compilation links them.
    Neither constructing nor binding a domain invokes its callbacks. Enumeration
    is advisory: every assignment goes through membership independently.
    ``ordered`` states that the cases have an order by construction, which the
    enumeration follows. ``order(values, *, ...)`` ranks the enumerated values and may
    read other values. Each callable receives the inputs its signature names: the
    membership's inputs (``membership_inputs``) are those ``accepts`` and
    ``candidates`` name, the order's (``order_inputs``) those ``order`` names.
    """

    dependencies: tuple[tuple[str, object], ...]
    accepts: Callable[..., bool | QueryResult[bool]]
    candidates: Callable[..., Iterable[T] | QueryResult[Iterable[T]]] | None = None
    value_semantics: ValueSemantics[T] | None = None
    _finite_values: tuple[T, ...] | None = None
    ordered: bool = False
    order: Callable[..., Iterable[T]] | None = None
    _accepts_inputs: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _candidates_inputs: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _order_inputs: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.dependencies)
        if len(set(names)) != len(names):
            raise DefinitionError.broken(
                "domain-dependencies-duplicate", _OWNER, "domain dependency names must be unique"
            )
        if any(not name.isidentifier() or name == "candidate" for name in names):
            raise DefinitionError.broken(
                "domain-dependency-name-invalid",
                _OWNER,
                "domain dependency names must be identifiers other than candidate",
            )
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if not callable(self.accepts):
            raise DefinitionError.broken(
                "domain-membership-not-callable", _OWNER, "domain membership must be callable"
            )
        if self.candidates is not None and not callable(self.candidates):
            raise DefinitionError.broken(
                "domain-enumeration-not-callable", _OWNER, "domain enumeration must be callable"
            )
        if self.order is not None and not callable(self.order):
            raise DefinitionError.broken(
                "domain-order-not-callable", _OWNER, "a domain's order must be callable"
            )
        if self.order is not None and self.candidates is None:
            raise DefinitionError.broken(
                "domain-order-unlisted",
                _OWNER,
                "a domain known by membership only lists nothing to order",
            )
        callables: tuple[tuple[str, Callable[..., object] | None, tuple[str, ...]], ...] = (
            ("_accepts_inputs", self.accepts, ("candidate",)),
            ("_candidates_inputs", self.candidates, ()),
            ("_order_inputs", self.order, ()),
        )
        for attribute, function, skip in callables:
            if function is None:
                continue
            if attribute == "_order_inputs":
                skip = _first_parameter(function)  # the values it ranks
            object.__setattr__(self, attribute, _named(function, names, skip))

    @property
    def membership_inputs(self) -> tuple[str, ...]:
        """The inputs membership and enumeration read (by name, in declaration order)."""
        named = {*self._accepts_inputs, *self._candidates_inputs}
        return tuple(name for name, _ in self.dependencies if name in named)

    @property
    def order_inputs(self) -> tuple[str, ...]:
        """The inputs the order reads (by name, in declaration order)."""
        return tuple(name for name, _ in self.dependencies if name in self._order_inputs)

    def signatures(self) -> tuple[tuple[str, Callable[..., object], tuple[str, ...]], ...]:
        """Each callable, by role, with the inputs its signature names (compilation checks
        each is declared and binds)."""
        found: list[tuple[str, Callable[..., object], tuple[str, ...]]] = [
            ("membership", self.accepts, self._accepts_inputs)
        ]
        if self.candidates is not None:
            found.append(("enumeration", self.candidates, self._candidates_inputs))
        if self.order is not None:
            found.append(("order", self.order, self._order_inputs))
        return tuple(found)

    def with_semantics(self, semantics: ValueSemantics[T]) -> Domain[T]:
        """Bind finite values to their Decision's declared snapshot and equality."""

        if (
            self.value_semantics is not None
            and self.value_semantics.type_token is not semantics.type_token
        ):
            raise DefinitionError.broken(
                "domain-semantics-incompatible",
                _OWNER,
                "domain and decision have incompatible value semantics",
            )
        if self._finite_values is not None and self.order is None:
            return finite(self._finite_values, semantics, ordered=self.ordered)
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
                raise TypeError(unrecognized(semantics))
            if self._finite_values is not None:
                result: bool | QueryResult[bool] = _contains(
                    self._finite_values, candidate, semantics
                )
            else:
                inputs = {name: dependency_values[name] for name in self._accepts_inputs}
                result = self.accepts(candidate=candidate, **inputs)
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
            inputs = {name: dependency_values[name] for name in self._candidates_inputs}
            result = self.candidates(**inputs)
            if isinstance(result, (Inapplicable, Rejected, Unresolved)):
                return owned_result(result, owner)
            values = result.value if isinstance(result, Available) else result
            return Available(tuple(semantics.freeze(value) for value in values))
        except (EvaluationError, ValueUnavailableError):
            raise
        except Exception as cause:
            raise EvaluationError(owner, "domain enumeration", str(cause)) from cause

    def rank(
        self,
        values: tuple[T, ...],
        dependency_values: Mapping[str, object],
        *,
        semantics: ValueSemantics[T],
        owner: str,
    ) -> tuple[T, ...]:
        """``values`` (its enumeration) as its order ranks them, checked to be a permutation
        of them by the values' semantics; ``values`` themselves where it has no order."""
        if self.order is None:
            return values
        try:
            inputs = {name: dependency_values[name] for name in self._order_inputs}
            ranked = tuple(semantics.freeze(value) for value in self.order(values, **inputs))
        except (EvaluationError, ValueUnavailableError):
            raise
        except Exception as cause:
            raise EvaluationError(owner, "domain order", str(cause)) from cause
        rest = list(values)
        for value in ranked:
            found = next(
                (i for i, held in enumerate(rest) if semantics.values_equal(held, value)), None
            )
            if found is None:
                rest = [value]  # a value it was not given, or one given twice
                break
            del rest[found]
        if rest:
            raise EvaluationError(
                owner, "domain order", "the order is not a permutation of the values it ranks"
            )
        return ranked


def _first_parameter(function: Callable[..., object]) -> tuple[str, ...]:
    """The name of ``function``'s first positional parameter (an order's values)."""
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return ()
    if parameters and parameters[0].kind in (
        parameters[0].POSITIONAL_ONLY,
        parameters[0].POSITIONAL_OR_KEYWORD,
    ):
        return (parameters[0].name,)
    return ()


def finite(
    values: Iterable[T], semantics: ValueSemantics[T] | None = None, *, ordered: bool = False
) -> Domain[T]:
    """Finite membership uses declared equality and supports unhashable values.

    A Decision binds this domain to its semantics before use. Passing semantics
    here also makes direct domain use safe for mutable definition values. The
    values are unordered unless ``ordered``: listed in their order.
    """

    listed = tuple(values)
    if semantics is not None:
        listed = tuple(semantics.freeze(value) for value in listed)

    def accepts(*, candidate: T) -> bool:
        if semantics is None:
            raise DefinitionError.broken(
                "domain-semantics-unbound",
                _OWNER,
                "a finite domain must be bound to value semantics before evaluation",
            )
        return _contains(listed, candidate, semantics)

    def candidates() -> tuple[T, ...]:
        # The public enumeration method snapshots each result. The callback is
        # declaration data consumed only through that evaluation boundary.
        return listed

    return Domain((), accepts, candidates, semantics, listed, ordered=ordered)


def domain(
    *,
    accepts: Callable[..., bool | QueryResult[bool]],
    candidates: Callable[..., Iterable[T] | QueryResult[Iterable[T]]] | None = None,
    semantics: ValueSemantics[T] | None = None,
    ordered: bool = False,
    order: Callable[..., Iterable[T]] | None = None,
    **dependencies: object,
) -> Domain[T]:
    """A domain over ``dependencies``, each callable taking those its signature names;
    ``ordered`` when its cases have an order by construction (an integer range), which
    ``candidates`` follows; ``order(values, *, ...)`` ranks the enumerated values,
    reading what it names (other choices among them)."""
    return Domain(
        tuple(dependencies.items()), accepts, candidates, semantics, ordered=ordered, order=order
    )


def divisors_of(extent: object) -> Domain[int]:
    """Positive divisors of a declared positive integer extent, ascending (ordered)."""

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
        ordered=True,
        extent=extent,
    )


__all__ = [
    "Domain",
    "divisors_of",
    "domain",
    "finite",
]
