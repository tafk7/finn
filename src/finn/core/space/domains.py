# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Decision membership and optional enumeration, over declared inputs.

A domain states a Decision's values and nothing else: its membership and its
enumeration read facts only (the root's facts, values that read only facts, formals
bound to those), and assigning a value checks it against the domain alone (its type
check). A listed domain lists all its values: each listed value is checked by
membership too (one it rejects is not listed), and an empty list means none. A rule
linking a value to anything else (``ultra`` needs UltraRAM, two choices go together) is
a constraint of the Space that owns the choice. Compilation checks it: a domain whose
membership or enumeration reaches a choice is a ``DefinitionError``
(``domain-reads-choice``), naming the way it reads it (``_compose.check_domains``); an input
under the guards deciding whether the choice itself exists reads as a fact.

Each of the domain's callables takes the inputs its signature names, of those the domain
declares: ``accepts(*, candidate, ...)`` and ``candidates(*, ...)``. Its listed values are
answered in the order its enumeration lists them. A domain ranks nothing: a ranking of a
kernel's values (by the size a memory holds) is a view of the kernel, which a compiler's
policy names (FINN's ``FirstValid(order=...)``); ``domain(order=...)`` is refused.

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
from .semantics import NotAValue, ValueSemantics, canonical, default_semantics, same, unrecognized

_OWNER = "space.domains"

T = TypeVar("T")


def _contains(values: tuple[T, ...], candidate: T) -> bool:
    """Finite membership: one of ``values`` is the same value (``semantics.same``)."""
    return any(
        same(candidate, allowed, owner=_OWNER, role="domain membership") for allowed in values
    )


def _value(semantics: ValueSemantics[T], value: object) -> T:
    """A domain's value, type-checked by its Decision's ``semantics`` and in its canonical
    form (``semantics.canonical``); a ``TypeError`` or ``NotAValue`` where it is neither."""
    if not semantics.accepts(value):
        raise TypeError(unrecognized(semantics))
    return cast(T, canonical(value))


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
    """A membership callback and an optional enumeration over declared inputs.

    Dependency objects are author declarations until compilation links them.
    Neither constructing nor binding a domain invokes its callbacks. Enumeration
    is advisory: every assignment goes through membership independently.
    ``ordered`` states that the cases have an order by construction, which the
    enumeration follows. Each callable receives the inputs its signature names: the
    membership's inputs (``membership_inputs``) are those ``accepts`` and
    ``candidates`` name.
    """

    dependencies: tuple[tuple[str, object], ...]
    accepts: Callable[..., bool | QueryResult[bool]]
    candidates: Callable[..., Iterable[T] | QueryResult[Iterable[T]]] | None = None
    value_semantics: ValueSemantics[T] | None = None
    _finite_values: tuple[T, ...] | None = None
    ordered: bool = False
    _accepts_inputs: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _candidates_inputs: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

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
        callables: tuple[tuple[str, Callable[..., object] | None, tuple[str, ...]], ...] = (
            ("_accepts_inputs", self.accepts, ("candidate",)),
            ("_candidates_inputs", self.candidates, ()),
        )
        for attribute, function, skip in callables:
            if function is None:
                continue
            object.__setattr__(self, attribute, _named(function, names, skip))

    @property
    def membership_inputs(self) -> tuple[str, ...]:
        """The inputs membership and enumeration read (by name, in declaration order)."""
        named = {*self._accepts_inputs, *self._candidates_inputs}
        return tuple(name for name, _ in self.dependencies if name in named)

    def signatures(self) -> tuple[tuple[str, Callable[..., object], tuple[str, ...]], ...]:
        """Each callable, by role, with the inputs its signature names (compilation checks
        each is declared and binds)."""
        found: list[tuple[str, Callable[..., object], tuple[str, ...]]] = [
            ("membership", self.accepts, self._accepts_inputs)
        ]
        if self.candidates is not None:
            found.append(("enumeration", self.candidates, self._candidates_inputs))
        return tuple(found)

    def with_semantics(self, semantics: ValueSemantics[T]) -> Domain[T]:
        """Bind the domain to its Decision's type check: listed values are checked by it, and
        taken in their canonical form."""

        if self.value_semantics is not None and not self.value_semantics.is_compatible_with(
            cast(ValueSemantics[object], semantics)
        ):
            raise DefinitionError.broken(
                "domain-semantics-incompatible",
                _OWNER,
                "domain and decision have incompatible value semantics",
            )
        if self._finite_values is not None:
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
        """Evaluate membership and preserve explicit refusal/nonvalues."""

        try:
            if not semantics.accepts(candidate):
                raise TypeError(unrecognized(semantics))
            if self._finite_values is not None:
                result: bool | QueryResult[bool] = _contains(self._finite_values, candidate)
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
        """Read the optional candidate provider without adopting any value: each candidate
        type-checked, in its canonical form."""

        if self.candidates is None:
            return Inapplicable()
        try:
            inputs = {name: dependency_values[name] for name in self._candidates_inputs}
            result = self.candidates(**inputs)
            if isinstance(result, (Inapplicable, Rejected, Unresolved)):
                return owned_result(result, owner)
            values = result.value if isinstance(result, Available) else result
            return Available(tuple(_value(semantics, value) for value in values))
        except (EvaluationError, ValueUnavailableError):
            raise
        except Exception as cause:
            raise EvaluationError(owner, "domain enumeration", str(cause)) from cause


def finite(
    values: Iterable[T], semantics: ValueSemantics[T] | None = None, *, ordered: bool = False
) -> Domain[T]:
    """Listed values: a candidate is a member where one of them is the same value
    (``semantics.same``).

    A Decision binds this domain to its type check before use (``Domain.with_semantics``),
    which checks each value and takes it in its canonical form. The values are unordered
    unless ``ordered``: listed in their order.
    """

    listed = tuple(values)
    if semantics is not None:
        try:
            listed = tuple(_value(semantics, value) for value in listed)
        except NotAValue as refusal:
            raise DefinitionError.broken(
                "domain-value-not-value", _OWNER, f"a listed value: {refusal}"
            ) from None

    def accepts(*, candidate: T) -> bool:
        if semantics is None:
            raise DefinitionError.broken(
                "domain-semantics-unbound",
                _OWNER,
                "a finite domain must be bound to its Decision's type before evaluation",
            )
        return _contains(listed, candidate)

    def candidates() -> tuple[T, ...]:
        # The values as the Decision's type check took them (``with_semantics``).
        return listed

    return Domain((), accepts, candidates, semantics, listed, ordered=ordered)


def domain(
    *,
    accepts: Callable[..., bool | QueryResult[bool]],
    candidates: Callable[..., Iterable[T] | QueryResult[Iterable[T]]] | None = None,
    ordered: bool = False,
    **dependencies: object,
) -> Domain[T]:
    """A domain over ``dependencies``, each callable taking those its signature names;
    ``ordered`` when its cases have an order by construction (an integer range), which
    ``candidates`` follows. Its values' type is its Decision's annotation. ``order=`` is
    refused (``domain-order``): a domain ranks nothing; so is ``semantics=``
    (``domain-semantics``)."""
    if "semantics" in dependencies:
        raise DefinitionError.broken(
            "domain-semantics",
            _OWNER,
            "semantics= is gone: a domain's values are of its Decision's annotated type, "
            "compared by ==",
        )
    if "order" in dependencies:
        raise DefinitionError.broken(
            "domain-order",
            _OWNER,
            "a domain ranks nothing: its candidates are its listed order. Write a ranking as a "
            "view of the kernel (ram_preference), named by a compiler's policy (FINN's "
            "FirstValid(order=Kernel.ram_preference)); ordered=True states an order by "
            "construction",
        )
    return Domain(tuple(dependencies.items()), accepts, candidates, ordered=ordered)


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

    made = domain(
        accepts=accepts,
        candidates=cast(Callable[..., Iterable[int] | QueryResult[Iterable[int]]], candidates),
        ordered=True,
        extent=extent,
    )
    return replace(made, value_semantics=default_semantics(int))


__all__ = [
    "Domain",
    "divisors_of",
    "domain",
    "finite",
]
