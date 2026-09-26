# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed declarations for the Space language.

Declarations describe structure and preserve authored value types. Descriptors
dispatch to bound configuration operations when read through a Space instance.
"""

# Runtime dispatch is deliberately lazy to keep declaration collection independent.
# ruff: noqa: PLC0415

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast, overload

from typing_extensions import Self

from .domains import Domain, finite
from .errors import DefinitionError
from .graph import LOCATED, Located
from .results import (
    QueryResult,
)
from .semantics import ValueSemantics, semantics_for

if TYPE_CHECKING:
    from ._configuration import BoundView, ChoiceView, Space
    from .expressions import Expr

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound="Space")
S_co = TypeVar("S_co", bound="Space", covariant=True)


def local_name(value: str, role: str) -> str:
    """Keep authored identity segments disjoint from generated node names."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise DefinitionError(f"{role} must be one nonempty ASCII name segment")
    return value


def class_namespace(space_type: type[Space]) -> dict[str, object]:
    namespace: dict[str, object] = {}
    for base in reversed(space_type.__mro__):
        namespace.update(vars(base))
    namespace[space_type.__name__] = space_type
    return namespace


class Declaration:
    """Identity-bearing source declaration; names are diagnostic hints only."""

    name: str | None = None
    owner: type[object] | None = None
    when: ValueRef[bool] | None = None

    def __set_name__(self, owner: type[object], name: str) -> None:
        if self.owner is not None and (self.owner is not owner or self.name != name):
            raise DefinitionError(
                f"{owner.__qualname__}.{name}: declaration already belongs to "
                f"{self.owner.__qualname__}.{self.name}; create a fresh declaration"
            )
        self.owner, self.name = owner, name

    def __bool__(self) -> bool:
        raise TypeError("a declaration has no truth value")


class ValueRef(Declaration, Generic[T_co]):
    """A typed value handle, including scoped aliases and accepted outputs."""

    semantics: ValueSemantics[T_co] | None

    @property
    def value_semantics(self) -> ValueSemantics[T_co] | None:
        return self.semantics

    def __add__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("add", self, other)

    def __radd__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("add", other, self)

    def __sub__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("sub", self, other)

    def __rsub__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("sub", other, self)

    def __mul__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mul", self, other)

    def __rmul__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mul", other, self)

    def __floordiv__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("floordiv", self, other)

    def __rfloordiv__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("floordiv", other, self)

    def __mod__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mod", self, other)

    def __rmod__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mod", other, self)

    def __neg__(self: ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("neg", self)


class ValueDecl(ValueRef[T_co], Generic[T_co]):
    """A class member descriptor; scoped handles remain ordinary typed objects."""

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> T_co: ...

    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Self | T_co:
        if instance is None:
            return self
        from .occurrence import read_value

        return read_value(instance, self)


class Param(ValueDecl[T], Generic[T]):
    """A required or optional formal input supplied when a root is started."""

    def __init__(
        self,
        value_type: type[T] | ValueSemantics[T],
        *,
        required: bool = True,
        semantics: ValueSemantics[T] | None = None,
    ) -> None:
        self.semantics = semantics if semantics is not None else semantics_for(value_type)
        self.required = required


class Const(ValueDecl[T], Generic[T]):
    """A definition-owned frozen value."""

    def __init__(self, value: T, *, semantics: ValueSemantics[T] | None = None) -> None:
        self.semantics = semantics if semantics is not None else semantics_for(type(value))
        self.value = self.semantics.freeze(value)


class Decision(ValueDecl[T], Generic[T]):
    """An independently editable choice. Its type parameter is invariant."""

    def __init__(
        self,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: Domain[T] | None = None,
        values: Iterable[T] | None = None,
        semantics: ValueSemantics[T] | None = None,
        when: ValueRef[bool] | None = None,
    ) -> None:
        if (domain is None) == (values is None):
            raise DefinitionError("a Decision needs exactly one of domain= or values=")
        self.semantics = semantics if semantics is not None else semantics_for(value_type)
        self.when = when
        self.domain = (
            domain.with_semantics(self.semantics)
            if domain is not None
            else finite(cast(Iterable[T], values), self.semantics)
        )


class Derived(ValueDecl[T], Generic[T]):
    """A function whose dependencies are bound after effective collection."""

    def __init__(
        self,
        function: Callable[..., T | QueryResult[T]],
        *,
        semantics: ValueSemantics[T] | None = None,
        aliases: Mapping[str, object] | None = None,
        when: ValueRef[bool] | None = None,
    ) -> None:
        self.function = function
        self.semantics = semantics
        self.aliases = MappingProxyType(dict(aliases or {}))
        self.when = when


class _DerivedDecorator:
    def __init__(self, aliases: Mapping[str, object], when: ValueRef[bool] | None) -> None:
        self.aliases, self.when = aliases, when

    def __call__(self, function: Callable[..., T]) -> Derived[T]:
        return Derived(function, aliases=self.aliases, when=self.when)


class _SemanticDerivedDecorator(Generic[T]):
    def __init__(
        self,
        semantics: ValueSemantics[T],
        aliases: Mapping[str, object],
        when: ValueRef[bool] | None,
    ) -> None:
        self.semantics, self.aliases, self.when = semantics, aliases, when

    def __call__(self, function: Callable[..., T | QueryResult[T]]) -> Derived[T]:
        return Derived(function, semantics=self.semantics, aliases=self.aliases, when=self.when)


@overload
def derived(function: Callable[..., T], /) -> Derived[T]: ...


@overload
def derived(
    *, semantics: ValueSemantics[T], when: ValueRef[bool] | None = None, **aliases: object
) -> _SemanticDerivedDecorator[T]: ...


@overload
def derived(
    *, semantics: None = None, when: ValueRef[bool] | None = None, **aliases: object
) -> _DerivedDecorator: ...


def derived(
    function: Callable[..., object] | None = None,
    /,
    *,
    semantics: object = None,
    when: ValueRef[bool] | None = None,
    **aliases: object,
) -> object:
    if function is not None:
        return Derived(function, aliases=aliases, when=when)
    if semantics is not None:
        if not isinstance(semantics, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        return _SemanticDerivedDecorator(semantics, aliases, when)
    return _DerivedDecorator(aliases, when)


class Constraint(Declaration):
    def __init__(
        self,
        function: Callable[..., bool | QueryResult[bool]],
        *,
        aliases: Mapping[str, object] | None = None,
        when: ValueRef[bool] | None = None,
    ) -> None:
        self.function = function
        self.aliases = MappingProxyType(dict(aliases or {}))
        self.semantics = semantics_for(bool)
        self.when = when


class _ConstraintDecorator:
    def __init__(self, aliases: Mapping[str, object], when: ValueRef[bool] | None) -> None:
        self.aliases, self.when = aliases, when

    def __call__(self, function: Callable[..., bool | QueryResult[bool]]) -> Constraint:
        return Constraint(function, aliases=self.aliases, when=self.when)


@overload
def constraint(function: Callable[..., bool | QueryResult[bool]], /) -> Constraint: ...


@overload
def constraint(
    *, when: ValueRef[bool] | None = None, **aliases: object
) -> _ConstraintDecorator: ...


def constraint(
    function: Callable[..., bool | QueryResult[bool]] | None = None,
    /,
    *,
    when: ValueRef[bool] | None = None,
    **aliases: object,
) -> Constraint | _ConstraintDecorator:
    if function is not None:
        return Constraint(function, aliases=aliases, when=when)
    return _ConstraintDecorator(aliases, when)


class ConstraintGroup(Declaration):
    def __init__(self, *constraints: Constraint) -> None:
        self.constraints = constraints


# A view's obligations: constraints, groups, other views (their acceptance),
# accepted references, and member families (each member's acceptance).
Obligation: TypeAlias = (
    "Constraint | ConstraintGroup | View[Any] | AcceptedViewRef[Any] | Members[Any]"
)


class View(Declaration, Generic[T]):
    """One assessment declaration for either a value or an authored function."""

    def __init__(
        self,
        source: ValueRef[T],
        *,
        constraints: Sequence[Obligation] = (),
        when: ValueRef[bool] | None = None,
    ) -> None:
        self.source: ValueRef[T] | None = source
        self.function: Callable[..., T | QueryResult[T]] | None = None
        self.aliases: Mapping[str, object] = MappingProxyType({})
        self.semantics: ValueSemantics[T] | None = source.semantics
        self.constraints = tuple(constraints)
        self.when = when

    @classmethod
    def from_function(
        cls,
        function: Callable[..., T | QueryResult[T]],
        *,
        semantics: ValueSemantics[T] | None,
        aliases: Mapping[str, object],
        constraints: Sequence[Obligation],
        when: ValueRef[bool] | None = None,
    ) -> View[T]:
        result = cls.__new__(cls)
        result.source = None
        result.function = function
        result.aliases = MappingProxyType(dict(aliases))
        result.semantics = semantics
        result.constraints = tuple(constraints)
        result.when = when
        return result

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> BoundView[T]: ...

    def __get__(
        self, instance: Space | None, owner: type[object] | None = None
    ) -> Self | BoundView[T]:
        from ._configuration import BoundView

        return self if instance is None else BoundView(instance, self)


class _ViewDecorator:
    def __init__(
        self,
        aliases: Mapping[str, object],
        constraints: Sequence[Obligation],
        when: ValueRef[bool] | None,
    ) -> None:
        self.aliases, self.constraints = aliases, constraints
        self.when = when

    def __call__(self, function: Callable[..., T]) -> View[T]:
        return View.from_function(
            function,
            semantics=None,
            aliases=self.aliases,
            constraints=self.constraints,
            when=self.when,
        )


class _SemanticViewDecorator(Generic[T]):
    def __init__(self, semantics: ValueSemantics[T], decorator: _ViewDecorator) -> None:
        self.semantics, self.decorator = semantics, decorator

    def __call__(self, function: Callable[..., T | QueryResult[T]]) -> View[T]:
        return View.from_function(
            function,
            semantics=self.semantics,
            aliases=self.decorator.aliases,
            constraints=self.decorator.constraints,
            when=self.decorator.when,
        )


@overload
def view(function: Callable[..., T], /) -> View[T]: ...


@overload
def view(
    *,
    semantics: ValueSemantics[T],
    when: ValueRef[bool] | None = None,
    constraints: Sequence[Obligation] = (),
    **aliases: object,
) -> _SemanticViewDecorator[T]: ...


@overload
def view(
    *,
    semantics: None = None,
    when: ValueRef[bool] | None = None,
    constraints: Sequence[Obligation] = (),
    **aliases: object,
) -> _ViewDecorator: ...


def view(
    function: Callable[..., object] | None = None,
    /,
    *,
    semantics: object = None,
    when: ValueRef[bool] | None = None,
    constraints: Sequence[Obligation] = (),
    **aliases: object,
) -> object:
    decorator = _ViewDecorator(aliases, constraints, when)
    if function is not None:
        return decorator(function)
    if semantics is not None:
        if not isinstance(semantics, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        return _SemanticViewDecorator(semantics, decorator)
    return decorator


class ValueKey(Generic[T_co]):
    """A typed export contract independent of a concrete child class."""

    def __init__(self, name: str, value_type: type[T_co] | ValueSemantics[T_co]) -> None:
        self.name, self.semantics = local_name(name, "export name"), semantics_for(value_type)


class ViewKey(Generic[T_co]):
    def __init__(self, name: str, value_type: type[T_co] | ValueSemantics[T_co]) -> None:
        self.name, self.semantics = local_name(name, "export name"), semantics_for(value_type)


class ScopedValueRef(ValueRef[T], Generic[T]):
    def __init__(
        self,
        placement: Subspace[Space] | SubspaceChoice,
        member: ValueRef[T] | ValueKey[T],
    ) -> None:
        self.placement, self.member = placement, member
        self.semantics = member.semantics


class DecisionRef(ScopedValueRef[T], Generic[T]):
    """Editable handle; compilation verifies that this placement owns a choice."""


class AcceptedViewRef(ValueRef[T], Generic[T]):
    def __init__(
        self, placement: Subspace[Space] | SubspaceChoice, member: View[T] | ViewKey[T]
    ) -> None:
        self.placement, self.member = placement, member
        self.semantics = member.semantics


class Subspace(Declaration, Generic[S_co]):
    def __init__(
        self,
        space_type: type[S_co],
        *,
        when: ValueRef[bool] | None = None,
        bindings: Mapping[ValueRef[object], object] | None = None,
        **parameters: object,
    ) -> None:
        from ._configuration import Space

        if not isinstance(space_type, type) or not issubclass(space_type, Space):
            raise DefinitionError("a Subspace requires a Space subclass")
        self.space_type = space_type
        if bindings is not None and not isinstance(bindings, Mapping):
            raise DefinitionError("bindings= requires a declaration-keyed mapping")
        if bindings is not None and any(not isinstance(key, ValueRef) for key in bindings):
            raise DefinitionError("binding keys must be direct or scoped Param references")
        self.bindings = MappingProxyType(dict(parameters))
        self.parameter_bindings = MappingProxyType(dict(bindings if bindings is not None else {}))
        self.when = when

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> S_co: ...

    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Self | S_co:
        if instance is None:
            return self
        from .occurrence import child

        return child(instance, self)

    def ref(self, member: ValueRef[T] | ValueKey[T]) -> ValueRef[T]:
        return ScopedValueRef(cast("Subspace[Space]", self), member)

    def decision_ref(self, member: ValueRef[T]) -> DecisionRef[T]:
        return DecisionRef(cast("Subspace[Space]", self), member)

    def accepted(self, member: View[T] | ViewKey[T]) -> AcceptedViewRef[T]:
        return AcceptedViewRef(cast("Subspace[Space]", self), member)

    def at(self, member: ValueRef[T] | View[T] | ValueKey[T] | ViewKey[T]) -> LocatedRef[T]:
        """The member's value together with this node's identity (a ``Located``)."""
        return LocatedRef(cast("Subspace[Space]", self), member)


class SubspaceChoice(Declaration):
    def __init__(
        self,
        alternatives: Mapping[str, Subspace[Space]],
        *,
        exports: Sequence[ValueKey[object] | ViewKey[object]] = (),
        when: ValueRef[bool] | None = None,
    ) -> None:
        if not isinstance(alternatives, Mapping) or not alternatives:
            raise DefinitionError("a SubspaceChoice requires at least one alternative")
        for key, placement in alternatives.items():
            local_name(key, "case name")
            if not isinstance(placement, Subspace):
                raise DefinitionError(f"case {key}: expected a Subspace placement")
        self.alternatives = MappingProxyType(dict(alternatives))
        self.exports = tuple(exports)
        names: set[str] = set()
        for export in self.exports:
            if not isinstance(export, (ValueKey, ViewKey)):
                raise DefinitionError("a choice export requires a typed value or view key")
            if export.name in names:
                raise DefinitionError(f"duplicate choice export {export.name}")
            names.add(export.name)
        self.when = when

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> ChoiceView: ...

    def __get__(
        self, instance: Space | None, owner: type[object] | None = None
    ) -> Self | ChoiceView:
        if instance is None:
            return self
        from .occurrence import choice

        return choice(instance, self)

    def ref(self, member: ValueKey[T]) -> ValueRef[T]:
        return ScopedValueRef(self, member)

    def accepted(self, member: ViewKey[T]) -> AcceptedViewRef[T]:
        return AcceptedViewRef(self, member)

    def at(self, member: ValueKey[T] | ViewKey[T]) -> LocatedRef[T]:
        """The selected case's export, located at this choice's node."""
        return LocatedRef(self, member)

    def case(self) -> ChoiceCaseRef:
        """The selected case name, read-only; the selector owns the commitment."""
        if len(self.alternatives) < 2:
            raise DefinitionError("a singleton choice has no selected-case reference")
        return ChoiceCaseRef(self)


# -- Graph primitives ------------------------------------------------------------------


class ChoiceCaseRef(ScopedValueRef[str]):
    """Read-only reference to the case a structural choice selects."""

    def __init__(self, placement: SubspaceChoice) -> None:
        super().__init__(placement, ValueKey("case", str))

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> str: ...

    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Self | str:
        if instance is None:
            return self
        from .occurrence import read_value

        return read_value(instance, self)


class LocatedRef(ValueDecl[Located[T]], Generic[T]):
    """A member's value together with the identity of the node that holds it.

    ``placement`` is a child placement, a structural choice (its selected case),
    or None for the enclosing Space's own member (see ``located``).
    """

    def __init__(
        self,
        placement: Subspace[Space] | SubspaceChoice | None,
        member: ValueRef[T] | View[T] | ValueKey[T] | ViewKey[T],
    ) -> None:
        self.placement, self.member = placement, member
        self.semantics = cast("ValueSemantics[Located[T]]", LOCATED)


def located(member: ValueRef[T] | View[T]) -> LocatedRef[T]:
    """One of this Space's own members, located at the Space itself (node None)."""
    return LocatedRef(None, member)


class Present(ValueDecl[T], Generic[T]):
    """The value of whichever one of ``sources`` is present (applicable).

    Unresolved while any source is unresolved; refused if two are present;
    unsupplied (unresolved) if none is.
    """

    def __init__(self, *sources: ValueRef[T], semantics: ValueSemantics[T] | None = None) -> None:
        if not sources or any(not isinstance(source, ValueRef) for source in sources):
            raise DefinitionError("Present requires one or more value references")
        self.sources = sources
        self.semantics = semantics if semantics is not None else sources[0].semantics


class Bind(Declaration, Generic[T]):
    """An edge: supply a descendant's unbound formal from ``source``.

    Declared in the enclosing Space, after the nodes it joins, so edges may
    point forward or around a cycle. Several binds to one formal are resolved
    like ``Present``: at most one may be present.
    """

    def __init__(
        self,
        target: ValueRef[T],
        source: ValueRef[T] | T,
        *,
        when: ValueRef[bool] | None = None,
    ) -> None:
        if not isinstance(target, ScopedValueRef):
            raise DefinitionError("a Bind targets a descendant's formal through a scoped ref")
        self.target, self.source, self.when = target, source, when
        self.semantics = target.semantics


class Members(ValueDecl[tuple[Any, ...]], Generic[T]):
    """Every present child node exporting ``key``, as ``Located`` values in
    declaration order; used as an obligation, each member's acceptance counts."""

    def __init__(self, key: ViewKey[T]) -> None:
        if not isinstance(key, ViewKey):
            raise DefinitionError("Members requires a ViewKey")
        self.key = key
        self.semantics = cast("ValueSemantics[tuple[Any, ...]]", semantics_for(tuple))
