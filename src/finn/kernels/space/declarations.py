# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed declarations for the Space language.

Declarations describe structure. Runtime operations dispatch lazily so authoring
and collection do not depend on an evaluator or an existing compiled model.
"""

# Runtime dispatch is deliberately lazy to keep declaration collection independent.
# ruff: noqa: PLC0415

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar, cast, overload

from typing_extensions import Self

from .domains import Domain, finite
from .edits import Change, ChangeRequest, ConfigurationResult
from .errors import DefinitionError
from .results import (
    QueryResult,
    ConstraintAssessment,
    DecisionState,
    MissingInput,
    NotApplicable,
    ReadinessAssessment,
    ViewAssessment,
)
from .semantics import ValueSemantics, semantics_for

if TYPE_CHECKING:
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


class Dependency(Generic[T_co]):
    """A use-site mode; it does not create or change a value declaration."""

    def __init__(self, source: ValueRef[object], mode: Literal["optional", "result"]) -> None:
        self.source, self.mode = source, mode


def optional(source: ValueRef[T]) -> Dependency[T | MissingInput | NotApplicable]:
    return Dependency(source, "optional")


def full_result(source: ValueRef[T]) -> Dependency[QueryResult[T]]:
    return Dependency(source, "result")


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

    def _choice_type(self) -> T:
        """Static marker used by typed change construction; never evaluated."""

        raise RuntimeError("choice type markers are not runtime operations")


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


class Readiness(Declaration):
    def __init__(self, *requires: ValueRef[object] | Constraint | ConstraintGroup) -> None:
        self.requires = requires


class BoundView(Generic[T]):
    def __init__(self, instance: Space, declaration: View[T]) -> None:
        self.instance, self.declaration = instance, declaration

    def __call__(self) -> ViewAssessment[T]:
        return self.instance.assess(self.declaration)

    def assess(self) -> ViewAssessment[T]:
        return self()

    def result(self) -> QueryResult[T]:
        return self().accepted_result


class BoundValue(Generic[T]):
    """A typed value reference bound to one configuration snapshot."""

    def __init__(self, instance: Space, reference: ValueRef[T]) -> None:
        self.instance, self.reference = instance, reference

    def result(self) -> QueryResult[T]:
        return self.instance.query(self.reference)


class BoundDecision(BoundValue[T], Generic[T]):
    @property
    def state(self) -> QueryResult[DecisionState[T]]:
        from .occurrence import decision_state

        return decision_state(self.instance, cast(Decision[T] | DecisionRef[T], self.reference))

    def candidates(self) -> QueryResult[tuple[T, ...]] | None:
        from .occurrence import candidates

        return candidates(self.instance, cast(Decision[T] | DecisionRef[T], self.reference))

    def change(self, value: T) -> Change[T]:
        from .occurrence import change

        return change(self.instance, cast(Decision[T] | DecisionRef[T], self.reference), value)

    def clear(self) -> Change[T]:
        from .occurrence import clear

        return clear(self.instance, cast(Decision[T] | DecisionRef[T], self.reference))


class BoundViewField(Generic[T]):
    def __init__(self, instance: Space, reference: View[T]) -> None:
        self.instance, self.reference = instance, reference

    def assess(self) -> ViewAssessment[T]:
        return self.instance.assess(self.reference)

    def result(self) -> QueryResult[T]:
        return self.assess().accepted_result


class View(Declaration, Generic[T]):
    """One assessment declaration for either a value or an authored function."""

    def __init__(
        self,
        source: ValueRef[T],
        *,
        constraints: Sequence[Constraint | ConstraintGroup] = (),
        requires: Sequence[ValueRef[object] | Constraint | ConstraintGroup | Readiness] = (),
        when: ValueRef[bool] | None = None,
    ) -> None:
        self.source: ValueRef[T] | None = source
        self.function: Callable[..., T | QueryResult[T]] | None = None
        self.aliases: Mapping[str, object] = MappingProxyType({})
        self.semantics: ValueSemantics[T] | None = source.semantics
        self.constraints = tuple(constraints)
        self.requires = tuple(requires)
        self.when = when

    @classmethod
    def from_function(
        cls,
        function: Callable[..., T | QueryResult[T]],
        *,
        semantics: ValueSemantics[T] | None,
        aliases: Mapping[str, object],
        constraints: Sequence[Constraint | ConstraintGroup],
        requires: Sequence[ValueRef[object] | Constraint | ConstraintGroup | Readiness],
        when: ValueRef[bool] | None = None,
    ) -> View[T]:
        result = cls.__new__(cls)
        result.source = None
        result.function = function
        result.aliases = MappingProxyType(dict(aliases))
        result.semantics = semantics
        result.constraints, result.requires = tuple(constraints), tuple(requires)
        result.when = when
        return result

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> BoundView[T]: ...

    def __get__(
        self, instance: Space | None, owner: type[object] | None = None
    ) -> Self | BoundView[T]:
        return self if instance is None else BoundView(instance, self)


class _ViewDecorator:
    def __init__(
        self,
        aliases: Mapping[str, object],
        constraints: Sequence[Constraint | ConstraintGroup],
        requires: Sequence[ValueRef[object] | Constraint | ConstraintGroup | Readiness],
        when: ValueRef[bool] | None,
    ) -> None:
        self.aliases, self.constraints, self.requires = aliases, constraints, requires
        self.when = when

    def __call__(self, function: Callable[..., T]) -> View[T]:
        return View.from_function(
            function,
            semantics=None,
            aliases=self.aliases,
            constraints=self.constraints,
            requires=self.requires,
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
            requires=self.decorator.requires,
            when=self.decorator.when,
        )


@overload
def view(function: Callable[..., T], /) -> View[T]: ...


@overload
def view(
    *,
    semantics: ValueSemantics[T],
    when: ValueRef[bool] | None = None,
    constraints: Sequence[Constraint | ConstraintGroup] = (),
    requires: Sequence[ValueRef[object] | Constraint | ConstraintGroup | Readiness] = (),
    **aliases: object,
) -> _SemanticViewDecorator[T]: ...


@overload
def view(
    *,
    semantics: None = None,
    when: ValueRef[bool] | None = None,
    constraints: Sequence[Constraint | ConstraintGroup] = (),
    requires: Sequence[ValueRef[object] | Constraint | ConstraintGroup | Readiness] = (),
    **aliases: object,
) -> _ViewDecorator: ...


def view(
    function: Callable[..., object] | None = None,
    /,
    *,
    semantics: object = None,
    when: ValueRef[bool] | None = None,
    constraints: Sequence[Constraint | ConstraintGroup] = (),
    requires: Sequence[ValueRef[object] | Constraint | ConstraintGroup | Readiness] = (),
    **aliases: object,
) -> object:
    decorator = _ViewDecorator(aliases, constraints, requires, when)
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

    def _choice_type(self) -> T:
        """Static marker used by typed change construction; never evaluated."""

        raise RuntimeError("choice type markers are not runtime operations")


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

    def accepted(self, member: View[T] | ViewKey[T]) -> ValueRef[T]:
        return AcceptedViewRef(cast("Subspace[Space]", self), member)


class ChoiceView:
    def __init__(self, instance: Space, declaration: SubspaceChoice) -> None:
        self.instance, self.declaration = instance, declaration

    @property
    def alternatives(self) -> tuple[str, ...]:
        from .occurrence import choice_alternatives

        return choice_alternatives(self)

    def select(self, case: str) -> ChoiceView:
        from .occurrence import select

        return select(self, case)

    def alternative(self, case: str) -> Space:
        from .occurrence import alternative

        return alternative(self, case)


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

    def accepted(self, member: ViewKey[T]) -> ValueRef[T]:
        return AcceptedViewRef(self, member)


class SpaceMeta(type):
    """Construct configurations and protect prepared declaration structure."""

    def __call__(
        cls: SpaceMeta,
        parameters: Mapping[object, object] | None = None,
        /,
        **keyword_parameters: object,
    ) -> Space:
        from .compiler import compile_space

        return compile_space(cast(type[Space], cls)).bind(parameters, **keyword_parameters)

    def __setattr__(cls, name: str, value: object) -> None:
        if cls.__dict__.get("_space_definition_finalized", False) and not name.startswith(
            "_space_"
        ):
            existing = cls.__dict__.get(name)
            if (
                name == "exports"
                or isinstance(existing, Declaration)
                or isinstance(value, Declaration)
            ):
                raise DefinitionError(
                    f"{cls.__qualname__}.{name}: prepared declaration structure is finalized"
                )
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if cls.__dict__.get("_space_definition_finalized", False):
            existing = getattr(cls, name, None)
            if name == "exports" or isinstance(existing, Declaration):
                raise DefinitionError(
                    f"{cls.__qualname__}.{name}: prepared declaration structure is finalized"
                )
        super().__delattr__(name)


class Space(metaclass=SpaceMeta):
    """Authored family and scoped configuration type over an immutable runtime state."""

    _state: object
    _scope: int
    exports: ClassVar[Mapping[ValueKey[object] | ViewKey[object], Declaration]] = MappingProxyType(
        {}
    )

    def __init__(
        self,
        parameters: Mapping[object, object] | None = None,
        /,
        **keyword_parameters: object,
    ) -> None:
        """Typing signature only; SpaceMeta performs framework-owned construction."""

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_state", "_scope"} and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        current = getattr(self, "_state", None)
        scope_index = getattr(self, "_scope", None)
        if current is not None and type(scope_index) is int:
            from .occurrence import state

            scope = state(self).model.linked.scopes[scope_index]
            if name in scope.named_members or name in scope.named_children:
                raise AttributeError(
                    f"{name} is an immutable configuration field; use with_choices()"
                )
        object.__setattr__(self, name, value)

    def query(self, value: ValueRef[T] | View[T]) -> QueryResult[T]:
        from .occurrence import query

        return query(self, value)

    @overload
    def assess(self, view: View[T]) -> ViewAssessment[T]: ...

    @overload
    def assess(self, view: Constraint | ConstraintGroup) -> ConstraintAssessment: ...

    @overload
    def assess(self, view: Readiness) -> ReadinessAssessment: ...

    def assess(
        self, view: View[T] | Constraint | ConstraintGroup | Readiness
    ) -> ViewAssessment[T] | ConstraintAssessment | ReadinessAssessment:
        from .occurrence import assess

        return assess(self, view)

    @overload
    def field(self, reference: Decision[T] | DecisionRef[T]) -> BoundDecision[T]: ...

    @overload
    def field(self, reference: View[T]) -> BoundViewField[T]: ...

    @overload
    def field(self, reference: ValueRef[T]) -> BoundValue[T]: ...

    def field(
        self, reference: ValueRef[T] | View[T]
    ) -> BoundValue[T] | BoundDecision[T] | BoundViewField[T]:
        from .occurrence import bind_field

        return bind_field(self, reference)

    def with_choices(self, *changes: ChangeRequest, **choices: object) -> Self:
        from .occurrence import with_choices

        return with_choices(self, *changes, **choices)

    def try_with_choices(
        self, *changes: ChangeRequest, **choices: object
    ) -> ConfigurationResult[Self]:
        from .occurrence import try_with_choices

        return try_with_choices(self, *changes, **choices)

    @property
    def root(self) -> Space:
        from .occurrence import root

        return root(self)
