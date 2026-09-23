# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed declarations for the candidate Space language.

Declarations describe structure. Runtime operations dispatch lazily so authoring
and collection do not depend on an evaluator or an existing compiled model.
"""

# Runtime dispatch is deliberately lazy to keep declaration collection independent.
# ruff: noqa: PLC0415

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import re
from types import MappingProxyType
from typing import ClassVar, Generic, Literal, TypeVar, cast, overload

from typing_extensions import Self

from .domains import Domain, finite
from .edits import Edit, EditRequest, RefinementReport
from .errors import DefinitionError
from .results import (
    Answer,
    ConstraintAssessment,
    DecisionState,
    MissingInput,
    NotApplicable,
    ReadinessAssessment,
    ViewAssessment,
)
from .semantics import ValueSemantics, semantics_for

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

    def __init__(self, source: ValueRef[object], mode: Literal["optional", "answer"]) -> None:
        self.source, self.mode = source, mode


def optional(source: ValueRef[T]) -> Dependency[T | MissingInput | NotApplicable]:
    return Dependency(source, "optional")


def full_answer(source: ValueRef[T]) -> Dependency[Answer[T]]:
    return Dependency(source, "answer")


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
        function: Callable[..., T | Answer[T]],
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

    def __call__(self, function: Callable[..., T | Answer[T]]) -> Derived[T]:
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
        function: Callable[..., bool | Answer[bool]],
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

    def __call__(self, function: Callable[..., bool | Answer[bool]]) -> Constraint:
        return Constraint(function, aliases=self.aliases, when=self.when)


@overload
def constraint(function: Callable[..., bool | Answer[bool]], /) -> Constraint: ...


@overload
def constraint(
    *, when: ValueRef[bool] | None = None, **aliases: object
) -> _ConstraintDecorator: ...


def constraint(
    function: Callable[..., bool | Answer[bool]] | None = None,
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
    def __init__(self, occurrence: Space, declaration: View[T]) -> None:
        self.occurrence, self.declaration = occurrence, declaration

    def __call__(self) -> ViewAssessment[T]:
        return self.occurrence.assess(self.declaration)


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
        self.function: Callable[..., T | Answer[T]] | None = None
        self.aliases: Mapping[str, object] = MappingProxyType({})
        self.semantics: ValueSemantics[T] | None = source.semantics
        self.constraints = tuple(constraints)
        self.requires = tuple(requires)
        self.when = when

    @classmethod
    def from_function(
        cls,
        function: Callable[..., T | Answer[T]],
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

    def __call__(self, function: Callable[..., T | Answer[T]]) -> View[T]:
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


class AcceptedViewRef(ValueRef[T], Generic[T]):
    def __init__(
        self, placement: Subspace[Space] | SubspaceChoice, member: View[T] | ViewKey[T]
    ) -> None:
        self.placement, self.member = placement, member
        self.semantics = member.semantics


class Subspace(Declaration, Generic[S_co]):
    def __init__(
        self, space_type: type[S_co], *, when: ValueRef[bool] | None = None, **bindings: object
    ) -> None:
        if not isinstance(space_type, type) or not issubclass(space_type, Space):
            raise DefinitionError("a Subspace requires a Space subclass")
        self.space_type = space_type
        self.bindings = MappingProxyType(dict(bindings))
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
    def __init__(self, occurrence: Space, declaration: SubspaceChoice) -> None:
        self.occurrence, self.declaration = occurrence, declaration

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


class Space:
    """Authored family and scoped occurrence type over an immutable runtime state."""

    _state: object
    _scope: int
    exports: ClassVar[Mapping[ValueKey[object] | ViewKey[object], Declaration]] = MappingProxyType(
        {}
    )

    @classmethod
    def start(cls, parameters: Mapping[object, object] | None = None) -> Self:
        from .compiler import compile_space

        return compile_space(cls).start(parameters if parameters is not None else {})

    def answer(self, value: ValueRef[T]) -> Answer[T]:
        from .occurrence import answer

        return answer(self, value)

    def assign(self, decision: Decision[T] | DecisionRef[T], value: T) -> Self:
        from .occurrence import assign

        return assign(self, decision, value)

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

    def decision_state(self, decision: Decision[T] | DecisionRef[T]) -> Answer[DecisionState[T]]:
        from .occurrence import decision_state

        return decision_state(self, decision)

    def candidates(self, decision: Decision[T] | DecisionRef[T]) -> Answer[tuple[T, ...]] | None:
        from .occurrence import candidates

        return candidates(self, decision)

    def edit(self, decision: Decision[T] | DecisionRef[T], value: T) -> Edit[T]:
        from .occurrence import edit

        return edit(self, decision, value)

    def refine(self, *edits: EditRequest) -> RefinementReport[Self]:
        from .occurrence import refine

        return refine(self, *edits)

    @property
    def root(self) -> Space:
        from .occurrence import root

        return root(self)
