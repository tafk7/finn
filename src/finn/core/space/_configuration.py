# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed configuration objects and their bound value, view, and choice accessors.

Descriptors dispatch into these objects; evaluation and revision are implemented
by occurrence and change operations. Construction always prepares a Space model.
"""

# Local dispatch imports keep configuration types independent of their operations.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Generic, Self, TypeVar, cast, overload

from .declarations import (
    Constraint,
    ConstraintGroup,
    Decision,
    DecisionRef,
    Declaration,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    View,
    ViewKey,
)
from .edits import Change, ChangeRequest, ConfigurationResult
from .errors import DefinitionError
from .results import (
    ConstraintAssessment,
    DecisionState,
    QueryResult,
    ViewAssessment,
)

T = TypeVar("T")


class BoundView(Generic[T]):
    def __init__(self, instance: Space, declaration: View[T]) -> None:
        self.instance, self.declaration = instance, declaration

    def __call__(self) -> T:
        from .occurrence import read_value

        return read_value(self.instance, self.declaration)

    def get(self) -> T:
        return self()

    def inspect(self) -> ViewAssessment[T]:
        return self.instance.inspect(self.declaration)

    def query(self) -> QueryResult[T]:
        return self.instance.query(self.declaration)


class BoundValue(Generic[T]):
    """A typed value reference bound to one configuration snapshot."""

    def __init__(self, instance: Space, reference: ValueRef[T]) -> None:
        self.instance, self.reference = instance, reference

    def get(self) -> T:
        from .occurrence import read_value

        return read_value(self.instance, self.reference)

    def query(self) -> QueryResult[T]:
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
        from ._changes import change

        return change(self.instance, cast(Decision[T] | DecisionRef[T], self.reference), value)

    def clear(self) -> Change[T]:
        from ._changes import clear

        return clear(self.instance, cast(Decision[T] | DecisionRef[T], self.reference))


class ChoiceView:
    def __init__(self, instance: Space, declaration: SubspaceChoice) -> None:
        self.instance, self.declaration = instance, declaration

    @property
    def alternatives(self) -> tuple[str, ...]:
        from .occurrence import choice_alternatives

        return choice_alternatives(self)

    def select(self, case: str) -> ChoiceView:
        from ._changes import select

        return select(self, case)

    def alternative(self, case: str) -> Space:
        from .occurrence import alternative

        return alternative(self, case)


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
            existing = getattr(cls, name, None)
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


def _check_configuration_mutation(point: Space, name: str) -> None:
    from ._execution import driver_only
    from .occurrence import state

    driver_only("configuration mutation")
    current = getattr(point, "_state", None)
    scope_index = getattr(point, "_scope", None)
    if current is not None and type(scope_index) is int:
        scope = state(point).model.linked.scopes[scope_index]
        choice_names = {
            declaration.name
            for declaration in scope.choices
            if isinstance(declaration, SubspaceChoice)
        }
        if name in scope.named_members or name in scope.named_children or name in choice_names:
            raise AttributeError(f"{name} is an immutable configuration field; use with_choices()")


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
        _check_configuration_mutation(self, name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        _check_configuration_mutation(self, name)
        object.__delattr__(self, name)

    def query(self, value: ValueRef[T] | View[T]) -> QueryResult[T]:
        from .occurrence import query

        return query(self, value)

    @overload
    def inspect(self, view: View[T]) -> ViewAssessment[T]: ...

    @overload
    def inspect(self, view: Constraint | ConstraintGroup) -> ConstraintAssessment: ...

    def inspect(
        self, view: View[T] | Constraint | ConstraintGroup
    ) -> ViewAssessment[T] | ConstraintAssessment:
        from .occurrence import inspect

        return inspect(self, view)

    def view(self, reference: View[T]) -> BoundView[T]:
        from .occurrence import bind_view

        return bind_view(self, reference)

    @overload
    def field(self, reference: Decision[T] | DecisionRef[T]) -> BoundDecision[T]: ...

    @overload
    def field(self, reference: View[T]) -> BoundView[T]: ...

    @overload
    def field(self, reference: ValueRef[T]) -> BoundValue[T]: ...

    def field(
        self, reference: ValueRef[T] | View[T]
    ) -> BoundValue[T] | BoundDecision[T] | BoundView[T]:
        from .occurrence import bind_field

        return bind_field(self, reference)

    def with_choices(self, /, *changes: ChangeRequest, **choices: object) -> Self:
        from ._changes import with_choices

        return with_choices(self, *changes, **choices)

    def try_with_choices(
        self, /, *changes: ChangeRequest, **choices: object
    ) -> ConfigurationResult[Self]:
        from ._changes import try_with_choices

        return try_with_choices(self, *changes, **choices)

    @property
    def root(self) -> Space:
        from .occurrence import root

        return root(self)
