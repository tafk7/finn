# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The Space family type, its node declarations, and bound value and view accessors.

A ``Space`` object is one of two things. Calling a family, ``Room(area=12)``,
returns a *node declaration*: a template with bindings that reads its members
as symbolic references. ``configure(node)`` returns a *configuration*: a point
of the compiled space whose members read values. Both are typed as the family.
"""

# Local dispatch imports keep configuration types independent of their operations.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, overload

from typing_extensions import Self, dataclass_transform

from .declarations import (
    Constraint,
    ConstraintGroup,
    Decision,
    Declaration,
    Param,
    ValueRef,
    View,
    ViewKey,
    declared_path,
)
from .edits import Change, ChangeRequest, ConfigurationResult
from .errors import DefinitionError
from .results import (
    ConstraintAssessment,
    DecisionState,
    QueryResult,
    ViewAssessment,
)

if TYPE_CHECKING:
    from .references import DecisionHandle

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

    def __init__(self, instance: Space, reference: object) -> None:
        self.instance, self.reference = instance, reference

    def get(self) -> T:
        from .occurrence import read_value

        return read_value(self.instance, cast(ValueRef[T], self.reference))

    def query(self) -> QueryResult[T]:
        return cast(QueryResult[T], self.instance.query(cast(ValueRef[T], self.reference)))


class BoundDecision(BoundValue[T], Generic[T]):
    @property
    def state(self) -> QueryResult[DecisionState[T]]:
        from .occurrence import decision_state

        return cast(QueryResult[DecisionState[T]], decision_state(self.instance, self.reference))

    def candidates(self) -> QueryResult[tuple[T, ...]] | None:
        from .occurrence import candidates

        return cast(QueryResult[tuple[T, ...]] | None, candidates(self.instance, self.reference))

    def change(self, value: T) -> Change[T]:
        from ._changes import change

        return change(self.instance, self.reference, value)

    def clear(self) -> Change[T]:
        from ._changes import clear

        return cast(Change[T], clear(self.instance, self.reference))


class _When:
    """Typing surface of the reserved ``when=`` keyword of every family call."""

    def __get__(self, instance: object, owner: type[object] | None = None) -> _When:
        if instance is not None:
            raise AttributeError("when is the reserved guard keyword of a family call")
        return self

    def __set__(self, instance: object, value: ValueRef[bool] | bool | None) -> None:
        raise AttributeError("when is the reserved guard keyword of a family call")


def _when_field(*, default: None = None, kw_only: bool = True) -> Any:
    return _When()


@dataclass_transform(
    kw_only_default=True,
    eq_default=False,
    field_specifiers=(Param, _when_field),
)
class SpaceMeta(type):
    """Declare nodes, and protect prepared declaration structure.

    ``dataclass_transform`` types each family's call from its annotated
    formals (``area: Param[int] = Param(int)``) plus the ``when`` guard.
    """

    if not TYPE_CHECKING:

        def __call__(cls, *args: object, **keywords: object) -> Space:
            if args:
                raise DefinitionError(
                    f"{cls.__qualname__}: formals are bound by keyword, as in "
                    f"{cls.__qualname__}(name=value)"
                )
            from ._nodes import declare_node

            return declare_node(cls, keywords)

    def __setattr__(cls, name: str, value: object) -> None:
        if cls.__dict__.get("_space_definition_finalized", False) and not name.startswith(
            "_space_"
        ):
            existing = cls.__dict__.get(name, getattr(cls, name, None))
            if name == "exports" or _structural(existing) or _structural(value):
                raise DefinitionError(
                    f"{cls.__qualname__}.{name}: prepared declaration structure is finalized"
                )
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if cls.__dict__.get("_space_definition_finalized", False):
            existing = cls.__dict__.get(name)
            if name == "exports" or _structural(existing):
                raise DefinitionError(
                    f"{cls.__qualname__}.{name}: prepared declaration structure is finalized"
                )
        super().__delattr__(name)


def _structural(value: object) -> bool:
    from ._nodes import NodeChoice

    return isinstance(value, (Declaration, NodeChoice)) or (
        isinstance(value, Space) and declared_path(value) is not None
    )


def _check_configuration_mutation(point: Space, name: str) -> None:
    from ._execution import driver_only
    from .occurrence import state

    if declared_path(point) is not None:
        raise AttributeError(f"{name}: a node declaration is immutable")
    driver_only("configuration mutation")
    current = getattr(point, "_state", None)
    scope_index = getattr(point, "_scope", None)
    if current is not None and type(scope_index) is int:
        scope = state(point).model.linked.scopes[scope_index]
        if name in scope.named_members or name in scope.named_children:
            raise AttributeError(f"{name} is an immutable configuration field; use with_choices()")


class Space(metaclass=SpaceMeta):
    """A family of design spaces: calling it declares a node, ``configure`` compiles one."""

    when: _When = _when_field(default=None, kw_only=True)
    _state: ClassVar[object]
    _scope: ClassVar[int]
    exports: ClassVar[Mapping[ViewKey[object], Declaration]] = MappingProxyType({})

    def __set_name__(self, owner: type[object], name: str) -> None:
        path = declared_path(self)
        if path is None or len(path) != 1:
            raise DefinitionError(
                f"{owner.__qualname__}.{name}: only a node declaration can be placed in a family"
            )
        from ._nodes import NodeDecl, place_in_class

        place_in_class(cast(NodeDecl, path[0]), owner, name)

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[object] | None = None) -> Self: ...

    def __get__(self, instance: object, owner: type[object] | None = None) -> Self:
        if instance is None:
            return self
        path = declared_path(self)
        assert path is not None and len(path) == 1
        outer = declared_path(instance)
        if outer is not None:
            from ._nodes import path_proxy

            return cast(Self, path_proxy(type(self), (*outer, path[0])))
        from .occurrence import child

        return cast(Self, child(cast(Space, instance), path[0]))

    def __setattr__(self, name: str, value: object) -> None:
        _check_configuration_mutation(self, name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        _check_configuration_mutation(self, name)
        object.__delattr__(self, name)

    def __repr__(self) -> str:
        path = declared_path(self)
        if path is not None:
            from ._nodes import NodeDecl

            last = path[-1]
            if len(path) == 1 and isinstance(last, NodeDecl):
                return f"<{last.describe()}>"
            names = ".".join(str(item.name) for item in path)
            return f"<{type(self).__qualname__} node path {names}>"
        return object.__repr__(self)

    @overload
    def query(self, value: ValueRef[T] | View[T] | BoundView[T]) -> QueryResult[T]: ...

    @overload
    def query(self, value: T) -> QueryResult[T]: ...

    def query(self, value: object) -> QueryResult[Any]:
        """Query a member or a reference (``House.kitchen.finish``) of this configuration."""
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
    def field(self, reference: Decision[T] | DecisionHandle[T]) -> BoundDecision[T]: ...

    @overload
    def field(self, reference: View[T]) -> BoundView[T]: ...

    @overload
    def field(self, reference: ValueRef[T]) -> BoundValue[T]: ...

    def field(
        self, reference: ValueRef[T] | View[T]
    ) -> BoundValue[T] | BoundDecision[T] | BoundView[T]:
        from .occurrence import bind_field

        return bind_field(self, reference)

    def with_choices(
        self, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
    ) -> Self:
        """Commit choices atomically: ``{House.kitchen.finish: 2}``, Change objects,
        or keywords for this configuration's own decisions."""
        from ._changes import with_choices

        return with_choices(self, *changes, **choices)

    def try_with_choices(
        self, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
    ) -> ConfigurationResult[Self]:
        from ._changes import try_with_choices

        return try_with_choices(self, *changes, **choices)

    @property
    def root(self) -> Space:
        from .occurrence import root

        return root(self)
