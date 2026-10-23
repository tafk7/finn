# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The Space base class, its node declarations, and bound value accessors.

A ``Space`` object is one of two things. Calling a Space class, ``Room(area=12)``,
returns a *declaration*: a template with bindings that reads its members as
symbolic references. ``design_space(node)`` returns a design space read at its
root (every choice open; ``occurrence.held``: the design space value and the scope it
holds), whose members read values, as does every Space object a change or a child read
returns. Both are typed as the Space class. Every value-like member follows this rule,
views included: a view reads as its accepted value, and its assessment is
``point.inspect(Room.view)``.
"""

# Local dispatch imports keep configuration types independent of their operations.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    cast,
    dataclass_transform,
    overload,
)

from .declarations import (
    Constraint,
    Decision,
    Declaration,
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

_OWNER = "space.configuration"

T = TypeVar("T")


class BoundValue(Generic[T]):
    """A typed value reference bound to one configuration snapshot.

    Any member that reads as a value binds as one: a formal, a derived value,
    a view (its accepted value) or a reference into a child.
    """

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
        """The choice's listed values (its domain's enumeration, each value its membership
        accepts), in their listed order (``None``: known by membership only), under its
        guard."""
        from .occurrence import candidates

        return cast(QueryResult[tuple[T, ...]] | None, candidates(self.instance, self.reference))

    def change(self, value: T) -> Change[T]:
        from .occurrence import change

        return change(self.instance, self.reference, value)

    def clear(self) -> Change[T]:
        from .occurrence import clear

        return cast(Change[T], clear(self.instance, self.reference))


class _When:
    """Typing surface of the reserved ``when=`` keyword of every call on a Space class."""

    def __get__(self, instance: object, owner: type[object] | None = None) -> _When:
        if instance is not None:
            raise AttributeError("when is the reserved guard keyword of a call on a Space class")
        return self

    def __set__(self, instance: object, value: ValueRef[bool] | bool | None) -> None:
        raise AttributeError("when is the reserved guard keyword of a call on a Space class")


def _when_field(*, default: None = None, kw_only: bool = True) -> Any:
    return _When()


@dataclass_transform(
    kw_only_default=True,
    eq_default=False,
    field_specifiers=(_when_field,),
)
class SpaceMeta(type):
    """Declare nodes, and protect prepared declaration structure.

    ``dataclass_transform`` types each Space class's call from its annotated
    formals and Decisions (``area: int = Param()``) plus the ``when`` guard.
    Param and Decision are deliberately not field specifiers: their call is
    then a default, so every member is optional at the call and a bare
    ``Room()`` type-checks (a later assignment may supply it), while each
    keyword and assignment is typed by the annotation.
    """

    if not TYPE_CHECKING:

        def __call__(cls, *args: object, **keywords: object) -> Space:
            if args:
                raise DefinitionError.broken(
                    "formals-by-keyword",
                    _OWNER,
                    f"{cls.__qualname__}: formals are bound by keyword, as in "
                    f"{cls.__qualname__}(name=value)",
                    declaration=cls.__qualname__,
                )
            from ._nodes import declare_node

            return declare_node(cls, keywords)

    def __setattr__(cls, name: str, value: object) -> None:
        if cls.__dict__.get("_space_definition_finalized", False) and not name.startswith(
            "_space_"
        ):
            existing = cls.__dict__.get(name, getattr(cls, name, None))
            if name == "exports" or _structural(existing) or _structural(value):
                raise DefinitionError.broken(
                    "structure-finalized",
                    _OWNER,
                    f"{cls.__qualname__}.{name}: prepared declaration structure is finalized",
                    declaration=f"{cls.__qualname__}.{name}",
                )
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if cls.__dict__.get("_space_definition_finalized", False):
            existing = cls.__dict__.get(name)
            if name == "exports" or _structural(existing):
                raise DefinitionError.broken(
                    "structure-finalized",
                    _OWNER,
                    f"{cls.__qualname__}.{name}: prepared declaration structure is finalized",
                    declaration=f"{cls.__qualname__}.{name}",
                )
        super().__delattr__(name)


def _structural(value: object) -> bool:
    from ._nodes import NodeChoice

    return isinstance(value, (Declaration, NodeChoice)) or (
        isinstance(value, Space) and declared_path(value) is not None
    )


def _check_configuration_mutation(point: Space, name: str) -> None:
    from ._failure import driver_only
    from .occurrence import model_of

    driver_only("configuration mutation")
    current = getattr(point, "_held", None)
    scope_index = getattr(point, "_scope", None)
    if current is not None and type(scope_index) is int:
        scope = model_of(point).linked.scopes[scope_index]
        if name in scope.named_members or name in scope.named_children:
            raise AttributeError(f"{name} is an immutable configuration field; use with_choices()")


class Space(metaclass=SpaceMeta):
    """The base of every Space class.

    A Space class describes a design space: calling it declares a node, and
    ``design_space`` opens one.
    """

    when: _When = _when_field(default=None, kw_only=True)
    # A design space read at a scope (``_running.surface.At``), or a method's reader.
    _held: ClassVar[object]
    _scope: ClassVar[int]
    # A key maps to one view or derived value (a derived value is typed as its value).
    exports: ClassVar[Mapping[ViewKey[Any], object]] = MappingProxyType({})

    def __set_name__(self, owner: type[object], name: str) -> None:
        path = declared_path(self)
        if path is None or len(path) != 1:
            raise DefinitionError.broken(
                "space-holds-non-node",
                _OWNER,
                f"{owner.__qualname__}.{name}: only a node declaration can be placed in a Space "
                "class",
                declaration=f"{owner.__qualname__}.{name}",
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

    if not TYPE_CHECKING:
        # Hidden from type checkers: a declared __setattr__ would make mypy accept
        # assignment to any attribute. Members are typed by their annotations instead.

        def __setattr__(self, name: str, value: object) -> None:
            if declared_path(self) is not None:
                # A declaration: supply one of its formals (``hall.area = kitchen.area``).
                from ._nodes import assign

                assign(self, name, value)
                return
            _check_configuration_mutation(self, name)
            object.__setattr__(self, name, value)

        def __delattr__(self, name: str) -> None:
            if declared_path(self) is not None:
                raise AttributeError(f"{name}: an assignment cannot be removed")
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
    def query(self, value: ValueRef[T] | View[T]) -> QueryResult[T]: ...

    @overload
    def query(self, value: T) -> QueryResult[T]: ...

    def query(self, value: object) -> QueryResult[Any]:
        """Query a member or a reference (``House.kitchen.finish``) of this configuration.

        The answer is what reading it would give, as a result rather than a
        raise: a view (``House.total``) answers its accepted result. A node (a
        child or a candidate handle) answers its configuration: inapplicable when
        the node is absent, unresolved while its presence is undecided.
        """
        from .occurrence import query

        return query(self, value)

    def present(self, node: object) -> bool:
        """Whether a node (a child or a candidate) is present, or a value input is supplied.

        Read like a value: undecided presence raises ``ValueUnavailableError``
        (inside a method it halts the method as unresolved). A value input (a ``Param``) is
        present when it is supplied: given a literal, or bound to a source that
        applies (its guards hold and the choices it is read through select it);
        an optional input omitted at start, or one that no present source
        supplies, is not. The value is not evaluated: a source that refuses is
        present, and its refusal surfaces where the value is read, with its
        findings. A source whose presence is still undecided raises.
        """
        from .occurrence import present

        return present(self, node)

    @overload
    def inspect(self, view: View[T]) -> ViewAssessment[T]: ...

    @overload
    def inspect(  # type: ignore[overload-overlap]
        self, view: Constraint
    ) -> ConstraintAssessment: ...

    @overload
    def inspect(self, view: T) -> ViewAssessment[T]: ...

    def inspect(self, view: object) -> ViewAssessment[Any] | ConstraintAssessment:
        """The assessment of a view or a constraint of this configuration.

        ``point.inspect(House.total)`` is the view's ``ViewAssessment``: its raw
        output, its Space's validity, readiness and accepted result. A view of a
        child is inspected on the child's configuration,
        ``point.kitchen.inspect(Room.cost)`` (typed by the view declaration),
        or through a path, ``point.inspect(House.kitchen.cost)`` (a reference
        typed as its value, so a non-view member is refused only at runtime).
        """
        from .occurrence import inspect

        return inspect(self, cast("View[Any] | Constraint", view))

    @overload
    def field(self, reference: Decision[T] | DecisionHandle[T]) -> BoundDecision[T]: ...

    @overload
    def field(self, reference: ValueRef[T] | View[T]) -> BoundValue[T]: ...  # type: ignore[overload-overlap]

    @overload
    def field(self, reference: Space | None) -> BoundDecision[str]: ...

    @overload
    def field(self, reference: T) -> BoundDecision[T]: ...

    def field(self, reference: object) -> BoundValue[Any] | BoundDecision[Any]:
        """A bound accessor. A view binds as a value accessor of its accepted
        value, like a derived member. A member typed as its value binds as a
        decision accessor: its decision operations refuse a member that is not one."""
        from .occurrence import bind_field

        return bind_field(self, cast("ValueRef[Any]", reference))

    def with_choices(
        self, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
    ) -> Self:
        """Commit choices atomically: ``{House.kitchen.finish: 2}``, Change objects,
        or keywords for this configuration's own decisions. Refused (``ConfigurationError``,
        its report as ``try_with_choices`` answers it) where a value is outside its domain,
        or the change would make a rule newly false, naming the rule and the stored choices
        standing in its way."""
        from .occurrence import with_choices

        return with_choices(self, *changes, **choices)

    def try_with_choices(
        self, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
    ) -> ConfigurationResult[Self]:
        from .occurrence import try_with_choices

        return try_with_choices(self, *changes, **choices)

    def with_facts(
        self, facts: Mapping[Any, object], /, *changes: ChangeRequest | Mapping[Any, object]
    ) -> Self:
        """A new design space from other values for root formals (``{House.budget:
        120}``), keeping every evaluation that read none of them and the choices that still
        fall within their domains, then ``changes``, a change of choices on it. A change of
        facts is never refused: a stored choice outside its new domain is dropped, and a
        constraint the new facts make false reads refused (``try_with_facts`` reports
        both). Raises ``ConfigurationError`` only where the batch's choices are refused,
        by the rules of ``with_choices`` judged against the new design space."""
        from .occurrence import with_facts

        return with_facts(self, facts, *changes)

    def try_with_facts(
        self, facts: Mapping[Any, object], /, *changes: ChangeRequest | Mapping[Any, object]
    ) -> ConfigurationResult[Self]:
        from .occurrence import try_with_facts

        return try_with_facts(self, facts, *changes)

    @property
    def root(self) -> Space:
        from .occurrence import root

        return root(self)
