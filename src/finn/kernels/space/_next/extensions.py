# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Build reusable declaration bundles as ordinary named child Space templates.

The builder owns only authoring data. Finishing creates one subclass through
normal Python class construction; placement uses the same linker and evaluator
as a handwritten Subspace. Parent classes and compiled records are never edited.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, cast, overload

from .collection import collect_placement, collect_space
from .declarations import (
    Const,
    Constraint,
    ConstraintGroup,
    Declaration,
    Decision,
    Derived,
    Param,
    Readiness,
    Space,
    Subspace,
    ValueKey,
    ValueRef,
    View,
    ViewKey,
    local_name,
)
from .domains import Domain
from .errors import DefinitionError
from .results import Answer
from .semantics import ValueSemantics

T = TypeVar("T")
S = TypeVar("S", bound=Space)
D = TypeVar("D", bound=Declaration)


class _ExportRecorder(Protocol):
    def _record_export(
        self, key: ValueKey[object] | ViewKey[object], source: Declaration
    ) -> None: ...


class ValueExport(Generic[T]):
    """A key fixes T before its value is supplied, preserving type checking."""

    def __init__(self, builder: _ExportRecorder, key: ValueKey[T]) -> None:
        self._builder, self._key = builder, key

    def value(self, source: ValueRef[T]) -> None:
        self._builder._record_export(self._key, source)


class ViewExport(Generic[T]):
    def __init__(self, builder: _ExportRecorder, key: ViewKey[T]) -> None:
        self._builder, self._key = builder, key

    def view(self, source: View[T]) -> None:
        self._builder._record_export(self._key, source)


class ScopeBuilder(Generic[S]):
    """A mutable authoring session which seals into one reusable child template.

    Base-class fields keep their Python types. New fields are accessed through
    their returned declarations or typed export keys. Bind external suppliers
    explicitly; callbacks receive named dependencies and no implicit instance.
    """

    def __init__(self, base: type[S], *, name: str | None = None) -> None:
        if not isinstance(base, type) or not issubclass(base, Space):
            raise DefinitionError("ScopeBuilder requires a Space base class")
        self._base = base
        self._name = local_name(
            name if name is not None else f"{base.__name__}Bundle", "extension template name"
        )
        effective = collect_space(base)
        self._base_members = {key: record.declaration for key, record in effective.members.items()}
        self._base_names = {key for cls in base.__mro__ for key in vars(cls)}
        self._names_by_identity = {id(source): key for source, key in effective.aliases.items()}
        self._members: dict[str, Declaration] = {}
        self._exports = dict(effective.exports)
        self._bindings: dict[str, object] = {}
        self._sealed = False
        self._template: type[S] | None = None
        self._failure: Exception | None = None

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _open(self) -> None:
        if self._sealed:
            raise DefinitionError(f"{self._name}: extension construction is sealed")

    def add(self, name: str, declaration: D) -> D:
        self._open()
        local_name(name, "extension member name")
        if name in self._members or name in self._base_names:
            raise DefinitionError(f"{self._name}.{name}: duplicate or inherited member name")
        if not isinstance(declaration, Declaration):
            raise DefinitionError(f"{self._name}.{name}: expected an authoring declaration")
        if declaration.owner is not None:
            raise DefinitionError(
                f"{self._name}.{name}: declaration already belongs to "
                f"{declaration.owner.__qualname__}.{declaration.name}; "
                "inherited fields need no re-add"
            )
        if id(declaration) in self._names_by_identity:
            raise DefinitionError(
                f"{self._name}.{name}: this declaration already has a member name"
            )
        self._members[name] = declaration
        self._names_by_identity[id(declaration)] = name
        return declaration

    def param(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        required: bool = True,
        semantics: ValueSemantics[T] | None = None,
    ) -> Param[T]:
        self._open()
        return self.add(name, Param(value_type, required=required, semantics=semantics))

    def const(self, name: str, value: T, *, semantics: ValueSemantics[T] | None = None) -> Const[T]:
        self._open()
        return self.add(name, Const(value, semantics=semantics))

    def decision(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: Domain[T] | None = None,
        values: Iterable[T] | None = None,
        semantics: ValueSemantics[T] | None = None,
        when: ValueRef[bool] | None = None,
    ) -> Decision[T]:
        self._open()
        return self.add(
            name, Decision(value_type, domain=domain, values=values, semantics=semantics, when=when)
        )

    @overload
    def derived(
        self,
        name: str,
        function: Callable[..., T],
        *,
        semantics: None = None,
        when: ValueRef[bool] | None = None,
        **aliases: object,
    ) -> Derived[T]: ...

    @overload
    def derived(
        self,
        name: str,
        function: Callable[..., T | Answer[T]],
        *,
        semantics: ValueSemantics[T],
        when: ValueRef[bool] | None = None,
        **aliases: object,
    ) -> Derived[T]: ...

    def derived(
        self,
        name: str,
        function: Callable[..., object],
        *,
        semantics: object = None,
        when: ValueRef[bool] | None = None,
        **aliases: object,
    ) -> object:
        self._open()
        if semantics is not None and not isinstance(semantics, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        return self.add(name, Derived(function, semantics=semantics, aliases=aliases, when=when))

    def constraint(
        self,
        name: str,
        function: Callable[..., bool | Answer[bool]],
        *,
        when: ValueRef[bool] | None = None,
        **aliases: object,
    ) -> Constraint:
        self._open()
        return self.add(name, Constraint(function, aliases=aliases, when=when))

    def view(
        self,
        name: str,
        source: ValueRef[T],
        *,
        constraints: Sequence[Constraint | ConstraintGroup] = (),
        requires: Sequence[ValueRef[object] | Constraint | ConstraintGroup | Readiness] = (),
        when: ValueRef[bool] | None = None,
    ) -> View[T]:
        self._open()
        return self.add(name, View(source, constraints=constraints, requires=requires, when=when))

    @overload
    def export(self, key: ValueKey[T]) -> ValueExport[T]: ...

    @overload
    def export(self, key: ViewKey[T]) -> ViewExport[T]: ...

    def export(self, key: ValueKey[T] | ViewKey[T]) -> ValueExport[T] | ViewExport[T]:
        self._open()
        if isinstance(key, ValueKey):
            return ValueExport(self, key)
        if isinstance(key, ViewKey):
            return ViewExport(self, key)
        raise DefinitionError("extension exports require typed ValueKey or ViewKey objects")

    def _record_export(self, key: ValueKey[object] | ViewKey[object], source: Declaration) -> None:
        self._open()
        if id(source) not in self._names_by_identity:
            raise DefinitionError(
                f"{self._name}: export {key.name!r} is not a local or inherited member"
            )
        if any(existing.name == key.name for existing in self._exports):
            raise DefinitionError(f"{self._name}: duplicate export name {key.name!r}")
        if (isinstance(key, ValueKey) and not isinstance(source, ValueRef)) or (
            isinstance(key, ViewKey) and not isinstance(source, View)
        ):
            raise DefinitionError(f"{self._name}: export {key.name!r} has the wrong kind")
        self._exports[key] = source

    def bind(self, parameter: Param[T], supplier: T | ValueRef[T]) -> None:
        self._open()
        if not isinstance(parameter, Param):
            raise DefinitionError("extension bindings require a Param declaration")
        name = self._names_by_identity.get(id(parameter))
        if name is None:
            raise DefinitionError(
                f"{self._name}: binding parameter is not a local or inherited member"
            )
        if name in self._bindings:
            raise DefinitionError(f"{self._name}.{name}: parameter is already bound")
        assert parameter.semantics is not None
        if isinstance(supplier, ValueRef):
            if supplier.semantics is not None and not parameter.semantics.is_compatible_with(
                cast(ValueSemantics[object], supplier.semantics)
            ):
                raise DefinitionError(f"{self._name}.{name}: incompatible supplier value semantics")
            self._bindings[name] = supplier
        else:
            try:
                self._bindings[name] = parameter.semantics.freeze(supplier)
            except Exception as cause:
                raise DefinitionError(
                    f"{self._name}.{name}: invalid literal binding: {cause}"
                ) from cause

    def finish(self) -> type[S]:
        """Create and validate one child class, then reject further authoring mutations."""
        if self._template is not None:
            return self._template
        if self._failure is not None:
            raise DefinitionError(
                f"{self._name}: extension construction previously failed"
            ) from self._failure
        self._sealed = True
        namespace: dict[str, object] = {
            "__module__": self._base.__module__,
            "exports": MappingProxyType(dict(self._exports)),
            **self._members,
        }
        try:
            template = cast(type[S], type(self._name, (self._base,), namespace))
            collect_space(template)
        except Exception as cause:
            self._failure = cause
            if isinstance(cause, DefinitionError):
                raise
            raise DefinitionError(
                f"{self._name}: extension construction failed: {cause}"
            ) from cause
        self._template = template
        return template

    def place(self, *, when: ValueRef[bool] | None = None) -> Subspace[S]:
        """Make an independent placement with the complete explicitly supplied bindings."""
        template = self.finish()
        declarations = {**self._base_members, **self._members}
        bindings: dict[str, object] = {}
        for name, supplier in self._bindings.items():
            parameter = declarations[name]
            assert isinstance(parameter, Param) and parameter.semantics is not None
            try:
                bindings[name] = (
                    supplier
                    if isinstance(supplier, ValueRef)
                    else parameter.semantics.freeze(supplier)
                )
            except Exception as cause:
                raise DefinitionError(
                    f"{self._name}.{name}: invalid literal binding: {cause}"
                ) from cause
        placement = Subspace(template, when=when, **bindings)
        collect_placement(placement)
        return placement


__all__ = ["ScopeBuilder", "ValueExport", "ViewExport"]
