# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile node declarations into immutable linked models, and configure them.

``configure(node)`` is the one step from a declaration to a configuration.
A root whose bindings are all plain values reuses its family's model and
binds the values as runtime inputs; a root that supplies structure (a node, a
reference, or a fresh Decision) is compiled for that declaration.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar, cast

from ._configuration import Space
from ._linker import link_space
from ._nodes import FamilyFormal, NodeDecl, family_formals, node_record
from .declarations import MISSING, UNSUPPLIED, Param, at
from .errors import DefinitionError, RequestError
from .ir import LinkedModel
from .references import ValueHandle, resolve_decision, resolve_reference

S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class SpaceModel(Generic[S]):
    """A compiled definition: frozen indexes, no values, assignments or caches."""

    space_type: type[S]
    linked: LinkedModel

    def resolve(self, scope: int, reference: object) -> int:
        """Resolve a declaration in this exact scope without consulting its class."""

        if isinstance(reference, ValueHandle):
            if type(scope) is not int or not 0 <= scope < len(self.linked.scopes):
                raise RequestError("reference scope does not belong to this model")
            return reference._resolve(self.linked)
        return resolve_reference(
            self.linked.nodes, self.linked.scopes, self.linked.choices, scope, reference
        )

    def decision(self, scope: int, reference: object) -> int:
        """The owning decision ``reference`` may edit in this scope."""

        if isinstance(reference, ValueHandle):
            index = self.resolve(scope, reference)
            if self.linked.nodes[index].kind != "decision":
                raise RequestError("parameter aliases are not independently editable")
            return index
        return resolve_decision(
            self.linked.nodes,
            self.linked.scopes,
            self.linked.choices,
            scope,
            reference,
            self.linked.editable_aliases,
        )


_PREPARATION_LOCK = RLock()


def _constructor_families(space_type: type[Space]) -> tuple[type[object], ...]:
    """Return every class whose constructor protocol affects this family."""

    result: list[type[object]] = []
    for base in space_type.__mro__:
        if base is Space:
            break
        result.append(base)
    return tuple(result)


def _validate_constructors(space_types: tuple[type[Space], ...]) -> None:
    checked: set[type[object]] = set()
    for space_type in space_types:
        for base in _constructor_families(space_type):
            if base in checked:
                continue
            checked.add(base)
            if "__init__" in base.__dict__:
                raise DefinitionError(
                    f"{base.__qualname__}: custom instance __init__ is unsupported; "
                    "use declarations and ordinary helper methods"
                )
            if "__new__" in base.__dict__:
                raise DefinitionError(
                    f"{base.__qualname__}: custom instance __new__ is unsupported"
                )


def _definition_families(space_types: tuple[type[Space], ...]) -> tuple[type[Space], ...]:
    """Include concrete families and Space bases contributing effective declarations."""

    result: list[type[Space]] = []
    seen: set[type[Space]] = set()
    for space_type in space_types:
        for base in space_type.__mro__:
            if base is Space:
                break
            if issubclass(base, Space) and base not in seen:
                seen.add(base)
                result.append(base)
    return tuple(result)


def _publish(space_type: type[S], linked: LinkedModel) -> SpaceModel[S]:
    scope_types = tuple(dict.fromkeys(scope.space_type for scope in linked.scopes))
    _validate_constructors(scope_types)
    model = SpaceModel(space_type, linked)
    # Publication happens only after the complete definition linked successfully.
    for family in _definition_families(scope_types):
        type.__setattr__(family, "_space_definition_finalized", True)
    return model


def compile_space(space_type: type[S]) -> SpaceModel[S]:
    """The canonical model of a family whose formals are all runtime inputs."""

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError("compile_space requires a Space subclass")
    with _PREPARATION_LOCK:
        cached = space_type.__dict__.get("_space_prepared_model")
        if cached is not None:
            if not isinstance(cached, SpaceModel) or cached.space_type is not space_type:
                raise DefinitionError("invalid prepared-model cache on Space family")
            return cached
        model = _publish(space_type, link_space(space_type))
        type.__setattr__(space_type, "_space_prepared_model", model)
        return model


def _structural(record: NodeDecl) -> bool:
    from ._bindings import placement_plan

    plan = placement_plan(record.family, record, root=True)
    return bool(record.open) or any(item.kind != "parameter" for item in plan.bindings.values())


def compile_node(record: NodeDecl) -> SpaceModel[Space]:
    """The model of one root declaration; plain values stay runtime inputs."""

    with _PREPARATION_LOCK:
        if not _structural(record):
            return cast(SpaceModel[Space], compile_space(record.family))
        cached = record.model
        if isinstance(cached, SpaceModel):
            return cached
        model: SpaceModel[Space] = _publish(record.family, link_space(record.family, record))
        record.model = model
        return model


def root_record(node: object) -> NodeDecl:
    record = node_record(node)
    if record is None or isinstance(record, FamilyFormal):
        raise RequestError(
            "configure() takes a node declaration, as in configure(House(budget=100))"
        )
    if record.placement is not None:
        raise RequestError(
            f"configure() takes a root: {record.describe()} is already placed; configure the "
            "node that contains it"
        )
    return record


def root_parameters(model: SpaceModel[Space], record: NodeDecl) -> dict[int, object]:
    """The root's plain values and value defaults, by parameter node."""

    values: dict[int, object] = {}
    root = model.linked.scopes[0]
    for name, formal in family_formals(record.family).items():
        index = root.named_members.get(name)
        if index is None or model.linked.nodes[index].kind != "param":
            continue
        if name in record.bindings:
            values[index] = record.bindings[name]
        elif isinstance(formal, Param):
            default = cast(Param[object], formal).default
            if default is not MISSING and default is not UNSUPPLIED:
                values[index] = default
    return values


def configure(node: S) -> S:
    """Compile a root node declaration into its initial configuration.

    ``configure(House(budget=100))`` is typed as ``House``. The name follows
    the result: a configuration, edited with ``with_choices``.
    """

    from .occurrence import bind  # noqa: PLC0415 - keep compilation evaluator-independent

    record = root_record(node)
    try:
        model = compile_node(record)
    except DefinitionError as error:
        raise DefinitionError(f"{error}{at(record.origin)}", findings=error.findings) from error
    return cast(S, bind(model, root_parameters(model, record)))


__all__ = ["SpaceModel", "compile_space", "configure"]
