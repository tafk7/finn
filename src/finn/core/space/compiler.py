# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile authored declarations into an immutable, validated linked model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

from ._linker import link_space, resolve_reference
from .declarations import Space
from .errors import DefinitionError, RequestError
from .ir import LinkedModel, Node
from .results import Finding, FindingKind
from .references import ValueHandle

S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class SpaceModel(Generic[S]):
    """A prepared definition with frozen indexes and a typed configuration factory."""

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

    def bind(
        self,
        parameters: Mapping[object, object] | None = None,
        /,
        **keyword_parameters: object,
    ) -> S:
        from .occurrence import bind  # noqa: PLC0415 - keep compilation evaluator-independent

        return bind(self, parameters if parameters is not None else {}, keyword_parameters)


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


def _validated_order(nodes: tuple[Node, ...]) -> tuple[int, ...]:
    """Validate known structural/explicit edges and order them in O(V + E).

    Arbitrary self-method dependencies are discovered when their reads execute.
    Their reached cycles belong to evaluation, rather than this static check.
    """

    dependencies = tuple(node.dependencies for node in nodes)
    reverse: list[list[int]] = [[] for _ in nodes]
    for node in nodes:
        for dependency in dependencies[node.index]:
            reverse[dependency].append(node.index)

    # Finish dependencies before their users. Iterator frames avoid Python's
    # recursion limit even when the dependency depth equals the model size.
    visited: set[int] = set()
    finished: list[int] = []
    for root in range(len(nodes)):
        if root in visited:
            continue
        visited.add(root)
        stack = [(root, iter(dependencies[root]))]
        while stack:
            current, adjacent = stack[-1]
            child = next(adjacent, None)
            if child is None:
                finished.append(current)
                stack.pop()
            elif child not in visited:
                visited.add(child)
                stack.append((child, iter(dependencies[child])))

    assigned: set[int] = set()
    cycles: list[tuple[str, ...]] = []
    for root in reversed(finished):
        if root in assigned:
            continue
        component: list[int] = []
        pending = [root]
        assigned.add(root)
        while pending:
            current = pending.pop()
            component.append(current)
            for child in reverse[current]:
                if child not in assigned:
                    assigned.add(child)
                    pending.append(child)
        if len(component) > 1 or root in dependencies[root]:
            cycles.append(tuple(sorted(nodes[index].key for index in component)))
    if cycles:
        findings = tuple(
            Finding(
                FindingKind.AUTHORING,
                "cyclic-dependency",
                members[0],
                "cyclic dependencies: " + ", ".join(members),
                details=(("members", members),),
            )
            for members in sorted(cycles)
        )
        raise DefinitionError("model contains cyclic dependencies", findings=findings)
    return tuple(finished)


def compile_space(space_type: type[S]) -> SpaceModel[S]:
    """Return the canonical prepared definition for one root family."""

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError("compile_space requires a Space subclass")
    with _PREPARATION_LOCK:
        cached = space_type.__dict__.get("_space_prepared_model")
        if cached is not None:
            if not isinstance(cached, SpaceModel) or cached.space_type is not space_type:
                raise DefinitionError("invalid prepared-model cache on Space family")
            return cached
        linked = link_space(space_type, _validated_order)
        scope_types = tuple(dict.fromkeys(scope.space_type for scope in linked.scopes))
        _validate_constructors(scope_types)
        model = SpaceModel(space_type, linked)
        # Publication happens only after the complete definition linked successfully.
        for family in _definition_families(scope_types):
            type.__setattr__(family, "_space_definition_finalized", True)
        type.__setattr__(space_type, "_space_prepared_model", model)
        return model


__all__ = ["SpaceModel", "compile_space"]
