# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile authored declarations into an immutable, validated linked model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    """A reusable family. Its frozen indexes remain authoritative after compile."""

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

    def start(self, parameters: Mapping[object, object] | None = None) -> S:
        from .occurrence import start  # noqa: PLC0415 - keep compilation evaluator-independent

        return start(self, parameters if parameters is not None else {})


def _validated_order(nodes: tuple[Node, ...]) -> tuple[int, ...]:
    """Iterative Kosaraju validation and dependency-first order in O(V + E)."""

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
    """Validate the family once. No user evaluator runs while linking it."""

    return SpaceModel(space_type, link_space(space_type, _validated_order))


__all__ = ["SpaceModel", "compile_space"]
