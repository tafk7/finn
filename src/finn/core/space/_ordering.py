# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Iterative dependency ordering with precise, deterministic cycle diagnostics."""

from __future__ import annotations

from .errors import DefinitionError
from .results import Finding, FindingKind


def dependency_order(
    dependencies: tuple[tuple[int, ...], ...],
    names: tuple[str, ...],
) -> tuple[int, ...]:
    """Validate known structural/explicit edges and order them in O(V + E).

    Arbitrary self-method dependencies are discovered when their reads execute.
    Their reached cycles belong to evaluation, rather than this static check.
    """

    reverse: list[list[int]] = [[] for _ in dependencies]
    for index, inputs in enumerate(dependencies):
        for dependency in inputs:
            reverse[dependency].append(index)

    # Finish dependencies before their users. Iterator frames avoid Python's
    # recursion limit even when the dependency depth equals the model size.
    visited: set[int] = set()
    finished: list[int] = []
    for root in range(len(dependencies)):
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
            cycles.append(tuple(sorted(names[index] for index in component)))
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
