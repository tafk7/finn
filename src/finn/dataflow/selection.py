# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Small FINN-side finite coordinator for jointly constrained decisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from finn.dataflow.design import (
    Absent,
    Decided,
    DesignPoint,
    Engine,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
)

_SELECTION_PATH = QualifiedPath("compiler.finite_selection")


@dataclass(frozen=True)
class FiniteSelectionResult:
    """All feasible points in deterministic assignment order."""

    points: tuple[DesignPoint, ...]
    findings: tuple[Finding, ...] = ()


def _value_key(value: object) -> tuple[object, ...]:
    if isinstance(value, Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, str(value.value))
    if value is None or type(value) in {bool, int, float, str}:
        return (type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", *(_value_key(item) for item in value))
    raise TypeError(
        "finite selection requires scalar, enum, or tuple decision values for stable ordering"
    )


def _point_key(point: DesignPoint) -> tuple[object, ...]:
    return tuple(
        (path.value, _value_key(value))
        for path, value in sorted(point.assignments.items(), key=lambda item: item[0])
    )


def _known_false(point: DesignPoint, engine: Engine, constraint_set: str) -> bool:
    assessment = engine.evaluate_constraint_set(point, constraint_set)
    return any(
        isinstance(answer, Decided) and answer.value is False
        for answer in assessment.answers.values()
    )


def enumerate_feasible_points(
    engine: Engine,
    point: DesignPoint,
    decision_paths: Iterable[QualifiedPath | str],
    *,
    constraint_set: str,
    traversal_order: Iterable[QualifiedPath | str] | None = None,
) -> FiniteSelectionResult:
    """Exhaustively coordinate finite decisions without greedy commitment.

    ``traversal_order`` may prioritize queries, but all feasible assignments are
    returned and sorted independently of that order.
    """
    requested = tuple(sorted({QualifiedPath.parse(path) for path in decision_paths}))
    if traversal_order is None:
        ordered = requested
    else:
        priority = tuple(QualifiedPath.parse(path) for path in traversal_order)
        if set(priority) != set(requested) or len(priority) != len(requested):
            raise ValueError("traversal_order must contain each coordinated decision exactly once")
        ordered = priority
    solutions: dict[tuple[object, ...], DesignPoint] = {}
    observed_findings: dict[tuple[str, str, str], Finding] = {}

    def remember(findings: tuple[Finding, ...]) -> None:
        for finding in findings:
            observed_findings[(finding.path.value, finding.kind.value, finding.code)] = finding

    def visit(current: DesignPoint) -> None:
        if _known_false(current, engine, constraint_set):
            return
        unresolved_paths: list[QualifiedPath] = []
        enumerable: list[tuple[QualifiedPath, tuple[object, ...]]] = []
        for path in ordered:
            state = engine.decision_state(current, path)
            if isinstance(state, Absent):
                remember(state.findings)
                continue
            if isinstance(state, Unresolved):
                remember(state.findings)
                unresolved_paths.append(path)
                continue
            if state.value.status == "committed":
                continue
            candidate_answer = engine.enumerate_candidates(current, path)
            if isinstance(candidate_answer, Decided):
                enumerable.append((path, candidate_answer.value))
            else:
                remember(candidate_answer.findings)
                unresolved_paths.append(path)
        if not enumerable:
            if unresolved_paths:
                return
            assessment = engine.evaluate_constraint_set(current, constraint_set)
            for answer in assessment.answers.values():
                if isinstance(answer, (Absent, Unresolved)):
                    remember(answer.findings)
            if assessment.verdict is True:
                solutions[_point_key(current)] = current
            return
        path, candidate_values = enumerable[0]
        for candidate in candidate_values:
            result = engine.commit_assignments(current, {path: candidate})
            outcome = result.outcomes[0]
            remember(outcome.findings)
            if outcome.disposition in {"committed", "unchanged"}:
                visit(result.point)

    visit(point)
    if not solutions and not observed_findings:
        finding = Finding(
            FindingKind.REJECTION,
            "finite-selection-no-feasible-point",
            _SELECTION_PATH,
            "finite candidate domains contain no assignment satisfying the constraint set",
        )
        observed_findings[(finding.path.value, finding.kind.value, finding.code)] = finding
    return FiniteSelectionResult(
        tuple(solutions[key] for key in sorted(solutions)),
        tuple(observed_findings[key] for key in sorted(observed_findings)),
    )


__all__ = ["FiniteSelectionResult", "enumerate_feasible_points"]
