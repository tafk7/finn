# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional empirical checks for adapter-owned monotonicity obligations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .engine import Engine
from .errors import RequestError
from .points import DesignPoint
from .primitives import QualifiedPath, ValueSemantics
from .results import Absent, Answer, Decided, DecisionState


@dataclass(frozen=True, slots=True)
class ConformanceSnapshot:
    decisions: Mapping[QualifiedPath, Answer[Any]]
    properties: Mapping[QualifiedPath, Answer[Any]]
    candidates: Mapping[QualifiedPath, Answer[Any]]
    constraints: Mapping[QualifiedPath, Answer[Any]]


@dataclass(frozen=True, slots=True)
class MonotonicityViolation:
    path: QualifiedPath
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    checked_successors: int
    violations: tuple[MonotonicityViolation, ...]
    skipped: tuple[tuple[QualifiedPath, str], ...] = ()

    @property
    def conformant(self) -> bool | None:
        if self.violations:
            return False
        return True if self.checked_successors else None


class MonotonicityHarness:
    """Try to falsify monotonicity over caller-supplied assignment samples."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def snapshot(self, point: DesignPoint) -> ConformanceSnapshot:
        decisions = {
            path: self._engine.decision_state(point, path)
            for path in sorted(point.design_space.decisions)
        }
        properties = {
            path: self._engine.query_property(point, path)
            for path in sorted(point.design_space.properties)
        }
        candidates = {
            path: self._engine.enumerate_candidates(point, path)
            for path, declaration in sorted(point.design_space.decisions.items())
            if declaration.domain.candidates is not None
        }
        constraints = dict(self._engine.evaluate_constraints(point).answers)
        return ConformanceSnapshot(decisions, properties, candidates, constraints)

    @staticmethod
    def _stable(answer: Answer[object]) -> bool:
        return isinstance(answer, Absent) or (
            isinstance(answer, Decided)
            and not (
                isinstance(answer.value, DecisionState) and answer.value.status == "unassigned"
            )
        )

    @staticmethod
    def _equal_decided(
        left: object, right: object, semantics: ValueSemantics[object] | None
    ) -> bool:
        if isinstance(left, DecisionState) and isinstance(right, DecisionState):
            if (left.path, left.status, left.origin, left.has_proposal) != (
                right.path,
                right.status,
                right.origin,
                right.has_proposal,
            ):
                return False
            if left.status == "unassigned":
                return True
            return semantics is not None and semantics.values_equal(left.value, right.value)
        if semantics is not None:
            return semantics.values_equal(left, right)
        return left == right

    @classmethod
    def _equal(
        cls,
        left: Answer[object],
        right: Answer[object],
        semantics: ValueSemantics[object] | None,
    ) -> bool:
        if isinstance(left, Decided) and isinstance(right, Decided):
            return cls._equal_decided(left.value, right.value, semantics)
        return left == right

    def compare(
        self,
        point: DesignPoint,
        base: ConformanceSnapshot,
        successor: ConformanceSnapshot,
    ) -> tuple[MonotonicityViolation, ...]:
        semantics = {
            **{path: item.value_semantics for path, item in point.design_space.decisions.items()},
            **{path: item.value_semantics for path, item in point.design_space.properties.items()},
        }
        violations: list[MonotonicityViolation] = []
        for category, before, after in (
            ("decision", base.decisions, successor.decisions),
            ("property", base.properties, successor.properties),
            ("constraint", base.constraints, successor.constraints),
        ):
            for path, answer in before.items():
                if self._stable(answer) and not self._equal(
                    answer, after[path], semantics.get(path)
                ):
                    violations.append(
                        MonotonicityViolation(
                            path,
                            category,
                            "a final answer changed after assignment commitment",
                        )
                    )
        for path, answer in base.candidates.items():
            after_answer = successor.candidates[path]
            same = answer == after_answer
            if isinstance(answer, Decided) and isinstance(after_answer, Decided):
                candidate_semantics = point.design_space.decisions[path].value_semantics
                same = len(answer.value) == len(after_answer.value) and all(
                    candidate_semantics.values_equal(left, right)
                    for left, right in zip(answer.value, after_answer.value, strict=True)
                )
            if self._stable(answer) and not same:
                violations.append(
                    MonotonicityViolation(
                        path,
                        "candidates",
                        "a final candidate enumeration changed after assignment commitment",
                    )
                )
        return tuple(sorted(violations, key=lambda item: (item.path, item.category)))

    def verify(
        self,
        point: DesignPoint,
        samples: Mapping[QualifiedPath | str, Iterable[object]],
    ) -> ConformanceResult:
        baseline = self.snapshot(point)
        violations: list[MonotonicityViolation] = []
        skipped: list[tuple[QualifiedPath, str]] = []
        checked = 0
        for raw_path, values in samples.items():
            try:
                path = QualifiedPath.parse(raw_path)
            except (TypeError, ValueError):
                skipped.append((QualifiedPath("invalid-sample"), "invalid sample path"))
                continue
            if path not in point.design_space.decisions:
                skipped.append((path, "no decision is declared at this path"))
                continue
            for value in values:
                try:
                    result = self._engine.commit_assignments(point, {path: value})
                except RequestError:
                    skipped.append((path, "the assignment request was rejected"))
                    continue
                if result.point is point:
                    skipped.append((path, "the candidate produced no successor point"))
                    continue
                checked += 1
                violations.extend(self.compare(point, baseline, self.snapshot(result.point)))
        return ConformanceResult(checked, tuple(violations), tuple(skipped))


__all__ = [
    "ConformanceResult",
    "ConformanceSnapshot",
    "MonotonicityHarness",
    "MonotonicityViolation",
]
