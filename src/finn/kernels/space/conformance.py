# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Optional empirical monotonicity checks over explicitly supplied refinements.

These checks can falsify an author's contract on sampled successors. They do
not prove a law, generate candidates, choose defaults, or run a search policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from . import inspection
from .declarations import Space
from .edits import Edit, EditRequest
from .errors import EvaluationError, RequestError
from .results import Answer, Decided, DecisionState, Unresolved, ViewAssessment
from .semantics import ValueSemantics, default_semantics

Sample: TypeAlias = EditRequest | Iterable[EditRequest]
Category = Literal[
    "value", "decision_state", "constraint", "view", "view_output", "readiness", "candidates"
]
_BOOL = cast(ValueSemantics[object], default_semantics(bool))


@dataclass(frozen=True, slots=True)
class MonotonicityViolation:
    sample: int
    key: str
    category: Category
    message: str


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    sample: int
    status: Literal["checked", "skipped", "noop"]
    reason: str


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    checked_successors: int
    violations: tuple[MonotonicityViolation, ...]
    outcomes: tuple[SampleOutcome, ...]

    @property
    def conformant(self) -> bool | None:
        if self.violations:
            return False
        return True if self.checked_successors else None

    @property
    def skipped(self) -> tuple[SampleOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status == "skipped")

    @property
    def noops(self) -> tuple[SampleOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status == "noop")


@dataclass(frozen=True, slots=True)
class _Observation:
    key: str
    owner: str
    category: Category
    answer: Answer[object]
    semantics: ValueSemantics[object] | None


def _observe(point: Space) -> tuple[_Observation, ...]:
    observations: list[_Observation] = []
    for member in inspection.members(point):
        category: Category = (
            "constraint"
            if member.kind in {"constraint", "group"}
            else "readiness"
            if member.kind == "readiness"
            else "view"
            if member.kind == "view"
            else "value"
        )
        semantics = member.reference.semantics
        if category in {"constraint", "readiness"}:
            semantics = _BOOL
        if category == "view":
            evidence = inspection.explain(point, member.reference)
            observations.append(
                _Observation(member.key, member.owner, category, evidence.answer, semantics)
            )
            assessment = evidence.assessment
            if isinstance(assessment, ViewAssessment):
                observations.append(
                    _Observation(
                        member.key,
                        member.owner,
                        "view_output",
                        assessment.output_answer,
                        semantics,
                    )
                )
                observations.append(
                    _Observation(
                        member.key,
                        member.owner,
                        "readiness",
                        cast(Answer[object], assessment.readiness.answer),
                        _BOOL,
                    )
                )
        else:
            observations.append(
                _Observation(
                    member.key,
                    member.owner,
                    category,
                    point.answer(member.reference),
                    semantics,
                )
            )
    for decision in inspection.decisions(point):
        observations.append(
            _Observation(
                decision.key,
                decision.owner,
                "decision_state",
                cast(Answer[object], point.decision_state(decision.reference)),
                decision.reference.semantics,
            )
        )
        candidates = point.candidates(decision.reference)
        if candidates is not None:
            observations.append(
                _Observation(
                    decision.key,
                    decision.owner,
                    "candidates",
                    cast(Answer[object], candidates),
                    decision.reference.semantics,
                )
            )
    return tuple(sorted(observations, key=lambda item: (item.key, item.category)))


def _settled(answer: Answer[object]) -> bool:
    return not isinstance(answer, Unresolved) and not (
        isinstance(answer, Decided)
        and isinstance(answer.value, DecisionState)
        and answer.value.status == "unassigned"
    )


def _equal(before: _Observation, after: _Observation) -> bool:
    left, right = before.answer, after.answer
    if not isinstance(left, Decided) or not isinstance(right, Decided):
        return left == right
    semantics = before.semantics
    if semantics is None:
        raise RequestError(f"{before.key}: a decided observation has no declared value semantics")
    try:
        if before.category == "decision_state":
            first, second = left.value, right.value
            if not isinstance(first, DecisionState) or not isinstance(second, DecisionState):
                return False
            if (first.owner, first.status, first.origin) != (
                second.owner,
                second.status,
                second.origin,
            ):
                return False
            return semantics.values_equal(
                semantics.freeze(first.value), semantics.freeze(second.value)
            )
        if before.category == "candidates":
            old_values, new_values = (
                cast(tuple[object, ...], left.value),
                cast(tuple[object, ...], right.value),
            )
            return len(old_values) == len(new_values) and all(
                semantics.values_equal(semantics.freeze(first), semantics.freeze(second))
                for first, second in zip(old_values, new_values)
            )
        return semantics.values_equal(semantics.freeze(left.value), semantics.freeze(right.value))
    except Exception as cause:
        raise EvaluationError(before.owner, "conformance equality", str(cause)) from cause


class MonotonicityHarness:
    """Compare settled observations on independently sampled atomic successors.

    Supply individual edits or batches made from the same root checkpoint.
    Candidate refusal and malformed requests are reported as skipped samples;
    equal recommits and empty batches are no-ops. Programmer failures propagate
    with the ordinary evaluator context. A positive result covers only the
    supplied samples and queried declarations, never all possible refinements.
    """

    def verify(self, base: Space, samples: Iterable[Sample]) -> ConformanceResult:
        if base.root is not base:
            raise RequestError("conformance refinement samples require a root occurrence")
        supplied = tuple(samples)
        if not supplied:
            return ConformanceResult(0, (), ())
        baseline = _observe(base)
        outcomes: list[SampleOutcome] = []
        violations: list[MonotonicityViolation] = []
        checked = 0
        for index, sample in enumerate(supplied):
            if isinstance(sample, Edit):
                batch: tuple[EditRequest, ...] = (sample,)
            else:
                try:
                    batch = tuple(cast(Iterable[EditRequest], sample))
                except TypeError:
                    outcomes.append(
                        SampleOutcome(index, "skipped", "sample is neither an edit nor a batch")
                    )
                    continue
            try:
                report = base.refine(*batch)
            except RequestError as error:
                outcomes.append(
                    SampleOutcome(index, "skipped", f"invalid refinement request: {error}")
                )
                continue
            if not report.accepted:
                outcomes.append(SampleOutcome(index, "skipped", "refinement refused"))
                continue
            if report.point is base:
                outcomes.append(SampleOutcome(index, "noop", "refinement made no commitments"))
                continue
            checked += 1
            outcomes.append(SampleOutcome(index, "checked", "compared sampled successor"))
            successor = {(item.key, item.category): item for item in _observe(report.point)}
            for observation in baseline:
                if _settled(observation.answer) and not _equal(
                    observation, successor[(observation.key, observation.category)]
                ):
                    violations.append(
                        MonotonicityViolation(
                            index,
                            observation.key,
                            observation.category,
                            "a settled answer changed after atomic refinement",
                        )
                    )
        return ConformanceResult(
            checked,
            tuple(sorted(violations, key=lambda item: (item.sample, item.key, item.category))),
            tuple(outcomes),
        )


__all__ = [
    "ConformanceResult",
    "MonotonicityHarness",
    "MonotonicityViolation",
    "Sample",
    "SampleOutcome",
]
