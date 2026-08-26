# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demand-limited evaluation over the compiled runtime plan."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast
from weakref import WeakKeyDictionary

from .declarations import (
    ABSENT,
    AbsenceMode,
    Constraint,
    Decision,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    EvaluatorSpec,
)
from .errors import EvaluationError
from .facts import FactKey, FactKind, render_provenance
from .points import CommitResult, DesignPoint, ProposalAdoptionResult, make_successor
from .primitives import QualifiedPath, ValueSemantics
from .results import (
    Absent,
    Answer,
    ConstraintAssessment,
    Decided,
    DecisionState,
    Finding,
    FindingKind,
    ItemOutcome,
    ItemSource,
    ProposalAdoptionMode,
    ReadinessAssessment,
    Unresolved,
    ordered_findings,
)

T = TypeVar("T")


@dataclass(slots=True)
class _PointCache:
    facts: dict[FactKey, Answer[object]] = field(default_factory=dict)
    candidates: dict[QualifiedPath, Answer[tuple[object, ...]]] = field(default_factory=dict)


def _finding(
    kind: FindingKind,
    code: str,
    path: QualifiedPath,
    message: str,
    *,
    values: tuple[tuple[str, object], ...] = (),
    trace: tuple[QualifiedPath, ...] = (),
) -> Finding:
    return Finding(kind, code, path, message, values, trace)


def _traced(answer: Answer[T], owner: QualifiedPath) -> Answer[T]:
    if isinstance(answer, Decided):
        return answer
    traced = tuple(
        Finding(
            finding.kind,
            finding.code,
            finding.path,
            finding.message,
            finding.values,
            finding.trace
            if finding.trace and finding.trace[0] == owner
            else (owner, *(finding.trace or (finding.path,))),
        )
        for finding in answer.findings
    )
    return Absent(traced) if isinstance(answer, Absent) else Unresolved(traced)


class EvaluationKernel:
    """Demand-limited, cache-transparent evaluator for validated spaces."""

    def __init__(self) -> None:
        self._caches: WeakKeyDictionary[DesignPoint, _PointCache] = WeakKeyDictionary()

    def clear_cache(self, point: DesignPoint) -> None:
        self._caches.pop(point, None)

    def _cache(self, point: DesignPoint) -> _PointCache:
        return self._caches.setdefault(point, _PointCache())

    def _discard(self, points: Iterable[DesignPoint]) -> None:
        for point in points:
            self._caches.pop(point, None)

    def resolve(self, point: DesignPoint, target: FactKey) -> Answer[object]:
        cache = self._cache(point).facts
        if target in cache:
            return cache[target]
        for key in point.design_space._plan.evaluation_order((target,)):
            if key not in cache:
                cache[key] = self._compute_fact(point, key)
        try:
            return cache[target]
        except KeyError:
            raise EvaluationError(
                target.path, "evaluation plan", f"no value produced for {target}"
            ) from None

    def _dependency_answer(self, point: DesignPoint, ref: DependencyRef) -> Answer[object]:
        kind = FactKind.VALUE if ref.kind is DependencyKind.DECISION else FactKind.PROPERTY
        key = FactKey(kind, ref.path)
        try:
            return self._cache(point).facts[key]
        except KeyError:
            raise EvaluationError(
                ref.path, "evaluation plan", "a declared dependency was not evaluated"
            ) from None

    def _prepare(
        self,
        point: DesignPoint,
        owner: QualifiedPath,
        dependencies: tuple[DependencyRef, ...],
    ) -> DependencyView | Absent | Unresolved:
        values: dict[str, object] = {}
        absent: list[Finding] = []
        unresolved: list[Finding] = []
        for ref in dependencies:
            if ref.kind is DependencyKind.PROBLEM:
                if ref.path in point.problem:
                    values[ref.name] = point.problem[ref.path]
                elif ref.absence is AbsenceMode.ALLOWS_ABSENT:
                    values[ref.name] = ABSENT
                else:
                    unresolved.append(
                        _finding(
                            FindingKind.LIMITATION,
                            "problem-field-unavailable",
                            ref.path,
                            "an omitted problem field is unavailable for this immutable problem",
                            trace=(owner,),
                        )
                    )
                continue
            answer = _traced(self._dependency_answer(point, ref), owner)
            if isinstance(answer, Decided):
                values[ref.name] = answer.value
            elif isinstance(answer, Unresolved):
                unresolved.extend(answer.findings)
            elif ref.absence is AbsenceMode.ALLOWS_ABSENT:
                values[ref.name] = ABSENT
            else:
                absent.extend(
                    answer.findings
                    or (
                        _finding(
                            FindingKind.LIMITATION,
                            "required-dependency-absent",
                            ref.path,
                            "a required dependency does not apply",
                            trace=(owner,),
                        ),
                    )
                )
        if absent:
            return Absent(ordered_findings(absent))
        if unresolved:
            return Unresolved(ordered_findings(unresolved))
        return DependencyView(values)

    def _call(
        self,
        owner: QualifiedPath,
        role: str,
        evaluator: EvaluatorSpec[object],
        prepared: DependencyView,
    ) -> Answer[object]:
        try:
            raw = evaluator.evaluator(prepared)
        except Exception as exc:
            raise EvaluationError(owner, role, "evaluator raised an exception") from exc
        if not isinstance(raw, (Decided, Absent, Unresolved)):
            raise EvaluationError(
                owner, role, f"evaluator returned unsupported result {type(raw).__name__}"
            )
        return cast(Answer[object], raw)

    def _evaluate_applicability(
        self,
        point: DesignPoint,
        owner: QualifiedPath,
        evaluator: EvaluatorSpec[object],
    ) -> Answer[object]:
        prepared = self._prepare(point, owner, evaluator.dependencies)
        if not isinstance(prepared, DependencyView):
            return prepared
        answer = self._call(owner, "applicability", evaluator, prepared)
        if isinstance(answer, Decided) and type(answer.value) is not bool:
            raise EvaluationError(owner, "applicability", "evaluator must decide a bool")
        return answer

    def _applicability(
        self, point: DesignPoint, declaration: Decision | DerivedProperty | Constraint
    ) -> Answer[object]:
        if declaration.applies_if is None:
            return Decided(True)
        answer = self._evaluate_applicability(
            point, declaration.path, cast(EvaluatorSpec[object], declaration.applies_if)
        )
        if isinstance(answer, Decided) and answer.value is False:
            return Absent()
        return answer

    @staticmethod
    def _snapshot_value(
        path: QualifiedPath,
        role: str,
        answer: Answer[object],
        semantics: ValueSemantics[object],
    ) -> Answer[object]:
        if not isinstance(answer, Decided):
            return answer
        try:
            return Decided(semantics.freeze(answer.value))
        except Exception as exc:
            raise EvaluationError(path, role, "result could not be snapshotted") from exc

    @staticmethod
    def _gate_result(
        owner: QualifiedPath, prepared: DependencyView | Absent | Unresolved
    ) -> Answer[object]:
        if isinstance(prepared, DependencyView):
            return Decided(None)
        if isinstance(prepared, Unresolved):
            return prepared
        findings = tuple(
            Finding(
                FindingKind.LIMITATION,
                "domain-gate-absent",
                finding.path,
                "the decision applies but a required domain input does not",
                finding.values,
                finding.trace or (owner,),
            )
            for finding in prepared.findings
        )
        return Unresolved(
            findings
            or (
                _finding(
                    FindingKind.LIMITATION,
                    "domain-gate-absent",
                    owner,
                    "the decision applies but a required domain input does not",
                ),
            )
        )

    def _compute_fact(self, point: DesignPoint, key: FactKey) -> Answer[object]:
        space = point.design_space
        if key.kind is FactKind.APPLIES:
            declaration = cast(
                Decision | DerivedProperty | Constraint,
                space.decisions.get(key.path)
                or space.properties.get(key.path)
                or space.constraints.get(key.path),
            )
            return self._applicability(point, declaration)

        if key.kind is FactKind.DOMAIN_READY:
            applies = self._cache(point).facts[FactKey(FactKind.APPLIES, key.path)]
            if not isinstance(applies, Decided):
                return applies
            decision = space.decisions[key.path]
            return self._gate_result(
                key.path, self._prepare(point, key.path, decision.domain.dependencies)
            )

        if key.kind is FactKind.VALUE:
            applies = self._cache(point).facts[FactKey(FactKind.APPLIES, key.path)]
            if not isinstance(applies, Decided):
                return applies
            ready = self._cache(point).facts[FactKey(FactKind.DOMAIN_READY, key.path)]
            if not isinstance(ready, Decided):
                return ready
            if key.path in point.assignments:
                return Decided(point.assignments[key.path])
            return Unresolved(
                (
                    _finding(
                        FindingKind.BLOCKER,
                        "decision-unassigned",
                        key.path,
                        "applicable decision has no committed value",
                    ),
                )
            )

        if key.kind is FactKind.PROPERTY:
            applies = self._cache(point).facts[FactKey(FactKind.APPLIES, key.path)]
            if not isinstance(applies, Decided):
                return applies
            declaration = space.properties[key.path]
            prepared = self._prepare(point, key.path, declaration.evaluator.dependencies)
            if not isinstance(prepared, DependencyView):
                return prepared
            answer = self._call(
                key.path, "property", cast(EvaluatorSpec[object], declaration.evaluator), prepared
            )
            return self._snapshot_value(key.path, "property", answer, declaration.value_semantics)

        if key.kind is FactKind.CONSTRAINT:
            applies = self._cache(point).facts[FactKey(FactKind.APPLIES, key.path)]
            if not isinstance(applies, Decided):
                return applies
            declaration = space.constraints[key.path]
            prepared = self._prepare(point, key.path, declaration.evaluator.dependencies)
            if not isinstance(prepared, DependencyView):
                return prepared
            answer = self._call(
                key.path, "constraint", cast(EvaluatorSpec[object], declaration.evaluator), prepared
            )
            if isinstance(answer, Decided) and type(answer.value) is not bool:
                raise EvaluationError(key.path, "constraint", "evaluator must decide a bool")
            return answer

        decision = space.decisions[key.path]
        applies = self._cache(point).facts[FactKey(FactKind.APPLIES, key.path)]
        if not isinstance(applies, Decided):
            return applies
        ready = self._cache(point).facts[FactKey(FactKind.DOMAIN_READY, key.path)]
        if not isinstance(ready, Decided):
            return ready
        if decision.proposal is None:
            return Absent()
        prepared = self._prepare(point, key.path, decision.proposal.dependencies)
        if not isinstance(prepared, DependencyView):
            return prepared
        answer = self._call(
            key.path, "proposal", cast(EvaluatorSpec[object], decision.proposal), prepared
        )
        return self._snapshot_value(key.path, "proposal", answer, decision.value_semantics)

    def _ensure_dependencies(
        self, point: DesignPoint, dependencies: tuple[DependencyRef, ...]
    ) -> None:
        for ref in dependencies:
            if ref.kind is DependencyKind.PROBLEM:
                continue
            kind = FactKind.VALUE if ref.kind is DependencyKind.DECISION else FactKind.PROPERTY
            self.resolve(point, FactKey(kind, ref.path))

    def check_candidate(
        self, point: DesignPoint, decision: Decision, value: object
    ) -> Answer[bool]:
        applies = self.resolve(point, FactKey(FactKind.APPLIES, decision.path))
        if not isinstance(applies, Decided):
            return cast(Answer[bool], applies)
        ready = self.resolve(point, FactKey(FactKind.DOMAIN_READY, decision.path))
        if not isinstance(ready, Decided):
            return cast(Answer[bool], ready)
        prepared = self._prepare(point, decision.path, decision.domain.dependencies)
        if not isinstance(prepared, DependencyView):
            return cast(Answer[bool], self._gate_result(decision.path, prepared))
        try:
            raw = decision.domain.accepts(value, prepared)
        except Exception as exc:
            raise EvaluationError(decision.path, "domain", "evaluator raised an exception") from exc
        if not isinstance(raw, (Decided, Absent, Unresolved)):
            raise EvaluationError(
                decision.path,
                "domain",
                f"evaluator returned unsupported result {type(raw).__name__}",
            )
        if isinstance(raw, Decided) and type(raw.value) is not bool:
            raise EvaluationError(decision.path, "domain", "evaluator must decide a bool")
        return raw

    def decision_state(self, point: DesignPoint, path: QualifiedPath) -> Answer[DecisionState]:
        applies = self.resolve(point, FactKey(FactKind.APPLIES, path))
        if not isinstance(applies, Decided):
            return cast(Answer[DecisionState], applies)
        decision = point.design_space.decisions[path]
        if path in point.assignments:
            return Decided(
                DecisionState(
                    path,
                    "committed",
                    point.assignments[path],
                    point.origins[path],
                    decision.proposal is not None,
                )
            )
        ready = self.resolve(point, FactKey(FactKind.DOMAIN_READY, path))
        if not isinstance(ready, Decided):
            return cast(Answer[DecisionState], ready)
        return Decided(
            DecisionState(path, "unassigned", has_proposal=decision.proposal is not None)
        )

    def query_property(self, point: DesignPoint, path: QualifiedPath) -> Answer[object]:
        return self.resolve(point, FactKey(FactKind.PROPERTY, path))

    def enumerate_candidates(
        self, point: DesignPoint, path: QualifiedPath
    ) -> Answer[tuple[object, ...]]:
        cache = self._cache(point).candidates
        if path in cache:
            return cache[path]
        decision = point.design_space.decisions[path]
        applies = self.resolve(point, FactKey(FactKind.APPLIES, path))
        if not isinstance(applies, Decided):
            return cast(Answer[tuple[object, ...]], applies)
        ready = self.resolve(point, FactKey(FactKind.DOMAIN_READY, path))
        if not isinstance(ready, Decided):
            return cast(Answer[tuple[object, ...]], ready)
        declaration = decision.domain.candidates
        if declaration is None:
            answer: Answer[tuple[object, ...]] = Absent(
                (
                    _finding(
                        FindingKind.LIMITATION,
                        "candidate-enumerator-unavailable",
                        path,
                        "the decision domain does not expose a finite candidate enumeration",
                    ),
                )
            )
        else:
            self._ensure_dependencies(point, declaration.dependencies)
            prepared = self._prepare(point, path, declaration.dependencies)
            if not isinstance(prepared, DependencyView):
                answer = cast(Answer[tuple[object, ...]], prepared)
            else:
                raw = self._call(
                    path,
                    "candidate enumeration",
                    cast(EvaluatorSpec[object], declaration),
                    prepared,
                )
                if isinstance(raw, Decided):
                    if type(raw.value) is not tuple:
                        raise EvaluationError(
                            path, "candidate enumeration", "evaluator must decide a tuple"
                        )
                    frozen: list[object] = []
                    for value in raw.value:
                        try:
                            frozen.append(decision.value_semantics.freeze(value))
                        except Exception as exc:
                            raise EvaluationError(
                                path,
                                "candidate enumeration",
                                "a candidate could not be snapshotted",
                            ) from exc
                    answer = Decided(tuple(frozen))
                else:
                    answer = cast(Answer[tuple[object, ...]], raw)
        cache[path] = answer
        return answer

    @staticmethod
    def _answer_outcome(
        path: QualifiedPath,
        answer: Answer[object],
        source: ItemSource,
        value: object | None = None,
    ) -> ItemOutcome:
        if isinstance(answer, Unresolved):
            return ItemOutcome(path, "unresolved", source, value, answer.findings)
        findings = answer.findings if isinstance(answer, Absent) else ()
        if not findings:
            findings = (
                _finding(
                    FindingKind.REJECTION,
                    "item-unavailable",
                    path,
                    "the item is not applicable or was not accepted",
                ),
            )
        return ItemOutcome(path, "rejected", source, value, findings)

    def commit_assignments(
        self, point: DesignPoint, assignments: Mapping[QualifiedPath, object]
    ) -> CommitResult:
        provisional = point
        created: list[DesignPoint] = []
        outcomes: list[ItemOutcome] = []
        order = point.design_space._plan.assignment_order(frozenset(assignments))
        for path in order:
            decision = point.design_space.decisions[path]
            value = assignments[path]
            if path in provisional.assignments:
                try:
                    equal = decision.value_semantics.values_equal(
                        provisional.assignments[path], value
                    )
                except Exception as exc:
                    raise EvaluationError(path, "value equality", "comparison raised") from exc
                if equal:
                    outcomes.append(ItemOutcome(path, "unchanged", "explicit", value))
                else:
                    outcomes.append(
                        ItemOutcome(
                            path,
                            "rejected",
                            "conflict",
                            value,
                            (
                                _finding(
                                    FindingKind.REJECTION,
                                    "assignment-conflict",
                                    path,
                                    "a different value is already committed",
                                ),
                            ),
                        )
                    )
                continue
            applies = self.resolve(provisional, FactKey(FactKind.APPLIES, path))
            if not isinstance(applies, Decided):
                outcomes.append(
                    self._answer_outcome(
                        path, cast(Answer[object], applies), "applicability", value
                    )
                )
                continue
            domain = self.check_candidate(provisional, decision, value)
            if isinstance(domain, Decided) and domain.value is True:
                assignments_next = dict(provisional.assignments)
                origins_next = dict(provisional.origins)
                assignments_next[path] = value
                origins_next[path] = "explicit"
                provisional = make_successor(provisional, assignments_next, origins_next)
                created.append(provisional)
                outcomes.append(ItemOutcome(path, "committed", "explicit", value))
            elif isinstance(domain, Decided):
                outcomes.append(
                    ItemOutcome(
                        path,
                        "rejected",
                        "domain",
                        value,
                        (
                            _finding(
                                FindingKind.REJECTION,
                                "candidate-outside-domain",
                                path,
                                "candidate is outside the decision domain",
                            ),
                        ),
                    )
                )
            else:
                outcomes.append(
                    self._answer_outcome(path, cast(Answer[object], domain), "domain", value)
                )
        self._discard(created[:-1])
        return CommitResult(provisional, tuple(outcomes))

    def _adopt_one(
        self, point: DesignPoint, path: QualifiedPath
    ) -> tuple[DesignPoint, ItemOutcome]:
        decision = point.design_space.decisions[path]
        if path in point.assignments:
            return point, ItemOutcome(path, "unchanged", "proposal", point.assignments[path])
        applies = self.resolve(point, FactKey(FactKind.APPLIES, path))
        if isinstance(applies, Absent):
            return point, ItemOutcome(path, "skipped", "applicability", findings=applies.findings)
        if isinstance(applies, Unresolved):
            return point, ItemOutcome(
                path, "unresolved", "applicability", findings=applies.findings
            )
        if decision.proposal is None:
            return point, ItemOutcome(
                path,
                "skipped",
                "declaration",
                findings=(
                    _finding(
                        FindingKind.AUTHORING,
                        "proposal-missing",
                        path,
                        "decision declares no proposal",
                    ),
                ),
            )
        dead = point.design_space._plan.dead_proposals.get(path)
        if dead is not None:
            return point, ItemOutcome(
                path,
                "unresolved",
                "proposal",
                findings=(
                    _finding(
                        FindingKind.AUTHORING,
                        "dead-proposal",
                        path,
                        "proposal requires its own uncommitted decision value",
                        values=(
                            ("facts", tuple(str(node) for node in dead.fact_trace)),
                            ("provenance", render_provenance(dead.provenance)),
                        ),
                        trace=tuple(node.path for node in dead.fact_trace),
                    ),
                ),
            )
        proposal = self.resolve(point, FactKey(FactKind.PROPOSAL, path))
        if isinstance(proposal, Absent):
            return point, ItemOutcome(
                path,
                "skipped",
                "proposal",
                findings=proposal.findings
                or (
                    _finding(
                        FindingKind.REJECTION,
                        "proposal-absent",
                        path,
                        "proposal produced no candidate",
                    ),
                ),
            )
        if isinstance(proposal, Unresolved):
            return point, ItemOutcome(path, "unresolved", "proposal", findings=proposal.findings)
        domain = self.check_candidate(point, decision, proposal.value)
        if isinstance(domain, Decided) and domain.value is True:
            assignments = dict(point.assignments)
            origins = dict(point.origins)
            assignments[path] = proposal.value
            origins[path] = "proposal"
            successor = make_successor(point, assignments, origins)
            return successor, ItemOutcome(path, "committed", "proposal", proposal.value)
        if isinstance(domain, Decided):
            return point, ItemOutcome(
                path,
                "rejected",
                "domain",
                proposal.value,
                (
                    _finding(
                        FindingKind.REJECTION,
                        "candidate-outside-domain",
                        path,
                        "candidate is outside the decision domain",
                    ),
                ),
            )
        return point, self._answer_outcome(
            path, cast(Answer[object], domain), "domain", proposal.value
        )

    def adopt_proposals(
        self,
        point: DesignPoint,
        targets: tuple[QualifiedPath, ...],
        mode: ProposalAdoptionMode,
    ) -> ProposalAdoptionResult:
        ordered = point.design_space._plan.adoption_order(frozenset(targets))
        provisional = point
        created: list[DesignPoint] = []
        passes: list[tuple[ItemOutcome, ...]] = []
        while True:
            outcomes: list[ItemOutcome] = []
            committed = 0
            for path in ordered:
                successor, outcome = self._adopt_one(provisional, path)
                outcomes.append(outcome)
                if successor is not provisional:
                    provisional = successor
                    created.append(successor)
                    committed += 1
            passes.append(tuple(outcomes))
            if mode is ProposalAdoptionMode.ONCE or committed == 0:
                break
        self._discard(created[:-1])
        return ProposalAdoptionResult(provisional, tuple(passes))

    def evaluate_constraints(
        self, point: DesignPoint, paths: tuple[QualifiedPath, ...]
    ) -> ConstraintAssessment:
        answers = {
            path: cast(Answer[bool], self.resolve(point, FactKey(FactKind.CONSTRAINT, path)))
            for path in paths
        }
        complete = all(isinstance(answer, (Decided, Absent)) for answer in answers.values())
        verdict = (
            all(answer.value for answer in answers.values() if isinstance(answer, Decided))
            if complete
            else None
        )
        return ConstraintAssessment(answers, verdict)

    def check_readiness(
        self,
        point: DesignPoint,
        profile_name: str,
        decisions: tuple[QualifiedPath, ...],
        properties: tuple[QualifiedPath, ...],
        constraints: tuple[QualifiedPath, ...],
    ) -> ReadinessAssessment:
        answers: dict[QualifiedPath, Answer[object]] = {}
        for path in decisions:
            state = self.decision_state(point, path)
            if isinstance(state, Decided) and state.value.status == "unassigned":
                answers[path] = Unresolved(
                    (
                        _finding(
                            FindingKind.BLOCKER,
                            "readiness-decision-unassigned",
                            path,
                            "readiness requires a committed decision value",
                        ),
                    )
                )
            else:
                answers[path] = cast(Answer[object], state)
        for path in properties:
            answers[path] = self.query_property(point, path)
        for path in constraints:
            answers[path] = self.resolve(point, FactKey(FactKind.CONSTRAINT, path))
        ready = (
            True
            if all(isinstance(answer, (Decided, Absent)) for answer in answers.values())
            else None
        )
        return ReadinessAssessment(profile_name, answers, ready)


__all__ = ["EvaluationKernel"]
