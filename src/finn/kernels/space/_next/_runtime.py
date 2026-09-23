# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demand-limited iterative evaluation over one immutable snapshot."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import TypeAlias, cast

from .errors import EvaluationError
from .ir import Argument, LinkedModel, Node
from .results import (
    Answer,
    ConstraintAssessment,
    Decided,
    DecisionState,
    Finding,
    FindingKind,
    Inapplicable,
    MissingInput,
    NonValue,
    NotApplicable,
    ReadinessAssessment,
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_constraints,
    assess_readiness,
    assess_view,
    constraint_answer,
    owned_answer,
)

Assessment: TypeAlias = ViewAssessment[object] | ConstraintAssessment | ReadinessAssessment


@dataclass(frozen=True, slots=True)
class Evaluation:
    answer: Answer[object]
    dependencies: tuple[int, ...] = ()
    assessment: Assessment | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Snapshot:
    """One root binding and commitment set with a local evaluation cache."""

    linked: LinkedModel
    parameters: Mapping[int, object]
    assignments: Mapping[int, object] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)
    cache: dict[int, Evaluation] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MappingProxyType):
            object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))

    def successor(self, assignments: Mapping[int, object]) -> Snapshot:
        return Snapshot(self.linked, self.parameters, assignments, self.lock)


class _TrialSnapshot(Snapshot):
    """Unpublished state used only inside one locked atomic refinement.

    Edits are admitted in conservative dependency order. Every decision that a
    previously evaluated node could depend on has therefore already been
    processed. Admitting the next edit cannot invalidate the trial's cache.
    Self-dependent domains/guards are rejected as cycles during compilation.

    The backing assignment map is private and is never installed in a public
    snapshot. Publication copies it once and starts a separate empty cache.
    """

    __slots__ = ("_pending", "_published")

    _pending: dict[int, object]
    _published: bool

    def __init__(self, base: Snapshot) -> None:
        pending = dict(base.assignments)
        super().__init__(base.linked, base.parameters, {}, base.lock)
        object.__setattr__(self, "assignments", MappingProxyType(pending))
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_published", False)

    def admit(self, node_index: int, value: object) -> None:
        if self._published:
            raise RuntimeError("a published refinement trial is closed")
        if node_index in self._pending or node_index in self.cache:
            raise RuntimeError(
                "refinement dependency order admitted a previously resolved decision"
            )
        self._pending[node_index] = value

    def publish(self) -> Snapshot:
        if self._published:
            raise RuntimeError("a refinement trial can only be published once")
        object.__setattr__(self, "_published", True)
        return Snapshot(self.linked, self.parameters, self._pending, self.lock)


def _blocked(answers: list[Answer[object]]) -> NonValue | None:
    """Required inputs retain all findings of the highest-precedence nonvalue."""

    for variant in (Unresolved, Rejected, Inapplicable):
        matches = tuple(answer for answer in answers if isinstance(answer, variant))
        if matches:
            return variant(tuple(finding for answer in matches for finding in answer.findings))
    return None


def _clone(node: Node, value: object, *, owner: str, role: str) -> object:
    if node.semantics is None:
        raise EvaluationError(owner, role, f"{node.key} has no value semantics")
    try:
        return node.semantics.freeze(value)
    except Exception as cause:
        raise EvaluationError(owner, role, str(cause)) from cause


def _argument_value(
    linked: LinkedModel, argument: Argument, answer: Answer[object], owner: str
) -> tuple[bool, object]:
    source = linked.nodes[argument.node]
    if argument.mode == "answer":
        if isinstance(answer, Decided):
            return True, Decided(
                _clone(source, answer.value, owner=owner, role="dependency snapshot")
            )
        return True, answer
    if isinstance(answer, Decided):
        return True, _clone(source, answer.value, owner=owner, role="dependency snapshot")
    if argument.mode == "optional":
        if isinstance(answer, Inapplicable):
            return True, NotApplicable(source.owner)
        supplier = source
        while supplier.kind == "alias" and supplier.output is not None:
            supplier = linked.nodes[supplier.output]
        if (
            isinstance(answer, Unresolved)
            and supplier.kind == "param"
            and not supplier.required
            and all(
                finding.code == "input-missing" and finding.owner == supplier.owner
                for finding in answer.findings
            )
        ):
            return True, MissingInput(supplier.owner)
    return False, answer


def _arguments(
    linked: LinkedModel, arguments: tuple[Argument, ...], owner: str
) -> Generator[int, Answer[object], tuple[dict[str, object], NonValue | None]]:
    values: dict[str, object] = {}
    failures: list[Answer[object]] = []
    for argument in arguments:
        answer = yield argument.node
        available, value = _argument_value(linked, argument, answer, owner)
        if available:
            values[argument.name] = value
        else:
            failures.append(cast(Answer[object], value))
    return values, _blocked(failures)


def _guard_result(node: Node, answer: Answer[object]) -> NonValue | None:
    if not isinstance(answer, Decided):
        return answer
    if type(answer.value) is not bool:
        raise EvaluationError(node.owner, "applicability", "guard must return bool")
    return None if answer.value else Inapplicable()


def _guard_assessment(node: Node, answer: NonValue) -> Assessment | None:
    if node.kind == "view":
        return assess_view(answer, owner=node.key, applicability=answer)
    if node.kind == "constraint" or node.kind == "group":
        return ConstraintAssessment({node.key: answer}, answer)
    if node.kind == "readiness":
        return ReadinessAssessment({}, answer)
    return None


def _constraint_members(
    snapshot: Snapshot, reference: int, answer: Answer[object]
) -> Mapping[str, Answer[bool]]:
    assessment = snapshot.cache[reference].assessment
    if isinstance(assessment, ConstraintAssessment) and assessment.answers:
        return assessment.answers
    return {snapshot.linked.nodes[reference].key: cast(Answer[bool], answer)}


def _readiness_members(
    snapshot: Snapshot, reference: int, answer: Answer[object]
) -> Mapping[str, Answer[object]]:
    assessment = snapshot.cache[reference].assessment
    if isinstance(assessment, (ConstraintAssessment, ReadinessAssessment)) and assessment.answers:
        return {key: cast(Answer[object], value) for key, value in assessment.answers.items()}
    return {snapshot.linked.nodes[reference].key: answer}


def _frame(snapshot: Snapshot, node: Node) -> Generator[int, Answer[object], Evaluation]:
    if node.guard is not None:
        guard = yield node.guard
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive, assessment=_guard_assessment(node, inactive))

    if node.kind == "param":
        if node.index in snapshot.parameters:
            return Evaluation(Decided(snapshot.parameters[node.index]))
        return Evaluation(
            Unresolved(
                (
                    Finding(
                        FindingKind.LIMITATION,
                        "input-missing",
                        node.owner,
                        "optional input was omitted at start",
                    ),
                )
            )
        )
    if node.kind == "const":
        return Evaluation(Decided(node.value))
    if node.kind == "decision":
        if node.index in snapshot.assignments:
            return Evaluation(Decided(snapshot.assignments[node.index]))
        return Evaluation(
            Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "decision-unassigned",
                        node.owner,
                        "decision requires a commitment",
                    ),
                )
            )
        )
    if node.kind in {"alias", "guard"}:
        if node.output is None:
            raise EvaluationError(node.owner, node.kind, "missing output reference")
        return Evaluation((yield node.output))
    if node.kind == "select":
        if node.selector is None:
            if len(node.alternatives) != 1:
                raise EvaluationError(node.owner, "selection", "missing selector")
            selected = node.alternatives[0][1]
        else:
            selector = yield node.selector
            if not isinstance(selector, Decided):
                return Evaluation(selector)
            matches = [target for key, target in node.alternatives if key == selector.value]
            if not matches:
                raise EvaluationError(node.owner, "selection", "selector is not a declared case")
            selected = matches[0]
        return Evaluation((yield selected))
    if node.kind == "view":
        if node.output is None:
            raise EvaluationError(node.owner, "view", "missing output reference")
        output = yield node.output
        constraints: dict[str, Answer[bool]] = {}
        for reference in node.constraints:
            answer = yield reference
            constraints.update(_constraint_members(snapshot, reference, answer))
        requires: dict[str, Answer[object]] = {}
        for reference in node.requires:
            answer = yield reference
            requires.update(_readiness_members(snapshot, reference, answer))
        view = assess_view(output, owner=node.key, requires=requires, constraints=constraints)
        return Evaluation(view.accepted_answer, assessment=view)
    if node.kind == "group":
        answers: dict[str, Answer[bool]] = {}
        for reference in node.constraints:
            answer = yield reference
            answers.update(_constraint_members(snapshot, reference, answer))
        group = assess_constraints(answers)
        return Evaluation(cast(Answer[object], group.answer), assessment=group)
    if node.kind == "readiness":
        obligations: dict[str, Answer[object]] = {}
        for reference in node.requires:
            answer = yield reference
            obligations.update(_readiness_members(snapshot, reference, answer))
        readiness = assess_readiness(obligations)
        return Evaluation(cast(Answer[object], readiness.answer), assessment=readiness)

    arguments, failure = yield from _arguments(snapshot.linked, node.arguments, node.owner)
    if failure is not None:
        return Evaluation(failure, assessment=_guard_assessment(node, failure))
    if node.function is None:
        raise EvaluationError(node.owner, node.kind, "missing callback")
    try:
        result = node.function(**arguments)
        if node.kind == "constraint":
            normalized = constraint_answer(cast(bool | Answer[bool], result), node.owner)
            assessment = ConstraintAssessment({node.key: normalized}, normalized)
            return Evaluation(cast(Answer[object], normalized), assessment=assessment)
        if isinstance(result, (Inapplicable, Rejected, Unresolved)):
            return Evaluation(owned_answer(result, node.owner))
        value = result.value if isinstance(result, Decided) else result
        if node.semantics is None:
            raise TypeError("derived output has no value semantics")
        return Evaluation(Decided(node.semantics.freeze(value)))
    except Exception as cause:
        raise EvaluationError(node.owner, node.kind, str(cause)) from cause


@dataclass(slots=True)
class _Task:
    index: int
    frame: Generator[int, Answer[object], Evaluation]
    dependencies: list[int] = field(default_factory=list)
    incoming: Answer[object] | None = None


def evaluate(snapshot: Snapshot, node_index: int) -> Evaluation:
    """Evaluate only demanded dependencies using an explicit stack of frames."""

    with snapshot.lock:
        cached = snapshot.cache.get(node_index)
        if cached is not None:
            return cached
        tasks = [_Task(node_index, _frame(snapshot, snapshot.linked.nodes[node_index]))]
        active = {node_index}
        while tasks:
            task = tasks[-1]
            try:
                if task.incoming is None:
                    demanded = next(task.frame)
                else:
                    incoming, task.incoming = task.incoming, None
                    demanded = task.frame.send(incoming)
            except StopIteration as completion:
                result = cast(Evaluation, completion.value)
                result = replace(result, dependencies=tuple(dict.fromkeys(task.dependencies)))
                snapshot.cache[task.index] = result
                tasks.pop()
                active.remove(task.index)
                if tasks:
                    tasks[-1].incoming = result.answer
                continue
            task.dependencies.append(demanded)
            found = snapshot.cache.get(demanded)
            if found is not None:
                task.incoming = found.answer
            else:
                if demanded in active:
                    raise EvaluationError(
                        snapshot.linked.nodes[demanded].owner, "dependency", "cyclic evaluation"
                    )
                active.add(demanded)
                tasks.append(_Task(demanded, _frame(snapshot, snapshot.linked.nodes[demanded])))
        return snapshot.cache[node_index]


def _applicability(snapshot: Snapshot, node: Node) -> NonValue | None:
    if node.guard is None:
        return None
    return _guard_result(node, evaluate(snapshot, node.guard).answer)


def decision_state(snapshot: Snapshot, node_index: int) -> Answer[DecisionState[object]]:
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.kind != "decision":
            raise EvaluationError(
                node.owner, "decision state", "reference is not an owning decision"
            )
        inactive = _applicability(snapshot, node)
        if inactive is not None:
            return inactive
        if node_index not in snapshot.assignments:
            return Decided(DecisionState(node.owner))
        value = _clone(
            node, snapshot.assignments[node_index], owner=node.owner, role="decision state"
        )
        return Decided(DecisionState(node.owner, "committed", value, "explicit"))


def _domain_inputs(snapshot: Snapshot, node: Node) -> tuple[dict[str, object], NonValue | None]:
    inactive = _applicability(snapshot, node)
    if inactive is not None:
        return {}, inactive
    values: dict[str, object] = {}
    failures: list[Answer[object]] = []
    for argument in node.domain_arguments:
        answer = evaluate(snapshot, argument.node).answer
        available, value = _argument_value(snapshot.linked, argument, answer, node.owner)
        if available:
            values[argument.name] = value
        else:
            failures.append(cast(Answer[object], value))
    return values, _blocked(failures)


def candidate_values(snapshot: Snapshot, node_index: int) -> Answer[tuple[object, ...]] | None:
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.domain is None or node.semantics is None:
            raise EvaluationError(
                node.owner, "domain enumeration", "reference has no decision domain"
            )
        arguments, failure = _domain_inputs(snapshot, node)
        if failure is not None:
            return failure
        if node.domain.candidates is None:
            return None
        return node.domain.enumerate(arguments, semantics=node.semantics, owner=node.owner)


def membership(snapshot: Snapshot, node_index: int, value: object) -> Answer[bool]:
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.domain is None or node.semantics is None:
            raise EvaluationError(
                node.owner, "domain membership", "reference has no decision domain"
            )
        arguments, failure = _domain_inputs(snapshot, node)
        if failure is not None:
            return failure
        candidate = _clone(node, value, owner=node.owner, role="domain candidate snapshot")
        return node.domain.membership(
            candidate, arguments, semantics=node.semantics, owner=node.owner
        )


def copy_answer(snapshot: Snapshot, node_index: int, answer: Answer[object]) -> Answer[object]:
    """Detach a public answer from cached values using its declared semantics."""

    if not isinstance(answer, Decided):
        return answer
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.kind in {"group", "readiness"} and type(answer.value) is bool:
            # Aggregate verdicts are intrinsic immutable booleans; these
            # declarations need no separate user-owned value semantics.
            return answer
        return Decided(_clone(node, answer.value, owner=node.owner, role="public value snapshot"))


def _copy_readiness(snapshot: Snapshot, assessment: ReadinessAssessment) -> ReadinessAssessment:
    answers: dict[str, Answer[object]] = {}
    for key, answer in assessment.answers.items():
        index = snapshot.linked.keys[key]
        if isinstance(answer, Decided) and isinstance(answer.value, DecisionState):
            state = answer.value
            node = snapshot.linked.nodes[index]
            if state.status == "committed":
                value = _clone(node, state.value, owner=node.owner, role="public state snapshot")
                state = replace(state, value=value)
            answers[key] = Decided(state)
        else:
            answers[key] = copy_answer(snapshot, index, answer)
    return ReadinessAssessment(answers, assessment.answer)


def copy_assessment(snapshot: Snapshot, node_index: int, assessment: Assessment) -> Assessment:
    """Detach every mutable value visible in an assessment from its cache."""

    with snapshot.lock:
        if isinstance(assessment, ConstraintAssessment):
            # Validated constraint values are exact bool, and findings are
            # immutable primitives. Their scalar answers can safely be shared.
            return assessment
        if isinstance(assessment, ReadinessAssessment):
            return _copy_readiness(snapshot, assessment)
        node = snapshot.linked.nodes[node_index]
        output_index = node.output if node.output is not None else node_index
        output = copy_answer(snapshot, output_index, assessment.output_answer)
        accepted = (
            output
            if assessment.accepted_answer is assessment.output_answer
            else copy_answer(snapshot, node_index, assessment.accepted_answer)
        )
        return ViewAssessment(
            output,
            _copy_readiness(snapshot, assessment.readiness),
            assessment.constraints,
            accepted,
        )


__all__ = [
    "Evaluation",
    "Snapshot",
    "candidate_values",
    "copy_answer",
    "copy_assessment",
    "decision_state",
    "evaluate",
    "membership",
]
