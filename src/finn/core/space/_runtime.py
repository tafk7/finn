# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demand-limited iterative evaluation over one immutable snapshot."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

from . import _execution
from .errors import EvaluationError
from .ir import Argument, LinkedModel, Node
from .results import (
    QueryResult,
    ConstraintAssessment,
    Available,
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
    constraint_result,
    owned_result,
)

if TYPE_CHECKING:
    from .compiler import SpaceModel
    from .declarations import Space

Assessment: TypeAlias = ViewAssessment[object] | ConstraintAssessment | ReadinessAssessment


@dataclass(frozen=True, slots=True)
class Evaluation:
    result: QueryResult[object]
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
    model: SpaceModel[Space] | None = field(default=None, kw_only=True, repr=False)
    work: _execution.Work = field(default_factory=_execution.Work, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MappingProxyType):
            object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))

    def successor(self, assignments: Mapping[int, object]) -> Snapshot:
        return Snapshot(self.linked, self.parameters, assignments, model=self.model)


class _TrialSnapshot(Snapshot):
    """Private complete candidates, admitted on demand before value publication.

    Existing assignments may seed a monotone trial. Replacement starts with no
    trusted assignments, so all retained candidates must pass admission again.
    """

    __slots__ = ("_pending", "_published", "_candidates")

    _pending: dict[int, object]
    _published: bool
    _candidates: Mapping[int, object]

    def __init__(self, base: Snapshot, candidates: Mapping[int, object]) -> None:
        pending = dict(base.assignments)
        super().__init__(base.linked, base.parameters, {}, base.lock, model=base.model)
        object.__setattr__(self, "assignments", MappingProxyType(pending))
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_candidates", MappingProxyType(dict(candidates)))
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
        return Snapshot(self.linked, self.parameters, self._pending, model=self.model)


def _blocked(answers: list[QueryResult[object]]) -> NonValue | None:
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
    linked: LinkedModel, argument: Argument, answer: QueryResult[object], owner: str
) -> tuple[bool, object]:
    source = linked.nodes[argument.node]
    if argument.mode == "result":
        if isinstance(answer, Available):
            return True, Available(
                _clone(source, answer.value, owner=owner, role="dependency snapshot")
            )
        return True, answer
    if isinstance(answer, Available):
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
) -> Generator[int, object, tuple[dict[str, object], NonValue | None]]:
    values: dict[str, object] = {}
    failures: list[QueryResult[object]] = []
    for argument in arguments:
        answer = cast(QueryResult[object], (yield argument.node))
        available, value = _argument_value(linked, argument, answer, owner)
        if available:
            values[argument.name] = value
        else:
            failures.append(cast(QueryResult[object], value))
    return values, _blocked(failures)


def _guard_result(node: Node, answer: QueryResult[object]) -> NonValue | None:
    if not isinstance(answer, Available):
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
    snapshot: Snapshot, reference: int, answer: QueryResult[object]
) -> Mapping[str, QueryResult[bool]]:
    assessment = snapshot.cache[reference].assessment
    if isinstance(assessment, ConstraintAssessment) and assessment.results:
        return assessment.results
    return {snapshot.linked.nodes[reference].key: cast(QueryResult[bool], answer)}


def _readiness_members(
    snapshot: Snapshot, reference: int, answer: QueryResult[object]
) -> Mapping[str, QueryResult[object]]:
    assessment = snapshot.cache[reference].assessment
    if isinstance(assessment, (ConstraintAssessment, ReadinessAssessment)) and assessment.results:
        return {key: cast(QueryResult[object], value) for key, value in assessment.results.items()}
    return {snapshot.linked.nodes[reference].key: answer}


def _frame(snapshot: Snapshot, node: Node) -> Generator[int | _execution.Call, object, Evaluation]:
    if node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive, assessment=_guard_assessment(node, inactive))

    if node.kind == "param":
        if node.index in snapshot.parameters:
            return Evaluation(Available(snapshot.parameters[node.index]))
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
        return Evaluation(Available(node.value))
    if node.kind == "decision":
        if node.index in snapshot.assignments:
            return Evaluation(Available(snapshot.assignments[node.index]))
        if isinstance(snapshot, _TrialSnapshot) and node.index in snapshot._candidates:
            candidate = snapshot._candidates[node.index]
            admission = yield from _membership_frame(snapshot, node, candidate, check_guard=False)
            if isinstance(admission.result, Available) and admission.result.value is True:
                snapshot.admit(node.index, candidate)
                return Evaluation(Available(snapshot.assignments[node.index]))
            return Evaluation(admission.result)
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
        return Evaluation(cast(QueryResult[object], (yield node.output)))
    if node.kind == "select":
        if node.selector is None:
            if len(node.alternatives) != 1:
                raise EvaluationError(node.owner, "selection", "missing selector")
            selected = node.alternatives[0][1]
        else:
            selector = cast(QueryResult[object], (yield node.selector))
            if not isinstance(selector, Available):
                return Evaluation(selector)
            case = selector.value
            if type(case) is not str:
                raise EvaluationError(node.owner, "selection", "selector is not a declared case")
            if node.selection_index is not None:
                target = node.selection_index.get(case)
            elif len(node.alternatives) == 1 and node.alternatives[0][0] == case:
                target = node.alternatives[0][1]
            else:
                target = None
            if target is None:
                raise EvaluationError(node.owner, "selection", "selector is not a declared case")
            selected = target
        return Evaluation(cast(QueryResult[object], (yield selected)))
    if node.kind == "view":
        if node.output is None:
            raise EvaluationError(node.owner, "view", "missing output reference")
        output = cast(QueryResult[object], (yield node.output))
        constraints: dict[str, QueryResult[bool]] = {}
        for reference in node.constraints:
            answer = cast(QueryResult[object], (yield reference))
            constraints.update(_constraint_members(snapshot, reference, answer))
        requires: dict[str, QueryResult[object]] = {}
        for reference in node.requires:
            answer = cast(QueryResult[object], (yield reference))
            requires.update(_readiness_members(snapshot, reference, answer))
        view = assess_view(output, owner=node.key, requires=requires, constraints=constraints)
        return Evaluation(view.accepted_result, assessment=view)
    if node.kind == "group":
        answers: dict[str, QueryResult[bool]] = {}
        for reference in node.constraints:
            answer = cast(QueryResult[object], (yield reference))
            answers.update(_constraint_members(snapshot, reference, answer))
        group = assess_constraints(answers)
        return Evaluation(cast(QueryResult[object], group.result), assessment=group)
    if node.kind == "readiness":
        obligations: dict[str, QueryResult[object]] = {}
        for reference in node.requires:
            answer = cast(QueryResult[object], (yield reference))
            obligations.update(_readiness_members(snapshot, reference, answer))
        readiness = assess_readiness(obligations)
        return Evaluation(cast(QueryResult[object], readiness.result), assessment=readiness)

    arguments, failure = yield from _arguments(snapshot.linked, node.arguments, node.owner)
    if failure is not None:
        return Evaluation(failure, assessment=_guard_assessment(node, failure))
    if node.function is None:
        raise EvaluationError(node.owner, node.kind, "missing callback")
    try:
        positional = (_self_point(snapshot, node.scope),) if node.call_style == "self" else ()
        called = yield _execution.Call(node.function, positional, arguments, node.kind)
        if isinstance(called, _execution._Halt):
            blocked = _blocked(list(called.results))
            assert blocked is not None
            return Evaluation(blocked, assessment=_guard_assessment(node, blocked))
        assert isinstance(called, _execution._Returned)
        result = called.value
        if node.kind == "constraint":
            normalized = constraint_result(cast(bool | QueryResult[bool], result), node.owner)
            assessment = ConstraintAssessment({node.key: normalized}, normalized)
            return Evaluation(cast(QueryResult[object], normalized), assessment=assessment)
        if isinstance(result, (Inapplicable, Rejected, Unresolved)):
            return Evaluation(owned_result(result, node.owner))
        value = result.value if isinstance(result, Available) else result
        if node.semantics is None:
            raise TypeError("derived output has no value semantics")
        return Evaluation(Available(node.semantics.freeze(value)))
    except Exception as cause:
        raise EvaluationError(node.owner, node.kind, str(cause)) from cause


def _self_point(snapshot: Snapshot, scope: int) -> Space:
    from .compiler import SpaceModel  # noqa: PLC0415 - preparation/execution boundary
    from .occurrence import OccurrenceState, _attach  # noqa: PLC0415

    model = snapshot.model
    if model is None:
        model = SpaceModel(snapshot.linked.scopes[0].space_type, snapshot.linked)
    return _attach(OccurrenceState(model, snapshot), scope)


def evaluate(snapshot: Snapshot, node_index: int) -> Evaluation:
    """Demand one result through the canonical native dispatcher."""
    with snapshot.lock:
        cached = snapshot.cache.get(node_index)
        return cached if cached is not None else _execution.run(snapshot, node_index)


def _membership_frame(
    snapshot: Snapshot, node: Node, value: object, *, check_guard: bool = True
) -> Generator[int | _execution.Call, object, Evaluation]:
    if check_guard and node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive)
    arguments, failure = yield from _arguments(snapshot.linked, node.domain_arguments, node.owner)
    if failure is not None:
        return Evaluation(failure)
    if node.domain is None or node.semantics is None:
        raise EvaluationError(node.owner, "domain membership", "reference has no domain")
    candidate = _clone(node, value, owner=node.owner, role="domain candidate snapshot")
    called = yield _execution.Call(
        node.domain.membership,
        (candidate, arguments),
        {"semantics": node.semantics, "owner": node.owner},
        "domain membership",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return Evaluation(blocked)
    assert isinstance(called, _execution._Returned)
    return Evaluation(cast(QueryResult[object], called.value))


def _enumeration_frame(
    snapshot: Snapshot, node: Node
) -> Generator[int | _execution.Call, object, Evaluation]:
    if node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive)
    arguments, failure = yield from _arguments(snapshot.linked, node.domain_arguments, node.owner)
    if failure is not None:
        return Evaluation(failure)
    assert node.domain is not None and node.semantics is not None
    if node.domain.candidates is None:
        return Evaluation(Available(None))
    called = yield _execution.Call(
        node.domain.enumerate,
        (arguments,),
        {"semantics": node.semantics, "owner": node.owner},
        "domain enumeration",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return Evaluation(blocked)
    assert isinstance(called, _execution._Returned)
    return Evaluation(cast(QueryResult[object], called.value))


def _applicability(snapshot: Snapshot, node: Node) -> NonValue | None:
    if node.guard is None:
        return None
    return _guard_result(node, evaluate(snapshot, node.guard).result)


def decision_state(snapshot: Snapshot, node_index: int) -> QueryResult[DecisionState[object]]:
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
            return Available(DecisionState(node.owner))
        value = _clone(
            node, snapshot.assignments[node_index], owner=node.owner, role="decision state"
        )
        return Available(DecisionState(node.owner, "committed", value, "explicit"))


def candidate_values(snapshot: Snapshot, node_index: int) -> QueryResult[tuple[object, ...]] | None:
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.domain is None or node.semantics is None:
            raise EvaluationError(node.owner, "domain enumeration", "reference has no domain")
        result = _execution.run(snapshot, node_index, _enumeration_frame(snapshot, node)).result
        if isinstance(result, Available) and result.value is None:
            return None
        return cast(QueryResult[tuple[object, ...]], result)


def membership(snapshot: Snapshot, node_index: int, value: object) -> QueryResult[bool]:
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        result = _execution.run(
            snapshot, node_index, _membership_frame(snapshot, node, value)
        ).result
        return cast(QueryResult[bool], result)


def copy_result(
    snapshot: Snapshot, node_index: int, answer: QueryResult[object]
) -> QueryResult[object]:
    """Detach a public answer from cached values using its declared semantics."""

    if not isinstance(answer, Available):
        return answer
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.kind in {"group", "readiness"} and type(answer.value) is bool:
            # Aggregate verdicts are intrinsic immutable booleans; these
            # declarations need no separate user-owned value semantics.
            return answer
        return Available(_clone(node, answer.value, owner=node.owner, role="public value snapshot"))


def _copy_readiness(snapshot: Snapshot, assessment: ReadinessAssessment) -> ReadinessAssessment:
    answers: dict[str, QueryResult[object]] = {}
    for key, answer in assessment.results.items():
        index = snapshot.linked.keys[key]
        if isinstance(answer, Available) and isinstance(answer.value, DecisionState):
            state = answer.value
            node = snapshot.linked.nodes[index]
            if state.status == "committed":
                value = _clone(node, state.value, owner=node.owner, role="public state snapshot")
                state = replace(state, value=value)
            answers[key] = Available(state)
        else:
            answers[key] = copy_result(snapshot, index, answer)
    return ReadinessAssessment(answers, assessment.result)


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
        output = copy_result(snapshot, output_index, assessment.output_result)
        accepted = (
            output
            if assessment.accepted_result is assessment.output_result
            else copy_result(snapshot, node_index, assessment.accepted_result)
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
    "copy_result",
    "copy_assessment",
    "decision_state",
    "evaluate",
    "membership",
]
