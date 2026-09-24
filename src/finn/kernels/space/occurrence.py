# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed occurrence operations over one compiled model and immutable snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypeVar, cast, overload

from . import _runtime
from .compiler import SpaceModel
from .declarations import (
    ChoiceView,
    Constraint,
    ConstraintGroup,
    Decision,
    DecisionRef,
    Readiness,
    Space,
    Subspace,
    SubspaceChoice,
    ValueRef,
    View,
)
from .edits import Edit, EditOutcome, EditRequest, RefinementReport
from .errors import EvaluationError, RefinementError, RequestError
from .ir import Choice, Node
from .results import (
    Answer,
    ConstraintAssessment,
    Decided,
    DecisionState,
    Inapplicable,
    ReadinessAssessment,
    ViewAssessment,
    reject,
)

T = TypeVar("T")
S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class OccurrenceState:
    model: SpaceModel[Space]
    snapshot: _runtime.Snapshot


def state(point: Space) -> OccurrenceState:
    current = getattr(point, "_state", None)
    if not isinstance(current, OccurrenceState):
        raise RequestError("an occurrence must be created by SpaceModel.start")
    return current


def _attach(current: OccurrenceState, scope: int) -> Space:
    space_type = current.model.linked.scopes[scope].space_type
    instance = object.__new__(space_type)
    instance._state = current
    instance._scope = scope
    return instance


def _recognize_request_value(node: Node, value: object, role: str) -> None:
    assert node.semantics is not None
    try:
        recognized = node.semantics.accepts(value)
    except Exception as cause:
        raise EvaluationError(node.owner, f"{role} recognition", str(cause)) from cause
    if not recognized:
        raise RequestError(f"{node.key}: expected value of nominal type {node.semantics.name}")


def _snapshot_request_value(node: Node, value: object, role: str) -> object:
    assert node.semantics is not None
    try:
        return node.semantics.freeze(value)
    except Exception as cause:
        raise EvaluationError(node.owner, f"{role} snapshot", str(cause)) from cause


def start(model: SpaceModel[S], parameters: Mapping[object, object]) -> S:
    """Validate all bindings and freeze external values before any evaluation."""
    if not isinstance(parameters, Mapping):
        raise RequestError("parameters must be a declaration-keyed mapping")
    pending: dict[int, object] = {}
    for ref, value in parameters.items():
        index = model.resolve(0, ref)
        node = model.linked.nodes[index]
        if node.kind != "param" or index not in model.linked.parameters:
            raise RequestError(f"{node.key} is not an exposed parameter")
        if index in pending:
            raise RequestError(f"{node.key} is bound more than once")
        pending[index] = value
    missing = [
        model.linked.nodes[i].key
        for i in model.linked.parameters
        if model.linked.nodes[i].required and i not in pending
    ]
    if missing:
        raise RequestError(f"missing required parameters: {', '.join(missing)}")
    # Validate every nominal value before invoking snapshot adapters. A bad
    # later binding cannot start freezing an otherwise valid earlier binding.
    for index, value in pending.items():
        _recognize_request_value(model.linked.nodes[index], value, "parameter")
    frozen = {
        index: _snapshot_request_value(model.linked.nodes[index], value, "parameter")
        for index, value in pending.items()
    }
    snapshot = _runtime.Snapshot(model.linked, frozen)
    return cast(S, _attach(OccurrenceState(cast(SpaceModel[Space], model), snapshot), 0))


def answer(point: Space, reference: ValueRef[T]) -> Answer[T]:
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    result = _runtime.evaluate(current.snapshot, index).answer
    return cast(Answer[T], _runtime.copy_answer(current.snapshot, index, result))


def read_value(point: Space, reference: ValueRef[T]) -> T:
    result = answer(point, reference)
    if not isinstance(result, Decided):
        raise RequestError(f"value is {type(result).__name__}; inspect point.answer(reference)")
    return result.value


@overload
def assess(point: Space, reference: View[T]) -> ViewAssessment[T]: ...


@overload
def assess(point: Space, reference: Constraint | ConstraintGroup) -> ConstraintAssessment: ...


@overload
def assess(point: Space, reference: Readiness) -> ReadinessAssessment: ...


def assess(
    point: Space,
    reference: View[T] | Constraint | ConstraintGroup | Readiness,
) -> ViewAssessment[T] | ConstraintAssessment | ReadinessAssessment:
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    entry = _runtime.evaluate(current.snapshot, index)
    if entry.assessment is None:
        raise RequestError("this declaration is not assessable")
    return cast(
        ViewAssessment[T] | ConstraintAssessment | ReadinessAssessment,
        _runtime.copy_assessment(current.snapshot, index, entry.assessment),
    )


def decision_state(
    point: Space,
    reference: Decision[T] | DecisionRef[T],
) -> Answer[DecisionState[T]]:
    current = state(point)
    index = _decision(point, reference)
    # Runtime state reads already snapshot committed values and contextualize
    # adapter failures before crossing this boundary.
    result = _runtime.decision_state(current.snapshot, index)
    return cast(Answer[DecisionState[T]], result)


def _decision(point: Space, reference: object) -> int:
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    if current.model.linked.nodes[index].kind != "decision":
        raise RequestError("assignment requires an owning Decision or DecisionRef")
    # A plain parameter alias must not gain edit rights through its supplier.
    if not isinstance(reference, (Decision, DecisionRef)):
        raise RequestError("parameter aliases are not independently editable")
    return index


def candidates(
    point: Space,
    reference: Decision[T] | DecisionRef[T],
) -> Answer[tuple[T, ...]] | None:
    current = state(point)
    return cast(
        Answer[tuple[T, ...]] | None,
        _runtime.candidate_values(current.snapshot, _decision(point, reference)),
    )


def edit(point: Space, reference: Decision[T] | DecisionRef[T], value: T) -> Edit[T]:
    current = state(point)
    index = _decision(point, reference)
    return Edit(current.snapshot, current.model.linked.nodes[index].scope, index, value)


def refine(point: S, *edits: EditRequest) -> RefinementReport[S]:
    """Normalize a complete batch, trial it in graph order, publish atomically."""
    current = state(point)
    if type(point._scope) is not int or point._scope != 0:
        raise RequestError("atomic refinement is a root operation")
    pending: dict[int, object] = {}
    # Validate structure for the entire batch before invoking any value adapter.
    # In particular, a foreign or duplicate final edit cannot run the first
    # candidate's snapshot or membership callback.
    for item in edits:
        if not isinstance(item, Edit):
            raise RequestError("refine expects scoped Edit requests")
        if item.base is not current.snapshot:
            raise RequestError("all edits must target this exact base snapshot")
        if type(item.scope) is not int or not 0 <= item.scope < len(current.model.linked.scopes):
            raise RequestError("edit scope does not belong to this model")
        if type(item.node) is not int or not 0 <= item.node < len(current.model.linked.nodes):
            raise RequestError("edit node does not belong to this model")
        node = current.model.linked.nodes[item.node]
        if node.kind != "decision" or node.scope != item.scope:
            raise RequestError("edit does not identify an owned decision in its scope")
        if item.node in pending:
            raise RequestError(f"duplicate edit for {node.key}")
        pending[item.node] = item.value
    with current.snapshot.lock:
        for index, value in pending.items():
            _recognize_request_value(current.model.linked.nodes[index], value, "candidate")
        prepared = {
            index: _snapshot_request_value(current.model.linked.nodes[index], value, "candidate")
            for index, value in pending.items()
        }
        trial = _runtime._TrialSnapshot(current.snapshot)
        changed = False
        outcomes: dict[int, EditOutcome] = {}
        for index in sorted(prepared, key=current.model.linked.ranks.__getitem__):
            node = current.model.linked.nodes[index]
            candidate = prepared[index]
            assert node.semantics is not None
            if index in trial.assignments:
                try:
                    prior = node.semantics.freeze(trial.assignments[index])
                    comparison = node.semantics.freeze(candidate)
                    equal = node.semantics.values_equal(prior, comparison)
                except Exception as cause:
                    raise EvaluationError(node.owner, "commitment equality", str(cause)) from cause
                if equal:
                    outcomes[index] = EditOutcome(node.owner, Decided(True), "unchanged")
                else:
                    outcomes[index] = EditOutcome(
                        node.owner,
                        reject(
                            "commitment-conflict",
                            "a committed decision cannot change",
                            owner=node.owner,
                        ),
                        "refused",
                    )
                continue
            admissible = _runtime.membership(trial, index, candidate)
            if isinstance(admissible, Decided) and admissible.value is True:
                trial.admit(index, candidate)
                changed = True
                outcomes[index] = EditOutcome(node.owner, admissible, "provisional")
            else:
                outcomes[index] = EditOutcome(node.owner, admissible, "refused")
        accepted = all(item.status != "refused" for item in outcomes.values())
        if len(outcomes) != len(prepared):
            raise RequestError("compiled refinement order is incomplete")
        published = point
        if accepted and changed:
            published = cast(S, _attach(OccurrenceState(current.model, trial.publish()), 0))
        ordered = tuple(outcomes[item.node] for item in edits)
        if accepted:
            ordered = tuple(
                replace(item, status="committed") if item.status == "provisional" else item
                for item in ordered
            )
        return RefinementReport(published, accepted, ordered)


def assign(point: S, reference: Decision[T] | DecisionRef[T], value: T) -> S:
    current = state(point)
    report = refine(root(point), cast(Edit[object], edit(point, reference, value)))
    if not report.accepted:
        raise RefinementError(report)
    successor = state(report.point)
    if successor.snapshot is current.snapshot:
        return point
    return cast(S, _attach(successor, point._scope))


def root(point: Space) -> Space:
    return point if point._scope == 0 else _attach(state(point), 0)


def child(point: Space, placement: Subspace[S]) -> S:
    current = state(point)
    scope = current.model.linked.scopes[point._scope]
    try:
        child_scope = scope.children[placement]
    except KeyError as error:
        raise RequestError("child placement is not part of this compiled scope") from error
    return cast(S, _attach(current, child_scope))


def choice(point: Space, declaration: SubspaceChoice) -> ChoiceView:
    view = ChoiceView(point, declaration)
    _choice_record(view)
    return view


def _choice_record(view: ChoiceView) -> tuple[OccurrenceState, Choice]:
    current = state(view.occurrence)
    scope = current.model.linked.scopes[view.occurrence._scope]
    try:
        index = scope.choices[view.declaration]
    except (KeyError, TypeError) as cause:
        raise RequestError("choice is not part of this compiled scope") from cause
    return current, current.model.linked.choices[index]


def choice_alternatives(view: ChoiceView) -> tuple[str, ...]:
    _, declaration = _choice_record(view)
    return tuple(key for key, _ in declaration.cases)


def _case_scope(declaration: Choice, case: str) -> int:
    if type(case) is not str:
        raise RequestError("a choice case must be a string key")
    for key, scope in declaration.cases:
        if key == case:
            return scope
    raise RequestError(f"{declaration.key}: unknown choice case {case!r}")


def select(view: ChoiceView, case: str) -> ChoiceView:
    current, declaration = _choice_record(view)
    _case_scope(declaration, case)
    if declaration.selector is None:
        # A singleton case is structurally selected, but an enclosing guard
        # still determines whether selection applies at this snapshot.
        if declaration.guard is not None:
            guard = _runtime.evaluate(current.snapshot, declaration.guard).answer
            if not isinstance(guard, Decided) or guard.value is not True:
                refusal: Answer[bool] = (
                    Inapplicable() if isinstance(guard, Decided) else cast(Answer[bool], guard)
                )
                report = RefinementReport(
                    root(view.occurrence),
                    False,
                    (EditOutcome(declaration.key, refusal, "refused"),),
                )
                raise RefinementError(report)
        return view
    selector = current.model.linked.nodes[declaration.selector]
    report = refine(
        root(view.occurrence),
        Edit(current.snapshot, selector.scope, selector.index, case),
    )
    if not report.accepted:
        raise RefinementError(report)
    successor = state(report.point)
    if successor.snapshot is current.snapshot:
        return view
    owner = _attach(successor, view.occurrence._scope)
    return ChoiceView(owner, view.declaration)


def alternative(view: ChoiceView, case: str) -> Space:
    current, declaration = _choice_record(view)
    return _attach(current, _case_scope(declaration, case))
