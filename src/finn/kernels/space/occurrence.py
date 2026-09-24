# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed occurrence operations over one compiled model and immutable snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, TypeVar, cast, overload

from . import _runtime
from .compiler import SpaceModel
from .declarations import (
    BoundDecision,
    BoundValue,
    BoundViewField,
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
from .edits import (
    Change,
    ChangeOutcome,
    ChangeRequest,
    CommitmentReport,
    ConfigurationResult,
)
from .errors import ConfigurationError, EvaluationError, RequestError
from .ir import Choice, Node
from .results import (
    QueryResult,
    ConstraintAssessment,
    Available,
    DecisionState,
    Inapplicable,
    ReadinessAssessment,
    ViewAssessment,
    reject,
    require_value,
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
        raise RequestError("an instance must be created by Space construction or SpaceModel.bind")
    return current


def _attach(current: OccurrenceState, scope: int) -> Space:
    space_type = current.model.linked.scopes[scope].space_type
    instance = object.__new__(space_type)
    object.__setattr__(instance, "_state", current)
    object.__setattr__(instance, "_scope", scope)
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


def bind(
    model: SpaceModel[S],
    parameters: Mapping[object, object],
    keyword_parameters: Mapping[str, object],
) -> S:
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
    root_scope = model.linked.scopes[0]
    for name, value in keyword_parameters.items():
        keyword_index = root_scope.named_members.get(name)
        if keyword_index is None:
            raise RequestError(f"unknown parameter {name!r}")
        node = model.linked.nodes[keyword_index]
        if node.kind != "param" or keyword_index not in model.linked.parameters:
            raise RequestError(f"{node.key} is not an exposed parameter")
        if keyword_index in pending:
            raise RequestError(f"{node.key} is bound more than once")
        pending[keyword_index] = value
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


def query(point: Space, reference: ValueRef[T] | View[T]) -> QueryResult[T]:
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    result = _runtime.evaluate(current.snapshot, index).result
    return cast(QueryResult[T], _runtime.copy_result(current.snapshot, index, result))


def read_value(point: Space, reference: ValueRef[T]) -> T:
    return require_value(query(point, reference), context=(point, reference))


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
) -> QueryResult[DecisionState[T]]:
    current = state(point)
    index = _decision(point, reference)
    # Runtime state reads already snapshot committed values and contextualize
    # adapter failures before crossing this boundary.
    result = _runtime.decision_state(current.snapshot, index)
    return cast(QueryResult[DecisionState[T]], result)


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
) -> QueryResult[tuple[T, ...]] | None:
    current = state(point)
    return cast(
        QueryResult[tuple[T, ...]] | None,
        _runtime.candidate_values(current.snapshot, _decision(point, reference)),
    )


def change(point: Space, reference: Decision[T] | DecisionRef[T], value: T) -> Change[T]:
    current = state(point)
    index = _decision(point, reference)
    return Change(current.snapshot, current.model.linked.nodes[index].scope, index, value)


def clear(point: Space, reference: Decision[T] | DecisionRef[T]) -> Change[T]:
    current = state(point)
    index = _decision(point, reference)
    return Change(current.snapshot, current.model.linked.nodes[index].scope, index, remove=True)


def bind_field(
    point: Space, reference: ValueRef[T] | View[T]
) -> BoundValue[T] | BoundDecision[T] | BoundViewField[T]:
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    node = current.model.linked.nodes[index]
    if isinstance(reference, View):
        return BoundViewField(point, reference)
    if node.kind == "decision" and isinstance(reference, (Decision, DecisionRef)):
        return BoundDecision(point, reference)
    return BoundValue(point, reference)


def _normalize_changes(
    point: Space, changes: tuple[ChangeRequest, ...]
) -> dict[int, Change[object]]:
    current = state(point)
    pending: dict[int, Change[object]] = {}
    for item in changes:
        if not isinstance(item, Change):
            raise RequestError("changes must come from a bound field or refinement.change()")
        if item.base is not current.snapshot:
            raise RequestError("all changes must target this exact base snapshot")
        if type(item.scope) is not int or not 0 <= item.scope < len(current.model.linked.scopes):
            raise RequestError("change scope does not belong to this model")
        if type(item.node) is not int or not 0 <= item.node < len(current.model.linked.nodes):
            raise RequestError("change node does not belong to this model")
        node = current.model.linked.nodes[item.node]
        if node.kind != "decision" or node.scope != item.scope:
            raise RequestError("change does not identify an owned decision in its scope")
        if item.node in pending:
            raise RequestError(f"duplicate change for {node.key}")
        pending[item.node] = cast(Change[object], item)
    return pending


def commit(point: S, *changes: ChangeRequest) -> CommitmentReport[S]:
    """Atomically add monotone commitments without revising existing choices."""

    current = state(point)
    normalized = _normalize_changes(point, changes)
    if any(item.remove for item in normalized.values()):
        raise RequestError("monotone commitment does not accept removal requests")
    pending: dict[int, object] = {}
    for index, item in normalized.items():
        pending[index] = item.value
    with current.snapshot.lock:
        for index, value in pending.items():
            _recognize_request_value(current.model.linked.nodes[index], value, "candidate")
        prepared = {
            index: _snapshot_request_value(current.model.linked.nodes[index], value, "candidate")
            for index, value in pending.items()
        }
        trial = _runtime._TrialSnapshot(current.snapshot)
        changed = False
        outcomes: dict[int, ChangeOutcome] = {}
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
                    outcomes[index] = ChangeOutcome(node.owner, Available(True), "unchanged")
                else:
                    outcomes[index] = ChangeOutcome(
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
            if isinstance(admissible, Available) and admissible.value is True:
                trial.admit(index, candidate)
                changed = True
                outcomes[index] = ChangeOutcome(node.owner, admissible, "admissible")
            else:
                outcomes[index] = ChangeOutcome(node.owner, admissible, "refused")
        accepted = all(item.status != "refused" for item in outcomes.values())
        if len(outcomes) != len(prepared):
            raise RequestError("compiled refinement order is incomplete")
        published = point
        if accepted and changed:
            published = cast(S, _attach(OccurrenceState(current.model, trial.publish()), 0))
        ordered = tuple(outcomes[item.node] for item in changes)
        if accepted:
            ordered = tuple(
                replace(item, status="committed") if item.status == "admissible" else item
                for item in ordered
            )
        if state(published).snapshot is not current.snapshot and point._scope != 0:
            published = cast(S, _attach(state(published), point._scope))
        return CommitmentReport(published, accepted, ordered)


def _keyword_change(point: Space, name: str, value: object) -> Change[object]:
    current = state(point)
    scope = current.model.linked.scopes[point._scope]
    index = scope.named_members.get(name)
    if index is None:
        for declaration, choice_index in scope.choices.items():
            if isinstance(declaration, SubspaceChoice) and declaration.name == name:
                choice = current.model.linked.choices[choice_index]
                if choice.selector is None:
                    if type(value) is not str or value not in tuple(
                        case for case, _ in choice.cases
                    ):
                        raise RequestError(f"{choice.key}: unknown choice case {value!r}")
                    raise RequestError(f"{choice.key} is a singleton structural choice")
                index = choice.selector
                break
    if index is None:
        raise RequestError(f"unknown direct choice {name!r}")
    node = current.model.linked.nodes[index]
    if node.kind != "decision" or node.scope != point._scope:
        raise RequestError(f"{node.key} is not a direct owned choice in this scope")
    return Change(current.snapshot, node.scope, index, value)


def _values_equal(node: Node, left: object, right: object) -> bool:
    assert node.semantics is not None
    try:
        return node.semantics.values_equal(
            node.semantics.freeze(left), node.semantics.freeze(right)
        )
    except Exception as cause:
        raise EvaluationError(node.owner, "configuration equality", str(cause)) from cause


def try_with_choices(
    point: S, /, *changes: ChangeRequest, **choices: object
) -> ConfigurationResult[S]:
    """Build and validate a replacement choice set over the same frozen facts."""

    current = state(point)
    keyword_changes = tuple(_keyword_change(point, name, value) for name, value in choices.items())
    all_changes = (*changes, *keyword_changes)
    normalized = _normalize_changes(point, all_changes)
    with current.snapshot.lock:
        for index, item in normalized.items():
            if not item.remove:
                _recognize_request_value(current.model.linked.nodes[index], item.value, "choice")
        prepared = {
            index: (
                item
                if item.remove
                else replace(
                    item,
                    value=_snapshot_request_value(
                        current.model.linked.nodes[index], item.value, "choice"
                    ),
                )
            )
            for index, item in normalized.items()
        }
        merged = dict(current.snapshot.assignments)
        for index, item in prepared.items():
            if item.remove:
                merged.pop(index, None)
            else:
                merged[index] = item.value

        semantically_same = len(merged) == len(current.snapshot.assignments) and all(
            index in current.snapshot.assignments
            and _values_equal(
                current.model.linked.nodes[index], value, current.snapshot.assignments[index]
            )
            for index, value in merged.items()
        )
        if semantically_same:
            no_op_outcomes = tuple(
                ChangeOutcome(
                    current.model.linked.nodes[item.node].owner,
                    Available(True),
                    "unchanged",
                )
                for item in all_changes
            )
            return ConfigurationResult(point, True, no_op_outcomes)

        base = _runtime.Snapshot(
            current.model.linked, current.snapshot.parameters, {}, current.snapshot.lock
        )
        trial = _runtime._TrialSnapshot(base)
        validation: dict[int, QueryResult[bool]] = {}
        for index in sorted(merged, key=current.model.linked.ranks.__getitem__):
            candidate = merged[index]
            admissible = _runtime.membership(trial, index, candidate)
            validation[index] = admissible
            if isinstance(admissible, Available) and admissible.value is True:
                trial.admit(index, candidate)

        refused = {
            index
            for index, result in validation.items()
            if not (isinstance(result, Available) and result.value is True)
        }
        outcomes: list[ChangeOutcome] = []
        requested_nodes = set(normalized)
        for request in all_changes:
            item = normalized[request.node]
            node = current.model.linked.nodes[item.node]
            result = validation.get(item.node, Available(True))
            status: Literal["refused", "admissible"] = (
                "refused" if item.node in refused else "admissible"
            )
            outcomes.append(ChangeOutcome(node.owner, result, status))
        for index in sorted(refused - requested_nodes, key=current.model.linked.ranks.__getitem__):
            node = current.model.linked.nodes[index]
            outcomes.append(
                ChangeOutcome(node.owner, validation[index], "refused", requested=False)
            )
        if refused:
            return ConfigurationResult(point, False, tuple(outcomes))

        snapshot = trial.publish()
        successor = cast(S, _attach(OccurrenceState(current.model, snapshot), point._scope))
        published: list[ChangeOutcome] = []
        for request, outcome in zip(all_changes, outcomes):
            item = prepared[request.node]
            if item.remove and item.node in current.snapshot.assignments:
                final_status: Literal["removed", "unchanged", "changed"] = "removed"
            elif item.remove:
                final_status = "unchanged"
            elif item.node in current.snapshot.assignments and _values_equal(
                current.model.linked.nodes[item.node],
                item.value,
                current.snapshot.assignments[item.node],
            ):
                final_status = "unchanged"
            else:
                final_status = "changed"
            published.append(replace(outcome, status=final_status))
        return ConfigurationResult(successor, True, tuple(published))


def with_choices(point: S, /, *changes: ChangeRequest, **choices: object) -> S:
    report = try_with_choices(point, *changes, **choices)
    if not report.accepted:
        raise ConfigurationError(report)
    return report.instance


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
    current = state(view.instance)
    scope = current.model.linked.scopes[view.instance._scope]
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
            guard = _runtime.evaluate(current.snapshot, declaration.guard).result
            if not isinstance(guard, Available) or guard.value is not True:
                refusal: QueryResult[bool] = (
                    Inapplicable()
                    if isinstance(guard, Available)
                    else cast(QueryResult[bool], guard)
                )
                report = ConfigurationResult(
                    root(view.instance),
                    False,
                    (ChangeOutcome(declaration.key, refusal, "refused"),),
                )
                raise ConfigurationError(report)
        return view
    selector = current.model.linked.nodes[declaration.selector]
    report = try_with_choices(
        root(view.instance),
        Change(current.snapshot, selector.scope, selector.index, case),
    )
    if not report.accepted:
        raise ConfigurationError(report)
    successor = state(report.instance)
    if successor.snapshot is current.snapshot:
        return view
    owner = _attach(successor, view.instance._scope)
    return ChoiceView(owner, view.declaration)


def alternative(view: ChoiceView, case: str) -> Space:
    current, declaration = _choice_record(view)
    return _attach(current, _case_scope(declaration, case))
