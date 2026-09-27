# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed occurrence operations over one compiled model and immutable snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar, cast, overload

from . import _execution, _runtime
from ._configuration import BoundDecision, BoundValue, BoundView, ChoiceView, Space
from ._runtime import Snapshot
from .compiler import SpaceModel
from .declarations import (
    Constraint,
    ConstraintGroup,
    Decision,
    DecisionRef,
    Subspace,
    SubspaceChoice,
    ValueRef,
    View,
)
from .errors import EvaluationError, RequestError
from .ir import Choice, LinkedModel, Node
from .results import (
    ConstraintAssessment,
    DecisionState,
    QueryResult,
    ViewAssessment,
    require_value,
)

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def state(point: Space) -> Snapshot:
    current = getattr(point, "_state", None)
    if not isinstance(current, Snapshot):
        raise RequestError("an instance must be created by Space construction or SpaceModel.bind")
    return current


def _attach(current: Snapshot, scope: int) -> Space:
    space_type = current.linked.scopes[scope].space_type
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


def _prepare_values(
    linked: LinkedModel,
    pending: Mapping[int, object],
    role: str,
) -> dict[int, object]:
    """Recognize the entire request before any snapshot adapter is invoked."""
    for index, value in pending.items():
        _recognize_request_value(linked.nodes[index], value, role)
    return {
        index: _snapshot_request_value(linked.nodes[index], value, role)
        for index, value in pending.items()
    }


def bind(
    model: SpaceModel[S],
    parameters: Mapping[object, object],
    keyword_parameters: Mapping[str, object],
) -> S:
    """Validate all bindings and freeze external values before any evaluation."""
    _execution.driver_only("configuration binding")
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
    frozen = _prepare_values(model.linked, pending, "parameter")
    snapshot = _runtime.Snapshot(cast(SpaceModel[Space], model), frozen)
    return cast(S, _attach(snapshot, 0))


def query(point: Space, reference: ValueRef[T] | View[T]) -> QueryResult[T]:
    _execution.driver_only("query inspection")
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    result = _runtime.evaluate(current, index).result
    return cast(QueryResult[T], _runtime.copy_result(current, index, result))


def read_value(point: Space, reference: ValueRef[T] | View[T]) -> T:
    active = _execution.current()
    try:
        return _read_value(point, reference)
    except BaseException as cause:
        if active is not None:
            _execution._remember_exception(active, cause)
        raise


def _read_value(point: Space, reference: ValueRef[T] | View[T]) -> T:
    snapshot = state(point)
    _execution.check_snapshot(snapshot)
    index = snapshot.model.resolve(point._scope, reference)
    snapshot.work.getter_attempts += 1
    active = _execution.current()
    if active is None:
        entry = _runtime.evaluate(snapshot, index)
    else:
        active.dependencies[index] = None
        if index in snapshot.cache:
            outcome: object = snapshot.cache[index]
        else:
            outcome = active.native_parent.switch(index)
        entry = _execution.accept_read(active, index, outcome)
    answer = _runtime.copy_result(snapshot, index, entry.result)
    return cast(T, require_value(answer, context=snapshot.model.linked.nodes[index].owner))


def bind_view(point: Space, reference: View[T]) -> BoundView[T]:
    current = state(point)
    _execution.check_snapshot(current)
    index = current.model.resolve(point._scope, reference)
    if current.linked.nodes[index].kind != "view":
        raise RequestError("view binding requires a View declaration")
    return BoundView(point, reference)


@overload
def inspect(point: Space, reference: View[T]) -> ViewAssessment[T]: ...


@overload
def inspect(point: Space, reference: Constraint | ConstraintGroup) -> ConstraintAssessment: ...


def inspect(
    point: Space,
    reference: View[T] | Constraint | ConstraintGroup,
) -> ViewAssessment[T] | ConstraintAssessment:
    _execution.driver_only("assessment inspection")
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    entry = _runtime.evaluate(current, index)
    if entry.assessment is None:
        raise RequestError("this declaration is not assessable")
    return cast(
        ViewAssessment[T] | ConstraintAssessment,
        _runtime.copy_assessment(current, index, entry.assessment),
    )


def decision_state(
    point: Space,
    reference: Decision[T] | DecisionRef[T],
) -> QueryResult[DecisionState[T]]:
    _execution.driver_only("decision state")
    current = state(point)
    index = _decision(point, reference)
    # Runtime state reads already snapshot committed values and contextualize
    # adapter failures before crossing this boundary.
    result = _runtime.decision_state(current, index)
    return cast(QueryResult[DecisionState[T]], result)


def _decision(point: Space, reference: object) -> int:
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    if current.linked.nodes[index].kind != "decision":
        raise RequestError("assignment requires an owning Decision or DecisionRef")
    # A plain parameter alias must not gain edit rights through its supplier.
    if not isinstance(reference, (Decision, DecisionRef)):
        raise RequestError("parameter aliases are not independently editable")
    return index


def candidates(
    point: Space,
    reference: Decision[T] | DecisionRef[T],
) -> QueryResult[tuple[T, ...]] | None:
    _execution.driver_only("domain enumeration")
    current = state(point)
    return cast(
        QueryResult[tuple[T, ...]] | None,
        _runtime.candidate_values(current, _decision(point, reference)),
    )


def bind_field(
    point: Space, reference: ValueRef[T] | View[T]
) -> BoundValue[T] | BoundDecision[T] | BoundView[T]:
    current = state(point)
    _execution.check_snapshot(current)
    index = current.model.resolve(point._scope, reference)
    node = current.linked.nodes[index]
    if isinstance(reference, View):
        return BoundView(point, reference)
    if node.kind == "decision" and isinstance(reference, (Decision, DecisionRef)):
        return BoundDecision(point, reference)
    return BoundValue(point, reference)


def root(point: Space) -> Space:
    current = state(point)
    _execution.check_snapshot(current)
    return point if point._scope == 0 else _attach(current, 0)


def child(point: Space, placement: Subspace[S]) -> S:
    current = state(point)
    _execution.check_snapshot(current)
    scope = current.linked.scopes[point._scope]
    try:
        child_scope = scope.children[placement]
    except KeyError as error:
        raise RequestError("child placement is not part of this compiled scope") from error
    return cast(S, _attach(current, child_scope))


def choice(point: Space, declaration: SubspaceChoice) -> ChoiceView:
    _execution.check_snapshot(state(point))
    view = ChoiceView(point, declaration)
    _choice_record(view)
    return view


def _choice_record(view: ChoiceView) -> tuple[Snapshot, Choice]:
    current = state(view.instance)
    scope = current.linked.scopes[view.instance._scope]
    try:
        index = scope.choices[view.declaration]
    except (KeyError, TypeError) as cause:
        raise RequestError("choice is not part of this compiled scope") from cause
    return current, current.linked.choices[index]


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


def alternative(view: ChoiceView, case: str) -> Space:
    current, declaration = _choice_record(view)
    _execution.check_snapshot(current)
    return _attach(current, _case_scope(declaration, case))
