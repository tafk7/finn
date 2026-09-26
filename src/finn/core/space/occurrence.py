# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed occurrence operations over one compiled model and immutable snapshot."""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar, cast, overload

from . import _execution, _runtime
from ._configuration import BoundDecision, BoundValue, BoundView, Space
from ._runtime import Snapshot
from .compiler import SpaceModel
from .declarations import (
    Constraint,
    ConstraintGroup,
    Declaration,
    ValueRef,
    View,
    declared_path,
)
from .errors import EvaluationError, RequestError
from .ir import LinkedModel, Node
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
    current = vars(point).get("_state")
    if not isinstance(current, Snapshot):
        if declared_path(point) is not None:
            raise RequestError(
                f"{point!r} is a node declaration, not a configuration; configure() its root"
            )
        raise RequestError("an instance must be created by configure()")
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


def bind(model: SpaceModel[S], parameters: Mapping[int, object]) -> S:
    """Validate all root inputs and freeze them before any evaluation."""
    _execution.driver_only("configuration binding")
    for index in parameters:
        if index not in model.linked.parameters:
            raise RequestError(f"{model.linked.nodes[index].key} is not an exposed parameter")
    missing = [
        model.linked.nodes[i].key
        for i in model.linked.parameters
        if model.linked.nodes[i].required and i not in parameters
    ]
    if missing:
        raise RequestError(f"missing required parameters: {', '.join(missing)}")
    frozen = _prepare_values(model.linked, parameters, "parameter")
    snapshot = _runtime.Snapshot(cast(SpaceModel[Space], model), frozen)
    return cast(S, _attach(snapshot, 0))


def query(point: Space, reference: object) -> QueryResult[object]:
    _execution.driver_only("query inspection")
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    result = _runtime.evaluate(current, index).result
    return _runtime.copy_result(current, index, result)


def read_value(point: Space, reference: ValueRef[T] | View[T]) -> T:
    active = _execution.current()
    try:
        return _read_value(point, reference)
    except BaseException as cause:
        if active is not None:
            _execution._remember_exception(active, cause)
        raise


def _read_index(point: Space, index: int) -> object:
    snapshot = state(point)
    _execution.check_snapshot(snapshot)
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
    return require_value(answer, context=snapshot.model.linked.nodes[index].owner)


def _read_value(point: Space, reference: ValueRef[T] | View[T]) -> T:
    snapshot = state(point)
    _execution.check_snapshot(snapshot)
    index = snapshot.model.resolve(point._scope, reference)
    return cast(T, _read_index(point, index))


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


def decision_state(point: Space, reference: object) -> QueryResult[DecisionState[object]]:
    _execution.driver_only("decision state")
    current = state(point)
    index = decision_index(point, reference)
    # Runtime state reads already snapshot committed values and contextualize
    # adapter failures before crossing this boundary.
    return _runtime.decision_state(current, index)


def decision_index(point: Space, reference: object) -> int:
    current = state(point)
    return current.model.decision(point._scope, reference)


def candidates(point: Space, reference: object) -> QueryResult[tuple[object, ...]] | None:
    _execution.driver_only("domain enumeration")
    current = state(point)
    return _runtime.candidate_values(current, decision_index(point, reference))


def bind_field(
    point: Space, reference: ValueRef[T] | View[T]
) -> BoundValue[T] | BoundDecision[T] | BoundView[T]:
    current = state(point)
    _execution.check_snapshot(current)
    index = current.model.resolve(point._scope, reference)
    node = current.linked.nodes[index]
    if isinstance(reference, View):
        return BoundView(point, reference)
    if node.kind == "view":
        return BoundView(point, cast(View[T], reference))
    try:
        current.model.decision(point._scope, reference)
    except RequestError:
        return BoundValue(point, reference)
    return BoundDecision(point, reference)


def root(point: Space) -> Space:
    current = state(point)
    _execution.check_snapshot(current)
    return point if point._scope == 0 else _attach(current, 0)


def child(point: Space, record: Declaration) -> Space:
    current = state(point)
    _execution.check_snapshot(current)
    scope = current.linked.scopes[point._scope]
    try:
        child_scope = scope.children[record]
    except KeyError as error:
        raise RequestError("child node is not part of this compiled scope") from error
    return _attach(current, child_scope)


def selected_candidate(point: Space, decision: Declaration) -> Space | None:
    """The configuration of the candidate a Decision over nodes selects, or None."""
    current = state(point)
    _execution.check_snapshot(current)
    scope = current.linked.scopes[point._scope]
    try:
        choice = current.linked.choices[scope.choices[decision]]
    except KeyError as error:
        raise RequestError("the Decision over nodes is not part of this compiled scope") from error
    case = _read_index(point, choice.selector)
    for key, candidate in choice.cases:
        if key == case:
            return None if candidate is None else _attach(current, candidate)
    raise EvaluationError(choice.key, "selection", "selector is not a declared case")


def candidate(point: Space, decision: object, case: str) -> Space | None:
    """A candidate's configuration whether or not it is selected (None for a None case)."""
    from ._nodes import unwrap

    current = state(point)
    _execution.check_snapshot(current)
    if type(case) is not str:
        raise RequestError("a candidate key must be a string")
    record = unwrap(decision)
    scope = current.linked.scopes[point._scope]
    try:
        choice = current.linked.choices[scope.choices[record]]
    except (KeyError, TypeError) as error:
        raise RequestError("the Decision over nodes is not part of this compiled scope") from error
    for key, index in choice.cases:
        if key == case:
            return None if index is None else _attach(current, index)
    raise RequestError(f"{choice.key}: unknown candidate {case!r}")
