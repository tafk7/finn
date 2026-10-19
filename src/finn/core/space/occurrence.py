# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The read boundary: typed operations on a configuration, or on a method's receiver.

A Space object that reads values is one of two things. A **configuration**
(``design_space``, a change) reads through its state (``_state.State``): each read is
a driver demand, evaluated in the configuration's world under the model's lock. A
**receiver**, the ``self`` a ``self`` method runs with, reads through the method's
reader (``_world.MethodReader``): each read is evaluated in the world the method runs
in, recorded, and checked against what the method's source may read. A configuration
read during an authored call is that call's failure (a method reads through its
receiver), and so is a driver-only operation.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar, cast, overload

from . import _state
from ._configuration import BoundDecision, BoundValue, Space
from ._failure import check_transformation, current, driver_only, fault
from ._state import State
from ._world import MethodReader
from .compiler import Model
from .declarations import (
    Constraint,
    Declaration,
    Param,
    ValueRef,
    View,
    declared_path,
)
from .errors import EvaluationError, RequestError
from .ir import Choice, LinkedModel, NodeId
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    Unresolved,
    ViewAssessment,
)
from .semantics import snapshot

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def _reader(point: Space) -> MethodReader | None:
    """The method reader ``point`` reads through, where it is a method's receiver."""
    found = vars(point).get("_state")
    return found if isinstance(found, MethodReader) else None


def state(point: Space) -> State:
    """A configuration's state."""
    found = vars(point).get("_state")
    if not isinstance(found, State):
        if declared_path(point) is not None:
            raise RequestError(
                f"{point!r} is a node declaration, not a configuration; open its root's "
                "design space with design_space()"
            )
        raise RequestError("a configuration is created by design_space()")
    return found


def _driver(point: Space) -> State:
    """A configuration's state, for a read: refused in a value transformation, and inside
    an authored call (it reads through its receiver), sticky there."""
    found = state(point)
    check_transformation()
    active = current()
    if active is not None:
        error = EvaluationError(active.owner, "field read", "cross-configuration access")
        fault(active, error)
        raise error
    return found


def model_of(point: Space) -> Model[Space]:
    """The compiled model a configuration or a receiver reads."""
    reader = _reader(point)
    if reader is not None:
        return reader.world.evaluator.model
    return state(point).model


def attach(current: State, scope: int) -> Space:
    """The configuration of ``scope`` in ``current``."""
    space_type = current.linked.scopes[scope].space_type
    instance = object.__new__(space_type)
    object.__setattr__(instance, "_state", current)
    object.__setattr__(instance, "_scope", scope)
    return instance


def prepare_values(
    linked: LinkedModel,
    pending: Mapping[NodeId, object],
    role: str,
) -> dict[NodeId, object]:
    """Recognize every value before any is snapshot, then snapshot each."""
    return _state.prepare(linked, {index: (value, role) for index, value in pending.items()})


def bind(model: Model[S], parameters: Mapping[NodeId, object]) -> S:
    """Validate all root inputs and freeze them before any evaluation."""
    driver_only("configuration binding")
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
    frozen = prepare_values(model.linked, parameters, "parameter")
    return cast(S, attach(_state.bind(cast(Model[Space], model), frozen), 0))


def _node_scope(linked: LinkedModel, point: Space, reference: object) -> int | None:
    """For a node reference, the node's scope; None for anything else."""
    from ._nodes import NodeDecl
    from .references import descend

    if not isinstance(reference, Space):
        return None
    path = declared_path(reference)
    if path is None:
        return None
    *outer, last = path
    if not isinstance(last, NodeDecl):
        return None
    scopes = linked.scopes
    holder = descend(scopes, point._scope, outer)
    scope = scopes[holder]
    if last is scope.record:
        return holder
    target = scope.children.get(last)
    if target is None:
        raise RequestError("node is not part of this compiled scope")
    return target


def query(point: Space, reference: object) -> QueryResult[object]:
    driver_only("query inspection")
    current = state(point)
    linked = current.linked
    target = _node_scope(linked, point, reference)
    if target is not None:
        presence = linked.scopes[target].guard
        if presence is not None:
            answer = current.world.result(presence)
            if not isinstance(answer, Available) or answer.value is not True:
                return answer if not isinstance(answer, Available) else Inapplicable()
        return Available(attach(current, target))
    index = current.model.resolve(point._scope, reference)
    return _state.answer(current, index)


def present(point: Space, node: object) -> bool:
    """Whether a node, or a value input, is present, read like a value (undecided presence
    raises)."""
    reader = _reader(point)
    if reader is not None:  # a method reads through the world it runs in
        return reader.present(point, node)
    current = _driver(point)
    target = _node_scope(current.linked, point, node)
    if target is None:
        if isinstance(node, Param):
            return _supplied(current, point, node)
        raise RequestError("present() takes a node (a child or a candidate) or a value input")
    presence = current.linked.scopes[target].guard
    if presence is None:
        return True
    answer = current.world.result(presence)
    if isinstance(answer, Available):
        return answer.value is True
    if isinstance(answer, Inapplicable) or (isinstance(answer, Unresolved) and _unsupplied(answer)):
        return False
    return bool(_state.value(current, presence))  # undecided: raises like any value read


def _supplied(current: State, point: Space, formal: Param[object]) -> bool:
    """Whether a value input is supplied: a source applies (``_state.source_applies``).
    The value is never evaluated. Undecided presence raises like any value read."""
    applies = _state.source_applies(current, current.model.resolve(point._scope, formal))
    if isinstance(applies, bool):
        return applies
    _state.value(current, applies)  # undecided: raises like any value read
    raise AssertionError("an undecided presence read halts")


def _unsupplied(answer: Unresolved) -> bool:
    findings = answer.findings
    return bool(findings) and all(finding.code == "input-unsupplied" for finding in findings)


def read_value(point: Space, reference: ValueRef[T] | View[T]) -> T:
    """A member read as a value: the configuration's, or, on a receiver, the method's
    read (recorded and checked)."""
    reader = _reader(point)
    if reader is not None:  # a method reads through the world it runs in
        return cast(T, reader.read_value(point, reference))
    current = _driver(point)
    return cast(T, _state.value(current, current.model.resolve(point._scope, reference)))


@overload
def inspect(point: Space, reference: View[T]) -> ViewAssessment[T]: ...


@overload
def inspect(point: Space, reference: Constraint) -> ConstraintAssessment: ...


def inspect(
    point: Space,
    reference: View[T] | Constraint,
) -> ViewAssessment[T] | ConstraintAssessment:
    driver_only("assessment inspection")
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    node = current.linked.nodes[index]
    if node.kind not in {"view", "constraint"}:
        raise RequestError(
            f"{node.key} is not a view or a constraint: inspect() assesses those; "
            "read or query() any other member"
        )
    assessment = current.world.entry(index).assessment
    assert assessment is not None
    return cast(
        ViewAssessment[T] | ConstraintAssessment,
        _state.copy_assessment(current.linked, index, assessment),
    )


def decision_state(point: Space, reference: object) -> QueryResult[DecisionState[object]]:
    driver_only("decision state")
    return _state.decision_state(state(point), decision_index(point, reference))


def decision_index(point: Space, reference: object) -> NodeId:
    return state(point).model.decision(point._scope, reference)


def candidates(point: Space, reference: object) -> QueryResult[tuple[object, ...]] | None:
    """A choice's enumerated values, ranked by its domain's order where it has one (None:
    known by membership only), under its guard."""
    from ._world import enumeration

    driver_only("domain enumeration")
    current = state(point)
    index = decision_index(point, reference)
    result = enumeration(current.world, index)
    if isinstance(result, Available) and result.value is None:
        return None
    if isinstance(result, Available):
        node = current.linked.nodes[index]
        assert node.semantics is not None
        values = cast(tuple[object, ...], result.value)
        return Available(
            tuple(
                snapshot(node.semantics, value, owner=node.owner, role="public value snapshot")
                for value in values
            )
        )
    return cast(QueryResult[tuple[object, ...]], result)


def bind_field(point: Space, reference: ValueRef[T] | View[T]) -> BoundValue[T] | BoundDecision[T]:
    """A decision accessor for a Decision, a value accessor for anything else read as a value.

    A view binds as a value accessor: ``get()`` is its accepted value and
    ``query()`` its accepted result, exactly the attribute read and
    ``point.query``; its assessment is ``point.inspect``.
    """
    reader = _reader(point)
    if reader is not None:
        reader._enter()
        model = reader.world.evaluator.model
    else:
        model = _driver(point).model
    model.resolve(point._scope, reference)
    try:
        model.decision(point._scope, reference)
    except RequestError:
        return BoundValue(point, reference)
    return BoundDecision(point, reference)


def root(point: Space) -> Space:
    reader = _reader(point)
    if reader is not None:  # a method reads through the world it runs in
        return reader.root(point)
    current = _driver(point)
    return point if point._scope == 0 else attach(current, 0)


def child(point: Space, record: Declaration) -> Space:
    """A child node's configuration (on a receiver, the child's receiver)."""
    reader = _reader(point)
    if reader is not None:  # a method reads through the world it runs in
        return reader.child(point, record)
    current = _driver(point)
    scope = current.linked.scopes[point._scope]
    child_scope = scope.children.get(record)
    if child_scope is None:
        raise RequestError("child node is not part of this compiled scope")
    return attach(current, child_scope)


def _scope_choice(linked: LinkedModel, scope: int, decision: object) -> Choice:
    """The Choice of a Decision over nodes in compiled scope ``scope``."""
    from ._nodes import unwrap

    record = unwrap(decision)
    choices = linked.scopes[scope].choices
    try:
        return linked.choices[choices[record]]
    except (KeyError, TypeError) as error:
        raise RequestError("the Decision over nodes is not part of this compiled scope") from error


def selected_candidate(point: Space, decision: Declaration) -> Space | None:
    """The configuration of the candidate a Decision over nodes selects, or None."""
    reader = _reader(point)
    if reader is not None:  # a method reads through the world it runs in
        return reader.selected_candidate(point, decision)
    current = _driver(point)
    choice = _scope_choice(current.linked, point._scope, decision)
    case = _state.value(current, choice.selector)
    for key, candidate in choice.cases:
        if key == case:
            return None if candidate is None else attach(current, candidate)
    raise EvaluationError(choice.key, "selection", "selector is not a declared case")


def candidate(point: Space, decision: object, case: str) -> Space | None:
    """A candidate's configuration whether or not it is selected (None for a None case)."""
    reader = _reader(point)
    if reader is not None:  # a method reads through the world it runs in
        return reader.candidate(point, decision, case)
    current = _driver(point)
    if type(case) is not str:
        raise RequestError("a candidate key must be a string")
    choice = _scope_choice(current.linked, point._scope, decision)
    for key, index in choice.cases:
        if key == case:
            return None if index is None else attach(current, index)
    raise RequestError(f"{choice.key}: unknown candidate {case!r}")
