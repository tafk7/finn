# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed occurrence operations over one compiled model and immutable snapshot."""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar, cast, overload

from . import _execution, _runtime
from ._collapse import forwards, guard_implies
from ._configuration import BoundDecision, BoundValue, Space
from ._runtime import Snapshot
from .compiler import Model
from .declarations import (
    Constraint,
    ConstraintGroup,
    Declaration,
    Param,
    ValueRef,
    View,
    declared_path,
)
from .errors import EvaluationError, RequestError
from .ir import LinkedModel, Node
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    Unresolved,
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
                f"{point!r} is a node declaration, not a configuration; open its root's "
                "design space with design_space()"
            )
        raise RequestError("a configuration is created by design_space()")
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


def bind(model: Model[S], parameters: Mapping[int, object]) -> S:
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
    snapshot = _runtime.Snapshot(cast(Model[Space], model), frozen)
    return cast(S, _attach(snapshot, 0))


def _node_scope(point: Space, reference: object) -> tuple[int, int | None] | None:
    """For a node reference: (the scope holding it, the node's scope or None if unsupplied)."""
    from ._configuration import Space
    from ._nodes import NodeDecl, is_reference_input
    from .references import _descend

    if isinstance(reference, Space):
        path = declared_path(reference)
        if path is None:
            return None
        *outer, last = path
    elif is_reference_input(reference):
        outer, last = [], cast(Declaration, reference)
    else:
        return None
    if not isinstance(last, NodeDecl) and not is_reference_input(last):
        return None
    current = state(point)
    scopes = current.linked.scopes
    holder = _descend(scopes, point._scope, outer)
    scope = scopes[holder]
    if last is scope.record:
        return holder, holder
    target = scope.children.get(last, scope.references.get(last))
    if target is None and last not in scope.members:
        raise RequestError("node is not part of this compiled scope")
    return holder, target


def _presence(point: Space, holder: int, target: int | None, last: object) -> int | None:
    """The node answering a node's presence: its guard, or a reference input's presence node."""
    scopes = state(point).linked.scopes
    if last in scopes[holder].members:  # a reference input
        return scopes[holder].members[last]
    return None if target is None else scopes[target].guard


def query(point: Space, reference: object) -> QueryResult[object]:
    _execution.driver_only("query inspection")
    current = state(point)
    located = _node_scope(point, reference)
    if located is not None:
        holder, target = located
        last = cast(tuple[Declaration, ...], declared_path(reference) or (reference,))[-1]
        presence = _presence(point, holder, target, last)
        if presence is not None:
            answer = _runtime.evaluate(current, presence).result
            if not isinstance(answer, Available) or answer.value is not True:
                return answer if not isinstance(answer, Available) else Inapplicable()
        assert target is not None
        return Available(_attach(current, target))
    index = current.model.resolve(point._scope, reference)
    result = _runtime.evaluate(current, index).result
    return _runtime.copy_result(current, index, result)


def present(point: Space, node: object) -> bool:
    """Whether a node, or a value input, is present, read like a value (undecided presence
    raises)."""
    located = _node_scope(point, node)
    if located is None:
        if isinstance(node, Param):
            return _supplied(point, node)
        raise RequestError(
            "present() takes a node (a child, a candidate or a reference input) or a value input"
        )
    holder, target = located
    last = cast(tuple[Declaration, ...], declared_path(node) or (node,))[-1]
    presence = _presence(point, holder, target, last)
    if presence is None:
        return target is not None
    answer = _read_result(point, presence)
    if isinstance(answer, Available):
        return answer.value is True
    if isinstance(answer, Inapplicable) or (isinstance(answer, Unresolved) and _unsupplied(answer)):
        return False
    return bool(_read_index(point, presence))  # undecided: raises like any value read


def _supplied(point: Space, formal: Param[object]) -> bool:
    """Whether a value input is supplied: read without halting, its answer is available.

    Bound to an enclosing formal, it answers as that one does. Omitted at start
    (``input-missing``), or a formal no present source supplies
    (``input-unsupplied``), it is not supplied. Bound to a guarded view or
    derived value, it is supplied when that source applies: its guard is read,
    a fact, and the value is not computed (whether a stream has a value is
    known before the value is).
    """
    index = state(point).model.resolve(point._scope, formal)
    guards = _source_guards(state(point).linked, index)
    if guards is not None:
        for guard in guards:
            held = _read_result(point, guard)
            if isinstance(held, Inapplicable) or (
                isinstance(held, Available) and held.value is not True
            ):
                return False
            if not isinstance(held, Available):
                _read_index(point, guard)  # undecided: raises like any value read
        return True
    answer = _read_result(point, index)
    if isinstance(answer, Available):
        return True
    if isinstance(answer, Inapplicable) or (
        isinstance(answer, Unresolved) and _unsupplied(answer, "input-missing")
    ):
        return False
    _read_index(point, index)  # undecided: raises like any value read
    return True


def _source_guards(linked: LinkedModel, index: int) -> tuple[int, ...] | None:
    """The guards along a value input's forwarding chain to a guarded view or derived
    value, ending with that source's; None when the source is no guarded computation."""
    guards: list[int] = []
    node = linked.nodes[index]
    while forwards(node):
        if node.guard is not None:
            guards.append(node.guard)
        node = linked.nodes[cast(int, node.output)]
    if node.kind not in ("view", "derived") or node.guard is None:
        return None
    return (*guards, node.guard)


def _read_result(point: Space, index: int) -> QueryResult[object]:
    """A node's answer without halting the reading method on a non-value."""
    snapshot = state(point)
    _execution.check_snapshot(snapshot)
    active = _execution.current()
    if active is None:
        return _runtime.evaluate(snapshot, index).result
    active.dependencies[index] = None
    outcome: object = (
        snapshot.cache[index] if index in snapshot.cache else active.native_parent.switch(index)
    )
    if not isinstance(outcome, _runtime.Evaluation):
        _execution.accept_read(active, index, outcome)  # a transported failure: raises
    return cast(_runtime.Evaluation, outcome).result


def _unsupplied(answer: Unresolved, *also: str) -> bool:
    codes = ("input-unsupplied", *also)
    return bool(answer.findings) and all(finding.code in codes for finding in answer.findings)


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
    active = _execution.current()
    if active is not None:
        # A method's read of a forwarding alias goes straight to its source
        # when the alias applies whenever the reading node does.
        linked = snapshot.linked
        source = linked.forward[index]
        if source != index and guard_implies(
            linked.nodes, linked.nodes[index].guard, linked.nodes[active.index].guard
        ):
            active.via[index] = source
            index = source
    return cast(T, _read_index(point, index))


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
    node = current.linked.nodes[index]
    if node.kind not in {"view", "constraint", "group"}:
        raise RequestError(
            f"{node.key} is not a view or a constraint: inspect() assesses those; "
            "read or query() any other member"
        )
    entry = _runtime.evaluate(current, index)
    assert entry.assessment is not None
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


def bind_field(point: Space, reference: ValueRef[T] | View[T]) -> BoundValue[T] | BoundDecision[T]:
    """A decision accessor for a Decision, a value accessor for anything else read as a value.

    A view binds as a value accessor: ``get()`` is its accepted value and
    ``query()`` its accepted result, exactly the attribute read and
    ``point.query``; its assessment is ``point.inspect``.
    """
    current = state(point)
    _execution.check_snapshot(current)
    current.model.resolve(point._scope, reference)
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
    """A child node's configuration; through a reference input, the referenced node's.

    Every read through a referenced node that is absent is inapplicable, as for
    any guarded node. An unsupplied optional reference input has no node: it
    reads its presence, which is unsupplied, exactly like a value read.
    """
    from ._nodes import is_reference_input

    current = state(point)
    _execution.check_snapshot(current)
    scope = current.linked.scopes[point._scope]
    child_scope = scope.children.get(record, scope.references.get(record))
    if child_scope is None and is_reference_input(record) and record in scope.members:
        _read_index(point, scope.members[record])  # raises: unsupplied
    if child_scope is None:
        raise RequestError("child node is not part of this compiled scope")
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
