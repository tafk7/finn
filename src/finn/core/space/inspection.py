# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public metadata, conservative dependencies, and demanded query evidence.

Metadata inspection does not invoke evaluators or candidate providers. Explain
evaluates one query and follows only cached demanded edges for that snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast, overload

from . import _execution, _runtime
from ._configuration import Space
from .compiler import SpaceModel
from .declarations import (
    Constraint,
    ConstraintGroup,
    Decision,
    DecisionRef,
    ValueRef,
    View,
)
from .errors import RequestError
from .ir import LinkedModel, NodeKind
from .occurrence import state
from .references import DecisionHandle, ValueHandle, decision_key
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    QueryResult,
    ViewAssessment,
)

T = TypeVar("T")
S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """One computation identity and its authored diagnostic owner."""

    reference: ValueHandle[object]
    key: str
    kind: NodeKind
    scope: str
    owner: str
    generated: bool
    guard: ValueHandle[bool] | None


@dataclass(frozen=True, slots=True)
class DecisionInfo(Generic[T]):
    """An owning choice. Its stable key never exposes internal selector names."""

    reference: DecisionHandle[T]
    key: str
    scope: str
    owner: str
    selector: bool
    cases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseInfo:
    name: str
    scope: str
    space_type: type[Space]


@dataclass(frozen=True, slots=True)
class ChoiceInfo:
    key: str
    scope: str
    cases: tuple[CaseInfo, ...]
    selector: DecisionHandle[str] | None
    guard: ValueHandle[bool] | None


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    """Detached result and actual outgoing demands for one visited computation.

    Input presence is recorded only when parameter evaluation reached its body.
    An inactive parameter therefore exposes no unused external value.
    """

    declaration: NodeInfo
    result: QueryResult[object]
    dependencies: tuple[ValueHandle[object], ...]
    input_presence: Literal["supplied", "omitted"] | None = None
    decision_state: QueryResult[DecisionState[object]] | None = None
    selector: bool = False
    is_guard: bool = False


@dataclass(frozen=True, slots=True)
class QueryEvidence(Generic[T]):
    query: NodeInfo
    result: QueryResult[T]
    nodes: tuple[EvidenceNode, ...]
    assessment: ViewAssessment[T] | ConstraintAssessment | None = None


@dataclass(frozen=True, slots=True)
class ModelStatistics:
    """Structural counts for one compiled family, independent of runtime state.

    Authored declarations count effective value/constraint/view members once in
    every instantiated scope, plus each child placement (including choice cases)
    and each structural choice. Reused templates therefore count at every
    placement. A fresh binding replaces its child formal and is counted once;
    generated output, guard, selector and selected-export nodes are excluded.
    """

    authored_declarations: int
    scopes: int
    nodes: int
    potential_edges: int
    choices: int
    owning_decisions: int


def statistics(model: SpaceModel[S]) -> ModelStatistics:
    """Count direct structure on demand, with no runtime or closure retention."""

    if not isinstance(model, SpaceModel):
        raise RequestError("statistics requires a compiled model")
    linked = model.linked
    placements = len(linked.scopes) - 1
    authored = sum(len(scope.named_members) for scope in linked.scopes)
    return ModelStatistics(
        authored + placements + len(linked.choices),
        len(linked.scopes),
        len(linked.nodes),
        sum(len(node.dependencies) for node in linked.nodes),
        len(linked.choices),
        len(linked.decisions),
    )


def _context(subject: Space | SpaceModel[S]) -> tuple[SpaceModel[Space], int]:
    if isinstance(subject, SpaceModel):
        return cast(SpaceModel[Space], subject), 0
    if isinstance(subject, Space):
        current = state(subject)
        return current.model, subject._scope
    raise RequestError("inspection requires a compiled model or attached configuration")


def _scope_set(linked: LinkedModel, root: int) -> set[int]:
    # Scopes are allocated after their parent. No repeated ancestor traversal
    # or stored transitive closure is needed to inspect a subtree.
    included = {root}
    for scope in linked.scopes:
        if scope.parent in included:
            included.add(scope.index)
    return included


def _node_info(linked: LinkedModel, index: int) -> NodeInfo:
    node = linked.nodes[index]
    reference = (
        DecisionHandle[object](linked, index)
        if node.kind == "decision"
        else ValueHandle[object](linked, index)
    )
    return NodeInfo(
        reference,
        node.key,
        node.kind,
        linked.scopes[node.scope].name,
        node.owner,
        node.source_owner is not None,
        None if node.guard is None else ValueHandle[bool](linked, node.guard),
    )


def _decision_info(
    linked: LinkedModel,
    index: int,
) -> DecisionInfo[object]:
    node = linked.nodes[index]
    if node.kind != "decision":
        raise RequestError("decision inspection requires an owning Decision")
    choice_index = linked.selector_choices.get(index)
    choice = None if choice_index is None else linked.choices[choice_index]
    return DecisionInfo(
        DecisionHandle[object](linked, index),
        decision_key(linked, index),
        linked.scopes[node.scope].name,
        node.owner,
        choice is not None,
        () if choice is None else tuple(name for name, _ in choice.cases),
    )


def decisions(subject: Space | SpaceModel[S]) -> tuple[DecisionInfo[object], ...]:
    """Discover all owning decisions below this scope, including inactive cases."""

    model, scope = _context(subject)
    included = _scope_set(model.linked, scope)
    result = (
        _decision_info(model.linked, index)
        for index in model.linked.decisions
        if model.linked.nodes[index].scope in included
    )
    return tuple(sorted(result, key=lambda item: item.key))


def decision_info(
    subject: Space | SpaceModel[S],
    reference: Decision[T] | DecisionRef[T],
) -> DecisionInfo[T]:
    model, scope = _context(subject)
    if not isinstance(reference, (Decision, DecisionRef)):
        raise RequestError("a parameter alias cannot become an editable decision handle")
    index = model.resolve(scope, reference)
    return cast(DecisionInfo[T], _decision_info(model.linked, index))


def decision_handle(
    subject: Space | SpaceModel[S],
    reference: Decision[T] | DecisionRef[T],
) -> DecisionHandle[T]:
    """Bind a typed owning decision while retaining its candidate value type."""

    return decision_info(subject, reference).reference


@overload
def value_handle(
    subject: Space | SpaceModel[S],
    reference: ValueRef[T] | View[T],
) -> ValueHandle[T]: ...


@overload
def value_handle(
    subject: Space | SpaceModel[S],
    reference: Constraint | ConstraintGroup,
) -> ValueHandle[bool]: ...


def value_handle(subject: Space | SpaceModel[S], reference: object) -> object:
    """Bind a typed value, accepted view result, or Boolean assessment result."""

    model, scope = _context(subject)
    return ValueHandle(model.linked, model.resolve(scope, reference))


def choices(subject: Space | SpaceModel[S]) -> tuple[ChoiceInfo, ...]:
    model, scope = _context(subject)
    linked = model.linked
    included = _scope_set(linked, scope)
    return tuple(
        ChoiceInfo(
            choice.key,
            linked.scopes[choice.scope].name,
            tuple(
                CaseInfo(name, linked.scopes[index].name, linked.scopes[index].space_type)
                for name, index in choice.cases
            ),
            None if choice.selector is None else DecisionHandle[str](linked, choice.selector),
            None if choice.guard is None else ValueHandle[bool](linked, choice.guard),
        )
        for choice in sorted(linked.choices, key=lambda item: item.key)
        if choice.scope in included
    )


def members(subject: Space | SpaceModel[S]) -> tuple[NodeInfo, ...]:
    """Inspect authored and deliberately exposed members below this scope."""

    model, scope = _context(subject)
    included = _scope_set(model.linked, scope)
    indices = {
        index
        for current in model.linked.scopes
        if current.index in included
        for index in current.named_members.values()
    }
    return tuple(
        _node_info(model.linked, index)
        for index in sorted(indices, key=lambda item: model.linked.nodes[item].key)
    )


def dependencies(
    subject: Space | SpaceModel[S],
    reference: object,
) -> tuple[NodeInfo, ...]:
    """Return known structural inputs; self-method reads are discovered at runtime."""

    model, scope = _context(subject)
    node = model.linked.nodes[model.resolve(scope, reference)]
    return tuple(_node_info(model.linked, index) for index in node.dependencies)


@overload
def explain(point: Space, reference: ValueRef[T] | View[T]) -> QueryEvidence[T]: ...


@overload
def explain(
    point: Space,
    reference: Constraint | ConstraintGroup,
) -> QueryEvidence[bool]: ...


def explain(point: Space, reference: object) -> object:
    """Evaluate a query and detach evidence of exactly its demanded computation."""

    _execution.driver_only("dependency inspection")
    current = state(point)
    linked, snapshot = current.model.linked, current
    root = current.model.resolve(point._scope, reference)
    selectors = linked.selector_choices
    with snapshot.lock:
        result = _runtime.evaluate(snapshot, root)
        pending = [root]
        visited: set[int] = set()
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            pending.extend(snapshot.cache[index].dependencies)
        guard_nodes = {
            linked.nodes[index].guard for index in visited if linked.nodes[index].guard is not None
        }
        evidence: list[EvidenceNode] = []
        for index in sorted(visited, key=lambda item: linked.nodes[item].key):
            node, entry = linked.nodes[index], snapshot.cache[index]
            active = True
            if node.guard is not None:
                guard = snapshot.cache[node.guard].result
                active = isinstance(guard, Available) and guard.value is True
            presence: Literal["supplied", "omitted"] | None = None
            if node.kind == "param" and active:
                presence = "supplied" if index in snapshot.parameters else "omitted"
            evidence.append(
                EvidenceNode(
                    _node_info(linked, index),
                    _runtime.copy_result(snapshot, index, entry.result),
                    tuple(ValueHandle[object](linked, target) for target in entry.dependencies),
                    presence,
                    _runtime.decision_state(snapshot, index) if node.kind == "decision" else None,
                    index in selectors,
                    index in guard_nodes,
                )
            )
        assessment = (
            None
            if result.assessment is None
            else _runtime.copy_assessment(snapshot, root, result.assessment)
        )
        return QueryEvidence(
            _node_info(linked, root),
            _runtime.copy_result(snapshot, root, result.result),
            tuple(evidence),
            assessment,
        )


__all__ = [
    "CaseInfo",
    "ChoiceInfo",
    "DecisionInfo",
    "EvidenceNode",
    "NodeInfo",
    "ModelStatistics",
    "QueryEvidence",
    "choices",
    "decision_handle",
    "decision_info",
    "decisions",
    "dependencies",
    "explain",
    "members",
    "statistics",
    "value_handle",
]
