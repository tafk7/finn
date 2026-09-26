# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public metadata, conservative dependencies, and demanded query evidence.

Metadata inspection does not invoke evaluators or candidate providers. Explain
evaluates one query and follows only cached demanded edges for that snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar, cast, overload

from . import _execution, _runtime
from ._configuration import Space
from ._nodes import NodeChoice, NodeDecl, family_formals, node_record, unsupplied_formals
from .collection import collect_space
from .compiler import SpaceModel, compile_space
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Constraint,
    ConstraintGroup,
    Decision,
    Declaration,
    MemberRef,
    Param,
    ValueRef,
    View,
    declared_path,
)
from .errors import RequestError
from .ir import LinkedModel, NodeKind
from .occurrence import candidate as _candidate
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
    origin: str | None = None


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
    """One candidate of a Decision over nodes; a None candidate has no scope."""

    name: str
    scope: str | None
    space_type: type[Space] | None


@dataclass(frozen=True, slots=True)
class ChoiceInfo:
    key: str
    scope: str
    cases: tuple[CaseInfo, ...]
    selector: DecisionHandle[str]
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

    Authored declarations count effective members once in every instantiated
    scope (a Decision over nodes is one member), plus each placed node
    (including every candidate). Reused templates therefore count at every
    placement. A fresh binding replaces its child formal and is counted once;
    generated output, guard, selection and located nodes are excluded.
    """

    authored_declarations: int
    scopes: int
    nodes: int
    potential_edges: int
    choices: int
    owning_decisions: int


def statistics(subject: Space | SpaceModel[S] | type[Space]) -> ModelStatistics:
    """Count direct structure on demand, with no runtime or closure retention."""

    if not isinstance(subject, (SpaceModel, Space, type)):
        raise RequestError("statistics requires a compiled model")
    linked = _context(subject)[0].linked
    placements = len(linked.scopes) - 1
    authored = sum(len(scope.named_members) for scope in linked.scopes)
    return ModelStatistics(
        authored + placements,
        len(linked.scopes),
        len(linked.nodes),
        sum(len(node.dependencies) for node in linked.nodes),
        len(linked.choices),
        len(linked.decisions),
    )


def _context(subject: Space | SpaceModel[S] | type[Space]) -> tuple[SpaceModel[Space], int]:
    if isinstance(subject, SpaceModel):
        return cast(SpaceModel[Space], subject), 0
    if isinstance(subject, type) and issubclass(subject, Space):
        return cast(SpaceModel[Space], compile_space(subject)), 0
    if isinstance(subject, Space):
        current = state(subject)
        return current.model, subject._scope
    raise RequestError("inspection requires a configuration, a compiled model or a family")


def model(subject: Space | SpaceModel[S] | type[Space]) -> SpaceModel[Space]:
    """The compiled model of a configuration, or of a family compiled alone."""
    return _context(subject)[0]


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
        node.origin,
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


def decisions(subject: Space | SpaceModel[S] | type[Space]) -> tuple[DecisionInfo[object], ...]:
    """Discover all owning decisions below this scope, including inactive cases."""

    compiled, scope = _context(subject)
    included = _scope_set(compiled.linked, scope)
    result = (
        _decision_info(compiled.linked, index)
        for index in compiled.linked.decisions
        if compiled.linked.nodes[index].scope in included
    )
    return tuple(sorted(result, key=lambda item: item.key))


@overload
def decision_info(
    subject: Space | SpaceModel[S] | type[Space], reference: Decision[T]
) -> DecisionInfo[T]: ...


@overload
def decision_info(
    subject: Space | SpaceModel[S] | type[Space], reference: object
) -> DecisionInfo[Any]: ...


def decision_info(subject: Space | SpaceModel[S] | type[Space], reference: object) -> object:
    compiled, scope = _context(subject)
    index = compiled.decision(scope, reference)
    return _decision_info(compiled.linked, index)


@overload
def decision_handle(
    subject: Space | SpaceModel[S] | type[Space], reference: Decision[T]
) -> DecisionHandle[T]: ...


@overload
def decision_handle(
    subject: Space | SpaceModel[S] | type[Space], reference: object
) -> DecisionHandle[Any]: ...


def decision_handle(subject: Space | SpaceModel[S] | type[Space], reference: object) -> object:
    """Bind a typed owning decision while retaining its candidate value type."""

    return decision_info(subject, reference).reference


@overload
def value_handle(
    subject: Space | SpaceModel[S] | type[Space],
    reference: ValueRef[T] | View[T],
) -> ValueHandle[T]: ...


@overload
def value_handle(
    subject: Space | SpaceModel[S] | type[Space],
    reference: Constraint | ConstraintGroup,
) -> ValueHandle[bool]: ...


@overload
def value_handle(
    subject: Space | SpaceModel[S] | type[Space], reference: object
) -> ValueHandle[Any]: ...


def value_handle(subject: Space | SpaceModel[S] | type[Space], reference: object) -> object:
    """Bind a typed value, accepted view result, or Boolean assessment result."""

    compiled, scope = _context(subject)
    return ValueHandle(compiled.linked, compiled.resolve(scope, reference))


def choices(subject: Space | SpaceModel[S] | type[Space]) -> tuple[ChoiceInfo, ...]:
    """Every Decision over nodes below this scope, with its candidates."""
    compiled, scope = _context(subject)
    linked = compiled.linked
    included = _scope_set(linked, scope)
    return tuple(
        ChoiceInfo(
            choice.key,
            linked.scopes[choice.scope].name,
            tuple(
                CaseInfo(name, None, None)
                if index is None
                else CaseInfo(name, linked.scopes[index].name, linked.scopes[index].space_type)
                for name, index in choice.cases
            ),
            DecisionHandle[str](linked, choice.selector),
            None if choice.guard is None else ValueHandle[bool](linked, choice.guard),
        )
        for choice in sorted(linked.choices, key=lambda item: item.key)
        if choice.scope in included
    )


def members(subject: Space | SpaceModel[S] | type[Space]) -> tuple[NodeInfo, ...]:
    """Inspect authored and deliberately exposed members below this scope."""

    compiled, scope = _context(subject)
    included = _scope_set(compiled.linked, scope)
    indices = {
        index
        for current in compiled.linked.scopes
        if current.index in included
        for index in current.named_members.values()
    }
    return tuple(
        _node_info(compiled.linked, index)
        for index in sorted(indices, key=lambda item: compiled.linked.nodes[item].key)
    )


def dependencies(
    subject: Space | SpaceModel[S] | type[Space],
    reference: object,
) -> tuple[NodeInfo, ...]:
    """Return known structural inputs; self-method reads are discovered at runtime."""

    compiled, scope = _context(subject)
    node = compiled.linked.nodes[compiled.resolve(scope, reference)]
    return tuple(_node_info(compiled.linked, index) for index in node.dependencies)


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


@dataclass(frozen=True, slots=True)
class NodeDeclaration:
    """A node declaration read without compiling it.

    ``bindings`` maps each bound formal to its supplier: a frozen value, a
    reference, a fresh Decision, or (for a reference input) a node.
    ``nested`` maps the formals of descendants assigned through a path
    (``"port.dtype"``) to their suppliers. ``unsupplied`` names the required
    formals nothing supplies yet, and ``frozen`` says why assignment is closed.
    """

    family: type[Space]
    name: str | None
    placement: str | None
    members: tuple[str, ...]
    bindings: Mapping[str, object]
    nested: Mapping[str, object]
    unsupplied: tuple[str, ...]
    frozen: str | None
    when: object
    origin: str | None


@dataclass(frozen=True, slots=True)
class ReferenceInfo:
    """A symbolic reference read without compiling: its node path and member."""

    path: tuple[str | None, ...]
    member: str
    origin: str | None


def declaration(node: object) -> NodeDeclaration:
    """Inspect a node declaration (or a Decision over nodes' candidate) before compilation."""

    record = node_record(node)
    if record is None and isinstance(node, NodeDecl):
        record = node
    if record is None:
        raise RequestError("declaration() takes a node declaration, as returned by a family call")
    formals = family_formals(record.family)
    bindings: dict[str, object] = {}
    for name, value in record.bindings.items():
        formal = formals.get(name)
        if isinstance(value, NodeDecl):
            value = value.instance
        elif isinstance(formal, Param) and not isinstance(value, (ValueRef, View)):
            # A detached copy: the declaration keeps its own frozen literal.
            semantics = cast(Param[object], formal).semantics
            assert semantics is not None
            value = semantics.freeze(value)
        bindings[name] = value
    nested = {
        ".".join((*(str(item.name) for item in path), name)): (
            value.instance if isinstance(value, NodeDecl) else value
        )
        for path, supplies in record.nested.items()
        for name, (value, _) in supplies.items()
    }
    return NodeDeclaration(
        record.family,
        record.name,
        record.placement,
        tuple(collect_space(record.family).members),
        MappingProxyType(bindings),
        MappingProxyType(nested),
        tuple(sorted(unsupplied_formals(record))),
        record.frozen,
        record.when,
        record.origin,
    )


def reference(value: object) -> ReferenceInfo:
    """Inspect a symbolic reference (``kitchen.finish``, ``heating.kw``) before compilation."""

    if isinstance(value, NodeChoice):
        path: tuple[Declaration, ...] = value._space_path
        return ReferenceInfo(tuple(item.name for item in path), "", path[-1].origin)
    if isinstance(value, (MemberRef, ChoiceMemberRef, CaseRef)):
        member = value.member if isinstance(value, (MemberRef, ChoiceMemberRef)) else "$case"
        name = member if isinstance(member, str) else member.name
        return ReferenceInfo(tuple(item.name for item in value.path), str(name), value.origin)
    raise RequestError("reference() takes a symbolic reference such as node.member")


def candidate(point: Space, decision: object, case: str) -> Space | None:
    """A candidate's configuration, selected or not; None for a None candidate."""

    return _candidate(point, decision, case)


def is_declaration(value: object) -> bool:
    """Whether ``value`` is a node declaration or reference path rather than a configuration."""

    return isinstance(value, Space) and declared_path(value) is not None


__all__ = [
    "CaseInfo",
    "ChoiceInfo",
    "DecisionInfo",
    "EvidenceNode",
    "NodeInfo",
    "ModelStatistics",
    "QueryEvidence",
    "NodeDeclaration",
    "ReferenceInfo",
    "candidate",
    "choices",
    "declaration",
    "decision_handle",
    "decision_info",
    "decisions",
    "dependencies",
    "explain",
    "is_declaration",
    "members",
    "model",
    "reference",
    "statistics",
    "value_handle",
]
