# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public metadata, conservative dependencies, and demanded query evidence.

Metadata inspection does not invoke evaluators or candidate providers. Explain
evaluates one query and follows only cached demanded edges for that snapshot.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar, cast, overload

from . import _execution, _forcing, _runtime
from ._configuration import Space
from ._forcing import Forced, Open, Viable
from ._nodes import NodeChoice, NodeDecision, NodeDecl, node_record, unsupplied_formals
from .collection import collect_space
from .compiler import Model, compile_model
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Constraint,
    ConstraintGroup,
    Decision,
    Declaration,
    MemberRef,
    ValueRef,
    View,
    declared_path,
)
from .errors import RequestError
from .ir import Layer, LinkedModel, NodeKind, Provenance
from .occurrence import candidate as _candidate
from .occurrence import state
from .references import DecisionHandle, ValueHandle, decision_key, descend
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
    # Who set this member's value, for a member some body supplied or overrode.
    provenance: Provenance | None = None


@dataclass(frozen=True, slots=True)
class DecisionInfo(Generic[T]):
    """An owning choice. Its stable key never exposes internal selector names.

    ``space_type`` is the Space class whose scope declares it (a candidate's, for a
    choice nested under a Decision over nodes); ``ordered``, whether its domain
    states an order of its cases (``Domain.ordered``); ``required``, whether it
    has no safe baseline (``Decision(required=True)``): no completion takes its
    first case.
    """

    reference: DecisionHandle[T]
    key: str
    scope: str
    owner: str
    selector: bool
    cases: tuple[str, ...] = ()
    space_type: type[Space] | None = None
    ordered: bool = False
    required: bool = False


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
    # None when an enclosing body pinned the choice: its key is then listed as pinned.
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
    # Forwarding aliases this computation read through (by their authored
    # names): the dependency is the alias's source, which evaluation reached
    # directly, and the alias is kept here so evidence still names it.
    via: tuple[NodeInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryEvidence(Generic[T]):
    query: NodeInfo
    result: QueryResult[T]
    nodes: tuple[EvidenceNode, ...]
    assessment: ViewAssessment[T] | ConstraintAssessment | None = None


@dataclass(frozen=True, slots=True)
class ModelStatistics:
    """Structural counts for one compiled Space class, independent of runtime state.

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


def statistics(subject: Space | Model[S] | type[Space]) -> ModelStatistics:
    """Count direct structure on demand, with no runtime or closure retention."""

    if not isinstance(subject, (Model, Space, type)):
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


def _context(subject: Space | Model[S] | type[Space]) -> tuple[Model[Space], int]:
    if isinstance(subject, Model):
        return cast(Model[Space], subject), 0
    if isinstance(subject, type) and issubclass(subject, Space):
        return cast(Model[Space], compile_model(subject)), 0
    if isinstance(subject, Space):
        current = state(subject)
        return current.model, subject._scope
    raise RequestError("inspection requires a configuration, a compiled model or a Space class")


def model(subject: Space | Model[S] | type[Space]) -> Model[Space]:
    """The compiled model of a configuration, or of a Space class compiled alone."""
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
        linked.provenance.get(index),
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
        linked.scopes[node.scope].space_type,
        choice is None and node.domain is not None and node.domain.ordered,
        node.required_choice,
    )


def decisions(subject: Space | Model[S] | type[Space]) -> tuple[DecisionInfo[object], ...]:
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
    subject: Space | Model[S] | type[Space], reference: Decision[T]
) -> DecisionInfo[T]: ...


@overload
def decision_info(
    subject: Space | Model[S] | type[Space], reference: DecisionHandle[T] | ValueRef[T]
) -> DecisionInfo[T]: ...


@overload
def decision_info(
    subject: Space | Model[S] | type[Space], reference: Space | None
) -> DecisionInfo[str]: ...


@overload
def decision_info(subject: Space | Model[S] | type[Space], reference: T) -> DecisionInfo[T]: ...


def decision_info(subject: Space | Model[S] | type[Space], reference: object) -> object:
    compiled, scope = _context(subject)
    index = compiled.decision(scope, reference)
    return _decision_info(compiled.linked, index)


@overload
def decision_handle(
    subject: Space | Model[S] | type[Space], reference: Decision[T]
) -> DecisionHandle[T]: ...


@overload
def decision_handle(
    subject: Space | Model[S] | type[Space], reference: DecisionHandle[T] | ValueRef[T]
) -> DecisionHandle[T]: ...


@overload
def decision_handle(
    subject: Space | Model[S] | type[Space], reference: Space | None
) -> DecisionHandle[str]: ...


@overload
def decision_handle(subject: Space | Model[S] | type[Space], reference: T) -> DecisionHandle[T]: ...


def decision_handle(subject: Space | Model[S] | type[Space], reference: object) -> object:
    """Bind a typed owning decision while retaining its candidate value type."""

    return decision_info(subject, reference).reference


@overload
def value_handle(
    subject: Space | Model[S] | type[Space],
    reference: ValueRef[T] | View[T],
) -> ValueHandle[T]: ...


@overload
def value_handle(
    subject: Space | Model[S] | type[Space],
    reference: Constraint | ConstraintGroup,
) -> ValueHandle[bool]: ...


@overload
def value_handle(
    subject: Space | Model[S] | type[Space], reference: Space | None
) -> ValueHandle[Any]: ...


@overload
def value_handle(subject: Space | Model[S] | type[Space], reference: T) -> ValueHandle[T]: ...


def value_handle(subject: Space | Model[S] | type[Space], reference: object) -> object:
    """Bind a typed value, accepted view result, or Boolean assessment result."""

    compiled, scope = _context(subject)
    return ValueHandle(compiled.linked, compiled.resolve(scope, reference))


def choices(subject: Space | Model[S] | type[Space]) -> tuple[ChoiceInfo, ...]:
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
            DecisionHandle[str](linked, choice.selector)
            if linked.nodes[choice.selector].kind == "decision"
            else None,
            None if choice.guard is None else ValueHandle[bool](linked, choice.guard),
        )
        for choice in sorted(linked.choices, key=lambda item: item.key)
        if choice.scope in included
    )


def members(subject: Space | Model[S] | type[Space]) -> tuple[NodeInfo, ...]:
    """Inspect every named member below this scope (including forwarding aliases)."""

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


def provenance(subject: Space | Model[S] | type[Space], reference: object) -> Provenance | None:
    """Who set a member's effective value, and what it overrides; None if nothing set it.

    ``reference`` is a member (``House.kitchen.area``), a child node
    (``House.kitchen.sub``, replaced by an enclosing body) or a handle. The
    text form reads ``kitchen.area = 16 (set by House at house.py:42; declared
    12 at room.py:10)``.
    """
    compiled, scope = _context(subject)
    linked = compiled.linked
    record = declared_path(reference) if isinstance(reference, Space) else None
    if record is not None:
        return linked.scope_provenance.get(descend(linked.scopes, scope, record))
    return linked.provenance.get(compiled.resolve(scope, reference))


def admission(candidate: Space) -> QueryResult[object] | None:
    """A candidate's own refusal of its configuration: its ``admission`` member, if any.

    A group refuses as soon as one of its constraints does, even while another
    still waits on an open choice: a core that cannot target the DSP is refused
    before its folding factors are chosen.
    """
    _execution.driver_only("admission")
    current = state(candidate)
    with current.lock:
        return _forcing.admitted(current, candidate._scope)[0]


def forced(point: Space) -> tuple[Forced, ...]:
    """The open Decisions below this scope that the configuration forces (one viable
    case each), by key, each with why every other case is not viable. Derived at read
    time and never stored: the Decision's state stays ``unassigned``."""
    _execution.driver_only("forced inspection")
    current = state(point)
    found = _forcing.forced(current) if current.forcing else _forcing.NOTHING
    linked = current.linked
    included = _scope_set(linked, point._scope)
    with current.lock:
        result = [
            Forced(
                decision_key(linked, index),
                cast(
                    Available[object], _runtime.copy_result(current, index, Available(value))
                ).value,
                found.verdicts[index].reasons,
            )
            for index, value in found.values.items()
            if linked.nodes[index].scope in included
        ]
    return tuple(sorted(result, key=lambda item: item.key))


def viable(point: Space) -> tuple[Viable, ...]:
    """The open Decisions below this scope that the configuration leaves to choose:
    applicable, neither committed nor forced, each with its viable cases and why every
    other case is not viable, in rank order (an enclosing Decision before the ones it
    guards). A Decision whose guard waits on an open choice, or whose cases are not
    enumerable (a domain known by membership only), is not listed: forcing reads
    no verdict for it. A refused Decision is listed with no case."""
    _execution.driver_only("viable inspection")
    current = state(point)
    found = _forcing.forced(current) if current.forcing else _forcing.NOTHING
    linked = current.linked
    included = _scope_set(linked, point._scope)
    with current.lock:
        result = [
            Viable(
                decision_key(linked, index),
                tuple(
                    cast(
                        Available[object], _runtime.copy_result(current, index, Available(case))
                    ).value
                    for case in verdict.cases
                ),
                verdict.reasons,
            )
            for index, verdict in sorted(
                found.verdicts.items(), key=lambda item: linked.ranks[item[0]]
            )
            if verdict.cases is not None
            and index not in current.assignments
            and index not in found.values
            and linked.nodes[index].scope in included
        ]
    return tuple(result)


def open(point: Space) -> tuple[Open, ...]:  # noqa: A001 - the inspection's name
    """The open Decisions below this scope whose cases the engine cannot enumerate (a
    domain known by membership only, a FIFO's depth): applicable, neither committed
    nor forced, in rank order. A policy proposes values for them, which a commitment
    checks against the domain like any other (``domain-membership``)."""
    _execution.driver_only("open inspection")
    current = state(point)
    found = _forcing.forced(current) if current.forcing else _forcing.NOTHING
    linked = current.linked
    included = _scope_set(linked, point._scope)
    return tuple(
        Open(decision_key(linked, index), bool(domain and domain.ordered))
        for index, verdict in sorted(found.verdicts.items(), key=lambda item: linked.ranks[item[0]])
        if verdict.membership
        and index not in current.assignments
        and index not in found.values
        and linked.nodes[index].scope in included
        for domain in (linked.nodes[index].domain,)
    )


def pinned(subject: Space | Model[S] | type[Space]) -> tuple[Provenance, ...]:
    """Every decision key an override removed by pinning its coordinate, with who pinned it."""
    compiled, scope = _context(subject)
    prefix = compiled.linked.scopes[scope].name
    return tuple(
        item
        for key, item in sorted(compiled.linked.pinned.items())
        if not prefix or key.startswith(prefix + ".")
    )


def dependencies(
    subject: Space | Model[S] | type[Space],
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


@overload
def explain(point: Space, reference: T) -> QueryEvidence[T]: ...


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
            demanded = set(entry.dependencies)
            aliases = dict.fromkeys(
                alias for alias, source in (*node.via, *entry.via) if source in demanded
            )
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
                    tuple(_node_info(linked, alias) for alias in aliases),
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

    ``bindings`` maps each member this node's own body set (at the call or by
    direct assignment) to its supplier: a value, a reference, a fresh Decision,
    or a node. ``nested`` maps the members of descendants assigned through a
    path (``"port.dtype"``), which override what the bodies inside set.
    ``unsupplied`` names the required formals nothing supplies yet, and
    ``frozen`` says why assignment is closed.
    """

    space_type: type[Space]
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
        raise RequestError(
            "declaration() takes a node declaration, as returned by calling a Space class"
        )

    def public(value: object) -> object:
        if isinstance(value, NodeDecl):
            return value.instance
        if isinstance(value, NodeDecision):
            return value.proxy
        if not isinstance(value, (ValueRef, View)):
            # A detached copy: the declaration keeps its own frozen literal.
            return copy.deepcopy(value)
        return value

    bindings = {name: public(value) for name, value in record.bindings.items()}
    nested = {path: public(value) for path, value in record.nested.items()}
    return NodeDeclaration(
        record.space_type,
        record.name,
        record.placement,
        tuple(collect_space(record.space_type).members),
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


__all__ = [
    "CaseInfo",
    "ChoiceInfo",
    "DecisionInfo",
    "EvidenceNode",
    "Forced",
    "Layer",
    "NodeInfo",
    "Provenance",
    "ModelStatistics",
    "Open",
    "QueryEvidence",
    "NodeDeclaration",
    "ReferenceInfo",
    "Viable",
    "admission",
    "candidate",
    "choices",
    "declaration",
    "decision_handle",
    "decision_info",
    "decisions",
    "dependencies",
    "explain",
    "forced",
    "members",
    "model",
    "open",
    "pinned",
    "provenance",
    "reference",
    "statistics",
    "value_handle",
    "viable",
]
