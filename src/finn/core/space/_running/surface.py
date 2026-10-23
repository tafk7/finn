# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The rest of the public questions, each a function of a design space value: what today's
configuration, its bound accessors and ``inspection`` answer beyond presence, validity, the
states, the possible values, a change and a bind (``questions``). Each returns today's
public result type, its values as they are held (values are immutable), a mutable
container a derived value or a view answers copied (``answers.handed_out``).

A configuration of today's API is a design space read at one of its scopes: ``At``, the
design space and the scope (the root, a child's, a candidate's, selected or not). What is
the compiled class's alone (resolving a reference, the members, the decisions, the Decisions
over nodes, a scope's children and candidates, the statistics, the handles) stays the
compiled class's; here it is only read, to address the answer.

- ``query``: a member's answer, or a node's configuration (``At``) where it is present;
  ``value``: a member read as a value (a non-value raises ``ValueUnavailableError``);
- ``present``: whether a node is present;
- ``inspect``: a view's or a constraint's assessment;
- ``decision_state``: a Decision's commitment; ``candidates``: a choice's listed values,
  in their listed order, under its guard;
- ``field``: a bound accessor (``Field``, ``DecisionField``) asking the above;
- ``explain``: what an answer read, each node through what it read (``QueryEvidence``);
- ``root``, ``child``, ``candidate``, ``selected_candidate``: the configurations below.

Each question is one run of its session (``evaluate.Session.run``): a failure raises as
today's driver raises it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .._configuration import Space
from .._nodes import NodeDecl
from ..declarations import declared_path
from ..errors import EvaluationError, RequestError
from ..inspection import EvidenceNode, QueryEvidence, _node_info
from ..references import ValueHandle, descend
from ..results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    ViewAssessment,
    require_value,
)
from .answers import _guard_result, copy_assessment, copy_result
from .evaluate import INAPPLICABLE, Ans, Evaluation, Session, evaluation
from .questions import presence
from .receiver import not_a_node
from .receiver import scope_choice as _scope_choice
from .value import DesignSpace

if TYPE_CHECKING:
    from .._scopes import Address, ScopeRecord

Assessment = ViewAssessment[object] | ConstraintAssessment


@dataclass(frozen=True)
class At:
    """A design space read at one of its scopes (its scope record; the root's by default):
    today's configuration of that scope. Equal by value, as its design space is."""

    ds: DesignSpace
    scope: ScopeRecord = cast("ScopeRecord", None)

    def __post_init__(self) -> None:
        if self.scope is None:
            object.__setattr__(self, "scope", self.ds.model.scopes.root)


def _evaluation(at: At, session: Session) -> Evaluation:
    return evaluation(at.ds, session)


def _node_scope(at: At, reference: object) -> ScopeRecord | None:
    """For a node reference (a child or a candidate), the node's scope; ``None`` for
    anything else."""
    if not isinstance(reference, Space):
        return None
    path = declared_path(reference)
    if path is None:
        return None
    *outer, last = path
    if not isinstance(last, NodeDecl):
        return None
    holder = descend(at.scope, outer)
    if last is holder.record:
        return holder
    target = holder.children.get(last)
    if target is None:
        raise RequestError("node is not part of this compiled scope")
    return target


def _public(ev: Evaluation, index: Address) -> QueryResult[object]:
    """``index``'s answer, as a reader receives it (``answers.copy_result``)."""
    return copy_result(index, ev.result(index))


# -- a member's answer, a value, presence -----------------------------------------------


def query(at: At, reference: object, session: Session) -> QueryResult[object]:
    """A member's answer (``Space.query``): what reading it would give, as a result. A node
    (a child or a candidate) answers its configuration where it is present: inapplicable
    where it is absent, its guard's answer while its presence waits or is refused."""
    ev = _evaluation(at, session)
    target = _node_scope(at, reference)
    if target is not None:
        guard = target.guard
        if guard is not None:
            answer = ev.result(guard)
            if not isinstance(answer, Available):
                return answer
            if answer.value is not True:
                return INAPPLICABLE
        return Available(At(at.ds, target))
    return _public(ev, at.ds.model.resolve(at.scope, reference))


def value(at: At, reference: object, session: Session) -> object:
    """A member read as a value (``point.member``): a non-value raises
    ``ValueUnavailableError`` carrying it."""
    index = at.ds.model.resolve(at.scope, reference)
    ev = _evaluation(at, session)
    return require_value(_public(ev, index), context=index.owner)


def present(at: At, node: object, session: Session) -> bool:
    """Whether a node (a child or a candidate) is present, read like a value: undecided
    presence raises ``ValueUnavailableError`` (``Space.present``). A value input is known,
    never absent: it is no node (``not_a_node``)."""
    ev = _evaluation(at, session)
    target = _node_scope(at, node)
    if target is None:
        raise not_a_node(node)
    guard = target.guard
    if guard is None:
        return True
    answer = ev.result(guard)
    if isinstance(answer, Available):
        return answer.value is True
    if isinstance(answer, Inapplicable):
        return False
    return bool(require_value(answer, context=guard.owner))


# -- assessments, a Decision's state and its candidates ---------------------------------


def inspect(at: At, reference: object, session: Session) -> Assessment:
    """A view's or a constraint's assessment (``Space.inspect``), detached."""
    index = at.ds.model.resolve(at.scope, reference)
    ev = _evaluation(at, session)
    if index.kind not in {"view", "constraint"}:
        raise RequestError(
            f"{index.key} is not a view or a constraint: inspect() assesses those; "
            "read or query() any other member"
        )
    assessment = ev.get(index).assessment
    assert assessment is not None
    return copy_assessment(index, assessment)


def _decision_state(ev: Evaluation, index: Address) -> QueryResult[DecisionState[object]]:
    """A Decision's state: committed with its stored value (whatever a read of it
    answers), unassigned where it is present and open, else as its guard answers."""
    node = index
    if node.kind != "decision":
        raise EvaluationError(node.owner, "decision state", "reference is not an owning decision")
    stored = ev.ds.stored
    if index in stored:
        return Available(DecisionState(node.owner, "committed", stored[index]))
    found, answer = presence(ev, index)
    if found == "present":
        return Available(DecisionState(node.owner))
    if isinstance(answer, Available):
        return Inapplicable()
    return cast(QueryResult[DecisionState[object]], answer)


def decision_state(
    at: At, reference: object, session: Session
) -> QueryResult[DecisionState[object]]:
    """A Decision's state (``BoundDecision.state``)."""
    index = at.ds.model.decision(at.scope, reference)
    return _decision_state(_evaluation(at, session), index)


def candidates(
    at: At, reference: object, session: Session
) -> QueryResult[tuple[object, ...]] | None:
    """A choice's listed values (its domain's enumeration, each value its membership
    accepts, read as if the choice existed), in their listed order, under its guard:
    ``None`` where it is known by membership only (``BoundDecision.candidates``)."""
    index = at.ds.model.decision(at.scope, reference)
    ev = _evaluation(at, session)
    node = index
    found = ev.run(lambda: _candidates(ev, node))
    if isinstance(found, Available) and found.value is None:
        return None
    return cast(QueryResult[tuple[object, ...]], found)


def _candidates(ev: Evaluation, node: Address) -> QueryResult[object]:
    domain, semantics = node.domain, node.semantics
    if domain is None or semantics is None:
        raise EvaluationError(node.owner, "domain enumeration", "reference has no domain")
    if node.guard is not None:
        inactive = _guard_result(node, ev.result(node.guard))
        if inactive is not None:
            return inactive
    return ev.membership(node).listed(node, [])


# -- bound accessors --------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """A member bound to a configuration (``BoundValue``): its value and its answer."""

    at: At
    reference: object
    session: Session

    def get(self) -> object:
        return value(self.at, self.reference, self.session)

    def query(self) -> QueryResult[object]:
        return query(self.at, self.reference, self.session)


@dataclass(frozen=True)
class DecisionField(Field):
    """A Decision bound to a configuration (``BoundDecision``): also its state and its
    candidates."""

    @property
    def state(self) -> QueryResult[DecisionState[object]]:
        return decision_state(self.at, self.reference, self.session)

    def candidates(self) -> QueryResult[tuple[object, ...]] | None:
        return candidates(self.at, self.reference, self.session)


def field(at: At, reference: object, session: Session) -> Field:
    """A bound accessor (``Space.field``): a Decision's where the reference edits one, a
    value's for anything else read as a value."""
    model = at.ds.model
    model.resolve(at.scope, reference)
    try:
        model.decision(at.scope, reference)
    except RequestError:
        return Field(at, reference, session)
    return DecisionField(at, reference, session)


# -- the configurations below -----------------------------------------------------------


def root(at: At) -> At:
    return at if at.scope.parent is None else At(at.ds, at.ds.model.scopes.root)


def child(at: At, node: object) -> At:
    """A child node's configuration (``point.child``), the node as its class declares it."""
    found = _node_scope(at, node)
    if found is None:
        raise RequestError("child() takes a node of this compiled scope")
    return At(at.ds, found)


def candidate(at: At, decision: object, case: str) -> At | None:
    """A candidate's configuration whether or not it is selected (``None`` for a case
    placing nothing; ``inspection.candidate``)."""
    if type(case) is not str:
        raise RequestError("a case name must be a string")
    choice = _scope_choice(at.scope, decision)
    for key, index in choice.cases:
        if key == case:
            return None if index is None else At(at.ds, index)
    raise RequestError(f"{choice.key}: unknown case {case!r}")


def selected_candidate(at: At, decision: object, session: Session) -> At | None:
    """The configuration of the candidate a Decision over nodes selects, or ``None``: its
    selector read as a value."""
    choice = _scope_choice(at.scope, decision)
    case = value(at, ValueHandle[str](at.ds.model, choice.selector), session)
    for key, index in choice.cases:
        if key == case:
            return None if index is None else At(at.ds, index)
    raise EvaluationError(choice.key, "selection", "selector is not a declared case")


# -- evidence ---------------------------------------------------------------------------


def explain(at: At, reference: object, session: Session) -> QueryEvidence[object]:
    """What an answer read (``inspection.explain``): the node asked, and each node it
    read, through what each of those read, as their answers' traces record it, each with
    its detached answer, the nodes it read (``dependencies``, in order) and the formals
    bound by reference it read at their source (``via``).

    Every node is the one evaluation read: an alias is read as a node of its own, and
    ``via`` names only the formals a reader within their scope read at their source (no
    alias is passed over otherwise)."""
    index = at.ds.model.resolve(at.scope, reference)
    ev = _evaluation(at, session)
    return ev.run(lambda: _evidence(ev, index))


def _evidence(ev: Evaluation, index: Address) -> QueryEvidence[object]:
    model = ev.model
    found = ev.get(index)
    pending = [index]
    visited: dict[Address, Ans] = {}
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        entry = visited[current] = ev.get(current)
        pending.extend(dependency for dependency, _ in entry.trace)
    guards = {item.guard for item in visited if item.guard is not None}
    nodes: list[EvidenceNode] = []
    for item in sorted(visited, key=lambda each: each.key):
        node, entry = item, visited[item]
        demanded = tuple(dict.fromkeys(dependency for dependency, _ in entry.trace))
        nodes.append(
            EvidenceNode(
                _node_info(model, item),
                copy_result(item, entry.result),
                tuple(ValueHandle[object](model, target) for target in demanded),
                _decision_state(ev, item) if node.kind == "decision" else None,
                model.scopes.selector_choice(item) is not None,
                item in guards,
                tuple(_node_info(model, formal) for formal in _read_through(ev, node, demanded)),
            )
        )
    assessment = None if found.assessment is None else copy_assessment(index, found.assessment)
    return QueryEvidence(
        _node_info(model, index),
        copy_result(index, found.result),
        tuple(nodes),
        cast(Any, assessment),
    )


def _read_through(
    ev: Evaluation, node: Address, demanded: tuple[Address, ...]
) -> tuple[Address, ...]:
    """The formals bound by reference ``node`` read at their source: each node its
    definition names (its output, arguments, domain inputs, alternatives, selector, and a
    method's possible reads, in the order it read their sources) that ``node`` reads at
    another node it then read."""
    named: list[Address] = []
    if node.output is not None:
        named.append(node.output)
    named.extend(argument.node for argument in node.arguments)
    named.extend(argument.node for argument in node.domain_arguments)
    named.extend(target for _, target in node.alternatives)
    if node.selector is not None:
        named.append(node.selector)
    if node.call_style == "self":
        order = {source: position for position, source in enumerate(demanded)}
        possible = sorted(ev.inferred.methods.get(node) or ())
        named.extend(
            sorted(possible, key=lambda formal: order.get(ev.source(formal, node), len(order)))
        )
    read = set(demanded)
    found: dict[Address, None] = {}
    for target in named:
        source = ev.source(target, node)
        if source != target and source in read:
            found[target] = None
    return tuple(found)


__all__ = [
    "At",
    "DecisionField",
    "Field",
    "candidate",
    "candidates",
    "child",
    "decision_state",
    "explain",
    "field",
    "inspect",
    "present",
    "query",
    "root",
    "selected_candidate",
    "value",
]
