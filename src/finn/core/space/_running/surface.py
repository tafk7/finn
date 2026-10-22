# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The rest of the public questions, each a function of a design space value: what today's
configuration, its bound accessors and ``inspection`` answer beyond presence, validity, the
states, the possible values, a change and a bind (``questions``). Each returns today's
public result type, its values detached as today's are (by each member's semantics).

A configuration of today's API is a design space read at one of its scopes: ``At``, the
design space and the scope (the root, a child's, a candidate's, selected or not). What is
the compiled class's alone (resolving a reference, the members, the decisions, the Decisions
over nodes, a scope's children and candidates, the statistics, the handles) stays the
compiled class's; here it is only read, to address the answer.

- ``query``: a member's answer, or a node's configuration (``At``) where it is present;
  ``value``: a member read as a value (a non-value raises ``ValueUnavailableError``);
- ``present``: whether a node is present, or a value input supplied;
- ``inspect``: a view's or a constraint's assessment;
- ``decision_state``: a Decision's commitment; ``candidates``: a choice's listed values,
  ranked by its domain's order where it has one, under its guard;
- ``field``: a bound accessor (``Field``, ``DecisionField``) asking the above;
- ``explain``: what an answer read, each node through what it read (``QueryEvidence``);
- ``root``, ``child``, ``candidate``, ``selected_candidate``: the configurations below.

Each question is one run of its session (``evaluate.Session.run``): a failure raises as
today's driver raises it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from .. import _world as W
from .._configuration import Space
from .._failure import Call
from .._nodes import NodeDecl
from .._state import copy_assessment, copy_result
from ..declarations import Param, declared_path
from ..errors import EvaluationError, RequestError
from ..inspection import EvidenceNode, QueryEvidence, _node_info
from ..ir import Node, NodeId
from ..occurrence import _scope_choice
from ..references import ValueHandle, descend
from ..results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    Unresolved,
    ViewAssessment,
    require_value,
)
from ..semantics import snapshot
from .evaluate import INAPPLICABLE, Ans, Evaluation, Session, _Returned, evaluation, invoke
from .questions import presence
from .value import DesignSpace

Assessment = ViewAssessment[object] | ConstraintAssessment


@dataclass(frozen=True)
class At:
    """A design space read at one of its scopes (by index): today's configuration of that
    scope. Equal by value, as its design space is."""

    ds: DesignSpace
    scope: int = 0


def _evaluation(at: At, session: Session) -> Evaluation:
    return evaluation(at.ds, session)


def _node_scope(at: At, reference: object) -> int | None:
    """For a node reference (a child or a candidate), the node's scope; ``None`` for
    anything else (today's ``occurrence._node_scope``)."""
    if not isinstance(reference, Space):
        return None
    path = declared_path(reference)
    if path is None:
        return None
    *outer, last = path
    if not isinstance(last, NodeDecl):
        return None
    scopes = at.ds.model.linked.scopes
    holder = descend(scopes, at.scope, outer)
    scope = scopes[holder]
    if last is scope.record:
        return holder
    target = scope.children.get(last)
    if target is None:
        raise RequestError("node is not part of this compiled scope")
    return target


def _public(ev: Evaluation, index: NodeId) -> QueryResult[object]:
    """``index``'s answer, detached by its semantics (today's ``_state.answer``)."""
    return copy_result(ev.linked, index, ev.result(index))


# -- a member's answer, a value, presence -----------------------------------------------


def query(at: At, reference: object, session: Session) -> QueryResult[object]:
    """A member's answer (``Space.query``): what reading it would give, as a result. A node
    (a child or a candidate) answers its configuration where it is present: inapplicable
    where it is absent, its guard's answer while its presence waits or is refused."""
    ev = _evaluation(at, session)
    target = _node_scope(at, reference)
    if target is not None:
        guard = at.ds.model.linked.scopes[target].guard
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
    return require_value(_public(ev, index), context=ev.linked.header(index).owner)


def present(at: At, node: object, session: Session) -> bool:
    """Whether a node (a child or a candidate) is present, or a value input supplied (a
    source applies), read like a value: undecided presence raises
    ``ValueUnavailableError`` (``Space.present``)."""
    ev = _evaluation(at, session)
    target = _node_scope(at, node)
    if target is None:
        if isinstance(node, Param):
            index = at.ds.model.resolve(at.scope, node)
            applies = ev.run(lambda: _source_applies(ev, index))
            if isinstance(applies, bool):
                return applies
            require_value(_public(ev, applies), context=ev.linked.header(applies).owner)
            raise AssertionError("an undecided presence read halts")
        raise RequestError("present() takes a node (a child or a candidate) or a value input")
    guard = at.ds.model.linked.scopes[target].guard
    if guard is None:
        return True
    answer = ev.result(guard)
    if isinstance(answer, Available):
        return answer.value is True
    if isinstance(answer, Inapplicable) or (isinstance(answer, Unresolved) and _unsupplied(answer)):
        return False
    return bool(require_value(answer, context=ev.linked.header(guard).owner))


def _unsupplied(answer: Unresolved) -> bool:
    findings = answer.findings
    return bool(findings) and all(finding.code == "input-unsupplied" for finding in findings)


def _source_applies(ev: Evaluation, index: NodeId) -> bool | NodeId:
    """Whether a source applies at ``index`` (a value input is supplied), or the node whose
    answer still decides it: the edges a read of the value follows, reading only what
    decides presence, never a computation (today's ``_state.source_applies``)."""
    node = ev.raw[index]
    if node.guard is not None:
        held = ev.result(node.guard)
        if not isinstance(held, Available):
            return False if isinstance(held, Inapplicable) else node.guard
        if held.value is not True:
            return False
    if node.kind == "param":
        return isinstance(ev.result(index), Available)
    if node.kind in ("alias", "guard", "view"):
        return _source_applies(ev, cast(NodeId, node.output))
    if node.kind == "select":
        selector = cast(NodeId, node.selector)
        case = ev.result(selector)
        if not isinstance(case, Available):
            return False if isinstance(case, Inapplicable) else selector
        target = cast(Mapping[str, NodeId], node.selection_index).get(cast(str, case.value))
        return False if target is None else _source_applies(ev, target)
    if node.kind == "present":
        answers = [_source_applies(ev, target) for _, target in node.alternatives]
        if any(answer is True for answer in answers):
            return True
        return next((answer for answer in answers if answer is not False), False)
    return True


# -- assessments, a Decision's state and its candidates ---------------------------------


def inspect(at: At, reference: object, session: Session) -> Assessment:
    """A view's or a constraint's assessment (``Space.inspect``), detached."""
    index = at.ds.model.resolve(at.scope, reference)
    ev = _evaluation(at, session)
    header = ev.linked.header(index)
    if header.kind not in {"view", "constraint"}:
        raise RequestError(
            f"{header.key} is not a view or a constraint: inspect() assesses those; "
            "read or query() any other member"
        )
    assessment = ev.get(index).assessment
    assert assessment is not None
    return copy_assessment(ev.linked, index, assessment)


def _decision_state(ev: Evaluation, index: NodeId) -> QueryResult[DecisionState[object]]:
    """A Decision's state: committed with its stored value (whatever a read of it
    answers), unassigned where it is present and open, else as its guard answers
    (today's ``_state.decision_state``)."""
    node = ev.raw[index]
    if node.kind != "decision":
        raise EvaluationError(node.owner, "decision state", "reference is not an owning decision")
    stored = ev.ds.stored
    if index in stored:
        assert node.semantics is not None
        held = snapshot(node.semantics, stored[index], owner=node.owner, role="decision state")
        return Available(DecisionState(node.owner, "committed", held))
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
    accepts, read as if the choice existed), ranked by its domain's order where it has
    one, under its guard: ``None`` where it is known by membership only
    (``BoundDecision.candidates``)."""
    index = at.ds.model.decision(at.scope, reference)
    ev = _evaluation(at, session)
    node = ev.raw[index]
    found = ev.run(lambda: _candidates(ev, node))
    if isinstance(found, Available) and found.value is None:
        return None
    if not isinstance(found, Available):
        return cast(QueryResult[tuple[object, ...]], found)
    semantics = node.semantics
    assert semantics is not None
    return Available(
        tuple(
            snapshot(semantics, item, owner=node.owner, role="public value snapshot")
            for item in cast(tuple[object, ...], found.value)
        )
    )


def _candidates(ev: Evaluation, node: Node) -> QueryResult[object]:
    domain, semantics = node.domain, node.semantics
    if domain is None or semantics is None:
        raise EvaluationError(node.owner, "domain enumeration", "reference has no domain")
    if node.guard is not None:
        inactive = W._guard_result(node, ev.result(node.guard))
        if inactive is not None:
            return inactive
    listed = ev.membership(node.index).listed(node, [])
    if domain.order is None or not isinstance(listed, Available) or listed.value is None:
        return listed
    arguments = W.order_arguments(node)
    answers = [ev.result(argument.node) for argument in arguments]
    blocked = W._blocked([answer for answer in answers if not isinstance(answer, Available)])
    if blocked is not None:
        return blocked
    values = {
        argument.name: W._clone(
            ev.raw[argument.node],
            cast(Available[object], answer).value,
            owner=node.owner,
            role="dependency snapshot",
        )
        for argument, answer in zip(arguments, answers)
    }
    ev.counters.call("domain order")
    ranked = invoke(
        ev.session,
        Call(node.owner, "domain order"),
        domain.rank,
        (listed.value, values),
        {"semantics": semantics, "owner": node.owner},
    )
    return Available(cast(_Returned, ranked).value)


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
    return at if at.scope == 0 else At(at.ds, 0)


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
        raise RequestError("a candidate key must be a string")
    choice = _scope_choice(at.ds.model.linked, at.scope, decision)
    for key, index in choice.cases:
        if key == case:
            return None if index is None else At(at.ds, index)
    raise RequestError(f"{choice.key}: unknown candidate {case!r}")


def selected_candidate(at: At, decision: object, session: Session) -> At | None:
    """The configuration of the candidate a Decision over nodes selects, or ``None``: its
    selector read as a value."""
    choice = _scope_choice(at.ds.model.linked, at.scope, decision)
    case = value(at, ValueHandle[str](at.ds.model.linked, choice.selector), session)
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


def _evidence(ev: Evaluation, index: NodeId) -> QueryEvidence[object]:
    linked, raw = ev.linked, ev.raw
    found = ev.get(index)
    pending = [index]
    visited: dict[NodeId, Ans] = {}
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        entry = visited[current] = ev.get(current)
        pending.extend(dependency for dependency, _ in entry.trace)
    guards = {raw[item].guard for item in visited if raw[item].guard is not None}
    nodes: list[EvidenceNode] = []
    for item in sorted(visited, key=lambda each: linked.header(each).key):
        node, entry = raw[item], visited[item]
        demanded = tuple(dict.fromkeys(dependency for dependency, _ in entry.trace))
        active = True
        if node.guard is not None:
            guard = visited[node.guard].result
            active = isinstance(guard, Available) and guard.value is True
        supplied: Literal["supplied", "omitted"] | None = None
        if node.kind == "param" and active:
            supplied = "supplied" if item in ev.ds.facts else "omitted"
        nodes.append(
            EvidenceNode(
                _node_info(linked, item),
                copy_result(linked, item, entry.result),
                tuple(ValueHandle[object](linked, target) for target in demanded),
                supplied,
                _decision_state(ev, item) if node.kind == "decision" else None,
                item in linked.selector_choices,
                item in guards,
                tuple(_node_info(linked, formal) for formal in _read_through(ev, node, demanded)),
            )
        )
    assessment = (
        None if found.assessment is None else copy_assessment(linked, index, found.assessment)
    )
    return QueryEvidence(
        _node_info(linked, index),
        copy_result(linked, index, found.result),
        tuple(nodes),
        cast(Any, assessment),
    )


def _read_through(ev: Evaluation, node: Node, demanded: tuple[NodeId, ...]) -> tuple[NodeId, ...]:
    """The formals bound by reference ``node`` read at their source: each node its
    definition names (its output, arguments, domain inputs, alternatives, selector, and a
    method's possible reads) that ``node`` reads at another node it then read."""
    named: list[NodeId] = []
    if node.output is not None:
        named.append(node.output)
    named.extend(argument.node for argument in node.arguments)
    named.extend(argument.node for argument in node.domain_arguments)
    named.extend(target for _, target in node.alternatives)
    if node.selector is not None:
        named.append(node.selector)
    if node.call_style == "self":
        named.extend(sorted(ev.inferred.methods.get(node.index) or ()))
    read = set(demanded)
    found: dict[NodeId, None] = {}
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
