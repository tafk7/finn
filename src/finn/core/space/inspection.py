# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public metadata and demanded query evidence.

Metadata inspection does not invoke evaluators or candidate providers. Explain
evaluates one query and follows only cached demanded edges for that snapshot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast, overload

from . import _execution, _forcing, _runtime
from ._configuration import Space
from ._forcing import Forced, Open, Viable
from ._state import State, copy_assessment, copy_result, decision_state
from ._world import Entry, World
from .compiler import Model, compile_model
from .declarations import (
    Constraint,
    Decision,
    ValueRef,
    View,
)
from .errors import RequestError
from .ir import LinkedModel, NodeId, NodeKind
from .occurrence import candidate as _candidate
from .occurrence import state
from .references import DecisionHandle, ValueHandle, decision_key
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
    ViewAssessment,
)

T = TypeVar("T")
S = TypeVar("S", bound=Space)
A = TypeVar("A")


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
    """An owning choice. Its stable key never exposes internal selector names.

    ``space_type`` is the Space class whose scope declares it (a candidate's, for a
    choice nested under a Decision over nodes); ``ordered``, whether its domain
    states an order of its cases (``Domain.ordered``); ``required``: the Decision
    has no safe default (``Decision(required=True)``), so setting choices to their
    first case leaves it open.
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
    # None when its guard can never hold here: it has no key.
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
    generated output, guard and select nodes are excluded.
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
    """Discover all owning decisions below this scope, including inactive cases, sorted
    by key. Read once per compiled model and scope."""

    compiled, scope = _context(subject)
    found = compiled._read.get(("decisions", scope))
    if found is None:
        included = _scope_set(compiled.linked, scope)
        result = (
            _decision_info(compiled.linked, index)
            for index in compiled.linked.decisions
            if compiled.linked.nodes[index].scope in included
        )
        found = compiled._read[("decisions", scope)] = tuple(
            sorted(result, key=lambda item: item.key)
        )
    return cast(tuple[DecisionInfo[object], ...], found)


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

    compiled, scope = _context(subject)
    return _decision_info(compiled.linked, compiled.decision(scope, reference)).reference


@overload
def value_handle(
    subject: Space | Model[S] | type[Space],
    reference: ValueRef[T] | View[T],
) -> ValueHandle[T]: ...


@overload
def value_handle(
    subject: Space | Model[S] | type[Space],
    reference: Constraint,
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


def _answered(
    current: _runtime.Snapshot,
    question: str,
    scope: int,
    answer: Callable[[_Copies], A],
) -> A:
    """``answer(copies)``, kept on ``current`` (an immutable configuration) once its
    forced Decisions are found, where every value it copied is shared."""
    key = (question, scope)
    with current.lock:
        if key in current.answers:
            return cast(A, current.answers[key])
        copies = _Copies(current)
        found = answer(copies)
        if copies.shared and (not current.forcing or isinstance(current.found, _forcing.Found)):
            current.answers[key] = found
        return found


class _Copies:
    """Decision values detached from a configuration's evaluations (``copy_result``),
    and whether each copy is the value itself."""

    def __init__(self, current: _runtime.Snapshot) -> None:
        self.current = current
        self.shared = True

    def __call__(self, index: int, value: object) -> object:
        copied = cast(
            Available[object], _runtime.copy_result(self.current, index, Available(value))
        ).value
        self.shared = self.shared and copied is value
        return copied


@dataclass(frozen=True, slots=True)
class Piece:
    """One piece of a Space's validity that is not true: its owner (the key of a
    constraint, of a stored choice, or of an open Decision with no possible case; or the
    name of a placed Space whose presence waits or is refused) and its answer:
    ``Rejected`` (false, its findings the reasons) or ``Unresolved`` (unknown, its
    findings what it waits on)."""

    owner: str
    result: Rejected | Unresolved


@dataclass(frozen=True, slots=True)
class Validity:
    """A Space's validity: ``"valid"`` when every piece that counts is true, ``"refused"``
    when one is false, ``"undecided"`` otherwise (Kleene's AND: false beats unknown,
    unknown beats true). ``pieces`` are the false pieces of a refused Space, or the
    unknown pieces of an undecided one, in key order, each from the Space or the placed
    Space that owns it; ``result`` is the answer they combine to (``Available(True)``,
    ``Rejected``, ``Unresolved``; ``Inapplicable`` for a Space that is absent, a
    candidate not selected, which counts as valid)."""

    status: Literal["valid", "refused", "undecided", "open"]
    pieces: tuple[Piece, ...]
    result: QueryResult[bool]


def _pieces(current: _runtime.Snapshot, index: int, wanted: type[object]) -> list[Piece]:
    """The pieces of the validity node ``index`` (evaluated) whose answer is a ``wanted``,
    a placed Space's own pieces in place of its validity."""
    linked = current.linked
    found: list[Piece] = []
    entry = _runtime.evaluate(current, index)
    assessment = entry.assessment
    if not isinstance(assessment, ConstraintAssessment) or not assessment.results:
        # The Space's presence waits or is refused: its guard's answer stands for it.
        if isinstance(entry.result, (Rejected, Unresolved)) and isinstance(entry.result, wanted):
            name = linked.scopes[linked.nodes[index].scope].name
            found.append(Piece(name, entry.result))
        return found
    for key, answer in assessment.results.items():
        piece = linked.keys[key]
        if linked.nodes[piece].kind == "validity":
            found.extend(_pieces(current, piece, wanted))
        elif isinstance(answer, (Rejected, Unresolved)) and isinstance(answer, wanted):
            found.append(Piece(key, answer))
    return found


def validity(point: Space) -> Validity:
    """The validity of this configuration's Space (its root's is the configuration's):
    every constraint of the Space and of each placed Space that is present (each unknown
    while its presence waits), every stored choice's own check (an unchecked one is unknown;
    one whose domain reads a refused input is false, refused through that input), and every
    open Decision with no possible case (``decision-no-viable-case``, an empty list too). A
    candidate that is not selected contributes nothing. Valid, refused naming the false
    pieces, or undecided naming the unknown pieces and what each waits on (``Validity``)."""
    _execution.driver_only("validity")
    current = state(point)
    scope = point._scope

    def compute(_: _Copies) -> Validity:
        result, index = _forcing.validity(current, scope)
        answer = cast(QueryResult[bool], result)
        if isinstance(answer, Rejected):
            return Validity("refused", _ordered(_pieces(current, index, Rejected)), answer)
        if isinstance(answer, Unresolved):
            return Validity("undecided", _ordered(_pieces(current, index, Unresolved)), answer)
        return Validity("valid", (), answer)

    return _answered(current, "validity", scope, compute)


def _ordered(pieces: list[Piece]) -> tuple[Piece, ...]:
    return tuple(sorted(pieces, key=lambda piece: piece.owner))


def _pieces_in(
    world: World, index: NodeId, wanted: type[object], answer: QueryResult[bool] | None = None
) -> list[Piece]:
    """The pieces of the validity node ``index`` in ``world`` whose answer is a ``wanted``,
    a placed Space's own pieces in place of its validity (inside a run). ``answer``: the
    answer its parent's validity holds for it (a guard piece, where its guard is
    refused)."""
    linked = world.evaluator.linked
    found: list[Piece] = []
    entry = world._get(index)
    assessment = entry.assessment
    if not isinstance(assessment, ConstraintAssessment) or not assessment.results:
        # The Space's presence waits or is refused: its guard's answer stands for it.
        held = entry.result if answer is None else cast(QueryResult[object], answer)
        if isinstance(held, (Rejected, Unresolved)) and isinstance(held, wanted):
            name = linked.scopes[linked.nodes[index].scope].name
            found.append(Piece(name, held))
        return found
    for key, result in assessment.results.items():
        piece = linked.keys[key]
        if linked.nodes[piece].kind == "validity":
            found.extend(_pieces_in(world, piece, wanted, result))
        elif isinstance(result, (Rejected, Unresolved)) and isinstance(result, wanted):
            found.append(Piece(key, result))
    return found


def _validity_of(current: State, scope: int) -> Validity:
    """The validity of ``scope``'s Space in a configuration's world (``validity``)."""
    index = current.linked.scopes[scope].validity
    assert index is not None, "every scope has a validity node"
    world = current.world

    def walk() -> Validity:
        answer = cast(QueryResult[bool], world._get(index).result)
        if isinstance(answer, Rejected):
            return Validity("refused", _ordered(_pieces_in(world, index, Rejected)), answer)
        if isinstance(answer, Unresolved):
            return Validity("open", _ordered(_pieces_in(world, index, Unresolved)), answer)
        return Validity("valid", (), answer)

    return world.run(walk)


def _evidence(current: State, root: NodeId) -> QueryEvidence[object]:
    """The evidence of ``root``'s query in a configuration (``explain``): each node it
    read, through what each of those read, as the entries' traces record it."""
    linked, world = current.linked, current.world
    selectors = linked.selector_choices

    def walk() -> tuple[Entry, dict[NodeId, Entry]]:
        found = world._get(root)
        pending = [root]
        visited: dict[NodeId, Entry] = {}
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            entry = visited[index] = world._get(index)
            pending.extend(dependency for dependency, _ in entry.trace)
        return found, visited

    with current.evaluator.lock:
        result, visited = world.run(walk)
        guard_nodes = {
            linked.nodes[index].guard for index in visited if linked.nodes[index].guard is not None
        }
        evidence: list[EvidenceNode] = []
        for index in sorted(visited, key=lambda item: linked.nodes[item].key):
            node, entry = linked.nodes[index], visited[index]
            demanded = tuple(dict.fromkeys(dependency for dependency, _ in entry.trace))
            aliases = dict.fromkeys(
                alias for alias, source in (*node.via, *entry.via) if source in demanded
            )
            active = True
            if node.guard is not None:
                guard = visited[node.guard].result
                active = isinstance(guard, Available) and guard.value is True
            presence: Literal["supplied", "omitted"] | None = None
            if node.kind == "param" and active:
                presence = "supplied" if index in current.facts else "omitted"
            evidence.append(
                EvidenceNode(
                    _node_info(linked, index),
                    copy_result(linked, index, entry.result),
                    tuple(ValueHandle[object](linked, target) for target in demanded),
                    presence,
                    decision_state(current, index) if node.kind == "decision" else None,
                    index in selectors,
                    index in guard_nodes,
                    tuple(_node_info(linked, alias) for alias in aliases),
                )
            )
        assessment = (
            None if result.assessment is None else copy_assessment(linked, root, result.assessment)
        )
        return QueryEvidence(
            _node_info(linked, root),
            copy_result(linked, root, result.result),
            tuple(evidence),
            assessment,
        )


def forced(point: Space) -> tuple[Forced, ...]:
    """The open Decisions below this scope that the configuration forces (one viable
    case each), by key, each with why every other case is not viable. Derived at read
    time and never stored: the Decision's state stays ``unassigned``."""
    _execution.driver_only("forced inspection")
    current = state(point)
    found = _forcing.forced(current) if current.forcing else _forcing.NOTHING
    linked = current.linked
    included = _scope_set(linked, point._scope)

    def compute(copy: _Copies) -> tuple[Forced, ...]:
        result = [
            Forced(decision_key(linked, index), copy(index, value), found.verdicts[index].reasons)
            for index, value in found.values.items()
            if linked.nodes[index].scope in included
        ]
        return tuple(sorted(result, key=lambda item: item.key))

    return _answered(current, "forced", point._scope, compute)


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

    def compute(copy: _Copies) -> tuple[Viable, ...]:
        return tuple(
            Viable(
                decision_key(linked, index),
                tuple(copy(index, case) for case in verdict.cases),
                verdict.reasons,
            )
            for index, verdict in sorted(
                found.verdicts.items(), key=lambda item: linked.ranks[item[0]]
            )
            if verdict.cases is not None
            and index not in current.assignments
            and index not in found.values
            and linked.nodes[index].scope in included
        )

    return _answered(current, "viable", point._scope, compute)


@dataclass(frozen=True, slots=True)
class Unchecked:
    """A stored choice whose own check cannot conclude yet, for inspection: its key, the
    value stored, and what it waits on: the keys of the choices still open (or unchecked
    themselves) and of the inputs not supplied that its check, and its guard, read, and,
    for an input of its domain that does not apply, of what decides that it applies (the
    choices and inputs its guard reads, or the selector whose case leaves it out). A read
    of the choice answers unresolved until a change makes its check succeed; a change then
    checks it again, and refuses it if the check fails."""

    key: str
    value: object
    blockers: tuple[str, ...]


def unchecked(point: Space) -> tuple[Unchecked, ...]:
    """The choices below this scope that are stored and unchecked (see ``Unchecked``), by
    key. A configuration that holds one is not complete: a choice kept unchecked is a
    commitment with pending evidence, not an accepted value, and what generates hardware
    from a configuration is to refuse it. A choice kept whose domain reads a refused input
    is not listed: no commitment settles it, and it reads refused through that input."""
    _execution.driver_only("unchecked inspection")
    current = state(point)
    linked = current.linked
    included = _scope_set(linked, point._scope)

    def blockers(index: int) -> tuple[str, ...]:
        node = linked.nodes[index]
        membership = _runtime.membership(current, index, current.assignments[index])
        entries = [membership]
        if node.guard is not None:
            entries.append(_runtime.evaluate(current, node.guard))
        read = {item for entry in entries for item in (*entry.decisions, *entry.facts)}
        waits = {
            item
            for item in read
            if not isinstance(_runtime.evaluate(current, item).result, Available)
        }
        # An input of the domain that does not apply: the check waits on what decides
        # that it applies (its guard's reads, or the selector leaving it out), known or not.
        for argument in node.domain_arguments:
            if isinstance(_runtime.evaluate(current, argument.node).result, Inapplicable):
                waits.update(_runtime.absence_reads(current, argument.node) or (argument.node,))
        return tuple(sorted(linked.nodes[item].key for item in waits))

    def compute(copy: _Copies) -> tuple[Unchecked, ...]:
        return tuple(
            Unchecked(decision_key(linked, index), copy(index, current.assignments[index]), found)
            for index in sorted(current.unchecked, key=lambda item: linked.nodes[item].key)
            if linked.nodes[index].scope in included and index not in current.input_refused
            for found in (blockers(index),)
        )

    return _answered(current, "unchecked", point._scope, compute)


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
    return _answered(
        current,
        "open",
        point._scope,
        lambda _: tuple(
            Open(decision_key(linked, index), bool(domain and domain.ordered))
            for index, verdict in sorted(
                found.verdicts.items(), key=lambda item: linked.ranks[item[0]]
            )
            if verdict.membership
            and index not in current.assignments
            and index not in found.values
            and linked.nodes[index].scope in included
            for domain in (linked.nodes[index].domain,)
        ),
    )


@overload
def explain(point: Space, reference: ValueRef[T] | View[T]) -> QueryEvidence[T]: ...


@overload
def explain(
    point: Space,
    reference: Constraint,
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


def candidate(point: Space, decision: object, case: str) -> Space | None:
    """A candidate's configuration, selected or not; None for a None candidate."""

    return _candidate(point, decision, case)


__all__ = [
    "CaseInfo",
    "ChoiceInfo",
    "DecisionInfo",
    "EvidenceNode",
    "Forced",
    "NodeInfo",
    "ModelStatistics",
    "Open",
    "Piece",
    "QueryEvidence",
    "Unchecked",
    "Validity",
    "Viable",
    "candidate",
    "choices",
    "decision_handle",
    "decisions",
    "explain",
    "forced",
    "members",
    "model",
    "open",
    "statistics",
    "unchecked",
    "validity",
    "value_handle",
    "viable",
]
