# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public metadata, validity, possible values, choice states and demanded query evidence.

Metadata inspection does not invoke evaluators or candidate providers. Validity reads
a configuration's pieces in its world; ``viable`` tries each listed value of a choice as a
change would set it, publishing nothing; ``states`` reads each choice's state; explain
evaluates one query and follows the reads its entries recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, overload

from ._assign import check, placed, tried
from ._configuration import Space
from ._failure import driver_only
from ._state import State, copy_assessment, copy_result, decision_state, presence
from ._world import Entry, World, listed_values
from .compiler import Model, compile_model
from .declarations import (
    Constraint,
    Decision,
    ValueRef,
    View,
)
from .edits import BrokenRule
from .errors import RequestError
from .ir import Linked, NodeId, NodeKind
from .occurrence import candidate as _candidate
from .occurrence import decision_index, model_of, state
from .references import DecisionHandle, ValueHandle, decision_key
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
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
    has no safe value (``Decision(required=True)``): no policy gives it one, a strategy
    sets it; ``listed``: its domain lists its values (false: it is known by membership
    only, as a FIFO's depth).
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
    listed: bool = True


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
    """Count direct structure on demand, with no runtime or closure retention. It reads
    every scope and node: it addresses and resolves the whole design space (piece 3 counts
    it from templates)."""

    if not isinstance(subject, (Model, Space, type)):
        raise RequestError("statistics requires a compiled model")
    linked = _context(subject)[0].linked
    # By iteration (a structure's ``len(nodes)`` bounds its ids; it counts no nodes).
    scopes, nodes = tuple(linked.scopes), tuple(node for node in linked.nodes)
    placements = len(scopes) - 1
    authored = sum(len(scope.named_members) for scope in scopes)
    return ModelStatistics(
        authored + placements,
        len(scopes),
        len(nodes),
        sum(len(node.dependencies) for node in nodes),
        len(tuple(linked.choices)),
        len(linked.decisions),
    )


def _context(subject: Space | Model[S] | type[Space]) -> tuple[Model[Space], int]:
    if isinstance(subject, Model):
        return cast(Model[Space], subject), 0
    if isinstance(subject, type) and issubclass(subject, Space):
        return cast(Model[Space], compile_model(subject)), 0
    if isinstance(subject, Space):
        return model_of(subject), subject._scope
    raise RequestError("inspection requires a configuration, a compiled model or a Space class")


def model(subject: Space | Model[S] | type[Space]) -> Model[Space]:
    """The compiled model of a configuration, or of a Space class compiled alone."""
    return _context(subject)[0]


def _scope_set(linked: Linked, root: int) -> set[int]:
    # Scopes are allocated after their parent. No repeated ancestor traversal
    # or stored transitive closure is needed to inspect a subtree. It reads every scope:
    # it addresses the whole design space (piece 3 answers from templates).
    included = {root}
    for scope in linked.scopes:
        if scope.parent in included:
            included.add(scope.index)
    return included


def _node_info(linked: Linked, index: int) -> NodeInfo:
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
    linked: Linked,
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
        node.domain is not None and node.domain.candidates is not None,
    )


def decisions(subject: Space | Model[S] | type[Space]) -> tuple[DecisionInfo[object], ...]:
    """Discover all owning decisions below this scope, including inactive cases, in the
    flow order (``_flow``). Read once per compiled model and scope. The flow order reads
    the whole design space (piece 3 answers it from templates)."""

    compiled, scope = _context(subject)
    found = compiled._read.get(("decisions", scope))
    if found is None:
        found = compiled._read[("decisions", scope)] = tuple(
            _decision_info(compiled.linked, index) for index in _flow(compiled, scope)
        )
    return cast(tuple[DecisionInfo[object], ...], found)


def _flow(compiled: Model[Space], scope: int) -> tuple[NodeId, ...]:
    """The choices below ``scope`` in the flow order: each after every choice its
    presence, its list or its Space's facts read (a selector before the choices its cases
    hold), ties in declaration order (``_reads.flow_order``)."""
    linked = compiled.linked
    included = _scope_set(linked, scope)
    return tuple(
        index for index in compiled.evaluator.flow() if linked.nodes[index].scope in included
    )


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
    """Every Decision over nodes below this scope, with its candidates (it addresses the
    whole design space)."""
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
    """Inspect every named member below this scope (including forwarding aliases; it
    addresses the whole design space)."""

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
    current: State,
    question: str,
    scope: int,
    answer: Callable[[_Copies], A],
) -> A:
    """``answer(copies)``, kept on ``current`` (an immutable configuration) where every
    value it copied is shared."""
    key = (question, scope)
    with current.evaluator.lock:
        if key in current.answers:
            return cast(A, current.answers[key])
        copies = _Copies(current)
        found = answer(copies)
        if copies.shared:
            current.answers[key] = found
        return found


class _Copies:
    """Decision values detached from a configuration's evaluations (``copy_result``),
    and whether each copy is the value itself."""

    def __init__(self, current: State) -> None:
        self.current = current
        self.shared = True

    def __call__(self, index: NodeId, value: object) -> object:
        copied = cast(
            Available[object], copy_result(self.current.linked, index, Available(value))
        ).value
        self.shared = self.shared and copied is value
        return copied


@dataclass(frozen=True, slots=True)
class Piece:
    """One piece of a Space's validity that is not true: its owner (the key of a
    constraint, of a choice whose domain lists no value or whose guard is refused; or
    the name of a placed Space whose presence waits or is refused) and its answer:
    ``Rejected`` (false, its findings the reasons) or ``Unresolved`` (unknown, its
    findings what it waits on)."""

    owner: str
    result: Rejected | Unresolved


@dataclass(frozen=True, slots=True)
class Validity:
    """A Space's validity: ``"valid"`` when every piece that counts is true, ``"refused"``
    when one is false, ``"open"`` otherwise (Kleene's AND: false beats unknown, unknown
    beats true). ``pieces`` are the false pieces of a refused Space, or the unknown
    pieces of an open one, in key order, each from the Space or the placed Space that
    owns it; ``result`` is the answer they combine to (``Available(True)``,
    ``Rejected``, ``Unresolved``; ``Inapplicable`` for a Space that is absent, a
    candidate not selected, which counts as valid)."""

    status: Literal["valid", "refused", "open"]
    pieces: tuple[Piece, ...]
    result: QueryResult[bool]


def validity(point: Space) -> Validity:
    """The validity of this configuration's Space (its root's is the configuration's):
    every constraint of the Space and of each placed Space that is present (each unknown
    while its presence waits), every present choice whose domain lists no value
    (``domain-empty``; unknown while its presence waits), and every guard refused through
    a refused input (``guard-refused``, once per guard). A stored value's type check is
    no piece, and a candidate that is not selected contributes nothing. Valid, refused
    naming the false pieces, or open naming the unknown pieces and what each waits on
    (``Validity``)."""
    driver_only("validity")
    current = state(point)
    scope = point._scope
    return _answered(current, "validity", scope, lambda _: _validity_of(current, scope))


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


# -- which values are possible --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Possible:
    """A value nothing refuses: setting it alone is accepted. ``dormant``, ``in_play`` and
    ``opened``: the stored choices it would put out of play, those it would bring into
    play, and the open choices it would bring into presence, by key."""

    value: object
    dormant: tuple[str, ...] = ()
    in_play: tuple[str, ...] = ()
    opened: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Refused:
    """A value whose setting alone would break a rule: each rule, at its origin, with the
    stored choices standing in its way (none where the value and the facts alone break
    it)."""

    value: object
    broken: tuple[BrokenRule, ...]


@dataclass(frozen=True, slots=True)
class Viable:
    """One choice's listed values, each possible or refused, in its domain's list order.
    ``values`` is ``None`` where nothing can be listed: a domain known by membership only,
    or a list that waits on missing facts (``waits``, by key) or is refused through an
    input of the domain."""

    key: str
    values: tuple[Possible | Refused, ...] | None
    waits: tuple[str, ...] = ()

    @property
    def cases(self) -> tuple[object, ...]:
        """The possible values, in list order."""
        if self.values is None:
            return ()
        return tuple(item.value for item in self.values if isinstance(item, Possible))


def viable(point: Space, keys: Iterable[object]) -> tuple[Viable, ...]:
    """Each named choice's listed values (its domain over the facts, listed as if the
    choice existed), each tried alone as a change would set it, publishing nothing:
    possible, with what it would move, or refused, with each rule it would break and
    the stored choices standing in its way. A rule waiting on an open choice does not
    refuse; the value stored is possible (setting it changes nothing). In the order of
    ``keys``."""
    driver_only("possible values")
    current = state(point)
    indices = [decision_index(point, key) for key in keys]
    return tuple(_viable(current, index) for index in indices)


def _waits(result: object) -> tuple[str, ...]:
    """What an unresolved answer waits on: its findings' owners (the open choices and the
    missing facts), by key."""
    if not isinstance(result, Unresolved):
        return ()
    return tuple(sorted({finding.owner for finding in result.findings}))


def _viable(current: State, index: NodeId) -> Viable:
    linked = current.linked
    node = linked.nodes[index]
    listed = listed_values(placed(current.world, index), index)
    if not isinstance(listed, Available) or listed.value is None:
        return Viable(node.key, None, _waits(listed))
    copies = _Copies(current)
    found: list[Possible | Refused] = []
    for value in cast(tuple[object, ...], listed.value):
        trial = tried(current, index, value)
        public = copies(index, value)
        if trial.accepted:
            moved = trial.moved
            found.append(Possible(public, moved.dormant, moved.in_play, moved.opened))
        else:
            found.append(Refused(public, trial.broken))
    return Viable(node.key, tuple(found))


# -- every choice's state --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Set:
    """Stored and in play: it reads as its value."""

    value: object


@dataclass(frozen=True, slots=True)
class Waiting:
    """Stored, never read as a value: its presence is unknown, or its domain waits on a
    missing fact. ``waits``: what it waits on (open choices, missing facts), by key."""

    value: object
    waits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Dormant:
    """Stored, out of play: its scope is absent (a case not selected, a false guard) or
    under a guard refused through an input. Kept, and back in play with its scope."""

    value: object


@dataclass(frozen=True, slots=True)
class Open:
    """Unset, in a scope that is present (``waits`` empty) or whose presence is unknown
    (``waits``: what that presence waits on, by key). ``required``: no policy gives it a
    value."""

    waits: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True, slots=True)
class Absent:
    """Unset, its scope absent or under a guard refused through an input: not in play."""


ChoiceState: TypeAlias = Set | Waiting | Dormant | Open | Absent


def states(point: Space) -> tuple[tuple[str, ChoiceState], ...]:
    """Every choice below this scope, by key, with its state, in the flow order (each
    after every choice its presence, its list or its Space's facts read: a selector before
    the choices its cases hold)."""
    driver_only("choice states")
    current = state(point)
    nodes = current.linked.nodes
    return tuple(
        (nodes[index].key, _state_of(current, index))
        for index in _flow(current.model, point._scope)
    )


def _state_of(current: State, index: NodeId) -> ChoiceState:
    world = current.world
    node = current.linked.nodes[index]
    found, guard = presence(world, index)
    if index not in world.stored:
        if found == "absent":
            return Absent()
        return Open(_waits(guard) if found == "unknown" else (), node.required_choice)
    value = _Copies(current)(index, world.stored[index])
    if found == "absent":
        return Dormant(value)
    if found == "unknown":
        # Its presence's waits and its domain's missing facts, never itself.
        waits = {*_waits(guard), *_waits(check(world, index))}
        return Waiting(value, tuple(sorted(waits)))
    read = world.result(index)
    return Waiting(value, _waits(read)) if isinstance(read, Unresolved) else Set(value)


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

    driver_only("dependency inspection")
    current = state(point)
    return _evidence(current, current.model.resolve(point._scope, reference))


def candidate(point: Space, decision: object, case: str) -> Space | None:
    """A candidate's configuration, selected or not; None for a None candidate."""

    return _candidate(point, decision, case)


__all__ = [
    "Absent",
    "CaseInfo",
    "ChoiceInfo",
    "ChoiceState",
    "DecisionInfo",
    "Dormant",
    "EvidenceNode",
    "NodeInfo",
    "ModelStatistics",
    "Open",
    "Piece",
    "Possible",
    "QueryEvidence",
    "Refused",
    "Set",
    "Validity",
    "Viable",
    "Waiting",
    "candidate",
    "choices",
    "decision_handle",
    "decisions",
    "explain",
    "members",
    "model",
    "states",
    "statistics",
    "validity",
    "value_handle",
    "viable",
]
