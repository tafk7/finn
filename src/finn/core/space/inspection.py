# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public metadata, validity, possible values, choice states and demanded query evidence.

Metadata inspection does not invoke evaluators or candidate providers. The questions of a
design space are the running type's (``_running.questions``, ``_running.surface``), each
asked in the model's one session under its lock (``occurrence.ask``): validity reads the
pieces of a Space's validity; ``viable`` tries each listed value of a choice as a change
would set it, publishing nothing; ``states`` reads each choice's state; explain evaluates
one query and follows the reads its answers recorded. ``states`` and ``viable`` detach the
values they hand out, by each choice's semantics.
"""

# The running type imports this module's types: it is imported inside the bodies here.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast, overload

from ._configuration import Space
from ._failure import driver_only
from ._running.per_class import flow_below, scopes_below
from ._structure import Structure
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
from .occurrence import ask, held, model_of
from .occurrence import candidate as _candidate
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

if TYPE_CHECKING:
    from ._running.evaluate import Session

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
    """An owning choice. Its stable key never exposes internal selector names.

    ``space_type`` is the Space class whose scope declares it (a candidate's, for a
    choice nested under a Decision over nodes); ``ordered``, whether its domain
    states an order of its cases (``Domain.ordered``); ``listed``: its domain lists its
    values (false: it is known by membership only, as a FIFO's depth). Whether a policy
    gives it a value is the compiler's, never the engine's.
    """

    reference: DecisionHandle[T]
    key: str
    scope: str
    owner: str
    selector: bool
    cases: tuple[str, ...] = ()
    space_type: type[Space] | None = None
    ordered: bool = False
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
    """Count direct structure on demand, with no runtime or closure retention. A model's
    structure counts from its templates, each counted at every placement: once per shape
    of placement, times the placements of that shape (``Structure.statistics``; no node
    resolved); a table, from its nodes."""

    if not isinstance(subject, (Model, Space, type)):
        raise RequestError("statistics requires a compiled model")
    linked = _context(subject)[0].linked
    if isinstance(linked, Structure):
        return ModelStatistics(*linked.statistics())
    scopes, nodes = tuple(linked.scopes), tuple(linked.nodes)
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


def _scope_name(linked: Linked, scope: int) -> str:
    if isinstance(linked, Structure):
        return linked.records[scope].name
    return linked.scopes[scope].name


def _scope_type(linked: Linked, scope: int) -> type[Space]:
    if isinstance(linked, Structure):
        return linked.records[scope].template.space_type
    return linked.scopes[scope].space_type


def _node_info(linked: Linked, index: int) -> NodeInfo:
    node = linked.header(index)
    reference = (
        DecisionHandle[object](linked, index)
        if node.kind == "decision"
        else ValueHandle[object](linked, index)
    )
    return NodeInfo(
        reference,
        node.key,
        node.kind,
        _scope_name(linked, node.scope),
        node.owner,
        node.source_owner is not None,
        None if node.guard is None else ValueHandle[bool](linked, node.guard),
        node.origin,
    )


def _decision_info(
    linked: Linked,
    index: int,
) -> DecisionInfo[object]:
    node = linked.header(index)
    if node.kind != "decision":
        raise RequestError("decision inspection requires an owning Decision")
    choice_index = linked.selector_choices.get(index)
    choice = None if choice_index is None else linked.choices[choice_index]
    return DecisionInfo(
        DecisionHandle[object](linked, index),
        decision_key(linked, index),
        _scope_name(linked, node.scope),
        node.owner,
        choice is not None,
        () if choice is None else tuple(name for name, _ in choice.cases),
        _scope_type(linked, node.scope),
        choice is None and node.domain is not None and node.domain.ordered,
        node.domain is not None and node.domain.candidates is not None,
    )


def decisions(subject: Space | Model[S] | type[Space]) -> tuple[DecisionInfo[object], ...]:
    """Discover all owning decisions below this scope, including inactive cases, in the
    flow order (``_running.per_class.flow_below``). Read once per compiled model and scope,
    from its templates and scope records (``_whole``): no node is resolved."""

    compiled, scope = _context(subject)
    found = compiled._read.get(("decisions", scope))
    if found is None:
        found = compiled._read[("decisions", scope)] = tuple(
            _decision_info(compiled.linked, index) for index in flow_below(compiled, scope)
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
    """Every Decision over nodes below this scope, with its candidates (every scope
    addressed, no node resolved)."""
    compiled, scope = _context(subject)
    linked = compiled.linked
    included = scopes_below(linked, scope)
    return tuple(
        ChoiceInfo(
            choice.key,
            _scope_name(linked, choice.scope),
            tuple(
                CaseInfo(name, None, None)
                if index is None
                else CaseInfo(name, _scope_name(linked, index), _scope_type(linked, index))
                for name, index in choice.cases
            ),
            DecisionHandle[str](linked, choice.selector)
            if linked.header(choice.selector).kind == "decision"
            else None,
            None if choice.guard is None else ValueHandle[bool](linked, choice.guard),
        )
        for choice in sorted(linked.choices, key=lambda item: item.key)
        if choice.scope in included
    )


def members(subject: Space | Model[S] | type[Space]) -> tuple[NodeInfo, ...]:
    """Inspect every named member below this scope (including forwarding aliases; every
    scope below addressed, no node resolved)."""

    compiled, scope = _context(subject)
    linked = compiled.linked
    indices = {
        index
        for current in scopes_below(linked, scope)
        for index in linked.scopes[current].named_members.values()
    }
    return tuple(
        _node_info(linked, index)
        for index in sorted(indices, key=lambda item: linked.header(item).key)
    )


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
    from ._running.evaluate import evaluation
    from ._running.questions import validity as valid

    driver_only("validity")
    at = held(point)
    return ask(at.ds.model, lambda session: valid(evaluation(at.ds, session), at.scope))


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
    from ._running import questions

    driver_only("possible values")
    at = held(point)
    model = at.ds.model
    indices = [model.decision(at.scope, key) for key in keys]

    def answer(session: Session) -> tuple[Viable, ...]:
        return tuple(
            _detached_viable(model.linked, index, questions.viable(at.ds, index, session))
            for index in indices
        )

    return ask(model, answer)


def _detached(linked: Linked, index: NodeId, value: object) -> object:
    """A choice's value detached for a reader, by the choice's semantics."""
    from ._running.answers import copy_result as detached

    return cast(Available[object], detached(linked, index, Available(value))).value


def _detached_viable(linked: Linked, index: NodeId, found: Viable) -> Viable:
    """``found`` with each listed value detached (a domain's list is the facts layer's,
    shared by every design space of those facts)."""
    if found.values is None:
        return found
    values = tuple(
        replace(item, value=_detached(linked, index, item.value)) for item in found.values
    )
    return Viable(found.key, values, found.waits)


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
    (``waits``: what that presence waits on, by key)."""

    waits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Absent:
    """Unset, its scope absent or under a guard refused through an input: not in play."""


ChoiceState: TypeAlias = Set | Waiting | Dormant | Open | Absent


def states(point: Space) -> tuple[tuple[str, ChoiceState], ...]:
    """Every choice below this scope, by key, with its state, in the flow order (each
    after every choice its presence, its list or its Space's facts read: a selector before
    the choices its cases hold)."""
    from ._running.evaluate import evaluation
    from ._running.questions import state_of

    driver_only("choice states")
    at = held(point)
    model = at.ds.model
    linked = model.linked
    header = linked.header  # a choice's key, resolving nothing of an absent one

    def answer(session: Session) -> tuple[tuple[str, ChoiceState], ...]:
        ev = evaluation(at.ds, session)
        return tuple(
            (header(index).key, _detached_state(linked, index, state_of(ev, index)))
            for index in flow_below(model, at.scope)
        )

    return ask(model, answer)


def _detached_state(linked: Linked, index: NodeId, found: ChoiceState) -> ChoiceState:
    """A choice's state with its stored value detached (a stored value is the design
    space's own)."""
    if isinstance(found, (Set, Waiting, Dormant)):
        return replace(found, value=_detached(linked, index, found.value))
    return found


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

    from ._running.surface import explain as evidence

    driver_only("dependency inspection")
    at = held(point)
    return ask(at.ds.model, lambda session: evidence(at, reference, session))


def candidate(point: Space, decision: object, case: str) -> Space | None:
    """A candidate's Space object, selected or not; None for a None candidate."""

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
