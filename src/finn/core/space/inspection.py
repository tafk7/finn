# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public metadata, validity, possible values, choice states and demanded query evidence.

Metadata inspection does not invoke evaluators or candidate providers. The questions of a
design space are the running type's (``_running.questions``, ``_running.surface``), each
asked in the model's one session under its lock (``occurrence.ask``): validity reads the
pieces of a Space's validity; ``viable`` tries each listed value of a choice as a change
would set it, publishing nothing; ``states`` reads each choice's state; explain evaluates
one query and follows the reads its answers recorded. ``states`` and ``viable`` hand out
the choices' values themselves: a choice's value is immutable, in its canonical form
(``semantics.canonical``), and every answer type is a frozen dataclass of tuples.

**The lists**. A design space's ``decisions`` and ``states`` (and so its flow
order) leave out each choice whose existence the facts alone rule out, unless it is stored
(``_running.questions.pruned``). A choice asked by name is answered all the same:
its state, its possible values (``viable``), a change setting it (accepted, dormant). A
compiled model's or a Space class's ``decisions`` list every decision.
"""

# The running type imports this module's types: it is imported inside the bodies here.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast, overload

from ._configuration import Space
from ._failure import driver_only
from ._running.per_class import flow_below, scopes_below
from .compiler import Model, compile_model
from .declarations import (
    Constraint,
    Decision,
    ValueRef,
    View,
)
from .edits import BrokenRule
from .errors import RequestError
from .ir import NodeKind
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
    from ._scopes import Address, ScopeRecord

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
    """Detached result and actual outgoing demands for one visited computation."""

    declaration: NodeInfo
    result: QueryResult[object]
    dependencies: tuple[ValueHandle[object], ...]
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
    (including every candidate). Reused classes therefore count at every
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
    scope records count from its classes' compiled forms, each counted at every placement:
    once per shape of placement, times the placements of that shape
    (``Scopes.statistics``; no member evaluated)."""

    if not isinstance(subject, (Model, Space, type)):
        raise RequestError("statistics requires a compiled model")
    return ModelStatistics(*_context(subject)[0].scopes.statistics())


def _context(subject: Space | Model[S] | type[Space]) -> tuple[Model[Space], ScopeRecord]:
    if isinstance(subject, Model):
        model = cast(Model[Space], subject)
        return model, model.scopes.root
    if isinstance(subject, type) and issubclass(subject, Space):
        model = cast(Model[Space], compile_model(subject))
        return model, model.scopes.root
    if isinstance(subject, Space):
        return model_of(subject), subject._scope
    raise RequestError("inspection requires a configuration, a compiled model or a Space class")


def model(subject: Space | Model[S] | type[Space]) -> Model[Space]:
    """The compiled model of a configuration, or of a Space class compiled alone."""
    return _context(subject)[0]


def _node_info(model: Model[Space], index: Address) -> NodeInfo:
    node = index
    reference = (
        DecisionHandle[object](model, index)
        if node.kind == "decision"
        else ValueHandle[object](model, index)
    )
    return NodeInfo(
        reference,
        node.key,
        node.kind,
        node.scope.name,
        node.owner,
        node.source_owner is not None,
        None if node.guard is None else ValueHandle[bool](model, node.guard),
        node.origin,
    )


def _decision_info(model: Model[Space], index: Address) -> DecisionInfo[object]:
    node = index
    if node.kind != "decision":
        raise RequestError("decision inspection requires an owning Decision")
    choice = model.scopes.selector_choice(index)
    return DecisionInfo(
        DecisionHandle[object](model, index),
        decision_key(index),
        node.scope.name,
        node.owner,
        choice is not None,
        () if choice is None else tuple(name for name, _ in choice.cases),
        node.scope.space_type,
        choice is None and node.domain is not None and node.domain.ordered,
        node.domain is not None and node.domain.candidates is not None,
    )


def decisions(subject: Space | Model[S] | type[Space]) -> tuple[DecisionInfo[object], ...]:
    """The owning decisions below this scope, in the flow order
    (``_running.per_class.flow_below``). Of a compiled model or a Space class, every one,
    inactive cases included: read once per compiled model and scope, from its compiled
    forms and scope records (``_whole``), no member evaluated. Of a design space, those in
    its lists: every one but each choice whose existence the facts alone rule out,
    unless it is stored (``_running.questions.pruned``). A decision asked by name is
    answered whether it is listed or not."""

    compiled, scope = _context(subject)
    found = compiled._read.get(("decisions", scope))
    if found is None:
        found = compiled._read[("decisions", scope)] = tuple(
            _decision_info(compiled, index) for index in flow_below(compiled, scope)
        )
    every = cast(tuple[DecisionInfo[object], ...], found)
    if not isinstance(subject, Space):
        return every
    from ._running.evaluate import evaluation
    from ._running.questions import listed

    driver_only("decisions")
    at = held(subject)

    def answer(session: Session) -> tuple[DecisionInfo[object], ...]:
        kept = {index.key for index in listed(evaluation(at.ds, session), at.scope)}
        if len(kept) == len(every):
            return every
        return tuple(item for item in every if item.key in kept)

    return ask(compiled, answer)


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
    return _decision_info(compiled, compiled.decision(scope, reference)).reference


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
    return ValueHandle(compiled, compiled.resolve(scope, reference))


def choices(subject: Space | Model[S] | type[Space]) -> tuple[ChoiceInfo, ...]:
    """Every Decision over nodes below this scope, with its candidates (every scope
    made, no member evaluated)."""
    compiled, scope = _context(subject)
    included = scopes_below(compiled.scopes, scope)
    return tuple(
        ChoiceInfo(
            choice.key,
            choice.scope.name,
            tuple(
                CaseInfo(name, None, None)
                if child is None
                else CaseInfo(name, child.name, child.space_type)
                for name, child in choice.cases
            ),
            DecisionHandle[str](compiled, choice.selector)
            if choice.selector.kind == "decision"
            else None,
            None if choice.guard is None else ValueHandle[bool](compiled, choice.guard),
        )
        for choice in sorted(compiled.scopes.choices(), key=lambda item: item.key)
        if choice.scope in included
    )


def members(subject: Space | Model[S] | type[Space]) -> tuple[NodeInfo, ...]:
    """Inspect every named member below this scope (including forwarding aliases; every
    scope below made, no member evaluated)."""

    compiled, scope = _context(subject)
    indices = {
        index
        for current in scopes_below(compiled.scopes, scope)
        for index in current.named_members.values()
    }
    return tuple(
        _node_info(compiled, index) for index in sorted(indices, key=lambda item: item.key)
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


def presence(point: Space) -> tuple[tuple[str, QueryResult[bool]], ...]:
    """Every real Space's presence below this point, including rule-free children.

    Available(True/False) means present/absent; Inapplicable means an ancestor is
    absent. Unresolved or Rejected presence is not complete. This asks only the
    ordinary guards, never member values or validity, and preserves dormant cases.
    """
    from ._running.evaluate import evaluation

    driver_only("scope presence")
    at = held(point)

    def answer(session: Session) -> tuple[tuple[str, QueryResult[bool]], ...]:
        ev = evaluation(at.ds, session)
        return tuple(
            (
                scope.name,
                Available(True)
                if scope.guard is None
                else cast(QueryResult[bool], ev.result(scope.guard)),
            )
            for scope in scopes_below(at.ds.model.scopes, at.scope)
        )

    return ask(at.ds.model, answer)


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
    or a list that waits (``waits``, by key) or is refused through an input of the
    domain."""

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
    choice existed; a choice the design space's lists leave out is answered all the same),
    each tried alone as a change would set it, publishing nothing:
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
        return tuple(questions.viable(at.ds, index, session) for index in indices)

    return ask(model, answer)


# -- every choice's state --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Set:
    """Stored and in play: it reads as its value."""

    value: object


@dataclass(frozen=True, slots=True)
class Waiting:
    """Stored, never read as a value: its presence is unknown. ``waits``: what it waits on
    (open choices), by key."""

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
    """Every choice below this scope in the design space's lists, by key, with its state,
    in the flow order (each after every choice its presence, its list or its Space's facts
    read: a selector before the choices its cases hold). A choice whose existence the facts
    alone rule out is in no list, unless it is stored
    (``_running.questions.pruned``)."""
    from ._running.evaluate import evaluation
    from ._running.questions import listed, state_of

    driver_only("choice states")
    at = held(point)
    model = at.ds.model

    def answer(session: Session) -> tuple[tuple[str, ChoiceState], ...]:
        ev = evaluation(at.ds, session)
        return tuple((index.key, state_of(ev, index)) for index in listed(ev, at.scope))

    return ask(model, answer)


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
    "presence",
    "states",
    "statistics",
    "validity",
    "value_handle",
    "viable",
]
