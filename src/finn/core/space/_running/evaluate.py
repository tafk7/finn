# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every answer of one design space value, by node, memoized within it.

An ``Evaluation`` is one design space's answers: a node's answer is its frame's (one per
node kind), each node evaluated once and kept in the evaluation's memo, which dies with
the design space. Nothing is taken from another design space but by the session's node
cache, where it holds one: no worlds, no entries taken from a parent, no early cutoff.
Every node is read as the model's structure resolves it (``raw_nodes``): no collapsed node
is read anywhere.

**Reading a node** (``Evaluation.read``) names its reader: the node being evaluated, or,
for a ``when``'s condition read past its guard, that guard. A formal bound by reference
is read at the node it is bound to where the reader's scope is the formal's scope or one
placed below it, checked at each hop of a formal bound to a formal against the same
reader (``structure.Record.source``); read from anywhere else, it answers as a member of
its scope, its guard first. A ``self`` method is a plain call on a receiver reading
through the evaluation (``Reader``), the method's node its reader; what it read, in order,
is its answer's trace.

**Validity's pieces.** A piece whose guard is refused through an input counts once, at
the highest refused guard of its chain (each guard's own guard read in turn). Counted at
a guard above its own, the piece is that guard's owner's, keyed by the owner where the
owner is a node: a selector's cases make one piece, the selector's. A ``when`` refused
through an input below a guard that waits (a case of an open selector, or a parent whose
guard waits) is refused only once all above it holds: until then what it guards waits, a
Space's validity included (``Evaluation.refused_waiting``), and its piece is unknown,
waiting on what that guard waits on, counted once at the highest such ``when`` of the
chain and keyed as a refused guard's piece is. This holds in both places pieces are made,
a Space's validity and a choice's piece.

**Membership readings.** A choice's membership is read as if the choice existed, every
time: an evaluation of its facts layer with nothing stored (``FactsLayer.bare``), its
existence guards, where it has any, taken as holding (``Evaluation.membership``). By the
domain rule it reads facts only: a membership reading that reads a decision raises
``MembershipReadsChoice``, naming the choice and the decision, for a choice with no guard
as for any other. That is a definition the compile check should have refused, never a
case to fall back on.

**Depth.** Evaluation recurses on the Python stack up to ``DEPTH`` evaluations in flight
(today's guard, ``_world.DEPTH``). Past it, ``Deeper`` unwinds to the outermost read
(``Session.drive``), which evaluates the deep node first, then asks again what it
abandoned: authored functions are pure, so asking again answers the same. A failure is
raised, and kept for the rest of its drive (a later drive calls the function again).

**The session** holds the counters, and may hold a node cache (``cache.NodeCache``):
where it does, each answer carries the choices and facts its evaluation read
(``Ans.reads``), the key a cache keeps it by. Every node's evaluation asks the cache
first; a hit is taken as the answer, into the memo, its read set noted as read, exactly as
the evaluation would have left them. A miss is evaluated, and its answer kept, with its
read set, after. A membership reading asks under its guards taken as holding. A node
abandoned by the depth guard, or failing, keeps nothing.

Single-threaded: an evaluation, its session and the model's record take no lock.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from .. import _world as W
from .._collapse import guard_implies
from .._failure import (
    ACTIVE,
    Call,
    CleanupFailure,
    exception,
    from_exception,
    remember_exception,
)
from ..errors import EvaluationError, ValueUnavailableError
from ..graph import Located
from ..ir import Node, NodeId
from ..results import (
    Available,
    ConstraintAssessment,
    Inapplicable,
    NonValue,
    QueryResult,
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_constraints,
    assess_view,
    merged_findings,
)
from .structure import Record, record
from .value import DesignSpace, hashkey

if TYPE_CHECKING:
    from .._structure import Structure

Trace = tuple[tuple[NodeId, QueryResult[object]], ...]
Assessment = ViewAssessment[object] | ConstraintAssessment
TRUE: QueryResult[object] = Available(True)
INAPPLICABLE = Inapplicable()
_MISSING = object()
_OWN_BIT = ("param", "decision")
_CANCELLATIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)
DEPTH = W.DEPTH
"""How many evaluations may be in flight on the Python stack before the depth guard
unwinds (today's)."""


@dataclass(frozen=True, slots=True)
class Ans:
    """One node's answer in one design space: its result, what it read in order (its
    trace), a view's or a constraint's assessment, whether it refused on its own values,
    and what a method read only to know whether something exists."""

    result: QueryResult[object]
    trace: Trace = ()
    assessment: Assessment | None = None
    own_refusal: bool = False
    existence: frozenset[NodeId] = frozenset()
    # the choices and facts it read, through anything (kept only where a node cache is on)
    reads: frozenset[NodeId] = frozenset()


class MembershipReadsChoice(EvaluationError):
    """A membership reading read a decision: the domain rule (membership reads facts
    only) does not hold for ``choice``, whose membership read ``decision``."""

    def __init__(self, choice: str, decision: str) -> None:
        super().__init__(
            choice,
            "membership reads a choice",
            f"reading {choice}'s membership as if it existed read the choice {decision}",
        )
        self.choice = choice
        self.decision = decision


class Deeper(W.Deeper):
    """The depth guard: ``node`` was demanded in ``ev`` with ``DEPTH`` evaluations in
    flight. Everything it passes through is abandoned (nothing kept); the drive evaluates
    the node first, then again what it abandoned. Today's ``Deeper``, so that a method's
    reader passes it through as today's does."""

    def __init__(self, ev: Evaluation, node: NodeId) -> None:
        BaseException.__init__(self, node)
        self.ev = ev
        self.node = node


class NodeCache(Protocol):
    """A cache of answers across design spaces: ``lookup`` gives an answer kept for the
    same reads, or ``None``; ``keep`` offers one, with its ``reads``."""

    def lookup(self, ev: Evaluation, index: NodeId) -> Ans | None: ...

    def keep(self, ev: Evaluation, index: NodeId, found: Ans) -> None: ...


@dataclass
class Counters:
    """Work done: nodes evaluated, by kind; authored calls by kind; the most evaluations
    in flight at once; the depth guard's restarts. Where a node cache is on: its lookups
    and hits, the values it hashed, the members it kept a read set for, the read sets it
    kept, and the most read sets one member held at once (``cache.NodeCache``)."""

    evaluated: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    deepest: int = 0
    restarts: int = 0
    lookups: int = 0
    hits: int = 0
    hashed: int = 0
    members: int = 0
    sets: int = 0
    most_sets: int = 0

    def call(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1


@dataclass(eq=False)
class Session:
    """What evaluations share: their counters, the depth guard's drive (how many
    evaluations are in flight, what it abandoned, the failures met in it) and, where one
    is given, a node cache."""

    cache: NodeCache | None = None
    counters: Counters = field(default_factory=Counters)
    depth: int = 0
    driving: bool = False
    # what each evaluation in flight read, innermost last (where a cache is on)
    stack: list[set[NodeId]] = field(default_factory=list)
    pending: set[tuple[Evaluation, NodeId]] = field(default_factory=set)
    failures: dict[tuple[Evaluation, NodeId], Exception] = field(default_factory=dict)

    def drive(self, ev: Evaluation, index: NodeId) -> Ans:
        """``index``'s answer in ``ev``, read with nothing in flight: past the depth
        guard, the deep node is evaluated first, then what it abandoned, again. A failure
        is raised in what read it (asked again from the drive's stack), and finally here."""
        stack: list[tuple[Evaluation, NodeId]] = [(ev, index)]
        self.driving = True
        try:
            while True:
                at, node = stack[-1]
                try:
                    found = at._get(node)
                except Deeper as deeper:
                    self.counters.restarts += 1
                    self.pending.add(stack[-1])
                    stack.append((deeper.ev, deeper.node))
                    continue
                except Exception:
                    if len(stack) == 1:
                        raise
                    stack.pop()
                    self.pending.discard(stack[-1])
                    continue
                stack.pop()
                if not stack:
                    return found
                self.pending.discard(stack[-1])
        finally:
            self.driving = False
            self.pending.clear()
            self.failures.clear()


def evaluation(
    ds: DesignSpace, session: Session, assumed: frozenset[NodeId] | None = None
) -> Evaluation:
    """The evaluation of ``ds`` in ``session`` (one per design space value, session and
    guards taken as holding: ``None`` for the design space's own, a frozen set for a
    membership reading)."""
    found = ds.evaluations.get((session, assumed))
    if found is None:
        found = ds.evaluations[(session, assumed)] = Evaluation(ds, session, assumed)
    return cast(Evaluation, found)


@dataclass(frozen=True, slots=True)
class _Returned:
    value: object


@dataclass(frozen=True, slots=True)
class _Halted:
    results: tuple[NonValue, ...]


def invoke(
    call: Call,
    function: Callable[..., object],
    arguments: tuple[object, ...] = (),
    keywords: Mapping[str, object] | None = None,
) -> _Returned | _Halted:
    """Run one authored function as the active call: its value, or its halt on a
    non-value read (sticky, even where it caught the read); its failure raised; abandoned
    by the depth guard, the guard's ``Deeper`` raised."""
    token = ACTIVE.set(call)
    call.outer = sys.exception()
    try:
        try:
            value = function(*arguments, **(keywords or {}))
        except BaseException as cause:
            if call.abandoned is not None and not isinstance(cause, _CANCELLATIONS):
                raise call.abandoned from None
            failure = remember_exception(call, cause)
            if failure is not None:
                raise exception(failure) from cause
            if call.blocked:
                return _Halted(tuple(call.blocked.values()))
            raise EvaluationError(call.owner, call.role, str(cause)) from cause
        if call.abandoned is not None:
            raise call.abandoned
        if call.failure is not None:
            raise exception(call.failure)
        if call.blocked:
            return _Halted(tuple(call.blocked.values()))
        return _Returned(value)
    finally:
        ACTIVE.reset(token)


def answer_key(answer: QueryResult[object]) -> object:
    """An answer as a key: a value by ``hashkey``, a non-value by its variant and
    findings; ``None`` where it cannot be hashed."""
    if isinstance(answer, Available):
        found = hashkey(answer.value)
        return None if found is None else ("value", found)
    try:
        hash(answer.findings)
    except TypeError:
        return None
    return (type(answer), answer.findings)


_Tr = list[tuple[NodeId, QueryResult[object]]]


class Evaluation:
    """Every answer of one design space value (see the module docstring). ``assumed``:
    ``None`` for the design space's own evaluation; for a membership reading (an
    evaluation of the facts layer's design space with nothing stored, ``membership``), the
    guards taken as holding, none for a choice with no guard."""

    def __init__(
        self, ds: DesignSpace, session: Session, assumed: frozenset[NodeId] | None = None
    ) -> None:
        self.ds = ds
        self.session = session
        self.counters = session.counters
        self.model = ds.model
        self.linked: Structure = ds.layer.structure
        self.raw = self.linked.raw_nodes  # each node as resolved, nothing collapsed
        self.inferred = ds.model.evaluator.reads  # inferred reads, existence and case guards
        self.record: Record = record(ds.model)
        self.assumed = assumed
        self.memo: dict[NodeId, Ans] = {}
        self.active: dict[NodeId, None] = {}
        # a membership reading: the choice whose membership is being read
        self.asking: NodeId | None = None

    # -- reading

    def source(self, index: NodeId, reader: Node) -> NodeId:
        """What ``reader`` reads when it reads ``index`` (``structure.Record.source``)."""
        return self.record.source(self.linked, index, reader.scope)

    def note(self, reads: frozenset[NodeId] | set[NodeId]) -> None:
        """What the node being evaluated read (where a node cache is on)."""
        stack = self.session.stack
        if stack:
            stack[-1].update(reads)

    def get(self, index: NodeId) -> Ans:
        """``index``'s answer here: its memo's, else evaluated (driven from here where
        no drive is under way)."""
        found = self.memo.get(index)
        if found is not None:
            if self.session.cache is not None:
                self.note(found.reads)
            return found
        if not self.session.driving:
            return self.session.drive(self, index)
        return self._get(index)

    def _get(self, index: NodeId) -> Ans:
        found = self.memo.get(index)
        if found is not None:
            if self.session.cache is not None:
                self.note(found.reads)
            return found
        session = self.session
        kept = session.failures.get((self, index))
        if kept is not None:
            raise kept
        if index in self.active or (self, index) in session.pending:
            path = [self.raw[item].owner for item in self.active]
            owner = self.raw[index].owner
            raise EvaluationError(owner, "dependency cycle", " -> ".join((*path, owner)))
        cache = session.cache
        if cache is not None:
            found = cache.lookup(self, index)
            if found is not None:
                self.memo[index] = found
                self.note(found.reads)
                return found
        if session.depth >= DEPTH:
            raise Deeper(self, index)
        session.depth += 1
        if session.depth > self.counters.deepest:
            self.counters.deepest = session.depth
        self.active[index] = None
        if cache is not None:
            session.stack.append(set())
        read: set[NodeId] | None = None
        outer = sys.exception()
        try:
            found = self._evaluate(index)
        except Deeper:
            raise
        except Exception as cause:
            # Raised as today's driver raises it: the failure's ``NativeEvaluationError``.
            node = self.raw[index]
            failure = from_exception(cause, owner=node.owner, role=node.kind, outer=outer)
            error = exception(failure)
            if isinstance(error, Exception):
                session.failures[(self, index)] = error
            raise error from cause  # one failure: its cause is not a second one
        finally:
            session.depth -= 1
            del self.active[index]
            if cache is not None:
                read = session.stack.pop()
        if cache is not None and read is not None:
            found = replace(found, reads=frozenset(read))
            cache.keep(self, index, found)
            self.note(read)
        self.memo[index] = found
        return found

    def result(self, index: NodeId) -> QueryResult[object]:
        return self.get(index).result

    def read(self, trace: _Tr, index: NodeId, reader: Node) -> Ans:
        """``reader``'s read of ``index``: its answer, at its source where the formal is
        transparent to the reader, recorded in ``trace``."""
        index = self.source(index, reader)
        found = self.get(index)
        trace.append((index, found.result))
        return found

    def membership(self, index: NodeId) -> Evaluation:
        """Where the choice ``index``'s membership is read as if it existed: its facts
        layer with nothing stored, its existence guards (if any) taken as holding."""
        guards = frozenset(self.inferred.guards(index))
        found = evaluation(self.ds.layer.bare, self.session, guards)
        found.asking = index
        return found

    def _reads_choice(self, decision: NodeId) -> None:
        """A membership reading reading a decision: the domain rule broken."""
        if self.assumed is not None:
            choice = self.raw[self.asking].key if self.asking is not None else "?"
            raise MembershipReadsChoice(choice, self.raw[decision].key)

    def case(self, index: NodeId) -> tuple[NodeId, str] | None:
        return self.inferred.cases.get(index)

    # -- a node

    def _evaluate(self, index: NodeId) -> Ans:
        node = self.raw[index]
        if self.assumed and index in self.assumed:
            return Ans(TRUE)
        self.counters.evaluated += 1
        self.counters.by_kind[node.kind] = self.counters.by_kind.get(node.kind, 0) + 1
        trace: _Tr = []
        if node.guard is not None:
            guard = self.read(trace, node.guard, node)
            inactive = W._guard_result(node, guard.result)
            if isinstance(inactive, (Unresolved, Rejected)) and (
                node.kind == "guard" or self.case(index) is not None
            ):
                # Presence is the AND over the path: a condition that does not hold makes
                # its scope absent, whatever waits or is refused above it.
                if self._holds_here(node, trace) is False:
                    return Ans(Available(False), tuple(trace))
            if node.kind == "constraint" and isinstance(inactive, Rejected):
                inactive = INAPPLICABLE  # the guard's piece stands for it
            if inactive is not None and not (
                node.kind == "validity"
                and isinstance(inactive, Unresolved)
                and self.refused_waiting(node.guard, trace, node) is None
            ):
                return Ans(inactive, tuple(trace), W._guard_assessment(node, inactive))
        return _FRAMES[node.kind](self, node, trace, node)

    def _holds_here(self, node: Node, trace: _Tr) -> bool | None:
        """A guard's own condition, read while a guard above it waits or is refused: false
        where its case's selector stores another case or its ``when`` reads false. The
        guard is the reader of its condition, read past the condition's own guard where
        the guard implies it."""
        held = self.case(node.index)
        if held is not None:
            selector, case = held
            self._reads_choice(selector)
            if self.session.cache is not None:
                self.note({selector})
            value = self.ds.stored.get(selector, _MISSING)
            return None if value is _MISSING or value == case else False
        if node.output is None:
            return None
        answer = self._condition(node, trace)
        return False if isinstance(answer, Available) and answer.value is False else None

    def _condition(self, node: Node, trace: _Tr) -> QueryResult[object]:
        """A ``when`` guard's condition (``node``: the guard, its reader, under a guard of
        its own), read past the condition's own guard where the guard implies it."""
        assert node.output is not None and node.guard is not None
        output = self.source(node.output, node)
        condition = self.raw[output]
        if (
            condition.guard is None
            or condition.kind in _OWN_BIT
            or not guard_implies(self.raw, condition.guard, node.guard)
        ):
            return self.read(trace, output, node).result
        # Past its own guard: its frame alone, the guard its reader.
        found = _FRAMES[condition.kind](self, condition, [], node)
        trace.extend(found.trace)
        return found.result

    def refused_waiting(self, guard: NodeId, trace: _Tr, reader: Node) -> NodeId | None:
        """The highest ``when`` guard in the chain of ``guard`` (which waits) refused
        through an input below a guard that waits, or ``None``: climbing from ``guard``
        while each guard waits, a ``when`` guard whose own condition, read past its guard,
        is refused. What it guards waits, on what the guard above it waits on (refused
        takes effect once all above it holds); where nothing above waits, the guard
        answers refused, and the refused guard's piece counts (``_guard_refused``)."""
        found: NodeId | None = None
        index = guard
        while (outer := self.raw[index].guard) is not None:
            node = self.raw[index]
            if (
                node.kind == "guard"
                and node.output is not None
                and self.case(index) is None
                and isinstance(self._condition(node, trace), Rejected)
            ):
                found = index
            index = outer
            if not isinstance(self.read(trace, index, reader).result, Unresolved):
                break
        return found

    # -- authored functions

    def authored(self, kind: str, compute: Callable[[], W.Called]) -> W.Called:
        """An authored function's outcome, counted by kind."""
        self.counters.call(kind)
        return compute()

    def domain_outcome(
        self, key: tuple[NodeId, str], kind: str, inputs: Trace, compute: Callable[[], W.Called]
    ) -> QueryResult[object]:
        """A domain's outcome, kept on the facts layer by its inputs' answers (a domain
        reads facts only, so every design space of the layer shares it)."""
        outcomes = self.ds.layer.outcomes
        local = tuple(answer_key(answer) for _, answer in inputs)
        memo_key = None if any(part is None for part in local) else (key, local)
        if memo_key is not None:
            held = outcomes.get(memo_key)
            if held is not None:
                return held.result
        called = self.authored(kind, compute)
        if memo_key is not None:
            outcomes[memo_key] = called
        return called.result

    def call_explicit(self, node: Node, answers: list[QueryResult[object]]) -> W.Called:
        arguments = {
            argument.name: W._clone(
                self.raw[argument.node],
                cast(Available[object], answer).value,
                owner=node.owner,
                role="dependency snapshot",
            )
            for argument, answer in zip(node.arguments, answers)
        }
        outcome = invoke(
            Call(node.owner, node.kind), cast(Callable[..., object], node.function), (), arguments
        )
        return W._finished(node, cast(_Returned, outcome).value)

    def call_method(self, node: Node, reader: Node) -> tuple[W.Called, Trace]:
        call = Call(node.owner, node.kind)
        receiver = Reader(self, node, call, reader)
        outcome = invoke(
            call, cast(Callable[..., object], node.function), (W.receiver(receiver, node.scope),)
        )
        existence = frozenset(receiver.existence - receiver.valued)
        if isinstance(outcome, _Halted):
            blocked = cast(NonValue, W._blocked(outcome.results))
            called = W.Called(blocked, W._guard_assessment(node, blocked), existence=existence)
        else:
            done = W._finished(node, outcome.value)
            called = W.Called(done.result, done.assessment, done.own_refusal, (), existence)
        return called, tuple(receiver.trace)

    # -- domains (each read by the choice it belongs to)

    def domain_inputs(
        self, node: Node, trace: _Tr
    ) -> tuple[tuple[QueryResult[object], ...], NonValue | None]:
        arguments = W.membership_arguments(node)
        answers = tuple(self.read(trace, argument.node, node).result for argument in arguments)
        absent = [
            self.raw[argument.node].owner
            for argument, answer in zip(arguments, answers)
            if isinstance(answer, Inapplicable)
        ]
        if absent:
            return answers, W._inputs_absent(node, absent)
        return answers, W._blocked(
            [answer for answer in answers if not isinstance(answer, Available)]
        )

    def domain_values(
        self, node: Node, answers: tuple[QueryResult[object], ...]
    ) -> dict[str, object]:
        return {
            argument.name: W._clone(
                self.raw[argument.node],
                cast(Available[object], answer).value,
                owner=node.owner,
                role="dependency snapshot",
            )
            for argument, answer in zip(W.membership_arguments(node), answers)
        }

    def _inputs(self, node: Node, answers: tuple[QueryResult[object], ...]) -> Trace:
        return tuple(
            (argument.node, answer)
            for argument, answer in zip(W.membership_arguments(node), answers)
        )

    def type_check(self, node: Node, value: Available[object], trace: _Tr) -> QueryResult[object]:
        """Whether ``node``'s domain holds ``value`` here."""
        domain, semantics = node.domain, node.semantics
        assert domain is not None and semantics is not None
        answers, blocked = self.domain_inputs(node, trace)
        if blocked is not None:
            return blocked

        def membership() -> W.Called:
            candidate = W._clone(node, value.value, owner=node.owner, role="domain candidate")
            outcome = invoke(
                Call(node.owner, "domain membership"),
                domain.membership,
                (candidate, self.domain_values(node, answers)),
                {"semantics": semantics, "owner": node.owner},
            )
            return W.Called(cast(QueryResult[object], cast(_Returned, outcome).value))

        inputs = ((node.index, value), *self._inputs(node, answers))
        return self.domain_outcome(
            (node.index, "membership"), "domain membership", inputs, membership
        )

    def _enumerate(
        self, node: Node, answers: tuple[QueryResult[object], ...]
    ) -> QueryResult[object]:
        domain, semantics = node.domain, node.semantics
        assert domain is not None and semantics is not None

        def enumerate_() -> W.Called:
            outcome = invoke(
                Call(node.owner, "domain enumeration"),
                domain.enumerate,
                (self.domain_values(node, answers),),
                {"semantics": semantics, "owner": node.owner},
            )
            return W.Called(cast(QueryResult[object], cast(_Returned, outcome).value))

        return self.domain_outcome(
            (node.index, "enumeration"),
            "domain enumeration",
            self._inputs(node, answers),
            enumerate_,
        )

    def _list(self, node: Node, answers: tuple[QueryResult[object], ...]) -> W.Called:
        domain, semantics = node.domain, node.semantics
        assert domain is not None and semantics is not None
        listed = self._enumerate(node, answers)
        if isinstance(listed, Available) and domain._finite_values is None:
            kept = invoke(
                Call(node.owner, "domain membership"),
                W._listed_members,
                (domain, listed.value, self.domain_values(node, answers)),
                {"semantics": semantics, "owner": node.owner},
            )
            listed = Available(cast(_Returned, kept).value)
        return W.Called(listed)

    def listed(self, node: Node, trace: _Tr) -> QueryResult[object]:
        """A choice's listed values here (``Available(None)``: membership only)."""
        domain = node.domain
        assert domain is not None
        answers, blocked = self.domain_inputs(node, trace)
        if blocked is not None:
            return blocked
        if domain.candidates is None:
            return Available(None)
        return self.domain_outcome(
            (node.index, "listing"),
            "domain listing",
            self._inputs(node, answers),
            lambda: self._list(node, answers),
        )

    def lists_a_value(self, node: Node, trace: _Tr) -> QueryResult[object]:
        """Whether a listed domain lists a value here: from its list where it was made,
        else membership asked up to the first value it holds."""
        domain, semantics = node.domain, node.semantics
        assert domain is not None and domain.candidates is not None and semantics is not None
        answers, blocked = self.domain_inputs(node, trace)
        if blocked is not None:
            return blocked
        inputs = self._inputs(node, answers)
        local = tuple(answer_key(answer) for _, answer in inputs)
        made = self.ds.layer.outcomes.get(((node.index, "listing"), local))
        if made is not None:
            found = made.result
            return Available(found.value != ()) if isinstance(found, Available) else found

        def lists() -> W.Called:
            enumerated = self._enumerate(node, answers)
            if not isinstance(enumerated, Available):
                return W.Called(enumerated)
            values = cast(tuple[object, ...], enumerated.value)
            if domain._finite_values is not None or not values:
                return W.Called(Available(bool(values)))
            kept = invoke(
                Call(node.owner, "domain membership"),
                W._first_member,
                (domain, values, self.domain_values(node, answers)),
                {"semantics": semantics, "owner": node.owner},
            )
            return W.Called(Available(cast(_Returned, kept).value))

        return self.domain_outcome((node.index, "lists a value"), "domain listing", inputs, lists)


# -- the receiver a method reads through ------------------------------------------------


class _World:
    """What ``MethodReader`` reads of the world it runs in: the model's evaluator (its
    structure, inferred reads and resolution), nothing of a world."""

    __slots__ = ("evaluator",)

    def __init__(self, evaluator: W.Evaluator) -> None:
        self.evaluator = evaluator


class Reader(W.MethodReader):
    """A method's reads, on an evaluation: today's reader (resolution, the read check,
    ``present``, candidates), each read answered by the evaluation from the raw nodes and
    recorded in order. ``reader``: the node whose scope decides which formals bound by
    reference it reads at their source (the method's node, or the guard a ``when``'s
    condition is read for)."""

    __slots__ = ("ev", "reader")

    def __init__(self, ev: Evaluation, node: Node, call: Call, reader: Node) -> None:
        super().__init__(cast(W.World, _World(ev.model.evaluator)), node, call)
        self.ev = ev
        self.reader = reader

    def _source(self, index: NodeId) -> NodeId:
        return self.ev.source(index, self.reader)

    def _take(self, index: NodeId) -> Any:
        try:
            found = self.ev.get(index)
        except Deeper as deeper:
            self.call.abandoned = deeper
            raise
        except EvaluationError as error:
            remember_exception(self.call, error)
            raise
        self.trace.append((index, found.result))
        return found

    def _read_value(self, point: Any, reference: object) -> object:
        self._enter()
        index = self.ev.model.resolve(point._scope, reference)
        self._check(index)
        index = self._source(index)
        self.valued.add(index)
        return self._value(index)

    def _answer(self, index: NodeId) -> QueryResult[object]:
        self._check(index)
        index = self._source(index)
        self.existence.add(index)
        return cast(QueryResult[object], self._take(index).result)

    def _value(self, index: NodeId) -> object:
        index = self._source(index)
        answer = self._take(index).result
        node = self.ev.raw[index]
        if isinstance(answer, Available):
            return W.public_value(node, answer)
        self.call.blocked[index] = answer
        if self.call.failure is not None:
            self.call.failure.add(CleanupFailure(node.owner, "cleanup read", result=answer))
        raise ValueUnavailableError(answer, context=node.owner)

    def _source_applies(self, index: NodeId) -> bool | NodeId:
        index = self._source(index)
        node = self.ev.raw[index]
        if node.guard is not None:
            holds = self._holds(node.guard)
            if holds is not True:
                return holds
        if node.kind == "param":
            return isinstance(self._answer(index), Available)
        if node.kind in ("alias", "guard", "view"):
            return self._source_applies(cast(NodeId, node.output))
        if node.kind == "select":
            selector = cast(NodeId, node.selector)
            case = self._answer(selector)
            if not isinstance(case, Available):
                return False if isinstance(case, Inapplicable) else selector
            target = cast(Mapping[str, NodeId], node.selection_index).get(cast(str, case.value))
            return False if target is None else self._source_applies(target)
        if node.kind == "present":
            answers = [self._source_applies(target) for _, target in node.alternatives]
            if any(answer is True for answer in answers):
                return True
            return next((answer for answer in answers if answer is not False), False)
        return True


# -- frames: each kind of node (``reader``: whose scope its reads are made from) ----------


def _param(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    if ev.session.cache is not None:
        ev.note({node.index})
    if node.index in ev.ds.facts:
        return Ans(Available(ev.ds.facts[node.index]), tuple(trace))
    return Ans(W._input_missing(node), tuple(trace))


def _const(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    return Ans(Available(node.value), tuple(trace))


def _decision(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    ev._reads_choice(node.index)
    if ev.session.cache is not None:
        ev.note({node.index})
    if node.index not in ev.ds.stored:
        return Ans(W._unassigned(node), tuple(trace))
    value: Available[object] = Available(ev.ds.stored[node.index])
    if node.domain is None:
        return Ans(value, tuple(trace))
    check = ev.type_check(node, value, trace)
    if isinstance(check, Available) and check.value is True:
        return Ans(value, tuple(trace))
    return Ans(check, tuple(trace))


def _forward(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    if node.output is None:
        raise EvaluationError(node.owner, node.kind, "missing output reference")
    return Ans(ev.read(trace, node.output, reader).result, tuple(trace))


def _select(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    assert node.selector is not None and node.selection_index is not None
    selector = ev.read(trace, node.selector, reader).result
    if not isinstance(selector, Available):
        return Ans(selector, tuple(trace))
    target = node.selection_index.get(cast(str, selector.value))
    if target is None:
        return Ans(INAPPLICABLE, tuple(trace))
    return Ans(ev.read(trace, target, reader).result, tuple(trace))


def _waiting(ev: Evaluation, index: NodeId) -> dict[str, QueryResult[object]]:
    """The pieces still unresolved in the refused Space whose validity is ``index``."""
    keys, found = ev.linked.keys, {}
    pending = [index]
    while pending:
        assessment = ev.get(pending.pop()).assessment
        if not isinstance(assessment, ConstraintAssessment):
            continue
        for key, answer in assessment.results.items():
            if isinstance(answer, Unresolved):
                found[key] = cast(QueryResult[object], answer)
            elif isinstance(answer, Rejected) and ev.raw[keys[key]].kind == "validity":
                pending.append(keys[key])
    return found


def _view(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    assert node.output is not None and len(node.constraints) == 1
    output = ev.read(trace, node.output, reader).result
    validity = ev.read(trace, node.constraints[0], reader).result
    waiting = _waiting(ev, node.constraints[0]) if isinstance(validity, Rejected) else None
    view = assess_view(
        output,
        owner=node.key,
        validity=cast(QueryResult[bool], validity),
        validity_owner=ev.linked.header(node.constraints[0]).key,
        waiting=waiting,
    )
    return Ans(view.accepted_result, tuple(trace), view)


Piece = tuple[str, QueryResult[bool]]


def _validity(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    """Kleene's AND of a Space's pieces, a placed Space's read from its guard first; a
    piece whose guard is refused through an input counted once, at the highest refused
    guard of its chain (``_guard_refused``), and one refused below a guard that waits,
    unknown, at the highest such ``when`` (``_guard_waiting``)."""
    linked = ev.linked
    answers: dict[str, QueryResult[bool]] = {}
    guards: set[NodeId] = set()
    for index in node.constraints:
        header = linked.header(index)
        placed: Ans | None = None
        if header.kind == "validity" and header.guard is not None:
            placed = ev.read(trace, header.guard, reader)
            held = placed.result
            if isinstance(held, Inapplicable):
                answers[header.key] = held
                continue
            if isinstance(held, Available) and held.value is False:
                answers[header.key] = INAPPLICABLE
                continue
        piece = ev.raw[index]
        if piece.kind == "decision":
            found = _choice_piece(ev, piece, trace, reader, guards)
            if found is not None:
                answers[found[0]] = found[1]
            continue
        answer = cast(QueryResult[bool], ev.read(trace, index, reader).result)
        if isinstance(answer, Unresolved) and piece.guard is not None:
            guard = placed if placed is not None else ev.read(trace, piece.guard, reader)
            if isinstance(guard.result, Unresolved):
                top = ev.refused_waiting(piece.guard, trace, reader)
                if top is not None:
                    counted = _guard_waiting(ev, piece, top, trace, reader, guards)
                    if counted is not None:
                        answers[counted[0]] = counted[1]
                    continue
        if isinstance(answer, (Inapplicable, Rejected)) and piece.guard is not None:
            guard = placed if placed is not None else ev.read(trace, piece.guard, reader)
            if isinstance(guard.result, Rejected):
                owner = linked.scopes[piece.scope].name if piece.kind == "validity" else piece.owner
                counted = _guard_refused(ev, piece, owner, trace, reader, guards)
                if counted is not None:
                    answers[counted[0]] = counted[1]
                continue
        answers[piece.key] = answer
    assessment = assess_constraints(answers)
    return Ans(cast(QueryResult[object], assessment.result), tuple(trace), assessment)


def _guard_refused(
    ev: Evaluation, piece: Node, owner: str, trace: _Tr, reader: Node, guards: set[NodeId]
) -> Piece | None:
    """The guard piece of ``piece``, whose guard is refused through an input: counted
    once, at the highest refused guard of its chain, or ``None`` where that guard is
    counted already. Counted at a guard above its own, it is that guard's owner's, keyed
    by the owner where the owner is a node (a selector: its cases' pieces are its one
    piece)."""
    top = cast(NodeId, piece.guard)
    found = ev.read(trace, top, reader).result
    while (outer := ev.raw[top].guard) is not None:
        above = ev.read(trace, outer, reader).result
        if not isinstance(above, Rejected):
            break
        top, found = outer, above
    if top in guards:
        return None
    guards.add(top)
    key = piece.key
    if top != piece.guard:
        owner = ev.raw[top].owner
        if owner in ev.linked.keys:
            key = owner
    return key, W._guard_piece(owner, cast(Rejected, found))


def _guard_waiting(
    ev: Evaluation, piece: Node, top: NodeId, trace: _Tr, reader: Node, guards: set[NodeId]
) -> Piece | None:
    """The piece of ``piece``, below ``top``, a ``when`` refused through an input below a
    guard that waits (``Evaluation.refused_waiting``): unknown, waiting on what the guard
    above ``top`` waits on, counted once at ``top`` and keyed as ``_guard_refused`` keys a
    refused guard's piece; ``None`` where ``top`` is counted already."""
    if top in guards:
        return None
    guards.add(top)
    key = piece.key
    if top != piece.guard:
        owner = ev.raw[top].owner
        if owner in ev.linked.keys:
            key = owner
    above = cast(NodeId, ev.raw[top].guard)
    return key, cast(QueryResult[bool], ev.read(trace, above, reader).result)


def _choice_piece(
    ev: Evaluation, piece: Node, trace: _Tr, reader: Node, guards: set[NodeId]
) -> Piece | None:
    """A choice's piece by key, or ``None`` where it is no piece: a guard refused through
    an input (``_guard_refused``), or refused below a guard that waits
    (``_guard_waiting``); else read from its guard and its listed values (never from its
    stored value)."""
    waiting: Unresolved | None = None
    if piece.guard is not None:
        answer = ev.read(trace, piece.guard, reader).result
        if isinstance(answer, Rejected):
            return _guard_refused(ev, piece, piece.owner, trace, reader, guards)
        if isinstance(answer, Unresolved):
            top = ev.refused_waiting(piece.guard, trace, reader)
            if top is not None:
                return _guard_waiting(ev, piece, top, trace, reader, guards)
            waiting = answer
        elif W._guard_result(piece, answer) is not None:
            return None
    if piece.domain is None or piece.domain.candidates is None:
        return None
    if waiting is not None:
        listed = ev.membership(piece.index).lists_a_value(piece, [])
        if isinstance(listed, Available):
            return None if listed.value else (piece.key, waiting)
        if isinstance(listed, Unresolved):
            return piece.key, Unresolved(merged_findings((waiting, listed)))
        return piece.key, waiting
    listed = ev.lists_a_value(piece, trace)
    if isinstance(listed, Available):
        return None if listed.value else (piece.key, W._empty_domain(piece))
    return piece.key, cast(QueryResult[bool], listed)


def _graph(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    if node.kind == "present":
        answers = [ev.read(trace, target, reader).result for _, target in node.alternatives]
        several = sum(1 for answer in answers if isinstance(answer, Available)) > 1
        return Ans(W._present(node, answers), tuple(trace), own_refusal=several)
    located: list[Located[object]] = []
    failures: list[QueryResult[object]] = []
    for (name, target), member in zip(node.alternatives, cast(tuple[str, ...], node.value)):
        answer = ev.read(trace, target, reader).result
        if isinstance(answer, Inapplicable):
            continue
        if isinstance(answer, Available):
            located.append(Located(name or None, member, answer.value))
        else:
            failures.append(answer)
    blocked = W._blocked(failures)
    return Ans(blocked if blocked is not None else Available(tuple(located)), tuple(trace))


def _callback(ev: Evaluation, node: Node, trace: _Tr, reader: Node) -> Ans:
    if node.function is None:
        raise EvaluationError(node.owner, node.kind, "missing callback")
    if node.call_style == "self":
        ev.counters.call(node.kind)
        called, read = ev.call_method(node, reader)
        return Ans(
            called.result, (*trace, *read), called.assessment, called.own_refusal, called.existence
        )
    answers = [ev.read(trace, argument.node, reader).result for argument in node.arguments]
    blocked = W._blocked([answer for answer in answers if not isinstance(answer, Available)])
    if blocked is not None:
        return Ans(blocked, tuple(trace), W._guard_assessment(node, blocked))
    called = ev.authored(node.kind, lambda: ev.call_explicit(node, answers))
    return Ans(called.result, tuple(trace), called.assessment, called.own_refusal)


_FRAMES: Mapping[str, Callable[[Evaluation, Node, _Tr, Node], Ans]] = {
    "param": _param,
    "const": _const,
    "decision": _decision,
    "alias": _forward,
    "guard": _forward,
    "select": _select,
    "view": _view,
    "validity": _validity,
    "present": _graph,
    "members": _graph,
    "derived": _callback,
    "constraint": _callback,
}

__all__ = [
    "DEPTH",
    "INAPPLICABLE",
    "TRUE",
    "Ans",
    "Counters",
    "Deeper",
    "Evaluation",
    "MembershipReadsChoice",
    "NodeCache",
    "Reader",
    "Session",
    "answer_key",
    "evaluation",
    "invoke",
]
