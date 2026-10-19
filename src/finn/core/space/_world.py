# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The evaluation core: worlds, their entries, early cutoff, and each kind of node.

A **world** is a reading of every choice and fact of one compiled model: the facts
bound (root formals) and the stored choices, dormant ones included. Nothing else is
read: a choice nobody stored is open and reads unassigned, and a stored choice whose
scope is absent reads inapplicable (dormancy is a reading, not a state). An **entry**
is one node's evaluation in one world: its answer, the choices and facts it read
(``ReadSet``, through everything it read), and what it read, in order, with the
answers it got (its trace).

A world made from another (``World.successor``) names the choices and facts it reads
otherwise (its ``delta``). It takes, lazily, each entry of its parent that read none
of them. An entry that did read one is not thrown away: the node is evaluated again
from its trace (**early cutoff**): where each dependency answers the same here, by its
value semantics, the parent's answer stands, the same object, so what read it is kept
in turn.

Authored functions are pure, so their outcomes are kept across every world of the
model (``_memo.Memo``): an explicit function by its arguments' answers, a ``self``
method by its traces. A ``self`` method is a plain call; its receiver reads through a
``MethodReader`` on the world it runs in, which records each read.

Nothing outside this module's tests evaluates through it yet: the configuration's
reads, changes and inspection still run on ``_runtime``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, TypeAlias, cast

from finn.core.reporting.findings import Finding, FindingKind

from ._memo import BitTable, Counters, Memo, ReadSet, Trace
from ._reads import ModelReads, infer
from .errors import EvaluationError, ValueUnavailableError
from .graph import Located
from .ir import LinkedModel, Node, NodeId, NodeKind
from .results import (
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
    constraint_result,
    merged_findings,
    owned_result,
    reject,
)
from .semantics import equal
from .semantics import snapshot as snapshot_value

if TYPE_CHECKING:
    from ._configuration import Space
    from .compiler import Model

Assessment: TypeAlias = ViewAssessment[object] | ConstraintAssessment

_INAPPLICABLE = Inapplicable()

# The nodes that read their own bit (a fact, a choice): they are never cut early.
_OWN_BIT: frozenset[NodeKind] = frozenset(("param", "decision"))


@dataclass(frozen=True, slots=True, eq=False)
class Entry:
    """One node's evaluation in one world.

    ``reads``: every choice and fact it read, directly or through another entry.
    ``trace``: what it read, in order, with the answers it got. ``assessment``: a
    view's or a constraint's (and a validity node's). ``own_refusal``: it refused on
    its own values (a constraint false, a derived value or a view's body refusing),
    not through a refused input. ``via``: a method's reads of a forwarding alias that
    went to its source, as (alias, source)."""

    result: QueryResult[object]
    reads: ReadSet
    trace: Trace
    assessment: Assessment | None = None
    own_refusal: bool = False
    via: tuple[tuple[NodeId, NodeId], ...] = ()


@dataclass(frozen=True, slots=True)
class Called:
    """What an authored function's call came to, as a memo keeps it."""

    result: QueryResult[object]
    assessment: Assessment | None = None
    own_refusal: bool = False
    via: tuple[tuple[NodeId, NodeId], ...] = ()


class Evaluator:
    """What every world of one model shares: its structure and its methods' inferred
    reads, the bit table, the memo of authored functions, the work counters, and the
    model's lock (worlds share their parents' entries and the memo, so evaluation of
    one model is serialized)."""

    __slots__ = ("model", "linked", "nodes", "reads", "bits", "memo", "counters", "lock")

    def __init__(self, model: Model[Space]) -> None:
        self.model = model
        self.linked: LinkedModel = model.linked
        self.nodes = self.linked.nodes
        self.reads: ModelReads = infer(self.linked)
        self.bits = BitTable()
        self.memo: Memo[Called] = Memo()
        self.counters = Counters()
        self.lock = RLock()


_MISSING = object()


class World:
    """A reading of every choice and fact of one model, and the entries evaluated in it.

    ``facts`` and ``stored`` (dormant choices included) by node; ``values``: one answer
    object per fact and stored value, shared with the parent for what did not change;
    ``parent`` and ``delta``: the world this one takes entries from, and the bits of the
    choices and facts it reads otherwise; ``cache``: the entries found here (only ever
    added to)."""

    __slots__ = ("evaluator", "facts", "stored", "values", "parent", "delta", "cache")

    def __init__(
        self,
        evaluator: Evaluator,
        facts: Mapping[NodeId, object],
        stored: Mapping[NodeId, object],
        values: dict[NodeId, Available[object]],
        parent: World | None = None,
        delta: ReadSet = 0,
    ) -> None:
        self.evaluator = evaluator
        self.facts = facts
        self.stored = stored
        self.values = values
        self.parent = parent
        self.delta = delta
        self.cache: dict[NodeId, Entry] = {}

    @classmethod
    def start(
        cls, evaluator: Evaluator, facts: Mapping[NodeId, object], stored: Mapping[NodeId, object]
    ) -> World:
        """A world of its own: facts and stored values already recognized and frozen."""
        values = {node: Available(value) for node, value in (*facts.items(), *stored.items())}
        return cls(evaluator, dict(facts), dict(stored), values)

    def successor(
        self,
        *,
        facts: Mapping[NodeId, object] | None = None,
        stored: Mapping[NodeId, object] | None = None,
    ) -> World:
        """The world of these facts and stored values (this world's where not given), made
        from this one: its delta is the choices and facts whose values differ, by their
        value semantics."""
        evaluator = self.evaluator
        values = dict(self.values)
        delta = 0
        new_facts = self.facts if facts is None else dict(facts)
        new_stored = self.stored if stored is None else dict(stored)
        for old, new in ((self.facts, new_facts), (self.stored, new_stored)):
            if old is new:
                continue
            for node, value in new.items():
                held = old.get(node, _MISSING)
                if held is _MISSING or not _same_value(evaluator.nodes[node], held, value):
                    delta |= evaluator.bits.bit(node)
                    values[node] = Available(value)
            for node in old:
                if node not in new:
                    delta |= evaluator.bits.bit(node)
                    del values[node]
        return World(evaluator, new_facts, new_stored, values, self, delta)

    # -- evaluation ----------------------------------------------------------------------

    def entry(self, node: NodeId) -> Entry:
        """``node``'s entry in this world: the one way to evaluate (from a driver)."""
        with self.evaluator.lock:
            return self._get(node)

    def result(self, node: NodeId) -> QueryResult[object]:
        return self.entry(node).result

    def _get(self, node: NodeId) -> Entry:
        """``node``'s entry: held here, taken from a parent that read nothing this world
        reads otherwise, or evaluated (with the nearest parent's entry for early cutoff)."""
        found = self.cache.get(node)
        if found is not None:
            return found
        previous: Entry | None = None
        world: World = self
        reached = 0
        while world.parent is not None:
            reached |= world.delta
            held = world.parent.cache.get(node)
            if held is not None:
                if not held.reads & reached:
                    self.evaluator.counters.inherited += 1
                    return self.cache.setdefault(node, held)
                previous = held
                break
            world = world.parent
        entry = self._evaluate(node, previous)
        return self.cache.setdefault(node, entry)

    def _evaluate(self, index: NodeId, previous: Entry | None) -> Entry:
        evaluator = self.evaluator
        node = evaluator.nodes[index]
        if previous is not None and node.kind not in _OWN_BIT and not _refused_view(previous):
            cut = self._cut(previous)
            if cut is not None:
                return cut
        evaluator.counters.evaluated += 1
        trace: list[tuple[NodeId, QueryResult[object]]] = []
        reads = 0
        if node.guard is not None:
            guard, reads = self.read(trace, node.guard, reads)
            inactive = _guard_result(node, guard.result)
            # A Space's validity under a guard that waits is still its pieces' AND: each
            # piece inside reads the same guard first and answers unknown.
            if inactive is not None and not (
                node.kind == "validity" and isinstance(inactive, Unresolved)
            ):
                return Entry(inactive, reads, tuple(trace), _guard_assessment(node, inactive))
        return _FRAMES[node.kind](self, node, trace, reads)

    def _cut(self, previous: Entry) -> Entry | None:
        """Early cutoff: ``previous`` (a parent's entry of the node) stands, the same answer
        object, where everything its trace read answers the same here."""
        reads = 0
        for dependency, answer in previous.trace:
            found = self._get(dependency)
            if not self.same(dependency, found.result, answer):
                return None
            reads |= found.reads
        self.evaluator.counters.cut += 1
        return Entry(
            previous.result,
            reads,
            previous.trace,
            previous.assessment,
            previous.own_refusal,
            previous.via,
        )

    def same(self, node: NodeId, left: QueryResult[object], right: QueryResult[object]) -> bool:
        """Whether two answers of ``node`` are the same: the same object, or equal by the
        node's value semantics (never bare ``==``: ``True == 1``, and an array has no
        truth value); two non-values of one variant with equal findings."""
        counters = self.evaluator.counters
        counters.compared += 1
        if left is right:
            counters.identical += 1
            return True
        counters.semantic += 1
        if type(left) is not type(right):
            return False
        if isinstance(left, Available):
            return _same_value(
                self.evaluator.nodes[node], left.value, cast(Available[object], right).value
            )
        try:
            return bool(left.findings == cast(NonValue, right).findings)
        except Exception:  # noqa: BLE001 - findings without equality are never the same
            return False

    def read(
        self, trace: list[tuple[NodeId, QueryResult[object]]], node: NodeId, reads: ReadSet
    ) -> tuple[Entry, ReadSet]:
        """A frame's read of ``node``: its entry, recorded in ``trace``, its reads added."""
        entry = self._get(node)
        trace.append((node, entry.result))
        return entry, reads | entry.reads


def _refused_view(entry: Entry) -> bool:
    """A view whose Space was refused: its readiness reads the pieces' assessments, which
    its trace does not hold."""
    assessment = entry.assessment
    return isinstance(assessment, ViewAssessment) and isinstance(assessment.validity, Rejected)


def _same_value(node: Node, left: object, right: object) -> bool:
    if left is right:
        return True
    if node.semantics is None:
        return False
    return equal(node.semantics, left, right, owner=node.owner, role="early cutoff")


# -- answers ----------------------------------------------------------------------------


def _blocked(answers: Sequence[QueryResult[object]]) -> NonValue | None:
    """Required inputs keep every distinct finding of the highest-precedence non-value:
    refused, then unresolved, then inapplicable."""
    for variant in (Rejected, Unresolved, Inapplicable):
        matches = tuple(answer for answer in answers if isinstance(answer, variant))
        if matches:
            return variant(merged_findings(matches))
    return None


def _guard_result(node: Node, answer: QueryResult[object]) -> NonValue | None:
    """What a node answers under its guard's answer: nothing (it applies), inapplicable,
    or the guard's own non-value."""
    if not isinstance(answer, Available):
        return answer
    if type(answer.value) is not bool:
        raise EvaluationError(node.owner, "applicability", "guard must return bool")
    return None if answer.value else _INAPPLICABLE


def _guard_assessment(node: Node, answer: NonValue) -> Assessment | None:
    if node.kind == "view":
        return assess_view(answer, owner=node.key, applicability=answer)
    if node.kind == "constraint":
        return ConstraintAssessment({node.key: answer}, answer)
    if node.kind == "validity":
        return ConstraintAssessment({}, answer)
    return None


def _clone(node: Node, value: object, *, owner: str, role: str) -> object:
    if node.semantics is None:
        raise EvaluationError(owner, role, f"{node.key} has no value semantics")
    return snapshot_value(node.semantics, value, owner=owner, role=role)


def public_value(node: Node, answer: Available[object]) -> object:
    """A value as a reader receives it: detached from the entry by its semantics (a
    validity's verdict is an immutable bool)."""
    if node.kind == "validity" and type(answer.value) is bool:
        return answer.value
    return _clone(node, answer.value, owner=node.owner, role="public value snapshot")


def _unassigned(node: Node) -> Unresolved:
    return Unresolved(
        (
            Finding(
                FindingKind.BLOCKER,
                "decision-unassigned",
                node.owner,
                "decision requires a commitment",
            ),
        )
    )


def _input_missing(node: Node) -> Unresolved:
    return Unresolved(
        (
            Finding(
                FindingKind.LIMITATION,
                "input-missing",
                node.owner,
                "optional input was omitted at start",
            ),
        )
    )


def _inputs_absent(node: Node, absent: list[str]) -> Unresolved:
    """A present choice's domain whose input does not apply: membership cannot be
    decided (inputs exist under the choice's own existence guards, so this needs an
    input guarded by something else)."""
    return Unresolved(
        tuple(
            Finding(
                FindingKind.BLOCKER,
                "domain-input-absent",
                node.owner,
                "an input of the domain does not apply where the choice is present",
                details=(("input", owner),),
            )
            for owner in absent
        )
    )


def _guard_piece(owner: str, refusal: Rejected) -> Rejected:
    """A validity piece for a guard refused through a refused input: the guarded scope or
    member is not in play, and the refusal is the input's, carried beside."""
    return Rejected(
        (
            Finding(
                FindingKind.REJECTION,
                "guard-refused",
                owner,
                "its guard reads a refused input",
            ),
            *refusal.findings,
        )
    )


def _empty_domain(node: Node) -> Rejected:
    return reject("domain-empty", "the domain lists no value", owner=node.owner)


# -- frames: each kind of node ----------------------------------------------------------

_Trace = list[tuple[NodeId, QueryResult[object]]]


def _param(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    reads |= world.evaluator.bits.bit(node.index)
    value = world.values.get(node.index)
    if value is not None:
        return Entry(value, reads, tuple(trace))
    return Entry(_input_missing(node), reads, tuple(trace))


def _const(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    return Entry(Available(node.value), reads, tuple(trace))


def _decision(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """A present choice: its stored value where its type check holds (unresolved while a
    fact it needs is missing, refused through a refused input of its domain, refused
    outside it), or unassigned when nothing is stored. Its guard was read first: a
    stored value under an absent scope is dormant and reads inapplicable."""
    reads |= world.evaluator.bits.bit(node.index)
    value = world.values.get(node.index)
    if value is None:
        return Entry(_unassigned(node), reads, tuple(trace))
    if node.domain is None:
        return Entry(value, reads, tuple(trace))
    check, reads = _type_check(world, node, value, trace, reads)
    if isinstance(check, Available) and check.value is True:
        return Entry(value, reads, tuple(trace))
    return Entry(check, reads, tuple(trace))


def _forward(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """An alias or a guard: its output's answer."""
    if node.output is None:
        raise EvaluationError(node.owner, node.kind, "missing output reference")
    entry, reads = world.read(trace, node.output, reads)
    return Entry(entry.result, reads, tuple(trace))


def _select(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """A member read through a Decision over nodes: the selected candidate's member, or
    inapplicable where the candidate places nothing or has no such member."""
    if node.selector is None or node.selection_index is None:
        raise EvaluationError(node.owner, "selection", "missing selector")
    selector, reads = world.read(trace, node.selector, reads)
    if not isinstance(selector.result, Available):
        return Entry(selector.result, reads, tuple(trace))
    case = selector.result.value
    if type(case) is not str:
        raise EvaluationError(node.owner, "selection", "selector is not a declared case")
    target = node.selection_index.get(case)
    if target is None:
        return Entry(_INAPPLICABLE, reads, tuple(trace))
    entry, reads = world.read(trace, target, reads)
    return Entry(entry.result, reads, tuple(trace))


def _view(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """An active view: its output, then its Space's validity, combined by
    ``assess_view``. Where the Space is refused, the pieces still unresolved are its
    readiness, read from the pieces' assessments (so such a view is never cut early:
    its trace does not hold them)."""
    if node.output is None or len(node.constraints) != 1:
        raise EvaluationError(node.owner, "view", "missing output or validity reference")
    output, reads = world.read(trace, node.output, reads)
    validity, reads = world.read(trace, node.constraints[0], reads)
    waiting = None
    if isinstance(validity.result, Rejected):
        waiting = _waiting(world, node.constraints[0])
    view = assess_view(
        output.result,
        owner=node.key,
        validity=cast(QueryResult[bool], validity.result),
        validity_owner=world.evaluator.nodes[node.constraints[0]].key,
        waiting=waiting,
    )
    return Entry(view.accepted_result, reads, tuple(trace), view)


def _waiting(world: World, index: NodeId) -> dict[str, QueryResult[object]]:
    """The pieces still unresolved in the refused Space whose validity is ``index``, its
    refused placed Spaces' with them, by key: what readiness reports beside a refusal,
    which hides them (false beats unknown)."""
    nodes, keys, found = world.evaluator.nodes, world.evaluator.linked.keys, {}
    pending = [index]
    while pending:
        assessment = world._get(pending.pop()).assessment
        if not isinstance(assessment, ConstraintAssessment):
            continue
        for key, answer in assessment.results.items():
            if isinstance(answer, Unresolved):
                found[key] = cast(QueryResult[object], answer)
            elif isinstance(answer, Rejected) and nodes[keys[key]].kind == "validity":
                pending.append(keys[key])
    return found


def _validity(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """A Space's validity: Kleene's AND of its pieces, refused when one is false, else
    unresolved when one is unknown, else true. The pieces (``Node.constraints``):

    - each constraint, as it answers (inactive counts true);
    - each placed Space's validity (absent counts true);
    - each choice, only where it is a piece: present with a listed domain that lists no
      value (refused, ``domain-empty``); presence unknown and its list empty or not yet
      known (unknown). A stored value's type check is no piece.

    A constraint, a choice or a placed Space whose guard is refused through a refused
    input is one guard piece (``guard-refused``, the input's refusal beside), once per
    guard node."""
    nodes = world.evaluator.nodes
    answers: dict[str, QueryResult[bool]] = {}
    guards: set[NodeId] = set()
    for index in node.constraints:
        piece = nodes[index]
        if piece.kind == "decision":
            found, reads = _choice_piece(world, piece, trace, reads, guards)
            if found is not None:
                answers[piece.key] = found
            continue
        entry, reads = world.read(trace, index, reads)
        answer = cast(QueryResult[bool], entry.result)
        if isinstance(answer, Rejected) and piece.guard is not None:
            guard, reads = world.read(trace, piece.guard, reads)
            if isinstance(guard.result, Rejected):
                if piece.guard in guards:
                    continue
                guards.add(piece.guard)
                owner = (
                    world.evaluator.linked.scopes[piece.scope].name
                    if piece.kind == "validity"
                    else piece.owner
                )
                answer = _guard_piece(owner, guard.result)
        answers[piece.key] = answer
    assessment = assess_constraints(answers)
    return Entry(cast(QueryResult[object], assessment.result), reads, tuple(trace), assessment)


def _choice_piece(
    world: World, piece: Node, trace: _Trace, reads: ReadSet, guards: set[NodeId]
) -> tuple[QueryResult[bool] | None, ReadSet]:
    """A choice's validity piece, read from its guard and its listed values (never from
    its stored value), or ``None`` where it is no piece."""
    waiting: Unresolved | None = None
    if piece.guard is not None:
        guard, reads = world.read(trace, piece.guard, reads)
        answer = guard.result
        if isinstance(answer, Rejected):
            if piece.guard in guards:
                return None, reads
            guards.add(piece.guard)
            return _guard_piece(piece.owner, answer), reads
        if isinstance(answer, Unresolved):
            waiting = answer
        elif _guard_result(piece, answer) is not None:
            return None, reads  # absent
    if piece.domain is None:
        return None, reads
    listed, reads = _listed(world, piece, trace, reads)
    known = isinstance(listed, Available)
    values = cast(Available[tuple[object, ...] | None], listed).value if known else None
    if waiting is not None:
        # Presence unknown: an empty list (or one not yet known) is an unknown piece.
        return (None if known and values != () else waiting), reads
    if known:
        return (_empty_domain(piece) if values == () else None), reads
    return cast(QueryResult[bool], listed), reads


def _present(node: Node, answers: list[QueryResult[object]]) -> QueryResult[object]:
    """The single present source: two known present ones refuse at once; else a refused
    source refuses, and while one is unresolved the answer is unresolved."""
    active = [answer for answer in answers if not isinstance(answer, Inapplicable)]
    failures = [answer for answer in active if not isinstance(answer, Available)]
    known = len(active) - len(failures)
    if known > 1:
        failures.append(
            reject(
                "multiple-suppliers",
                f"{known} sources are present; at most one may supply this value",
                owner=node.owner,
            )
        )
    blocked = _blocked(failures)
    if blocked is not None:
        return blocked
    if not active:
        return Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "input-unsupplied",
                    node.owner,
                    "no present source supplies this value",
                ),
            )
        )
    return active[0]


def _graph(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """A ``present`` node (the one present source of a formal) or ``Members`` (each
    present member's export, located)."""
    if node.kind == "present":
        answers: list[QueryResult[object]] = []
        for _, target in node.alternatives:
            entry, reads = world.read(trace, target, reads)
            answers.append(entry.result)
        return Entry(_present(node, answers), reads, tuple(trace))
    located: list[Located[object]] = []
    failures: list[QueryResult[object]] = []
    for (name, target), member in zip(node.alternatives, cast(tuple[str, ...], node.value)):
        entry, reads = world.read(trace, target, reads)
        answer = entry.result
        if isinstance(answer, Inapplicable):
            continue
        if isinstance(answer, Available):
            located.append(Located(name or None, member, answer.value))
        else:
            failures.append(answer)
    blocked = _blocked(failures)
    result: QueryResult[object] = blocked if blocked is not None else Available(tuple(located))
    return Entry(result, reads, tuple(trace))


def _finish(node: Node, returned: object) -> Called:
    """A callback's returned value as the node's answer and assessment."""
    if node.kind == "constraint":
        normalized = constraint_result(cast(bool | QueryResult[bool], returned), node.owner)
        return Called(
            cast(QueryResult[object], normalized),
            ConstraintAssessment({node.key: normalized}, normalized),
            isinstance(normalized, Rejected),
        )
    if isinstance(returned, (Inapplicable, Rejected, Unresolved)):
        return Called(owned_result(returned, node.owner), None, isinstance(returned, Rejected))
    if node.semantics is None:
        raise EvaluationError(node.owner, node.kind, "derived output has no value semantics")
    value = returned.value if isinstance(returned, Available) else returned
    return Called(
        Available(snapshot_value(node.semantics, value, owner=node.owner, role=node.kind))
    )


def _entry(called: Called, reads: ReadSet, trace: Trace) -> Entry:
    return Entry(called.result, reads, trace, called.assessment, called.own_refusal, called.via)


def _callback(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """A derived value, a constraint or a view's body: its authored function."""
    if node.function is None:
        raise EvaluationError(node.owner, node.kind, "missing callback")
    if node.call_style == "self":
        return _method(world, node, trace, reads)
    answers: list[QueryResult[object]] = []
    for argument in node.arguments:
        entry, reads = world.read(trace, argument.node, reads)
        answers.append(entry.result)
    blocked = _blocked([answer for answer in answers if not isinstance(answer, Available)])
    if blocked is not None:
        return Entry(blocked, reads, tuple(trace), _guard_assessment(node, blocked))
    evaluator = world.evaluator
    held = evaluator.memo.explicit(node.index, "call", tuple(answers))
    if held is not None:
        evaluator.counters.hit(node.kind)
        return _entry(held.outcome, reads, tuple(trace))
    arguments = {
        argument.name: _clone(
            evaluator.nodes[argument.node],
            cast(Available[object], answer).value,
            owner=node.owner,
            role="dependency snapshot",
        )
        for argument, answer in zip(node.arguments, answers)
    }
    evaluator.counters.call(node.kind)
    function = node.function
    try:
        called = _finish(node, function(**arguments))
    except (EvaluationError, ValueUnavailableError):
        raise
    except Exception as cause:
        raise EvaluationError(node.owner, node.kind, str(cause)) from cause
    evaluator.memo.keep_explicit(node.index, "call", tuple(answers), called)
    return _entry(called, reads, tuple(trace))


class MethodReader:
    """The reads of one running ``self`` method, on the world it runs in: each read is
    evaluated there and recorded, in order, with its answer; a read of a non-value is
    kept (``blocked``) and halts the method."""

    __slots__ = ("world", "node", "trace", "reads", "blocked", "via")

    def __init__(self, world: World, node: Node) -> None:
        self.world = world
        self.node = node
        self.trace: list[tuple[NodeId, QueryResult[object]]] = []
        self.reads: ReadSet = 0
        self.blocked: dict[NodeId, NonValue] = {}
        self.via: list[tuple[NodeId, NodeId]] = []

    def answer(self, index: NodeId) -> QueryResult[object]:
        """``index``'s answer, recorded; a non-value does not halt the method."""
        entry = self.world._get(index)
        self.trace.append((index, entry.result))
        self.reads |= entry.reads
        return entry.result

    def value(self, index: NodeId) -> object:
        """``index``'s value, recorded and detached; a non-value halts the method."""
        answer = self.answer(index)
        node = self.world.evaluator.nodes[index]
        if isinstance(answer, Available):
            return public_value(node, answer)
        self.blocked[index] = answer
        raise ValueUnavailableError(answer, context=node.owner)


def receiver(reader: MethodReader, scope: int) -> Space:
    """A Space of ``scope``'s class reading through ``reader``: a method's ``self``."""
    space_type = reader.world.evaluator.linked.scopes[scope].space_type
    instance = object.__new__(space_type)
    object.__setattr__(instance, "_state", reader)
    object.__setattr__(instance, "_scope", scope)
    return instance


def _method(world: World, node: Node, trace: _Trace, reads: ReadSet) -> Entry:
    """A ``self`` method: a plain call on a receiver reading through the world, memoized
    by its traces. A kept trace whose every read answers the same here gives its
    outcome without a call."""
    evaluator = world.evaluator
    for recorded, called in evaluator.memo.traces(node.index):
        more = 0
        for dependency, answer in recorded:
            found = world._get(dependency)
            if not world.same(dependency, found.result, answer):
                break
            more |= found.reads
        else:
            evaluator.counters.hit(node.kind)
            return _entry(called, reads | more, (*trace, *recorded))
    reader = MethodReader(world, node)
    function = cast(Callable[[object], object], node.function)
    evaluator.counters.call(node.kind)
    returned: object = None
    try:
        returned = function(receiver(reader, node.scope))
    except ValueUnavailableError:
        if not reader.blocked:
            raise
    except EvaluationError:
        raise
    except Exception as cause:
        if not reader.blocked:
            raise EvaluationError(node.owner, node.kind, str(cause)) from cause
    if reader.blocked:
        blocked = cast(NonValue, _blocked(list(reader.blocked.values())))
        called = Called(blocked, _guard_assessment(node, blocked), via=tuple(reader.via))
    else:
        finished = _finish(node, returned)
        called = Called(
            finished.result, finished.assessment, finished.own_refusal, tuple(reader.via)
        )
    recorded = tuple(reader.trace)
    evaluator.memo.keep_method(node.index, recorded, called)
    return _entry(called, reads | reader.reads, (*trace, *recorded))


# -- domains ----------------------------------------------------------------------------


def _domain_inputs(
    world: World, node: Node, trace: _Trace, reads: ReadSet
) -> tuple[tuple[QueryResult[object], ...], NonValue | None, ReadSet]:
    """The answers of a Decision's domain inputs, read in ``world`` (and traced), and the
    non-value that keeps the domain from being asked (an input that does not apply where
    the choice is present cannot decide it)."""
    answers: list[QueryResult[object]] = []
    for argument in node.domain_arguments:
        entry, reads = world.read(trace, argument.node, reads)
        answers.append(entry.result)
    blocked = _blocked([answer for answer in answers if not isinstance(answer, Available)])
    if isinstance(blocked, Inapplicable):
        nodes = world.evaluator.nodes
        absent = [
            nodes[argument.node].owner
            for argument, answer in zip(node.domain_arguments, answers)
            if isinstance(answer, Inapplicable)
        ]
        blocked = _inputs_absent(node, absent)
    return tuple(answers), blocked, reads


def _domain_values(
    world: World, node: Node, answers: tuple[QueryResult[object], ...]
) -> dict[str, object]:
    nodes = world.evaluator.nodes
    return {
        argument.name: _clone(
            nodes[argument.node],
            cast(Available[object], answer).value,
            owner=node.owner,
            role="dependency snapshot",
        )
        for argument, answer in zip(node.domain_arguments, answers)
    }


def _type_check(
    world: World, node: Node, value: Available[object], trace: _Trace, reads: ReadSet
) -> tuple[QueryResult[object], ReadSet]:
    """Whether a choice's domain holds ``value`` here (memoized on the value's answer and
    the domain inputs' answers)."""
    assert node.domain is not None and node.semantics is not None
    answers, blocked, reads = _domain_inputs(world, node, trace, reads)
    if blocked is not None:
        return blocked, reads
    evaluator = world.evaluator
    key = (value, *answers)
    held = evaluator.memo.explicit(node.index, "membership", key)
    if held is not None:
        evaluator.counters.hit("domain membership")
        return held.outcome.result, reads
    candidate = _clone(node, value.value, owner=node.owner, role="domain candidate snapshot")
    evaluator.counters.call("domain membership")
    result = node.domain.membership(
        candidate,
        _domain_values(world, node, answers),
        semantics=node.semantics,
        owner=node.owner,
    )
    evaluator.memo.keep_explicit(
        node.index, "membership", key, Called(cast(QueryResult[object], result))
    )
    return cast(QueryResult[object], result), reads


def _listed(
    world: World, node: Node, trace: _Trace, reads: ReadSet
) -> tuple[QueryResult[object], ReadSet]:
    """A choice's listed values here (``Available(None)``: membership only, nothing is
    listed): its domain's enumeration, each value checked by membership (memoized on the
    domain inputs' answers)."""
    assert node.domain is not None and node.semantics is not None
    answers, blocked, reads = _domain_inputs(world, node, trace, reads)
    if blocked is not None:
        return blocked, reads
    if node.domain.candidates is None:
        return Available(None), reads
    evaluator = world.evaluator
    held = evaluator.memo.explicit(node.index, "listing", answers)
    if held is not None:
        evaluator.counters.hit("domain listing")
        return held.outcome.result, reads
    arguments = _domain_values(world, node, answers)
    evaluator.counters.call("domain listing")
    listed = cast(
        QueryResult[object],
        node.domain.enumerate(arguments, semantics=node.semantics, owner=node.owner),
    )
    if isinstance(listed, Available) and node.domain._finite_values is None:
        kept = tuple(
            value
            for value in cast(tuple[object, ...], listed.value)
            if not isinstance(
                node.domain.membership(
                    value, arguments, semantics=node.semantics, owner=node.owner
                ),
                Rejected,
            )
        )
        listed = Available(kept)
    evaluator.memo.keep_explicit(node.index, "listing", answers, Called(listed))
    return listed, reads


def listed_values(world: World, node: NodeId) -> QueryResult[object]:
    """A choice's listed values in ``world`` (``Available(None)``: membership only)."""
    with world.evaluator.lock:
        found, _ = _listed(world, world.evaluator.nodes[node], [], 0)
        return found


_FRAMES: Mapping[NodeKind, Callable[[World, Node, _Trace, ReadSet], Entry]] = {
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
    "Called",
    "Entry",
    "Evaluator",
    "MethodReader",
    "World",
    "listed_values",
    "public_value",
    "receiver",
]
