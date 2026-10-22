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
model (``_memo.Memo``), each with the trace it was found from (an explicit function's
arguments' answers, a ``self`` method's reads), and found again where every read of the
trace answers the same: the same answer object, else equal by value semantics. A
``self`` method is a plain call; its receiver reads through a ``MethodReader`` on the
world it runs in, which records each read and checks it against the reads inferred
from the method's source (``_reads``).

Each driver demand (``World.entry``) is a **run**. Its authored calls each run in a
copy of the context the driver entered with, and fail as ``_failure`` says (the first
failure primary, cleanup in order, failures kept for the run only). A node demanded
while it is being evaluated is a dependency cycle. Evaluation recurses on the Python
stack to ``DEPTH`` evaluations; past it the **depth guard** unwinds (``Deeper``), and
the run evaluates the deep node first, then again what it abandoned: authored
functions are pure, so one run again answers the same (it may start more than once;
it completes once).

Both the memo and the depth guard rest on that purity. The **purity check**
(``check_purity``, a test mode) tests it: every memo hit runs its function again, and so
does every method the depth guard abandoned once it completes; an answer that differs
by value semantics is the function's ``EvaluationError``.

A configuration (``_state.State``) is its facts and stored choices and the world they
read in; a change makes the world of the configuration it would publish from its
receiver's (``_assign``). A chain of such worlds is kept short by sharing its bulk
(``World.flatten``): a world published more than ``FLATTEN`` worlds below one standing
alone takes, as its parent, its ancestor holding the most entries (its **base**, made
to stand alone first), and folds only the layers between into its own entries. A
lookup then goes no further than the world and its base, and nothing large is copied.
A flattened world whose own entries reach ``BASE_SHARE`` of its base's stands alone in
turn, the base of the worlds made after it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.reporting.manifest import note

from ._collapse import guard_implies
from ._configuration import Space
from ._failure import (
    ACTIVE,
    Call,
    CleanupFailure,
    Failing,
    Failure,
    check_transformation,
    current,
    driver_only,
    exception,
    fault,
    from_exception,
    raise_failure,
    remember_exception,
    remember_failure,
)
from ._memo import BitTable, Counters, Kept, Key, Memo, ReadSet, Trace
from ._nodes import NodeDecl, unwrap
from ._reads import ModelReads, infer
from .declarations import Param, declared_path
from .errors import EvaluationError, RequestError, ValueUnavailableError
from .graph import Located
from .ir import Argument, Choice, Linked, Node, NodeId, NodeKind, Questions
from .references import descend
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
from .semantics import equal, recognize
from .semantics import snapshot as snapshot_value

if TYPE_CHECKING:
    from .compiler import Model
    from .domains import Domain
    from .semantics import ValueSemantics

Assessment: TypeAlias = ViewAssessment[object] | ConstraintAssessment
T = TypeVar("T")

INAPPLICABLE = Inapplicable()

# How many evaluations deep a run goes on the Python stack before the depth guard
# unwinds it. Each level through a method's read takes about a dozen Python frames.
DEPTH = 32

# How many worlds a chain of changes may stack before the newest is flattened.
FLATTEN = 8

# A flattened world stands alone once its own entries are this share of its base's: the
# base's entries it takes then cost at most twice what it already holds.
BASE_SHARE = 0.5

# The purity check (``check_purity``; on from the start where the environment variable
# FINN_SPACE_CHECK_PURITY is 1): every memo hit runs its function again, and so does a
# method the depth guard abandoned, once it completes; where the two answers differ by
# value semantics, the function is impure, its ``EvaluationError``.
_PURITY = os.environ.get("FINN_SPACE_CHECK_PURITY") == "1"


def check_purity(on: bool) -> bool:
    """Switch the purity check on or off, for every model; the setting it replaces."""
    global _PURITY  # noqa: PLW0603 - a test mode, switched for the process
    previous, _PURITY = _PURITY, on
    return previous


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
    went to its source, as (alias, source). ``existence``: what a method read only to
    know whether something exists (a selector, to find its candidate; a guard, for
    ``present``), never as a value."""

    result: QueryResult[object]
    reads: ReadSet
    trace: Trace
    assessment: Assessment | None = None
    own_refusal: bool = False
    via: tuple[tuple[NodeId, NodeId], ...] = ()
    existence: frozenset[NodeId] = frozenset()


@dataclass(frozen=True, slots=True)
class Called:
    """What an authored function's call came to, as a memo keeps it."""

    result: QueryResult[object]
    assessment: Assessment | None = None
    own_refusal: bool = False
    via: tuple[tuple[NodeId, NodeId], ...] = ()
    existence: frozenset[NodeId] = frozenset()


class Evaluator:
    """What every world of one model shares: its structure and its methods' inferred
    reads (both answered node by node, as read), the order its choices flow in, the bit
    table, the memo of authored functions, the work counters, and the model's lock (worlds
    share their parents' entries and the memo, so evaluation of one model is serialized;
    the structure resolves under it too)."""

    __slots__ = (
        "model",
        "linked",
        "nodes",
        "reads",
        "bits",
        "memo",
        "counters",
        "lock",
        "run",
        "_flow",
    )

    def __init__(self, model: Model[Space]) -> None:
        self.model = model
        self.linked: Linked = model.linked
        self.nodes = self.linked.nodes
        self.reads: ModelReads = infer(self.linked)
        self.bits = BitTable()
        self.memo: Memo[Called] = Memo()
        self.counters = Counters()
        self.lock = model.lock
        # The run in progress (under the lock), if any.
        self.run: _Run | None = None
        self._flow: tuple[NodeId, ...] | None = None

    def _whole(self) -> Questions:
        whole = self.linked.whole
        if whole is None:
            raise RequestError(
                "a table built by hand has no templates: it answers no question about its "
                "whole design space (its flow order, its presence readers)"
            )
        return whole

    def flow(self) -> tuple[NodeId, ...]:
        """The model's choices in the flow order (found once per model, on first use, from
        its templates: ``_whole``). Each loop in it is noted, a finding stated without
        refusing (``flow-loop``): its choices are taken in declaration order."""
        with self.lock:
            if self._flow is None:
                found = self._whole().flow()
                for loop in found.loops:
                    keys = tuple(self.linked.header(index).key for index in loop)
                    note(
                        Finding(
                            FindingKind.LIMITATION,
                            "flow-loop",
                            keys[0],
                            "choices whose presence, list or facts read one another: taken "
                            "in declaration order",
                            details=(("choices", keys),),
                        )
                    )
                self._flow = found.order
            return self._flow

    def presence_readers(self, source: NodeId) -> tuple[NodeId, ...]:
        """The choices whose presence may read the choice or fact ``source`` (found once
        per model, on first use, from its templates: ``_whole``)."""
        with self.lock:
            return self._whole().readers(source)


class Deeper(BaseException):
    """The depth guard: ``node`` was demanded in ``world`` deeper than ``DEPTH``. Every
    evaluation and call it passes through is abandoned (nothing kept); the run evaluates
    the node first, then again what it abandoned."""

    def __init__(self, world: World, node: NodeId) -> None:
        super().__init__(node)
        self.world = world
        self.node = node


_Key: TypeAlias = "tuple[World, NodeId]"


@dataclass(slots=True, eq=False)
class _Run:
    """One driver demand: the context its authored calls copy, the failures met (kept
    for this run only: a later demand runs the function again), what is being
    evaluated, on the Python stack (``active``) and abandoned to the run's own stack
    (``pending``), for cycles, and the methods the depth guard abandoned (for the purity
    check)."""

    context: Context
    failures: dict[_Key, Failure] = field(default_factory=dict)
    active: dict[_Key, None] = field(default_factory=dict)
    pending: dict[_Key, None] = field(default_factory=dict)
    depth: int = 0
    abandoned: set[_Key] = field(default_factory=set)  # methods the depth guard abandoned


def _drive(evaluator: Evaluator, demand: Callable[[], T]) -> T:
    """Run ``demand`` (a driver's evaluation) under the model's lock: past the depth
    guard, the deep node is evaluated from the run's own stack, then the demand again;
    a failure is raised as its exception."""
    driver_only("evaluation")
    with evaluator.lock:
        run = _Run(copy_context())
        outer, evaluator.run = evaluator.run, run
        failure: Failure | None = None
        try:
            stack: list[_Key] = []
            while True:
                try:
                    if not stack:
                        return demand()
                    world, node = stack[-1]
                    world._get(node)
                except Deeper as deeper:
                    if stack:
                        run.pending[stack[-1]] = None
                    stack.append((deeper.world, deeper.node))
                    continue
                except Failing:
                    if not stack:
                        raise
                stack.pop()
                if stack:
                    del run.pending[stack[-1]]
        except Failing as failing:
            failure = failing.failure
        finally:
            evaluator.run = outer
        raise_failure(failure)


_MISSING = object()


class World:
    """A reading of every choice and fact of one model, and the entries evaluated in it.

    ``facts`` and ``stored`` (dormant choices included) by node; ``values``: one answer
    object per fact and stored value, shared with the parent for what did not change;
    ``parent`` and ``delta``: the world this one takes entries from, and the bits of the
    choices and facts it reads otherwise; ``cache``: the entries found here (only ever
    added to). ``assumed``: guards taken as holding, in a world where a choice's
    membership is read (``membership``); its own such worlds are kept in
    ``memberships``."""

    __slots__ = (
        "evaluator",
        "facts",
        "stored",
        "values",
        "parent",
        "delta",
        "cache",
        "assumed",
        "memberships",
    )

    def __init__(
        self,
        evaluator: Evaluator,
        facts: Mapping[NodeId, object],
        stored: Mapping[NodeId, object],
        values: dict[NodeId, Available[object]],
        parent: World | None = None,
        delta: ReadSet = 0,
        assumed: frozenset[NodeId] = frozenset(),
    ) -> None:
        self.evaluator = evaluator
        self.facts = facts
        self.stored = stored
        self.values = values
        self.parent = parent
        self.delta = delta
        self.cache: dict[NodeId, Entry] = {}
        self.assumed = assumed
        self.memberships: dict[tuple[NodeId, ...], World] | None = None

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
        with evaluator.lock:  # the bit table is the model's
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

    def membership(self, node: NodeId) -> World:
        """Where the choice ``node``'s membership is read as if the choice existed: a world
        of this one's facts and stored choices in which every guard deciding whether it
        exists (the selectors above it at its case, the ``when`` and ``supplied`` guards of
        its scope and every enclosing one) is taken as holding, and every other guard is
        read as usual. Its membership reads facts only, so it is decided there wherever the
        choice sits. This world itself for a choice with no guard; kept on this world, by
        the guards (a world of its own, taking no entry of another)."""
        guards = self.evaluator.reads.guards(node)
        if not guards:
            return self
        if self.memberships is None:
            self.memberships = {}
        found = self.memberships.get(guards)
        if found is None:
            found = self.memberships[guards] = World(
                self.evaluator, self.facts, self.stored, self.values, assumed=frozenset(guards)
            )
        return found

    @property
    def depth(self) -> int:
        """How many worlds stand above this one."""
        depth, world = 0, self.parent
        while world is not None:
            depth, world = depth + 1, world.parent
        return depth

    def flatten(self) -> None:
        """Keep the chain above this world short by sharing its bulk: its ancestor holding
        the most entries (the oldest of equals) becomes its parent, its **base**, standing
        alone first; the layers between are folded into this world (each entry of theirs
        that read nothing this world reads otherwise, nearest first), and this world reads
        otherwise than its base only in the choices and facts whose values differ there.
        A lookup then walks two worlds, and nothing of the base is copied. Where this
        world's own entries reach ``BASE_SHARE`` of its base's, it stands alone in turn: a
        new base, for the worlds made after it.

        Only a world nothing has made a successor of yet is flattened; a base standing
        alone keeps its answers, and so do the worlds made from it (their lookups stop at
        it, which now holds what they found above it)."""
        with self.evaluator.lock:
            layers: list[World] = []
            world = self.parent
            while world is not None:
                layers.append(world)
                world = world.parent
            if not layers:
                return
            base = max(reversed(layers), key=lambda layer: len(layer.cache))
            base.stand_alone()
            reached = self._fold(layers[: layers.index(base)])
            self.parent = base
            self.delta = self._differing(base, reached)
            if len(self.cache) >= BASE_SHARE * len(base.cache):
                self.stand_alone()

    def stand_alone(self) -> None:
        """Take every entry of the worlds above this one that read nothing it reads
        otherwise, and drop them: a world of its own, whose answers are unchanged."""
        with self.evaluator.lock:
            layers: list[World] = []
            world = self.parent
            while world is not None:
                layers.append(world)
                world = world.parent
            self._fold(layers)
            self.parent = None
            self.delta = 0

    def _fold(self, layers: Sequence[World]) -> ReadSet:
        """Take into this world's cache each entry of ``layers`` (the worlds above it,
        nearest first) that read nothing this world reads otherwise than that layer; the
        choices and facts it then reads otherwise than the world above the last layer."""
        reached = self.delta
        cache = self.cache
        for layer in layers:
            for node, entry in layer.cache.items():
                if not entry.reads & reached:
                    cache.setdefault(node, entry)
            reached |= layer.delta
        return reached

    def _differing(self, base: World, reached: ReadSet) -> ReadSet:
        """The bits of ``reached`` whose choice or fact has another value here than in
        ``base`` (a value set back to the base's is read as the base reads it)."""
        evaluator = self.evaluator
        differing = 0
        for node in evaluator.bits.nodes(reached):
            here, there = self.values.get(node), base.values.get(node)
            if here is there:
                continue
            if (
                here is None
                or there is None
                or not _same_value(evaluator.nodes[node], here.value, there.value)
            ):
                differing |= evaluator.bits.bit(node)
        return differing

    # -- evaluation ----------------------------------------------------------------------

    def run(self, demand: Callable[[], T]) -> T:
        """``demand`` as one driver demand on this world's model: under the model's lock,
        its evaluations (``World._get``) in one run. A failure is raised as
        ``NativeEvaluationError`` (or as its cancellation)."""
        return _drive(self.evaluator, demand)

    def entry(self, node: NodeId) -> Entry:
        """``node``'s entry in this world: the one way to evaluate, from a driver. A
        failure is raised as ``NativeEvaluationError`` (or as its cancellation)."""
        return _drive(self.evaluator, lambda: self._get(node))

    def result(self, node: NodeId) -> QueryResult[object]:
        return self.entry(node).result

    def _get(self, node: NodeId) -> Entry:
        """``node``'s entry, inside a run: held here, taken from a parent that read
        nothing this world reads otherwise, or evaluated (with the nearest parent's entry
        for early cutoff). A failure met in this run raises again; a node being evaluated
        is a cycle; past ``DEPTH`` the depth guard unwinds."""
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
        run = self.evaluator.run
        assert run is not None, "an evaluation outside a run"
        key = (self, node)
        failure = run.failures.get(key)
        if failure is not None:
            raise Failing(failure)
        if key in run.active or key in run.pending:
            raise Failing(self._cycle(run, node))
        if run.depth >= DEPTH:
            raise Deeper(self, node)
        run.depth += 1
        run.active[key] = None
        outer = sys.exception()
        try:
            entry = self._evaluate(node, previous)
            return self.cache.setdefault(node, entry)
        except Deeper:
            raise
        except Failing as failing:
            run.failures[key] = failing.failure
            raise
        except BaseException as cause:
            spec = self.evaluator.nodes[node]
            failure = from_exception(cause, owner=spec.owner, role=spec.kind, outer=outer)
            run.failures[key] = failure
            raise Failing(failure) from cause
        finally:
            run.depth -= 1
            del run.active[key]

    def _cycle(self, run: _Run, node: NodeId) -> Failure:
        """A dependency cycle: ``node`` demanded while it is being evaluated."""
        nodes = self.evaluator.nodes
        path = [f"{nodes[index].owner} ({nodes[index].kind})" for _, index in run.pending]
        path += [f"{nodes[index].owner} ({nodes[index].kind})" for _, index in run.active]
        owner = nodes[node].owner
        return Failure(EvaluationError(owner, "dependency cycle", " -> ".join((*path, owner))))

    def _evaluate(self, index: NodeId, previous: Entry | None) -> Entry:
        evaluator = self.evaluator
        node = evaluator.nodes[index]
        if index in self.assumed:
            return Entry(Available(True), 0, ())
        cases = evaluator.reads.cases
        if (
            previous is not None
            and node.kind not in _OWN_BIT
            and index not in cases  # a case reads its selector's stored value: never cut
            and not _refused_view(previous)
        ):
            cut = self._cut(previous)
            if cut is not None:
                return cut
        evaluator.counters.evaluated += 1
        trace: list[tuple[NodeId, QueryResult[object]]] = []
        reads = 0
        if node.guard is not None:
            guard, reads = self.read(trace, node.guard, reads)
            inactive = _guard_result(node, guard.result)
            if isinstance(inactive, (Unresolved, Rejected)) and (
                node.kind == "guard" or index in cases
            ):
                # Presence is the AND over the whole path: a condition that does not hold
                # makes its scope absent, whatever is still undecided or refused above it.
                held, reads = self._holds_here(node, trace, reads)
                if held is False:
                    return Entry(Available(False), reads, tuple(trace))
            if node.kind == "constraint" and isinstance(inactive, Rejected):
                # Under a guard refused through an input its scope is not in play: the
                # constraint does not apply, and the guard's piece stands for it.
                inactive = INAPPLICABLE
            # A Space's validity under a guard that waits is still its pieces' AND: each
            # piece inside reads the same guard first and answers unknown. Under a ``when``
            # refused below a guard that waits it waits whole, on what that guard waits on
            # (reading 8's second half: ``refused_waiting``).
            evaluated = node.kind == "validity" and isinstance(inactive, Unresolved)
            if evaluated:
                top, reads = self.refused_waiting(node.guard, trace, reads)
                evaluated = top is None
            if inactive is not None and not evaluated:
                return Entry(inactive, reads, tuple(trace), _guard_assessment(node, inactive))
        return _FRAMES[node.kind](self, node, trace, reads)

    def _holds_here(
        self, node: Node, trace: list[tuple[NodeId, QueryResult[object]]], reads: ReadSet
    ) -> tuple[bool | None, ReadSet]:
        """A guard's own condition, read while a guard above it is undecided or refused:
        false where a case's selector stores another case, or a ``when`` reads false;
        ``None`` where it holds or is not known.

        A ``when``'s condition computed in the scope it is written in (a derived value,
        an alias) carries that scope's guard, the one undecided here: it is evaluated past
        that guard (``_unguarded``), its reads recorded as this guard's own. What it reads
        keeps its own presence: a choice of a case still open reads unknown, and so does
        a condition that is such a choice."""
        evaluator = self.evaluator
        held = evaluator.reads.cases.get(node.index)
        if held is not None:
            selector, case = held
            reads |= evaluator.bits.bit(selector)
            value = self.values.get(selector)
            return (None if value is None or value.value == case else False), reads
        answer, reads = self.condition(node, trace, reads)
        return (False if isinstance(answer, Available) and answer.value is False else None), reads

    def condition(
        self, node: Node, trace: list[tuple[NodeId, QueryResult[object]]], reads: ReadSet
    ) -> tuple[QueryResult[object] | None, ReadSet]:
        """A ``when`` guard's own condition, read whatever the guard above it answers
        (``_holds_here``), or ``None`` for a node that is no such guard (a case's guard
        reads its selector)."""
        evaluator = self.evaluator
        if node.kind != "guard" or node.output is None or node.index in evaluator.reads.cases:
            return None, reads
        condition = evaluator.nodes[node.output]
        if (
            condition.guard is None
            or condition.kind in _OWN_BIT
            or not guard_implies(evaluator.nodes, condition.guard, node.guard)
        ):
            entry, reads = self.read(trace, node.output, reads)
        else:
            entry = self._unguarded(condition)
            trace.extend(entry.trace)
            reads |= entry.reads
        return entry.result, reads

    def refused_waiting(
        self, guard: NodeId, trace: list[tuple[NodeId, QueryResult[object]]], reads: ReadSet
    ) -> tuple[NodeId | None, ReadSet]:
        """The highest ``when`` guard in the chain of ``guard`` (which waits) refused through
        an input below a guard that waits, or ``None``: climbing from ``guard`` while each
        guard above waits, a ``when`` guard whose own condition, read past the guard above
        it (``condition``), is refused. What it guards waits, on what the guard above it
        waits on: refused takes effect once all above it holds (reading 8's second half).
        Where nothing above waits the guard answers refused, and the refused guard's piece
        counts (``_guard_refused``)."""
        nodes = self.evaluator.nodes
        found: NodeId | None = None
        index = guard
        while (outer := nodes[index].guard) is not None:
            answer, reads = self.condition(nodes[index], trace, reads)
            if isinstance(answer, Rejected):
                found = index
            index = outer
            above, reads = self.read(trace, index, reads)
            if not isinstance(above.result, Unresolved):
                break
        return found, reads

    def _unguarded(self, node: Node) -> Entry:
        """``node`` evaluated here as if its own guard held (its frame alone, kept nowhere):
        what it reads, read as usual."""
        return _FRAMES[node.kind](self, node, [], 0)

    def _cut(self, previous: Entry) -> Entry | None:
        """Early cutoff: ``previous`` (a parent's entry of the node) stands, the same answer
        object, where everything its trace read answers the same here. A read that fails
        here leaves the node to be evaluated (its cleanup, if a method, runs)."""
        reads = 0
        for dependency, answer in previous.trace:
            try:
                found = self._get(dependency)
            except Failing:
                return None
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
            previous.existence,
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
    """Whether two values of ``node`` are equal by its semantics, each compared as a
    detached copy (equality may not change a held value)."""
    if left is right:
        return True
    semantics = node.semantics
    if semantics is None:
        return False
    role = "early cutoff"
    owner = node.owner
    if not (
        recognize(semantics, left, owner=owner, role=role)
        and recognize(semantics, right, owner=owner, role=role)
    ):
        return False
    left = snapshot_value(semantics, left, owner=owner, role=role)
    right = snapshot_value(semantics, right, owner=owner, role=role)
    return equal(semantics, left, right, owner=owner, role=role)


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
    return None if answer.value else INAPPLICABLE


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
    validity's answer is an immutable bool)."""
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


def _inputs_absent(node: Node, absent: list[str]) -> Rejected:
    """A present choice's domain whose input does not apply: the input does not exist
    for these facts (inputs under the choice's own existence guards are present whenever
    it is, so it sits under a guard over facts that is false), and nothing the
    configuration could set would make the domain: refused, naming each input."""
    return Rejected(
        tuple(
            Finding(
                FindingKind.REJECTION,
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
        return Entry(INAPPLICABLE, reads, tuple(trace))
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

    - each constraint, as it answers (inactive counts true; under a guard refused through
      an input it does not apply);
    - each placed Space's validity (absent counts true);
    - each choice, only where it is a piece: present with a listed domain that lists no
      value (refused, ``domain-empty``); presence unknown and its list empty or not yet
      known (unknown). A stored value's type check is no piece.

    A constraint, a choice or a placed Space whose guard is refused through a refused
    input is one guard piece (``guard-refused``, the input's refusal beside), counted
    once at the highest refused guard of its chain (``_guard_refused``). One below a
    ``when`` refused through an input under a guard that waits is unknown, waiting on what
    that guard waits on, counted once at the highest such ``when`` (``_guard_waiting``,
    reading 8's second half: refused takes effect once all above it holds).

    A placed Space's piece is read from its guard first (the guard its placing scope
    places it under, a node of this Space's scope or above): where the guard is false or
    does not apply, the placed Space is absent and its validity answers inapplicable
    whatever its pieces are, so the piece is that answer, read from the guard alone, and
    the placed Space's validity is neither read nor resolved (an absent scope resolves no
    node for it). Elsewhere the placed Space's validity is read as it answers, after its
    guard. The trace holds what decides the piece either way: the guard, and the placed
    Space's validity where it was read."""
    linked = world.evaluator.linked
    nodes = world.evaluator.nodes
    answers: dict[str, QueryResult[bool]] = {}
    guards: set[NodeId] = set()
    for index in node.constraints:
        header = linked.header(index)
        placed: Entry | None = None
        if header.kind == "validity" and header.guard is not None:
            placed, reads = world.read(trace, header.guard, reads)
            held = placed.result
            if isinstance(held, Inapplicable):
                answers[header.key] = held  # what the placed Space's validity answers
                continue
            if isinstance(held, Available) and held.value is False:
                answers[header.key] = INAPPLICABLE
                continue
        piece = nodes[index]
        if piece.kind == "decision":
            found, reads = _choice_piece(world, piece, trace, reads, guards)
            if found is not None:
                answers[found[0]] = found[1]
            continue
        entry, reads = world.read(trace, index, reads)
        answer = cast(QueryResult[bool], entry.result)
        if isinstance(answer, Unresolved) and piece.guard is not None:
            if placed is None:
                guard, reads = world.read(trace, piece.guard, reads)
            else:
                guard = placed  # read above, first
            if isinstance(guard.result, Unresolved):
                top, reads = world.refused_waiting(piece.guard, trace, reads)
                if top is not None:
                    waiting, reads = _guard_waiting(world, piece, top, trace, reads, guards)
                    if waiting is not None:
                        answers[waiting[0]] = waiting[1]
                    continue
        if isinstance(answer, (Inapplicable, Rejected)) and piece.guard is not None:
            if placed is None:
                guard, reads = world.read(trace, piece.guard, reads)
            else:
                guard = placed  # read above, first
            if isinstance(guard.result, Rejected):
                if piece.kind == "validity":
                    owner = linked.scopes[piece.scope].name
                else:
                    owner = piece.owner
                counted, reads = _guard_refused(world, piece, owner, trace, reads, guards)
                if counted is not None:
                    answers[counted[0]] = counted[1]
                continue
        answers[piece.key] = answer
    assessment = assess_constraints(answers)
    return Entry(cast(QueryResult[object], assessment.result), reads, tuple(trace), assessment)


def _guard_refused(
    world: World,
    piece: Node,
    owner: str,
    trace: _Trace,
    reads: ReadSet,
    guards: set[NodeId],
) -> tuple[tuple[str, QueryResult[bool]] | None, ReadSet]:
    """The guard piece of ``piece``, whose guard is refused through an input: counted
    once, at the highest refused guard of its chain (reading 8: "at the highest such
    guard, scopes under it add nothing"), or ``None`` where that guard is counted
    already. Counted at a guard above its own, it is that guard's owner's, keyed by it
    where the owner is a node (a selector: its cases' pieces are its one piece)."""
    nodes = world.evaluator.nodes
    top = cast(NodeId, piece.guard)
    entry, reads = world.read(trace, top, reads)
    while (outer := nodes[top].guard) is not None:
        above, reads = world.read(trace, outer, reads)
        if not isinstance(above.result, Rejected):
            break
        top, entry = outer, above
    if top in guards:
        return None, reads
    guards.add(top)
    key = piece.key
    if top != piece.guard:
        owner = nodes[top].owner
        if owner in world.evaluator.linked.keys:
            key = owner
    return (key, _guard_piece(owner, cast(Rejected, entry.result))), reads


def _guard_waiting(
    world: World,
    piece: Node,
    top: NodeId,
    trace: _Trace,
    reads: ReadSet,
    guards: set[NodeId],
) -> tuple[tuple[str, QueryResult[bool]] | None, ReadSet]:
    """The piece of ``piece``, below ``top``, a ``when`` refused through an input below a
    guard that waits (``World.refused_waiting``): unknown, waiting on what the guard above
    ``top`` waits on, counted once at ``top`` and keyed as ``_guard_refused`` keys a refused
    guard's piece (by ``top``'s owner where it is a node and not ``piece``'s own guard);
    ``None`` where ``top`` is counted already."""
    if top in guards:
        return None, reads
    guards.add(top)
    nodes = world.evaluator.nodes
    key = piece.key
    if top != piece.guard:
        owner = nodes[top].owner
        if owner in world.evaluator.linked.keys:
            key = owner
    above, reads = world.read(trace, cast(NodeId, nodes[top].guard), reads)
    return (key, cast(QueryResult[bool], above.result)), reads


def _choice_piece(
    world: World, piece: Node, trace: _Trace, reads: ReadSet, guards: set[NodeId]
) -> tuple[tuple[str, QueryResult[bool]] | None, ReadSet]:
    """A choice's validity piece by key, or ``None`` where it is no piece: a guard
    refused through an input (``_guard_refused``), or refused below a guard that waits
    (``_guard_waiting``); else read from its listed values."""
    if piece.guard is not None:
        guard, reads = world.read(trace, piece.guard, reads)
        if isinstance(guard.result, Rejected):
            return _guard_refused(world, piece, piece.owner, trace, reads, guards)
        if isinstance(guard.result, Unresolved):
            top, reads = world.refused_waiting(piece.guard, trace, reads)
            if top is not None:
                return _guard_waiting(world, piece, top, trace, reads, guards)
    answer, reads = _listed_piece(world, piece, trace, reads)
    return (None if answer is None else (piece.key, answer)), reads


def _listed_piece(
    world: World, piece: Node, trace: _Trace, reads: ReadSet
) -> tuple[QueryResult[bool] | None, ReadSet]:
    """A choice's validity piece where its guard is not refused, read from its guard and
    its listed values (never from its stored value), or ``None`` where it is no piece. A
    domain that lists nothing (membership only) is never a piece: its inputs are not read,
    so neither their wait nor their refusal makes it one."""
    waiting: Unresolved | None = None
    if piece.guard is not None:
        guard, reads = world.read(trace, piece.guard, reads)
        answer = guard.result
        if isinstance(answer, Unresolved):
            waiting = answer
        elif _guard_result(piece, answer) is not None:
            return None, reads  # absent
    if piece.domain is None or piece.domain.candidates is None:
        return None, reads
    if waiting is not None:
        # Presence unknown: its list as if it existed; empty (or not known) it is an
        # unknown piece. What that list read is traced as it answered there, so the piece
        # is found again only where those reads answer the same.
        place = world.membership(piece.index)
        listed, found = _lists_a_value(place, piece, [], 0)
        bits = place.evaluator.bits
        trace.extend((source, place._get(source).result) for source in bits.nodes(found))
        reads |= found
        if isinstance(listed, Available):
            return (None if listed.value else waiting), reads
        if isinstance(listed, Unresolved):  # it waits on a missing fact too
            return Unresolved(merged_findings((waiting, listed))), reads
        return waiting, reads
    listed, reads = _lists_a_value(world, piece, trace, reads)
    if isinstance(listed, Available):
        return (None if listed.value else _empty_domain(piece)), reads
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
        # Two known present sources refuse at the formal: its own refusal, an origin.
        several = sum(1 for answer in answers if isinstance(answer, Available)) > 1
        return Entry(_present(node, answers), reads, tuple(trace), own_refusal=several)
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
    return Entry(
        called.result,
        reads,
        trace,
        called.assessment,
        called.own_refusal,
        called.via,
        called.existence,
    )


@dataclass(frozen=True, slots=True)
class _Returned:
    value: object


@dataclass(frozen=True, slots=True)
class _Halted:
    """A method halted by reads of non-values (its blocked reads' answers)."""

    results: tuple[NonValue, ...]


_ABANDONED = object()
# Never swallowed by an abandoned call: they fail it, keeping their identity.
_CANCELLATIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)


def _invoke(
    call: Call,
    function: Callable[..., object],
    arguments: tuple[object, ...],
    keywords: Mapping[str, object],
) -> object:
    """Run one authored function as ``call`` (the active call): its value, its halt, its
    failure (sticky: one it caught still counts), or abandoned by the depth guard."""
    token = ACTIVE.set(call)
    call.outer = sys.exception()
    try:
        try:
            value = function(*arguments, **keywords)
        except BaseException as cause:
            if call.abandoned is not None and not isinstance(cause, _CANCELLATIONS):
                return _ABANDONED
            failure = remember_exception(call, cause)
            if failure is not None:
                return failure
            if call.blocked:
                return _Halted(tuple(call.blocked.values()))
            # A value-unavailable error raised by hand, with no read behind it.
            error = EvaluationError(call.owner, call.role, str(cause))
            error.__cause__ = cause
            return remember_failure(call, Failure(error))
        if call.abandoned is not None:
            return _ABANDONED
        if call.failure is not None:
            return call.failure
        if call.blocked:
            return _Halted(tuple(call.blocked.values()))
        return _Returned(value)
    finally:
        ACTIVE.reset(token)


def _authored(
    world: World,
    call: Call,
    function: Callable[..., object],
    arguments: tuple[object, ...] = (),
    keywords: Mapping[str, object] | None = None,
) -> _Returned | _Halted:
    """Call an authored function as ``call``, in a copy of its driver's context: a failure
    goes on as ``Failing``, an abandoned call as the depth guard's ``Deeper``."""
    run = world.evaluator.run
    assert run is not None, "an authored call outside a run"
    outcome = run.context.copy().run(_invoke, call, function, arguments, keywords or {})
    if outcome is _ABANDONED:
        raise cast(BaseException, call.abandoned)
    if isinstance(outcome, Failure):
        raise Failing(outcome)
    return cast(_Returned | _Halted, outcome)


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
    key = (node.index, "call")
    called_on = _inputs(node.arguments, answers)
    held = _recall(world, key, called_on)
    if held is not None:
        evaluator.counters.hit(node.kind, node.index)
        if _PURITY:
            _again(node, node.kind, held.outcome.result, _call(world, node, answers).result)
        return _entry(held.outcome, reads, tuple(trace))
    evaluator.counters.call(node.kind, node.index)
    called = _call(world, node, answers)
    evaluator.memo.keep(key, called_on, called)
    return _entry(called, reads, tuple(trace))


def _call(world: World, node: Node, answers: Sequence[QueryResult[object]]) -> Called:
    """An explicit function called on its arguments' answers (each a value), detached."""
    nodes = world.evaluator.nodes
    arguments = {
        argument.name: _clone(
            nodes[argument.node],
            cast(Available[object], answer).value,
            owner=node.owner,
            role="dependency snapshot",
        )
        for argument, answer in zip(node.arguments, answers)
    }
    function = cast(Callable[..., object], node.function)
    outcome = _authored(world, Call(node.owner, node.kind), function, (), arguments)
    return _finished(node, cast(_Returned, outcome).value)


def _again(node: Node, role: str, held: QueryResult[object], found: QueryResult[object]) -> None:
    """The purity check: ``found``, a function's answer run again on the same reads, is
    ``held``, its answer before, by value semantics; else the function is impure."""
    if not _same_outcome(node, held, found):
        raise EvaluationError(
            node.owner,
            role,
            "answered differently when run again on the same reads: an authored function "
            "must answer from what it reads alone",
        )


def _same_outcome(node: Node, left: QueryResult[object], right: QueryResult[object]) -> bool:
    """Whether two answers of one function of ``node`` are the same: non-values with equal
    findings; values equal by the node's semantics (a domain's: each value of the choice
    by its semantics, a bool or nothing listed exactly; a constraint's, the bool)."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if not isinstance(left, Available):
        try:
            return bool(left.findings == cast(NonValue, right).findings)
        except Exception:  # noqa: BLE001 - findings without equality are never the same
            return False
    one, other = left.value, cast(Available[object], right).value
    if node.kind == "decision":
        if isinstance(one, tuple) and isinstance(other, tuple):
            return len(one) == len(other) and all(
                _same_value(node, a, b) for a, b in zip(one, other)
            )
        return type(one) is type(other) and one == other
    if node.kind == "constraint":
        return one is other
    return _same_value(node, one, other)


def _inputs(arguments: Sequence[Argument], answers: Sequence[QueryResult[object]]) -> Trace:
    """An explicit function's trace: each argument's node, with the answer it got."""
    return tuple(zip((argument.node for argument in arguments), answers))


def _recall(world: World, key: Key, trace: Trace) -> Kept[Called] | None:
    """An outcome the memo keeps for ``key``'s function whose trace answers as ``trace``:
    every read the same answer object (tried first), else equal by its node's value
    semantics."""
    memo = world.evaluator.memo
    kept = memo.kept(key)
    found = next((each for each in kept if _identical(each.trace, trace)), None)
    if found is None:
        found = next((each for each in kept if _alike(world, each.trace, trace)), None)
    if found is not None:
        memo.used(found)
    return found


def _identical(kept: Trace, trace: Trace) -> bool:
    return len(kept) == len(trace) and all(
        node == other and answer is found for (node, answer), (other, found) in zip(kept, trace)
    )


def _alike(world: World, kept: Trace, trace: Trace) -> bool:
    return len(kept) == len(trace) and all(
        node == other and world.same(node, answer, found)
        for (node, answer), (other, found) in zip(kept, trace)
    )


def _finished(node: Node, returned: object) -> Called:
    """``_finish``, where a returned value that fails as an answer is the owner's error."""
    try:
        return _finish(node, returned)
    except EvaluationError:
        raise
    except Exception as cause:
        raise EvaluationError(node.owner, node.kind, str(cause)) from cause


class MethodReader:
    """The reads of one running ``self`` method, on the world it runs in.

    A method's receiver (``receiver``) reads through it: each read is evaluated in the
    world and recorded, in order, with its answer; a read of a non-value is kept on the
    call (``Call.blocked``) and halts the method. Every read is checked against what the
    method's source may read (``_reads``, always on: an undeclared read is the method's
    failure), and is refused outside the method's own call (another computation's
    receiver, a value transformation). A read of a forwarding alias goes to its source
    where the alias applies whenever the method does, recorded in ``via``. What it reads
    only to know whether something exists (``existence``: a selector, for its candidate;
    a guard or a source's presence, for ``present``) is told apart from what it reads as
    a value (``valued``)."""

    __slots__ = ("world", "node", "call", "trace", "reads", "via", "existence", "valued")

    def __init__(self, world: World, node: Node, call: Call) -> None:
        self.world = world
        self.node = node
        self.call = call
        self.trace: list[tuple[NodeId, QueryResult[object]]] = []
        self.reads: ReadSet = 0
        self.via: list[tuple[NodeId, NodeId]] = []
        self.existence: set[NodeId] = set()
        self.valued: set[NodeId] = set()

    # -- the read boundary -------------------------------------------------------------

    def _enter(self) -> None:
        """Any read through this reader: only from its own call, never in a value
        transformation; an abandoned call reads nothing more."""
        check_transformation()
        active = current()
        if active is not self.call:
            if active is None:
                raise EvaluationError(
                    self.node.owner, "field read", "a method's receiver is read after it returned"
                )
            error = EvaluationError(active.owner, "field read", "cross-configuration access")
            fault(active, error)
            raise error
        if self.call.abandoned is not None:
            raise self.call.abandoned

    def _check(self, index: NodeId) -> None:
        """A read the method's source does not name is its failure."""
        allowed = self.world.evaluator.reads.methods.get(self.node.index)
        if allowed is None or index in allowed:
            return
        member = self.world.evaluator.nodes[index].key
        error = EvaluationError(
            self.node.owner,
            "read-undeclared",
            f"read {member}, which its source does not name: read it through a path from "
            "its receiver, or declare it in __reads__",
        )
        fault(self.call, error)
        raise error

    def _take(self, index: NodeId) -> Entry:
        """``index``'s entry, recorded. A failure met is the call's (its primary, or a
        cleanup failure) and is raised in the method; the depth guard abandons it."""
        try:
            entry = self.world._get(index)
        except Deeper as deeper:
            self.call.abandoned = deeper
            raise
        except Failing as failing:
            # Interrupted here, the failure still reaches the call: it is the context of
            # what interrupts (``remember_exception`` follows it).
            remember_failure(self.call, failing.failure)
            raise exception(failing.failure)
        self.trace.append((index, entry.result))
        self.reads |= entry.reads
        return entry

    def _answer(self, index: NodeId) -> QueryResult[object]:
        """``index``'s answer, checked and recorded, read to know whether something exists;
        a non-value does not halt."""
        self._check(index)
        self.existence.add(index)
        return self._take(index).result

    def _value(self, index: NodeId) -> object:
        """``index``'s value, recorded and detached; a non-value halts the method (in
        cleanup, after a failure, it is a cleanup read)."""
        answer = self._take(index).result
        node = self.world.evaluator.nodes[index]
        if isinstance(answer, Available):
            return public_value(node, answer)
        self.call.blocked[index] = answer
        if self.call.failure is not None:
            self.call.failure.add(CleanupFailure(node.owner, "cleanup read", result=answer))
        raise ValueUnavailableError(answer, context=node.owner)

    # -- what a receiver reads ---------------------------------------------------------

    def read_value(self, point: Space, reference: object) -> object:
        """A member read as a value (``self.x``). Whatever it raises is the call's."""
        try:
            return self._read_value(point, reference)
        except BaseException as cause:
            if current() is self.call and not isinstance(cause, Deeper):
                remember_exception(self.call, cause)
            raise

    def _read_value(self, point: Space, reference: object) -> object:
        self._enter()
        evaluator = self.world.evaluator
        index = evaluator.model.resolve(point._scope, reference)
        self._check(index)
        linked = evaluator.linked
        source = linked.forward[index]
        if source != index and guard_implies(
            linked.nodes, linked.nodes[index].guard, self.node.guard
        ):
            self.via.append((index, source))
            index = source
        self.valued.add(index)
        return self._value(index)

    def present(self, point: Space, node: object) -> bool:
        """Whether a node is present, or a value input supplied (``occurrence.present``)."""
        self._enter()
        target = self._node_scope(point, node)
        if target is None:
            if isinstance(node, Param):
                return self._supplied(point, node)
            raise RequestError("present() takes a node (a child or a candidate) or a value input")
        presence = self.world.evaluator.linked.scopes[target].guard
        if presence is None:
            return True
        answer = self._answer(presence)
        if isinstance(answer, Available):
            return answer.value is True
        if isinstance(answer, Inapplicable) or (
            isinstance(answer, Unresolved) and _unsupplied(answer)
        ):
            return False
        return bool(self._value(presence))  # undecided: halts like any value read

    def _node_scope(self, point: Space, reference: object) -> int | None:
        """For a node reference, the node's scope; ``None`` for anything else."""
        if not isinstance(reference, Space):
            return None
        path = declared_path(reference)
        if path is None:
            return None
        *outer, last = path
        if not isinstance(last, NodeDecl):
            return None
        scopes = self.world.evaluator.linked.scopes
        holder = descend(scopes, point._scope, outer)
        scope = scopes[holder]
        if last is scope.record:
            return holder
        target = scope.children.get(last)
        if target is None:
            raise RequestError("node is not part of this compiled scope")
        return target

    def _supplied(self, point: Space, formal: object) -> bool:
        """Whether a value input is supplied: a source applies (never its value)."""
        index = self.world.evaluator.model.resolve(point._scope, formal)
        applies = self._source_applies(index)
        if isinstance(applies, bool):
            return applies
        self._value(applies)  # undecided: halts like any value read
        raise AssertionError("an undecided presence read halts")

    def _source_applies(self, index: NodeId) -> bool | NodeId:
        """Whether a source applies at ``index``, or the presence node still undecided."""
        node = self.world.evaluator.nodes[index]
        if node.guard is not None:
            holds = self._holds(node.guard)
            if holds is not True:
                return holds
        if node.kind == "param":
            return isinstance(self._answer(index), Available)  # given at start
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
            if any(answer is True for answer in answers):  # a node may be 1
                return True
            return next((answer for answer in answers if answer is not False), False)
        return True  # a const, a decision, a computation: the source itself

    def _holds(self, guard: NodeId) -> bool | NodeId:
        """Whether a guard holds (inapplicable: it does not), or the guard while undecided."""
        answer = self._answer(guard)
        if isinstance(answer, Available):
            return answer.value is True
        return False if isinstance(answer, Inapplicable) else guard

    def child(self, point: Space, record: object) -> Space:
        """A child node's receiver."""
        self._enter()
        scope = self.world.evaluator.linked.scopes[point._scope]
        child_scope = scope.children.get(record)
        if child_scope is None:
            raise RequestError("child node is not part of this compiled scope")
        return receiver(self, child_scope)

    def _choice(self, scope: int, decision: object) -> Choice:
        linked = self.world.evaluator.linked
        try:
            return linked.choices[linked.scopes[scope].choices[unwrap(decision)]]
        except (KeyError, TypeError) as error:
            raise RequestError(
                "the Decision over nodes is not part of this compiled scope"
            ) from error

    def selected_candidate(self, point: Space, decision: object) -> Space | None:
        """The receiver of the candidate a Decision over nodes selects, or ``None``."""
        self._enter()
        choice = self._choice(point._scope, decision)
        self._check(choice.selector)
        self.existence.add(choice.selector)
        case = self._value(choice.selector)
        for key, candidate in choice.cases:
            if key == case:
                return None if candidate is None else receiver(self, candidate)
        raise EvaluationError(choice.key, "selection", "selector is not a declared case")

    def candidate(self, point: Space, decision: object, case: str) -> Space | None:
        """A candidate's receiver, selected or not (``None`` for a case placing nothing)."""
        self._enter()
        if type(case) is not str:
            raise RequestError("a candidate key must be a string")
        choice = self._choice(point._scope, decision)
        for key, index in choice.cases:
            if key == case:
                return None if index is None else receiver(self, index)
        raise RequestError(f"{choice.key}: unknown candidate {case!r}")

    def root(self, point: Space) -> Space:
        """The root's receiver."""
        self._enter()
        return point if point._scope == 0 else receiver(self, 0)


def _unsupplied(answer: Unresolved) -> bool:
    findings = answer.findings
    return bool(findings) and all(finding.code == "input-unsupplied" for finding in findings)


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
    outcome without a call (a read that fails here leaves it to the call)."""
    evaluator = world.evaluator
    key = (node.index, "call")
    for kept in evaluator.memo.kept(key):
        more = 0
        try:
            for dependency, answer in kept.trace:
                found = world._get(dependency)
                if not world.same(dependency, found.result, answer):
                    break
                more |= found.reads
            else:
                evaluator.memo.used(kept)
                evaluator.counters.hit(node.kind, node.index)
                if _PURITY:
                    _again(
                        node, node.kind, kept.outcome.result, _call_method(world, node)[0].result
                    )
                return _entry(kept.outcome, reads | more, (*trace, *kept.trace))
        except Failing:
            break
    evaluator.counters.call(node.kind, node.index)
    called, reader = _call_method(world, node)
    if _PURITY and (world, node.index) in cast(_Run, evaluator.run).abandoned:
        _again(node, node.kind, called.result, _call_method(world, node)[0].result)
    recorded = tuple(reader.trace)
    evaluator.memo.keep(key, recorded, called)
    return _entry(called, reads | reader.reads, (*trace, *recorded))


def _call_method(world: World, node: Node) -> tuple[Called, MethodReader]:
    """A ``self`` method called on a receiver reading through ``world``: what it came to,
    and its reader (what it read). Abandoned by the depth guard, it is noted in its run."""
    call = Call(node.owner, node.kind)
    reader = MethodReader(world, node, call)
    try:
        outcome = _authored(
            world, call, cast(Callable[..., object], node.function), (receiver(reader, node.scope),)
        )
    except Deeper:
        cast(_Run, world.evaluator.run).abandoned.add((world, node.index))
        raise
    via = tuple(reader.via)
    existence = frozenset(reader.existence - reader.valued)
    if isinstance(outcome, _Halted):
        blocked = cast(NonValue, _blocked(outcome.results))
        called = Called(blocked, _guard_assessment(node, blocked), via=via, existence=existence)
        return called, reader
    finished = _finished(node, outcome.value)
    called = Called(finished.result, finished.assessment, finished.own_refusal, via, existence)
    return called, reader


# -- domains ----------------------------------------------------------------------------


def membership_arguments(node: Node) -> tuple[Argument, ...]:
    """A Decision's domain inputs its membership and enumeration read (an order's own
    inputs aside)."""
    domain = node.domain
    if domain is None:
        return ()
    named = set(domain.membership_inputs)
    return tuple(argument for argument in node.domain_arguments if argument.name in named)


def order_arguments(node: Node) -> tuple[Argument, ...]:
    """A Decision's domain inputs its order reads."""
    domain = node.domain
    if domain is None:
        return ()
    named = set(domain.order_inputs)
    return tuple(argument for argument in node.domain_arguments if argument.name in named)


def _domain_inputs(
    world: World, node: Node, trace: _Trace, reads: ReadSet
) -> tuple[tuple[QueryResult[object], ...], NonValue | None, ReadSet]:
    """The answers of a Decision's membership inputs, read in ``world`` (and traced), and
    the non-value that keeps the domain from being asked. An input that does not apply
    where the choice is present cannot decide it, whatever the other inputs answer: the
    domain is refused at the choice, naming it."""
    arguments = membership_arguments(node)
    answers: list[QueryResult[object]] = []
    for argument in arguments:
        entry, reads = world.read(trace, argument.node, reads)
        answers.append(entry.result)
    nodes = world.evaluator.nodes
    absent = [
        nodes[argument.node].owner
        for argument, answer in zip(arguments, answers)
        if isinstance(answer, Inapplicable)
    ]
    if absent:
        return tuple(answers), _inputs_absent(node, absent), reads
    blocked = _blocked([answer for answer in answers if not isinstance(answer, Available)])
    return tuple(answers), blocked, reads


def _domain_values(
    world: World,
    node: Node,
    answers: tuple[QueryResult[object], ...],
    arguments: tuple[Argument, ...] | None = None,
) -> dict[str, object]:
    """The values of a Decision's domain inputs (its membership inputs, unless
    ``arguments`` names others), detached, by name."""
    nodes = world.evaluator.nodes
    return {
        argument.name: _clone(
            nodes[argument.node],
            cast(Available[object], answer).value,
            owner=node.owner,
            role="dependency snapshot",
        )
        for argument, answer in zip(
            membership_arguments(node) if arguments is None else arguments, answers
        )
    }


def _type_check(
    world: World, node: Node, value: Available[object], trace: _Trace, reads: ReadSet
) -> tuple[QueryResult[object], ReadSet]:
    """Whether a choice's domain holds ``value`` here (memoized on the value's answer and
    the domain inputs' answers)."""
    domain, semantics = node.domain, node.semantics
    assert domain is not None and semantics is not None
    answers, blocked, reads = _domain_inputs(world, node, trace, reads)
    if blocked is not None:
        return blocked, reads

    def membership() -> QueryResult[object]:
        candidate = _clone(node, value.value, owner=node.owner, role="domain candidate snapshot")
        outcome = _authored(
            world,
            Call(node.owner, "domain membership"),
            domain.membership,
            (candidate, _domain_values(world, node, answers)),
            {"semantics": semantics, "owner": node.owner},
        )
        return cast(QueryResult[object], cast(_Returned, outcome).value)

    called_on = ((node.index, value), *_inputs(membership_arguments(node), answers))
    return _memoized(world, node, "membership", called_on, membership), reads


def _memoized(
    world: World,
    node: Node,
    purpose: str,
    called_on: Trace,
    compute: Callable[[], QueryResult[object]],
) -> QueryResult[object]:
    """What a Decision's domain answers for ``purpose`` (its membership, listing,
    enumeration or order) on the reads ``called_on``: kept in the model's memo, or
    computed and kept. Under the purity check a kept answer is computed again and
    compared."""
    evaluator = world.evaluator
    kind = _DOMAIN_KINDS[purpose]
    key = (node.index, purpose)
    held = _recall(world, key, called_on)
    if held is not None:
        evaluator.counters.hit(kind, node.index)
        if _PURITY:
            _again(node, kind, held.outcome.result, compute())
        return held.outcome.result
    evaluator.counters.call(kind, node.index)
    result = compute()
    evaluator.memo.keep(key, called_on, Called(result))
    return result


# What each domain purpose is counted as.
_DOMAIN_KINDS = {
    "membership": "domain membership",
    "listing": "domain listing",
    "lists a value": "domain listing",
    "enumeration": "domain enumeration",
    "order": "domain order",
}


def _listed_members(
    domain: Domain[object],
    values: tuple[object, ...],
    arguments: Mapping[str, object],
    *,
    semantics: ValueSemantics[object],
    owner: str,
) -> tuple[object, ...]:
    """The listed values a domain's membership does not refuse, in list order."""
    return tuple(
        value
        for value in values
        if not isinstance(
            domain.membership(value, arguments, semantics=semantics, owner=owner), Rejected
        )
    )


def _listed(
    world: World, node: Node, trace: _Trace, reads: ReadSet
) -> tuple[QueryResult[object], ReadSet]:
    """A choice's listed values here (``Available(None)``: membership only, nothing is
    listed): its domain's enumeration, each value checked by membership (memoized on the
    domain inputs' answers)."""
    domain = node.domain
    assert domain is not None
    answers, blocked, reads = _domain_inputs(world, node, trace, reads)
    if blocked is not None:
        return blocked, reads
    if domain.candidates is None:
        return Available(None), reads
    called_on = _inputs(membership_arguments(node), answers)
    return _memoized(world, node, "listing", called_on, lambda: _list(world, node, answers)), reads


def _list(
    world: World, node: Node, answers: tuple[QueryResult[object], ...]
) -> QueryResult[object]:
    """A domain's listed values on its inputs' answers (each available)."""
    domain, semantics = node.domain, node.semantics
    assert domain is not None and semantics is not None
    listed = _enumerate(world, node, answers)
    if isinstance(listed, Available) and domain._finite_values is None:
        kept = _authored(
            world,
            Call(node.owner, "domain membership"),
            _listed_members,
            (domain, listed.value, _domain_values(world, node, answers)),
            {"semantics": semantics, "owner": node.owner},
        )
        listed = Available(cast(_Returned, kept).value)
    return listed


def _first_member(
    domain: Domain[object],
    values: tuple[object, ...],
    arguments: Mapping[str, object],
    *,
    semantics: ValueSemantics[object],
    owner: str,
) -> bool:
    """Whether a domain's membership holds any of ``values``, asked in list order up to
    the first it holds."""
    return any(
        not isinstance(
            domain.membership(value, arguments, semantics=semantics, owner=owner), Rejected
        )
        for value in values
    )


def _lists_a_value(
    world: World, node: Node, trace: _Trace, reads: ReadSet
) -> tuple[QueryResult[object], ReadSet]:
    """Whether a choice's listed domain lists a value here: ``Available(True)`` or
    ``Available(False)``, or the non-value its inputs answer. Membership is asked up to
    the first value it holds (the full list, where it was already made, answers at once);
    memoized on the domain inputs' answers."""
    domain = node.domain
    assert domain is not None and domain.candidates is not None
    answers, blocked, reads = _domain_inputs(world, node, trace, reads)
    if blocked is not None:
        return blocked, reads
    called_on = _inputs(membership_arguments(node), answers)
    if _recall(world, (node.index, "listing"), called_on) is not None:
        found = _memoized(world, node, "listing", called_on, lambda: _list(world, node, answers))
        if isinstance(found, Available):
            found = Available(found.value != ())
        return found, reads
    found = _memoized(world, node, "lists a value", called_on, lambda: _lists(world, node, answers))
    return found, reads


def _lists(
    world: World, node: Node, answers: tuple[QueryResult[object], ...]
) -> QueryResult[object]:
    """Whether a domain lists a value on its inputs' answers (each available)."""
    domain, semantics = node.domain, node.semantics
    assert domain is not None and semantics is not None
    enumerated = _enumerate(world, node, answers)
    if not isinstance(enumerated, Available):
        return enumerated
    values = cast(tuple[object, ...], enumerated.value)
    if domain._finite_values is not None or not values:
        return Available(bool(values))
    kept = _authored(
        world,
        Call(node.owner, "domain membership"),
        _first_member,
        (domain, values, _domain_values(world, node, answers)),
        {"semantics": semantics, "owner": node.owner},
    )
    return Available(cast(_Returned, kept).value)


def listed_values(world: World, node: NodeId) -> QueryResult[object]:
    """A choice's listed values in ``world`` (``Available(None)``: membership only), from
    a driver."""
    spec = world.evaluator.nodes[node]
    return _drive(world.evaluator, lambda: _listed(world, spec, [], 0)[0])


def type_check(world: World, node: NodeId, value: Available[object]) -> QueryResult[object]:
    """Whether a choice's domain holds ``value``, its inputs read in ``world``, from a
    driver: ``Available(True)``; refused outside the domain, or through an input refusing
    or not applying; unresolved while an input waits (a missing fact)."""
    spec = world.evaluator.nodes[node]
    if spec.domain is None:
        return Available(True)
    return _drive(world.evaluator, lambda: _type_check(world, spec, value, [], 0)[0])


def _candidates(world: World, node: Node) -> QueryResult[object]:
    """A choice's candidates: its listed values (its domain's enumeration, each value its
    membership accepts), ranked by its order where it has one (``Available(None)``:
    membership only), under its guard: inapplicable where the choice is absent, the
    guard's answer while it waits or is refused; the order's inputs' non-value while one
    of them is not a value (memoized on the inputs' answers)."""
    domain, semantics = node.domain, node.semantics
    if domain is None or semantics is None:
        raise EvaluationError(node.owner, "domain enumeration", "reference has no domain")
    if node.guard is not None:
        inactive = _guard_result(node, world._get(node.guard).result)
        if inactive is not None:
            return inactive
    answers, blocked, _ = _domain_inputs(world, node, [], 0)
    if blocked is not None:
        return blocked
    if domain.candidates is None:
        return Available(None)
    called_on = _inputs(membership_arguments(node), answers)
    listed = _memoized(world, node, "listing", called_on, lambda: _list(world, node, answers))
    if domain.order is None or not isinstance(listed, Available):
        return listed
    return _ranked(world, node, answers, listed)


def _ranked(
    world: World,
    node: Node,
    inputs: tuple[QueryResult[object], ...],
    listed: Available[object],
) -> QueryResult[object]:
    """Listed values (found from the membership inputs' answers ``inputs``) as the
    domain's order ranks them, over the order's own inputs (memoized on both inputs'
    answers)."""
    domain, semantics = node.domain, node.semantics
    assert domain is not None and semantics is not None
    arguments = order_arguments(node)
    answers = tuple(world._get(argument.node).result for argument in arguments)
    blocked = _blocked([answer for answer in answers if not isinstance(answer, Available)])
    if blocked is not None:
        return blocked

    def rank() -> QueryResult[object]:
        ranked = _authored(
            world,
            Call(node.owner, "domain order"),
            domain.rank,
            (listed.value, _domain_values(world, node, answers, arguments)),
            {"semantics": semantics, "owner": node.owner},
        )
        return Available(cast(_Returned, ranked).value)

    called_on = (*_inputs(membership_arguments(node), inputs), *_inputs(arguments, answers))
    return _memoized(world, node, "order", called_on, rank)


def _enumerate(
    world: World, node: Node, answers: tuple[QueryResult[object], ...]
) -> QueryResult[object]:
    """A domain's enumeration on its inputs' answers (each available), memoized on them:
    what the listing filters by membership."""
    domain, semantics = node.domain, node.semantics
    assert domain is not None and semantics is not None

    def enumerate_() -> QueryResult[object]:
        enumerated = _authored(
            world,
            Call(node.owner, "domain enumeration"),
            domain.enumerate,
            (_domain_values(world, node, answers),),
            {"semantics": semantics, "owner": node.owner},
        )
        return cast(QueryResult[object], cast(_Returned, enumerated).value)

    called_on = _inputs(membership_arguments(node), answers)
    return _memoized(world, node, "enumeration", called_on, enumerate_)


def candidates(world: World, node: NodeId) -> QueryResult[object]:
    """A choice's candidates in ``world`` (``Available(None)``: membership only), from a
    driver: its listed values, in its domain's order."""
    spec = world.evaluator.nodes[node]
    return _drive(world.evaluator, lambda: _candidates(world, spec))


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
    "DEPTH",
    "FLATTEN",
    "INAPPLICABLE",
    "Called",
    "Entry",
    "Evaluator",
    "MethodReader",
    "World",
    "candidates",
    "check_purity",
    "listed_values",
    "membership_arguments",
    "order_arguments",
    "public_value",
    "receiver",
    "type_check",
]
