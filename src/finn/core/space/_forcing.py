# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Forced Decisions: an open Decision with exactly one viable case reads as that case.

A forced Decision is derived at read time and never stored. It stays
unassigned (``DecisionState`` has no other status), it is not among a
configuration's commitments, and each snapshot finds its own. The public reads are
in ``inspection``: ``forced`` and ``viable`` report the verdicts, ``validity``
a Space's answer.

A case of an open Decision is **viable** (possible) unless a constraint that reads the
Decision refuses it, read in the configuration plus the case (a **trial**), and, for a
Decision over nodes, unless the candidate it places is refused, its validity read in
the same trial (a Space refuses as soon as one of its pieces does, whatever else it
still waits on). The constraints are every one whose evaluation may read the Decision,
wherever it is owned: its own Space's, a placed Space's, an ancestor's, or one that
reads it through a formal (``_rules``); a selector's are those outside its candidates.
A constraint or a validity still waiting on an open choice, with nothing refused, does
not refuse, and a Decision over values none of the constraints reads costs no trial.
One viable case forces the Decision, several leave it open, and none refuses it
(``decision-no-viable-case``, each case with its reason): the same for both kinds.

A snapshot finds its forced Decisions once, on the first read of an open
Decision: rounds over the open Decisions in rank order, each forced value
visible to the next (an adapter after the core it feeds is found in the same
round), evaluated on copies that do not force themselves. A successor of a base
whose forced values were found starts its rounds from them (``_find``): the base's
forced values, replayed in the order the base found them, each held only while its
verdict's reads hold, are assigned on the first copy, and the rounds then find only
what the change reaches; a verdict is held only while a read of each Decision it read
answers what it did (the value, open, or unresolved as an unchecked choice, which is never
assigned); the result is kept only where it is a derivation of the
plain rounds' forcing, and otherwise the plain rounds run. The first copy starts
from the snapshot's evaluations that read no open Decision (but a seed). Each copy (one per
forced value, one per case of a Decision over nodes) extends the one before it by
one Decision and starts from that one's evaluations that did not read it, so a
case's trial re-derives only what the case reaches. The snapshot then keeps what
the last copy evaluated that read no refused Decision: there, every other open
Decision reads as it does on the snapshot. An evaluation records every Decision it
read (``Evaluation.decisions``), so each of these is one pass: found, in a large
cache, from the readers of the Decisions a copy adds and not by looking at every
evaluation (``_runtime.inherit``), and what the last copy evaluated is all the snapshot
gains. A trial reads its
base's forced values for the Decisions it does not change; where the base
forces nothing, the forced values of the configuration it would publish, found
once and published with it, so a batch may commit a choice nested under a
selector that another choice of the batch forces (``_runtime``). Each
Decision's **verdict** (its viable cases and why
the others are not) records the Decisions it read and the value it saw of
each, and the formals it read, and is reused while none of them changed: within the
rounds, and by a successor from its base, so a change re-checks only the verdicts it
reaches. A configuration with other values for a formal (``try_with_facts``) finds
again the verdicts that read it. A forced Decision whose verdict read a Decision the
rounds forced after it (one a constraint reading the first reads too, such as a choice
inside the candidate it forces) keeps that verdict as its step of the derivation, and is
given the verdict found again under the final values (``Verdict.settled``), which the
final check and a change's reach read.

A selector's verdict rests on its candidates' validity, read on copies where an open
Decision reads as unassigned and a stored one as its value: there a Decision's piece
refuses only through its guard, so the verdict records the reads of the constraints,
the guards and the placed Spaces' validity, not the Decisions inside the candidate
(``_refusal_roots``), and a choice committed inside a candidate does not reach it unless
a constraint reading the selector reads it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from finn.core.reporting.findings import Finding, FindingKind

from . import _runtime
from ._runtime import Snapshot
from .ir import Choice, LinkedModel
from .results import Available, ConstraintAssessment, QueryResult, Rejected
from .semantics import equal

if TYPE_CHECKING:
    from ._configuration import Space
    from .compiler import Model


class _Reading:
    """What a read of a Decision answered, where it answered no value: the Decision was
    open (``OPEN``), or stored and unresolved because its own check cannot conclude
    (``UNCHECKED``). A read answers its value, or one of these: a verdict records that, not
    the stored candidate, so a candidate that stays equal while its check concludes is a
    changed answer."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


OPEN = _Reading("OPEN")
UNCHECKED = _Reading("UNCHECKED")
_BUSY = object()


@dataclass(frozen=True)
class Verdict:
    """A Decision's viable cases (None: not applicable, or not enumerable here), why
    each other case is not viable, and the Decisions read with the answer seen (its
    value, ``OPEN`` or ``UNCHECKED``), and the formals read (``facts``). ``membership``:
    applicable, and known by its domain's membership only.

    ``settled``: for a forced Decision whose verdict read a Decision still open that the
    rounds forced later (one that a constraint reading it reads too, such as a choice
    inside the candidate it forces), the verdict found again under the rounds' final values with the
    Decision itself open, which forces the same case. This verdict stays the step of the
    derivation a successor replays; the settled one is what the final values give, which
    the rounds' final check and a change's reach read (``_settle``)."""

    cases: tuple[object, ...] | None
    reasons: Mapping[str, str]
    reads: Mapping[int, object]
    membership: bool = False
    # The formals it read: a configuration with other values for any of them finds it again.
    facts: frozenset[int] = frozenset()
    settled: Verdict | None = None

    @property
    def every_fact(self) -> frozenset[int]:
        """The formals this verdict, or its settled one, read."""
        return self.facts if self.settled is None else self.facts | self.settled.facts


@dataclass(frozen=True)
class Found:
    """A snapshot's forced values (in the order they were found, which a successor replays)
    and refused Decisions by node, and every verdict found so far (a successor's to reuse)."""

    values: Mapping[int, object] = field(default_factory=dict)
    refused: Mapping[int, Rejected] = field(default_factory=dict)
    verdicts: Mapping[int, Verdict] = field(default_factory=dict)
    # Who reads whom among the snapshot's open Decisions, and who reads which formal (both
    # by Decision), and which are unsettled: built on the first ``reach``.
    _graph: list[
        tuple[Mapping[int, frozenset[int]], Mapping[int, frozenset[int]], frozenset[int]]
    ] = field(default_factory=list, init=False, repr=False, compare=False)

    def reach(
        self, base: Snapshot, seeds: Collection[int], facts: Collection[int] = ()
    ) -> set[int]:
        """The Decisions open in ``base`` (the snapshot these were found on) that a change
        to ``seeds``, or to other values of the formals ``facts``, reaches: those whose
        verdict read a seed or a changed formal, or read one of these, to a fixed point;
        and those with no verdict, or whose verdict does not hold under ``base``'s own
        final values. Such a verdict records a point in the rounds (a Decision it read
        was still open and forced later), not the answer that stands, so what it
        decided is not known to be stable."""
        if not self._graph:
            self._graph.append(self._reads_by(base))
        readers, formal_readers, unsettled = self._graph[0]
        reached = set(unsettled)
        pending = [*seeds, *unsettled]
        for fact in facts:
            for reader in formal_readers.get(fact, ()):
                if reader not in reached:
                    reached.add(reader)
                    pending.append(reader)
        while pending:
            for reader in readers.get(pending.pop(), ()):
                if reader not in reached:
                    reached.add(reader)
                    pending.append(reader)
        return reached

    def _reads_by(
        self, base: Snapshot
    ) -> tuple[Mapping[int, frozenset[int]], Mapping[int, frozenset[int]], frozenset[int]]:
        """The open Decisions whose verdicts read each Decision, those whose verdicts read
        each formal, and the open Decisions whose verdicts are not known to hold
        (``reach``)."""
        linked = base.linked
        final = {**base.assignments, **self.values}
        unchecked = base.unchecked
        readers: dict[int, set[int]] = {}
        formal_readers: dict[int, set[int]] = {}
        unsettled: set[int] = set()
        for index in linked.decisions:
            if index in base.assignments:
                continue
            verdict = _holding(linked, self.verdicts.get(index), final, unchecked)
            if verdict is None:
                unsettled.add(index)
                continue
            for read in verdict.reads:
                readers.setdefault(read, set()).add(index)
            for fact in verdict.facts:
                formal_readers.setdefault(fact, set()).add(index)
        return (
            {read: frozenset(each) for read, each in readers.items()},
            {fact: frozenset(each) for fact, each in formal_readers.items()},
            frozenset(unsettled),
        )


NOTHING = Found()


@dataclass(frozen=True)
class Viable:
    """An open Decision, for inspection: neither committed nor forced, its viable cases
    (several, or none when the Decision is refused) and why each other case is not."""

    key: str
    cases: tuple[object, ...]
    refused: Mapping[str, str]


@dataclass(frozen=True)
class Open:
    """An open Decision whose cases are not enumerable, for inspection: known by its
    domain's membership only (a FIFO's depth), and whether that domain is ordered."""

    key: str
    ordered: bool


@dataclass(frozen=True)
class Forced:
    """A forced Decision, for inspection: its value, and why every other case is not
    viable (empty for a Decision over values whose domain has one case)."""

    key: str
    value: object
    refused: Mapping[str, str]


def forced(snapshot: Snapshot) -> Found:
    """The snapshot's forced Decisions, found once."""
    with snapshot.lock:
        found = snapshot.found
        if isinstance(found, Found):
            return found
        if found is _BUSY:
            return NOTHING
        object.__setattr__(snapshot, "found", _BUSY)
        try:
            result = _find(snapshot)
        except BaseException:
            object.__setattr__(snapshot, "found", None)
            raise
        object.__setattr__(snapshot, "found", result)
        return result


def inherited(base: Snapshot) -> Mapping[int, object]:
    """The verdicts a successor of ``base`` starts from: those ``base`` found, or else
    the ones it inherited itself."""
    found = base.found
    return found.verdicts if isinstance(found, Found) else base.verdicts


def carried(base: Snapshot) -> Mapping[int, object]:
    """The forced values a successor of ``base`` replays as seeds, in the order ``base``
    found them: none where ``base``'s were never found (nothing is cached to carry)."""
    found = base.found
    return found.values if isinstance(found, Found) else {}


def validity(snapshot: Snapshot, scope: int) -> tuple[QueryResult[object], int]:
    """A Space's validity read in ``snapshot``, and the node that answers it."""
    index = snapshot.linked.scopes[scope].validity
    assert index is not None, "every scope has a validity node"
    return _runtime.evaluate(snapshot, index).result, index


def _rules(model: Model[Space], index: int) -> tuple[int, ...]:
    """The constraints that may read the Decision ``index`` in some configuration, wherever
    they are owned: those a case of it is tried against. For a Decision over nodes, not
    those of its candidates (the candidate's validity counts them, and a candidate not
    selected leaves them inactive). Found once per model and Decision and kept with the
    model (``_may_read``)."""
    key = ("rules", index)
    found = model._read.get(key)
    if found is None:
        linked = model.linked
        rules = _may_read(model).get(index, ())
        choice = linked.selector_choices.get(index)
        if choice is not None:
            inside = _below(linked, [scope for _, scope in linked.choices[choice].cases])
            rules = tuple(rule for rule in rules if linked.nodes[rule].scope not in inside)
        found = model._read[key] = rules
    return cast(tuple[int, ...], found)


def _below(linked: LinkedModel, scopes: Iterable[int | None]) -> set[int]:
    """``scopes`` (None aside) and every scope placed under one of them, at any depth."""
    below = {scope for scope in scopes if scope is not None}
    for item in linked.scopes:  # a scope is allocated after its parent
        if item.parent in below:
            below.add(item.index)
    return below


def _may_read(model: Model[Space]) -> Mapping[int, tuple[int, ...]]:
    """For each Decision, the constraints whose evaluation may read it in some
    configuration: those from which it is reached through the nodes' structural edges
    (``Node.dependencies``; a Decision's read reads its guard only), where a callback
    called on its Space (``self``) may read any node of its Space and of every Space
    below it. Found once per model and kept with it.

    A constraint that did not read a Decision in one configuration may read it in another
    (it waited on another choice first, or took another branch), so a Decision's verdict
    records what every one of these read, not only what its readers did; one that can
    never read it is left out, so a change to what it reads does not reach the verdict."""
    key = ("may read", 0)
    found = model._read.get(key)
    if found is None:
        found = model._read[key] = _readers_index(model.linked)
    return cast(Mapping[int, tuple[int, ...]], found)


def _readers_index(linked: LinkedModel) -> Mapping[int, tuple[int, ...]]:
    """``_may_read``: the read graph's strongly connected components (Tarjan's, iterative;
    a node ``n`` of scope ``s`` has the successor ``len(nodes) + s`` when called on its
    Space, and that one every node of ``s`` and the same for each scope placed in ``s``),
    each with the Decisions reached from it as a bit set, found sinks first."""
    nodes, scopes = linked.nodes, linked.scopes
    size = len(nodes)
    in_scope: list[list[int]] = [[] for _ in scopes]
    for node in nodes:
        in_scope[node.scope].append(node.index)
    for item in scopes:
        if item.parent is not None:
            in_scope[item.parent].append(size + item.index)
    bit = {decision: 1 << ordinal for ordinal, decision in enumerate(linked.decisions)}

    def successors(vertex: int) -> tuple[int, ...]:
        if vertex >= size:
            return tuple(in_scope[vertex - size])
        node = nodes[vertex]
        if node.kind == "decision":
            return () if node.guard is None else (node.guard,)
        edges = (*node.dependencies, *(source for _, source in node.via))
        return (*edges, size + node.scope) if node.call_style == "self" else edges

    total = size + len(scopes)
    order = [0] * total  # discovery order + 1; 0: not visited
    low = [0] * total
    component = [-1] * total
    masks: list[int] = []
    stack: list[int] = []
    counter = 0
    for root in range(total):
        if order[root]:
            continue
        counter += 1
        order[root] = low[root] = counter
        stack.append(root)
        frames = [(root, iter(successors(root)))]
        while frames:
            vertex, pending = frames[-1]
            advanced = False
            for child in pending:
                if not order[child]:
                    counter += 1
                    order[child] = low[child] = counter
                    stack.append(child)
                    frames.append((child, iter(successors(child))))
                    advanced = True
                    break
                if component[child] < 0:
                    low[vertex] = min(low[vertex], order[child])
            if advanced:
                continue
            frames.pop()
            if frames:
                parent = frames[-1][0]
                low[parent] = min(low[parent], low[vertex])
            if low[vertex] != order[vertex]:
                continue
            # ``vertex`` roots a component: every component it reaches is found already.
            members: list[int] = []
            while True:
                member = stack.pop()
                component[member] = len(masks)
                members.append(member)
                if member == vertex:
                    break
            mask = 0
            for member in members:
                mask |= bit.get(member, 0)
                for child in successors(member):
                    if component[child] != len(masks):
                        mask |= masks[component[child]]
            masks.append(mask)
    readers: dict[int, list[int]] = {}
    for node in nodes:
        if node.kind != "constraint":
            continue
        mask = masks[component[node.index]]
        while mask:
            lowest = mask & -mask
            readers.setdefault(linked.decisions[lowest.bit_length() - 1], []).append(node.index)
            mask ^= lowest
    return MappingProxyType({decision: tuple(each) for decision, each in readers.items()})


def _describe(result: Rejected) -> str:
    return "; ".join(map(str, result.findings))


def _reads(snapshot: Snapshot, roots: Iterable[int], own: int) -> dict[int, object]:
    """Every Decision the evaluation of ``roots`` read in ``snapshot`` (but ``own``), with
    what a read of it answered there: its value, ``UNCHECKED`` where it is stored and its
    check cannot conclude, ``OPEN`` if it has none."""
    nodes, cache = snapshot.linked.nodes, snapshot.cache
    assignments, unchecked = snapshot.assignments, snapshot.unchecked
    found: dict[int, object] = {}
    for root in roots:
        entry = cache.get(root)
        read = (root,) if nodes[root].kind == "decision" else ()
        for index in (*read, *(entry.decisions if entry is not None else ())):
            if index != own:
                found[index] = UNCHECKED if index in unchecked else assignments.get(index, OPEN)
    return found


def _refusal_roots(snapshot: Snapshot, index: int) -> list[int]:
    """What a refusal of the validity node ``index`` (evaluated in ``snapshot``, a copy that
    does not force) can rest on: its guard, each constraint, each placed Space's (the same,
    below it), and each Decision's guard. A Decision is read there as unassigned or as the
    value assigned on the copy, which never refuses: only its guard can (``_piece``), so a
    verdict that rests on the validity reads the guard and not the Decision."""
    nodes, cache = snapshot.linked.nodes, snapshot.cache
    roots: list[int] = []
    pending = [index]
    while pending:
        current = nodes[pending.pop()]
        if current.guard is not None:
            roots.append(current.guard)
        entry = cache.get(current.index)
        if entry is None or not isinstance(entry.assessment, ConstraintAssessment):
            continue
        if not entry.assessment.results:  # inactive, or its guard refused or waits
            continue
        for piece in current.constraints:
            kind = nodes[piece].kind
            if kind == "validity":
                pending.append(piece)
            elif kind == "decision":
                if nodes[piece].guard is not None:
                    roots.append(cast(int, nodes[piece].guard))
            else:
                roots.append(piece)
    return roots


def _facts(snapshot: Snapshot, roots: Iterable[int]) -> frozenset[int]:
    """Every formal the evaluation of ``roots`` read in ``snapshot``."""
    nodes, cache = snapshot.linked.nodes, snapshot.cache
    found: set[int] = set()
    for root in roots:
        entry = cache.get(root)
        if nodes[root].kind == "param":
            found.add(root)
        if entry is not None:
            found.update(entry.facts)
    return frozenset(found)


def _extended(base: Snapshot, assignments: Mapping[int, object], added: int) -> Snapshot:
    """A copy of ``base`` (one that does not force) with ``assignments``: ``base``'s and the
    Decision ``added``. It starts from every evaluation of ``base`` that did not read
    ``added``, directly or through another: those answer the same in the copy."""
    copy = Snapshot(
        base.model, base.parameters, assignments, forcing=False, unchecked=base.unchecked
    )
    _runtime.inherit(copy, base, (added,), keep_fresh=True)
    return copy


def _holding(
    linked: LinkedModel,
    verdict: Verdict | None,
    values: Mapping[int, object],
    unchecked: frozenset[int] = frozenset(),
) -> Verdict | None:
    """The verdict that holds under ``values``: ``verdict`` itself, or else its settled one
    (``Verdict.settled``); None where neither does."""
    if verdict is None:
        return None
    if _unchanged(linked, verdict, values, unchecked):
        return verdict
    settled = verdict.settled
    if settled is not None and _unchanged(linked, settled, values, unchecked):
        return settled
    return None


def _unchanged(
    linked: LinkedModel,
    verdict: Verdict,
    values: Mapping[int, object],
    unchecked: frozenset[int] = frozenset(),
) -> bool:
    """Whether a read of every Decision a verdict read still answers what it saw: the
    value, or open, or unresolved as a stored choice (``unchecked``) whose check cannot
    conclude. An unchecked candidate is never a value here."""
    for index, seen in verdict.reads.items():
        now = UNCHECKED if index in unchecked else values.get(index, OPEN)
        if now is seen:
            continue
        if isinstance(now, _Reading) or isinstance(seen, _Reading):
            return False
        semantics = linked.nodes[index].semantics
        assert semantics is not None
        if not equal(
            semantics, seen, now, owner=linked.nodes[index].owner, role="forcing equality"
        ):
            return False
    return True


def _cases(
    current: Snapshot, index: int, candidates: tuple[object, ...], choice: Choice | None
) -> tuple[tuple[object, ...], dict[str, str], dict[int, object], frozenset[int]]:
    """An open Decision's viable cases: those that no constraint reading the Decision
    refuses in the configuration plus the case (a **trial**), nor, for a Decision over
    nodes, the candidate's validity; why each other case is not viable; and what was read:
    the Decisions with the values seen, and the formals.

    The constraints are every one that may read the Decision, wherever it is owned
    (``_rules``), each read in ``current`` with the Decision open: one whose evaluation
    read the Decision (directly or through another) is tried with each case assumed; one
    that did not answers the same whatever the case. What every one of them read is the
    verdict's, since a constraint that stopped before the Decision (it waits on another
    choice) may read it once that choice is made. A case a constraint waits on is still
    viable. A Decision over values none of them reads costs no trial; a Decision over
    nodes tries each case that places a Space, against its candidate's validity
    (``_refusal_roots``: the reads it rests on)."""
    rules = _rules(current.model, index)
    readers = [rule for rule in rules if index in _runtime.evaluate(current, rule).decisions]
    reads = _reads(current, rules, index)
    facts = set(_facts(current, rules))
    scopes = {} if choice is None else dict(choice.cases)
    viable: list[object] = []
    reasons: dict[str, str] = {}
    for case in candidates:
        scope = scopes.get(str(case))
        if scope is None and not readers:
            viable.append(case)
            continue
        # The configuration plus the case, without revalidating the rest: a committed
        # choice nested under a selector not forced yet is not this case's refusal.
        trial = _extended(current, {**current.assignments, index: case}, index)
        refused: list[Rejected] = []
        if scope is not None:
            verdict, node = validity(trial, scope)
            roots = _refusal_roots(trial, node)
            reads.update(_reads(trial, roots, index))
            facts |= _facts(trial, roots)
            if isinstance(verdict, Rejected):
                refused.append(verdict)
        for rule in readers:
            answer = _runtime.evaluate(trial, rule).result
            if isinstance(answer, Rejected):
                refused.append(answer)
        reads.update(_reads(trial, readers, index))
        facts |= _facts(trial, readers)
        if refused:
            reasons[str(case) if choice is not None else repr(case)] = "; ".join(
                _describe(answer) for answer in refused
            )
        else:
            viable.append(case)  # waiting on an open choice is not a refusal
    return tuple(viable), reasons, reads, frozenset(facts)


def _verdict(current: Snapshot, index: int) -> Verdict:
    """An open Decision's verdict, evaluated on a copy that does not force."""
    linked = current.linked
    node = linked.nodes[index]
    guard = () if node.guard is None else (node.guard,)
    if node.domain is None:
        return Verdict(None, {}, {})  # no cases to read
    applicable = _runtime.decision_state(current, index)
    if not isinstance(applicable, Available):
        # Inapplicable, or its guard waits on an open choice.
        return Verdict(None, {}, _reads(current, guard, index), facts=_facts(current, guard))
    listed = _runtime.enumeration(current, index)
    roots = (*guard, *listed.dependencies)
    reads = _reads(current, roots, index)
    facts = _facts(current, roots)
    if not isinstance(listed.result, Available) or listed.result.value is None:
        return Verdict(None, {}, reads, isinstance(listed.result, Available), facts)
    candidates = tuple(cast("Iterable[object]", listed.result.value))
    choice = linked.selector_choices.get(index)
    cases, reasons, more, seen = _cases(
        current, index, candidates, None if choice is None else linked.choices[choice]
    )
    return Verdict(
        cases,
        MappingProxyType(reasons),
        MappingProxyType({**reads, **more}),
        facts=facts | seen,
    )


def _find(snapshot: Snapshot) -> Found:
    """The snapshot's forced values, refused Decisions and verdicts.

    A configuration with no base runs the rank-order rounds (``_rounds``). A successor
    of a base whose forced values were found first replays them as **seeds** and runs
    the rounds from there, which keeps what it derived only when it is one derivation
    of the forcing the plain rounds make:

    - the seeds replay the base's derivation (``_replay``): in the order the base found
      them, each still open, with exactly one case, and every Decision its verdict read
      at the value it saw (a Decision seen open still open) at that point of the replay.
      A seed is therefore a step the plain rounds could take, given the seeds before
      it, and two seeds that hold only given each other are not both seeded;
    - every forced verdict, seeded or found in the rounds, still holds under the final
      values, or its settled verdict does (``_settle``);
    - no Decision is refused.

    Adding a value only shrinks what is viable, so without a refusal two complete
    derivations from the same commitments force the same Decisions to the same values,
    whatever order either took, and the seeds make the seeded run one. Where a check
    fails the seeds are dropped and the plain rounds run from the snapshot, so the
    result is the plain rounds' (the order in which Decisions are visited can show in a
    refusal, which the seeded run never reports). Verdicts that rested on a point of
    either run are reused only by value (``_unchanged``), so they save work and never
    change an answer.
    """
    linked = snapshot.linked
    verdicts: dict[int, Verdict] = {
        index: verdict
        for index, verdict in snapshot.verdicts.items()
        if isinstance(verdict, Verdict)
    }
    open_ = sorted(
        (index for index in linked.decisions if index not in snapshot.assignments),
        key=linked.ranks.__getitem__,
    )
    seeds = _replay(snapshot, verdicts)
    if seeds:
        values, refused, current = _rounds(snapshot, verdicts, open_, seeds)
        final = {**snapshot.assignments, **values}
        if not refused and all(
            _holding(linked, verdicts[index], final, snapshot.unchecked) is not None
            for index in values
        ):
            return _publish(snapshot, verdicts, values, refused, current)
    values, refused, current = _rounds(snapshot, verdicts, open_, {})
    return _publish(snapshot, verdicts, values, refused, current)


def _replay(snapshot: Snapshot, verdicts: Mapping[int, Verdict]) -> dict[int, object]:
    """The forced values the snapshot's base found that this snapshot can start from:
    walked in the order the base found them, each one still open here, its verdict
    (inherited) with that one case, and every read of the verdict holding under the
    commitments and the seeds accepted so far."""
    linked = snapshot.linked
    assigned = dict(snapshot.assignments)
    seeds: dict[int, object] = {}
    for index in snapshot.carried:
        verdict = verdicts.get(index)
        if (
            index not in assigned
            and index in linked.decisions
            and verdict is not None
            and verdict.cases is not None
            and len(verdict.cases) == 1
            and _unchanged(linked, verdict, assigned, snapshot.unchecked)
        ):
            seeds[index] = assigned[index] = verdict.cases[0]
    return seeds


def _rounds(
    snapshot: Snapshot,
    verdicts: dict[int, Verdict],
    open_: list[int],
    seeds: Mapping[int, object],
) -> tuple[dict[int, object], dict[int, Rejected], Snapshot]:
    """Rounds over the open Decisions in rank order from ``seeds`` (already forced, in
    the order given), each forced value visible to the next, until a round forces
    nothing more. It reuses ``verdicts`` by value and records the ones it finds. The
    forced values, the refused Decisions, and the last copy."""
    linked = snapshot.linked
    values: dict[int, object] = dict(seeds)
    # The copy starts from the snapshot's evaluations that read no open Decision but a
    # seed: an open Decision reads as unassigned on the copy, and as forced on the
    # snapshot, and a seed is assigned on the copy, forced to the same value on the snapshot.
    current = Snapshot(
        snapshot.model,
        snapshot.parameters,
        {**snapshot.assignments, **values},
        forcing=False,
        unchecked=snapshot.unchecked,
    )
    _runtime.inherit(current, snapshot, [index for index in open_ if index not in values])
    refused: dict[int, Rejected] = {}
    progressed = True
    while progressed:
        progressed = False
        refused.clear()
        for index in open_:
            if index in values:
                continue
            verdict = verdicts.get(index)
            if verdict is None or not _unchanged(
                linked, verdict, current.assignments, snapshot.unchecked
            ):
                verdict = verdicts[index] = _verdict(current, index)
            if verdict.cases is None:
                continue
            if len(verdict.cases) == 1:
                values[index] = verdict.cases[0]
                current = _extended(current, {**snapshot.assignments, **values}, index)
                progressed = True
            elif not verdict.cases and verdict.reasons:
                detail = "; ".join(f"{case}: {why}" for case, why in verdict.reasons.items())
                refused[index] = Rejected(
                    (
                        Finding(
                            FindingKind.REJECTION,
                            "decision-no-viable-case",
                            linked.nodes[index].owner,
                            f"no case is viable: {detail}",
                        ),
                    )
                )
    if not refused:
        _settle(snapshot, verdicts, values, current)
    return values, refused, current


def _settle(
    snapshot: Snapshot,
    verdicts: dict[int, Verdict],
    values: Mapping[int, object],
    current: Snapshot,
) -> None:
    """Give each forced value whose verdict does not hold under the final values its
    settled verdict (``Verdict.settled``): found again on a copy of ``current`` (which
    holds every forced value) with the Decision itself open, kept where it forces the
    same case. One found before is reused while it holds. A verdict is found again only
    where a Decision it read open was forced after it: one a constraint reading the
    Decision reads too, such as a choice inside a candidate the Decision forces."""
    linked = snapshot.linked
    final = {**snapshot.assignments, **values}
    for index, value in values.items():
        verdict = verdicts[index]
        if _unchanged(linked, verdict, final, snapshot.unchecked):
            continue
        held = verdict.settled
        if held is not None and _unchanged(linked, held, final, snapshot.unchecked):
            continue
        copy = Snapshot(
            snapshot.model,
            snapshot.parameters,
            {key: item for key, item in final.items() if key != index},
            forcing=False,
            unchecked=snapshot.unchecked,
        )
        _runtime.inherit(copy, current, (index,))
        again = _verdict(copy, index)
        cases = again.cases
        if cases is not None and len(cases) == 1 and _same_case(linked, index, cases[0], value):
            verdicts[index] = replace(verdict, settled=again)
        elif held is not None:
            verdicts[index] = replace(verdict, settled=None)


def _same_case(linked: LinkedModel, index: int, left: object, right: object) -> bool:
    if left is right:
        return True
    semantics = linked.nodes[index].semantics
    assert semantics is not None
    return equal(semantics, left, right, owner=linked.nodes[index].owner, role="forcing equality")


def _publish(
    snapshot: Snapshot,
    verdicts: dict[int, Verdict],
    values: dict[int, object],
    refused: Mapping[int, Rejected],
    current: Snapshot,
) -> Found:
    """The result of the rounds, and what their last copy evaluated, given to the snapshot."""
    # The last copy assigns every forced value as the snapshot reads it, and an open
    # Decision that is not forced reads as unassigned on both, unless it is refused:
    # what the copy evaluated that read no refused Decision answers the same here.
    for index, entry in _runtime.unaffected(_runtime.fresh(current), refused).items():
        snapshot.cache.setdefault(index, entry)
    return Found(
        MappingProxyType(values), MappingProxyType(dict(refused)), MappingProxyType(verdicts)
    )


__all__ = [
    "NOTHING",
    "Forced",
    "Found",
    "Open",
    "Verdict",
    "Viable",
    "carried",
    "forced",
    "inherited",
    "validity",
]
