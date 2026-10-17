# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Forced Decisions: an open Decision with exactly one viable case reads as that case.

A forced Decision is derived at read time and never stored. It stays
unassigned (``DecisionState`` has no other status), it is not among a
configuration's commitments, and each snapshot finds its own. The public reads are
in ``inspection``: ``forced`` and ``viable`` report the verdicts, ``admission``
a candidate's own refusal.

A case of a Decision over nodes is **viable** when the candidate it places does
not refuse the configuration through its ``admission`` member (a
``ConstraintGroup``, a constraint or a view; a group refuses as soon as one of
its constraints does). An admission still waiting on an open choice does not
refuse, and the Decision's owner is not consulted. A case of a Decision over
values is viable when every requirement it states holds (``requiring``), read
through the domain's membership; without requirements every enumerated case
is. One viable case forces the Decision, several leave it open, and none
refuses it (``decision-no-viable-case``, each case with its reason): the same
for both kinds.

A snapshot finds its forced Decisions once, on the first read of an open
Decision: rounds over the open Decisions in rank order, each forced value
visible to the next (an adapter after the core it feeds is found in the same
round), evaluated on copies that do not force themselves. A successor of a base
whose forced values were found starts its rounds from them (``_find``): the base's
forced values, replayed in the order the base found them, each held only while its
verdict's reads hold, are assigned on the first copy, and the rounds then find only
what the change reaches; the result is kept only where it is a derivation of the
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
each, and is reused while none of them changed: within the rounds, and by a
successor from its base, so a change re-checks only the verdicts it reaches.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from finn.core.reporting.findings import Finding, FindingKind

from . import _runtime
from ._runtime import Snapshot
from .declarations import Constraint, ConstraintGroup, View
from .ir import Choice, LinkedModel
from .results import Available, ConstraintAssessment, QueryResult, Rejected
from .semantics import equal


class _Open:
    """A Decision a verdict read while it was open."""

    def __repr__(self) -> str:
        return "OPEN"


OPEN = _Open()
_BUSY = object()


@dataclass(frozen=True)
class Verdict:
    """A Decision's viable cases (None: not applicable, or not enumerable here), why
    each other case is not viable, and the Decisions read with the value seen.
    ``membership``: applicable, and known by its domain's membership only."""

    cases: tuple[object, ...] | None
    reasons: Mapping[str, str]
    reads: Mapping[int, object]
    membership: bool = False


@dataclass(frozen=True)
class Found:
    """A snapshot's forced values (in the order they were found, which a successor replays)
    and refused Decisions by node, and every verdict found so far (a successor's to reuse)."""

    values: Mapping[int, object] = field(default_factory=dict)
    refused: Mapping[int, Rejected] = field(default_factory=dict)
    verdicts: Mapping[int, Verdict] = field(default_factory=dict)
    # Who reads whom among the snapshot's open Decisions, built on the first ``reach``.
    _graph: list[tuple[Mapping[int, frozenset[int]], frozenset[int]]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def reach(self, base: Snapshot, seeds: Collection[int]) -> set[int]:
        """The Decisions open in ``base`` (the snapshot these were found on) that a change
        to ``seeds`` reaches: those whose verdict read a seed, or read one of these, to a
        fixed point; and those with no verdict, or whose verdict does not hold under
        ``base``'s own final values. Such a verdict records a point in the rounds (a
        Decision it read was still open and forced later), not the answer that stands,
        so what it decided is not known to be stable."""
        if not self._graph:
            self._graph.append(self._reads_by(base))
        readers, unsettled = self._graph[0]
        reached = set(unsettled)
        pending = [*seeds, *unsettled]
        while pending:
            for reader in readers.get(pending.pop(), ()):
                if reader not in reached:
                    reached.add(reader)
                    pending.append(reader)
        return reached

    def _reads_by(self, base: Snapshot) -> tuple[Mapping[int, frozenset[int]], frozenset[int]]:
        """The open Decisions whose verdicts read each Decision, and the open Decisions
        whose verdicts are not known to hold (``reach``)."""
        linked = base.linked
        final = {**base.assignments, **self.values}
        readers: dict[int, set[int]] = {}
        unsettled: set[int] = set()
        for index in linked.decisions:
            if index in base.assignments:
                continue
            verdict = self.verdicts.get(index)
            if verdict is None or not _unchanged(linked, verdict, final):
                unsettled.add(index)
                continue
            for read in verdict.reads:
                readers.setdefault(read, set()).add(index)
        return {read: frozenset(each) for read, each in readers.items()}, frozenset(unsettled)


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


def admitted(snapshot: Snapshot, scope: int) -> tuple[QueryResult[object] | None, int | None]:
    """A scope's ``admission`` answer and the node that gave it, read in ``snapshot``."""
    member = getattr(snapshot.linked.scopes[scope].space_type, "admission", None)
    if not isinstance(member, (ConstraintGroup, Constraint, View)):
        return None, None
    index = snapshot.model.resolve(scope, member)
    entry = _runtime.evaluate(snapshot, index)
    if isinstance(member, ConstraintGroup) and isinstance(entry.assessment, ConstraintAssessment):
        refused = [
            result for result in entry.assessment.results.values() if isinstance(result, Rejected)
        ]
        if refused:
            return Rejected(tuple(item for result in refused for item in result.findings)), index
    return entry.result, index


def _describe(result: Rejected) -> str:
    return "; ".join(map(str, result.findings))


def _reads(snapshot: Snapshot, roots: Iterable[int], own: int) -> dict[int, object]:
    """Every Decision the evaluation of ``roots`` read in ``snapshot`` (but ``own``), with
    the value it had there (``OPEN`` if none)."""
    nodes, cache, assignments = snapshot.linked.nodes, snapshot.cache, snapshot.assignments
    found: dict[int, object] = {}
    for root in roots:
        entry = cache.get(root)
        read = (root,) if nodes[root].kind == "decision" else ()
        for index in (*read, *(entry.decisions if entry is not None else ())):
            if index != own:
                found[index] = assignments.get(index, OPEN)
    return found


def _extended(base: Snapshot, assignments: Mapping[int, object], added: int) -> Snapshot:
    """A copy of ``base`` (one that does not force) with ``assignments``: ``base``'s and the
    Decision ``added``. It starts from every evaluation of ``base`` that did not read
    ``added``, directly or through another: those answer the same in the copy."""
    copy = Snapshot(base.model, base.parameters, assignments, forcing=False)
    _runtime.inherit(copy, base, (added,), keep_fresh=True)
    return copy


def _unchanged(linked: LinkedModel, verdict: Verdict, values: Mapping[int, object]) -> bool:
    """Whether every Decision a verdict read still has the value it saw."""
    for index, seen in verdict.reads.items():
        now = values.get(index, OPEN)
        if now is seen:
            continue
        if now is OPEN or seen is OPEN:
            return False
        semantics = linked.nodes[index].semantics
        assert semantics is not None
        if not equal(
            semantics, seen, now, owner=linked.nodes[index].owner, role="forcing equality"
        ):
            return False
    return True


def _nodes(
    current: Snapshot, index: int, choice: Choice, candidates: tuple[object, ...]
) -> tuple[tuple[object, ...], dict[str, str], dict[int, object]]:
    """A Decision over nodes' viable cases: those whose candidate's admission does not
    refuse the configuration plus the case."""
    scopes = dict(choice.cases)
    viable: list[object] = []
    reasons: dict[str, str] = {}
    reads: dict[int, object] = {}
    for case in candidates:
        scope = scopes[str(case)]
        if scope is None:
            viable.append(case)
            continue
        # The configuration plus the case, without revalidating the rest: a committed
        # choice nested under a selector not forced yet is not this case's refusal.
        trial = _extended(current, {**current.assignments, index: case}, index)
        verdict, node = admitted(trial, scope)
        if node is not None:
            reads.update(_reads(trial, (node,), index))
        if isinstance(verdict, Rejected):
            reasons[str(case)] = _describe(verdict)
            continue
        viable.append(case)
    return tuple(viable), reasons, reads


def _values(
    current: Snapshot, index: int, candidates: tuple[object, ...]
) -> tuple[tuple[object, ...], dict[str, str], dict[int, object]]:
    """A Decision over values' viable cases: those its domain's membership admits here,
    every requirement holding."""
    viable: list[object] = []
    reasons: dict[str, str] = {}
    reads: dict[int, object] = {}
    for case in candidates:
        entry = _runtime.membership(current, index, case)
        reads.update(_reads(current, entry.dependencies, index))
        if isinstance(entry.result, Rejected):
            reasons[repr(case)] = _describe(entry.result)
        else:
            viable.append(case)  # waiting on an open choice is not a refusal
    return tuple(viable), reasons, reads


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
        return Verdict(None, {}, _reads(current, guard, index))
    listed = _runtime.enumeration(current, index)
    reads = _reads(current, (*guard, *listed.dependencies), index)
    if not isinstance(listed.result, Available) or listed.result.value is None:
        return Verdict(None, {}, reads, isinstance(listed.result, Available))
    candidates = tuple(cast("Iterable[object]", listed.result.value))
    choice = linked.selector_choices.get(index)
    if choice is not None:
        cases, reasons, more = _nodes(current, index, linked.choices[choice], candidates)
    elif node.domain is not None and node.domain.requirements:
        cases, reasons, more = _values(current, index, candidates)
    else:
        cases, reasons, more = candidates, {}, {}
    return Verdict(cases, MappingProxyType(reasons), MappingProxyType({**reads, **more}))


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
      values;
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
        if not refused and all(_unchanged(linked, verdicts[index], final) for index in values):
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
            and _unchanged(linked, verdict, assigned)
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
        snapshot.model, snapshot.parameters, {**snapshot.assignments, **values}, forcing=False
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
            if verdict is None or not _unchanged(linked, verdict, current.assignments):
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
    return values, refused, current


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
    "admitted",
    "carried",
    "forced",
    "inherited",
]
