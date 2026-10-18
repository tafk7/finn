# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demand-limited iterative evaluation over one immutable snapshot."""

from __future__ import annotations

from collections.abc import Callable, Collection, Generator, Mapping
from dataclasses import dataclass, field
from itertools import islice
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

from finn.core.reporting.findings import Finding, FindingKind

from . import _execution
from .errors import EvaluationError
from .graph import Located
from .ir import Argument, LinkedModel, Node, NodeKind
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    NonValue,
    QueryResult,
    ReadinessAssessment,
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


_NONE: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class Evaluation:
    result: QueryResult[object]
    dependencies: tuple[int, ...] = ()
    assessment: Assessment | None = None
    # Method reads that went straight to a forwarding alias's source: (alias, source).
    via: tuple[tuple[int, int], ...] = ()
    # Every Decision read, directly or through another evaluation (``reads``).
    decisions: frozenset[int] = _NONE
    # Every formal read the same way: the snapshot's ``param`` nodes, whose values are its
    # facts. A configuration with other facts keeps what read none of the changed ones.
    facts: frozenset[int] = _NONE


def reads(
    snapshot: Snapshot, dependencies: Collection[int], via: Collection[tuple[int, int]]
) -> tuple[frozenset[int], frozenset[int]]:
    """The Decisions and the formals an evaluation of ``snapshot`` read: those among its
    dependencies, and every one its dependencies read, and a forwarding alias it read
    through its source, as far as the cache holds them. A successful evaluation's
    dependencies all succeeded first, so they are cached."""
    nodes, cache = snapshot.linked.nodes, snapshot.cache
    decisions = facts = _NONE
    for read in (*dependencies, *(alias for alias, _ in via)):
        entry = cache.get(read)
        if entry is not None:
            if entry.decisions and not entry.decisions <= decisions:
                decisions = entry.decisions if not decisions else decisions | entry.decisions
            if entry.facts and not entry.facts <= facts:
                facts = entry.facts if not facts else facts | entry.facts
        kind = nodes[read].kind
        if kind == "decision":
            if read not in decisions:
                decisions = decisions | {read}
        elif kind == "param" and read not in facts:
            facts = facts | {read}
    return decisions, facts


def unaffected(cache: Mapping[int, Evaluation], reached: Collection[int]) -> dict[int, Evaluation]:
    """The evaluations of ``cache`` that are none of ``reached`` (Decisions, or the formals
    of a configuration with other facts) and read none of them, directly or through
    another. Where only those answers differ, these answer the same: an evaluation is a
    function of what it read. What a kept evaluation read is kept with it, so its
    evidence stays complete. This is how a small cache is copied, and what a large one
    gives back (``inherit`` finds the same from the ``Readers`` of a large one without
    looking at each evaluation)."""
    seeds = reached if isinstance(reached, (set, frozenset)) else frozenset(reached)
    return {
        index: entry
        for index, entry in cache.items()
        if index not in seeds
        and seeds.isdisjoint(entry.decisions)
        and seeds.isdisjoint(entry.facts)
    }


# What makes a snapshot's cache: an exact ``dict``, so production pays nothing for the
# rule that a cache is only added to (``Snapshot``). The tests install a dict that refuses
# what the rule forbids (``tests/support/checked_cache.py``, from ``tests/conftest.py``).
# Called with no argument it makes an empty cache, and with a mapping a cache of its entries.
cache_factory: Callable[..., dict[int, Evaluation]] = dict


def new_cache() -> dict[int, Evaluation]:
    """An empty cache, of the kind ``cache_factory`` makes."""
    return cache_factory()


def _cache_of(entries: dict[int, Evaluation]) -> dict[int, Evaluation]:
    """``entries`` as a cache: themselves, where caches are exact dicts."""
    return entries if cache_factory is dict else cache_factory(entries)


# A cache of fewer evaluations than this is copied by looking at each (``unaffected``);
# from this many, ``inherit`` uses the cache's ``Readers``. Looking at each evaluation costs
# about 0.4-0.7 us an evaluation. The index (a copy of the dict, of the readers' table, and the
# readers of what goes undone) costs 0.03 us an evaluation where one cache is copied again
# and again (a change on a whole Space), and about 0.4 us where a chain of copies each gains
# back what it lost and reads it in (forcing's copies, on a trial's way). Measured on a
# cache of 100 evaluations the two ways are level for the first and the scan is 1.2-1.9
# times faster for the second; at 500 the index is 2.6 and 1.2-1.4 times faster; at 10,000
# 14 and 1.7-1.9 times. A network's member Space holds up to about 140 evaluations and
# copies most of them whole (the network trial at 64 kernels), a whole Space about 10,000
# and a change reaches a few of them. The answers are the same either way; only the work
# differs.
INDEX_FROM = 500


class Readers:
    """Who read which Decision or formal, in one cache: for each, the evaluations that
    read it (``Evaluation.decisions``, ``Evaluation.facts``). It is brought up to date from
    what was added to the cache since, in insertion order, at the time a change asks
    (``sync``), so what a change reaches is found from the Decisions it touches and not by
    looking at every evaluation. The cache is only added to (``Snapshot``), so the
    evaluations the index has read are the cache's leading ones. A set of readers is shared
    with the index it was copied from until this one changes it."""

    __slots__ = ("_readers", "_owned", "synced")

    def __init__(self) -> None:
        self._readers: dict[int, set[int]] = {}
        self._owned: set[int] = set()
        self.synced = 0  # how many of its cache's leading evaluations this index accounts for

    def sync(self, cache: Mapping[int, Evaluation]) -> None:
        """Read what was added to ``cache`` since."""
        if self.synced == len(cache):
            return
        for index, entry in islice(cache.items(), self.synced, None):
            self.read(index, entry)
        self.synced = len(cache)

    def read(self, index: int, entry: Evaluation) -> None:
        readers, owned = self._readers, self._owned
        for read in (*entry.decisions, *entry.facts) if entry.facts else entry.decisions:
            if read in owned:
                readers[read].add(index)
            else:
                readers[read] = {*readers.get(read, ()), index}
                owned.add(read)

    def unread(self, index: int, entry: Evaluation) -> None:
        readers, owned = self._readers, self._owned
        for read in (*entry.decisions, *entry.facts) if entry.facts else entry.decisions:
            if read not in owned:
                readers[read] = set(readers.get(read, ()))
                owned.add(read)
            readers[read].discard(index)

    def reached(self, seeds: Collection[int]) -> set[int]:
        """The ``seeds`` (Decisions or formals) and the evaluations that read any of them
        (up to date: ``sync``)."""
        found = set(seeds)
        for seed in seeds:
            found.update(self._readers.get(seed, ()))
        return found

    def copy(self) -> Readers:
        """An index for a copy of the cache, sharing the sets of readers with this one."""
        child = Readers()
        child._readers = dict(self._readers)
        child.synced = self.synced
        # What is shared is no longer this index's alone.
        self._owned = set()
        return child


def fresh(snapshot: Snapshot) -> dict[int, Evaluation]:
    """What was added to ``snapshot``'s cache since it was made, in order (``inherit`` may
    carry its base's along): all a successor or a forcing run gives back to the snapshot
    it started from."""
    return dict(islice(snapshot.cache.items(), snapshot.made, None))


def inherit(
    target: Snapshot,
    base: Snapshot,
    seeds: Collection[int],
    *,
    keep_fresh: bool = False,
    replacing: Mapping[int, Evaluation] | None = None,
) -> None:
    """Start ``target``'s cache from the evaluations of ``base``'s that are none of
    ``seeds`` and read none of them, directly or through another (``unaffected``): where
    only those Decisions' answers differ, these answer the same. ``replacing`` gives
    evaluations in place of those of the same index, which are fresh in it, and
    ``keep_fresh`` carries along what ``base`` itself gained since it was made.

    A cache of ``INDEX_FROM`` evaluations or more is copied from its ``Readers`` (made
    on the first asking and handed on to the copy, to be brought up to date when the copy
    is asked in turn): a copy of the dict and the readers of the seeds, not a look at
    each evaluation. A smaller one is looked through, and its copy has no index until
    it grows to that size. ``base`` is left as it was."""
    cache, made = base.cache, base.made
    given = replacing or {}
    readers: Readers | None = None
    if len(cache) < INDEX_FROM:
        kept = unaffected(cache, seeds)
        for index in given:
            kept.pop(index, None)
        carried = (
            sum(1 for index in islice(cache, made, None) if index in kept) if keep_fresh else 0
        )
        made = len(kept) - carried
    else:
        index_of = base.readers
        if index_of is None:
            index_of = Readers()
            object.__setattr__(base, "readers", index_of)
        index_of.sync(cache)
        kept = dict(cache)
        readers = index_of.copy()
        removed = index_of.reached(seeds) | given.keys()
        for index in removed:
            old = kept.pop(index, None)
            if old is not None:
                readers.unread(index, old)
        carried = (
            sum(1 for index in islice(cache, made, None) if index not in removed)
            if keep_fresh
            else 0
        )
        made = len(kept) - carried
    for index, entry in given.items():
        kept[index] = entry
        if readers is not None:
            readers.read(index, entry)
    if readers is not None:
        readers.synced = len(kept)
    object.__setattr__(target, "made", made)
    object.__setattr__(target, "cache", _cache_of(kept))
    object.__setattr__(target, "readers", readers)


@dataclass(frozen=True, slots=True, eq=False)
class Snapshot:
    """One root binding and commitment set with a local evaluation cache.

    An open Decision with one viable case reads as that case (``forcing``), false
    only on the copies forcing evaluates on. ``found`` holds the snapshot's forced
    Decisions once found; ``verdicts`` are the ones its base found, which forcing
    reuses where the change did not reach, and ``carried`` the forced values its base
    found, in the order it found them, which forcing starts its own from where their
    verdicts still hold (``finn.core.space._forcing``). A
    successor's cache starts from what its base and its trial evaluated that the
    change does not reach (a change reaches what it adds, removes or replaces, and
    each open Decision whose forced value's verdict read one of those:
    ``_touched``), and ``admissions`` holds the Decisions each assignment's admission
    read, and ``admission_facts`` the formals, which a later trial reuses where the change
    does not reach them (``_TrialSnapshot``; a choice whose own check was read with its
    guard's cases assumed has none, and is checked again on every change). ``unchecked``
    holds the assignments whose own check cannot conclude yet (it reads a choice still open,
    or an input not supplied): each
    is stored and read **unresolved** until a change makes its check succeed, and forcing never
    seeds it or reads it as a value (``_decision_frame``); a read of a Decision answers the
    value or unresolved, and what a verdict or an evaluation records of one is that answer
    (``_forcing.observed``). ``input_refused`` holds those of them whose own check reads an
    input that is itself refused (a derived value refusing on its facts): a refusal
    elsewhere never refuses a commit, so each is kept and never read as a value; a
    configuration that forces reads it refused through that input, with the input's
    findings, and a copy that does not force reads it unresolved, as any unchecked one, so
    the refusal is blamed once, at the input. ``answers`` holds what
    inspection answered of it, by question and scope (``inspection``), where the answer
    shares its values. The cache is an exact ``dict``, **only added to**: an evaluation is
    given by ``setdefault``, and none is replaced or removed once held, or ``readers`` would
    go on holding what the replaced evaluation read. A successor, a trial or a forcing copy
    starts from its base's by leaving out what a set of Decisions reaches (``inherit``),
    and each is its own: it holds nothing of the base's it leaves out. ``readers``, where
    the cache is large enough to pay for it, holds who read each Decision in it
    (``Readers``); ``made`` counts the leading evaluations that were inherited, so what
    was added since is ``fresh``. The rule that a cache is only added to is checked where
    it costs production nothing: the tests make every cache a dict that refuses what the
    rule forbids (``cache_factory``), and a test of the engine's source finds no write to
    a cache but those the rule allows.
    """

    model: Model[Space]
    parameters: Mapping[int, object]
    assignments: Mapping[int, object] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)
    forcing: bool = True
    verdicts: Mapping[int, object] = field(default_factory=dict, repr=False)
    carried: Mapping[int, object] = field(default_factory=dict, repr=False)
    admissions: Mapping[int, frozenset[int]] = field(default_factory=dict, repr=False)
    admission_facts: Mapping[int, frozenset[int]] = field(default_factory=dict, repr=False)
    unchecked: frozenset[int] = field(default_factory=frozenset)
    input_refused: frozenset[int] = field(default_factory=frozenset)
    cache: dict[int, Evaluation] = field(default_factory=new_cache, init=False, repr=False)
    readers: Readers | None = field(default=None, init=False, repr=False)
    made: int = field(default=0, init=False, repr=False)
    answers: dict[tuple[str, int], object] = field(default_factory=dict, init=False, repr=False)
    work: _execution.Work = field(default_factory=_execution.Work, init=False, repr=False)
    found: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MappingProxyType):
            object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))

    @property
    def linked(self) -> LinkedModel:
        return self.model.linked


def changed_facts(base: Snapshot, parameters: Mapping[int, object]) -> frozenset[int]:
    """The formals whose value in ``parameters`` is not ``base``'s: supplied in one and
    not the other, or unequal by the formal's value semantics."""
    nodes, held = base.linked.nodes, base.parameters
    changed = set()
    for index in held.keys() | parameters.keys():
        if (index in held) != (index in parameters):
            changed.add(index)
        elif held[index] is not parameters[index]:
            node = nodes[index]
            assert node.semantics is not None
            if not equal(
                node.semantics,
                held[index],
                parameters[index],
                owner=node.owner,
                role="parameter equality",
            ):
                changed.add(index)
    return frozenset(changed)


def _touched(
    base: Snapshot,
    candidates: Mapping[int, object],
    facts: Collection[int] = (),
    unchecked: Collection[int] = (),
) -> list[int]:
    """The Decisions a change from ``base`` to ``candidates``, and to other values of the
    formals ``facts``, touches: those it adds, removes or replaces (not the identical
    value), those ``base`` holds unchecked and those the change leaves unchecked
    (``unchecked``: what a read of either answers may change with its check, not only
    with its value), and the open ones its reach includes: every Decision open in both whose
    forced value's verdict, in what ``base`` found, read a touched one or a changed
    formal (to a fixed point), or has no verdict, or one that does not hold under the
    base's final values (``Found.reach``). A Decision
    open in both and not touched is **stable**: its verdict read nothing the change
    touched, so it reads the same, forced to the same value, refused alike or open
    alike, in the trial, in the successor and in the base. Where ``base``'s forced
    Decisions were never found there is nothing to bound with, and every Decision open
    in the base is touched, as a value it forces may differ."""
    from ._forcing import Found  # noqa: PLC0415 - runtime/forcing cycle

    assignments = base.assignments
    unsettled = base.unchecked.union(unchecked)
    changed = {
        index
        for index in base.linked.decisions
        if (index in assignments) != (index in candidates)
        or (index in candidates and assignments[index] is not candidates[index])
        or index in unsettled
    }
    found = base.found
    if not isinstance(found, Found):
        return [
            index for index in base.linked.decisions if index not in assignments or index in changed
        ]
    reached = found.reach(base, changed, facts)
    return [index for index in base.linked.decisions if index in changed or index in reached]


def _stable(
    base: Snapshot, candidates: Mapping[int, object], touched: Collection[int]
) -> frozenset[int]:
    """The Decisions open in both ``base`` and the candidates that the change does not
    touch (``_touched``)."""
    excluded = frozenset(touched)
    return frozenset(
        index
        for index in base.linked.decisions
        if index not in base.assignments and index not in candidates and index not in excluded
    )


def _reusable(
    base: Snapshot,
    candidates: Mapping[int, object],
    touched: Collection[int],
    stable: Collection[int],
    facts: frozenset[int] = _NONE,
) -> dict[int, frozenset[int]]:
    """The retained choices whose admission read nothing the change touched (a Decision,
    or one of the changed formals ``facts``), with the Decisions it read: each would be
    admitted identically. An admission that read a
    retained choice through an evaluation of the base recorded that choice and not
    its admission's reads, so a choice is reusable only while every Decision its
    admission read is reusable too, or stable (it reads as the base forces it, as it
    did in that admission); a touched Decision never is."""
    excluded = frozenset(touched)
    reusable = {
        index: read
        for index, read in base.admissions.items()
        if index in candidates
        and index not in excluded
        and facts.isdisjoint(base.admission_facts.get(index, _NONE))
    }
    while True:
        allowed = reusable.keys() | stable
        lost = [index for index, read in reusable.items() if not read <= allowed]
        if not lost:
            return reusable
        for index in lost:
            del reusable[index]


def _case_guards(model: Model[Space]) -> Mapping[int, tuple[int, str]]:
    """Each node that holds where a selector is one of its cases (the ``$selected`` node a
    case's scope is guarded by, directly or under its ``when``), with that selector and
    case: found once per model and kept with it."""
    found = model._read.get(("case guards", 0))
    if found is None:
        linked = model.linked
        nodes, scopes = linked.nodes, linked.scopes
        guards: dict[int, tuple[int, str]] = {}
        for choice in linked.choices:
            selected = (Argument("selected", choice.selector),)
            for case, scope in choice.cases:
                guard = None if scope is None else scopes[scope].guard
                while guard is not None:
                    node = nodes[guard]
                    if node.kind == "derived" and node.arguments == selected:
                        guards[guard] = (choice.selector, case)
                        break
                    guard = node.guard
        found = model._read[("case guards", 0)] = MappingProxyType(guards)
    return cast(Mapping[int, tuple[int, str]], found)


class _TrialSnapshot(Snapshot):
    """Private complete candidates, admitted on demand before value publication.

    Every candidate, including retained assignments, is checked before publication,
    except a retained choice whose admission the change cannot reach (the first
    bullet), which keeps the admission the base recorded. A candidate's **own check**
    (its domain, on the stored value, with its guard's reads) is true, refused or
    unresolved, and ``own`` gives it: true admits the
    candidate, so that it reads as its value; refused refuses the change; unresolved
    (it reads a choice still open, or an input not supplied) keeps the candidate
    **unchecked**: stored, and read unresolved. A candidate's own check is read whatever
    its guard answers: where the guard is refused or waits, in the configuration with the
    selector cases its guard holds under committed too (``assumed``), since what the check
    reads in its own scope is under that guard as well. Which candidates are unchecked is given
    to the trial (``unchecked``), since the configuration it would publish forces by
    them: a change starts from none, and runs again with those its own checks found
    unresolved until they are the same (``_changes``). The base contributes its model,
    frozen facts and lock, its verdicts, and the values it forces, which the trial reads
    for the Decisions it does not change and stable. A Decision the base does not force
    reads as the configuration the trial would publish forces it (``successor``): found
    once, and that configuration is the one published.

    A trial may be over other values of the base's formals (``parameters``, as
    ``try_with_facts`` makes it): its facts are those, and the formals that differ from
    the base's are reached as a Decision is. Evaluations, admissions and verdicts that
    read one are left out (``changed_facts``, ``_reusable``, ``successor``), and every
    open Decision whose verdict read one is touched (``_touched``). The base's forced
    values may be of other facts, so a touched Decision reads as the successor forces
    it; a stable one still as the base forces it, the two being equal.

    What was evaluated is kept where the change cannot reach it. A change touches
    what it adds, removes or replaces and the open Decisions whose forced values it
    reaches (``_touched``); the other Decisions open in both are **stable**: each
    reads the same in the trial, in the successor and in the base, because its
    verdict read nothing the change touched.

    - A retained choice whose admission read no Decision the change touched, and
      only reused choices and stable Decisions (``_reusable``), is admitted
      identically: the trial takes its admission as the base recorded it. Every
      other candidate is admitted on its first read in the trial, and so is one whose
      own check was read in an assumed trial, whose admission is not recorded.
    - The trial starts from its base's evaluations that read no Decision but those
      reused choices and the stable Decisions (found, in a large cache, from the readers
      of the others, ``inherit``), with the reused choices' admissions as they were
      recorded. So nothing that read another
      candidate is inherited, and ``admit`` still refuses a candidate already
      resolved; a retained choice whose admission now fails refuses what reads it.
    - The successor starts from its base's evaluations that read no Decision the
      change touched, and on publication gains the trial's own (what it evaluated or
      was given: the rest is its base's, which the successor already has) that read no
      Decision open in the successor but a stable one (a stable Decision reads in the trial
      as the base forces it, which is how the successor reads it): the trial's
      candidates are the successor's assignments, each checked before it was
      read. A candidate's own answer is
      taken as the successor gives it: with the reads of its guard, which its
      frame makes first, and without the reads of its admission; what read it
      counts those reads among its own, which only keeps less. A choice whose
      admission was reused did not evaluate its guard, and the base's cache may not
      hold it, and one whose guard did not answer true, or that is unchecked, is read
      otherwise by the successor than by the trial: the successor takes neither that
      choice nor what read it, and evaluates it when something asks. The successor
      records each candidate's admission reads (``admissions``), but for a candidate
      whose own check was read in an assumed trial (``_read_assumed``): what that check
      read there is not what this configuration reads, and a forced value it read may
      follow a change that nothing here connects to it.
    """

    __slots__ = (
        "_pending",
        "_published",
        "_candidates",
        "_base",
        "_successor",
        "_touches",
        "_stable",
        "_changed",
        "_own",
        "_assumed",
        "_read_assumed",
        "_through",
    )

    _pending: dict[int, object]
    _published: bool
    _candidates: Mapping[int, object]
    _base: Snapshot
    _successor: Snapshot | None
    _touches: list[int]
    _stable: frozenset[int]
    _changed: frozenset[int]
    _own: dict[int, QueryResult[object]]
    _assumed: dict[frozenset[tuple[int, str]], _TrialSnapshot]
    _read_assumed: set[int]
    _through: set[int]

    def __init__(
        self,
        base: Snapshot,
        candidates: Mapping[int, object],
        parameters: Mapping[int, object] | None = None,
        *,
        unchecked: frozenset[int] = _NONE,
        input_refused: frozenset[int] = _NONE,
    ) -> None:
        facts = _NONE if parameters is None else changed_facts(base, parameters)
        super().__init__(
            base.model,
            base.parameters if parameters is None else parameters,
            {},
            base.lock,
            base.forcing,
            unchecked=unchecked,
            input_refused=input_refused,
        )
        candidates = MappingProxyType(dict(candidates))
        touched = _touched(base, candidates, facts, unchecked)
        stable = _stable(base, candidates, touched)
        reused = _reusable(base, candidates, touched, stable, facts)
        pending = {index: candidates[index] for index in reused}
        held = [
            index for index in base.linked.decisions if index not in reused and index not in stable
        ]
        # As the decision frame admits it, with its admission's reads.
        admitted = {
            index: Evaluation(
                Available(candidates[index]),
                decisions=read,
                facts=base.admission_facts.get(index, _NONE),
            )
            for index, read in reused.items()
        }
        inherit(self, base, [*held, *facts], replacing=admitted)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_successor", None)
        object.__setattr__(self, "_touches", touched)
        object.__setattr__(self, "_stable", stable)
        object.__setattr__(self, "_changed", facts)
        object.__setattr__(self, "_own", {})
        object.__setattr__(self, "_assumed", {})
        object.__setattr__(self, "_read_assumed", set())
        object.__setattr__(self, "_through", set())
        object.__setattr__(self, "assignments", MappingProxyType(pending))
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_candidates", candidates)
        object.__setattr__(self, "_published", False)

    def admit(self, node_index: int, value: object) -> None:
        if self._published:
            raise RuntimeError("a published admission trial is closed")
        if node_index in self._pending or node_index in self.cache:
            raise RuntimeError("admission dependency order admitted a previously resolved decision")
        self._pending[node_index] = value

    def successor(self) -> Snapshot:
        """The configuration this trial would publish: every candidate committed."""
        if self._successor is None:
            from ._forcing import (  # noqa: PLC0415 - runtime/forcing cycle
                Verdict,
                carried,
                inherited,
            )

            base = self._base
            verdicts = inherited(base)
            changed = self._changed
            if changed:
                # A verdict that read a formal with another value is found again.
                verdicts = {
                    index: verdict
                    for index, verdict in verdicts.items()
                    if not isinstance(verdict, Verdict) or changed.isdisjoint(verdict.every_fact)
                }
            successor = Snapshot(
                self.model,
                self.parameters,
                self._candidates,
                forcing=self.forcing,
                verdicts=verdicts,
                carried=carried(base),
                unchecked=self.unchecked,
                input_refused=self.input_refused,
            )
            inherit(successor, base, [*self._touches, *changed])
            object.__setattr__(self, "_successor", successor)
        assert self._successor is not None
        return self._successor

    def publish(self) -> Snapshot:
        if self._published:
            raise RuntimeError("an admission trial can only be published once")
        object.__setattr__(self, "_published", True)
        # Every candidate was admitted: the successor's assignments are the pending ones.
        successor = self.successor()
        candidates = self._candidates
        # An own check read in an assumed trial is not recorded, so it is never reused.
        recorded = [index for index in candidates if index not in self._read_assumed]
        object.__setattr__(
            successor,
            "admissions",
            MappingProxyType({index: self.cache[index].decisions for index in recorded}),
        )
        object.__setattr__(
            successor,
            "admission_facts",
            MappingProxyType({index: self.cache[index].facts for index in recorded}),
        )
        nodes = self.linked.nodes
        # A choice whose admission the trial reused did not evaluate its guard, and the
        # base's cache may not hold it. Its record would have no guard to name, so the
        # successor does not take it, nor what read it: it evaluates the choice when
        # something asks, as any snapshot does. Nor does it take a choice whose guard
        # did not answer true in the trial (a refused selector admits the choice its own
        # check holds for, which the configuration reads refused through the guard), or
        # one left unchecked (the configuration reads it unresolved, with other findings).
        unrecorded = self.unchecked | frozenset(
            index
            for index in candidates
            if nodes[index].guard is not None and not self._guard_holds(nodes[index].guard)
        )
        open_ = frozenset(
            index
            for index in self.linked.decisions
            if index not in candidates and index not in self._stable
        )
        for index, entry in unaffected(fresh(self), open_ | unrecorded).items():
            if index in candidates:
                entry = self._record(index, entry)
            successor.cache.setdefault(index, entry)
        return successor

    def own(self, index: int) -> QueryResult[object]:
        """The result of candidate ``index``'s own check in this trial, once it was read: true
        where it was admitted, the refusal where it was refused, unresolved where it cannot
        conclude (``_candidate_frame``), and the input's refusal where it reads a refused
        input (``through``). A choice whose admission the trial reused was admitted."""
        return self._own.get(index, Available(True))

    def through(self) -> frozenset[int]:
        """The candidates whose own check, read so far, reads an input that is itself
        refused: none of them refuses the change; each is kept, refused through that input
        (``Snapshot.input_refused``)."""
        return frozenset(self._through)

    def assumed(self, node: Node) -> _TrialSnapshot | None:
        """The trial of this change with each selector case ``node``'s guard holds under
        committed too (every case guard on its chain whose selector is no candidate), over
        the same base, facts and unchecked candidates: where a candidate's guard is refused
        or waits, its own check is read there, as the configuration reads it once that guard
        holds.
        None where there is no such case to commit. One per set of cases, kept for the
        trial's other candidates."""
        cases = _case_guards(self.model)
        nodes, candidates = self.linked.nodes, self._candidates
        committed: dict[int, str] = {}
        guard = node.guard
        while guard is not None:
            held = cases.get(guard)
            if held is not None and held[0] not in candidates:
                committed[held[0]] = held[1]
            guard = nodes[guard].guard
        if not committed:
            return None
        key = frozenset(committed.items())
        trial = self._assumed.get(key)
        if trial is None:
            trial = _TrialSnapshot(
                self._base,
                {**candidates, **committed},
                self.parameters if self._changed else None,
                unchecked=self.unchecked,
                input_refused=self.input_refused,
            )
            self._assumed[key] = trial
        return trial

    def _guard_holds(self, guard: int | None) -> bool:
        """Whether the trial evaluated ``guard`` and it answered true."""
        entry = self.cache.get(guard) if guard is not None else None
        return entry is not None and entry.result == Available(True)

    def _record(self, index: int, entry: Evaluation) -> Evaluation:
        """A candidate's entry as the successor's decision frame records it: its answer
        and what the frame reads, its guard (and so the Decisions the guard reads),
        and nothing of its admission."""
        guard = self.linked.nodes[index].guard
        if guard is None:
            return Evaluation(entry.result)
        if guard not in self.cache:
            raise RuntimeError("a published choice's guard was never evaluated in its trial")
        assert self._guard_holds(guard), "a recorded choice's guard answered true in its trial"
        decisions, facts = reads(self, (guard,), ())
        return Evaluation(entry.result, (guard,), decisions=decisions, facts=facts)


def _forced(snapshot: Snapshot, index: int) -> QueryResult[object] | None:
    """An open Decision's forced value (its one viable case), its refusal (no viable
    case), or None (several, or a snapshot that does not force). A trial reads its
    base's, and where the base forces nothing, the configuration's it would publish. A
    trial reads a stable Decision as its base forces it, which is how the configuration it
    would publish forces it too, and every other as that configuration does (the base's
    forced values may be of other facts, or of choices the change touches)."""
    from ._forcing import forced  # noqa: PLC0415 - runtime/forcing cycle

    if not snapshot.forcing:
        return None
    source = snapshot
    if isinstance(snapshot, _TrialSnapshot):
        if index in snapshot._stable:
            base = forced(snapshot._base)
            if index in base.values:
                return Available(base.values[index])
        source = snapshot.successor()
    found = forced(source)
    if index in found.values:
        return Available(found.values[index])
    return found.refused.get(index)


def _blocked(answers: list[QueryResult[object]]) -> NonValue | None:
    """Required inputs retain all distinct findings of the highest-precedence nonvalue:
    refused, then unresolved, then inapplicable. Every input is required, so a refused one
    settles the answer whatever the inputs still waiting become."""

    for variant in (Rejected, Unresolved, Inapplicable):
        matches = tuple(answer for answer in answers if isinstance(answer, variant))
        if matches:
            return variant(merged_findings(matches))
    return None


def _clone(node: Node, value: object, *, owner: str, role: str) -> object:
    if node.semantics is None:
        raise EvaluationError(owner, role, f"{node.key} has no value semantics")
    return snapshot_value(node.semantics, value, owner=owner, role=role)


def _arguments(
    linked: LinkedModel, arguments: tuple[Argument, ...], owner: str
) -> Generator[int, object, tuple[dict[str, object], NonValue | None]]:
    values: dict[str, object] = {}
    failures: list[QueryResult[object]] = []
    for argument in arguments:
        answer = cast(QueryResult[object], (yield argument.node))
        if isinstance(answer, Available):
            values[argument.name] = _clone(
                linked.nodes[argument.node], answer.value, owner=owner, role="dependency snapshot"
            )
        else:
            failures.append(answer)
    return values, _blocked(failures)


def _guard_result(node: Node, answer: QueryResult[object]) -> NonValue | None:
    if not isinstance(answer, Available):
        return answer
    if type(answer.value) is not bool:
        raise EvaluationError(node.owner, "applicability", "guard must return bool")
    return None if answer.value else Inapplicable()


def _guard_assessment(node: Node, answer: NonValue) -> Assessment | None:
    if node.kind == "view":
        return assess_view(answer, owner=node.key, applicability=answer)
    if node.kind == "constraint":
        return ConstraintAssessment({node.key: answer}, answer)
    if node.kind == "validity":
        return ConstraintAssessment({}, answer)
    return None


def _present(node: Node, answers: list[QueryResult[object]]) -> QueryResult[object]:
    """The single present source. A refused source refuses the answer. Otherwise, while
    any source is unresolved the answer is unresolved too: a later commitment could
    still make a second one present."""
    active = [answer for answer in answers if not isinstance(answer, Inapplicable)]
    blocked = _blocked([answer for answer in active if not isinstance(answer, Available)])
    if blocked is not None:
        return blocked
    if len(active) > 1:
        return reject(
            "multiple-suppliers",
            f"{len(active)} sources are present; at most one may supply this value",
            owner=node.owner,
        )
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


def _graph_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.kind == "present":
        answers: list[QueryResult[object]] = []
        for _, target in node.alternatives:
            answers.append(cast(QueryResult[object], (yield target)))
        return Evaluation(_present(node, answers))
    assert node.kind == "members"
    # Members: one (node name, export) alternative per entry, with the member
    # name of each entry (the key's name).
    located: list[Located[object]] = []
    failures: list[QueryResult[object]] = []
    for (name, target), member in zip(node.alternatives, cast(tuple[str, ...], node.value)):
        answer = cast(QueryResult[object], (yield target))
        if isinstance(answer, Inapplicable):
            continue
        if isinstance(answer, Available):
            located.append(Located(name or None, member, answer.value))
        else:
            failures.append(answer)
    blocked = _blocked(failures)
    return Evaluation(blocked if blocked is not None else Available(tuple(located)))


def _unchecked_answer(node: Node) -> Unresolved:
    """What a read of a stored choice whose own check cannot conclude answers."""
    return Unresolved(
        (
            Finding(
                FindingKind.BLOCKER,
                "decision-unchecked",
                node.owner,
                "the stored choice's own check cannot conclude yet",
            ),
        )
    )


def _frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """A node's evaluation: inapplicable while its guard does not hold, else its kind's.
    A Space's validity under a guard that waits is still its pieces' AND: a placed Space
    is not one piece, and each piece inside answers unknown under that guard (its own
    frame reads the guard first, and the body is never evaluated), so a Space with no
    pieces (only open choices and views) contributes nothing."""
    if isinstance(snapshot, _TrialSnapshot) and node.index in snapshot._candidates:
        return (yield from _candidate_frame(snapshot, node))
    if node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None and not (
            node.kind == "validity" and isinstance(inactive, Unresolved)
        ):
            return Evaluation(inactive, assessment=_guard_assessment(node, inactive))
    return (yield from _FRAMES[node.kind](snapshot, node))


def _candidate_frame(snapshot: _TrialSnapshot, node: Node) -> _execution.Frame:
    """A trial candidate's evaluation, and its own check (``_TrialSnapshot.own``).

    A choice's own check is its domain, on the stored value, read after its guard's reads.
    It is true, refused or unresolved, and the configuration reads the choice as its value
    once it is true, and unresolved while it cannot conclude (a candidate is read through
    its own check, never on its stored value, so private admission keeps its protection). A
    constraint is no part of it: a constraint does not gate a commit, and the
    configuration's validity reports it. A guard is the configuration's, not the
    choice's: a guard that waits leaves the choice waiting on it, and one that is refused is
    the configuration's refusal, which a change reports beside the choice's own check and
    never in its place. The own check is read whatever the guard answers, refused or
    waiting: what it reads in the choice's own scope (a domain over the scope's own values)
    is under the same guard, so it is read in the trial with the selector cases the guard
    holds under committed too (``_TrialSnapshot.assumed``). What that check read is not this
    evaluation's: a value the assumed configuration forces may follow what nothing in this
    configuration connects to the choice, so such a check is never reused, and is read again
    on every change (``publish`` records no admission for it). Under a refused guard, its own
    check true is then enough to admit the choice (the configuration reads it refused through
    its guard); false refuses it with both findings; unresolved keeps it unchecked. Under a
    guard that waits, its own check false refuses it with its own finding (a commit is a
    check of the value against its domain where the choice exists, which a value that can
    never hold there fails at once), and true or unresolved keeps it unchecked, waiting on
    its guard. An inapplicable guard refuses it.

    Only a domain that answers false for the value refuses it: a refusal elsewhere never
    does. Where an input of the domain is itself refused (a derived value refusing on its
    facts), whatever the guard answers, the own check is that refusal and the choice is
    kept (``through``), never read as a value: unresolved in this trial, and refused
    through the input in the configuration published (``Snapshot.input_refused``)."""
    candidate = snapshot._candidates[node.index]
    inactive: NonValue | None = None
    through = False
    if node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
    if not isinstance(inactive, Inapplicable):
        assumed = snapshot.assumed(node) if inactive is not None else None
        if assumed is not None:
            evaluate(assumed, node.index)
            own = assumed.own(node.index)
            through = node.index in assumed._through
            snapshot._read_assumed.add(node.index)
        else:
            own, through = yield from _domain_check(snapshot, node, candidate)
        holds = isinstance(own, Available) and own.value is True
        if through:
            pass  # kept, refused through its input
        elif isinstance(inactive, Unresolved):
            if not isinstance(own, Rejected):
                own = inactive
        elif inactive is not None and not holds and not isinstance(own, Unresolved):
            merged = _blocked([inactive, own])
            assert merged is not None
            own = merged
    else:
        own = inactive
    snapshot._own[node.index] = own
    if through:
        snapshot._through.add(node.index)
        return Evaluation(_unchecked_answer(node))
    if isinstance(own, Available) and own.value is True:
        if node.index in snapshot.unchecked:
            return Evaluation(_unchecked_answer(node))
        snapshot.admit(node.index, candidate)
        return Evaluation(Available(snapshot.assignments[node.index]))
    return Evaluation(own)


def _param_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    yield from ()  # answers at once
    if node.index in snapshot.parameters:
        return Evaluation(Available(snapshot.parameters[node.index]))
    return Evaluation(
        Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "input-missing",
                    node.owner,
                    "optional input was omitted at start",
                ),
            )
        )
    )


def _const_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    yield from ()  # answers at once
    return Evaluation(Available(node.value))


def _decision_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """A Decision: its stored value, its forced value or refusal, or unassigned. A stored
    choice whose own check reads a refused input (``Snapshot.input_refused``) reads, where
    the configuration forces, refused through that input, with the input's findings."""
    if node.index in snapshot.assignments:
        if node.index in snapshot.unchecked:
            if node.index in snapshot.input_refused and snapshot.forcing:
                _, failure = yield from _arguments(
                    snapshot.linked, node.domain_arguments, node.owner
                )
                if isinstance(failure, Rejected):
                    return Evaluation(failure)
            return Evaluation(_unchecked_answer(node))
        return Evaluation(Available(snapshot.assignments[node.index]))
    found = _forced(snapshot, node.index)
    if found is not None:
        return Evaluation(found)
    return Evaluation(
        Unresolved(
            (
                Finding(
                    FindingKind.BLOCKER,
                    "decision-unassigned",
                    node.owner,
                    "decision requires a commitment",
                ),
            )
        )
    )


def _forward_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """An alias or a guard: its output's answer."""
    if node.output is None:
        raise EvaluationError(node.owner, node.kind, "missing output reference")
    return Evaluation(cast(QueryResult[object], (yield node.output)))


def _select_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.selector is None or node.selection_index is None:
        raise EvaluationError(node.owner, "selection", "missing selector")
    selector = cast(QueryResult[object], (yield node.selector))
    if not isinstance(selector, Available):
        return Evaluation(selector)
    case = selector.value
    if type(case) is not str:
        raise EvaluationError(node.owner, "selection", "selector is not a declared case")
    target = node.selection_index.get(case)
    if target is None:
        # The selected candidate is None, or has no such member: absent.
        return Evaluation(Inapplicable())
    return Evaluation(cast(QueryResult[object], (yield target)))


def _view_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """An active view (``_frame`` read its guard first: an inactive view never reads its
    Space's validity): its output, then its Space's validity, combined by
    ``assess_view``."""
    if node.output is None or len(node.constraints) != 1:
        raise EvaluationError(node.owner, "view", "missing output or validity reference")
    output = cast(QueryResult[object], (yield node.output))
    validity = cast(QueryResult[bool], (yield node.constraints[0]))
    view = assess_view(
        output,
        owner=node.key,
        validity=validity,
        validity_owner=snapshot.linked.nodes[node.constraints[0]].key,
        waiting=_waiting(snapshot, node.constraints[0]) if isinstance(validity, Rejected) else None,
    )
    return Evaluation(view.accepted_result, assessment=view)


def _waiting(snapshot: Snapshot, index: int) -> dict[str, QueryResult[object]]:
    """The pieces of the refused Space whose validity node is ``index`` that are still
    unresolved, by key, its placed Spaces' with them: what readiness reports beside a
    refusal, which hides them (false beats unknown)."""
    nodes, found = snapshot.linked.nodes, {}
    pending = [index]
    while pending:
        entry = snapshot.cache.get(pending.pop())
        assessment = None if entry is None else entry.assessment
        if not isinstance(assessment, ConstraintAssessment):
            continue
        for key, answer in assessment.results.items():
            if isinstance(answer, Unresolved):
                found[key] = cast(QueryResult[object], answer)
            elif isinstance(answer, Rejected):
                piece = snapshot.linked.keys[key]
                if nodes[piece].kind == "validity":
                    pending.append(piece)
    return found


def _stored(snapshot: Snapshot, index: int) -> bool:
    """Whether a Decision holds a stored choice here (a trial's candidates are stored)."""
    if isinstance(snapshot, _TrialSnapshot):
        return index in snapshot._candidates
    return index in snapshot.assignments


def _piece(snapshot: Snapshot, piece: Node, answer: QueryResult[object]) -> QueryResult[bool]:
    """A validity piece's answer as a check. A constraint and a placed Space's validity
    answer as themselves. A Decision is its stored choice's own check: true once the
    choice reads as its value, unresolved while unchecked, refused through a refused
    guard or a refused input of its domain. An open Decision is a piece only where it
    has no possible case (forcing refuses it, ``decision-no-viable-case``); otherwise,
    forced or still to choose, it counts as true.

    On forcing's copies (which do not force) an open Decision reads unassigned and a
    stored one as its value or unresolved, so a Decision's piece is false there only
    through its guard: a selector's verdict, which rests on its candidate's validity,
    reads the guards and not the Decisions (``_forcing._refusal_roots``). That is checked
    here, so a change that breaks it fails loudly."""
    if piece.kind != "decision":
        return cast(QueryResult[bool], answer)
    if isinstance(answer, Available):
        return Available(True)
    if isinstance(answer, Rejected) and not snapshot.forcing:
        guard = None if piece.guard is None else snapshot.cache.get(piece.guard)
        if guard is None or not isinstance(guard.result, Rejected):
            raise EvaluationError(
                piece.owner,
                "validity",
                "a Decision's piece is refused on a copy that does not force, its guard not",
            )
    if _stored(snapshot, piece.index) or isinstance(answer, Rejected):
        return cast(QueryResult[bool], answer)
    return Available(True)


def _validity_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """A present Space's validity: Kleene's AND of its pieces (``Node.constraints``):
    refused when one is false, else unresolved when one is unknown, else true. An
    inactive piece (a constraint whose guard does not hold, a placed Space that is
    absent) counts as true. The assessment holds each piece's answer, by key."""
    nodes = snapshot.linked.nodes
    answers: dict[str, QueryResult[bool]] = {}
    for index in node.constraints:
        answer = cast(QueryResult[object], (yield index))
        answers[nodes[index].key] = _piece(snapshot, nodes[index], answer)
    assessment = assess_constraints(answers)
    return Evaluation(cast(QueryResult[object], assessment.result), assessment=assessment)


def _callback_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """A derived value or a constraint: its authored callback, called on its arguments.

    What the callback raises, or a result it returns that fails to snapshot, is
    the owner's failure; an engine invariant is raised as itself.
    """
    arguments, failure = yield from _arguments(snapshot.linked, node.arguments, node.owner)
    if failure is not None:
        return Evaluation(failure, assessment=_guard_assessment(node, failure))
    if node.function is None:
        raise EvaluationError(node.owner, node.kind, "missing callback")
    try:
        positional = (_self_point(snapshot, node.scope),) if node.call_style == "self" else ()
        called = yield _execution.Call(node.function, positional, arguments, node.kind)
        if isinstance(called, _execution._Halt):
            blocked = _blocked(list(called.results))
            assert blocked is not None
            return Evaluation(blocked, assessment=_guard_assessment(node, blocked))
        assert isinstance(called, _execution._Returned)
        result = called.value
        if node.kind == "constraint":
            normalized = constraint_result(cast(bool | QueryResult[bool], result), node.owner)
            assessment = ConstraintAssessment({node.key: normalized}, normalized)
            return Evaluation(cast(QueryResult[object], normalized), assessment=assessment)
        if isinstance(result, (Inapplicable, Rejected, Unresolved)):
            return Evaluation(owned_result(result, node.owner))
    except Exception as cause:
        raise EvaluationError(node.owner, node.kind, str(cause)) from cause
    if node.semantics is None:
        raise EvaluationError(node.owner, node.kind, "derived output has no value semantics")
    value = result.value if isinstance(result, Available) else result
    frozen = snapshot_value(node.semantics, value, owner=node.owner, role=node.kind)
    return Evaluation(Available(frozen))


_FRAMES: Mapping[NodeKind, Callable[[Snapshot, Node], _execution.Frame]] = {
    "param": _param_frame,
    "const": _const_frame,
    "present": _graph_frame,
    "members": _graph_frame,
    "decision": _decision_frame,
    "alias": _forward_frame,
    "guard": _forward_frame,
    "select": _select_frame,
    "view": _view_frame,
    "validity": _validity_frame,
    "derived": _callback_frame,
    "constraint": _callback_frame,
}


def _self_point(snapshot: Snapshot, scope: int) -> Space:
    from .occurrence import attach  # noqa: PLC0415 - scoped callback receiver

    return attach(snapshot, scope)


def evaluate(snapshot: Snapshot, node_index: int) -> Evaluation:
    """Demand one result through the canonical native dispatcher."""
    with snapshot.lock:
        cached = snapshot.cache.get(node_index)
        return cached if cached is not None else _execution.run(snapshot, node_index)


def _membership_frame(
    snapshot: Snapshot, node: Node, value: object, *, check_guard: bool = True
) -> _execution.Frame:
    if check_guard and node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive)
    result, _ = yield from _domain_check(snapshot, node, value)
    return Evaluation(result)


def _domain_check(
    snapshot: Snapshot, node: Node, value: object
) -> Generator[int | _execution.Call, object, tuple[QueryResult[object], bool]]:
    """Whether a Decision's domain admits ``value``, and whether an input of the domain is
    itself refused on its own (then the answer is that refusal, not the domain's): refused
    by a derived value's or a view's body, and through no Decision that reads refused (a
    choice its own check refuses, one with no possible case, one under a refused guard),
    whose refusal is that choice's, answered as before."""
    arguments, failure = yield from _arguments(snapshot.linked, node.domain_arguments, node.owner)
    if failure is not None:
        through = isinstance(failure, Rejected) and not _through_a_choice(
            snapshot, node.domain_arguments
        )
        return failure, through
    if node.domain is None or node.semantics is None:
        raise EvaluationError(node.owner, "domain membership", "reference has no domain")
    candidate = _clone(node, value, owner=node.owner, role="domain candidate snapshot")
    called = yield _execution.Call(
        node.domain.membership,
        (candidate, arguments),
        {"semantics": node.semantics, "owner": node.owner},
        "domain membership",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return blocked, False
    assert isinstance(called, _execution._Returned)
    return cast(QueryResult[object], called.value), False


def _through_a_choice(snapshot: Snapshot, arguments: tuple[Argument, ...]) -> bool:
    """Whether a refused one of ``arguments`` (just read in ``snapshot``) is, or read, a
    Decision that reads refused."""
    nodes, cache = snapshot.linked.nodes, snapshot.cache
    for argument in arguments:
        entry = cache.get(argument.node)
        if entry is None or not isinstance(entry.result, Rejected):
            continue
        own = (argument.node,) if nodes[argument.node].kind == "decision" else ()
        for index in (*own, *entry.decisions):
            read = cache.get(index)
            if read is not None and isinstance(read.result, Rejected):
                return True
    return False


def _enumeration_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive)
    arguments, failure = yield from _arguments(snapshot.linked, node.domain_arguments, node.owner)
    if failure is not None:
        return Evaluation(failure)
    assert node.domain is not None and node.semantics is not None
    if node.domain.candidates is None:
        return Evaluation(Available(None))
    called = yield _execution.Call(
        node.domain.enumerate,
        (arguments,),
        {"semantics": node.semantics, "owner": node.owner},
        "domain enumeration",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return Evaluation(blocked)
    assert isinstance(called, _execution._Returned)
    return Evaluation(cast(QueryResult[object], called.value))


def _applicability(snapshot: Snapshot, node: Node) -> NonValue | None:
    if node.guard is None:
        return None
    return _guard_result(node, evaluate(snapshot, node.guard).result)


def decision_state(snapshot: Snapshot, node_index: int) -> QueryResult[DecisionState[object]]:
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.kind != "decision":
            raise EvaluationError(
                node.owner, "decision state", "reference is not an owning decision"
            )
        inactive = _applicability(snapshot, node)
        if inactive is not None:
            return inactive
        if node_index not in snapshot.assignments:
            return Available(DecisionState(node.owner))
        if node_index in snapshot.unchecked:
            read = evaluate(snapshot, node_index).result
            if isinstance(read, Rejected):
                return read  # kept, refused through a refused input of its domain
            return _unchecked_answer(node)
        value = _clone(
            node, snapshot.assignments[node_index], owner=node.owner, role="decision state"
        )
        return Available(DecisionState(node.owner, "committed", value))


def enumeration(snapshot: Snapshot, node_index: int) -> Evaluation:
    """A Decision's enumerated candidates (``Available(None)``: not enumerable), with
    what the enumeration read."""
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.domain is None or node.semantics is None:
            raise EvaluationError(node.owner, "domain enumeration", "reference has no domain")
        return _execution.run(snapshot, node_index, _enumeration_frame(snapshot, node))


def candidate_values(snapshot: Snapshot, node_index: int) -> QueryResult[tuple[object, ...]] | None:
    result = enumeration(snapshot, node_index).result
    if isinstance(result, Available) and result.value is None:
        return None
    return cast(QueryResult[tuple[object, ...]], result)


def _listed_frame(
    snapshot: Snapshot, node: Node, candidates: tuple[object, ...]
) -> _execution.Frame:
    arguments, failure = yield from _arguments(snapshot.linked, node.domain_arguments, node.owner)
    if failure is not None:
        return Evaluation(failure)
    assert node.domain is not None and node.semantics is not None
    called = yield _execution.Call(
        node.domain.listed_members,
        (candidates, arguments),
        {"semantics": node.semantics, "owner": node.owner},
        "domain membership",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return Evaluation(blocked)
    assert isinstance(called, _execution._Returned)
    return Evaluation(Available(called.value))


def listed_members(
    snapshot: Snapshot, node_index: int, candidates: tuple[object, ...]
) -> Evaluation:
    """Whether a Decision's domain admits each of its listed values ``candidates`` here
    (``Available``: one ``QueryResult[bool]`` each), with what membership read."""
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.domain is None or node.semantics is None:
            raise EvaluationError(node.owner, "domain membership", "reference has no domain")
        return _execution.run(snapshot, node_index, _listed_frame(snapshot, node, candidates))


def membership(snapshot: Snapshot, node_index: int, value: object) -> Evaluation:
    """Whether a Decision's domain admits ``value`` here, with the refusal's finding and
    what membership read."""
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        frame = _membership_frame(snapshot, node, value, check_guard=False)
        return _execution.run(snapshot, node_index, frame)


def copy_result(
    snapshot: Snapshot, node_index: int, answer: QueryResult[object]
) -> QueryResult[object]:
    """Detach a public answer from cached values using its declared semantics."""

    if not isinstance(answer, Available):
        return answer
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.kind == "validity" and type(answer.value) is bool:
            # Aggregate verdicts are intrinsic immutable booleans; these
            # declarations need no separate user-owned value semantics.
            return answer
        return Available(_clone(node, answer.value, owner=node.owner, role="public value snapshot"))


def _copy_readiness(snapshot: Snapshot, assessment: ReadinessAssessment) -> ReadinessAssessment:
    return ReadinessAssessment(
        {
            key: copy_result(snapshot, snapshot.linked.keys[key], answer)
            for key, answer in assessment.results.items()
        },
        assessment.result,
    )


def copy_assessment(snapshot: Snapshot, node_index: int, assessment: Assessment) -> Assessment:
    """Detach every mutable value visible in an assessment from its cache."""

    with snapshot.lock:
        if isinstance(assessment, ConstraintAssessment):
            # Validated constraint values are exact bool, and findings are
            # immutable primitives. Their scalar answers can safely be shared.
            return assessment
        node = snapshot.linked.nodes[node_index]
        output_index = node.output if node.output is not None else node_index
        output = copy_result(snapshot, output_index, assessment.output_result)
        accepted = (
            output
            if assessment.accepted_result is assessment.output_result
            else copy_result(snapshot, node_index, assessment.accepted_result)
        )
        return ViewAssessment(
            output,
            _copy_readiness(snapshot, assessment.readiness),
            assessment.validity,
            accepted,
        )


__all__ = [
    "INDEX_FROM",
    "Evaluation",
    "Readers",
    "Snapshot",
    "cache_factory",
    "candidate_values",
    "copy_result",
    "copy_assessment",
    "decision_state",
    "enumeration",
    "evaluate",
    "fresh",
    "inherit",
    "listed_members",
    "membership",
    "new_cache",
    "reads",
    "unaffected",
]
