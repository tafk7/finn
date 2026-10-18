# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A seeded random walk of changes over one configuration, each step checked against a
recomputation.

A successor built by ``try_with_choices`` inherits evaluations, admissions and verdicts
from its base. A recomputation inherits nothing: a binding of the same model and
parameters with the successor's choices committed as one batch. The two must answer alike,
so every step compares the successor with its oracle, and the step's report with the
report of the same change on the oracle of its base. Every difference is reported with the
step, the change and the first keys that differ; the tallies say what the walk exercised.

A walk given other values for the root's formals (``facts``) also changes them: a step
may change a formal with ``try_with_facts``, and every step then goes through it. Its
oracle is the configuration of the new facts and the same choices, built from nothing: a
fresh binding of the model with the new values, its choices committed as one batch.

A choice whose own check cannot conclude (it reads a choice still open) is kept
**unchecked**: stored, and read unresolved. A step may commit one (``waiting``: a choice
under a selector the configuration leaves open, whose cases the walk learned from the
configurations it could reach), and the steps after it open the choice it reads or close it:
the choice comes out checked, or stays unchecked, or the change is refused. Every successor
is compared with its recomputation on which choices are unchecked, what they wait on, and
every answer, and each unchecked choice's blockers are checked to be open or unsupplied.

Engine-only: this module imports ``finn.core.space`` and nothing of FINN. The engine's
own walks (``core.space.test_bounded_change``) and the kernel chain's
(``kernel_ops.test_bounded_change_chain``) share it, imported as ``support.space_walk``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from finn.core.space import (
    Available,
    DecisionHandle,
    Rejected,
    Space,
    Unresolved,
    ValueHandle,
    _forcing,
    _runtime,
    inspection,
    occurrence,
)
from finn.core.space.edits import Change, ConfigurationResult
from finn.core.space.ir import Argument

SHOWN = 5  # the keys a difference names
FACT_STEPS = 0.3  # the share of a walk's steps that change a root formal, where it may


def same(left: object, right: object) -> bool:
    """Equality, falling back to ``repr`` where ``==`` raises."""
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - a value whose equality is not a bool
        return repr(left) == repr(right)


def fresh(point: Space) -> Space:
    """The recomputation: ``point``'s model, parameters and assignments, with no cache, no
    admissions and no verdicts. Which of the assignments are unchecked is found from nothing
    too: a binding of the model commits them as one batch, which must accept what ``point``
    holds."""
    current = occurrence.state(point)
    linked = current.linked
    empty = occurrence.bind(current.model, current.parameters)
    values = {DecisionHandle[object](linked, i): v for i, v in current.assignments.items()}
    report = empty.try_with_choices(values)
    assert report.accepted, [(o.owner, o.status, o.result) for o in report.outcomes]
    unchecked = occurrence.state(report.instance).unchecked
    snapshot = _runtime.Snapshot(
        current.model, current.parameters, current.assignments, unchecked=unchecked
    )
    return occurrence.attach(snapshot, point._scope)


@dataclass(frozen=True)
class Draw:
    """One step's change: values to commit and Decisions to clear, by node."""

    values: Mapping[int, object]
    clear: tuple[int, ...] = ()
    facts: Mapping[int, object] = field(default_factory=dict)  # root formals, by node


@dataclass
class Tallies:
    steps: int = 0
    accepted: int = 0
    refused: int = 0
    clears: int = 0
    pairs: int = 0
    forced_moved: int = 0  # steps that moved the forced value of a Decision they did not name
    nodes_compared: int = 0
    assignments_rechecked: int = 0
    unchecked_held: int = 0  # unchecked choices the successors held, counted at each step
    unchecked_kept: int = 0  # steps that kept a choice unchecked that was not, or newly
    unchecked_checked: int = 0  # steps that made an unchecked choice checked
    idle: int = 0  # steps with nothing to draw
    fact_steps: int = 0  # steps that changed a root formal
    facts_changed: int = 0  # of those, the accepted ones
    facts_refused: int = 0  # of those, the refused ones


@dataclass
class Report:
    tallies: Tallies = field(default_factory=Tallies)
    differences: list[str] = field(default_factory=list)


class _Walk:
    def __init__(
        self,
        start: Space,
        seed: int,
        *,
        everything: bool,
        facts: Mapping[str, Sequence[object]] | None,
        waiting: bool,
    ) -> None:
        self.everything = everything
        self.rng = random.Random(seed)
        self.point = start
        linked = occurrence.state(start).linked
        self.handles = {i: DecisionHandle[object](linked, i) for i in linked.decisions}
        self.keys = {i: linked.nodes[i].key for i in linked.decisions}
        self.index = {key: i for i, key in self.keys.items()}
        # The root formals a step may change, and the values it may give them.
        self.formals = {linked.keys[key]: tuple(values) for key, values in (facts or {}).items()}
        self.by_facts = facts is not None
        self.report = Report()
        self.step = 0
        self.change = ""
        # The cases of the choices the start leaves waiting on a selector, learned from the
        # configurations that commit the selectors: where a step may name one.
        self.cases = self.learn(start) if waiting else {}

    def learn(self, start: Space) -> dict[int, tuple[object, ...]]:
        """The cases of every Decision, as listed in ``start`` and in the configurations that
        commit a selector of it to each of its cases (breadth first, a few deep)."""
        cases: dict[int, tuple[object, ...]] = {}
        seen = 0
        pending = [start]
        while pending and seen < 40:
            point = pending.pop(0)
            seen += 1
            snapshot = occurrence.state(point)
            for i in self.keys:
                found = _runtime.candidate_values(snapshot, i)
                if isinstance(found, Available) and found.value and i not in cases:
                    cases[i] = tuple(found.value)
            for item in inspection.decisions(point):
                if item.selector and item.key in self.index:
                    for case in item.cases:
                        report = point.try_with_choices({item.reference: case})
                        if report.accepted:
                            pending.append(report.instance)
        return cases

    # -- drawing ------------------------------------------------------------------------

    def draw(self, point: Space) -> Draw | None:
        """A Decision that is committed, forced, or open with viable cases, set to another
        of its enumerated cases or (a fifth of the committed draws) cleared."""
        snapshot = occurrence.state(point)
        committed = dict(snapshot.assignments)
        forced = _forcing.forced(snapshot).values
        viable = {self.index[item.key] for item in inspection.viable(point)}
        pool = [(i, "committed") for i in committed] + [(i, "forced") for i in forced]
        pool += [(i, "open") for i in viable]
        # A choice the configuration cannot list the cases of (its selector is open): its
        # change is kept unchecked where its own check cannot conclude, refused where it fails.
        listed = {i for i, _ in pool}
        pool += [(i, "waiting") for i in sorted(self.cases) if i not in listed]
        for _ in range(200):
            if not pool:
                return None
            i, kind = self.rng.choice(pool)
            if kind == "committed" and self.rng.random() < 0.2:
                return Draw({}, (i,))
            found = _runtime.candidate_values(snapshot, i)
            if i in self.cases and not (isinstance(found, Available) and found.value):
                return Draw({i: self.rng.choice(self.cases[i])})
            if not isinstance(found, Available) or not found.value:
                continue
            current = committed[i] if i in committed else forced.get(i)
            options = [c for c in found.value if not same(c, current)] or list(found.value)
            return Draw({i: self.rng.choice(options)})
        return None

    def draw_fact(self, point: Space) -> dict[int, object]:
        """A root formal set to another of the values the walk was given for it."""
        snapshot = occurrence.state(point)
        index = self.rng.choice(sorted(self.formals))
        current = snapshot.parameters.get(index)
        options = [v for v in self.formals[index] if not same(v, current)]
        return {index: self.rng.choice(options)} if options else {}

    def describe(self, drawn: Draw) -> str:
        nodes = occurrence.state(self.point).linked.nodes
        parts = [f"fact {nodes[i].key} = {value!r}" for i, value in drawn.facts.items()]
        parts += [f"{self.keys[i]} = {value!r}" for i, value in drawn.values.items()]
        parts += [f"clear {self.keys[i]}" for i in drawn.clear]
        return "; ".join(parts)

    def apply(self, point: Space, drawn: Draw) -> ConfigurationResult[Space]:
        changes: list[Change[object]] = [point.field(self.handles[i]).clear() for i in drawn.clear]
        values = {self.handles[i]: value for i, value in drawn.values.items()}
        if not self.by_facts:
            return point.try_with_choices(values, *changes)
        linked = occurrence.state(point).linked
        facts = {ValueHandle[object](linked, i): value for i, value in drawn.facts.items()}
        return point.try_with_facts(facts, values, *changes)

    def oracle(self, base: Space, drawn: Draw) -> ConfigurationResult[Space]:
        """The same change on a configuration built from nothing: a fresh binding of the
        model with the new facts and every choice the change leaves, committed as one batch."""
        current = occurrence.state(base)
        if not self.by_facts:
            return self.apply(fresh(base), drawn)
        merged = {i: v for i, v in current.assignments.items() if i not in drawn.clear}
        merged.update(drawn.values)
        empty = occurrence.bind(current.model, {**current.parameters, **drawn.facts})
        values = {self.handles[i]: value for i, value in merged.items()}
        return empty.try_with_facts({}, values)

    # -- comparing ----------------------------------------------------------------------

    def differ(self, kind: str, keys: Iterable[str]) -> None:
        shown = sorted(keys)[:SHOWN]
        self.report.differences.append(
            f"step {self.step}, {kind}: {self.change}" + (f" ({', '.join(shown)})" if shown else "")
        )

    def compare_reports(
        self, got: ConfigurationResult[Space], want: ConfigurationResult[Space]
    ) -> None:
        if got.accepted != want.accepted:
            self.differ(f"accepted {got.accepted}, recomputation {want.accepted}", ())
            return
        # A choice the change names is reported unchecked as the recomputation reports it (which
        # names every choice, since it commits them all as one batch), and with the same findings.
        theirs = {o.owner: o for o in want.outcomes}
        named = [o for o in got.outcomes if o.requested and o.owner in theirs]
        if any(o.unchecked != theirs[o.owner].unchecked for o in named) or any(
            o.unchecked and not same(o.result, theirs[o.owner].result) for o in named
        ):
            self.differ(
                "unchecked choices reported",
                {o.owner for o in named if o.unchecked != theirs[o.owner].unchecked},
            )
        refused = {o.owner: o.result for o in got.outcomes if o.status == "refused"}
        expected = {o.owner: o.result for o in want.outcomes if o.status == "refused"}
        if set(refused) != set(expected):
            self.differ("refusing owners", set(refused) ^ set(expected))
        wrong = [
            owner
            for owner in refused
            if owner in expected and not same(refused[owner], expected[owner])
        ]
        if wrong:
            self.differ("refusal results", wrong)

    def compare_successor(
        self, successor: Space, base: Space, drawn: Draw, report: ConfigurationResult[Space]
    ) -> None:
        tallies = self.report.tallies
        got, want = occurrence.state(successor), occurrence.state(fresh(successor))
        keys = self.keys
        named = {*drawn.values, *drawn.clear}

        if not same(dict(got.assignments), dict(want.assignments)):
            self.differ(
                "assignments", (keys[i] for i in set(got.assignments) ^ set(want.assignments))
            )

        found, expected = _forcing.forced(got), _forcing.forced(want)
        if not same(dict(found.values), dict(expected.values)):
            moved = {
                i
                for i in {*found.values, *expected.values}
                if not same(found.values.get(i), expected.values.get(i))
            }
            self.differ("forced values", (keys[i] for i in moved))
        if set(found.refused) != set(expected.refused):
            self.differ(
                "refused Decisions", (keys[i] for i in set(found.refused) ^ set(expected.refused))
            )

        viable = [(item.key, item.cases) for item in inspection.viable(successor)]
        wanted = [
            (item.key, item.cases)
            for item in inspection.viable(occurrence.attach(want, successor._scope))
        ]
        if not same(viable, wanted):
            self.differ("viable", {key for key, _ in viable} ^ {key for key, _ in wanted})

        before = _forcing.forced(occurrence.state(base)).values
        if any(
            i not in named and not same(before.get(i), found.values.get(i))
            for i in {*before, *found.values}
        ):
            tallies.forced_moved += 1

        # Everything the successor can answer is read on it (``walk``'s ``everything``).
        for i in range(len(got.linked.nodes)) if self.everything else ():
            _runtime.evaluate(got, i)
        nodes = got.linked.nodes
        entries = dict(got.cache)
        others: dict[int, _runtime.Evaluation] = {}
        wrong: list[str] = []
        for i, entry in entries.items():
            tallies.nodes_compared += 1
            try:
                others[i] = _runtime.evaluate(want, i)
            except Exception as error:  # noqa: BLE001 - the recomputation failed where the successor held an answer
                wrong.append(f"{nodes[i].key} (recomputation raised {error!r})")
        wrong += [
            nodes[i].key for i, other in others.items() if not same(entries[i].result, other.result)
        ]
        unread = [
            nodes[i].key
            for i, other in others.items()
            if not other.decisions <= entries[i].decisions
        ]
        if wrong:
            self.differ("cached results", wrong)
        if unread:
            self.differ("cached evidence", unread)

        self.compare_unchecked(successor, base, occurrence.attach(want, successor._scope), report)

    def compare_unchecked(
        self,
        successor: Space,
        base: Space,
        recomputed: Space,
        report: ConfigurationResult[Space],
    ) -> None:
        """Which choices the successor holds unchecked and what each waits on, against the
        recomputation; and that every other stored choice's own check holds, and every
        unchecked one's cannot conclude or its guard waits (and its own check is not false),
        on the successor. Under a guard that does not hold, the own check is read with the
        guard's cases assumed (``own_check``)."""
        tallies = self.report.tallies
        got, keys = occurrence.state(successor), self.keys
        unchecked = inspection.unchecked(successor)
        if not same(unchecked, inspection.unchecked(recomputed)):
            self.differ("unchecked choices", {item.key for item in unchecked})
        if set(got.unchecked) != {self.index[item.key] for item in unchecked}:
            self.differ("unchecked choices listed", {item.key for item in unchecked})
        unadmitted: list[str] = []
        for i, value in got.assignments.items():
            tallies.assignments_rechecked += 1
            guard = got.linked.nodes[i].guard
            applies = Available(True) if guard is None else _runtime.evaluate(got, guard).result
            answer = (
                self.own_check(got, i, value)
                if isinstance(applies, (Rejected, Unresolved))
                else _runtime.membership(got, i, value).result
            )
            if i in got.unchecked:
                # Unresolved: its guard waits (and its own check is not false), or its own
                # check does.
                if not (
                    (isinstance(applies, Unresolved) and not isinstance(answer, Rejected))
                    or isinstance(answer, Unresolved)
                ):
                    unadmitted.append(keys[i])
            elif not (
                isinstance(answer, Available)
                and answer.value is True
                and (isinstance(applies, (Available, Rejected)))
            ):
                unadmitted.append(keys[i])
        if unadmitted:
            self.differ("assignments not as their own check says", unadmitted)
        for item in unchecked:
            blocked = [
                key
                for key in item.blockers
                if isinstance(_runtime.evaluate(got, got.linked.keys[key]).result, Available)
            ]
            if not item.blockers or blocked:
                self.differ("unchecked choice waits on nothing open", [item.key, *blocked])
        before = {item.key for item in inspection.unchecked(base)}
        now = {item.key for item in unchecked}
        flagged = {o.owner for o in report.outcomes if o.unchecked}
        if flagged - now or (now - before) - flagged:
            self.differ("unchecked choices reported", flagged ^ (now - before))
        tallies.unchecked_held += len(unchecked)
        tallies.unchecked_kept += bool({item.key for item in unchecked} - before)
        tallies.unchecked_checked += bool(before - {item.key for item in unchecked})

    @staticmethod
    def own_check(got: _runtime.Snapshot, i: int, value: object) -> object:
        """A stored choice's own check as the configuration reads it where the choice exists:
        in a snapshot of the same model, facts and choices with each selector case its guard
        sits under committed too (where that selector is not committed), built from nothing;
        in place where there is none."""
        linked = got.linked
        nodes = linked.nodes
        # The node that holds where a selector is one of its cases, for each case's scope.
        held: dict[int, tuple[int, str]] = {}
        for choice in linked.choices:
            selected = (Argument("selected", choice.selector),)
            for case, scope in choice.cases:
                guard = None if scope is None else linked.scopes[scope].guard
                while guard is not None and nodes[guard].arguments != selected:
                    guard = nodes[guard].guard
                if guard is not None:
                    held[guard] = (choice.selector, case)
        cases: dict[int, object] = {}
        guard = nodes[i].guard
        while guard is not None:
            if guard in held and held[guard][0] not in got.assignments:
                cases[held[guard][0]] = held[guard][1]
            guard = nodes[guard].guard
        if not cases:
            return _runtime.membership(got, i, value).result
        assumed = _runtime.Snapshot(
            got.model, got.parameters, {**got.assignments, **cases}, unchecked=got.unchecked
        )
        return _runtime.membership(assumed, i, value).result

    # -- the walk -----------------------------------------------------------------------

    def run(self, steps: int) -> Report:
        tallies = self.report.tallies
        for self.step in range(steps):
            base = self.point
            drawn = self.draw(base)
            if drawn is None:
                tallies.idle += 1
                continue
            if self.rng.random() < 0.1:
                more = self.draw(base)
                if more is not None and not (
                    {*more.values, *more.clear} & {*drawn.values, *drawn.clear}
                ):
                    drawn = Draw({**drawn.values, **more.values}, (*drawn.clear, *more.clear))
                    tallies.pairs += 1
            if self.formals and self.rng.random() < FACT_STEPS:
                drawn = Draw(drawn.values, drawn.clear, self.draw_fact(base))
                if drawn.facts:
                    tallies.fact_steps += 1
            self.change = self.describe(drawn)
            tallies.steps += 1
            tallies.clears += bool(drawn.clear)
            result = self.apply(base, drawn)
            self.compare_reports(result, self.oracle(base, drawn))
            if not result.accepted:
                tallies.refused += 1
                tallies.facts_refused += bool(drawn.facts)
                continue
            tallies.accepted += 1
            tallies.facts_changed += bool(drawn.facts)
            self.compare_successor(result.instance, base, drawn, result)
            self.point = result.instance
        return self.report


def walk(
    start: Space,
    *,
    seed: int,
    steps: int,
    everything: bool = True,
    facts: Mapping[str, Sequence[object]] | None = None,
    waiting: bool = True,
) -> Report:
    """``steps`` seeded changes from ``start`` (a configuration), each checked against a
    recomputation. A step is accepted or refused as its change is; the walk continues
    from the successor of an accepted one. ``everything``: each successor answers every
    node before the comparison, so that what it answers from inherited evaluations is
    compared too; otherwise only the nodes the step and the forced Decisions left cached
    are (a whole Space of many nodes). ``facts``: the root formals a step may change,
    by key, with the values it may give each; every step is then a ``try_with_facts``
    and its oracle a binding built from nothing. ``waiting``: a step may also name a choice
    the configuration cannot list the cases of (its selector is open), which is kept
    unchecked where its own check cannot conclude."""
    return _Walk(start, seed, everything=everything, facts=facts, waiting=waiting).run(steps)


def check(
    report: Report, *, forced: bool = True, facts: bool = False, unchecked: bool = False
) -> None:
    """Fail on any difference; and, so the walk cannot pass without exercising the engine,
    unless it accepted, refused, cleared, paired and (``forced``) moved a forced value.
    ``facts``: also unless a step that changed a root formal was accepted, and one was
    refused. ``unchecked``: also unless a step kept a choice unchecked and a later one made
    an unchecked choice checked."""
    assert not report.differences, "\n".join(report.differences[:SHOWN])
    tallies = report.tallies
    assert tallies.accepted and tallies.refused, tallies
    assert tallies.clears and tallies.pairs, tallies
    assert tallies.forced_moved or not forced, tallies
    assert not facts or (tallies.facts_changed and tallies.facts_refused), tallies
    assert not unchecked or (tallies.unchecked_kept and tallies.unchecked_checked), tallies
