# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A seeded random walk of changes over one configuration, each step checked against a
recomputation.

A successor built by ``try_with_choices`` inherits evaluations, admissions and verdicts
from its base. A recomputation inherits nothing: a configuration of the same model,
parameters and assignments built from nothing. The two must answer alike, so every
step compares the successor with its oracle, and the step's report with the report of
the same change on the oracle of its base. Every difference is reported with the step,
the change and the first keys that differ; the tallies say what the walk exercised.

A walk given other values for the root's formals (``facts``) also changes them: a step
may change a formal with ``try_with_facts``, and every step then goes through it. Its
oracle is the configuration of the new facts and the same choices, built from nothing: a
fresh binding of the model with the new values, its choices committed as one batch.

One difference of the engine as it stands is kept apart (``Report.known``), found by
these walks: a retained choice, admitted under the base's forced values, can wait on a
selector the change opened. ``check`` fails on every other, and on this only when asked
(``known=True``).

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
    Space,
    Unresolved,
    ValueHandle,
    _forcing,
    _runtime,
    inspection,
    occurrence,
)
from finn.core.space.edits import Change, ConfigurationResult

SHOWN = 5  # the keys a difference names
FACT_STEPS = 0.3  # the share of a walk's steps that change a root formal, where it may


def same(left: object, right: object) -> bool:
    """Equality, falling back to ``repr`` where ``==`` raises."""
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - a value whose equality is not a bool
        return repr(left) == repr(right)


def fresh(point: Space) -> Space:
    """The recomputation: ``point``'s model, parameters and assignments, with no cache,
    no admissions and no verdicts."""
    current = occurrence.state(point)
    snapshot = _runtime.Snapshot(current.model, current.parameters, current.assignments)
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
    gaps: int = 0  # entries of the known gap below, counted
    assignments_rechecked: int = 0
    idle: int = 0  # steps with nothing to draw
    fact_steps: int = 0  # steps that changed a root formal
    facts_changed: int = 0  # of those, the accepted ones
    facts_refused: int = 0  # of those, the refused ones


@dataclass
class Report:
    tallies: Tallies = field(default_factory=Tallies)
    differences: list[str] = field(default_factory=list)
    # What today's engine answers differently from a recomputation, in the way the walks
    # found (a finding for the engine, not a pass): by name, the steps and the first
    # nodes. ``check(known=True)`` fails on it too.
    known: dict[str, list[str]] = field(default_factory=dict)


class _Walk:
    def __init__(
        self,
        start: Space,
        seed: int,
        *,
        everything: bool,
        facts: Mapping[str, Sequence[object]] | None,
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
        for _ in range(200):
            if not pool:
                return None
            i, kind = self.rng.choice(pool)
            if kind == "committed" and self.rng.random() < 0.2:
                return Draw({}, (i,))
            found = _runtime.candidate_values(snapshot, i)
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

    def compare_successor(self, successor: Space, base: Space, drawn: Draw) -> None:
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

        unadmitted: list[str] = []
        waiting: list[str] = []
        for i, value in got.assignments.items():
            tallies.assignments_rechecked += 1
            answer = _runtime.membership(got, i, value).result
            if not (isinstance(answer, Available) and answer.value is True):
                (waiting if isinstance(answer, Unresolved) else unadmitted).append(keys[i])
        if unadmitted:
            self.differ("assignments not re-admitted", unadmitted)
        if waiting:
            self.gap("an assignment waits on a choice the change opened", waiting)

    def gap(self, kind: str, keys: Iterable[str]) -> None:
        shown = sorted(keys)
        self.report.tallies.gaps += len(shown)
        self.report.known.setdefault(kind, []).append(
            f"step {self.step}: {self.change} ({', '.join(shown[:SHOWN])})"
        )

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
            self.compare_successor(result.instance, base, drawn)
            self.point = result.instance
        return self.report


def walk(
    start: Space,
    *,
    seed: int,
    steps: int,
    everything: bool = True,
    facts: Mapping[str, Sequence[object]] | None = None,
) -> Report:
    """``steps`` seeded changes from ``start`` (a configuration), each checked against a
    recomputation. A step is accepted or refused as its change is; the walk continues
    from the successor of an accepted one. ``everything``: each successor answers every
    node before the comparison, so that what it answers from inherited evaluations is
    compared too; otherwise only the nodes the step and the forced Decisions left cached
    are (a whole Space of many nodes). ``facts``: the root formals a step may change,
    by key, with the values it may give each; every step is then a ``try_with_facts``
    and its oracle a binding built from nothing."""
    return _Walk(start, seed, everything=everything, facts=facts).run(steps)


def check(report: Report, *, forced: bool = True, known: bool = False, facts: bool = False) -> None:
    """Fail on any difference; and, so the walk cannot pass without exercising the engine,
    unless it accepted, refused, cleared, paired and (``forced``) moved a forced value.
    ``known``: fail on the engine's known gaps too (``Report.known``). ``facts``: also
    unless a step that changed a root formal was accepted, and one was refused."""
    assert not report.differences, "\n".join(report.differences[:SHOWN])
    assert not (known and report.known), "\n".join(
        f"{kind}: {steps[0]}" for kind, steps in report.known.items()
    )
    tallies = report.tallies
    assert tallies.accepted and tallies.refused, tallies
    assert tallies.clears and tallies.pairs, tallies
    assert tallies.forced_moved or not forced, tallies
    assert not facts or (tallies.facts_changed and tallies.facts_refused), tallies
