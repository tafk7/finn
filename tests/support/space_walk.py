# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A seeded random walk of changes over one configuration, each step checked against a
recomputation.

A successor built by ``try_with_choices`` takes what its world can from its base's
(the entries that read nothing the change touched, others kept by early cutoff) and
shares the model's memo of authored functions. A recomputation shares nothing of the
history: the configuration of the same model, facts and stored choices (dormant ones
included), built directly in a world of its own. A configuration is its facts and stored
choices, so the two must answer alike: every step compares the successor with its
recomputation (the stored choices, every node's answer and the choices and facts each
read, and the validity), and the step's report with the report of the same change on
the recomputation of its base: accepted or refused, the values refused, the rules
broken with the choices standing in each one's way, and the choices put dormant,
brought into play and opened. A step that sets a new value and is accepted holds no
false piece through an origin its base did not hold it through. Every difference is
reported with the step, the change and the first keys that differ; the tallies say what
the walk exercised.

A step sets a choice, stored or open, to one of its domain's listed values (in play,
waiting on a selector still open, or dormant under a case not selected), or clears a
stored one. A walk given other values for the root's formals (``facts``) also changes
them: a step may change a formal with ``try_with_facts``, and every step then goes
through it. Its recomputation is a fresh binding of the model with the new facts.

Engine-only: this module imports ``finn.core.space`` and nothing of FINN. The engine's
own walks (``core.space.test_bounded_change``) and the kernel chain's
(``kernel_ops.test_bounded_change_chain``) share it, imported as ``support.space_walk``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from finn.core.space import (
    Available,
    DecisionHandle,
    Rejected,
    Space,
    ValueHandle,
    inspection,
    occurrence,
)
from finn.core.space._assign import check as type_check
from finn.core.space._assign import placed
from finn.core.space._origins import pairs
from finn.core.space._state import State
from finn.core.space._world import World, listed_values
from finn.core.space.edits import Change, ConfigurationResult
from finn.core.space.ir import NodeId

SHOWN = 5  # the keys a difference names
FACT_STEPS = 0.3  # the share of a walk's steps that change a root formal, where it may


def same(left: object, right: object) -> bool:
    """Equality, falling back to ``repr`` where ``==`` raises."""
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - a value whose equality is not a bool
        return repr(left) == repr(right)


def fresh(point: Space) -> Space:
    """The recomputation: the configuration of ``point``'s model, facts and stored
    choices, built directly in a world of its own (no rule is checked: a configuration is
    its facts and stored choices); each stored value must pass its type check there."""
    current = occurrence.state(point)
    world = World.start(current.evaluator, dict(current.facts), dict(current.stored))
    for index in world.stored:
        found = type_check(world, index)
        assert not isinstance(found, Rejected), (current.linked.nodes[index].key, found)
    return occurrence.attach(State(current.model, world), point._scope)


def held(point: Space) -> set[tuple[str, NodeId]]:
    """Each false piece of the root's validity with each origin it is false through."""
    world = occurrence.state(point).world
    return {(piece, origin) for (piece, _), found in pairs(world).items() for origin in found}


@dataclass(frozen=True)
class Draw:
    """One step's change: values to set and Decisions to clear, by node."""

    values: Mapping[NodeId, object]
    clear: tuple[NodeId, ...] = ()
    facts: Mapping[NodeId, object] = field(default_factory=dict)  # root formals, by node


@dataclass
class Tallies:
    steps: int = 0
    accepted: int = 0
    refused: int = 0
    clears: int = 0
    pairs: int = 0
    nodes_compared: int = 0
    idle: int = 0  # steps with nothing to draw
    fact_steps: int = 0  # steps that changed a root formal
    facts_changed: int = 0  # of those, the accepted ones
    facts_refused: int = 0  # of those, the refused ones
    broken: int = 0  # refused steps that broke a rule
    dormant: int = 0  # accepted steps that put a stored choice out of play
    in_play: int = 0  # accepted steps that brought one back into play


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
    ) -> None:
        self.everything = everything
        self.rng = random.Random(seed)
        self.point = start
        linked = occurrence.state(start).linked
        self.handles = {i: DecisionHandle[object](linked, i) for i in linked.decisions}
        self.keys = {i: linked.nodes[i].key for i in linked.decisions}
        # The root formals a step may change, and the values it may give them.
        self.formals = {linked.keys[key]: tuple(values) for key, values in (facts or {}).items()}
        self.by_facts = facts is not None
        self.report = Report()
        self.step = 0
        self.change = ""

    # -- drawing ------------------------------------------------------------------------

    def listed(self, current: State, index: NodeId) -> tuple[object, ...]:
        """A choice's listed values here (its domain over the facts, as if it existed),
        whatever its presence; none where they cannot be listed."""
        found = listed_values(placed(current.world, index), index)
        if isinstance(found, Available) and found.value:
            return tuple(cast(tuple[object, ...], found.value))
        return ()

    def draw(self, point: Space) -> Draw | None:
        """A Decision, stored or open, set to another of its listed values or (a fifth
        of the stored draws) cleared."""
        current = occurrence.state(point)
        stored = dict(current.stored)
        pool = sorted(self.keys)
        for _ in range(200):
            i = self.rng.choice(pool)
            if i in stored and self.rng.random() < 0.2:
                return Draw({}, (i,))
            listed = self.listed(current, i)
            if not listed:
                continue
            options = [c for c in listed if not same(c, stored.get(i))] or list(listed)
            return Draw({i: self.rng.choice(options)})
        return None

    def draw_fact(self, point: Space) -> dict[NodeId, object]:
        """A root formal set to another of the values the walk was given for it."""
        current = occurrence.state(point)
        index = self.rng.choice(sorted(self.formals))
        options = [v for v in self.formals[index] if not same(v, current.facts.get(index))]
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
        """The same change on the recomputation of its base."""
        return self.apply(fresh(base), drawn)

    def sets(self, base: Space, drawn: Draw) -> bool:
        """Whether a change sets a new value: a fact or a choice to a value it does not
        have (a clear, or a commit of the value stored, sets nothing)."""
        current = occurrence.state(base)
        missing = object()
        return any(
            not same(current.facts.get(i, missing), v) for i, v in drawn.facts.items()
        ) or any(not same(current.stored.get(i, missing), v) for i, v in drawn.values.items())

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
        rules = {(rule.rule, rule.standing) for rule in got.broken}
        expected_rules = {(rule.rule, rule.standing) for rule in want.broken}
        if rules != expected_rules:
            self.differ("broken rules", (rule for rule, _ in rules ^ expected_rules))
        for name in ("dormant", "in_play", "opened"):
            moved, expected_moved = set(getattr(got, name)), set(getattr(want, name))
            if moved != expected_moved:
                self.differ(name, moved ^ expected_moved)

    def compare_successor(self, successor: Space) -> None:
        """The successor against its recomputation: the stored choices; every node it
        holds an entry for (every node, ``everything``), its answer, and the choices and
        facts it read, no fewer than the recomputation's; and the validity."""
        tallies = self.report.tallies
        recomputed = fresh(successor)
        got, want = occurrence.state(successor), occurrence.state(recomputed)
        keys = self.keys
        if not same(dict(got.stored), dict(want.stored)):
            self.differ("stored choices", (keys[i] for i in set(got.stored) ^ set(want.stored)))
        nodes = got.linked.nodes
        if self.everything:
            for i in range(len(nodes)):
                got.world.entry(i)
        wrong: list[str] = []
        unread: list[str] = []
        for i in sorted(got.world.cache):
            tallies.nodes_compared += 1
            entry = got.world.cache[i]
            try:
                other = want.world.entry(i)
            except Exception as error:  # noqa: BLE001 - the recomputation failed where the successor held an answer
                wrong.append(f"{nodes[i].key} (recomputation raised {error!r})")
                continue
            if not same(entry.result, other.result):
                wrong.append(nodes[i].key)
            elif other.reads & ~entry.reads:
                unread.append(nodes[i].key)
        if wrong:
            self.differ("answers", wrong)
        if unread:
            self.differ("reads", unread)
        if not same(inspection.validity(successor), inspection.validity(recomputed)):
            self.differ("validity", ())

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
                tallies.broken += bool(result.broken)
                tallies.facts_refused += bool(drawn.facts)
                continue
            tallies.accepted += 1
            tallies.facts_changed += bool(drawn.facts)
            tallies.dormant += bool(result.dormant)
            tallies.in_play += bool(result.in_play)
            if self.sets(base, drawn) and not held(result.instance) <= held(base):
                self.differ("a false piece through a new origin, accepted", ())
            self.compare_successor(result.instance)
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
    node before the comparison, so that what it answers from inherited entries is
    compared too; otherwise only the nodes the change and the comparison evaluated are
    (a whole Space of many nodes). ``facts``: the root formals a step may change, by key,
    with the values it may give each; every step is then a ``try_with_facts`` and its
    recomputation a binding built from nothing."""
    return _Walk(start, seed, everything=everything, facts=facts).run(steps)


def check(report: Report, *, refused: bool = True, facts: bool = False) -> None:
    """Fail on any difference; and, so the walk cannot pass without exercising the engine,
    unless it accepted, cleared and paired, and (``refused``) refused a step. ``facts``:
    also unless a step that changed a root formal was accepted, and (``refused``) one was
    refused."""
    assert not report.differences, "\n".join(report.differences[:SHOWN])
    tallies = report.tallies
    assert tallies.accepted and (tallies.refused or not refused), tallies
    assert tallies.clears and tallies.pairs, tallies
    assert not facts or tallies.facts_changed, tallies
    assert not (facts and refused) or tallies.facts_refused, tallies
