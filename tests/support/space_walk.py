# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A seeded random walk of changes over one design space, each step checked against a
recomputation.

A successor published by ``try_with_choices`` is answered in its model's one session,
whose node cache keeps answers across every design space of the model. A recomputation
shares nothing of that: the design space of the same model, facts and stored choices
(dormant ones included), bound from nothing, and evaluated in a session of its own (no
cache for a change; a fresh, empty cache where the choices and facts each answer read
are compared). A design space is its facts and stored choices, so the two must answer
alike: every step compares the successor with its recomputation (the stored choices,
every node's answer and the choices and facts each read, and the validity), and the
step's report with the report of the same change on the recomputation of its base,
asked of the running type directly: accepted or refused, the values refused or dropped, the
rules broken with the choices standing in each one's way, and the choices put dormant,
brought into play and opened. A step of choices that sets a new value and is accepted
holds no false piece through an origin its base did not hold it through. A step that
changes a fact instantiates a new design space, never refused for it: the rules it
reports broken are exactly the origins through which its successor holds a false piece
its base did not. A facts step that also sets a choice is its facts, then its choices as
a change of choices on their design space, which adds no pair: its successor holds no
pair the same facts alone, changed on its base, do not. Every difference is reported
with the step, the change and the first keys that differ; the tallies say what the walk
exercised.

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
    Space,
    ValueHandle,
    inspection,
    occurrence,
)
from finn.core.space._running import questions
from finn.core.space._running.cache import NodeCache
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space._running.surface import At
from finn.core.space._running.value import DesignSpace
from finn.core.space._structure import Structure
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


def _rebound(ds: DesignSpace, session: Session | None = None) -> DesignSpace:
    """The design space of ``ds``'s facts and stored choices bound from nothing (the facts'
    design space with nothing stored): a design space equal to ``ds``, another object. No
    rule refuses it (a design space is its facts and stored choices); each stored value
    must pass its type check there. Bound in ``session``, or in a session of its own with
    no cache."""
    if not ds.stored:
        return ds
    with ds.model.lock:
        saved = list(ds.stored.items())
        report = questions.bind(ds.layer.bare, saved, Session() if session is None else session)
    dropped = [o.owner for o in report.outcomes if o.status == "dropped"]
    assert not dropped and report.published is not None, dropped
    return report.published


def fresh(point: Space) -> Space:
    """The recomputation: the design space of ``point``'s model, facts and stored choices,
    bound from nothing in its model's session (``_rebound``), read at ``point``'s scope. A
    design space equal to ``point``'s: a test asks it in a session of its own to evaluate it
    from nothing."""
    at = occurrence.held(point)
    model = at.ds.model
    return occurrence.place(At(_rebound(at.ds, model.per_class.session()), at.scope))


def _pairs(ds: DesignSpace) -> dict[tuple[str, bool], dict[NodeId, object]]:
    with ds.model.lock:
        return cast(
            "dict[tuple[str, bool], dict[NodeId, object]]",
            questions.pairs(evaluation(ds, Session())),
        )


def held(point: Space) -> set[tuple[str, NodeId]]:
    """Each false piece of the root's validity with each origin it is false through."""
    found = _pairs(occurrence.held(point).ds)
    return {(piece, origin) for (piece, _), origins in found.items() for origin in origins}


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
    facts_refused: int = 0  # of those, the refused ones (by the choices beside them)
    dropped: int = 0  # accepted steps that dropped a stored value the new facts exclude
    made_false: int = 0  # accepted steps whose new facts made a rule false
    batches: int = 0  # accepted facts steps that also set a choice, held to their facts' pairs
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
        linked = inspection.model(start).linked
        self.linked = linked
        self.handles = {i: DecisionHandle[object](linked, i) for i in linked.decisions}
        self.keys = {i: linked.header(i).key for i in linked.decisions}
        # The root formals a step may change, and the values it may give them.
        self.formals = {linked.keys[key]: tuple(values) for key, values in (facts or {}).items()}
        self.by_facts = facts is not None
        self.report = Report()
        self.step = 0
        self.change = ""

    # -- drawing ------------------------------------------------------------------------

    def listed(self, ds: DesignSpace, index: NodeId) -> tuple[object, ...]:
        """A choice's listed values here (its domain over the facts, as if it existed),
        whatever its presence; none where they cannot be listed."""

        def ask(session: Session) -> object:
            ev = evaluation(ds, session)
            node = ev.raw[index]
            return ev.run(lambda: questions.placed(ev, index).listed(node, []))

        found = occurrence.ask(ds.model, ask)
        if isinstance(found, Available) and found.value:
            return tuple(cast(tuple[object, ...], found.value))
        return ()

    def draw(self, point: Space) -> Draw | None:
        """A Decision, stored or open, set to another of its listed values or (a fifth
        of the stored draws) cleared."""
        current = occurrence.held(point).ds
        stored = dict(current.stored)
        pool = list(self.keys)  # declaration order (never by NodeId: ids follow addressing)
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
        current = occurrence.held(point).ds
        index = self.rng.choice(list(self.formals))  # in the order the walk was given them
        options = [v for v in self.formals[index] if not same(v, current.facts.get(index))]
        return {index: self.rng.choice(options)} if options else {}

    def describe(self, drawn: Draw) -> str:
        header = self.linked.header
        parts = [f"fact {header(i).key} = {value!r}" for i, value in drawn.facts.items()]
        parts += [f"{self.keys[i]} = {value!r}" for i, value in drawn.values.items()]
        parts += [f"clear {self.keys[i]}" for i in drawn.clear]
        return "; ".join(parts)

    def apply(self, point: Space, drawn: Draw) -> ConfigurationResult[Space]:
        changes: list[Change[object]] = [point.field(self.handles[i]).clear() for i in drawn.clear]
        values = {self.handles[i]: value for i, value in drawn.values.items()}
        if not self.by_facts:
            return point.try_with_choices(values, *changes)
        linked = self.linked
        facts = {ValueHandle[object](linked, i): value for i, value in drawn.facts.items()}
        return point.try_with_facts(facts, values, *changes)

    def oracle(self, base: Space, drawn: Draw) -> questions.Report:
        """The same change on the recomputation of its base, asked of the running type in
        a session of its own with no cache: the values set, in order, then the clears."""
        ds = _rebound(occurrence.held(base).ds)
        requests: list[tuple[NodeId, object]] = [
            *drawn.values.items(),
            *((i, questions.REMOVE) for i in drawn.clear),
        ]
        with ds.model.lock:
            return questions.change(ds, dict(drawn.facts), requests, Session())

    def sets(self, base: Space, drawn: Draw) -> bool:
        """Whether a change sets a new value: a fact or a choice to a value it does not
        have (a clear, or a commit of the value stored, sets nothing)."""
        current = occurrence.held(base).ds
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
        self, got: ConfigurationResult[Space], want: ConfigurationResult[Space] | questions.Report
    ) -> None:
        if got.accepted != want.accepted:
            self.differ(f"accepted {got.accepted}, recomputation {want.accepted}", ())
            return
        for status in ("refused", "dropped"):
            found = {o.owner: o.result for o in got.outcomes if o.status == status}
            expected = {o.owner: o.result for o in want.outcomes if o.status == status}
            if set(found) != set(expected):
                self.differ(f"{status} owners", set(found) ^ set(expected))
            wrong = [
                owner
                for owner in found
                if owner in expected and not same(found[owner], expected[owner])
            ]
            if wrong:
                self.differ(f"{status} results", wrong)
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
        holds an answer for in its model's session (every node, ``everything``), its
        answer, and the choices and facts it read, no fewer than the recomputation's (read
        in a session of its own, with a fresh cache, so that each answer records them); and
        the validity."""
        tallies = self.report.tallies
        at = occurrence.held(successor)
        model = at.ds.model
        recomputed = _rebound(at.ds)
        keys = self.keys
        if not same(dict(at.ds.stored), dict(recomputed.stored)):
            mine, theirs = at.ds.stored, recomputed.stored
            self.differ("stored choices", (keys[i] for i in set(mine) ^ set(theirs)))
        header = self.linked.header
        wrong: list[str] = []
        unread: list[str] = []
        with model.lock:
            got = evaluation(at.ds, model.per_class.session())
            if self.everything:
                for index in cast(Structure, self.linked).all_nodes():
                    got.get(index)
            want = evaluation(recomputed, Session(cache=NodeCache()))
            for i in sorted(got.memo):
                tallies.nodes_compared += 1
                entry = got.memo[i]
                try:
                    other = want.get(i)
                except Exception as error:  # noqa: BLE001 - the recomputation failed where the successor held an answer
                    wrong.append(f"{header(i).key} (recomputation raised {error!r})")
                    continue
                if not same(entry.result, other.result):
                    wrong.append(header(i).key)
                elif other.reads - entry.reads:
                    unread.append(header(i).key)
            expected = questions.validity(want, at.scope)
        if wrong:
            self.differ("answers", wrong)
        if unread:
            self.differ("reads", unread)
        if not same(inspection.validity(successor), expected):
            self.differ("validity", ())

    def compare_made_false(self, base: Space, result: ConfigurationResult[Space]) -> None:
        """An accepted facts step's rules broken against the origins through which its
        successor holds a false piece its base did not."""
        header = self.linked.header
        new = {header(origin).owner for _, origin in held(result.instance) - held(base)}
        reported = {rule.rule for rule in result.broken}
        if new != reported:
            self.differ("rules made false by the facts", new ^ reported)

    def compare_batch(self, base: Space, drawn: Draw, result: ConfigurationResult[Space]) -> None:
        """An accepted facts step that also sets a choice against its facts alone, changed
        on its base: its choices, a change of choices on the facts' design space, add no
        pair to it."""
        facts = self.apply(base, Draw({}, (), drawn.facts))
        assert facts.accepted, "a change of facts alone is never refused"
        if not self.sets(facts.instance, Draw(drawn.values)):
            return
        self.report.tallies.batches += 1
        header = self.linked.header
        added = held(result.instance) - held(facts.instance)
        if added:
            self.differ(
                "a pair its choices add to its facts' design space, accepted",
                {header(origin).owner for _, origin in added},
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
                tallies.broken += bool(result.broken)
                tallies.facts_refused += bool(drawn.facts)
                continue
            tallies.accepted += 1
            tallies.facts_changed += bool(drawn.facts)
            tallies.dormant += bool(result.dormant)
            tallies.in_play += bool(result.in_play)
            tallies.dropped += any(o.status == "dropped" for o in result.outcomes)
            tallies.made_false += bool(result.broken)
            if drawn.facts:
                self.compare_made_false(base, result)
                self.compare_batch(base, drawn, result)
            elif self.sets(base, drawn) and not held(result.instance) <= held(base):
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
    """``steps`` seeded changes from ``start`` (a design space), each checked against a
    recomputation. A step is accepted or refused as its change is; the walk continues
    from the successor of an accepted one. ``everything``: each successor answers every
    node before the comparison, so that what it answers from the model's cache is
    compared too; otherwise only the nodes the change and the comparison evaluated are
    (a whole Space of many nodes). ``facts``: the root formals a step may change, by key,
    with the values it may give each; every step is then a ``try_with_facts`` and its
    recomputation a binding built from nothing."""
    return _Walk(start, seed, everything=everything, facts=facts).run(steps)


def check(report: Report, *, refused: bool = True, facts: bool = False) -> None:
    """Fail on any difference; and, so the walk cannot pass without exercising the engine,
    unless it accepted, cleared and paired, and (``refused``) refused a step. ``facts``:
    also unless a step that changed a root formal was accepted, one that also set a choice
    among them, and (``refused``) one was refused."""
    assert not report.differences, "\n".join(report.differences[:SHOWN])
    tallies = report.tallies
    assert tallies.accepted and (tallies.refused or not refused), tallies
    assert tallies.clears and tallies.pairs, tallies
    assert not facts or (tallies.facts_changed and tallies.batches), tallies
    assert not (facts and refused) or tallies.facts_refused, tallies
