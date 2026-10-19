# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change: choices set, replaced or cleared, and facts, as one batch.

A change is **assignment**. In order:

1. **Identities.** Every item names a choice of this configuration, once (a ``Change``
   of this very configuration); facts name root formals, once. A malformed or foreign
   item is a ``RequestError``, before anything is evaluated.
2. **Recognition**, then **snapshot**: every supplied value is recognized by its
   semantics, all before any is frozen (``_state.prepare``).
3. **Same values.** Where every fact and stored value equals the configuration's, the
   configuration is kept, each item ``"unchanged"``.
4. **The configuration it would publish**: the stored choices, less those cleared, with
   those set; the facts with those given; its world made from the receiver's.
5. **Type checks.** Each value set is checked against its domain, which reads facts
   only (``_world.type_check``): outside it, or an input of the domain refusing or not
   applying, refuses the change; a missing fact leaves the choice stored, waiting on it.
   A choice whose scope is absent or waits is checked where the selectors above it read
   its case (its domain's inputs exist only there); where another guard decides its
   presence, it is accepted and checked when it comes into play: a stored choice the
   change brings into play is checked again. A facts change checks every stored value
   again, dormant ones included.
6. **Publication:** with nothing refused, the configuration is published whole; else
   nothing is, and each item's outcome says why. Nothing is cleared or filled in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ._state import State, prepare, presence
from ._world import FLATTEN, World, type_check
from .edits import Change, ChangeOutcome
from .errors import RequestError
from .ir import Node, NodeId
from .results import Available, Inapplicable, QueryResult, Unresolved
from .semantics import equal, snapshot

_MISSING = object()


@dataclass(frozen=True, slots=True)
class Assigned:
    """What a change came to: the configuration it publishes (the receiver's state
    where it changes nothing; ``None`` where it is refused) and each item's outcome, in
    order: the facts given, the choices named, then the stored choices the change checked
    again (a facts change, or brought into play) and found refusing."""

    state: State | None
    outcomes: tuple[ChangeOutcome, ...]

    @property
    def accepted(self) -> bool:
        return self.state is not None


def normalize(current: State, changes: Sequence[Change[object]]) -> dict[NodeId, Change[object]]:
    """Each change by its choice: every one of this configuration, naming an owned
    Decision of its scope, once."""
    linked = current.linked
    pending: dict[NodeId, Change[object]] = {}
    for item in changes:
        if not isinstance(item, Change):
            raise RequestError("changes must come from a bound decision field")
        if item.base is not current:
            raise RequestError("all changes must target this exact base snapshot")
        if type(item.scope) is not int or not 0 <= item.scope < len(linked.scopes):
            raise RequestError("change scope does not belong to this model")
        if type(item.node) is not int or not 0 <= item.node < len(linked.nodes):
            raise RequestError("change node does not belong to this model")
        node = linked.nodes[item.node]
        if node.kind != "decision" or node.scope != item.scope:
            raise RequestError("change does not identify an owned decision in its scope")
        if item.node in pending:
            raise RequestError(f"duplicate change for {node.key}")
        pending[item.node] = item
    return pending


def _equal(node: Node, left: object, right: object, role: str, *, detach: bool) -> bool:
    """Whether two values of ``node`` are equal by its semantics (a choice's each
    detached first, so equality cannot change a stored value)."""
    if left is right:
        return True
    semantics = node.semantics
    assert semantics is not None
    if detach:
        left = snapshot(semantics, left, owner=node.owner, role=role)
        right = snapshot(semantics, right, owner=node.owner, role=role)
    return equal(semantics, left, right, owner=node.owner, role=role)


def placed(world: World, index: NodeId) -> World | QueryResult[object]:
    """Where a choice's type check is made: ``world``, where the choice is present there;
    else the world where every selector above it reads the case that holds it, if the
    choice is present there; else why it cannot be checked yet, from its guard
    (inapplicable: it is absent; unresolved: its presence waits)."""
    evaluator = world.evaluator
    node = evaluator.nodes[index]
    if node.guard is None:
        return world
    guard = world.result(node.guard)
    if isinstance(guard, Available) and guard.value is True:
        return world
    reads = evaluator.reads
    cases = {
        held[0]: held[1]
        for held in (reads.cases.get(item) for item in reads.guards(index))
        if held is not None
    }
    if any(world.stored.get(selector) != case for selector, case in cases.items()):
        place = world.successor(stored={**world.stored, **cases})
        guard = place.result(node.guard)
        if isinstance(guard, Available) and guard.value is True:
            return place
    if isinstance(guard, Unresolved):
        return guard
    return Inapplicable()


def check(world: World, index: NodeId) -> QueryResult[bool]:
    """A stored value's type check, made where its scope is in play (``placed``)."""
    place = placed(world, index)
    if not isinstance(place, World):
        return cast(QueryResult[bool], place)
    return cast(QueryResult[bool], type_check(place, index, world.values[index]))


def _into_play(before: World, after: World, index: NodeId) -> bool:
    """Whether a stored choice the change does not name comes into play: present after,
    and not before (its type check may not have been made where it now is)."""
    if before.evaluator.nodes[index].guard is None:
        return False
    return presence(after, index)[0] == "present" and presence(before, index)[0] != "present"


def _refuses(result: QueryResult[bool]) -> bool:
    """A type check that refuses the change: anything but true, unresolved (a missing
    fact, or presence still waiting) or inapplicable (the scope absent: checked when it
    comes into play)."""
    return not isinstance(result, (Unresolved, Inapplicable)) and not (
        isinstance(result, Available) and result.value is True
    )


def assign(
    current: State,
    facts: Mapping[NodeId, object],
    requests: Sequence[Change[object]],
) -> Assigned:
    """The change of ``facts`` (by root formal) and ``requests`` (in the caller's order)
    from ``current``, under the model's lock."""
    linked = current.linked
    nodes = linked.nodes
    normalized = normalize(current, requests)
    with current.evaluator.lock:
        values = prepare(
            linked,
            {
                **{index: (value, "parameter") for index, value in facts.items()},
                **{
                    index: (item.value, "choice")
                    for index, item in normalized.items()
                    if not item.remove
                },
            },
        )
        held_facts, held = current.world.facts, current.world.stored
        given: dict[NodeId, object] = {}
        changed_facts: set[NodeId] = set()
        for index in facts:
            old = held_facts.get(index, _MISSING)
            if old is not _MISSING and _equal(
                nodes[index], old, values[index], "parameter equality", detach=False
            ):
                given[index] = old  # the same fact: the held object, so nothing reads it anew
            else:
                given[index] = values[index]
                changed_facts.add(index)
        stored = dict(held)
        for index, item in normalized.items():
            if item.remove:
                stored.pop(index, None)
                continue
            old = held.get(index, _MISSING)
            if old is not _MISSING and _equal(
                nodes[index], values[index], old, "configuration equality", detach=True
            ):
                stored[index] = old  # the same value: the held object
            else:
                stored[index] = values[index]
        fact_outcomes = [
            ChangeOutcome(nodes[index].owner, Available(True), "unchanged") for index in facts
        ]
        if (
            not changed_facts
            and stored.keys() == held.keys()
            and all(stored[index] is held[index] for index in stored)
        ):
            unchanged = (
                ChangeOutcome(nodes[item.node].owner, Available(True), "unchanged")
                for item in requests
            )
            return Assigned(current, (*fact_outcomes, *unchanged))
        world = current.world.successor(
            facts={**held_facts, **given} if changed_facts else None, stored=stored
        )
        checks = {
            index: check(world, index) for index, item in normalized.items() if not item.remove
        }
        retained: dict[NodeId, QueryResult[bool]] = {}
        for index in sorted(stored, key=linked.ranks.__getitem__):
            if index in normalized:
                continue
            if changed_facts or _into_play(current.world, world, index):
                retained[index] = check(world, index)
        refused = {
            index for index, result in (*checks.items(), *retained.items()) if _refuses(result)
        }
        if refused:
            outcomes = [
                ChangeOutcome(outcome.owner, outcome.result, "checked") for outcome in fact_outcomes
            ]
            for item in requests:
                status: Literal["refused", "checked"] = (
                    "refused" if item.node in refused else "checked"
                )
                result = checks.get(item.node, Available(True))
                outcomes.append(ChangeOutcome(nodes[item.node].owner, result, status))
            outcomes.extend(
                ChangeOutcome(nodes[index].owner, result, "refused", requested=False)
                for index, result in retained.items()
                if index in refused
            )
            return Assigned(None, tuple(outcomes))
        if world.depth > FLATTEN:
            world.flatten()
        published = [
            ChangeOutcome(
                outcome.owner, outcome.result, "changed" if index in changed_facts else "unchanged"
            )
            for index, outcome in zip(facts, fact_outcomes)
        ]
        for item in requests:
            index = item.node
            final: Literal["removed", "unchanged", "changed"]
            if item.remove:
                final = "removed" if index in held else "unchanged"
            else:
                final = "unchanged" if held.get(index, _MISSING) is stored[index] else "changed"
            published.append(
                ChangeOutcome(nodes[index].owner, checks.get(index, Available(True)), final)
            )
        return Assigned(State(current.model, world), tuple(published))


__all__ = ["Assigned", "assign", "check", "normalize", "placed"]
