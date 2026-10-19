# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change: choices set, replaced or cleared, and facts, as one batch.

A change is **assignment**. In order:

1. **Identities.** Every item names a choice of this configuration, once (a ``Change``,
   a ``{reference: value}`` mapping, or a keyword of the scope's own choices); facts
   name root formals, once. A malformed or foreign
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
6. **Rules.** A change that sets a new value (a choice set to a value it does not hold,
   or a fact to a value it does not have) is refused where the configuration it would
   publish holds a false piece of validity through an origin it was not false through
   before (``_origins``): a piece newly present counts as not false before. Each such
   origin is a ``BrokenRule``, with the stored choices standing in its way. So no
   accepted change that sets adds a false piece and its origin: a configuration's false
   pieces are those its facts made false. A change of clears alone is never refused:
   clearing opens a choice, and an open choice turns what reads it unknown, never false.
   A change with a value refused by its type check reports those alone.
7. **Publication:** with nothing refused, the configuration is published whole; else
   nothing is, and each item's outcome, and each rule broken, says why. Nothing is
   cleared or filled in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

from ._configuration import Space
from ._failure import driver_only
from ._origins import Origin, Pairs, pairs, standing
from ._state import State, prepare, presence
from ._world import FLATTEN, World, type_check
from .edits import BrokenRule, Change, ChangeOutcome, ChangeRequest, ConfigurationResult
from .errors import ConfigurationError, RequestError
from .ir import Node, NodeId
from .occurrence import attach, decision_index, state
from .results import Available, Inapplicable, QueryResult, Unresolved
from .semantics import equal, snapshot

T = TypeVar("T")
S = TypeVar("S", bound=Space)

_MISSING = object()


@dataclass(frozen=True, slots=True)
class Assigned:
    """What a change came to: the configuration it publishes (the receiver's state
    where it changes nothing; ``None`` where it is refused); each item's outcome, in
    order: the facts given, the choices named, then the stored choices the change checked
    again (a facts change, or brought into play) and found refusing; and each rule it
    would break."""

    state: State | None
    outcomes: tuple[ChangeOutcome, ...]
    broken: tuple[BrokenRule, ...] = ()

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


def _pairs(state: State) -> Pairs:
    """A configuration's pairs (each false piece of its root's validity with each origin it
    is false through), kept on it."""
    found = state.answers.get(("pairs", 0))
    if found is None:
        found = state.answers[("pairs", 0)] = pairs(state.world)
    return cast(Pairs, found)


def _broken(current: State, world: World, named: frozenset[NodeId]) -> tuple[BrokenRule, ...]:
    """The rules a change would break: each origin through which ``world`` (the
    configuration it would publish) holds a false piece that ``current`` does not hold
    through it, with the stored choices standing in its way (``named``: the choices the
    change sets or clears)."""
    after = pairs(world)
    if not after:
        return ()
    before = _pairs(current)
    found: dict[NodeId, Origin] = {}
    for piece, origins in after.items():
        held = before.get(piece, {})
        for index, origin in origins.items():
            if index not in held:
                found.setdefault(index, origin)
    nodes = world.evaluator.nodes
    return tuple(
        sorted(
            (
                BrokenRule(nodes[index].owner, origin.result, standing(world, index, named))
                for index, origin in found.items()
            ),
            key=lambda rule: rule.rule,
        )
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
        sets = bool(changed_facts) or any(
            not item.remove and held.get(index, _MISSING) is not stored[index]
            for index, item in normalized.items()
        )
        broken = _broken(current, world, frozenset(normalized)) if sets else ()
        if broken:
            refused_outcomes = (
                *(ChangeOutcome(o.owner, o.result, "checked") for o in fact_outcomes),
                *(
                    ChangeOutcome(
                        nodes[item.node].owner, checks.get(item.node, Available(True)), "checked"
                    )
                    for item in requests
                ),
            )
            return Assigned(None, refused_outcomes, broken)
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


# -- the configuration's change methods ------------------------------------------------


def change(point: Space, reference: object, value: T) -> Change[T]:
    driver_only("change construction")
    current = state(point)
    index = decision_index(point, reference)
    return Change(current, current.linked.nodes[index].scope, index, value)


def clear(point: Space, reference: object) -> Change[object]:
    driver_only("clear construction")
    current = state(point)
    index = decision_index(point, reference)
    return Change(current, current.linked.nodes[index].scope, index, remove=True)


def _keyword_change(point: Space, name: str, value: object) -> Change[object]:
    current = state(point)
    scope = current.linked.scopes[point._scope]
    index = scope.named_members.get(name)
    if index is None:
        raise RequestError(f"unknown direct choice {name!r}")
    node = current.linked.nodes[index]
    if node.kind != "decision" or node.scope != point._scope:
        raise RequestError(f"{node.key} is not a direct owned choice in this scope")
    return Change(current, node.scope, index, value)


def _requests(
    point: Space, items: tuple[ChangeRequest | Mapping[Any, object], ...]
) -> list[Change[object]]:
    """Expand ``{reference: value}`` mappings in order; Change objects pass through."""
    current = state(point)
    result: list[Change[object]] = []
    for item in items:
        if isinstance(item, Mapping):
            for reference, value in item.items():
                index = decision_index(point, reference)
                result.append(Change(current, current.linked.nodes[index].scope, index, value))
        else:
            result.append(item)
    return result


def _fact_nodes(point: Space, facts: Mapping[Any, object]) -> dict[NodeId, object]:
    """``{formal: value}`` by node: each formal a root formal the model exposes."""
    current = state(point)
    nodes: dict[NodeId, object] = {}
    for reference, value in facts.items():
        index = current.model.resolve(point._scope, reference)
        if index not in current.linked.parameters:
            raise RequestError(f"{current.linked.nodes[index].key} is not an exposed parameter")
        if index in nodes:
            raise RequestError(f"duplicate fact for {current.linked.nodes[index].key}")
        nodes[index] = value
    return nodes


def _report(point: S, assigned: Assigned) -> ConfigurationResult[S]:
    """A change's report: the configuration it publishes (the receiver where it changes
    nothing or is refused), and each item's outcome."""
    if assigned.state is None:
        return ConfigurationResult(point, False, assigned.outcomes, assigned.broken)
    if assigned.state is state(point):
        return ConfigurationResult(point, True, assigned.outcomes)
    return ConfigurationResult(
        cast(S, attach(assigned.state, point._scope)), True, assigned.outcomes
    )


def try_with_choices(
    point: S, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
) -> ConfigurationResult[S]:
    """Set, replace and clear choices as one batch, over the same facts."""
    driver_only("configuration replacement")
    keyword_changes = tuple(_keyword_change(point, name, value) for name, value in choices.items())
    requests = (*_requests(point, changes), *keyword_changes)
    return _report(point, assign(state(point), {}, requests))


def try_with_facts(
    point: S, facts: Mapping[Any, object], /, *changes: ChangeRequest | Mapping[Any, object]
) -> ConfigurationResult[S]:
    """Other values for root formals, and the choices ``changes`` make, as one batch:
    every stored value is checked again against its domain over the new facts."""
    driver_only("configuration replacement")
    given = _fact_nodes(point, facts)
    return _report(point, assign(state(point), given, _requests(point, changes)))


def with_choices(
    point: S, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
) -> S:
    report = try_with_choices(point, *changes, **choices)
    if not report.accepted:
        raise ConfigurationError(report)
    return report.instance


def with_facts(
    point: S, facts: Mapping[Any, object], /, *changes: ChangeRequest | Mapping[Any, object]
) -> S:
    report = try_with_facts(point, facts, *changes)
    if not report.accepted:
        raise ConfigurationError(report)
    return report.instance


__all__ = [
    "Assigned",
    "assign",
    "change",
    "check",
    "clear",
    "normalize",
    "placed",
    "try_with_choices",
    "try_with_facts",
    "with_choices",
    "with_facts",
]
