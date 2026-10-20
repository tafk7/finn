# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change: choices set, replaced or cleared, and facts, as one batch; and a bind.

A change is **assignment**. In order:

1. **Identities.** Every item names a choice of this design space, once (a ``Change``,
   a ``{reference: value}`` mapping, or a keyword of the scope's own choices); facts
   name root formals, once. A malformed or foreign
   item is a ``RequestError``, before anything is evaluated.
2. **Recognition**, then **snapshot**: every supplied value is recognized by its
   semantics, all before any is frozen (``_state.prepare``).
3. **Same values.** Where every fact and stored value equals the receiver's, the
   receiver is kept, each item ``"unchanged"``.
4. **New facts instantiate a new design space** (``_instantiate``). Where a fact changes,
   every stored choice, dormant ones included, is checked again against its domain under
   the new facts (5). A value outside it, or whose domain's input refuses or does not
   apply, is dropped: left out of the stored choices before the new design space's
   world is made, and reported ``"dropped"`` (in the flow order, after the items) unless
   the change names the choice. A value
   waiting on a missing fact is kept, waiting. Checks read facts only, so what is
   dropped does not depend on order. Nothing of the rules refuses this step: a constraint
   the new facts make false reads refused. ``bind`` is the same instantiation, of saved
   choices on a design space's facts.
5. **Type checks.** Each value set is checked against its domain, which reads facts
   only (``_world.type_check``): outside it, or an input of the domain refusing or not
   applying, refuses the change; a missing fact leaves the choice stored, waiting on it.
   Membership is read as if the choice existed (``World.membership``: every guard
   deciding whether it exists taken as holding), so a choice is checked wherever it
   sits, dormant or waiting on its presence alike, and its check holds until the facts
   change.
6. **The design space it would publish**: that of 4 (the receiver's, where no fact
   changes), its stored choices less those cleared, with those set; its world made from
   that of 4.
7. **Rules.** A change that sets a new choice (a value it does not hold in the design
   space of 4) is refused where the design space it would publish holds a false piece of
   validity through an origin it was not false through in that of 4 (``_origins``): a
   piece newly present counts as not false before. Each such origin is a ``BrokenRule``,
   with the stored choices standing in its way. So no accepted change of choices adds a
   false piece and its origin, and a change of choices is never refused for what the
   facts broke. A change of clears alone is never refused, because clearing stays the
   remedy a refusal names ("clear it first"). A clear only removes information, but it
   may reveal a rule: where a Space is refused by its own rule, a rule reading one of its
   views is false through it, with no origin of its own; clearing the choice behind that
   refusal reopens the Space, and the rule may turn false at its own origin. What the
   clear reveals was hidden behind the refusal, not caused by it, and the published
   design space reads refused, naming it. A change with a value refused by its type
   check reports those alone. An accepted change of facts reports, in the same form, the rules
   newly false in the published design space: by its new facts, or revealed by a clear
   beside them. Each is an origin through which the published design space holds a false
   piece the receiver did not hold through it, with every stored choice standing in its
   way.
8. **Publication:** with nothing refused, the design space is published whole; else
   nothing is, and each item's outcome, and each rule broken, says why. Nothing is
   filled in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from .results import Available, QueryResult, Unresolved
from .semantics import equal, snapshot

T = TypeVar("T")
S = TypeVar("S", bound=Space)

_MISSING = object()


@dataclass(frozen=True, slots=True)
class Moved:
    """What an accepted change moves, by key: the stored choices it puts out of play
    (present or waiting before, absent after), those it brings into play (not present
    before, present after), and the open choices it brings into presence."""

    dormant: tuple[str, ...] = ()
    in_play: tuple[str, ...] = ()
    opened: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Assigned:
    """What a change came to: the design space it publishes (the receiver's state
    where it changes nothing; ``None`` where it is refused); each item's outcome, in
    order: the facts given, the choices named, then the stored choices new facts dropped
    (in the flow order); the rules it is refused by, or, accepted, the rules newly false
    in the published design space: by its new facts, or revealed by a clear beside them
    (a bind's: those its saved choices break); and what it moved."""

    state: State | None
    outcomes: tuple[ChangeOutcome, ...]
    broken: tuple[BrokenRule, ...] = ()
    moved: Moved = field(default_factory=lambda: Moved())

    @property
    def accepted(self) -> bool:
        return self.state is not None


def normalize(current: State, changes: Sequence[Change[object]]) -> dict[NodeId, Change[object]]:
    """Each change by its choice: every one of this design space, naming an owned
    Decision of its scope, once. Its scope and node are asked of the model
    (``holds_scope``, ``holds_node``: an addressed scope, a node of one), never compared
    with a length."""
    linked = current.linked
    pending: dict[NodeId, Change[object]] = {}
    for item in changes:
        if not isinstance(item, Change):
            raise RequestError("changes must come from a bound decision field")
        if item.base is not current:
            raise RequestError("all changes must target this exact base snapshot")
        if not linked.holds_scope(item.scope):
            raise RequestError("change scope does not belong to this model")
        if not linked.holds_node(item.node):
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


def placed(world: World, index: NodeId) -> World:
    """Where a choice's type check is made: ``world``, where every guard above the choice
    holds there; else the world where its membership is read as if it existed
    (``World.membership``: its existence guards taken as holding)."""
    node = world.evaluator.nodes[index]
    if node.guard is None:
        return world
    guard = world.result(node.guard)
    if isinstance(guard, Available) and guard.value is True:
        return world
    return world.membership(index)


def check(world: World, index: NodeId) -> QueryResult[bool]:
    """A stored value's type check, made as if the choice existed (``placed``)."""
    return cast(QueryResult[bool], type_check(placed(world, index), index, world.values[index]))


def _refuses(result: QueryResult[bool]) -> bool:
    """A type check the value fails: anything but true or unresolved (a missing fact: the
    choice is stored and waits on it)."""
    return not isinstance(result, Unresolved) and not (
        isinstance(result, Available) and result.value is True
    )


def _pairs(state: State) -> Pairs:
    """A configuration's pairs (each false piece of its root's validity with each origin it
    is false through), kept on it."""
    found = state.answers.get(("pairs", 0))
    if found is None:
        found = state.answers[("pairs", 0)] = pairs(state.world)
    return cast(Pairs, found)


def _held(current: State, world: World) -> Pairs:
    """The pairs of ``world``: ``current``'s own, kept on it, where ``world`` is its world."""
    return _pairs(current) if world is current.world else pairs(world)


def _value(node: Node, value: object, old: object) -> object:
    """A choice's ``value`` as stored: the held object ``old`` where the two are equal by
    the choice's semantics (so nothing reads it anew), else ``value``."""
    if old is not _MISSING and _equal(node, value, old, "configuration equality", detach=True):
        return old
    return value


def _same(left: Mapping[NodeId, object], right: Mapping[NodeId, object]) -> bool:
    """Whether two stored sets hold the same objects for the same choices."""
    return left.keys() == right.keys() and all(left[index] is right[index] for index in left)


def _moved(before: World, after: World, sources: set[NodeId]) -> Moved:
    """What a change moves: for each choice whose presence may read a choice or fact the
    change moved (``sources``), in the flow order, its presence after and, where that
    decides a list, before (``presence``: refused guards count as absent). A stored
    choice is ``dormant`` where it is absent after and was not absent before, ``in_play``
    where it is present after and was not present before; an open choice is ``opened``
    where it is present after and was not present before. So the presence before is read
    only for a choice absent after and stored, or present after: a choice of unknown
    presence after, or absent and open, is in no list whatever it was, and its presence
    before (which may read the scopes of every case its guards wait on) is not read. The
    lists are those of reading both everywhere. Keys are read from the nodes' headers."""
    evaluator = after.evaluator
    linked = evaluator.linked
    candidates = {choice for source in sources for choice in evaluator.presence_readers(source)}
    dormant: list[NodeId] = []
    in_play: list[NodeId] = []
    opened: list[NodeId] = []
    for index in sorted(candidates, key=linked.ranks.__getitem__):
        now = presence(after, index)[0]
        if now == "unknown":
            continue
        stored = index in after.stored
        if now == "absent":
            if stored and presence(before, index)[0] != "absent":
                dormant.append(index)
        elif presence(before, index)[0] != "present":
            (in_play if stored else opened).append(index)
    return Moved(
        tuple(linked.header(index).key for index in dormant),
        tuple(linked.header(index).key for index in in_play),
        tuple(linked.header(index).key for index in opened),
    )


def _moved_from(before: World, after: World, facts: set[NodeId]) -> Moved:
    """What a change moves from ``before`` to ``after``: its sources are the ``facts`` it
    changed and every choice whose stored value it set, cleared or dropped."""
    held, stored = before.stored, after.stored
    sources = set(facts) | {
        index
        for index in held.keys() | stored.keys()
        if held.get(index, _MISSING) is not stored.get(index, _MISSING)
    }
    return _moved(before, after, sources)


def _broken(
    before: Callable[[], Pairs], world: World, named: frozenset[NodeId]
) -> tuple[BrokenRule, ...]:
    """The rules newly false in ``world``: each origin through which it holds a false
    piece not held through that origin in the pairs ``before`` returns, with the stored
    choices standing in its way (``named``: the choices excluded for being named by the
    change). ``before`` is called only where ``world`` holds a false piece: the design
    space compared with may be open everywhere, and reading its validity whole there
    resolves every scope of unknown presence."""
    after = pairs(world)
    if not after:
        return ()
    held_before = before()
    found: dict[NodeId, Origin] = {}
    for piece, origins in after.items():
        held = held_before.get(piece, {})
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


def _instantiate(
    base: World, facts: Mapping[NodeId, object] | None, stored: Mapping[NodeId, object]
) -> tuple[World, dict[NodeId, QueryResult[bool]]]:
    """The design space of ``facts`` (``base``'s where ``None``) with the ``stored``
    choices that fall within their domains, made from ``base``: every stored value checked
    under those facts as if it existed (``check``, dormant ones included), and each it
    fails left out. Returns its world and each stored value's check by choice, in the flow
    order (``Evaluator.flow``). Checks read facts only, so what is left out does not depend
    on order; a value waiting on a missing fact is kept, and no rule is read."""
    made = base.successor(facts=facts, stored=stored)
    position = {index: rank for rank, index in enumerate(base.evaluator.flow())}
    checks = {index: check(made, index) for index in sorted(stored, key=position.__getitem__)}
    kept = {index: value for index, value in stored.items() if not _refuses(checks[index])}
    if len(kept) < len(stored):
        made = base.successor(facts=facts, stored=kept)
    return made, checks


def _publish(world: World, *, trial: bool) -> None:
    """A world as published: kept short (``World.flatten``) unless only tried."""
    if world.depth > FLATTEN and not trial:
        world.flatten()


def assign(
    current: State,
    facts: Mapping[NodeId, object],
    requests: Sequence[Change[object]],
    *,
    trial: bool = False,
) -> Assigned:
    """The change of ``facts`` (by root formal) and ``requests`` (in the caller's order)
    from ``current``, under the model's lock: new facts instantiate a new design space
    (``_instantiate``), and the requests are a change of choices on it. A ``trial``
    publishes nothing: the design space it would publish is answered, its world left as
    made."""
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
        # The design space the choices are changed on: the receiver's, or that of the new
        # facts, with the stored values new facts leave outside their domains dropped.
        instance = current.world
        rechecked: dict[NodeId, QueryResult[bool]] = {}
        if changed_facts:
            instance, rechecked = _instantiate(instance, {**held_facts, **given}, held)
        before = instance.stored
        stored = dict(before)
        for index, item in normalized.items():
            if item.remove:
                stored.pop(index, None)
            else:
                stored[index] = _value(nodes[index], values[index], before.get(index, _MISSING))
        fact_outcomes = [
            ChangeOutcome(nodes[index].owner, Available(True), "unchanged") for index in facts
        ]
        if not changed_facts and _same(stored, held):
            unchanged = (
                ChangeOutcome(nodes[item.node].owner, Available(True), "unchanged")
                for item in requests
            )
            return Assigned(current, (*fact_outcomes, *unchanged))
        world = instance if _same(stored, before) else instance.successor(stored=stored)
        checks = {
            index: check(world, index) for index, item in normalized.items() if not item.remove
        }
        refused = {index for index, result in checks.items() if _refuses(result)}
        checked = [ChangeOutcome(o.owner, o.result, "checked") for o in fact_outcomes]
        if refused:
            for item in requests:
                status: Literal["refused", "checked"] = (
                    "refused" if item.node in refused else "checked"
                )
                result = checks.get(item.node, Available(True))
                checked.append(ChangeOutcome(nodes[item.node].owner, result, status))
            return Assigned(None, tuple(checked))
        sets = any(
            not item.remove and before.get(index, _MISSING) is not stored[index]
            for index, item in normalized.items()
        )
        broken = (
            _broken(lambda: _held(current, instance), world, frozenset(normalized)) if sets else ()
        )
        if broken:
            checked.extend(
                ChangeOutcome(
                    nodes[item.node].owner, checks.get(item.node, Available(True)), "checked"
                )
                for item in requests
            )
            return Assigned(None, tuple(checked), broken)
        made_false = _broken(lambda: _pairs(current), world, frozenset()) if changed_facts else ()
        moved = _moved_from(current.world, world, changed_facts)
        _publish(world, trial=trial)
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
        published.extend(
            ChangeOutcome(nodes[index].owner, result, "dropped", requested=False)
            for index, result in rechecked.items()
            if index not in normalized and _refuses(result)
        )
        return Assigned(State(current.model, world), tuple(published), made_false, moved)


def bound(current: State, requests: Sequence[Change[object]]) -> Assigned:
    """``current``'s facts with exactly the saved choices ``requests`` set, under the
    model's lock: the design space of those facts instantiated with them (``_instantiate``),
    each value outside its domain dropped. Never refused; a clear is a ``RequestError``.
    Its whole report reads against the facts with nothing stored, whatever ``current``
    stores: each saved choice kept is ``"changed"`` and each dropped ``"dropped"``, its
    rules are those false through an origin that design space is not false through, and
    what it moves is from that design space. So the report is the same from any design
    space with these facts. Where the bound set is ``current``'s own (the same objects),
    ``current`` is the design space it publishes, with that same report."""
    linked = current.linked
    nodes = linked.nodes
    normalized = normalize(current, requests)
    if any(item.remove for item in normalized.values()):
        raise RequestError("a bind sets saved choices; it clears none")
    with current.evaluator.lock:
        values = prepare(
            linked, {index: (item.value, "choice") for index, item in normalized.items()}
        )
        held = current.world.stored
        saved = {
            index: _value(nodes[index], values[index], held.get(index, _MISSING))
            for index in normalized
        }
        empty = current.world.successor(stored={}) if held else current.world
        world, checks = _instantiate(empty, None, saved)
        kept = _same(world.stored, held)  # the receiver's own choices: the receiver
        if kept:
            world = current.world
        made_false = _broken(lambda: _held(current, empty), world, frozenset())
        # Each saved choice from nothing stored: set (``"changed"``) where kept, else
        # ``"dropped"``, whatever ``current`` holds.
        outcomes = tuple(
            ChangeOutcome(
                nodes[item.node].owner,
                checks[item.node],
                "changed" if item.node in world.stored else "dropped",
            )
            for item in requests
        )
        moved = _moved_from(empty, world, set())
        if kept:
            return Assigned(current, outcomes, made_false, moved)
        _publish(world, trial=False)
        return Assigned(State(current.model, world), outcomes, made_false, moved)


def tried(current: State, index: NodeId, value: object) -> Assigned:
    """The choice ``index`` set to ``value`` alone, from ``current``, as a change would set
    it, publishing nothing: one value's trial (``inspection.viable``)."""
    node = current.linked.nodes[index]
    return assign(current, {}, (Change(current, node.scope, index, value),), trial=True)


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
    """A change's report: the design space it publishes (the receiver where it changes
    nothing or is refused), each item's outcome, the rules it is refused by or, where
    accepted, those newly false in the published design space, and what it moved."""
    if assigned.state is None:
        return ConfigurationResult(point, False, assigned.outcomes, assigned.broken)
    moved = assigned.moved
    published = assigned.state
    return ConfigurationResult(
        point if published is state(point) else cast(S, attach(published, point._scope)),
        True,
        assigned.outcomes,
        assigned.broken,
        moved.dormant,
        moved.in_play,
        moved.opened,
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
    """A new design space from other values for root formals, keeping the choices that
    still fall within their domains, then the choices ``changes`` make on it, as one
    batch. A change of facts is never refused: every stored choice the batch does not
    name, dormant ones included, is checked again against its domain under the new facts,
    and one outside it, or whose domain's input refuses or does not apply, is dropped
    (``"dropped"``, not requested); one waiting on a missing fact keeps waiting. A
    constraint the new facts make false reads refused. The report's ``broken`` names each
    rule newly false in the published design space, by the new facts or revealed by a
    clear beside them, with the stored choices standing in its way. ``changes`` are a
    change of choices on the new design space, refused by the rules of
    ``try_with_choices`` judged against it; a refused batch publishes nothing."""
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


def bind(point: S, /, *choices: ChangeRequest | Mapping[Any, object]) -> ConfigurationResult[S]:
    """``point``'s facts with exactly the saved ``choices`` (``{reference: value}``
    mappings or ``Change`` objects of ``point``; a clear is a ``RequestError``). Binding
    facts to a Space's params instantiates a design space, and binding saved choices onto
    those facts is the same instantiation, keeping the choices that fall within their
    domains: each value is checked under ``point``'s facts as if its choice existed, and
    one outside its domain, or whose domain's input refuses or does not apply, is dropped
    (``"dropped"``); one waiting on a missing fact is kept, waiting. Each saved choice kept
    is ``"changed"``, as set on the facts with nothing stored, whatever ``point`` holds.
    Never refused for constraints: the report's ``broken`` names every rule the saved
    choices break under these facts (false through an origin the facts with nothing stored
    are not), each with the stored choices standing in its way, and the design space reads
    refused. Its ``dormant``, ``in_play`` and ``opened`` read against the same baseline, so
    the whole report (each outcome, the rules broken, what moved) is the same from any
    design space with these facts. A change of facts (``try_with_facts``) is this
    instantiation under the new facts. The point's own stored choices are replaced and not
    reported, so bind from a design space with nothing stored, as FINN will."""
    driver_only("configuration binding")
    return _report(point, bound(state(point), _requests(point, choices)))


__all__ = [
    "Assigned",
    "Moved",
    "assign",
    "bind",
    "bound",
    "change",
    "check",
    "clear",
    "normalize",
    "placed",
    "tried",
    "try_with_choices",
    "try_with_facts",
    "with_choices",
    "with_facts",
]
