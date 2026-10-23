# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The questions a design space answers, each a function of the value alone: presence,
validity, every choice's state, the pairs (false piece, origin) and what stands in an
origin's way, a change of choices or facts, a bind, and the possible values. A change
builds the design space it would publish as a new value and evaluates it from nothing
(within its own evaluation).

What stands in an origin's way is what the origin read, here: the stored choices it read
as values, through derived values and views' bodies, less those the change names and
those deciding whether it exists (everything its guard read). Its own guard is skipped at
every node read through, and so is what a method read only to know whether something
exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from ..edits import BrokenRule, ChangeOutcome
from ..inspection import (
    Absent,
    ChoiceState,
    Dormant,
    Open,
    Piece,
    Possible,
    Refused,
    Set,
    Validity,
    Viable,
    Waiting,
)
from ..results import (
    Available,
    ConstraintAssessment,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
)
from ..semantics import equal, snapshot
from .answers import _ordered, _waits, membership_arguments
from .evaluate import INAPPLICABLE, TRUE, Evaluation, Session, evaluation
from .facts import prepare
from .per_class import flow_below as _flow
from .structure import record
from .value import DesignSpace, FactsLayer

if TYPE_CHECKING:
    from .._scopes import Address, ScopeRecord

_MISSING = object()
HOLDS: QueryResult[bool] = Available(True)
REMOVE = object()
"""A request clearing its choice."""

Presence = Literal["present", "absent", "unknown"]


# -- presence, validity, states ---------------------------------------------------------


def presence(ev: Evaluation, index: Address) -> tuple[Presence, QueryResult[object]]:
    """Whether ``index`` is present: its existence guards read outermost first."""
    guards = ev.inferred.guards(index)
    if not guards:
        return "present", TRUE
    if ev.assumed is None:
        for outer in reversed(guards[1:]):
            held = ev.result(outer)
            if isinstance(held, Inapplicable):
                return "absent", held
            if isinstance(held, Available) and held.value is False:
                return "absent", INAPPLICABLE
            if not (isinstance(held, Available) and held.value is True):
                break
    answer = ev.result(guards[0])
    if isinstance(answer, Available):
        return ("present" if answer.value else "absent"), answer
    if isinstance(answer, Unresolved):
        return "unknown", answer
    return "absent", answer


def placed(ev: Evaluation, index: Address) -> Evaluation:
    """Where the choice ``index``'s values are checked and listed: its membership reading
    (as if it existed), its guard read here first, to place the check. The reading answers
    the same whatever the guard answers (a domain reads facts only); a failure in the
    guard, a dependency cycle through the choice among them, is met there first."""
    guard = index.guard
    if guard is not None:
        ev.result(guard)
    return ev.membership(index)


def check(ev: Evaluation, index: Address, value: object = _MISSING) -> QueryResult[bool]:
    """A value's type check (the stored one by default), where it is placed (``placed``)."""
    node = index
    reading = placed(ev, index)
    if node.domain is None:
        return HOLDS
    held = ev.ds.stored[index] if value is _MISSING else value
    return cast(QueryResult[bool], ev.run(lambda: reading.type_check(node, Available(held), [])))


def _refuses(result: QueryResult[bool]) -> bool:
    return not isinstance(result, Unresolved) and not (
        isinstance(result, Available) and result.value is True
    )


def _pieces_in(
    ev: Evaluation, index: Address, wanted: type[object], answer: QueryResult[Any] | None = None
) -> list[Piece]:
    keys = ev.scopes.keys
    found: list[Piece] = []
    entry = ev.get(index)
    assessment = entry.assessment
    if not isinstance(assessment, ConstraintAssessment) or not assessment.results:
        held = entry.result if answer is None else answer
        if isinstance(held, (Rejected, Unresolved)) and isinstance(held, wanted):
            found.append(Piece(index.scope.name, held))
        return found
    for key, result in assessment.results.items():
        if isinstance(result, Inapplicable):
            continue
        piece = keys[key]
        if piece.kind == "validity":
            found.extend(_pieces_in(ev, piece, wanted, result))
        elif isinstance(result, (Rejected, Unresolved)) and isinstance(result, wanted):
            found.append(Piece(key, result))
    return found


def validity(ev: Evaluation, scope: ScopeRecord | None = None) -> Validity:
    index = (ev.scopes.root if scope is None else scope).validity
    answer = cast(QueryResult[bool], ev.result(index))
    if isinstance(answer, Rejected):
        return Validity("refused", _ordered(_pieces_in(ev, index, Rejected)), answer)
    if isinstance(answer, Unresolved):
        return Validity("open", _ordered(_pieces_in(ev, index, Unresolved)), answer)
    return Validity("valid", (), answer)


def state_of(ev: Evaluation, index: Address) -> ChoiceState:
    found, guard = presence(ev, index)
    stored = ev.ds.stored
    if index not in stored:
        if found == "absent":
            return Absent()
        return Open(_waits(guard) if found == "unknown" else ())
    value = stored[index]
    if found == "absent":
        return Dormant(value)
    if found == "unknown":
        return Waiting(value, tuple(sorted({*_waits(guard), *_waits(check(ev, index))})))
    read = ev.result(index)
    return Waiting(value, _waits(read)) if isinstance(read, Unresolved) else Set(value)


def states(ev: Evaluation, scope: ScopeRecord | None = None) -> tuple[tuple[str, ChoiceState], ...]:
    """Every choice's state, in the flow order (today's, from the compiled forms)."""
    at = ev.scopes.root if scope is None else scope
    return tuple((i.key, state_of(ev, i)) for i in _flow(ev.model, at))


# -- pairs, origins, standing -----------------------------------------------------------

PieceId = tuple[str, bool]
Pairs = Mapping[PieceId, Mapping["Address", Rejected]]


def pairs(ev: Evaluation) -> dict[PieceId, dict[Address, Rejected]]:
    """Each false piece of the root's validity with each origin it is false through."""
    keys = ev.scopes.keys
    root = ev.scopes.root.validity
    found: dict[PieceId, dict[Address, Rejected]] = {}
    if not isinstance(ev.result(root), Rejected):
        return found
    pending = [root]
    while pending:
        assessment = ev.get(pending.pop()).assessment
        if not isinstance(assessment, ConstraintAssessment):
            continue
        for key, answer in assessment.results.items():
            if not isinstance(answer, Rejected):
                continue
            index = keys[key]
            node = index
            if node.guard is not None and isinstance(ev.result(node.guard), Rejected):
                found[(key, True)] = _origins(ev, (node.guard,))
            elif node.kind == "validity":
                pending.append(index)
            elif node.kind == "decision":
                found[(key, False)] = _domain_origins(ev, index, answer)
            else:
                found[(key, False)] = _origins(ev, (index,))
    return found


def _domain_origins(ev: Evaluation, index: Address, answer: Rejected) -> dict[Address, Rejected]:
    inputs = [(a.node, ev.result(a.node)) for a in membership_arguments(index)]
    if not any(isinstance(result, Inapplicable) for _, result in inputs):
        refused = [node for node, result in inputs if isinstance(result, Rejected)]
        if refused:
            return _origins(ev, refused)
    return {index: answer}


def _origins(ev: Evaluation, starts: Sequence[Address]) -> dict[Address, Rejected]:
    """Following what each refused node read that answered refused, to the nodes refused
    on their own values (never into a view's Space validity)."""
    found: dict[Address, Rejected] = {}
    seen: set[Address] = set()
    pending = list(reversed(tuple(starts)))
    while pending:
        index = pending.pop()
        if index in seen:
            continue
        seen.add(index)
        entry = ev.get(index)
        if not isinstance(entry.result, Rejected):
            continue
        if entry.own_refusal:
            found[index] = entry.result
            continue
        node = index
        space = node.constraints[0] if node.kind == "view" else None
        pending.extend(
            dep
            for dep, answer in reversed(entry.trace)
            if isinstance(answer, Rejected) and dep != space
        )
    return found


def standing(ev: Evaluation, origin: Address, named: frozenset[Address]) -> tuple[str, ...]:
    """The stored choices standing in an origin's way: those it read as values here,
    through derived values and views' bodies, less those ``named`` and those deciding
    whether it exists."""
    node = origin
    if node.kind == "decision":
        return ()
    deciders = _deciders(ev, origin)
    read: set[Address] = set()
    if node.kind == "present":
        for target, answer in ev.get(origin).trace:
            if not isinstance(answer, Inapplicable) and target != node.guard:
                read |= {i for i in _deciders(ev, target) if i.kind == "decision"}
    else:
        pending, seen = [origin], {origin}
        while pending:
            index = pending.pop()
            for dependency in _value_reads(ev, index):
                if dependency in seen:
                    continue
                seen.add(dependency)
                kind = dependency.kind
                if kind == "decision":
                    read.add(dependency)
                elif kind not in ("param", "const", "validity"):
                    pending.append(dependency)
    stored = ev.ds.stored
    return tuple(
        sorted(i.key for i in read if i in stored and i not in named and i not in deciders)
    )


def _value_reads(ev: Evaluation, index: Address) -> Sequence[Address]:
    """What ``index`` read here as values: its trace, less its own guard (read first),
    what it read only to know whether something exists, a view's Space validity and a
    selection's selector."""
    spec = index
    entry = ev.get(index)
    trace = entry.trace
    if spec.guard is not None and trace and trace[0][0] == spec.guard:
        trace = trace[1:]
    skipped = set(entry.existence)
    if spec.kind == "view":
        skipped.add(spec.constraints[0])
    elif spec.kind == "select" and spec.selector is not None:
        skipped.add(spec.selector)
    return [dep for dep, _ in trace if dep not in skipped]


def _deciders(ev: Evaluation, index: Address) -> set[Address]:
    """The choices and facts deciding whether ``index`` exists: everything its guard's
    evaluation read here, through anything."""
    guard = index.guard
    if guard is None:
        return set()
    found: set[Address] = set()
    pending, seen = [guard], {guard}
    while pending:
        current = pending.pop()
        node = current
        if node.kind in ("decision", "param"):
            found.add(current)
            if node.kind == "param":
                continue
        for dep, _ in ev.get(current).trace:
            if dep not in seen:
                seen.add(dep)
                pending.append(dep)
    return found


def newly_false(
    before: Callable[[], Pairs], after: Evaluation, named: frozenset[Address]
) -> tuple[BrokenRule, ...]:
    held_after = pairs(after)
    if not held_after:
        return ()
    held_before = before()
    found: dict[Address, Rejected] = {}
    for piece, origins in held_after.items():
        held = held_before.get(piece, {})
        for index, result in origins.items():
            if index not in held:
                found.setdefault(index, result)
    return tuple(
        sorted(
            (BrokenRule(i.owner, result, standing(after, i, named)) for i, result in found.items()),
            key=lambda rule: rule.rule,
        )
    )


# -- changes ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """A change's or a bind's report, as today's ``ConfigurationResult`` has it: the
    design space it publishes (``None`` where refused)."""

    published: DesignSpace | None
    outcomes: tuple[ChangeOutcome, ...]
    broken: tuple[BrokenRule, ...] = ()
    dormant: tuple[str, ...] = ()
    in_play: tuple[str, ...] = ()
    opened: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.published is not None


def moved(
    before: Evaluation, after: Evaluation, sources: set[Address]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The stored choices put dormant and brought into play, and the open choices brought
    into presence, by presence before and after, in the flow order (only the choices
    whose presence may read what moved: today's presence readers)."""
    model = after.model
    scopes = after.scopes
    candidates = {c for source in sources for c in model.per_class.presence_readers(source)}
    dormant: list[Address] = []
    in_play: list[Address] = []
    opened: list[Address] = []
    for index in sorted(candidates, key=scopes.position):
        now = presence(after, index)[0]
        if now == "unknown":
            continue
        stored = index in after.ds.stored
        if now == "absent":
            if stored and presence(before, index)[0] != "absent":
                dormant.append(index)
        elif presence(before, index)[0] != "present":
            (in_play if stored else opened).append(index)
    return (
        tuple(i.key for i in dormant),
        tuple(i.key for i in in_play),
        tuple(i.key for i in opened),
    )


def _moved_from(
    before: Evaluation, after: Evaluation, facts: set[Address]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    held, stored = before.ds.stored, after.ds.stored
    sources = set(facts) | {
        i
        for i in held.keys() | stored.keys()
        if held.get(i, _MISSING) is not stored.get(i, _MISSING)
    }
    return moved(before, after, sources)


def _instantiate(
    ds: DesignSpace,
    facts: Mapping[Address, object],
    stored: Mapping[Address, object],
    session: Session,
) -> tuple[DesignSpace, dict[Address, QueryResult[bool]]]:
    """The design space of ``facts`` (their layer) with the ``stored`` values of their
    type there."""
    layer = FactsLayer.of(ds.model, facts)
    made = DesignSpace(layer, stored)
    ev = evaluation(made, session)
    position = record(ds.model).ranks(ds.model)
    checks = {i: check(ev, i) for i in sorted(stored, key=position.__getitem__)}
    kept = {i: v for i, v in stored.items() if not _refuses(checks[i])}
    if len(kept) < len(stored):
        made = DesignSpace(layer, kept)
    return made, checks


def _same(left: Mapping[Address, object], right: Mapping[Address, object]) -> bool:
    return left.keys() == right.keys() and all(left[i] is right[i] for i in left)


def _equal(node: Address, left: object, right: object, role: str, *, detach: bool) -> bool:
    """Whether two values of ``node`` are equal by its semantics, in ``role``: a fact's
    ("parameter equality") as given, a choice's ("configuration equality") each detached
    first, so equality cannot change a stored value (today's assignment's)."""
    if left is right:
        return True
    semantics = node.semantics
    assert semantics is not None
    if detach:
        left = snapshot(semantics, left, owner=node.owner, role=role)
        right = snapshot(semantics, right, owner=node.owner, role=role)
    return equal(semantics, left, right, owner=node.owner, role=role)


def _stored(node: Address, value: object, old: object) -> object:
    """A choice's ``value`` as stored: the held object ``old`` where the two are equal by
    the choice's semantics (so nothing reads it anew), else ``value``."""
    if old is not _MISSING and _equal(node, value, old, "configuration equality", detach=True):
        return old
    return value


def change(
    ds: DesignSpace,
    facts: Mapping[Address, object],
    requests: Sequence[tuple[Address, object]],
    session: Session,
) -> Report:
    """``facts`` (by root formal) and ``requests`` ((choice, value or ``REMOVE``), in order)
    as one change of ``ds``: today's assignment, step for step, on values."""
    values = prepare(
        {
            **{i: (v, "parameter") for i, v in facts.items()},
            **{i: (v, "choice") for i, v in requests if v is not REMOVE},
        },
    )
    held_facts, held = ds.facts, ds.stored
    given: dict[Address, object] = {}
    changed: set[Address] = set()
    for i in facts:
        old = held_facts.get(i, _MISSING)
        if old is not _MISSING and _equal(i, old, values[i], "parameter equality", detach=False):
            given[i] = old
        else:
            given[i] = values[i]
            changed.add(i)
    reached, rechecked = ds, cast("dict[Address, QueryResult[bool]]", {})
    if changed:
        reached, rechecked = _instantiate(ds, {**held_facts, **given}, held, session)
    before = reached.stored
    stored = dict(before)
    for i, value in requests:
        if value is REMOVE:
            stored.pop(i, None)
        else:
            stored[i] = _stored(i, values[i], before.get(i, _MISSING))

    def owner(index: Address) -> str:
        return index.owner

    fact_outcomes = [ChangeOutcome(owner(i), HOLDS, "unchanged") for i in facts]
    if not changed and _same(stored, held):
        unchanged = (ChangeOutcome(owner(i), HOLDS, "unchanged") for i, _ in requests)
        return Report(ds, (*fact_outcomes, *unchanged))
    after_ds = reached if _same(stored, before) else reached.with_stored(stored)
    after = evaluation(after_ds, session)
    checks = {i: check(after, i) for i, v in requests if v is not REMOVE}
    refused = {i for i, result in checks.items() if _refuses(result)}
    checked = [ChangeOutcome(o.owner, o.result, "checked") for o in fact_outcomes]
    if refused:
        for i, _ in requests:
            status: Literal["refused", "checked"] = "refused" if i in refused else "checked"
            checked.append(ChangeOutcome(owner(i), checks.get(i, HOLDS), status))
        return Report(None, tuple(checked))
    named = frozenset(i for i, _ in requests)
    sets = any(v is not REMOVE and before.get(i, _MISSING) is not stored[i] for i, v in requests)
    broken = newly_false(lambda: pairs(evaluation(reached, session)), after, named) if sets else ()
    if broken:
        checked.extend(
            ChangeOutcome(owner(i), checks.get(i, HOLDS), "checked") for i, _ in requests
        )
        return Report(None, tuple(checked), broken)
    made_false = (
        newly_false(lambda: pairs(evaluation(ds, session)), after, frozenset()) if changed else ()
    )
    moves = _moved_from(evaluation(ds, session), after, changed)
    published = [
        ChangeOutcome(o.owner, o.result, "changed" if i in changed else "unchanged")
        for i, o in zip(facts, fact_outcomes)
    ]
    for i, value in requests:
        final: Literal["removed", "unchanged", "changed"]
        if value is REMOVE:
            final = "removed" if i in held else "unchanged"
        else:
            final = "unchanged" if held.get(i, _MISSING) is stored[i] else "changed"
        published.append(ChangeOutcome(owner(i), checks.get(i, HOLDS), final))
    published.extend(
        ChangeOutcome(owner(i), result, "dropped", requested=False)
        for i, result in rechecked.items()
        if i not in named and _refuses(result)
    )
    return Report(after_ds, tuple(published), made_false, *moves)


def bind(ds: DesignSpace, saved: Sequence[tuple[Address, object]], session: Session) -> Report:
    """``ds``'s facts with exactly the ``saved`` choices, each of its type there; never
    refused; its report read against the facts with nothing stored."""
    values = prepare({i: (v, "choice") for i, v in saved})
    held = ds.stored
    kept_values = {i: _stored(i, values[i], held.get(i, _MISSING)) for i, _ in saved}
    empty = ds.with_stored({}) if held else ds
    made, checks = _instantiate(empty, ds.facts, kept_values, session)
    if _same(made.stored, held):
        made = ds
    after = evaluation(made, session)
    broken = newly_false(lambda: pairs(evaluation(empty, session)), after, frozenset())
    outcomes = tuple(
        ChangeOutcome(i.owner, checks[i], "changed" if i in made.stored else "dropped")
        for i, _ in saved
    )
    return Report(made, outcomes, broken, *_moved_from(evaluation(empty, session), after, set()))


def viable(ds: DesignSpace, index: Address, session: Session) -> Viable:
    """One choice's listed values, each tried alone as a change would set it."""
    ev = evaluation(ds, session)
    node = index
    reading = placed(ev, index)
    listed = ev.run(lambda: reading.listed(node, []))
    if not isinstance(listed, Available) or listed.value is None:
        return Viable(node.key, None, _waits(listed))
    found: list[Possible | Refused] = []
    for value in cast(tuple[object, ...], listed.value):
        trial = change(ds, {}, ((index, value),), session)
        if trial.accepted:
            found.append(Possible(value, trial.dormant, trial.in_play, trial.opened))
        else:
            found.append(Refused(value, trial.broken))
    return Viable(node.key, tuple(found))


__all__ = [
    "REMOVE",
    "Report",
    "bind",
    "change",
    "check",
    "moved",
    "pairs",
    "placed",
    "presence",
    "standing",
    "state_of",
    "states",
    "validity",
    "viable",
]
