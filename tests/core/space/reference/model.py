# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An executable reference model of the Space engine's semantics.

Written from the semantics, not from the engine: it imports nothing of ``finn`` and
nothing outside the standard library. It is slow on purpose. Every answer is
recomputed from scratch for each configuration; one evaluation of one configuration
memoizes its own answers while it runs, and nothing survives from one configuration
to the next. A change evaluates its predecessor and its successor each from nothing.

The rules it states
===================

**A configuration** is its facts and its stored choices, dormant ones included: the
same two give the same answers, whatever the history. Nothing gives a choice a value:
a choice is set, or open. A stored value outside its domain, or whose domain's
membership reads a refused input, is no configuration (:class:`ConfigurationError`);
a stored set breaking a rule is one, and reads refused.

**Domains.** Membership reads facts only, so it is a type check fixed once the facts
are given: a value is a member, outside, waiting on a missing fact, refused through a
refused input, or refused because an input does not apply (reading 6). A listed
domain's membership list is its listed values that are members, in listed order; its
``order`` ranks that list and checks nothing.

**Presence.** A scope is present; absent (another case selected, or its ``when``
false); unknown (its selector open, or a ``when`` input waiting); or refused (its
``when`` reads a refused input: a derived value or a view refusing). A scope under a
refused guard is not in play: its stored choices are dormant, its open choices
absent, its constraints add nothing, and every value in it reads refused through the
guard.

**A choice** is set (stored, in play, read as its value), waiting (stored, its
presence unknown or its membership waiting on a missing fact; never read as a value),
dormant (stored, its scope absent or not in play), open (unset, its scope present or
unknown) or absent (unset, its scope absent or not in play).

**Reads.** A fact is known, or unresolved when not given. A derived value or a view
reads its inputs: all known, its body answers (a :class:`~.description.Refuse` is a
refusal at its origin); an input refused, it is refused through it; an input waiting,
it waits; an input inapplicable, it is inapplicable. A view also refuses, through,
when its Space's validity is false.

**Validity** of a scope is the strong three-valued AND
(:func:`~.answers.kleene_and`) of the pieces at or below it: each constraint (absent
or not in play: true, not applying; an input refused: false through it; an input
waiting: unknown; an input inapplicable: true, not applying; else its predicate on its
own values, false at its origin); each listed domain that lists no value (false), or
whose list cannot be made yet (reading 13); each guard refused through an input
(false). A listed domain that lists a value is no piece, whatever its presence. A
piece of a scope whose presence is unknown is unknown, whatever it would answer. No
choice is otherwise a piece.

**A change** (commit or replace choices, clear choices, set or unset facts, as one
batch) is assignment: the successor is published whole, or refused with each refusal
and the original kept.

1. *Assignment.* Each committed value is checked against its domain; a change of facts
   re-checks every stored value, dormant ones included. A value outside its domain, or
   whose membership reads a refused input, refuses the change
   (:class:`~.answers.AssignmentRefusal`). A value whose membership waits on a missing
   fact is accepted, and waits.
2. *Counted pairs.* An origin is a constraint false on its own values; a derived value
   or a view refusing on its own values; a present choice whose listed domain lists no
   value. A configuration's counted pairs are each false piece of the root's validity
   with each origin it is false through: the piece itself, or the origins its refused
   inputs lead back to.
3. *Refusal.* A change that sets a choice or a fact is refused when the successor
   holds a counted pair the predecessor does not: a piece false through an origin it
   was not false through before (a piece absent, unknown or true before counts as not
   false before). Each origin of such a pair refuses it once
   (:class:`~.answers.RuleRefusal`), naming the stored choices standing in its way. A
   rule false only through a refused input is never itself a refusal. So no accepted
   change that sets adds a pair. A change made only of clears and unsets is never
   refused (reading 3).
4. *Accepted*, the change reports the stored choices it puts dormant or brings into
   play and the open choices it brings into presence. A change that changes nothing
   returns the same configuration.

**Possible values** of a choice with a known membership list are one change per member
value, in membership's list order: possible (with what moves) or impossible (with
each refusal). The ranking is a separate answer; validity and refusals never read it.

Readings chosen where the semantics leave room
==============================================

The note is ``records/space-semantics-defaults-2026-10-18/NOTE.md`` in the design
corpus; § 16 holds the readings its author gave for this model.

1. *A view's body refusing on its own values is an origin* (§ 16.1), as a derived
   value's is. A view refused because its Space is refused is refused through, never an
   origin: its Space's false pieces are counted on their own.
2. *Only an origin feeding a false piece counts* (§ 16.2, kept by § 16.15). A piece's
   origins are found from it, following refused inputs back through derived values,
   views, and choices and values refused through a guard (to the guard's refused reads);
   a listed domain's list refused through its membership's input, back over the facts.
   A refusing derived value or view nothing rules on reads refused where it is asked,
   is in no pair, and refuses no change. A piece false only through a view refused with
   its Space has no origin of its own: its Space's false pieces are counted on their
   own (reading 1).
3. *A change is refused by a new pair* (§ 16.15, which replaces § 16.2-16.3's reach and
   supersedes § 16.13's first consequence and § 16.14). Pieces are told apart by kind,
   owner and subject; a pair is new when the predecessor lacks it, whatever moved.
   Consequences: setting ``x`` beside a rule already false through ``d``, which refuses
   on the facts alone, adds no pair and is accepted; switching to a case whose rule (or
   guard) is false through the facts alone is refused with nothing standing, the
   refusing derived value inside the case or outside it alike; a switch to a case whose
   rule is false through a derived value over a stored ``c`` is refused with ``c``
   standing; a configuration built with a broken rule accepts a change keeping that
   rule false through the same origin, and refuses one making it false through another.
   Every accepted change that sets leaves the set of pairs no larger, so the bind of a
   configuration reached by such changes from nothing is accepted (FN3), and no
   accepted addition leaves a Space newly refused (§ 16.14's exception cannot arise).
   *Clears and unsets are never refused* (§ 16.16; § 5.6 stands): a change made only
   of them is accepted whatever pairs it adds. It can add one in a single shape,
   **masking**: a piece false only through a view refused with its Space, whose Space
   the change reopens, so that the view answers and the piece is false at an origin of
   its own. The configuration reads refused, naming it; its bind may be refused (a
   named limit of FN3's invariant, until products leave the Space).
4. *What stands in the way* (§ 16.4): the stored choices among the origin's declared
   reads followed through derived values' and views' bodies (not its presence's
   reads), less every choice the change names (committed or cleared) and every choice
   deciding the origin's existence: the selectors of the scopes above it, its own
   included, and the choices its guards' ``when`` reads depend on. An empty domain has
   nothing standing (membership reads facts only).
5. *In play, waiting, dormant* (§ 16.5), by presence before and after, over the
   successor's stored choices: ``in_play``, not present before and present after;
   ``dormant``, present or unknown before and absent or not in play after; ``opened``,
   the open choices not present before and present after. Clearing a selector leaves
   its cases' stored choices waiting. An unset choice not in play is
   :class:`~.answers.Absent`, a state the note's table lacks.
6. *Membership that cannot be decided* (§ 16.6): a missing fact leaves a stored value
   waiting (never read as a value) and the facts change supplying it checks its type; a
   refusing input refuses the assignment, and a facts change making it refuse while a
   value is stored. Membership is evaluated over the facts alone: a derived value it
   reads is computed from its reads whatever its own scope's presence. *An input that
   does not apply is refused* (§ 16.17, ``INPUT_ABSENT``, the choice's reason naming the
   input), not unresolved: only a change of facts could settle it. Its existence is
   decided over the facts alone: the choice's own existence guards (its scope's and
   those above it) hold whenever it is present, and are not asked; any other guard on
   the input's scopes (or on those of the derived values it reads) is a ``when`` over
   facts, a selector or a ``when`` reaching a choice being ``domain-reads-choice``. A
   false one makes the input absent, a waiting one leaves membership waiting, a refused
   one refuses it through that guard's input. An omitted fact stays a missing one.
7. *The constructor* (§ 16.7) refuses a stored value outside its domain or whose
   membership input refuses, dormant ones included; a stored set breaking a rule is a
   legal configuration that reads refused. Only :func:`change` refuses it.
8. *A guard refused through an input* (§ 16.8) is one ``GUARD`` piece, owned by the
   guarded scope, at the highest such guard (scopes under it add nothing). A refused
   ``when`` below a selector or a parent that is still unknown leaves the scope unknown
   and its piece unknown: refused takes effect once all above it is present.
9. *A change newly emptying a present choice's listed domain is refused* (§ 16.9, in
   § 16.15's terms): a facts change emptying it, or a switch to the case holding it,
   adds the pair of its ``EMPTY_DOMAIN`` piece and itself, and is refused naming it
   with nothing standing. A domain already empty before the change adds no pair.
   **Not yet ruled on by the human.** The single switch is
   :data:`EMPTY_DOMAIN_REFUSES`; off, an empty domain never refuses a change and the
   configuration reads refused.
10. *A domain's order* (§ 16.10) ranks the membership list over ``order_reads``, which
    may read choices: :meth:`Solution.ordered` is the ranking, known, or waiting while a
    read waits. Possible values are answered in membership's list order.
11. *An unknown guard answers unknown* for every piece under it (constraints, empty
    listed domains, guards), not Kleene's implication (the canon's text in § 11: "a scope
    whose guard waits counts each of its pieces as unknown").
12. *Assignment comes first* (§ 5.1): a change with an assignment refusal reports those
    alone, and its ``validity`` is ``None``: a value of the wrong type makes no
    configuration whose rules could be read.
13. *A listed domain is a piece only when it lists no value* (§ 4.6, § 16.12): false
    when its list is empty and its choice present (the origin), unknown when it is
    empty under a guard that waits; unknown while the list waits on a missing fact, and
    false through the input when its membership reads a refused input (the list cannot
    be made; blamed at that input's origin). When an input does not apply (§ 16.17,
    reading 6) the list cannot be made either and the piece is false through it; the
    origin is the domain itself, as for an empty list (reading 9 decides it alike). A
    list with a value adds no piece, whatever its presence: an open selector whose case
    holds a listed choice and no rules leaves the configuration valid. A membership-only
    domain has no piece.
14. *A stored choice waits on what it waits on*: an unknown presence's waits and its
    membership's missing facts, never on itself. An open choice also waits on itself.
15. *A ``when`` reading an input that does not apply* leaves its scope unknown,
    waiting on that input (a guard over a value that does not exist neither holds nor
    refuses).
16. *Cycles* are the dependency rule written at :mod:`.description`.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .answers import (
    Absent,
    Answer,
    AssignmentRefusal,
    ChangeRefusal,
    ChoiceState,
    Dormant,
    Impossible,
    Inapplicable,
    Known,
    Open,
    Piece,
    PieceKind,
    Possible,
    Reason,
    ReasonKind,
    Refused,
    RuleRefusal,
    Set,
    Unresolved,
    Validity,
    Verdict,
    Waiting,
    kleene_and,
    piece_key,
    refusal_key,
    sorted_reasons,
)
from .description import Index, Refuse, Space

#: Reading 9 (§ 16.9), not yet ruled on by the human: a change newly emptying a present
#: choice's listed domain is refused. The one place the reading is decided.
EMPTY_DOMAIN_REFUSES = True


class ConfigurationError(ValueError):
    """A configuration or change naming a choice or fact the description lacks, or a
    configuration storing a value that is not of its choice's type."""


def same(a: Any, b: Any) -> bool:
    """Equal and of the same type, so ``True`` is not ``1``."""

    return type(a) is type(b) and bool(a == b)


def _contains(values: tuple[Any, ...], value: Any) -> bool:
    return any(same(item, value) for item in values)


def _distinct(values: tuple[Any, ...]) -> tuple[Any, ...]:
    out: list[Any] = []
    for value in values:
        if not _contains(tuple(out), value):
            out.append(value)
    return tuple(out)


def _same_mapping(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)


def _permutation(ranked: tuple[Any, ...], values: tuple[Any, ...]) -> bool:
    rest = list(values)
    for value in ranked:
        position = next((i for i, item in enumerate(rest) if same(item, value)), None)
        if position is None:
            return False
        del rest[position]
    return not rest


@dataclass(frozen=True)
class _Inputs:
    """A node's declared reads: their values when all are known, else what blocks."""

    values: dict[str, Any] | None
    refused: tuple[str, ...] = ()
    waits: frozenset[str] = frozenset()
    inapplicable: tuple[str, ...] = ()


def _inputs(answers: Mapping[str, Answer]) -> _Inputs:
    values: dict[str, Any] = {}
    refused: list[str] = []
    waits: set[str] = set()
    inapplicable: list[str] = []
    for path, answer in answers.items():
        if isinstance(answer, Known):
            values[path] = answer.value
        elif isinstance(answer, Refused):
            refused.append(path)
        elif isinstance(answer, Unresolved):
            waits.update(answer.waits)
        else:
            inapplicable.append(path)
    if refused or waits or inapplicable:
        return _Inputs(None, tuple(sorted(refused)), frozenset(waits), tuple(sorted(inapplicable)))
    return _Inputs(values)


# --- the facts alone: membership, lists ---------------------------------------------


class FactsOnly:
    """What the facts alone decide: membership, membership lists, and the derived
    values they read, computed from their reads whatever their scope's presence.

    Public for the harness, which reads a refused membership input as membership does
    (reading 6): ``FactsOnly(index, facts).value(path)``, ``.listing(choice)`` and
    ``.membership(choice, value)``."""

    def __init__(self, index: Index, facts: Mapping[str, Any]) -> None:
        self.index = index
        self.facts = facts
        self._values: dict[str, Answer] = {}
        self._applies: dict[tuple[str, str], Answer] = {}

    def value(self, path: str) -> Answer:
        """A fact, or a derived value over facts."""

        if path not in self._values:
            self._values[path] = self._value(path)
        return self._values[path]

    def _value(self, path: str) -> Answer:
        if path in self.index.facts:
            return Known(self.facts[path]) if path in self.facts else Unresolved((path,))
        derived, scope = self.index.derived[path]
        inputs = self.inputs(derived.reads)
        if inputs.refused:
            return Refused((Reason(ReasonKind.INPUT_REFUSED, path, scope, related=inputs.refused),))
        if inputs.values is None:
            return Unresolved(tuple(sorted(inputs.waits)))
        output = derived.compute(inputs.values)
        if isinstance(output, Refuse):
            return Refused((Reason(ReasonKind.DERIVED_BODY, path, scope, code=output.code),))
        return Known(output)

    def inputs(self, reads: tuple[str, ...]) -> _Inputs:
        return _inputs({path: self.value(path) for path in reads})

    def applies(self, path: str, within: str) -> Answer:
        """Whether the membership read ``path`` exists for a choice of the scope ``within``
        (reading 6, § 16.17): ``Known(True)``; ``Known(False)`` when a guard over facts
        beyond the choice's own existence guards is false; ``Unresolved`` while such a
        guard waits on a missing fact; ``Refused`` when it reads a refused input. A fact
        exists; a derived value exists when its scope does and every value it reads does.
        The choice's own guards (its scope's and those above it) hold whenever it is
        present, so they are not asked."""

        key = (path, within)
        if key not in self._applies:
            self._applies[key] = self._applies_now(path, within)
        return self._applies[key]

    def _applies_now(self, path: str, within: str) -> Answer:
        if path in self.index.facts:
            return Known(True)
        derived, scope = self.index.derived[path]
        answers = [self.applies(read, within) for read in derived.reads]
        current: str | None = scope
        while current is not None and not self.index.within(within, current):
            info = self.index.scopes[current]
            assert info.selector is None, (path, current)  # domain-reads-choice
            if info.when is not None:
                answers.extend(self.applies(read, within) for read in info.when.reads)
                inputs = self.inputs(info.when.reads)
                if inputs.refused:
                    reason = Reason(
                        ReasonKind.INPUT_REFUSED, current, current, related=inputs.refused
                    )
                    answers.append(Refused((reason,)))
                elif inputs.values is None:
                    answers.append(Unresolved(tuple(sorted(inputs.waits))))
                else:
                    answers.append(Known(bool(info.when.test(inputs.values))))
            current = info.parent
        if Known(False) in answers:
            return Known(False)
        refused = [r for answer in answers if isinstance(answer, Refused) for r in answer.reasons]
        if refused:
            return Refused(sorted_reasons(refused))
        waits = {w for answer in answers if isinstance(answer, Unresolved) for w in answer.waits}
        return Unresolved(tuple(sorted(waits))) if waits else Known(True)

    def _domain_inputs(self, choice: str) -> _Inputs | Refused:
        info = self.index.choices[choice]
        reads = info.domain.reads
        applies = {read: self.applies(read, info.scope) for read in reads}
        absent = tuple(read for read in reads if applies[read] == Known(False))
        if absent:
            return Refused((Reason(ReasonKind.INPUT_ABSENT, choice, info.scope, related=absent),))
        guarded = [read for read in reads if isinstance(applies[read], Refused)]
        inputs = self.inputs(reads)
        if inputs.refused or guarded:
            related = tuple(sorted({*inputs.refused, *guarded}))
            reasons = [Reason(ReasonKind.INPUT_REFUSED, choice, info.scope, related=related)]
            for path in related:
                answer = self.value(path) if path in inputs.refused else applies[path]
                assert isinstance(answer, Refused)
                reasons.extend(answer.reasons)
            return Refused(sorted_reasons(reasons))
        waits = {
            w for answer in applies.values() if isinstance(answer, Unresolved) for w in answer.waits
        }
        if waits:
            return _Inputs(None, waits=frozenset(waits) | inputs.waits)
        return inputs

    def listing(self, choice: str) -> Answer | None:
        """A listed domain's membership list (``Known`` of a tuple), waiting on missing
        facts, or refused through an input; ``None`` for a membership-only domain."""

        domain = self.index.choices[choice].domain
        if not domain.listed:
            return None
        inputs = self._domain_inputs(choice)
        if isinstance(inputs, Refused):
            return inputs
        if inputs.values is None:
            return Unresolved(tuple(sorted(inputs.waits)))
        if domain.values is not None:
            values = _distinct(domain.values)
        else:
            assert domain.enumerate is not None
            values = _distinct(tuple(domain.enumerate(inputs.values)))
        if domain.member is not None:
            member = domain.member
            reads = inputs.values
            values = tuple(value for value in values if member(value, reads))
        return Known(values)

    def membership(self, choice: str, value: Any) -> Answer:
        """``Known(True)`` for a member, ``Known(False)`` outside, ``Unresolved`` on
        missing facts, ``Refused`` through a refused input."""

        domain = self.index.choices[choice].domain
        if domain.listed:
            listing = self.listing(choice)
            assert listing is not None
            if isinstance(listing, Known):
                return Known(_contains(listing.value, value))
            return listing
        inputs = self._domain_inputs(choice)
        if isinstance(inputs, Refused):
            return inputs
        if inputs.values is None:
            return Unresolved(tuple(sorted(inputs.waits)))
        assert domain.member is not None
        return Known(bool(domain.member(value, inputs.values)))

    def origins(self, path: str, found: dict[str, tuple[str, tuple[Reason, ...]]]) -> None:
        """The derived values a refused ``path`` (over facts) is refused through, at
        their origin."""

        answer = self.value(path)
        if not isinstance(answer, Refused) or path in self.index.facts:
            return
        inputs = self.inputs(self.index.derived[path][0].reads)
        if inputs.refused:
            for read in inputs.refused:
                self.origins(read, found)
        else:
            found[path] = ("derived", answer.reasons)


# --- one configuration's evaluation --------------------------------------------------


class Presence(enum.Enum):
    """A scope's presence. ``REFUSED``: its ``when`` reads a refused input; the scope is
    not in play."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"
    REFUSED = "refused"


@dataclass(frozen=True)
class _Presence:
    state: Presence
    #: what an unknown presence waits on
    waits: frozenset[str] = frozenset()
    #: a refused guard's refused reads, and their reasons
    guards: tuple[str, ...] = ()
    reasons: tuple[Reason, ...] = ()


_PRESENT = _Presence(Presence.PRESENT)
_ABSENT = _Presence(Presence.ABSENT)
_OUT = (Presence.ABSENT, Presence.REFUSED)

#: an origin's kind and its own reasons, by path
_Origins = dict[str, tuple[str, tuple[Reason, ...]]]


class _Evaluation:
    """Every answer of one configuration (its facts and stored choices)."""

    def __init__(self, index: Index, facts: Mapping[str, Any], stored: Mapping[str, Any]) -> None:
        self.index = index
        self.facts = FactsOnly(index, facts)
        self.stored = stored
        self._memo: dict[tuple[str, str], Any] = {}

    def _memoized(self, kind: str, path: str, compute: Callable[[str], Any]) -> Any:
        key = (kind, path)
        if key not in self._memo:
            self._memo[key] = compute(path)
        return self._memo[key]

    # -- presence

    def presence(self, scope: str) -> _Presence:
        found: _Presence = self._memoized("presence", scope, self._presence)
        return found

    def parts(self, scope: str) -> tuple[_Presence, _Presence, _Presence]:
        """A scope's parent's presence, its selector's part and its guard's part."""

        found: tuple[_Presence, _Presence, _Presence] = self._memoized("parts", scope, self._parts)
        return found

    def _parts(self, scope: str) -> tuple[_Presence, _Presence, _Presence]:
        info = self.index.scopes[scope]
        assert info.parent is not None
        parent = self.presence(info.parent)
        selector = _PRESENT
        if info.selector is not None:
            if info.selector in self.stored:
                held = same(self.stored[info.selector], info.case)
                selector = _PRESENT if held else _ABSENT
            else:
                selector = _Presence(Presence.UNKNOWN, frozenset({info.selector}))
        when = _PRESENT
        if info.when is not None and Presence.ABSENT not in (parent.state, selector.state):
            inputs = self.inputs(info.when.reads)
            if inputs.values is not None:
                when = _PRESENT if info.when.test(inputs.values) else _ABSENT
            elif inputs.refused:
                reasons: list[Reason] = []
                for read in inputs.refused:
                    answer = self.read(read)
                    assert isinstance(answer, Refused)
                    reasons.extend(answer.reasons)
                when = _Presence(
                    Presence.REFUSED, guards=inputs.refused, reasons=sorted_reasons(reasons)
                )
            elif inputs.waits:
                when = _Presence(Presence.UNKNOWN, inputs.waits)
            else:
                when = _Presence(Presence.UNKNOWN, frozenset(inputs.inapplicable))
        return parent, selector, when

    def _presence(self, scope: str) -> _Presence:
        if self.index.scopes[scope].parent is None:
            return _PRESENT
        parts = self.parts(scope)
        parent, _, when = parts
        states = {part.state for part in parts}
        if Presence.ABSENT in states:
            return _ABSENT
        if parent.state is Presence.REFUSED:
            return parent
        if Presence.UNKNOWN in states:
            waits = frozenset().union(*(p.waits for p in parts if p.state is Presence.UNKNOWN))
            return _Presence(Presence.UNKNOWN, waits)
        if when.state is Presence.REFUSED:
            return when
        return _PRESENT

    # -- values

    def inputs(self, reads: tuple[str, ...]) -> _Inputs:
        return _inputs({path: self.read(path) for path in reads})

    def read(self, path: str) -> Answer:
        found: Answer = self._memoized("read", path, self._read)
        return found

    def _read(self, path: str) -> Answer:
        kind = self.index.kind(path)
        if kind == "fact":
            return self.facts.value(path)
        scope = self.index.scope_of(path)
        presence = self.presence(scope)
        if presence.state is Presence.ABSENT:
            return Inapplicable()
        if presence.state is Presence.REFUSED:
            reason = Reason(ReasonKind.INPUT_REFUSED, path, scope, related=presence.guards)
            return Refused(sorted_reasons((reason, *presence.reasons)))
        if kind == "choice":
            if path not in self.stored:
                return Unresolved(tuple(sorted(presence.waits | {path})))
            waits = set(presence.waits)
            member = self.facts.membership(path, self.stored[path])
            if isinstance(member, Unresolved):
                waits.update(member.waits)
            else:
                assert member == Known(True), (path, member)
            if waits:
                return Unresolved(tuple(sorted(waits)))
            return Known(self.stored[path])
        if presence.state is Presence.UNKNOWN:
            return Unresolved(tuple(sorted(presence.waits)))
        inputs = self.inputs(self.index.reads_of(path))
        body: Answer
        if inputs.refused:
            body = Refused((Reason(ReasonKind.INPUT_REFUSED, path, scope, related=inputs.refused),))
        elif inputs.waits:
            body = Unresolved(tuple(sorted(inputs.waits)))
        elif inputs.values is None:
            body = Inapplicable()
        else:
            if kind == "derived":
                output = self.index.derived[path][0].compute(inputs.values)
                reason_kind = ReasonKind.DERIVED_BODY
            else:
                output = self.index.views[path][0].body(inputs.values)
                reason_kind = ReasonKind.VIEW_BODY
            if isinstance(output, Refuse):
                body = Refused((Reason(reason_kind, path, scope, code=output.code),))
            else:
                body = Known(output)
        if kind == "derived":
            return body
        reasons = list(body.reasons) if isinstance(body, Refused) else []
        if self.validity(scope).verdict is Verdict.FALSE:
            reasons.append(Reason(ReasonKind.SPACE_REFUSED, path, scope))
        if reasons:
            return Refused(sorted_reasons(reasons))
        return body

    # -- pieces

    def constraint(self, path: str) -> Piece:
        found: Piece = self._memoized("constraint", path, self._constraint)
        return found

    def _constraint(self, path: str) -> Piece:
        rule, scope = self.index.constraints[path]
        kind = PieceKind.CONSTRAINT
        presence = self.presence(scope)
        if presence.state in _OUT:
            return Piece(kind, scope, path, Verdict.TRUE, applies=False)
        if presence.state is Presence.UNKNOWN:
            return Piece(kind, scope, path, Verdict.UNKNOWN, waits=tuple(sorted(presence.waits)))
        inputs = self.inputs(rule.reads)
        if inputs.refused:
            reason = Reason(ReasonKind.INPUT_REFUSED, path, scope, rule.code, inputs.refused)
            return Piece(kind, scope, path, Verdict.FALSE, reasons=(reason,))
        if inputs.waits:
            return Piece(kind, scope, path, Verdict.UNKNOWN, waits=tuple(sorted(inputs.waits)))
        if inputs.values is None:
            return Piece(kind, scope, path, Verdict.TRUE, applies=False)
        if rule.predicate(inputs.values):
            return Piece(kind, scope, path, Verdict.TRUE)
        reason = Reason(ReasonKind.CONSTRAINT, path, scope, rule.code)
        return Piece(kind, scope, path, Verdict.FALSE, reasons=(reason,))

    def domain(self, choice: str) -> Piece | None:
        """A listed domain's piece: none when it lists a value (reading 13), whatever
        its presence; else does it list a value?"""

        scope = self.index.choices[choice].scope
        kind = PieceKind.EMPTY_DOMAIN
        listing = self.facts.listing(choice)
        assert listing is not None
        if isinstance(listing, Known) and listing.value:
            return None
        presence = self.presence(scope)
        if presence.state in _OUT:
            return Piece(kind, scope, choice, Verdict.TRUE, applies=False)
        if presence.state is Presence.UNKNOWN:
            waits = set(presence.waits)
            if isinstance(listing, Unresolved):
                waits.update(listing.waits)
            return Piece(kind, scope, choice, Verdict.UNKNOWN, waits=tuple(sorted(waits)))
        if isinstance(listing, Unresolved):
            return Piece(kind, scope, choice, Verdict.UNKNOWN, waits=listing.waits)
        if isinstance(listing, Refused):
            return Piece(kind, scope, choice, Verdict.FALSE, reasons=listing.reasons)
        reason = Reason(ReasonKind.EMPTY_DOMAIN, choice, scope)
        return Piece(kind, scope, choice, Verdict.FALSE, reasons=(reason,))

    def guard(self, scope: str) -> Piece | None:
        """The piece of a guard refused through an input, at the highest such guard."""

        info = self.index.scopes[scope]
        if info.parent is None or info.when is None:
            return None
        parent, selector, when = self.parts(scope)
        if (
            when.state is not Presence.REFUSED
            or parent.state in _OUT
            or selector.state is Presence.ABSENT
        ):
            return None
        kind = PieceKind.GUARD
        unknown = [part for part in (parent, selector) if part.state is Presence.UNKNOWN]
        if unknown:
            waits = frozenset().union(*(part.waits for part in unknown))
            return Piece(kind, scope, scope, Verdict.UNKNOWN, waits=tuple(sorted(waits)))
        reason = Reason(ReasonKind.INPUT_REFUSED, scope, scope, related=when.guards)
        return Piece(
            kind, scope, scope, Verdict.FALSE, reasons=sorted_reasons((reason, *when.reasons))
        )

    # -- validity

    def validity(self, scope: str) -> Validity:
        found: Validity = self._memoized("validity", scope, self._validity)
        return found

    def _validity(self, scope: str) -> Validity:
        index = self.index
        pieces = [
            self.constraint(name)
            for name, (_, owner) in sorted(index.constraints.items())
            if index.within(owner, scope)
        ]
        for name, info in sorted(index.choices.items()):
            if info.domain.listed and index.within(info.scope, scope):
                domain = self.domain(name)
                if domain is not None:
                    pieces.append(domain)
        for name in sorted(index.scopes):
            if index.within(name, scope):
                guard = self.guard(name)
                if guard is not None:
                    pieces.append(guard)
        verdict = kleene_and(piece.verdict for piece in pieces)
        false = tuple(sorted((p for p in pieces if p.verdict is Verdict.FALSE), key=piece_key))
        unknown = tuple(sorted((p for p in pieces if p.verdict is Verdict.UNKNOWN), key=piece_key))
        return Validity(verdict, false, unknown)

    # -- choices

    def state(self, choice: str) -> ChoiceState:
        info = self.index.choices[choice]
        presence = self.presence(info.scope)
        if choice not in self.stored:
            if presence.state in _OUT:
                return Absent()
            return Open(tuple(sorted(presence.waits)), info.required)
        value = self.stored[choice]
        if presence.state in _OUT:
            return Dormant(value)
        read = self.read(choice)
        if isinstance(read, Unresolved):
            return Waiting(value, read.waits)
        return Set(value)

    # -- origins

    def pairs(self) -> dict[tuple[str, str, str], _Origins]:
        """The counted pairs (reading 2): each false piece of the root's validity, by its
        :func:`~.answers.piece_key`, with the origins it is false through."""

        found: dict[tuple[str, str, str], _Origins] = {}
        for piece in self.validity("").false:
            origins: _Origins = {}
            if piece.kind is PieceKind.CONSTRAINT:
                own = tuple(r for r in piece.reasons if r.kind is ReasonKind.CONSTRAINT)
                if own:
                    origins[piece.subject] = ("constraint", own)
                else:
                    for read in self.inputs(self.index.reads_of(piece.subject)).refused:
                        self._read_origins(read, origins)
            elif piece.kind is PieceKind.EMPTY_DOMAIN:
                listing = self.facts.listing(piece.subject)
                absent = isinstance(listing, Refused) and any(
                    r.kind is ReasonKind.INPUT_ABSENT for r in listing.reasons
                )
                if isinstance(listing, Known) or absent:
                    origins[piece.subject] = ("empty-domain", piece.reasons)
                else:
                    assert isinstance(listing, Refused)
                    reads = self.index.choices[piece.subject].domain.reads
                    for read in self.facts.inputs(reads).refused:
                        self.facts.origins(read, origins)
                    for reason in listing.reasons:  # a refused guard of an input
                        if (
                            reason.kind is ReasonKind.INPUT_REFUSED
                            and reason.subject in self.index.scopes
                        ):
                            for read in reason.related:
                                self.facts.origins(read, origins)
            else:
                for read in self.parts(piece.subject)[2].guards:
                    self._read_origins(read, origins)
            found[piece_key(piece)] = origins
        return found

    def _read_origins(self, path: str, found: _Origins) -> None:
        answer = self.read(path)
        if not isinstance(answer, Refused):
            return
        presence = self.presence(self.index.scope_of(path))
        if presence.state is Presence.REFUSED:
            for read in presence.guards:
                self._read_origins(read, found)
            return
        kind = self.index.kind(path)
        assert kind in ("derived", "view"), (path, kind)
        inputs = self.inputs(self.index.reads_of(path))
        if inputs.refused:
            for read in inputs.refused:
                self._read_origins(read, found)
            return
        body = (ReasonKind.DERIVED_BODY, ReasonKind.VIEW_BODY)
        own = tuple(r for r in answer.reasons if r.kind in body and r.subject == path)
        if own:
            found[path] = (kind, own)


# --- configurations and changes ---------------------------------------------------


@dataclass(frozen=True)
class Configuration:
    """A description, its facts (a missing fact is unknown) and its commitments
    (choice path to stored value), dormant ones included."""

    space: Space
    facts: Mapping[str, Any] = field(default_factory=dict)
    stored: Mapping[str, Any] = field(default_factory=dict)
    #: the validated description (a definition, never an answer)
    index: Index = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = self.space.validate()
        object.__setattr__(self, "index", index)
        for name in self.facts:
            if name not in index.facts:
                raise ConfigurationError(f"unknown fact {name}")
        for path in self.stored:
            if path not in index.choices:
                raise ConfigurationError(f"unknown choice {path}")
        object.__setattr__(self, "facts", dict(self.facts))
        object.__setattr__(self, "stored", dict(self.stored))
        checker = FactsOnly(index, self.facts)
        for path in sorted(self.stored):
            if _assignment(checker, path, self.stored[path]):
                raise ConfigurationError(f"{path}: {self.stored[path]!r} is not of its type")

    def solve(self) -> Solution:
        return Solution(self)


@dataclass(frozen=True)
class Change:
    """Commit or replace (``commit``), clear (``clear``), set facts (``facts``) and
    unset facts (``unset``), as one change."""

    commit: Mapping[str, Any] = field(default_factory=dict)
    clear: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)
    unset: tuple[str, ...] = ()


@dataclass(frozen=True)
class Accepted:
    """The published configuration; the stored choices put dormant and brought into
    play, and the open choices brought into presence; its validity."""

    configuration: Configuration
    dormant: tuple[str, ...]
    in_play: tuple[str, ...]
    opened: tuple[str, ...]
    validity: Validity


@dataclass(frozen=True)
class Rejected:
    """Each refusal, sorted; the would-be configuration's validity, ``None`` when an
    assignment refusal leaves no configuration (reading 12)."""

    refusals: tuple[ChangeRefusal, ...]
    validity: Validity | None


def _assignment(checker: FactsOnly, path: str, value: Any) -> tuple[Reason, ...]:
    """Why ``value`` is not of ``path``'s type, or nothing."""

    member = checker.membership(path, value)
    if isinstance(member, Refused):
        return member.reasons
    if member == Known(False):
        return (Reason(ReasonKind.OUTSIDE_DOMAIN, path, checker.index.choices[path].scope),)
    return ()


def _standing(
    index: Index, origin: str, kind: str, stored: Mapping[str, Any], named: set[str]
) -> tuple[str, ...]:
    """The stored choices standing in an origin's way (reading 4)."""

    if kind == "empty-domain":
        return ()
    reads = index.through_bodies(index.reads_of(origin))
    deciders = index.deciders(index.scope_of(origin))
    return tuple(sorted(path for path in reads if path in stored and path not in named | deciders))


def change(configuration: Configuration, edit: Change) -> Accepted | Rejected:
    """Apply ``edit`` as assignment: accepted with the configuration it publishes and
    what moved, or refused with each refusal."""

    index = configuration.index
    for path in (*edit.commit, *edit.clear):
        if path not in index.choices:
            raise ConfigurationError(f"unknown choice {path}")
    for name in (*edit.facts, *edit.unset):
        if name not in index.facts:
            raise ConfigurationError(f"unknown fact {name}")
    stored = dict(configuration.stored)
    for path in edit.clear:
        stored.pop(path, None)
    stored.update(edit.commit)
    facts = dict(configuration.facts)
    for name in edit.unset:
        facts.pop(name, None)
    facts.update(edit.facts)
    facts_changed = not _same_mapping(facts, configuration.facts)
    if not facts_changed and _same_mapping(stored, configuration.stored):
        return Accepted(configuration, (), (), (), Solution(configuration).validity())

    # 1. assignment: the named values, and every stored value under new facts
    checker = FactsOnly(index, facts)
    checked = sorted(stored) if facts_changed else sorted(edit.commit)
    assignments = [
        AssignmentRefusal(path, reasons)
        for path in checked
        if (reasons := _assignment(checker, path, stored[path]))
    ]
    if assignments:
        return Rejected(tuple(assignments), None)

    successor = Configuration(configuration.space, facts, stored)
    before = Solution(configuration)
    after = Solution(successor)

    # 2 and 3. each origin of a counted pair the predecessor lacks refuses a change that
    # sets, once; clears and unsets alone are never refused (§ 16.16)
    sets = bool(edit.commit) or bool(edit.facts)
    counted = before.evaluation.pairs()
    named = set(edit.commit) | set(edit.clear)
    refusals: dict[str, ChangeRefusal] = {}
    for piece, origins in sorted(after.evaluation.pairs().items() if sets else ()):
        for origin, (kind, reasons) in sorted(origins.items()):
            if origin in counted.get(piece, {}) or origin in refusals:
                continue
            if kind == "empty-domain" and not EMPTY_DOMAIN_REFUSES:
                continue
            standing = _standing(index, origin, kind, stored, named)
            refusals[origin] = RuleRefusal(origin, kind, reasons, standing)
    if refusals:
        return Rejected(tuple(sorted(refusals.values(), key=refusal_key)), after.validity())

    # 4. what moved
    dormant: list[str] = []
    in_play: list[str] = []
    opened: list[str] = []
    for path in index.ordered_choices():
        scope = index.choices[path].scope
        was = before.presence(scope)
        now = after.presence(scope)
        if path in stored:
            if was not in _OUT and now in _OUT:
                dormant.append(path)
            if was is not Presence.PRESENT and now is Presence.PRESENT:
                in_play.append(path)
        elif was is not Presence.PRESENT and now is Presence.PRESENT:
            opened.append(path)
    return Accepted(successor, tuple(dormant), tuple(in_play), tuple(opened), after.validity())


# --- a configuration's answers -------------------------------------------------------


class Solution:
    """Every answer of one configuration, computed from scratch."""

    def __init__(self, configuration: Configuration) -> None:
        self.configuration = configuration
        self.index = configuration.index
        self.evaluation = _Evaluation(self.index, configuration.facts, configuration.stored)

    def _choice(self, choice: str) -> None:
        if choice not in self.index.choices:
            raise ConfigurationError(f"unknown choice {choice}")

    def presence(self, scope: str) -> Presence:
        if scope not in self.index.scopes:
            raise ConfigurationError(f"unknown scope {scope}")
        return self.evaluation.presence(scope).state

    def state(self, choice: str) -> ChoiceState:
        self._choice(choice)
        return self.evaluation.state(choice)

    def states(self) -> dict[str, ChoiceState]:
        return {name: self.state(name) for name in self.index.ordered_choices()}

    def dormant(self) -> tuple[str, ...]:
        """The stored choices out of play, sorted."""

        return tuple(name for name, s in self.states().items() if isinstance(s, Dormant))

    def waiting(self) -> tuple[str, ...]:
        """The stored choices waiting, sorted."""

        return tuple(name for name, s in self.states().items() if isinstance(s, Waiting))

    def validity(self, scope: str = "") -> Validity:
        if scope not in self.index.scopes:
            raise ConfigurationError(f"unknown scope {scope}")
        return self.evaluation.validity(scope)

    def read(self, path: str) -> Answer:
        """A fact's, a choice's, a derived value's or a view's answer."""

        if self.index.kind(path) is None:
            raise ConfigurationError(f"unreadable {path}")
        return self.evaluation.read(path)

    def view(self, path: str) -> Answer:
        if path not in self.index.views:
            raise ConfigurationError(f"unknown view {path}")
        return self.evaluation.read(path)

    def piece(self, constraint: str) -> Piece:
        """A constraint's piece."""

        if constraint not in self.index.constraints:
            raise ConfigurationError(f"unknown constraint {constraint}")
        return self.evaluation.constraint(constraint)

    def possible(self, choice: str) -> tuple[tuple[Any, Possible | Impossible], ...] | None:
        """Each member value of ``choice``'s listed domain, in membership's list order,
        with what setting it alone answers; ``None`` when the list cannot be made (a
        membership-only domain, or a list waiting on a missing fact or refused through
        an input)."""

        self._choice(choice)
        listing = self.evaluation.facts.listing(choice)
        if not isinstance(listing, Known):
            return None
        out: list[tuple[Any, Possible | Impossible]] = []
        for value in listing.value:
            result = change(self.configuration, Change(commit={choice: value}))
            if isinstance(result, Accepted):
                out.append((value, Possible(result.dormant, result.in_play, result.opened)))
            else:
                out.append((value, Impossible(result.refusals)))
        return tuple(out)

    def ordered(self, choice: str) -> Answer | None:
        """The membership list ranked by the domain's order: known, waiting while the
        list or an order read waits, refused through a refused input, inapplicable when
        an order read does not apply; ``None`` for a membership-only domain."""

        self._choice(choice)
        info = self.index.choices[choice]
        listing = self.evaluation.facts.listing(choice)
        if not isinstance(listing, Known) or info.domain.order is None:
            return listing
        inputs = self.evaluation.inputs(info.domain.order_reads)
        if inputs.refused:
            reason = Reason(ReasonKind.INPUT_REFUSED, choice, info.scope, related=inputs.refused)
            return Refused((reason,))
        if inputs.waits:
            return Unresolved(tuple(sorted(inputs.waits)))
        if inputs.values is None:
            return Inapplicable()
        ranked = tuple(info.domain.order(listing.value, inputs.values))
        assert _permutation(ranked, listing.value), (choice, ranked, listing.value)
        return Known(ranked)
