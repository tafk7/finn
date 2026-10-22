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
a choice is set, or open. A configuration missing a required fact, or storing a value
outside its domain or whose domain's membership reads a refused input, is no
configuration (:class:`ConfigurationError`); a stored set breaking a rule is one, and
reads refused.

**Domains.** Membership reads facts only, so it is a type check fixed once the facts
are given: a value is a member, outside, refused through a refused input, or refused
because an input does not apply (reading 6). Facts are always known (SZ100 (5)), so
membership never waits. A listed
domain's membership list is its listed values that are members, in listed order; its
``order`` ranks that list and checks nothing.

**Presence.** A scope is present; absent (another case selected, or its ``when``
false); unknown (its selector open, or a ``when`` input waiting); or refused (its
``when`` reads a refused input: a derived value refusing; or it is a case of a derived
selector whose value refuses, reading 17). A scope under a
refused guard is not in play: its stored choices are dormant, its open choices
absent, its constraints add nothing, and every value in it reads refused through the
guard.

**A choice** is set (stored, in play, read as its value), waiting (stored, its
presence unknown; never read as a value),
dormant (stored, its scope absent or not in play), open (unset, its scope present or
unknown) or absent (unset, its scope absent or not in play).

**Reads.** A fact is always known (SZ100 (5)). A description declares each fact
required or optional (:class:`~.description.Space`, ``optional``): a required fact is
given with every configuration, and an optional fact left out, or unset, reads
``None``, a known value its readers handle. "Unresolved" belongs to open choices alone.
A derived value or a view
reads its inputs: all known, its body answers (a :class:`~.description.Refuse` is a
refusal at its origin); an input refused, it is refused through it; an input waiting,
it waits; an input inapplicable, it is inapplicable. A view also refuses, through,
when its Space's validity is false. *Views are read out* (SZ100 (3)): no constraint,
derived value, ``when`` guard or domain (its membership, its computed list, its
``order``) reads a view, and a description that does is no definition
(``view-read-in``); a view reads anything readable, other views included, its own
Space's and its children's. A ``Members`` (SZ100 (6)) reads its entries,
derived values in scopes below it: its value is the (entry, value) pairs of the
entries that answer a value, sorted by entry. That the model's ``Members`` ranges over
derived values is a restriction of the model, not of the semantics: views compose by
reading views, and aggregating views is a view reading views, so a ``Members`` over
views, read by views alone, would be view-side (read-out aggregation, as a Space
aggregating its members' views for its own views does). The model does not express that
view-side ``Members``, and refuses a ``Members`` naming a view as it refuses any other
reader of a view (``view-read-in``); the semantics would not. An entry answering
inapplicable (its scope absent, or present with an inapplicable value) is skipped.
Otherwise, in this order: any entry refused, by its value or by its presence, refuses
the whole through it, even while another waits; else any entry unresolved makes it
wait. Its reads made are the entries that answered a value (reading 4).

**Validity** of a scope is the strong three-valued AND
(:func:`~.answers.kleene_and`) of the pieces at or below it: each constraint (absent
or not in play: true, not applying; an input refused: false through it; an input
waiting: unknown; an input inapplicable: true, not applying; else its predicate on its
own values, false at its origin); each listed domain that lists no value (false), or
whose list cannot be made (reading 13); each guard refused through an input, a derived
selector's refusing value among them (false; readings 8 and 17). A listed domain that
lists a value is no piece, whatever its presence. A piece of a scope whose presence is
unknown is unknown, whatever it would answer. No choice is otherwise a piece.

**A change** (commit or replace choices, clear choices, set or unset facts, as one
batch) publishes its successor whole, or is refused with each refusal and the original
kept. A change of choices is assignment; a change of facts instantiates a new design
space, and is never refused for its facts (§ 16.24).

1. *Assignment.* Each committed value is checked against its domain. A value outside
   its domain, or whose membership reads an input that refuses or does not apply,
   refuses the change (:class:`~.answers.AssignmentRefusal`).
2. *Counted pairs.* An origin is a constraint false on its own values; a derived value
   refusing on its own values; a present choice whose listed domain lists no value (a
   view refusing on its own values feeds no piece: SZ100 (3)). A configuration's counted
   pairs are each false piece of the root's validity with each origin it is false
   through: the piece itself, or the origins its refused inputs lead back to.
3. *Refusal.* A change of choices that sets a choice to a new value is refused when the
   successor holds a counted pair the predecessor does not: a piece false through an
   origin it was not false through before (a piece absent, unknown or true before counts
   as not false before). Each origin of such a pair refuses it once
   (:class:`~.answers.RuleRefusal`), naming the stored choices standing in its way. A
   rule false only through a refused input is never itself a refusal. So no accepted
   change of choices that sets adds a pair. A change of choices that sets nothing new
   (its commits repeat stored values) and clears is never refused, and adds no pair
   (reading 3).
4. *New facts* (§ 16.24, § 16.25). A change of facts is the bind of the stored choices
   under the new facts (:func:`bind`), then its choices as a change of choices on that
   design space, published together or not at all. The bind type-checks every stored
   value the change does not name, dormant ones included, as if its choice existed
   (reading 6): outside its domain, or its domain's input refusing or not applying, the
   value is dropped (:class:`~.answers.Dropped`); inside, it is kept. Nothing of that
   refuses. Unsetting an optional fact sets it to ``None``, a change of facts like any
   other; a required fact cannot be unset (:class:`ConfigurationError`). A choice the
   change names is kept where its old value is inside its new domain and left out
   silently where not; its new value is then checked by 1, and the choices are judged by
   3 against the bind's pairs. The accepted report lists what was dropped and each rule
   the published design space holds newly false against the receiver (``broken``), with
   its standing choices read in the published design space, none left out for being
   named. A pair the bind added and the choices removed is not in the published design
   space, so it is not reported; nor is a pair already false before.
5. *Accepted*, the change reports the stored choices it puts dormant or brings into
   play and the open choices it brings into presence. A change that changes nothing
   returns the same configuration.

**Possible values** of a choice with a known membership list are one change per member
value, in membership's list order: possible (with what moves) or impossible (with
each refusal). The ranking is a separate answer; validity and refusals never read it.

Readings chosen where the semantics leave room
==============================================

The note is ``records/space-semantics-defaults-2026-10-18/NOTE.md`` in the design
corpus; § 16 holds the readings its author gave for this model.

1. *A view refused because its Space is refused is refused through, never an origin*
   (FN2 for views): its Space's false pieces are counted on their own. A view's body
   refusing on its own values refuses the view with its own reason (§ 16.1); no piece
   reads a view (SZ100 (3)), so neither is ever in a pair.
2. *Only an origin feeding a false piece counts* (§ 16.2, kept by § 16.15). A piece's
   origins are found from it, following refused inputs back through derived values,
   ``Members`` (to their refused entries; a ``Members`` is never an origin), and
   choices and values refused through a guard (to the guard's refused reads);
   a listed domain's list refused through its membership's input, back over the facts.
   A refusing derived value nothing rules on reads refused where it is asked, is in no
   pair, and refuses no change; nor is a view, which no piece reads (SZ100 (3)).
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
   Every accepted change of choices that sets leaves the set of pairs no larger (a
   change of facts may enlarge it, § 16.24), so a change from nothing committing the
   stored set of a design space reached by such changes is accepted (FN3), and no
   accepted addition leaves a Space newly refused (§ 16.14's exception cannot arise).
   *Clears are never refused* (§ 16.16; § 5.6 stands): clearing stays the remedy a
   refusal names; an unset is a change of facts, never refused either. "Sets" means a
   new value (§ 16.18): committing the value already stored sets nothing, so a change
   whose commits all repeat stored values is clear-only too (as a same-value update
   returns the receiver). Consequence (SZ100 (3)): a clear-only change adds no pair. A
   clear only moves answers towards unresolved, and the one shape that could add a pair,
   masking (a piece false only through a view refused with its Space, revealed when the
   change reopens that Space: § 16.14, § 16.16's note), has no subject, since no piece
   reads a view. So no accepted change of choices adds a pair, clears included.
4. *What stands in the way* (§ 16.4; SZ100 (6): standing is what the origin's
   evaluation read): the stored choices the origin's evaluation read for their value,
   in the design space judged, followed through the reads made by the bodies of the
   derived values, views and ``Members`` it read. The model evaluates every declared
   input before a body runs and has no short-circuit, so a node's reads made are its
   declared reads, but for a ``Members``: its reads made are the entries that answered
   a value (an entry answering inapplicable is skipped, never read). The exceptions,
   stated whole: reads made only to decide existence are not followed (each node's own
   guards, at every node read through and not only the origin; a member's existence
   reads; a selector read to find its case; a view's Space's validity); less every
   choice the change names (committed or cleared) and every choice deciding the
   origin's existence (the selectors of the scopes above it, its own included, and the
   choices its guards' ``when`` reads depend on). An empty domain, a choice's own
   domain as an origin, has nothing standing (membership reads facts only). A choice
   deciding only whether something the origin reads exists (``s`` for a root rule
   reading ``s.a.p``; the choice of a ``Members`` entry its scope's absence skips) is
   no read made, so never stands (§ 16.15); one the origin also reads as a value stands
   like any other read (§ 16.22). The model has no ``present`` origin.
5. *In play, waiting, dormant* (§ 16.5), by presence before and after, over the
   successor's stored choices: ``in_play``, not present before and present after;
   ``dormant``, present or unknown before and absent or not in play after; ``opened``,
   the open choices not present before and present after. Clearing a selector leaves
   its cases' stored choices waiting. An unset choice not in play is
   :class:`~.answers.Absent`, a state the note's table lacks.
6. *Membership that cannot be decided* (§ 16.6): a refusing input refuses the
   assignment, and a change of facts making it refuse while a value is stored drops the
   value, its domain piece false through the input. (SZ100 (5): no membership waits on
   a missing fact any more; an optional fact left out reads ``None``.) Membership is
   evaluated over the facts alone: a derived value it reads is computed from its reads
   whatever its own scope's presence. *An input that does not apply is refused* (§ 16.17,
   ``INPUT_ABSENT``, the choice's reason naming the input), not unresolved: only a change
   of facts could settle it. Its existence is decided over the facts alone: the choice's
   own existence guards (its scope's and those above it, a ``when`` over a choice alike)
   are taken as holding, wherever the choice sits (§ 16.20), and are not asked; any other
   guard on the input's scopes (or on those of the derived values it reads) is a
   ``when`` over facts, a selector or a ``when`` reaching a choice being
   ``domain-reads-choice``. A false one makes the input absent, a refused one refuses it
   through that guard's input. An optional fact left out is ``None`` there too: a guard
   over it holds or not as its test says (a former ``supplied()`` is a ``when`` on ``f
   is not None``, still absence: ``INPUT_ABSENT`` stays).
7. *The constructor* (§ 16.7) refuses a stored value outside its domain or whose
   membership input refuses, dormant ones included: a :class:`Configuration` never holds
   one. A change of facts and :func:`bind` drop such a value instead (§ 16.24). A stored
   set breaking a rule is a legal :class:`Configuration`, reading refused: a change of
   choices refuses it, a bind keeps it and reports the rule.
8. *A guard refused through an input* (§ 16.8) is one ``GUARD`` piece, owned by the
   guarded scope, at the highest such guard (scopes under it add nothing). A refused
   ``when`` below a selector or a parent that is still unknown leaves the scope unknown
   and its piece unknown: refused takes effect once all above it is present.
9. *A change of facts emptying a present choice's listed domain is accepted* (§ 16.9,
   settled the other way by § 16.24): the design space reads refused through the
   choice's ``EMPTY_DOMAIN`` piece, whose pair with itself the report lists in
   ``broken``, nothing standing. A selector switch bringing such a choice into play is
   still refused, by 3, as a newly present false piece, nothing standing. A domain
   already empty before the change adds no pair.
10. *A domain's order* (§ 16.10) ranks the membership list over ``order_reads``, which
    may read choices, never a view (SZ100 (3)): :meth:`Solution.ordered` is the
    ranking, known, or waiting while a read waits. Possible values are answered in
    membership's list order.
11. *An unknown guard answers unknown* for every piece under it (constraints, empty
    listed domains, guards), not Kleene's implication (the canon's text in § 11: "a scope
    whose guard waits counts each of its pieces as unknown").
12. *Assignment comes first* (§ 5.1): a change with an assignment refusal reports those
    alone, and its ``validity`` is ``None``: a value of the wrong type makes no
    configuration whose rules could be read.
13. *A listed domain is a piece only when it lists no value* (§ 4.6, § 16.12): false
    when its list is empty and its choice present (the origin), unknown when it is
    empty under a guard that waits (SZ100 (5): never while it waits on a fact), and
    false through the input when its membership reads a refused input (the list cannot
    be made; blamed at that input's origin). When an input does not apply (§ 16.17,
    reading 6) the list cannot be made either and the piece is false through it; the
    origin is the domain itself, as for an empty list (reading 9 holds for it alike). A
    list with a value adds no piece, whatever its presence: an open selector whose case
    holds a listed choice and no rules leaves the configuration valid. A membership-only
    domain has no piece.
14. *A stored choice waits on what it waits on*: an unknown presence's waits, never on
    itself (SZ100 (5): never on a fact). An open choice also waits on itself.
15. *A ``when`` reading an input that does not apply* leaves its scope unknown,
    waiting on that input (a guard over a value that does not exist neither holds nor
    refuses).
16. *Cycles* are the dependency rule written at :mod:`.description`.
17. *A selector picked by a derived value* (``Selector(by=d)``, the engine's
    ``Select(by=d)``; § 2.1's table of ``records/selectors-ownership-2026-10-21/NOTE.md``).
    It is no choice: no key, never stored, never listed. Its path reads its value: ``d``'s
    where that names one of its cases; refused ``"select-no-case"`` at the selector where
    it names none (an origin of kind ``"derived"``, the selector's own); refused through
    ``d`` where ``d`` refuses. A picker never answers "nothing": ``None`` names no case (a
    case that places nothing does). A case is present where the value names it and its
    holding scope is present; absent where it names another case; unknown while the value
    waits (or reads an input that does not apply, as reading 15), so the pieces of the
    rules inside are unknown, as under an open chosen selector (reading 13); and not in
    play where the value refuses, as under a guard refused through an input (reading 8):
    one ``GUARD`` piece, owned by and in the selector's own name, its cases adding none,
    their stored choices dormant. *Standing:* the selector reads its value to find its
    case, an existence read, so the choices ``d`` reads decide its cases' existence
    (reading 4's deciders). They never stand for an origin inside a case, even where it
    reads them as values; they stand for an origin outside the cases that reads them as
    values (§ 16.22). *The domain rule:* a domain's input existing under a case is
    ``domain-reads-choice`` exactly when ``d`` reads a choice (a picker over facts alone is
    a ``when`` over facts); an input in the choice's own case is a definition whatever
    picks that case, its guard being the choice's own existence guard (§ 16.20).
26. *Refused, then inapplicable, is a move up* (numbered as the semantics note records it;
    ruled in the design's review, § 10 of ``records/selectors-ownership-2026-10-21/NOTE.md``).
    Presence is Kleene's AND with absent dominating refused (§ 16.21:
    :meth:`_Evaluation._presence` asks absent first), so a scope refused through a guard
    above can still be found absent when another part of its path is decided (a chosen
    selector inside a case that is out of play is set): an answer refused through its
    scope's presence becomes inapplicable. It never becomes present. So an addition may move
    an answer refused **through its scope's presence** to inapplicable, and invariant 2
    (additions move answers only up) accepts exactly that move, nothing wider: an answer
    refused on its own never becomes inapplicable while its scope stays present.
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
    Dropped,
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
            # always known (SZ100 (5)): an optional fact left out is None
            if path in self.facts:
                return Known(self.facts[path])
            if path in self.index.optional:
                return Known(None)
            raise ConfigurationError(f"the required fact {path} is not given")
        derived, scope = self.index.derived[path]
        inputs = self.inputs(derived.reads)
        if inputs.refused:
            return Refused((Reason(ReasonKind.INPUT_REFUSED, path, scope, related=inputs.refused),))
        assert inputs.values is not None, path  # over facts, always known
        output = derived.compute(inputs.values)
        if isinstance(output, Refuse):
            return Refused((Reason(ReasonKind.DERIVED_BODY, path, scope, code=output.code),))
        return Known(output)

    def inputs(self, reads: tuple[str, ...]) -> _Inputs:
        return _inputs({path: self.value(path) for path in reads})

    def applies(self, path: str, within: str) -> Answer:
        """Whether the membership read ``path`` exists for a choice of the scope ``within``
        (reading 6, § 16.17): ``Known(True)``; ``Known(False)`` when a guard over facts
        beyond the choice's own existence guards is false; ``Refused`` when it reads a
        refused input. Never waiting: facts are always known (SZ100 (5)). A fact
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
            if info.picker is not None:
                # a derived selector's case, its value over facts here: does it name the case
                answers.append(self.applies(info.picker, within))
                picked = self.value(info.picker)
                if isinstance(picked, Refused):
                    reason = Reason(
                        ReasonKind.INPUT_REFUSED, current, current, related=(info.picker,)
                    )
                    answers.append(Refused((reason,)))
                else:
                    assert isinstance(picked, Known), (path, current)
                    answers.append(Known(same(picked.value, info.case)))
            if info.when is not None:
                answers.extend(self.applies(read, within) for read in info.when.reads)
                inputs = self.inputs(info.when.reads)
                if inputs.refused:
                    reason = Reason(
                        ReasonKind.INPUT_REFUSED, current, current, related=inputs.refused
                    )
                    answers.append(Refused((reason,)))
                else:
                    assert inputs.values is not None, current
                    answers.append(Known(bool(info.when.test(inputs.values))))
            current = info.parent
        if Known(False) in answers:
            return Known(False)
        refused = [r for answer in answers if isinstance(answer, Refused) for r in answer.reasons]
        if refused:
            return Refused(sorted_reasons(refused))
        return Known(True)

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
        assert inputs.values is not None, choice
        return inputs

    def listing(self, choice: str) -> Answer | None:
        """A listed domain's membership list (``Known`` of a tuple), or refused through an
        input; ``None`` for a membership-only domain."""

        domain = self.index.choices[choice].domain
        if not domain.listed:
            return None
        inputs = self._domain_inputs(choice)
        if isinstance(inputs, Refused):
            return inputs
        assert inputs.values is not None
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
        """``Known(True)`` for a member, ``Known(False)`` outside, ``Refused`` through a
        refused input."""

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
        assert domain.member is not None and inputs.values is not None
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
        elif info.picker is not None and parent.state is not Presence.ABSENT:
            selector = self.picked(info.picker, info.case)
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

    def picked(self, picker: str, case: str | None) -> _Presence:
        """A derived selector's part in its case's presence (reading 17): present where its
        value names the case, absent where it names another, unknown while it waits (or
        reads an input that does not apply: reading 15), refused where it refuses (through
        its input, or naming no case)."""

        answer = self.read(picker)
        if isinstance(answer, Known):
            return _PRESENT if same(answer.value, case) else _ABSENT
        if isinstance(answer, Unresolved):
            return _Presence(Presence.UNKNOWN, frozenset(answer.waits))
        if isinstance(answer, Refused):
            return _Presence(Presence.REFUSED, guards=(picker,), reasons=answer.reasons)
        return _Presence(Presence.UNKNOWN, frozenset({picker}))

    def _presence(self, scope: str) -> _Presence:
        if self.index.scopes[scope].parent is None:
            return _PRESENT
        parts = self.parts(scope)
        parent, selector, when = parts
        states = {part.state for part in parts}
        if Presence.ABSENT in states:
            return _ABSENT
        if parent.state is Presence.REFUSED:
            return parent
        if Presence.UNKNOWN in states:
            waits = frozenset().union(*(p.waits for p in parts if p.state is Presence.UNKNOWN))
            return _Presence(Presence.UNKNOWN, waits)
        if selector.state is Presence.REFUSED:
            return selector  # a derived selector refusing: its cases are not in play
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
            if presence.waits:
                return Unresolved(tuple(sorted(presence.waits)))
            member = self.facts.membership(path, self.stored[path])
            assert member == Known(True), (path, member)
            return Known(self.stored[path])
        if presence.state is Presence.UNKNOWN:
            return Unresolved(tuple(sorted(presence.waits)))
        if kind == "members":
            return self._members(path, scope)
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

    def _members(self, path: str, scope: str) -> Answer:
        """A ``Members``'s answer: its entries that answer a value, as sorted pairs; an
        inapplicable entry skipped; any refused entry refusing it through that entry,
        else any unresolved one making it wait."""

        entries = {entry: self.read(entry) for entry in self.index.reads_of(path)}
        refused = tuple(sorted(e for e, a in entries.items() if isinstance(a, Refused)))
        if refused:
            return Refused((Reason(ReasonKind.INPUT_REFUSED, path, scope, related=refused),))
        waits = {w for a in entries.values() if isinstance(a, Unresolved) for w in a.waits}
        if any(isinstance(a, Unresolved) for a in entries.values()):
            return Unresolved(tuple(sorted(waits)))
        return Known(
            tuple(sorted((e, a.value) for e, a in entries.items() if isinstance(a, Known)))
        )

    def reads_made(self, path: str) -> tuple[str, ...]:
        """What ``path``'s evaluation read (reading 4): its declared reads, but for a
        ``Members``, the entries that answered a value."""

        reads = self.index.reads_of(path)
        if path in self.index.members:
            return tuple(entry for entry in reads if isinstance(self.read(entry), Known))
        return reads

    def standing(self, origin: str, kind: str, named: set[str]) -> tuple[str, ...]:
        """The stored choices standing in an origin's way (reading 4): read for their
        value by its evaluation, followed through the reads made by the bodies it read,
        less ``named`` and the choices deciding its existence."""

        if kind == "empty-domain":
            return ()
        index = self.index
        found: set[str] = set()
        stack = list(self.reads_made(origin))
        while stack:
            path = stack.pop()
            if path in found:
                continue
            found.add(path)
            if path in index.derived or path in index.views or path in index.members:
                stack.extend(self.reads_made(path))
        deciders = index.deciders(index.scope_of(origin))
        return tuple(sorted(p for p in found if p in self.stored and p not in named | deciders))

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
            waits = tuple(sorted(presence.waits))
            return Piece(kind, scope, choice, Verdict.UNKNOWN, waits=waits)
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

    def picker_guard(self, picker: str) -> Piece | None:
        """A derived selector whose value refuses (through its input, or naming no case):
        one ``GUARD`` piece in the selector's own name, standing for its cases, none of which
        adds one (readings 8 and 17). None where its holding scope is not present: absent,
        refused above (that guard's piece counts), or unknown (its value then waits)."""

        owner = self.index.pickers[picker][0]
        if self.presence(owner).state is not Presence.PRESENT:
            return None
        answer = self.read(picker)
        if not isinstance(answer, Refused):
            return None
        reason = Reason(ReasonKind.INPUT_REFUSED, picker, picker, related=(picker,))
        return Piece(
            PieceKind.GUARD,
            picker,
            picker,
            Verdict.FALSE,
            reasons=sorted_reasons((reason, *answer.reasons)),
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
        for name, (owner, _, _) in sorted(index.pickers.items()):
            if index.within(owner, scope):
                guard = self.picker_guard(name)
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
            elif piece.subject in self.index.pickers:
                self._read_origins(piece.subject, origins)  # a derived selector refusing
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
        assert kind in ("derived", "members"), (path, kind)  # no piece reads a view
        inputs = self.inputs(self.index.reads_of(path))
        if inputs.refused or kind == "members":
            for read in inputs.refused:
                self._read_origins(read, found)
            return
        own = tuple(
            r for r in answer.reasons if r.kind is ReasonKind.DERIVED_BODY and r.subject == path
        )
        if own:
            found[path] = (kind, own)


# --- configurations and changes ---------------------------------------------------


@dataclass(frozen=True)
class Configuration:
    """A description, its facts and its commitments (choice path to stored value),
    dormant ones included. Every required fact is given (else
    :class:`ConfigurationError`); an optional fact left out is ``None`` (SZ100 (5)), and
    ``facts`` holds it so."""

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
        missing = sorted(index.facts - index.optional - set(self.facts))
        if missing:
            raise ConfigurationError(f"the required facts {missing} are not given")
        facts = {name: None for name in sorted(index.optional)}
        facts.update(self.facts)
        object.__setattr__(self, "facts", facts)
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
    play, and the open choices brought into presence; its validity. Under new facts
    (a change of facts, or :func:`bind`): the stored values left out (``dropped``, by
    choice) and the rules the published design space holds newly false (``broken``, by
    :func:`~.answers.refusal_key`), each with its standing choices (§ 16.24). Both are
    empty for a change of choices alone."""

    configuration: Configuration
    dormant: tuple[str, ...]
    in_play: tuple[str, ...]
    opened: tuple[str, ...]
    validity: Validity
    dropped: tuple[Dropped, ...] = ()
    broken: tuple[RuleRefusal, ...] = ()


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


def _sets_new(configuration: Configuration, edit: Change) -> bool:
    """Whether the change of choices ``edit`` sets a choice to a new value (§ 16.18): a
    committed value differing from the one stored. A change that does not is clear-only,
    never refused (§ 16.16)."""

    old = configuration.stored
    return any(path not in old or not same(old[path], value) for path, value in edit.commit.items())


def _standing(evaluation: _Evaluation, origin: str, kind: str, named: set[str]) -> tuple[str, ...]:
    """The stored choices standing in an origin's way (reading 4), read in the design
    space judged."""

    return evaluation.standing(origin, kind, named)


def _newly_false(
    before: Solution, after: Solution, named: set[str], sets: bool = True
) -> dict[str, RuleRefusal]:
    """Each origin of a counted pair ``after`` holds and ``before`` does not, once, with
    its standing choices in ``after`` less ``named`` (reading 4); nothing unless ``sets``."""

    counted = before.evaluation.pairs()
    found: dict[str, RuleRefusal] = {}
    for piece, origins in sorted(after.evaluation.pairs().items() if sets else ()):
        for origin, (kind, reasons) in sorted(origins.items()):
            if origin in counted.get(piece, {}) or origin in found:
                continue
            standing = _standing(after.evaluation, origin, kind, named)
            found[origin] = RuleRefusal(origin, kind, reasons, standing)
    return found


#: the stored choices put dormant, those brought into play, the open choices opened
_Moved = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def _moved(before: Solution, after: Solution) -> _Moved:
    """The stored choices put dormant and brought into play, and the open choices brought
    into presence, from ``before`` to ``after`` (reading 5)."""

    dormant: list[str] = []
    in_play: list[str] = []
    opened: list[str] = []
    index = after.index
    for path in index.ordered_choices():
        scope = index.choices[path].scope
        was = before.presence(scope)
        now = after.presence(scope)
        if path in after.configuration.stored:
            if was not in _OUT and now in _OUT:
                dormant.append(path)
            if was is not Presence.PRESENT and now is Presence.PRESENT:
                in_play.append(path)
        elif was is not Presence.PRESENT and now is Presence.PRESENT:
            opened.append(path)
    return tuple(dormant), tuple(in_play), tuple(opened)


def _instantiate(
    index: Index,
    space: Space,
    facts: Mapping[str, Any],
    stored: Mapping[str, Any],
    named: set[str],
) -> tuple[Configuration, tuple[Dropped, ...]]:
    """The design space of ``facts`` keeping each of ``stored`` that is of its type under
    them (§ 16.24): a value outside its domain, or whose domain's input refuses or does not
    apply, is left out, and reported unless ``named``. Membership reads facts only, so what
    is left out does not depend on order."""

    checker = FactsOnly(index, facts)
    kept: dict[str, Any] = {}
    dropped: list[Dropped] = []
    for path in sorted(stored):
        reasons = _assignment(checker, path, stored[path])
        if not reasons:
            kept[path] = stored[path]
        elif path not in named:
            dropped.append(Dropped(path, stored[path], reasons))
    return Configuration(space, facts, kept), tuple(dropped)


def _new_facts_report(
    receiver: Configuration, published: Configuration, dropped: tuple[Dropped, ...]
) -> Accepted:
    """The accepted report of new facts: what moved from ``receiver``, what was dropped,
    and each rule ``published`` holds newly false against ``receiver``, its standing
    choices read in ``published`` with none left out for being named (§ 16.24)."""

    before, after = Solution(receiver), Solution(published)
    broken = _newly_false(before, after, set())
    return Accepted(
        published,
        *_moved(before, after),
        after.validity(),
        dropped,
        tuple(sorted(broken.values(), key=refusal_key)),
    )


def bind(space: Space, facts: Mapping[str, Any], stored: Mapping[str, Any]) -> Accepted:
    """The design space of ``facts`` with exactly the ``stored`` choices (§ 16.24, SZ90):
    each is checked against its domain under ``facts``, and dropped where it falls outside
    (or its domain's input refuses or does not apply), kept where it is inside. Never
    refused for rules. Its baseline is the design space of ``facts``
    with nothing stored: ``broken`` lists every rule the kept choices break with those
    facts, and what moved is read against it."""

    baseline = Configuration(space, facts)
    for path in stored:
        if path not in baseline.index.choices:
            raise ConfigurationError(f"unknown choice {path}")
    published, dropped = _instantiate(baseline.index, space, facts, stored, set())
    return _new_facts_report(baseline, published, dropped)


def _choices(configuration: Configuration, edit: Change) -> Accepted | Rejected:
    """A change of choices alone (``edit`` changes no fact): assignment, then refusal by
    a new pair."""

    index = configuration.index
    stored = dict(configuration.stored)
    for path in edit.clear:
        stored.pop(path, None)
    stored.update(edit.commit)
    if _same_mapping(stored, configuration.stored):
        return Accepted(configuration, (), (), (), Solution(configuration).validity())

    # 1. assignment: the named values
    checker = FactsOnly(index, configuration.facts)
    assignments = [
        AssignmentRefusal(path, reasons)
        for path in sorted(edit.commit)
        if (reasons := _assignment(checker, path, stored[path]))
    ]
    if assignments:
        return Rejected(tuple(assignments), None)

    successor = Configuration(configuration.space, configuration.facts, stored)
    before = Solution(configuration)
    after = Solution(successor)

    # 2 and 3. each origin of a counted pair the predecessor lacks refuses a change that
    # sets a new value, once; clears alone are never refused (§ 16.16, § 16.18)
    named = set(edit.commit) | set(edit.clear)
    refusals = _newly_false(before, after, named, _sets_new(configuration, edit))
    if refusals:
        return Rejected(tuple(sorted(refusals.values(), key=refusal_key)), after.validity())

    # 4. what moved
    return Accepted(successor, *_moved(before, after), after.validity())


def change(configuration: Configuration, edit: Change) -> Accepted | Rejected:
    """Apply ``edit``: accepted with the design space it publishes and what moved, or
    refused with each refusal. A change of choices alone is assignment (:func:`_choices`).
    A change of facts is two steps, published together or not at all (§ 16.24, § 16.25):
    the receiver's stored choices bound under the new facts (:func:`bind`'s operation;
    a choice the change names is kept where its old value is inside its new domain and
    left out silently where not), then the choices as a change of choices on that design
    space. The first never refuses; a refusal is the second's."""

    index = configuration.index
    for path in (*edit.commit, *edit.clear):
        if path not in index.choices:
            raise ConfigurationError(f"unknown choice {path}")
    for name in (*edit.facts, *edit.unset):
        if name not in index.facts:
            raise ConfigurationError(f"unknown fact {name}")
    facts = dict(configuration.facts)
    for name in edit.unset:
        if name not in index.optional:
            raise ConfigurationError(f"the required fact {name} cannot be unset")
        facts[name] = None  # SZ100 (5): an optional fact unset is None
    facts.update(edit.facts)
    choices = Change(commit=edit.commit, clear=edit.clear)
    if _same_mapping(facts, configuration.facts):
        return _choices(configuration, choices)

    # step 1: the new design space; step 2: the choices on it, judged against its pairs
    named = set(edit.commit) | set(edit.clear)
    instantiated, dropped = _instantiate(
        index, configuration.space, facts, configuration.stored, named
    )
    result = _choices(instantiated, choices)
    if isinstance(result, Rejected):
        return result
    # a pair step 1 added and step 2 removed is not in the published design space
    return _new_facts_report(configuration, result.configuration, dropped)


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
        membership-only domain, or a list refused through an input)."""

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
        """The membership list ranked by the domain's order: known, waiting while an
        order read waits, refused through a refused input, inapplicable when an order read
        does not apply; ``None`` for a membership-only domain."""

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
