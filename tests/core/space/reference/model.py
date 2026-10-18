# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An executable reference model of the Space engine's semantics.

Written from the semantics, not from the engine: it imports nothing of ``finn`` and
nothing outside the standard library. It is slow on purpose. Every answer is
recomputed from scratch for each configuration, by exhaustive enumeration of small
listed domains. One evaluation of one state memoizes its own answers while it runs;
nothing survives from one state to the next, and nothing is reused across
configurations.

The rules it states
===================

**Pieces and validity.** A Space's validity is the strong three-valued AND
(:func:`~.answers.kleene_and`) of its pieces: every constraint of every present scope
at or below it; every stored choice's own check; every open choice with an exhaustive
list and no possible case; every conflict forcing finds. A piece in a scope that is not
present counts as true. A piece under a guard that is unknown answers unknown whatever
its predicate would say (the guard is not combined with the predicate as an
implication). A constraint reading an input that is refused answers false; reading one
that waits, unknown; reading one in a scope that is not present, true (it does not
apply).

**A choice's own check** is membership of its stored value in its domain, AND its
presence. Membership is evaluated whatever the presence, so a value outside its
domain is false even under an unknown or refused guard, and that is the choice's own
reason, never replaced by its guard's. A value inside its domain under an unknown
guard is unknown: the choice is kept **unchecked**, waiting on the guard. A choice
whose scope is not present is inactive.

**Readable.** A stored choice is read as its value only when its own check is true; an
unchecked choice reads as unresolved and is never used as a value, by any rule or by
forcing.

**Possible cases.** A listed case of an open choice is possible unless, with the case
assumed on a copy of the state (a trial), a piece turns false among: the choice's own
check; every constraint whose answer depends on the choice (through its reads, derived
values, views and the presence of what it reads); and, for a selector, the selected
case's validity (the constraints and stored choices' own checks at or below the case).
Unknown counts as possible. Another choice having no possible case is not one of those
pieces.

**Forcing** runs in frozen rounds. Each round trials every open choice that is present
and has an exhaustive list against one state; an open choice with exactly one possible
case is forced, and every value a round forces is installed together for the next
round, until a round forces nothing. A choice whose presence is not known is not
forced. Then one check: every constraint is evaluated under the forced values, and with
every no-case proof, starts a support (the valued choices it read; a no-case proof's
choice with the choices its refused cases read). A support is expanded backwards
through the evidence that forced each forced choice in it, and supports that overlap
are merged into components. A component holding a forced choice or a choice with no
possible case is a **conflict**: those uncommitted participants are refused, each
naming the others, and forced values are withdrawn; forcing then runs again from the
same commitments with the refused ones left out, until no new conflict appears. A
committed participant keeps its value while its own check holds; the configuration
reads refused, naming the conflict. Forcing never searches: it propagates one choice
at a time.

**A change** (commit, replace or clear choices; set or unset facts) builds the
configuration it would publish, forcing included, and type-checks every stored choice's
own check against it: a false check, or a stored choice that would be inactive,
refuses the change (``named`` when the change names the choice, ``reached``
otherwise); an unknown check keeps the choice unchecked. A constraint never refuses a
change; the configuration's validity is reported beside the result.

Readings chosen where the semantics leave room
==============================================

1. *Readability is the least fixed point.* Start with every stored choice unchecked;
   repeat: force from the readable choices, evaluate every stored choice's own check
   against the readable and forced values, and take the ones whose check is true as
   readable; until the readable set stops changing (:func:`_settle`). The other
   reading, a greatest fixed point (start with every stored choice readable and
   remove the ones that fail), would make two choices whose checks read each other
   readable together; the least one keeps them unchecked. One exception follows the
   rule that a committed participant of a conflict keeps its value: a readable
   choice in a conflict stays readable when withdrawing the conflict's forced values
   leaves its own check unknown (its scope's presence hung on a withdrawn forced
   selector); it is demoted only by a false check. Should the iteration
   revisit an earlier readable set (only possible when a conflict withdraws a forced
   value an own check read), the choices on the loop that are not readable in every
   set of it are kept unchecked, waiting on each other.
2. *A change re-checks every stored choice.* The semantics say the named and the
   reached ones. A retained choice whose check reads nothing the change touches has
   the same answer before and after, so the two readings agree on every
   configuration a change could have published; they differ only for a base that
   already holds a stored choice whose own check is false or inactive (no change
   publishes one), which this model refuses as ``reached`` on any change.
3. *Cycles* are the dependency rule written at :mod:`.description`: a constraint
   reading a view of its own Space or of a Space above it, directly or through
   derived values or other views; a view, derived value or guard closing a loop
   through validity; an own check reading a view of its own Space. Reading a child's
   view is not a cycle.
4. *"No possible value"*: an open choice that is present, with an exhaustive list
   (including an empty one) and no possible case, is refused; the piece is the
   choice's own (kind ``no-possible-value``, owned by the choice's scope) and counts
   in validity. It is also a conflict start: alone when its refused cases read no
   other valued choice, else with them. A stored value under an empty list is simply
   outside its domain (its own check false). The other reading would give the empty
   list a piece of its own kind beside the no-case piece.
5. *A refused guard beside an own check*: a selector with no possible case leaves its
   cases' presence unknown (not false). So a stored choice under it is refused for a
   value outside its domain (its own reason), and kept unchecked for a value inside
   it, never refused because of its guard; the selector's no-case piece is the
   configuration's refusal, reported beside. The other reading would admit the inside
   value as readable.
6. Others:

   a. *An unknown guard answers unknown* for every piece under it, not Kleene's
      implication (which would make a true predicate under an unknown guard true).
   b. *A trial on a value choice* checks its own check and the constraints reading
      it, but not other stored choices' own checks that read it; so a forced value can
      make a retained choice's own check false, which refuses the change. A
      selector's trial also checks the stored choices' own checks in the selected
      case (its validity).
   c. *Refused choices in checks and forcing*: a choice refused by forcing (no case,
      conflict) reads as unresolved inside trials, the final check and own checks, so
      a refusal is blamed once and never refuses a change through another choice's
      own check. In the published configuration (validity, views, derived values) it
      reads as refused, and a constraint reading it answers false, naming the input.
   d. *Dependents of withdrawn values* are recomputed: after a conflict, forcing runs
      again without the refused participants, so a value forced only because of a
      withdrawn one is no longer forced.
   e. *Blame is conservative*: supports are the valued choices a false piece's answer
      depends on (through views, that includes everything below the view's Space);
      overlapping supports merge, so a committed choice already refused by a plain
      constraint can be named as a peer when its support meets a conflict's.
   f. *A choice whose presence is unknown* is open (never forced, never refused for
      having no case); its possible cases are those whose own membership is not
      false.
   g. *Inputs of an own check or a guard that are refused or do not apply* make the
      check or the guard unknown, never false: only membership refuses a value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .answers import (
    Answer,
    CaseReasons,
    Check,
    ChoiceState,
    Conflict,
    Forced,
    Inactive,
    Inapplicable,
    Known,
    Open,
    Piece,
    PieceKind,
    Readable,
    Reason,
    ReasonKind,
    Refused,
    RefusedChoice,
    Unchecked,
    Unresolved,
    Validity,
    Verdict,
    kleene_and,
    piece_key,
    sorted_reasons,
)
from .description import Index, Refuse, Space, join


class ConfigurationError(ValueError):
    """A configuration or change naming a choice or fact the description lacks."""


def same(a: Any, b: Any) -> bool:
    """Equal and of the same type, so ``True`` is not ``1``."""

    return type(a) is type(b) and bool(a == b)


def _contains(values: Iterable[Any], value: Any) -> bool:
    return any(same(item, value) for item in values)


def _distinct(values: Iterable[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for value in values:
        if not _contains(out, value):
            out.append(value)
    return tuple(out)


# --- configurations and changes ---------------------------------------------------


@dataclass(frozen=True)
class Configuration:
    """A description, its facts (a missing fact is unknown) and its commitments
    (choice path to stored value)."""

    space: Space
    facts: Mapping[str, Any] = field(default_factory=dict)
    stored: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = self.space.validate()
        for name in self.facts:
            if name not in index.facts:
                raise ConfigurationError(f"unknown fact {name}")
        for path in self.stored:
            if path not in index.choices:
                raise ConfigurationError(f"unknown choice {path}")
        object.__setattr__(self, "facts", dict(self.facts))
        object.__setattr__(self, "stored", dict(self.stored))

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
class ChangeRefusal:
    """A choice that refuses a change: ``kind`` is ``"named"`` when the change names
    it, ``"reached"`` when it is retained and the change only reaches it."""

    choice: str
    kind: str
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class Accepted:
    configuration: Configuration
    unchecked: tuple[str, ...]
    validity: Validity


@dataclass(frozen=True)
class Rejected:
    refusals: tuple[ChangeRefusal, ...]
    validity: Validity


def change(configuration: Configuration, edit: Change) -> Accepted | Rejected:
    """Apply ``edit``: accepted with the configuration it publishes and the choices it
    keeps unchecked, or refused with each refusing choice."""

    index = configuration.space.validate()
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
    successor = Configuration(configuration.space, facts, stored)
    solution = Solution(successor)
    refusals = []
    for path in sorted(stored):
        kind = "named" if path in edit.commit else "reached"
        state = solution.state(path)
        if isinstance(state, RefusedChoice):
            refusals.append(ChangeRefusal(path, kind, state.reasons))
        elif isinstance(state, Inactive):
            reason = Reason(ReasonKind.INACTIVE, path, index.scope_of(path))
            refusals.append(ChangeRefusal(path, kind, (reason,)))
    if refusals:
        return Rejected(tuple(refusals), solution.validity())
    return Accepted(successor, solution.unchecked(), solution.validity())


# --- one state's evaluation ----------------------------------------------------------


@dataclass(frozen=True)
class _Inputs:
    """A node's declared reads: their values when all are known, else what blocks."""

    values: dict[str, Any] | None
    refused: tuple[str, ...] = ()
    waits: frozenset[str] = frozenset()
    inapplicable: tuple[str, ...] = ()


#: a presence: its verdict and what an unknown one waits on
_Presence = tuple[Verdict, frozenset[str]]


class _Evaluation:
    """Every answer under one fixed state: ``values`` (the usable values: readable
    stored choices and forced ones) and ``refused`` (choices refused, with why).

    ``published`` selects how a refused choice reads to constraints, derived values
    and views: as refused (the published configuration) or as unresolved (trials and
    forcing's final check). Own checks and guards always read it as unresolved.
    ``checks`` overrides the stored choices' own checks; ``extra`` adds forcing's
    pieces to validity in the published configuration.
    """

    def __init__(
        self,
        index: Index,
        facts: Mapping[str, Any],
        stored: Mapping[str, Any],
        values: Mapping[str, Any],
        refused: Mapping[str, tuple[Reason, ...]],
        published: bool,
        checks: Mapping[str, Check] | None = None,
        extra: tuple[Piece, ...] = (),
    ) -> None:
        self.index = index
        self.facts = facts
        self.stored = stored
        self.values = values
        self.refused = refused
        self.published = published
        self.checks = checks
        self.extra = extra
        self._memo: dict[tuple[str, str, bool], Any] = {}

    # -- presence

    def presence(self, scope: str) -> _Presence:
        key = ("presence", scope, False)
        if key not in self._memo:
            self._memo[key] = self._presence(scope)
        found: _Presence = self._memo[key]
        return found

    def _presence(self, scope: str) -> _Presence:
        info = self.index.scopes[scope]
        if info.parent is None:
            return Verdict.TRUE, frozenset()
        parts = [self.presence(info.parent)]
        if parts[0][0] is Verdict.FALSE:
            return parts[0]
        if info.selector is not None:
            answer = self.read(info.selector, published=False)
            if isinstance(answer, Known):
                parts.append(
                    (Verdict.TRUE if same(answer.value, info.case) else Verdict.FALSE, frozenset())
                )
            elif isinstance(answer, Unresolved):
                parts.append((Verdict.UNKNOWN, frozenset(answer.waits)))
            elif isinstance(answer, Refused):
                parts.append((Verdict.UNKNOWN, frozenset({info.selector})))
            else:
                parts.append((Verdict.FALSE, frozenset()))
        if info.when is not None:
            inputs = self.inputs(info.when.reads, published=False)
            if inputs.values is not None:
                held = bool(info.when.test(inputs.values))
                parts.append((Verdict.TRUE if held else Verdict.FALSE, frozenset()))
            else:
                blockers = inputs.waits | set(inputs.refused) | set(inputs.inapplicable)
                parts.append((Verdict.UNKNOWN, frozenset(blockers)))
        verdict = kleene_and(part[0] for part in parts)
        if verdict is Verdict.FALSE:
            return Verdict.FALSE, frozenset()
        waits = frozenset().union(*(part[1] for part in parts if part[0] is Verdict.UNKNOWN))
        return verdict, waits

    # -- values

    def inputs(self, reads: tuple[str, ...], published: bool) -> _Inputs:
        values: dict[str, Any] = {}
        refused: list[str] = []
        waits: set[str] = set()
        inapplicable: list[str] = []
        for path in reads:
            answer = self.read(path, published)
            if isinstance(answer, Known):
                values[path] = answer.value
            elif isinstance(answer, Refused):
                refused.append(path)
            elif isinstance(answer, Unresolved):
                waits.update(answer.waits)
            else:
                inapplicable.append(path)
        if refused or waits or inapplicable:
            return _Inputs(
                None, tuple(sorted(refused)), frozenset(waits), tuple(sorted(inapplicable))
            )
        return _Inputs(values)

    def read(self, path: str, published: bool | None = None) -> Answer:
        mode = self.published if published is None else published
        key = ("read", path, mode)
        if key not in self._memo:
            self._memo[key] = self._read(path, mode)
        found: Answer = self._memo[key]
        return found

    def _read(self, path: str, published: bool) -> Answer:
        kind = self.index.kind(path)
        if kind == "fact":
            if path in self.facts:
                return Known(self.facts[path])
            return Unresolved((path,))
        scope = self.index.scope_of(path)
        presence, blockers = self.presence(scope)
        if presence is Verdict.FALSE:
            return Inapplicable()
        if kind == "choice":
            if presence is Verdict.UNKNOWN:
                return Unresolved(tuple(sorted(blockers | {path})))
            if path in self.values:
                return Known(self.values[path])
            if path in self.refused and published:
                return Refused(self.refused[path])
            return Unresolved((path,))
        if presence is Verdict.UNKNOWN:
            return Unresolved(tuple(sorted(blockers)))
        if kind == "derived":
            derived = self.index.derived[path][0]
            inputs = self.inputs(derived.reads, published)
            if inputs.values is None:
                return self._blocked(path, scope, inputs)
            return Known(derived.compute(inputs.values))
        return self._view(path, scope, published)

    @staticmethod
    def _blocked(path: str, scope: str, inputs: _Inputs) -> Answer:
        if inputs.refused:
            reason = Reason(ReasonKind.INPUT_REFUSED, path, scope, related=inputs.refused)
            return Refused((reason,))
        if inputs.waits:
            return Unresolved(tuple(sorted(inputs.waits)))
        return Inapplicable()

    def _view(self, path: str, scope: str, published: bool) -> Answer:
        view = self.index.views[path][0]
        inputs = self.inputs(view.reads, published)
        body: Answer
        if inputs.values is None:
            body = self._blocked(path, scope, inputs)
        else:
            output = view.body(inputs.values)
            if isinstance(output, Refuse):
                body = Refused((Reason(ReasonKind.VIEW_BODY, path, scope, code=output.code),))
            else:
                body = Known(output)
        reasons = list(body.reasons) if isinstance(body, Refused) else []
        if self.validity(scope, published).verdict is Verdict.FALSE:
            reasons.append(Reason(ReasonKind.SPACE_REFUSED, path, scope))
        if reasons:
            return Refused(sorted_reasons(reasons))
        return body

    # -- checks

    def check(self, choice: str, value: Any) -> Check:
        """``choice``'s own check with ``value`` stored."""

        info = self.index.choices[choice]
        domain = info.domain
        member = Verdict.TRUE
        waits: set[str] = set()
        inputs = self.inputs(domain.reads, published=False)
        if inputs.values is None:
            member = Verdict.UNKNOWN
            waits.update(inputs.waits, inputs.refused, inputs.inapplicable)
        else:
            if domain.values is not None:
                held = _contains(domain.values, value)
            elif domain.enumerate is not None:
                held = _contains(domain.enumerate(inputs.values), value)
            else:
                held = True
            if held and domain.member is not None:
                held = bool(domain.member(value, inputs.values))
            member = Verdict.TRUE if held else Verdict.FALSE
        presence, blockers = self.presence(info.scope)
        if presence is Verdict.FALSE:
            return Check(None)
        if member is Verdict.FALSE:
            return Check(Verdict.FALSE, (Reason(ReasonKind.OUTSIDE_DOMAIN, choice, info.scope),))
        if member is Verdict.UNKNOWN or presence is Verdict.UNKNOWN:
            return Check(Verdict.UNKNOWN, waits=tuple(sorted(waits | blockers)))
        return Check(Verdict.TRUE)

    def check_piece(self, choice: str) -> Piece:
        if self.checks is not None and choice in self.checks:
            check = self.checks[choice]
        else:
            check = self.check(choice, self.stored[choice])
        scope = self.index.choices[choice].scope
        if check.verdict is None:
            return Piece(PieceKind.OWN_CHECK, scope, choice, Verdict.TRUE, applies=False)
        return Piece(
            PieceKind.OWN_CHECK, scope, choice, check.verdict, True, check.reasons, check.waits
        )

    def constraint(self, path: str, published: bool | None = None) -> Piece:
        mode = self.published if published is None else published
        key = ("constraint", path, mode)
        if key not in self._memo:
            self._memo[key] = self._constraint(path, mode)
        found: Piece = self._memo[key]
        return found

    def _constraint(self, path: str, published: bool) -> Piece:
        rule, scope = self.index.constraints[path]
        kind = PieceKind.CONSTRAINT
        presence, blockers = self.presence(scope)
        if presence is Verdict.FALSE:
            return Piece(kind, scope, path, Verdict.TRUE, applies=False)
        if presence is Verdict.UNKNOWN:
            return Piece(kind, scope, path, Verdict.UNKNOWN, waits=tuple(sorted(blockers)))
        inputs = self.inputs(rule.reads, published)
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

    # -- validity

    def validity(self, scope: str, published: bool | None = None) -> Validity:
        mode = self.published if published is None else published
        key = ("validity", scope, mode)
        if key not in self._memo:
            self._memo[key] = self._validity(scope, mode)
        found: Validity = self._memo[key]
        return found

    def _validity(self, scope: str, published: bool) -> Validity:
        index = self.index
        pieces = [
            self.constraint(name, published)
            for name, (_, owner) in sorted(index.constraints.items())
            if index.within(owner, scope)
        ]
        pieces.extend(
            self.check_piece(name)
            for name in sorted(self.stored)
            if index.within(index.choices[name].scope, scope)
        )
        if published:
            pieces.extend(piece for piece in self.extra if index.within(piece.owner, scope))
        verdict = kleene_and(piece.verdict for piece in pieces)
        false = tuple(sorted((p for p in pieces if p.verdict is Verdict.FALSE), key=piece_key))
        unknown = tuple(sorted((p for p in pieces if p.verdict is Verdict.UNKNOWN), key=piece_key))
        return Validity(verdict, false, unknown)


# --- trials and forcing -------------------------------------------------------------


@dataclass(frozen=True)
class _Trial:
    """One open choice's cases tried against one state."""

    presence: Verdict
    #: the listed cases in domain order, or None when they cannot be listed
    cases: tuple[Any, ...] | None
    waits: frozenset[str]
    possible: tuple[Any, ...] = ()
    impossible: tuple[CaseReasons, ...] = ()
    #: the valued choices, other than this one, that the refused cases' pieces read
    support: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Derivation:
    """A forced choice's value, or a choice with no possible case, with its evidence."""

    value: Any
    cases: tuple[CaseReasons, ...]
    support: frozenset[str]


@dataclass(frozen=True)
class _State:
    """The inputs every evaluation of one configuration shares."""

    index: Index
    facts: Mapping[str, Any]
    stored: Mapping[str, Any]

    def evaluation(
        self,
        values: Mapping[str, Any],
        refused: Mapping[str, tuple[Reason, ...]],
        published: bool = False,
        checks: Mapping[str, Check] | None = None,
        extra: tuple[Piece, ...] = (),
    ) -> _Evaluation:
        return _Evaluation(
            self.index, self.facts, self.stored, values, refused, published, checks, extra
        )


def _trial(
    state: _State,
    values: Mapping[str, Any],
    refused: Mapping[str, tuple[Reason, ...]],
    choice: str,
) -> _Trial:
    index = state.index
    info = index.choices[choice]
    base = state.evaluation(values, refused)
    presence, blockers = base.presence(info.scope)
    domain = info.domain
    if not domain.listed:
        return _Trial(presence, None, blockers)
    inputs = base.inputs(domain.reads, published=False)
    if inputs.values is None:
        waits = blockers | inputs.waits | set(inputs.refused) | set(inputs.inapplicable)
        return _Trial(presence, None, frozenset(waits))
    if domain.values is not None:
        cases = _distinct(domain.values)
    else:
        assert domain.enumerate is not None
        cases = _distinct(domain.enumerate(inputs.values))
    possible: list[Any] = []
    impossible: list[CaseReasons] = []
    support: set[str] = set()
    for case in cases:
        trial_values = dict(values)
        trial_values[choice] = case
        trial = state.evaluation(trial_values, refused)
        reasons: list[Reason] = []
        reads: set[str] = set()
        subjects: set[str] = set()
        check = trial.check(choice, case)
        if check.verdict is Verdict.FALSE:
            reasons.extend(check.reasons)
            reads |= index.check_reads[choice]
        for name in index.readers[choice]:
            piece = trial.constraint(name)
            if piece.verdict is Verdict.FALSE:
                reasons.extend(piece.reasons)
                reads |= index.constraint_reads[name]
        if info.cases is not None:
            selected = join(choice, case)
            for name in sorted(state.stored):
                if not index.within(index.choices[name].scope, selected):
                    continue
                piece = trial.check_piece(name)
                if piece.verdict is Verdict.FALSE:
                    reasons.extend(piece.reasons)
                    reads |= index.check_reads[name]
                    subjects.add(name)
        if reasons:
            impossible.append((case, sorted_reasons(reasons)))
            support |= subjects | {name for name in reads if name != choice and name in values}
        else:
            possible.append(case)
    return _Trial(presence, cases, blockers, tuple(possible), tuple(impossible), frozenset(support))


def _rounds(
    state: _State,
    readable: Mapping[str, Any],
    refused: Mapping[str, tuple[Reason, ...]],
) -> tuple[dict[str, _Derivation], dict[str, _Derivation], dict[str, Any]]:
    """Frozen rounds to quiescence: the forced choices, the ones with no possible case
    (of the last round), and the values (readable and forced)."""

    values = dict(readable)
    forced: dict[str, _Derivation] = {}
    while True:
        proposals: dict[str, _Derivation] = {}
        no_case: dict[str, _Derivation] = {}
        for choice in state.index.ordered_choices():
            if choice in state.stored or choice in values or choice in refused:
                continue
            trial = _trial(state, values, refused, choice)
            if trial.presence is not Verdict.TRUE or trial.cases is None:
                continue
            if len(trial.possible) == 1:
                proposals[choice] = _Derivation(trial.possible[0], trial.impossible, trial.support)
            elif not trial.possible:
                no_case[choice] = _Derivation(None, trial.impossible, trial.support)
        if not proposals:
            return forced, no_case, values
        for choice in sorted(proposals):
            values[choice] = proposals[choice].value
            forced[choice] = proposals[choice]


def _merge(groups: Iterable[frozenset[str]]) -> list[frozenset[str]]:
    """Overlapping groups merged, in a canonical order."""

    merged: list[set[str]] = []
    for group in groups:
        current = set(group)
        rest = []
        for other in merged:
            if other & current:
                current |= other
            else:
                rest.append(other)
        merged = [*rest, current]
    return sorted((frozenset(group) for group in merged), key=lambda g: tuple(sorted(g)))


@dataclass
class _Forcing:
    values: dict[str, Any]
    forced: dict[str, _Derivation]
    #: uncommitted participants refused by a conflict, with their derivation
    refused: dict[str, _Derivation] = field(default_factory=dict)
    #: of those, the ones with no possible case
    no_case: set[str] = field(default_factory=set)
    components: list[frozenset[str]] = field(default_factory=list)


def _force(state: _State, readable: Mapping[str, Any]) -> _Forcing:
    """Forcing from the readable choices: rounds, the final check, conflicts; again
    without the refused participants until no new conflict appears."""

    index = state.index
    refused: dict[str, _Derivation] = {}
    no_case_refused: set[str] = set()
    components: list[frozenset[str]] = []
    while True:
        blocked: dict[str, tuple[Reason, ...]] = {name: () for name in refused}
        forced, no_case, values = _rounds(state, readable, blocked)
        final = state.evaluation(values, blocked)
        starts: list[frozenset[str]] = []
        for name in sorted(index.constraints):
            if final.constraint(name).verdict is Verdict.FALSE:
                starts.append(frozenset(c for c in index.constraint_reads[name] if c in values))
        for name in sorted(no_case):
            starts.append(frozenset({name}) | no_case[name].support)
        fresh_found = False
        for group in _merge(_expand(start, forced) for start in starts):
            fresh = sorted(c for c in group if c in forced or c in no_case)
            if not fresh:
                continue
            fresh_found = True
            components.append(group)
            for name in fresh:
                if name in no_case:
                    refused[name] = no_case[name]
                    no_case_refused.add(name)
                else:
                    refused[name] = forced[name]
        if not fresh_found:
            return _Forcing(values, forced, refused, no_case_refused, _merge(components))


def _expand(start: frozenset[str], forced: Mapping[str, _Derivation]) -> frozenset[str]:
    """``start`` with the evidence of every forced choice in it, to closure."""

    found = set(start)
    expanded: set[str] = set()
    while True:
        pending = sorted(name for name in found if name in forced and name not in expanded)
        if not pending:
            return frozenset(found)
        for name in pending:
            expanded.add(name)
            found |= forced[name].support


def _keep_participants(
    readable: Mapping[str, Any], forcing: _Forcing, checks: dict[str, Check]
) -> None:
    """A readable choice that takes part in a conflict keeps its value while its own
    check is not false: withdrawing the conflict's forced values may leave its check
    waiting (its presence hung on a withdrawn selector), which does not demote it."""

    participants: set[str] = set()
    for group in forcing.components:
        participants |= group
    for name in readable:
        if name in participants and checks[name].verdict is Verdict.UNKNOWN:
            checks[name] = Check(Verdict.TRUE)


@dataclass
class _Settled:
    readable: dict[str, Any]
    forcing: _Forcing
    checks: dict[str, Check]


def _settle(state: _State) -> _Settled:
    """The least fixed point of readability (decision 1 of the module docstring)."""

    stored = state.stored
    readable: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    while True:
        forcing = _force(state, readable)
        blocked = {name: () for name in forcing.refused}
        evaluation = state.evaluation(forcing.values, blocked)
        checks = {name: evaluation.check(name, stored[name]) for name in sorted(stored)}
        _keep_participants(readable, forcing, checks)
        following = {
            name: stored[name] for name, check in checks.items() if check.verdict is Verdict.TRUE
        }
        if following == readable:
            return _Settled(readable, forcing, checks)
        if following in history:
            loop = [*history[history.index(following) :], readable]
            keep = set.intersection(*(set(item) for item in loop))
            unstable = set.union(*(set(item) for item in loop)) - keep
            readable = {name: stored[name] for name in sorted(keep)}
            forcing = _force(state, readable)
            blocked = {name: () for name in forcing.refused}
            evaluation = state.evaluation(forcing.values, blocked)
            checks = {name: evaluation.check(name, stored[name]) for name in sorted(stored)}
            _keep_participants(readable, forcing, checks)
            for name, check in checks.items():
                if check.verdict is Verdict.TRUE and name not in keep:
                    checks[name] = Check(Verdict.UNKNOWN, waits=tuple(sorted(unstable - {name})))
            return _Settled(readable, forcing, checks)
        history.append(readable)
        readable = following


# --- a configuration's answers -------------------------------------------------------


class Solution:
    """Every answer of one configuration, computed from scratch."""

    def __init__(self, configuration: Configuration) -> None:
        index = configuration.space.validate()
        self.configuration = configuration
        self.index = index
        self._state = _State(index, configuration.facts, configuration.stored)
        settled = _settle(self._state)
        self._settled = settled
        forcing = settled.forcing
        self._peers: dict[str, tuple[str, ...]] = {}
        for group in forcing.components:
            for name in group:
                self._peers[name] = tuple(sorted(group - {name}))
        refused: dict[str, tuple[Reason, ...]] = {}
        for name, check in settled.checks.items():
            if check.verdict is Verdict.FALSE:
                refused[name] = check.reasons
        extra: list[Piece] = []
        for name in sorted(forcing.refused):
            scope = index.choices[name].scope
            reasons: list[Reason] = []
            if name in forcing.no_case:
                reason = Reason(ReasonKind.NO_POSSIBLE_VALUE, name, scope)
                reasons.append(reason)
                extra.append(
                    Piece(
                        PieceKind.NO_POSSIBLE_VALUE, scope, name, Verdict.FALSE, reasons=(reason,)
                    )
                )
            peers = self._peers.get(name, ())
            if peers:
                owner = index.common_scope(tuple(sorted({name, *peers})))
                reasons.append(Reason(ReasonKind.CONFLICT, name, owner, related=peers))
            refused[name] = sorted_reasons(reasons)
        self._conflicts: list[Conflict] = []
        for group in forcing.components:
            participants = tuple(sorted(group))
            refused_here = tuple(name for name in participants if name in forcing.refused)
            if len(participants) < 2 and not any(name in forcing.forced for name in participants):
                continue
            owner = index.common_scope(participants)
            kept = tuple(name for name in participants if name not in forcing.refused)
            self._conflicts.append(Conflict(owner, participants, refused_here, kept))
            reason = Reason(ReasonKind.CONFLICT, "", owner, related=participants)
            extra.append(
                Piece(
                    PieceKind.CONFLICT,
                    owner,
                    "",
                    Verdict.FALSE,
                    reasons=(reason,),
                    related=participants,
                )
            )
        self._values = {
            name: value for name, value in forcing.values.items() if name not in forcing.refused
        }
        self._refused = refused
        self._published = self._state.evaluation(
            self._values, refused, True, settled.checks, tuple(extra)
        )
        self._frozen_refused: dict[str, tuple[Reason, ...]] = {name: () for name in forcing.refused}

    # -- the questions

    def own_check(self, choice: str) -> Check | None:
        """A stored choice's own check; ``None`` for a choice nothing is stored for."""

        if choice not in self.index.choices:
            raise ConfigurationError(f"unknown choice {choice}")
        return self._settled.checks.get(choice)

    def state(self, choice: str) -> ChoiceState:
        index = self.index
        if choice not in index.choices:
            raise ConfigurationError(f"unknown choice {choice}")
        info = index.choices[choice]
        stored = self.configuration.stored
        forcing = self._settled.forcing
        if choice in stored:
            value = stored[choice]
            check = self._settled.checks[choice]
            if check.verdict is None:
                return Inactive(True, value)
            if check.verdict is Verdict.TRUE:
                return Readable(value, self._peers.get(choice, ()))
            if check.verdict is Verdict.UNKNOWN:
                return Unchecked(value, check.waits)
            return RefusedChoice(check.reasons, True, value)
        presence, _ = self._published.presence(info.scope)
        if presence is Verdict.FALSE:
            return Inactive()
        if choice in forcing.refused:
            return RefusedChoice(
                self._refused[choice],
                cases=forcing.refused[choice].cases,
                peers=self._peers.get(choice, ()),
            )
        if choice in forcing.forced:
            derivation = forcing.forced[choice]
            return Forced(derivation.value, derivation.cases)
        trial = _trial(self._state, self._values, self._frozen_refused, choice)
        if trial.cases is None:
            return Open(None, (), tuple(sorted(trial.waits)), info.required)
        return Open(trial.possible, trial.impossible, tuple(sorted(trial.waits)), info.required)

    def states(self) -> dict[str, ChoiceState]:
        return {name: self.state(name) for name in self.index.ordered_choices()}

    def validity(self, scope: str = "") -> Validity:
        if scope not in self.index.scopes:
            raise ConfigurationError(f"unknown scope {scope}")
        return self._published.validity(scope)

    def read(self, path: str) -> Answer:
        """A fact's, a choice's, a derived value's or a view's answer."""

        if self.index.kind(path) is None:
            raise ConfigurationError(f"unreadable {path}")
        return self._published.read(path)

    def view(self, path: str) -> Answer:
        if path not in self.index.views:
            raise ConfigurationError(f"unknown view {path}")
        return self._published.read(path)

    def piece(self, constraint: str) -> Piece:
        """A constraint's piece in the published configuration."""

        return self._published.constraint(constraint)

    def possible(self, choice: str) -> tuple[Any, ...] | None:
        """The possible cases of a choice nothing is stored for: its forced value, its
        open cases (``None`` when they cannot be listed), or none when refused."""

        state = self.state(choice)
        if isinstance(state, Forced):
            return (state.value,)
        if isinstance(state, Open):
            return state.possible
        if isinstance(state, RefusedChoice) and not state.stored:
            return ()
        return None

    def forced(self) -> dict[str, Any]:
        forcing = self._settled.forcing
        return {
            name: forcing.forced[name].value
            for name in sorted(forcing.forced)
            if name not in forcing.refused
        }

    def unchecked(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, check in sorted(self._settled.checks.items())
            if check.verdict is Verdict.UNKNOWN
        )

    def conflicts(self) -> tuple[Conflict, ...]:
        return tuple(self._conflicts)
