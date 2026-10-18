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

**Guards.** A scope's presence is one of four: present; absent (another case
selected, or its ``when`` false); unknown (its selector open, or a ``when`` input
waiting); refused (its selector refused, by forcing or by its own check, or a ``when``
reading a refused choice). A refused guard gains no case without a revision, so
nothing waits on it.

**Pieces and validity.** A Space's validity is the strong three-valued AND
(:func:`~.answers.kleene_and`) of its pieces: every constraint of every scope at or
below it that is present (absent: true; unknown: unknown, whatever its predicate);
every stored choice's own check; every open choice with an exhaustive list and no
possible case; every conflict forcing finds. A scope under a refused guard adds no
constraint pieces of its own: the guard's own piece refuses, and its stored choices'
own checks still count. A constraint reading an input that is refused answers false;
one that waits, unknown; one in a scope that is absent, true (it does not apply).

**Under a refused guard** a constraint's, a derived value's or a view's body is not
evaluated. Inside trials, forcing's final check and own checks it reads unresolved, so
the refusal is blamed once, at the guard's piece; in the published configuration it
reads refused (input-refused, with the guard's reasons), queried directly or read from
outside. A stored choice there whose own check holds is admitted (never unchecked; a
change keeping it is accepted); it reads as its value only where its guard holds (a
trial assuming its case), and published it reads refused through its guard. An open
choice there reads refused through its guard, and its own pieces (no possible value)
add nothing: it is blamed at the guard.

**A choice's own check** is membership of its stored value in its domain, with its
presence. Under a guard false on known information (another case committed or forced,
a ``when`` false on facts) the key does not exist: the choice is inactive and its own
check is not evaluated; named, it refuses the change as inactive. Otherwise membership
is the type check, evaluated with its guard's cases assumed: a value outside its domain
is false under an unknown or refused guard too, and that is its own reason only. A
value inside its domain is: under an unknown guard, unknown (**unchecked**, waiting on
the guard); under a refused guard, true. Forcing never reads a stored choice whose own
check is false, so a guard never goes false because forcing read the bad value.

**Readable.** A stored choice is read as its value only when its own check is true; an
unchecked choice reads as unresolved and is never used as a value, by any rule or by
forcing, when its own check cannot conclude on its own inputs. One that waits only on
its presence is not unknown in a world that settles that presence: in a trial assuming
a case, or in the world an own check assumes, a stored choice made present there whose
own check holds there reads as its value (so a rule of a selector's case refusing it
makes the case not possible).

**Possible cases.** A listed case of an open choice is possible unless, with the case
assumed on a copy of the state (a trial), a piece turns false among: the choice's own
check; every constraint whose answer depends on the choice (through its reads, derived
values, views and the presence of what it reads); for a selector, the own checks of the
stored choices of the selected case. Unknown counts as possible. Another choice having
no possible case is not one of those pieces.

**The trial's reach** is a switch of the description
(:attr:`~.description.Space.trials_count_retained_checks`). Off (the default), the
above: a forced value may make a retained choice's own check false, and the change is
refused. On, a value case's trial also counts every other stored choice's own check
that depends on the choice (through its domain or its presence), so forcing never picks
a value under which a retained choice's own check is false; and a retained choice's own
check found false at forcing's final check starts a conflict, so a change is never
refused through a value nobody committed.

**Forcing** runs in frozen rounds. Each round trials every open choice that is present
and has an exhaustive list against one state; an open choice with exactly one possible
case is forced, and every value a round forces is installed together for the next
round, until a round forces nothing. A choice whose presence is not known is not
forced. Then one check: every constraint (and, with the trial's reach on, every
stored choice's own check) is evaluated under the forced values, and each false one,
with every no-case proof,
starts a support (the valued choices it read; a no-case proof's choice with the choices
its refused cases read). A support is expanded backwards through the evidence that
forced each forced choice in it, and supports that overlap are merged into components.
A component holding a forced choice is a **conflict**: its uncommitted participants
(forced, or with no possible case) are refused, each naming the others, and forced
values are withdrawn; forcing then runs again from the same commitments with the
refused ones left out, until no new conflict appears. A component without a forced
value has nothing to withdraw: a choice in it with no possible case is a **plain
refusal**, naming each case's reason, with no conflict and no peers. A committed
participant keeps its value while its own check holds; the configuration reads refused,
naming the conflict.
A selector refused by forcing is a refused guard for its cases. Forcing never searches:
it propagates one choice at a time.

**A change** (commit, replace or clear choices; set or unset facts) builds the
configuration it would publish, forcing included, and type-checks every stored choice's
own check against it: a false check, or a stored choice that would be inactive,
refuses the change (``named`` when the change names the choice, ``reached``
otherwise); an unknown check keeps the choice unchecked. A constraint never refuses a
change; the configuration's validity is reported beside the result.

Readings chosen where the semantics leave room
==============================================

1. *Readability is the least fixed point, recomputed each round from nothing
   readable.* A round forces from the readable choices, evaluates every stored
   choice's own check against the readable and forced values, and gives the choices
   whose check is true; the iteration starts from none and stops when a round gives
   back its input (:func:`settle_readable`). Nothing carries from one round to the
   next: a round's intermediate state is no configuration anyone held, and a mark
   taken from a value forced on too little would disagree with a build from scratch.
   The other reading, a greatest fixed point, would make two choices whose checks read
   each other readable together. A round that revisits an earlier set is reported as
   :class:`NonConvergence`, with the choices that move and the sets visited, never
   settled by an added rule (no test or random walk has reached one).
2. *A change re-checks every stored choice.* The semantics say the named and the
   reached ones. A retained choice whose check reads nothing the change touches has
   the same answer before and after, so the two readings agree on every
   configuration a change could have published; they differ only for a base that
   already holds a stored choice whose own check is false or inactive (no change
   publishes one), which this model refuses as ``reached`` on any change.
3. *Cycles* are the dependency rule written at :mod:`.description`: a constraint
   reading a view of its own Space or of a Space above it, directly or through
   derived values or other views; a view, derived value or guard closing a loop
   through validity; an own check reading a view of its own Space, or a cycle among
   own checks. Reading a child's view is not a cycle.
4. *"No possible value"*: an open choice that is present, with an exhaustive list
   (including an empty one) and no possible case, is refused; the piece is the
   choice's own (kind ``no-possible-value``, owned by the choice's scope) and counts
   in validity. It is also a conflict start: alone when its refused cases read no
   other valued choice, else with them. A stored value under an empty list is simply
   outside its domain (its own check false).
5. *Three guard states* (above): under a refused selector a stored choice is refused
   for a value outside its domain (its own reason), and admitted for a value inside
   it, read refused through its guard when published; the selector's own piece is the
   configuration's refusal, reported beside it.
6. Others:

   a. *An unknown guard answers unknown* for every piece under it, not Kleene's
      implication (which would make a true predicate under an unknown guard true).
   b. *With the trial's reach on, a false own check in forcing's final check starts a
      conflict*, as a false constraint does: when its support holds a forced value,
      that value is refused and withdrawn, and the stored choice waits on it
      (unchecked) rather than refusing the change. Without a forced value in its
      support it is a plain false own check, which refuses the change. Only a set of
      values forced together in one round makes one false: a value forced alone in a
      later round sees the earlier ones in its own trial.
   c. *Refused choices in checks and forcing*: a choice refused by forcing (no case,
      conflict) reads as unresolved inside trials, the final check and own checks, so
      a refusal is blamed once and never refuses a change through another choice's
      own check; as a guard it is a refused guard. In the published configuration it
      reads as refused, and a constraint reading it answers false, naming the input.
   d. *Dependents of withdrawn values* are recomputed: after a conflict, forcing runs
      again without the refused participants, so a value forced only because of a
      withdrawn one is no longer forced.
   e. *Blame is conservative*: supports are the valued choices a false piece's answer
      depends on (through views, that includes everything below the view's Space);
      overlapping supports merge, so a committed choice already refused by a plain
      constraint can be named as a peer when its support meets a conflict's.
   f. *A choice whose presence is unknown or refused* is never forced and never
      refused for having no case. Under an unknown guard it is open, its possible cases
      those whose own membership is not false; under a refused guard it reads refused
      through the guard.
   g. *An own check whose domain input is refused, or does not apply.* Only
      membership refuses a value, so neither refuses the commit. An input refused on
      facts (a derived value or a view refusing; not by forcing) gets the refused
      guard's treatment: the choice is kept, never read as a value, and not listed as
      waiting, since no commitment can settle it; inside trials, forcing's final check
      and other own checks its piece is unknown, so the refusal is blamed once, at the
      input; published, its piece is false, refused through the input with the input's
      own reasons, so the configuration reads refused. An input refused by forcing reads
      unresolved there, as in (c); a fact not supplied, or an input that does not
      apply, leaves the check unknown, waiting. A ``when`` guard reading a refused
      choice directly is a refused guard; reading one through a derived value, an
      unknown guard.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .answers import (
    Answer,
    CaseReasons,
    Check,
    ChoiceState,
    Conflict,
    Forced,
    GuardRefused,
    Inactive,
    Inapplicable,
    InputRefused,
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

_T = TypeVar("_T")


class ConfigurationError(ValueError):
    """A configuration or change naming a choice or fact the description lacks."""


class NonConvergence(Exception):
    """Readability's iteration revisited an earlier readable set without settling.

    Reported, never resolved: ``choices`` are the stored choices readable in some sets
    of the loop and not in others, ``visited`` the readable sets of the loop in order.
    """

    def __init__(self, choices: tuple[str, ...], visited: tuple[tuple[str, ...], ...]) -> None:
        super().__init__(f"readability does not settle: {', '.join(choices)} over {visited}")
        self.choices = choices
        self.visited = visited


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


class Guard(enum.Enum):
    """A scope's presence. ``REFUSED``: a guard that can never hold without a revision
    (its selector refused by forcing or by its own check, or a ``when`` reading such a
    choice)."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"
    REFUSED = "refused"


@dataclass(frozen=True)
class _Presence:
    state: Guard
    #: what an unknown presence waits on
    waits: frozenset[str] = frozenset()
    #: the refused guards (selectors or a guard's refused reads) and their reasons
    guards: tuple[str, ...] = ()
    reasons: tuple[Reason, ...] = ()


_PRESENT = _Presence(Guard.PRESENT)
_ABSENT = _Presence(Guard.ABSENT)


class _Evaluation:
    """Every answer under one fixed state: ``values`` (the usable values: readable
    stored choices and forced ones) and ``refused`` (choices refused by forcing, with
    why; in the published configuration also stored choices whose own check is false).

    ``published`` selects how something refused reads to constraints, derived values
    and views: as refused (the published configuration) or as unresolved (trials,
    forcing's final check), so a refusal is blamed once, at its own piece. Own checks
    read it as unresolved always. A guard reads it as a refused guard always.
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

    def refusal(self, choice: str) -> tuple[Reason, ...] | None:
        """Why ``choice`` is refused (by forcing, or by its own check), else ``None``."""

        if choice in self.values:
            return None
        if choice in self.refused:
            return self.refused[choice]
        if choice in self.stored:
            check = self.check(choice, self.stored[choice])
            if check.verdict is Verdict.FALSE or check.refused_input:
                return check.reasons
        return None

    def _presence(self, scope: str) -> _Presence:
        info = self.index.scopes[scope]
        if info.parent is None:
            return _PRESENT
        parts = [self.presence(info.parent)]
        if parts[0].state is Guard.ABSENT:
            return _ABSENT
        if info.selector is not None:
            selector = info.selector
            if selector in self.values:
                held = same(self.values[selector], info.case)
                parts.append(_PRESENT if held else _ABSENT)
            else:
                reasons = self.refusal(selector)
                if reasons is not None:
                    parts.append(_Presence(Guard.REFUSED, guards=(selector,), reasons=reasons))
                else:
                    waits = parts[0].waits | {selector}
                    parts.append(_Presence(Guard.UNKNOWN, frozenset(waits)))
        if info.when is not None:
            inputs = self.inputs(info.when.reads, published=False)
            if inputs.values is not None:
                parts.append(_PRESENT if info.when.test(inputs.values) else _ABSENT)
            else:
                refused = {
                    read: reasons
                    for read in info.when.reads
                    if self.index.kind(read) == "choice"
                    and (reasons := self.refusal(read)) is not None
                }
                if refused:
                    parts.append(
                        _Presence(
                            Guard.REFUSED,
                            guards=tuple(sorted(refused)),
                            reasons=tuple(r for read in sorted(refused) for r in refused[read]),
                        )
                    )
                else:
                    blockers = inputs.waits | set(inputs.refused) | set(inputs.inapplicable)
                    parts.append(_Presence(Guard.UNKNOWN, frozenset(blockers)))
        states = {part.state for part in parts}
        if Guard.ABSENT in states:
            return _ABSENT
        if Guard.REFUSED in states:
            refused_parts = [part for part in parts if part.state is Guard.REFUSED]
            return _Presence(
                Guard.REFUSED,
                guards=tuple(sorted({g for part in refused_parts for g in part.guards})),
                reasons=sorted_reasons(r for part in refused_parts for r in part.reasons),
            )
        if Guard.UNKNOWN in states:
            waits = frozenset().union(*(part.waits for part in parts))
            return _Presence(Guard.UNKNOWN, waits)
        return _PRESENT

    def _guard_refused(self, path: str, scope: str, presence: _Presence) -> Answer:
        """A node under a refused guard: refused when published (with the guard's
        reasons), unresolved inside trials and forcing's check."""

        if self.published:
            reason = Reason(ReasonKind.INPUT_REFUSED, path, scope, related=presence.guards)
            return Refused(sorted_reasons((reason, *presence.reasons)))
        return Unresolved(presence.guards)

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
        presence = self.presence(scope)
        if presence.state is Guard.ABSENT:
            return Inapplicable()
        if kind == "choice":
            if presence.state is Guard.UNKNOWN:
                return Unresolved(tuple(sorted(presence.waits | {path})))
            if presence.state is Guard.REFUSED and path not in self.stored:
                # an open choice under a refused guard is blamed at the guard
                if published:
                    return self._guard_refused(path, scope, presence)
                return Unresolved(tuple(sorted({*presence.guards, path})))
            reasons = self.refusal(path)
            if reasons is not None:
                if published:
                    return Refused(reasons)
                return Unresolved((path,))
            if presence.state is Guard.REFUSED:
                # admitted, but read as its value only where its guard holds
                if published:
                    return self._guard_refused(path, scope, presence)
                return Unresolved(tuple(sorted({*presence.guards, path})))
            if path in self.values:
                return Known(self.values[path])
            return Unresolved((path,))
        if presence.state is Guard.UNKNOWN:
            return Unresolved(tuple(sorted(presence.waits)))
        if presence.state is Guard.REFUSED:
            return (
                self._guard_refused(path, scope, presence)
                if published
                else Unresolved(presence.guards)
            )
        if kind == "derived":
            derived = self.index.derived[path][0]
            inputs = self.inputs(derived.reads, published)
            if inputs.values is None:
                return self._blocked(path, scope, inputs)
            output = derived.compute(inputs.values)
            if isinstance(output, Refuse):
                return Refused((Reason(ReasonKind.DERIVED_BODY, path, scope, code=output.code),))
            return Known(output)
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

        key = ("check", f"{choice}\0{type(value).__name__}\0{value!r}", False)
        if key not in self._memo:
            self._memo[key] = self._check(choice, value)
        found: Check = self._memo[key]
        return found

    def _check(self, choice: str, value: Any) -> Check:
        info = self.index.choices[choice]
        presence = self.presence(info.scope)
        if presence.state is Guard.ABSENT:
            # a guard false on known information: the key does not exist here, and its
            # own check is not evaluated
            return Check(None)
        member, found, through = self.membership(choice, value, promote=True)
        waits = set(found)
        if through:
            # an input refused on facts: no commitment can settle it; unknown inside,
            # refused through the input when published (the commit still accepted)
            return Check(Verdict.UNKNOWN, through, refused_input=True)
        if member is Verdict.FALSE:
            # its own reason only, under an unknown or refused guard too
            return Check(Verdict.FALSE, (Reason(ReasonKind.OUTSIDE_DOMAIN, choice, info.scope),))
        if presence.state is Guard.UNKNOWN:
            waits |= presence.waits
        if member is Verdict.UNKNOWN or presence.state is Guard.UNKNOWN:
            return Check(Verdict.UNKNOWN, waits=tuple(sorted(waits)))
        # present, or under a refused guard: a refused guard never gains a case
        # without a revision, so the choice is checked on its value alone
        return Check(Verdict.TRUE)

    def membership(
        self, choice: str, value: Any, promote: bool
    ) -> tuple[Verdict, frozenset[str], tuple[Reason, ...]]:
        """Whether ``value`` is one of ``choice``'s domain's values, what an unknown
        answer waits on, and, when an input of the domain is refused (other than by
        forcing, which reads unresolved here), why: the input-refused reason with the
        input's own. The type check: evaluated in the world assuming its guard's cases
        when its scope is not present (``promote`` reading there, as their values, the
        stored choices whose presence that world settles)."""

        info = self.index.choices[choice]
        domain = info.domain
        reader: _Evaluation = self
        if domain.reads and self.presence(info.scope).state is not Guard.PRESENT:
            reader = self.world(self._guard_cases(info.scope), promote)
        inputs = reader.inputs(domain.reads, published=False)
        if inputs.values is None:
            if inputs.refused:
                reasons = [
                    Reason(ReasonKind.INPUT_REFUSED, choice, info.scope, related=inputs.refused)
                ]
                for path in inputs.refused:
                    answer = reader.read(path, published=False)
                    if isinstance(answer, Refused):
                        reasons.extend(answer.reasons)
                return Verdict.UNKNOWN, frozenset(), sorted_reasons(reasons)
            waits = inputs.waits | set(inputs.inapplicable)
            return Verdict.UNKNOWN, frozenset(waits), ()
        if domain.values is not None:
            held = _contains(domain.values, value)
        elif domain.enumerate is not None:
            held = _contains(domain.enumerate(inputs.values), value)
        else:
            held = True
        if held and domain.member is not None:
            held = bool(domain.member(value, inputs.values))
        return (Verdict.TRUE if held else Verdict.FALSE), frozenset(), ()

    def world(self, assumed: Mapping[str, Any], promote: bool = True) -> _Evaluation:
        """This state with ``assumed`` values (a trial's case, or a guard's cases).

        With ``promote``, a stored choice that is unchecked here only because its
        presence waits reads as its value in that world when the world makes it present
        and its own check holds there: its membership holds with its guard assumed, so
        nothing but its presence was unknown. A choice whose own check cannot conclude
        on its own inputs is never read so.
        """

        values = dict(self.values)
        values.update(assumed)
        world = _Evaluation(self.index, self.facts, self.stored, values, self.refused, False)
        if not promote:
            return world
        while True:
            added = {}
            for name in sorted(self.stored):
                if name in values or name in self.refused:
                    continue
                scope = self.index.choices[name].scope
                if world.presence(scope).state is not Guard.PRESENT:
                    continue
                stored = self.stored[name]
                if self.membership(name, stored, promote=False)[0] is not Verdict.TRUE:
                    continue
                if world.check(name, stored).verdict is Verdict.TRUE:
                    added[name] = stored
            if not added:
                return world
            values.update(added)
            world = _Evaluation(self.index, self.facts, self.stored, values, self.refused, False)

    def _guard_cases(self, scope: str) -> dict[str, Any]:
        """Each selector above ``scope`` set to the case holding it."""

        cases: dict[str, Any] = {}
        current: str | None = scope
        while current is not None:
            info = self.index.scopes[current]
            if info.selector is not None:
                cases[info.selector] = info.case
            current = info.parent
        return cases

    def check_piece(self, choice: str) -> Piece:
        if self.checks is not None and choice in self.checks:
            check = self.checks[choice]
        else:
            check = self.check(choice, self.stored[choice])
        scope = self.index.choices[choice].scope
        if check.verdict is None or self.presence(scope).state is Guard.ABSENT:
            # an absent scope contributes nothing, its stored values included (one
            # outside its domain is still refused with its own reason by a change)
            return Piece(PieceKind.OWN_CHECK, scope, choice, Verdict.TRUE, applies=False)
        if check.refused_input:
            if self.published:
                return Piece(PieceKind.OWN_CHECK, scope, choice, Verdict.FALSE, True, check.reasons)
            refused = tuple(r for reason in check.reasons for r in reason.related)
            return Piece(
                PieceKind.OWN_CHECK, scope, choice, Verdict.UNKNOWN, waits=tuple(sorted(refused))
            )
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
        presence = self.presence(scope)
        if presence.state is Guard.ABSENT:
            return Piece(kind, scope, path, Verdict.TRUE, applies=False)
        if presence.state is Guard.UNKNOWN:
            return Piece(kind, scope, path, Verdict.UNKNOWN, waits=tuple(sorted(presence.waits)))
        if presence.state is Guard.REFUSED:
            # its body is not evaluated; blamed at the guard's own piece
            answer = self._guard_refused(path, scope, presence)
            if isinstance(answer, Refused):
                return Piece(kind, scope, path, Verdict.FALSE, reasons=answer.reasons)
            return Piece(kind, scope, path, Verdict.UNKNOWN, waits=presence.guards)
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
        # a scope under a refused guard adds no constraint pieces of its own: the
        # guard's piece refuses, and its stored choices' own checks still count
        pieces = [
            self.constraint(name, published)
            for name, (_, owner) in sorted(index.constraints.items())
            if index.within(owner, scope) and self.presence(owner).state is not Guard.REFUSED
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

    presence: Guard
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
    presence = base.presence(info.scope)
    blockers = presence.waits | set(presence.guards)
    domain = info.domain
    if not domain.listed:
        return _Trial(presence.state, None, blockers)
    inputs = base.inputs(domain.reads, published=False)
    if inputs.values is None:
        waits = blockers | inputs.waits | set(inputs.refused) | set(inputs.inapplicable)
        return _Trial(presence.state, None, frozenset(waits))
    if domain.values is not None:
        cases = _distinct(domain.values)
    else:
        assert domain.enumerate is not None
        cases = _distinct(domain.enumerate(inputs.values))
    reach = index.space.trials_count_retained_checks
    false_alone: dict[str, bool] = {}

    def already_false(name: str) -> bool:
        if name not in false_alone:
            check = base.check(name, state.stored[name])
            false_alone[name] = check.verdict is Verdict.FALSE
        return false_alone[name]

    possible: list[Any] = []
    impossible: list[CaseReasons] = []
    support: set[str] = set()
    for case in cases:
        trial = base.world({choice: case})
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
        # stored choices' own checks: on the switch, every one that reads this choice
        # (through its domain or its presence); off, a selector's selected case's. A
        # stored choice whose own check is already false is never read by forcing.
        for name in sorted(state.stored):
            if name == choice:
                continue
            if reach:
                counted = choice in index.check_reads[name]
            else:
                counted = info.cases is not None and index.within(
                    index.choices[name].scope, join(choice, case)
                )
            if counted and not already_false(name):
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
    return _Trial(
        presence.state, cases, blockers, tuple(possible), tuple(impossible), frozenset(support)
    )


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
            if trial.presence is not Guard.PRESENT or trial.cases is None:
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
        for name in sorted(state.stored) if index.space.trials_count_retained_checks else ():
            if final.check_piece(name).verdict is Verdict.FALSE:
                reads = index.check_reads[name]
                starts.append(frozenset({name} | {c for c in reads if c in values}))
        for name in sorted(no_case):
            starts.append(frozenset({name}) | no_case[name].support)
        fresh_found = False
        for group in _merge(_expand(start, forced) for start in starts):
            fresh = sorted(c for c in group if c in forced or c in no_case)
            if not fresh:
                continue
            fresh_found = True
            if not any(c in forced for c in group):
                # no forced value to withdraw: a plain refusal, no conflict, no peers
                for name in fresh:
                    refused[name] = no_case[name]
                    no_case_refused.add(name)
                continue
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


@dataclass
class _Settled:
    readable: dict[str, Any]
    forcing: _Forcing
    checks: dict[str, Check]


def settle_readable(round_: Callable[[frozenset[str]], tuple[frozenset[str], _T]]) -> _T:
    """Readability's least fixed point: from nothing readable, apply ``round_`` (which
    gives, from the readable set, the next one and what that round found) until a round
    reproduces its input, and return what that round found. A round revisiting an
    earlier set raises :class:`NonConvergence`."""

    readable: frozenset[str] = frozenset()
    history: list[frozenset[str]] = []
    while True:
        following, found = round_(readable)
        if following == readable:
            return found
        if following in history:
            loop = [*history[history.index(following) :], readable]
            moving = frozenset().union(*loop) - frozenset.intersection(*loop)
            raise NonConvergence(tuple(sorted(moving)), tuple(tuple(sorted(item)) for item in loop))
        history.append(readable)
        readable = following


def _settle(state: _State) -> _Settled:
    """The least fixed point of readability (decision 1 of the module docstring)."""

    stored = state.stored

    def round_(names: frozenset[str]) -> tuple[frozenset[str], _Settled]:
        readable = {name: stored[name] for name in sorted(names)}
        forcing = _force(state, readable)
        blocked = {name: () for name in forcing.refused}
        evaluation = state.evaluation(forcing.values, blocked)
        checks = {name: evaluation.check(name, stored[name]) for name in sorted(stored)}
        following = frozenset(
            name for name, check in checks.items() if check.verdict is Verdict.TRUE
        )
        return following, _Settled(readable, forcing, checks)

    return settle_readable(round_)


# --- a configuration's answers -------------------------------------------------------


class Solution:
    """Every answer of one configuration, computed from scratch."""

    def __init__(self, configuration: Configuration) -> None:
        index = configuration.index
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
        values = {name: v for name, v in forcing.values.items() if name not in forcing.refused}
        guards = self._state.evaluation(values, refused, True, settled.checks)
        for name in sorted(forcing.refused):
            scope = index.choices[name].scope
            reasons: list[Reason] = []
            under_refused = guards.presence(scope).state is Guard.REFUSED
            if name in forcing.no_case and not under_refused:
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
        self._values = values
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
                presence = self._published.presence(info.scope)
                if presence.state is Guard.REFUSED:
                    answer = self._published.read(choice)
                    assert isinstance(answer, Refused)
                    return GuardRefused(value, answer.reasons, self._peers.get(choice, ()))
                return Readable(value, self._peers.get(choice, ()))
            if check.refused_input:
                return InputRefused(value, check.reasons)
            if check.verdict is Verdict.UNKNOWN:
                return Unchecked(value, check.waits)
            return RefusedChoice(check.reasons, True, value)
        presence = self._published.presence(info.scope)
        if presence.state is Guard.ABSENT:
            return Inactive()
        if presence.state is Guard.REFUSED:
            # blamed at its guard: refused through it, its own pieces adding nothing
            answer = self._published.read(choice)
            assert isinstance(answer, Refused)
            return RefusedChoice(answer.reasons)
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
            if check.verdict is Verdict.UNKNOWN and not check.refused_input
        )

    def conflicts(self) -> tuple[Conflict, ...]:
        return tuple(self._conflicts)
