# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reference model's answers: plain, comparable data, never messages.

Every answer names what it is about by path and carries its reasons as structured
records (a kind, the node it is about, the scope that owns it, a code, and the
paths it relates to), so a harness compares kinds, codes and owners rather than
wording. Every tuple of paths or reasons is sorted, so an answer never depends on
the order a description declares its members in.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class Verdict(enum.Enum):
    """A check's truth in three values: unknown can still become true or false."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


def kleene_and(verdicts: Iterable[Verdict]) -> Verdict:
    """Strong three-valued AND: false beats unknown, unknown beats true."""

    seen = set(verdicts)
    if Verdict.FALSE in seen:
        return Verdict.FALSE
    if Verdict.UNKNOWN in seen:
        return Verdict.UNKNOWN
    return Verdict.TRUE


class ReasonKind(enum.Enum):
    """Why something is false or refused."""

    #: a constraint's predicate answered false
    CONSTRAINT = "constraint"
    #: a stored value is not one of its domain's values
    OUTSIDE_DOMAIN = "outside-domain"
    #: a stored choice sits in a scope that is not present
    INACTIVE = "inactive"
    #: an input the node reads is refused
    INPUT_REFUSED = "input-refused"
    #: a view's body refused, with its own code
    VIEW_BODY = "view-body"
    #: a derived value's function refused, with its own code
    DERIVED_BODY = "derived-body"
    #: a view's Space is refused
    SPACE_REFUSED = "space-refused"
    #: an open choice with an exhaustive list of which no case is possible
    NO_POSSIBLE_VALUE = "no-possible-value"
    #: forced or impossible choices whose evidence meets in a false piece
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Reason:
    """One structured reason.

    ``subject`` is the node the reason is about (a constraint, a choice, a view;
    empty for a conflict), ``owner`` the scope that owns it (empty for the root),
    ``code`` a constraint's or a view body's own code, and ``related`` the other
    paths involved: the refused inputs, or a conflict's participants.
    """

    kind: ReasonKind
    subject: str
    owner: str
    code: str = ""
    related: tuple[str, ...] = ()


def _reason_key(reason: Reason) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (reason.kind.value, reason.subject, reason.owner, reason.code, reason.related)


def sorted_reasons(reasons: Iterable[Reason]) -> tuple[Reason, ...]:
    """Distinct reasons in their canonical order."""

    return tuple(sorted(set(reasons), key=_reason_key))


# --- answers of a value: a fact, a choice, a derived value, a view -----------------


@dataclass(frozen=True)
class Known:
    """The value is available."""

    value: Any


@dataclass(frozen=True)
class Unresolved:
    """The value waits: ``waits`` lists the open or unchecked choices, the missing
    facts and the open selectors it waits on."""

    waits: tuple[str, ...]


@dataclass(frozen=True)
class Refused:
    """The value is refused, with why."""

    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class Inapplicable:
    """The value's scope is not present."""


Answer = Known | Unresolved | Refused | Inapplicable


# --- checks and validity -------------------------------------------------------------


class PieceKind(enum.Enum):
    """The pieces whose AND is a Space's validity."""

    CONSTRAINT = "constraint"
    OWN_CHECK = "own-check"
    NO_POSSIBLE_VALUE = "no-possible-value"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Piece:
    """One piece of validity and its answer.

    ``applies`` is false when the piece's scope is not present: it then counts as
    true. ``waits`` is set when the verdict is unknown, ``reasons`` when it is false.
    A conflict's ``related`` lists its participants.
    """

    kind: PieceKind
    owner: str
    subject: str
    verdict: Verdict
    applies: bool = True
    reasons: tuple[Reason, ...] = ()
    waits: tuple[str, ...] = ()
    related: tuple[str, ...] = ()


def piece_key(piece: Piece) -> tuple[str, str, str, tuple[str, ...]]:
    return (piece.kind.value, piece.owner, piece.subject, piece.related)


@dataclass(frozen=True)
class Validity:
    """A Space's validity: valid (``TRUE``), refused (``FALSE``, with the false
    pieces) or undecided (``UNKNOWN``, with the unknown pieces and what they wait on).
    Both lists are filled whatever the verdict."""

    verdict: Verdict
    false: tuple[Piece, ...]
    unknown: tuple[Piece, ...]

    @property
    def waits(self) -> tuple[str, ...]:
        return tuple(sorted({wait for piece in self.unknown for wait in piece.waits}))


@dataclass(frozen=True)
class Check:
    """A stored choice's own check: its value is one of its domain's, and it is present.

    ``verdict`` is ``None`` when the choice is inactive (its scope is not present).
    Membership is evaluated whatever the presence: a value outside its domain is
    false even under an unknown or refused guard; a value inside it is unknown under
    an unknown guard, waiting on the guard, and true under a refused one.

    ``refused_input`` marks a check whose domain reads an input refused other than by
    forcing (a derived value or view refusing on its facts): it cannot conclude, and
    never will without a revision, so ``verdict`` is unknown inside trials and forcing
    (blamed once, at the input) while ``reasons`` say why; published, its piece reads
    refused through that input. Under a refused guard the guard comes first: such a
    check has ``through_guard`` set, no reasons and no waits, and the choice is
    refused through its guard only.
    """

    verdict: Verdict | None
    reasons: tuple[Reason, ...] = ()
    waits: tuple[str, ...] = ()
    refused_input: bool = False
    #: under a refused guard, a check that cannot conclude only through a refused
    #: input: its answer is its guard's (no reasons of its own, nothing in validity)
    through_guard: bool = False


# --- a choice's state ----------------------------------------------------------------

#: one case of a listed domain and why it is not possible
CaseReasons = tuple[Any, tuple[Reason, ...]]


@dataclass(frozen=True)
class Readable:
    """Committed and checked: the stored value is read as the choice's value.
    ``conflict`` lists the other participants of a conflict this committed choice
    takes part in; it keeps its value while its own check holds."""

    value: Any
    conflict: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardRefused:
    """Committed, its own check true, under a refused guard (a selector refused by
    forcing or by its own check): admitted, never unchecked, a change keeping it is
    accepted. Where its guard holds (a trial assuming its case) it reads as its value;
    in the published configuration it reads as refused through its guard, with the
    guard's reasons in ``reasons``. ``conflict`` lists the other participants of a
    conflict it takes part in, as for :class:`Readable`."""

    value: Any
    reasons: tuple[Reason, ...]
    conflict: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputRefused:
    """Committed, its own check refused through an input of its domain that is refused
    on facts (not by forcing): the commit is accepted (only membership refuses a value),
    the choice is kept, never read as a value, and not waiting on anything a commitment
    could settle; published it reads refused through that input, ``reasons`` naming the
    input and its own reasons."""

    value: Any
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class Unchecked:
    """Committed, its own check unknown: kept, read as unresolved, never used as a
    value; ``waits`` is what the check waits on."""

    value: Any
    waits: tuple[str, ...]


@dataclass(frozen=True)
class Forced:
    """Open, with exactly one possible case of an exhaustive list. ``others`` gives
    each other case with why it is not possible. Never stored."""

    value: Any
    others: tuple[CaseReasons, ...]


@dataclass(frozen=True)
class Open:
    """Open. ``possible`` lists its possible cases in domain order, or is ``None``
    when they cannot be listed (a membership-only domain, or a listed domain whose
    list waits on something); ``impossible`` gives the other listed cases with why.
    ``waits`` is set when its presence or its list is unknown."""

    possible: tuple[Any, ...] | None
    impossible: tuple[CaseReasons, ...] = ()
    waits: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True)
class RefusedChoice:
    """Refused. A stored choice whose own check is false (``stored`` true, its value
    in ``value``), or an open choice with no possible case or in a conflict: then
    ``cases`` gives each case's reason and ``peers`` the conflict's other
    participants. A forced value of a conflict's participant is withdrawn."""

    reasons: tuple[Reason, ...]
    stored: bool = False
    value: Any = None
    cases: tuple[CaseReasons, ...] = ()
    peers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Inactive:
    """Its scope is not present. A stored inactive choice keeps its value here and
    refuses any change that would publish it."""

    stored: bool = False
    value: Any = None


ChoiceState = (
    Readable | GuardRefused | InputRefused | Unchecked | Forced | Open | RefusedChoice | Inactive
)


@dataclass(frozen=True)
class Conflict:
    """A conflict component: its participants, the uncommitted ones refused, the
    committed ones kept; ``owner`` is the lowest scope holding them all."""

    owner: str
    participants: tuple[str, ...]
    refused: tuple[str, ...]
    kept: tuple[str, ...]
