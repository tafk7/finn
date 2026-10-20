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
    """A piece's truth in three values: unknown can still become true or false."""

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

    #: a constraint's predicate answered false on its own values
    CONSTRAINT = "constraint"
    #: a value is not one of its domain's values
    OUTSIDE_DOMAIN = "outside-domain"
    #: a present choice's listed domain lists no value
    EMPTY_DOMAIN = "empty-domain"
    #: an input the node reads is refused
    INPUT_REFUSED = "input-refused"
    #: a present choice's membership input does not apply: a guard over facts beyond the
    #: choice's own is false (§ 16.17); ``related`` names the inputs
    INPUT_ABSENT = "input-absent"
    #: a view's body refused, with its own code
    VIEW_BODY = "view-body"
    #: a derived value's function refused, with its own code
    DERIVED_BODY = "derived-body"
    #: a view's Space is refused
    SPACE_REFUSED = "space-refused"


@dataclass(frozen=True)
class Reason:
    """One structured reason.

    ``subject`` is the node the reason is about (a constraint, a choice, a derived
    value, a view, a guarded scope), ``owner`` the scope that owns it (empty for the
    root), ``code`` a constraint's or a body's own code, and ``related`` the other
    paths involved: the refused inputs.
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
    """The value waits: ``waits`` lists the open choices, the missing facts and the
    open selectors (or a guard's waiting reads) it waits on."""

    waits: tuple[str, ...]


@dataclass(frozen=True)
class Refused:
    """The value is refused, with why."""

    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class Inapplicable:
    """The value's scope is absent."""


Answer = Known | Unresolved | Refused | Inapplicable


# --- validity ------------------------------------------------------------------------


class PieceKind(enum.Enum):
    """The pieces whose AND is a Space's validity."""

    #: a constraint of a present scope
    CONSTRAINT = "constraint"
    #: a present choice's listed domain, false when it lists no value
    EMPTY_DOMAIN = "empty-domain"
    #: a guard refused through a refused input, owned by the guarded scope
    GUARD = "guard"


@dataclass(frozen=True)
class Piece:
    """One piece of validity and its answer.

    ``applies`` is false when the piece's scope is absent, or not in play under a
    refused guard: it then counts as true. ``waits`` is set when the verdict is
    unknown, ``reasons`` when it is false.
    """

    kind: PieceKind
    owner: str
    subject: str
    verdict: Verdict
    applies: bool = True
    reasons: tuple[Reason, ...] = ()
    waits: tuple[str, ...] = ()


def piece_key(piece: Piece) -> tuple[str, str, str]:
    return (piece.kind.value, piece.owner, piece.subject)


@dataclass(frozen=True)
class Validity:
    """A Space's validity: valid (``TRUE``), refused (``FALSE``, with the false
    pieces) or open (``UNKNOWN``, with the unknown pieces and what they wait on).
    Both lists are filled whatever the verdict."""

    verdict: Verdict
    false: tuple[Piece, ...]
    unknown: tuple[Piece, ...]

    @property
    def waits(self) -> tuple[str, ...]:
        return tuple(sorted({wait for piece in self.unknown for wait in piece.waits}))


# --- a choice's state ----------------------------------------------------------------


@dataclass(frozen=True)
class Set:
    """Stored and in play: read as its value."""

    value: Any


@dataclass(frozen=True)
class Waiting:
    """Stored, never read as a value: its presence is unknown, or its membership waits
    on a missing fact. ``waits`` is what it waits on."""

    value: Any
    waits: tuple[str, ...]


@dataclass(frozen=True)
class Dormant:
    """Stored, out of play: its scope is absent, or under a refused guard. Kept, and
    back in play when its scope is present again."""

    value: Any


@dataclass(frozen=True)
class Open:
    """Unset, in a scope that is present or whose presence is unknown. ``waits`` is
    what its presence waits on, empty when present."""

    waits: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True)
class Absent:
    """Unset, its scope absent or under a refused guard: not in play."""


ChoiceState = Set | Waiting | Dormant | Open | Absent


# --- what a change, or a trial of one value, answers ---------------------------------


@dataclass(frozen=True)
class AssignmentRefusal:
    """A value the change names that is not of its choice's type: outside its domain, or
    its domain's membership reads an input that refuses or does not apply. A stored value
    the change does not name refuses nothing: under new facts it is dropped
    (:class:`Dropped`, § 16.24)."""

    choice: str
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class Dropped:
    """A stored value new facts leave out (§ 16.24): not of its choice's type under them,
    outside its domain, or its domain's input refusing or not applying. ``value`` is the
    old value, ``reasons`` the type check's, as an :class:`AssignmentRefusal` would carry
    them. The choice is then open where it is present, and absent where it is not."""

    choice: str
    value: Any
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class RuleRefusal:
    """An origin the change makes a piece newly false through (§ 16.15). ``origin`` is
    the node where the failure starts; ``kind`` is ``"constraint"``, ``"derived"``, ``"view"`` or
    ``"empty-domain"`` (``origin`` then the choice whose listed domain lists no
    value); ``reasons`` its own reason; ``standing`` the stored choices standing in the
    way, which the caller sets in the same change or clears first.

    An accepted change of facts reports the same records for the rules its facts make
    newly false (``Accepted.broken``, § 16.24): information, not a refusal; its standing
    choices are then read in the published design space, none left out for being named."""

    origin: str
    kind: str
    reasons: tuple[Reason, ...]
    standing: tuple[str, ...]


ChangeRefusal = AssignmentRefusal | RuleRefusal


def refusal_key(refusal: ChangeRefusal) -> tuple[int, str]:
    if isinstance(refusal, AssignmentRefusal):
        return (0, refusal.choice)
    return (1, refusal.origin)


@dataclass(frozen=True)
class Possible:
    """A value nothing refuses, with the choices setting it would put dormant, bring
    into play and open."""

    dormant: tuple[str, ...] = ()
    in_play: tuple[str, ...] = ()
    opened: tuple[str, ...] = ()


@dataclass(frozen=True)
class Impossible:
    """A value whose setting is refused, with each refusal."""

    refusals: tuple[ChangeRefusal, ...]
