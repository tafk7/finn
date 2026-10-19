# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one comparable form both sides' answers are put in: kinds, values, codes and owners,
never messages.

A **code** is ``(code, subject)``: the engine's finding code (or a rule's own code) and the
path of the node it is about. The reference model's reason kinds map to the engine's codes
in one table, ``compare.CODES``. **Blockers** are the paths an open answer waits on.

Some kinds only today's engine produces (``"forced"`` and ``"refused"`` choices, a
``"choice"`` refusal): the model never answers them. They exist so a legacy answer shows as
a disagreement, and go when the engine core lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: a reason: its code and the path of the node it is about
Code = tuple[str, str]

#: the model's five states; ``forced`` and ``refused`` are today's engine's only
StateKind = Literal["set", "waiting", "dormant", "open", "absent", "forced", "refused"]
AnswerKind = Literal["value", "unresolved", "refused", "inapplicable"]
RuleKind = Literal["true", "false", "unknown", "inapplicable"]
VerdictKind = Literal["valid", "refused", "open"]
#: ``choice`` is today's engine's only: a refusal per choice, named or reached
RefusalKind = Literal["assignment", "rule", "choice"]
OutcomeKind = Literal["possible", "impossible"]


@dataclass(frozen=True)
class StateForm:
    """A choice's state. ``value`` for set, waiting, dormant (and a legacy forced value);
    ``blockers`` for waiting, and for open while its presence waits."""

    kind: StateKind
    value: object = None
    blockers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AnswerForm:
    """A value's answer (a fact, a choice, a derived value, a view)."""

    kind: AnswerKind
    value: object = None
    codes: frozenset[Code] = frozenset()
    blockers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RuleForm:
    """One rule's answer: true, false (its codes), unknown (its blockers), or inapplicable
    (its scope is not present)."""

    kind: RuleKind
    codes: frozenset[Code] = frozenset()
    blockers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ValidityForm:
    """A configuration's validity: valid, refused (the false pieces' codes) or open (what
    the unknown pieces wait on). ``false`` is filled whatever the verdict."""

    verdict: VerdictKind
    false: frozenset[Code] = frozenset()
    blockers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RefusalForm:
    """One refusal of a change. ``assignment``: ``subject`` the choice whose value is not of
    its type; ``rule``: ``subject`` the origin (a constraint, a derived value, a view, or the
    choice whose listed domain lists no value), ``standing`` the stored choices in its way;
    ``choice`` (today's engine's only): ``subject`` a choice refused named or reached.
    ``codes`` are the refusal's own."""

    kind: RefusalKind
    subject: str
    codes: frozenset[Code] = frozenset()
    standing: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ChangeForm:
    """A change's result: accepted, with the stored choices it puts dormant or brings into
    play and the open choices it brings into presence, or refused, with each refusal."""

    accepted: bool
    dormant: frozenset[str] = frozenset()
    in_play: frozenset[str] = frozenset()
    opened: frozenset[str] = frozenset()
    refusals: frozenset[RefusalForm] = frozenset()


@dataclass(frozen=True)
class OutcomeForm:
    """What setting one value alone answers: possible, with what it moves (as
    ``ChangeForm``), or impossible, with each refusal."""

    kind: OutcomeKind
    dormant: frozenset[str] = frozenset()
    in_play: frozenset[str] = frozenset()
    opened: frozenset[str] = frozenset()
    refusals: frozenset[RefusalForm] = frozenset()


@dataclass(frozen=True)
class PossibleForm:
    """A choice's possible values: each member value with its outcome, in its membership
    list's order; ``None`` where no list is given (a membership-only domain, a list that
    waits or is refused, and today's engine's choices it does not list)."""

    values: tuple[tuple[object, OutcomeForm], ...] | None


@dataclass(frozen=True)
class DefinitionForm:
    """Whether a description is a definition at all: ``error`` the kind of definition
    error it is (``"cycle"``, ``"domain-reads-choice"``), or ``None`` where it builds and
    answers."""

    error: str | None


Form = StateForm | AnswerForm | RuleForm | ValidityForm | ChangeForm | PossibleForm | DefinitionForm
