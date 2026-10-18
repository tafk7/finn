# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one comparable form both sides' answers are put in: kinds, values, codes and owners,
never messages.

A **code** is ``(code, subject)``: the engine's finding code (or a rule's own code) and the
path of the node it is about. The reference model's reason kinds map to the engine's codes
in one table, ``compare.CODES``. **Blockers** are the paths an undecided answer waits on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: a reason: its code and the path of the node it is about
Code = tuple[str, str]

StateKind = Literal["readable", "unchecked", "forced", "open", "refused", "inactive"]
AnswerKind = Literal["value", "unresolved", "refused", "inapplicable"]
RuleKind = Literal["true", "false", "unknown", "inapplicable"]
VerdictKind = Literal["valid", "refused", "undecided"]


@dataclass(frozen=True)
class StateForm:
    """A choice's state. ``value`` for readable, unchecked and forced; ``possible`` for an
    open choice whose cases can be listed (``None`` when they cannot, or when its presence
    waits); ``codes`` for refused; ``blockers`` for unchecked and for open while waiting."""

    kind: StateKind
    value: object = None
    possible: tuple[object, ...] | None = None
    codes: frozenset[Code] = frozenset()
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
    """A configuration's validity: valid, refused (the false pieces' codes) or undecided
    (what the unknown pieces wait on). ``false`` is filled whatever the verdict."""

    verdict: VerdictKind
    false: frozenset[Code] = frozenset()
    blockers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Refusal:
    """One choice refusing a change: ``named`` when the change names it, ``reached``
    otherwise, with its own codes."""

    choice: str
    kind: Literal["named", "reached"]
    codes: frozenset[Code]


@dataclass(frozen=True)
class ChangeForm:
    """A change's result: accepted, with the stored choices it keeps unchecked, or refused,
    with each refusing choice."""

    accepted: bool
    unchecked: frozenset[str] = frozenset()
    refusals: frozenset[Refusal] = field(default_factory=frozenset)


@dataclass(frozen=True)
class DefinitionForm:
    """Whether a description is a definition at all: ``error`` the kind of definition
    error it is (``"cycle"``), or ``None`` where it builds and answers."""

    error: str | None


Form = StateForm | AnswerForm | RuleForm | ValidityForm | ChangeForm | DefinitionForm
