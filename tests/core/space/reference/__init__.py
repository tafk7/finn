# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A reference model of the Space engine's semantics, independent of the engine.

An oracle for the engine's tests: it imports nothing of ``finn`` and only the standard
library (its row in ``tests/support/layering.py`` says so). Describe a Space as data
(:mod:`.description`), give a :class:`Configuration` its facts and commitments, and
ask its :class:`Solution`; apply a :class:`Change` with :func:`change`. The rules it
states, and the readings it chose where the semantics leave room, are in
:mod:`.model`'s docstring.

    space = Space(("wide",), Scope("", choices=(Choice("x", listed(1, 2)),)))
    solution = Configuration(space, facts={"wide": True}, stored={"x": 1}).solve()
    solution.validity().verdict      # Verdict.TRUE
    change(solution.configuration, Change(commit={"x": 3}))   # Rejected: x, named
"""

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
)
from .description import (
    Choice,
    Constraint,
    DefinitionError,
    Derived,
    Domain,
    Index,
    Reads,
    Refuse,
    Scope,
    Selector,
    Space,
    View,
    When,
    computed,
    listed,
    membership,
)
from .model import (
    Accepted,
    Change,
    ChangeRefusal,
    Configuration,
    ConfigurationError,
    Rejected,
    Solution,
    change,
)

__all__ = [
    "Accepted",
    "Answer",
    "CaseReasons",
    "Change",
    "ChangeRefusal",
    "Check",
    "Choice",
    "ChoiceState",
    "Configuration",
    "ConfigurationError",
    "Conflict",
    "Constraint",
    "DefinitionError",
    "Derived",
    "Domain",
    "Forced",
    "GuardRefused",
    "Inactive",
    "Inapplicable",
    "Index",
    "Known",
    "Open",
    "Piece",
    "PieceKind",
    "Readable",
    "Reads",
    "Reason",
    "ReasonKind",
    "Refuse",
    "Refused",
    "RefusedChoice",
    "Rejected",
    "Scope",
    "Selector",
    "Solution",
    "Space",
    "Unchecked",
    "Unresolved",
    "Validity",
    "Verdict",
    "View",
    "When",
    "change",
    "computed",
    "kleene_and",
    "listed",
    "membership",
]
