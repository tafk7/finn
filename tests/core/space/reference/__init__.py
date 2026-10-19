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
    change(solution.configuration, Change(commit={"x": 3}))   # Rejected: x outside
"""

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
    EMPTY_DOMAIN_REFUSES,
    Accepted,
    Change,
    Configuration,
    ConfigurationError,
    Presence,
    Rejected,
    Solution,
    change,
    same,
)

__all__ = [
    "EMPTY_DOMAIN_REFUSES",
    "Absent",
    "Accepted",
    "Answer",
    "AssignmentRefusal",
    "Change",
    "ChangeRefusal",
    "Choice",
    "ChoiceState",
    "Configuration",
    "ConfigurationError",
    "Constraint",
    "DefinitionError",
    "Derived",
    "Domain",
    "Dormant",
    "Impossible",
    "Inapplicable",
    "Index",
    "Known",
    "Open",
    "Piece",
    "PieceKind",
    "Possible",
    "Presence",
    "Reads",
    "Reason",
    "ReasonKind",
    "Refuse",
    "Refused",
    "RuleRefusal",
    "Rejected",
    "Scope",
    "Selector",
    "Set",
    "Solution",
    "Space",
    "Unresolved",
    "Validity",
    "Verdict",
    "View",
    "Waiting",
    "When",
    "change",
    "computed",
    "kleene_and",
    "listed",
    "membership",
    "same",
]
