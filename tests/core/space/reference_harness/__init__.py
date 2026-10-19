# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness comparing the Space engine with its reference model (``core.space.reference``).

The model's package is the model's only: it imports the standard library alone. This
package imports both. :mod:`.engine` builds a model description as an engine Space and asks
it the model's questions (everything that depends on how the engine is authored or asked
at this commit is there); :mod:`.compare` puts the model's answers in the same form
(:mod:`.forms`) and compares them; :mod:`.matrix` records every disagreement expected at
this commit by the milestone that removes it; :mod:`.walks` compares the two over seeded
random walks (imported on its own, never by this package).

    CASE = Case("a toy", Space(("wide",), Scope("", choices=(Choice("x", listed(1, 2)),))))
    pair = CASE.start(facts={"wide": True})
    pair.check_change(Change(commit={"x": 3}))     # refused: x is not of its type
"""

from .compare import Case, Disagreement, Pair, SetupError, check_definition
from .engine import (
    Admission,
    ByName,
    NotTranslatable,
    Obligation,
    Plain,
    Requirement,
    Spelling,
)
from .forms import (
    AnswerForm,
    ChangeForm,
    DefinitionForm,
    OutcomeForm,
    PossibleForm,
    RefusalForm,
    RuleForm,
    StateForm,
    ValidityForm,
)
from .matrix import MILESTONES, ROWS, Row, expected, markdown

__all__ = [
    "MILESTONES",
    "ROWS",
    "Admission",
    "AnswerForm",
    "ByName",
    "Case",
    "ChangeForm",
    "DefinitionForm",
    "Disagreement",
    "NotTranslatable",
    "Obligation",
    "OutcomeForm",
    "Pair",
    "Plain",
    "PossibleForm",
    "RefusalForm",
    "Requirement",
    "Row",
    "RuleForm",
    "SetupError",
    "Spelling",
    "StateForm",
    "ValidityForm",
    "check_definition",
    "expected",
    "markdown",
]
