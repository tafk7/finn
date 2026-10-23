# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness comparing the Space engine with its reference model (``core.space.reference``).

The model's package is the model's only: it imports the standard library alone. This
package imports both. :mod:`.engine` is the adapter to the engine core: it builds a model
description as an engine Space and asks it the model's questions, each answered by the
engine's own reading in the model's terms (everything that depends on how the engine is
authored or asked is there); :mod:`.compare` puts the model's answers in the same form
(:mod:`.forms`) and compares them, the engine's side as the harness's default implementation
answers (:mod:`.implementation`: the public API unless a session names another) wherever a
test names none; :mod:`.matrix` records every disagreement a pending
milestone still expects, by the milestone that removes it (none: the engine core's and the
facts-instance milestones are accepted, and the engine agrees with the model); :mod:`.walks`
compares the two over seeded random walks (imported on its own, never by this package). The
engine has one read path, so each test and each walk asks it once.

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
    DroppedForm,
    OutcomeForm,
    PossibleForm,
    RefusalForm,
    RuleForm,
    StateForm,
    ValidityForm,
)
from .matrix import MILESTONES, ROWS, Reaching, Row, expected, markdown, reached

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
    "DroppedForm",
    "NotTranslatable",
    "Obligation",
    "OutcomeForm",
    "Pair",
    "Plain",
    "PossibleForm",
    "Reaching",
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
    "reached",
]
