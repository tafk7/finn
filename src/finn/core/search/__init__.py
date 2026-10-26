# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The generic half of a design space exploration: the seam over a composition of named
members (``seam``: the one change, the fold rule, a strategy's run, held choices, and their
records) and what explores through it (``explorers``: the protocol, chaining, ``Ranked`` and
``Hold``). What the seam cannot know of a composition (a choice's policy, its ranking, the
members the fold rule clears and the verdict reads) it asks of one hook its maker supplies
(``Hook``). It imports the engine (``finn.core.space``) and the findings
(``finn.core.reporting``) only."""

from finn.core.search.explorers import (
    Explorer,
    Hold,
    Ranked,
    RankPolicy,
    explore,
    next_choice,
    set_by_policy,
    unheld,
)
from finn.core.search.seam import (
    SEAM_CODES,
    Accepted,
    ByPolicy,
    Choice,
    Composition,
    ExploreError,
    Hook,
    Key,
    Refused,
    Restored,
    Run,
    Seam,
    Trial,
    Values,
    all_findings,
    dotted,
    first_case,
    key_reasons,
    no_valid_value,
    reasons,
    try_cases,
    unset,
)

__all__ = [
    "SEAM_CODES",
    "Accepted",
    "ByPolicy",
    "Choice",
    "Composition",
    "ExploreError",
    "Explorer",
    "Hold",
    "Hook",
    "Key",
    "RankPolicy",
    "Ranked",
    "Refused",
    "Restored",
    "Run",
    "Seam",
    "Trial",
    "Values",
    "all_findings",
    "dotted",
    "explore",
    "first_case",
    "key_reasons",
    "next_choice",
    "no_valid_value",
    "reasons",
    "set_by_policy",
    "try_cases",
    "unheld",
    "unset",
]
