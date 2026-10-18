# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The acceptance matrix: every disagreement between the engine and the reference model
expected at this commit, by the milestone that removes it.

Each row names the change that removes it, in one sentence, and its **witness**: the one
test that shows it, marked with :func:`expected` (a strict expected failure raising
:class:`~.compare.Disagreement`, so an engine error or a harness fault still fails). A
milestone is accepted when its rows' witnesses pass (each strict expected failure then
fails) and it removes the rows. A row's **kind** names the disagreement for a walk to
match against, where a walk can meet it.

``python -m core.space.reference_harness`` (from ``tests``) prints the matrix as a markdown
table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import TypeVar

import pytest

from .compare import Disagreement

#: in order: the line as this harness lands, the three lanes, the composition's pieces
MILESTONES = ("base", "lane 0", "lane 1", "lane 2", "E2")


@dataclass(frozen=True)
class Row:
    """One expected disagreement."""

    id: str
    milestone: str
    #: the change that removes it, in one sentence
    change: str
    #: the test showing it, ``module::function``
    witness: str
    #: what a walk can match it by, where a walk can meet it
    kind: str | None = None
    #: a qualification: unclassified, or waiting on a ruling
    note: str = ""


_CASES = "test_reference_cases"

ROWS: tuple[Row, ...] = (
    Row(
        "commit-under-open-selector",
        "lane 0",
        "a choice committed under a selector still open is accepted and kept unchecked, "
        "waiting on the selector, not refused",
        f"{_CASES}::test_a_choice_committed_under_an_open_selector_is_kept_unchecked",
        "commit-waits",
    ),
    Row(
        "facts-open-selector",
        "lane 0",
        "new facts that open the selector a retained choice sits under keep the choice "
        "unchecked, not refused",
        f"{_CASES}::test_new_facts_that_open_a_selector_keep_a_choice_under_it_unchecked",
        "commit-waits",
    ),
    Row(
        "commit-under-refused-selector",
        "lane 0",
        "a change reads the configuration it would publish, so a value inside its domain "
        "under a selector with no possible case is accepted, the configuration refused "
        "through the selector",
        f"{_CASES}::test_a_value_inside_its_domain_under_a_refused_selector_is_accepted",
        "commit-under-refused-selector",
    ),
    Row(
        "readable-under-refused-selector",
        "lane 0",
        "a stored value whose own check holds under a selector with no possible case reads "
        "as its value (readable), not refused through the selector",
        f"{_CASES}::test_a_value_inside_its_domain_under_a_refused_selector_is_readable",
        "state-under-refused-selector",
        "unclassified: no lane names it; lane 0's guard states are likeliest",
    ),
    Row(
        "value-case-trial",
        "lane 1",
        "a value case is tried against the rules whose answer depends on the choice, "
        "through the presence of what they read, so a case that makes one false is not "
        "possible",
        f"{_CASES}::test_a_guards_choice_takes_no_case_that_makes_a_rule_under_it_false",
        "value-case-trial",
    ),
    Row(
        "listed-value-not-member-possible",
        "lane 1",
        "every listed value is checked by the domain's membership, so a listed value it "
        "rejects is not possible",
        f"{_CASES}::test_a_listed_value_the_domain_rejects_is_not_possible",
        "listed-not-member",
    ),
    Row(
        "listed-value-not-member-forced",
        "lane 1",
        "every listed value is checked by the domain's membership, so the one listed value "
        "it accepts is forced",
        f"{_CASES}::test_a_listed_value_the_domain_rejects_leaves_the_other_forced",
        "listed-not-member",
    ),
    Row(
        "empty-list-choice",
        "lane 1",
        "an empty listed domain refuses its choice (no possible value), not left open",
        f"{_CASES}::test_an_empty_list_refuses_its_choice",
        "empty-list",
    ),
    Row(
        "empty-computed-list-choice",
        "lane 1",
        "a domain whose computed list is empty refuses its choice (no possible value), not "
        "left open",
        f"{_CASES}::test_an_empty_computed_list_refuses_its_choice",
        "empty-list",
    ),
    Row(
        "empty-list-validity",
        "lane 1",
        "a choice with an empty list is a false piece of validity: the configuration is refused",
        f"{_CASES}::test_an_empty_list_refuses_the_configuration",
        "empty-list",
    ),
)

BY_ID = {row.id: row for row in ROWS}

#: each row's witnesses, as the test modules mark them on import
USED: dict[str, list[str]] = {}

F = TypeVar("F", bound=Callable[..., object])


def expected(row_id: str) -> Callable[[F], F]:
    """Mark a test as the witness of the row ``row_id``: a strict expected failure raising
    ``Disagreement``, its reason the row's milestone and change."""
    row = BY_ID[row_id]

    def mark(test: F) -> F:
        USED.setdefault(row_id, []).append(f"{_module(test)}::{test.__name__}")
        reason = f"{row.milestone}: {row.change}" + (f" ({row.note})" if row.note else "")
        marked: F = pytest.mark.xfail(strict=True, raises=Disagreement, reason=reason)(test)
        return marked

    return mark


def _module(test: Callable[..., object]) -> str:
    return test.__module__.rsplit(".", 1)[-1]


def problems(modules: Iterable[ModuleType]) -> list[str]:
    """What breaks the matrix's rules: a row without exactly one witness, or with another
    than it names; an expected failure without a row."""
    found: list[str] = []
    for row in ROWS:
        if row.milestone not in MILESTONES[1:]:
            found.append(f"{row.id}: milestone {row.milestone!r} is not a later milestone")
        witnesses = USED.get(row.id, [])
        if witnesses != [row.witness]:
            found.append(f"{row.id}: witnessed by {witnesses}, names {row.witness}")
    witnessed = {witness for witnesses in USED.values() for witness in witnesses}
    for module in modules:
        for name, test in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(test):
                continue
            marks = getattr(test, "pytestmark", [])
            if any(mark.name == "xfail" for mark in marks):
                if f"{_module(test)}::{name}" not in witnessed:
                    found.append(f"{_module(test)}::{name}: an expected failure with no row")
    return found


def markdown() -> str:
    """The matrix as a markdown table, by milestone."""
    lines = [
        "| Row | Milestone | The change that removes it | Witness | Kind |",
        "|---|---|---|---|---|",
    ]
    order = {milestone: position for position, milestone in enumerate(MILESTONES)}
    for row in sorted(ROWS, key=lambda item: (order[item.milestone], item.id)):
        change = row.change + (f" ({row.note})" if row.note else "")
        lines.append(
            f"| {row.id} | {row.milestone} | {change} | `{row.witness}` | {row.kind or ''} |"
        )
    return "\n".join(lines)
