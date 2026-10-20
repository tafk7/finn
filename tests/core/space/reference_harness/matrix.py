# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The acceptance matrix: every disagreement between the engine and the reference model
expected at this commit, by the milestone that removes it. Its rows are the facts-instance
milestone's: the model takes a change of facts as a new design space, never refused (the
semantics note's § 16.24), and the engine still refuses it.

Each row names the change that removes it, in one sentence, and its **witness**: the one
test that shows it, marked with :func:`expected` (a strict expected failure raising
:class:`~.compare.Disagreement`, so an engine error or a harness fault still fails). A
milestone is accepted when its rows' witnesses pass (each strict expected failure then
fails) and it removes the rows. The engine core (the rewrite to the semantics the model
states: assignment, refusal at the origin, dormant choices) was the first, accepted when
it landed with the model: its rows' witnesses are plain tests.

A later milestone joins ``MILESTONES`` with its rows; while it has rows the walks count
their disagreements, and fail on each once none has.

``python -m core.space.reference_harness`` (from ``tests``) prints the matrix as a markdown
table.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pytest

from .compare import Disagreement

#: in order: the line as it is, then each later milestone with rows to remove
MILESTONES: tuple[str, ...] = ("base", "facts-instance")


@dataclass(frozen=True)
class Row:
    """One expected disagreement."""

    id: str
    milestone: str
    #: the change that removes it, in one sentence: what the milestone's engine must do
    change: str
    #: the test showing it, ``module::function``
    witness: str
    #: a qualification: how the engine answers it now
    note: str = ""


#: what the engine answers today for each row: the change of facts refused
_REFUSED = "today refused"

ROWS: tuple[Row, ...] = (
    Row(
        "facts-drop-present",
        "facts-instance",
        "a change of facts drops each stored value its new facts put outside its domain, and "
        "reports it with its old value, the choice open where present",
        "test_reference_rules::test_a_facts_change_drops_each_value_outside_its_new_domain",
        _REFUSED,
    ),
    Row(
        "facts-drop-dormant",
        "facts-instance",
        "a change of facts drops a dormant value its new facts put outside its domain, and "
        "reports it",
        "test_reference_rules::test_a_facts_change_drops_a_dormant_value_outside_its_new_domain",
        _REFUSED,
    ),
    Row(
        "facts-drop-waiting",
        "facts-instance",
        "a change of facts drops a value waiting on its selector that its new facts put "
        "outside its domain",
        "test_reference_cases::test_a_stored_value_new_facts_put_outside_its_domain_is_dropped",
        _REFUSED,
    ),
    Row(
        "facts-drop-arriving",
        "facts-instance",
        "the change of facts supplying the fact a waiting value lacks drops the value where "
        "it falls outside",
        "test_reference_cases::test_the_fact_a_waiting_choice_lacks_checks_its_type_when_it_comes",
        _REFUSED,
    ),
    Row(
        "facts-drop-input-refused",
        "facts-instance",
        "a change of facts making a stored choice's membership input refuse drops the value, "
        "the domain reading refused through the input",
        "test_reference_cases::test_new_facts_that_refuse_a_stored_choices_membership_input_drop_it",
        _REFUSED,
    ),
    Row(
        "facts-drop-input-absent",
        "facts-instance",
        "a change of facts making a stored choice's membership input absent drops the value, "
        "the domain reading refused (domain-input-absent) and reported",
        "test_reference_cases::"
        "test_new_facts_putting_a_stored_choices_membership_input_out_of_play_drop_it",
        _REFUSED,
    ),
    Row(
        "facts-rule-constraint",
        "facts-instance",
        "a change of facts making a constraint newly false is accepted, reading refused, and "
        "reports the constraint with its standing choices",
        "test_reference_rules::test_new_facts_lacking_a_capability_a_stored_case_needs_are_reported",
        _REFUSED,
    ),
    Row(
        "facts-rule-derived",
        "facts-instance",
        "a change of facts making a derived value refuse under a guard is accepted, reading "
        "refused, and reports the derived value",
        "test_reference_scopes::test_new_facts_refusing_a_guards_input_are_reported_at_the_input",
        _REFUSED,
    ),
)

BY_ID = {row.id: row for row in ROWS}

#: a test function, returned marked as it was given
F = TypeVar("F", bound=Callable[..., object])


def expected(row_id: str) -> Callable[[F], F]:
    """Mark a test as the witness of the row ``row_id``: a strict expected failure raising
    ``Disagreement``, its reason the row's milestone and change."""
    row = BY_ID[row_id]

    def mark(test: F) -> F:
        reason = f"{row.milestone}: {row.change}" + (f" ({row.note})" if row.note else "")
        marked: F = pytest.mark.xfail(strict=True, raises=Disagreement, reason=reason)(test)
        return marked

    return mark


def _witnesses(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    """From one case module's source, never imported: each row id its ``@expected`` tests
    witness, and every test marked as an expected failure in any other way."""
    module = path.stem
    witnessed: dict[str, list[str]] = {}
    unmarked: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            target = call.func if call is not None else decorator
            if isinstance(target, ast.Name) and target.id == "expected" and call is not None:
                argument = call.args[0]
                assert isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                witnessed.setdefault(argument.value, []).append(f"{module}::{node.name}")
            elif "xfail" in ast.unparse(target):
                unmarked.append(f"{module}::{node.name}")
    return witnessed, unmarked


def problems(paths: Iterable[Path]) -> list[str]:
    """What breaks the matrix's rules, read from the case modules' sources (a test module
    never imports another): a row without exactly one witness, or with another than it
    names; an expected failure marked otherwise than by its row."""
    found: list[str] = []
    witnessed: dict[str, list[str]] = {}
    for path in sorted(paths):
        rows, unmarked = _witnesses(path)
        for row_id, tests in rows.items():
            witnessed.setdefault(row_id, []).extend(tests)
        found.extend(f"{test}: an expected failure with no row" for test in unmarked)
    for row_id in sorted(set(witnessed) - set(BY_ID)):
        found.append(f"{row_id}: witnessed by {witnessed[row_id]}, but no such row")
    for row in ROWS:
        if row.milestone not in MILESTONES[1:]:
            found.append(f"{row.id}: milestone {row.milestone!r} is not a later milestone")
        if witnessed.get(row.id, []) != [row.witness]:
            found.append(f"{row.id}: witnessed by {witnessed.get(row.id, [])}, names {row.witness}")
    return found


def markdown() -> str:
    """The matrix as a markdown table, by milestone."""
    lines = [
        "| Row | Milestone | The change that removes it | Witness |",
        "|---|---|---|---|",
    ]
    order = {milestone: position for position, milestone in enumerate(MILESTONES)}
    for row in sorted(ROWS, key=lambda item: (order[item.milestone], item.id)):
        change = row.change + (f" ({row.note})" if row.note else "")
        lines.append(f"| {row.id} | {row.milestone} | {change} | `{row.witness}` |")
    return "\n".join(lines)
