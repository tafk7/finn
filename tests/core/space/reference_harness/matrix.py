# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The acceptance matrix: every disagreement between the engine and the reference model
expected at this commit, by the milestone that removes it. The model holds the human's
rulings on the core types (DECISIONS SZ100) before the engine does: T4b reads views out
(SZ100 (3)).

Each row names the change that removes it, in one sentence, and its **witness**: the one
test that shows it, marked with :func:`expected` (a strict expected failure raising
:class:`~.compare.Disagreement`, so an engine error or a harness fault still fails). A
milestone is accepted when its rows' witnesses pass (each strict expected failure then
fails) and it removes the rows. The engine core (the rewrite to the semantics the model
states: assignment, refusal at the origin, dormant choices) was the first, accepted when
it landed with the model: its rows' witnesses are plain tests. The facts-instance milestone
(a change of facts as a new design space, never refused: § 16.24) was the second, accepted
when the engine dropped and reported as the model does. T4a (the model's reading 8: a
refused guard counts once, at the highest refused guard of its chain, and a refused guard
below an unknown one leaves its piece unknown) was the third, accepted when the engine
counted and waited as the model does: its witnesses are plain tests. T4c (an optional fact
left out read as None, given None the one left out: SZ100 (5); no list holding a choice its
facts alone rule out, unless it is stored: SZ113, SZ119 (5)) was the fourth, accepted when
the engine read and pruned as the model does: its witnesses are plain tests.

A later milestone joins ``MILESTONES`` with its rows; while it has rows the walks count
their disagreements, and fail on each once none has.

**By implementation.** A row's expected failure holds for an implementation that has not
reached its milestone. An implementation declares the later milestones it has reached as
``milestones`` (``Reaching``; none where it declares none, as the public API), and
``expected`` marks a witness for the harness's default (``implementation.default``, a
session's ``--space-implementation``): a plain test where the default has reached the row's
milestone.

``python -m core.space.reference_harness`` (from ``tests``) prints the matrix as a markdown
table.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import pytest
from support.space_oracle import Implementation

from .compare import Disagreement
from .implementation import default

#: in order: the line as it is, then each later milestone with rows to remove
MILESTONES: tuple[str, ...] = ("base", "T4b")


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


ROWS: tuple[Row, ...] = (
    Row(
        "view-read-in",
        "T4b",
        "The engine refuses a definition whose constraint, derived value, guard or domain reads "
        "a view, as a definition error (view-read-in)",
        "test_reference_cases::test_a_rule_reading_a_view_is_no_definition",
        "the engine builds the Space and answers",
    ),
    Row(
        "view-read-in-derived",
        "T4b",
        "The engine refuses a definition whose derived value reads a view, as a definition "
        "error (view-read-in)",
        "test_reference_cases::test_a_derived_value_reading_a_view_is_no_definition",
        "the engine builds the Space and answers",
    ),
    Row(
        "view-read-in-guard",
        "T4b",
        "The engine refuses a definition whose guard reads a view, as a definition error "
        "(view-read-in)",
        "test_reference_cases::test_a_guard_reading_a_view_is_no_definition",
        "the engine builds the Space and lists the guarded choice",
    ),
    Row(
        "view-read-in-domain",
        "T4b",
        "The engine refuses a definition whose domain reads a view, as a definition error "
        "(view-read-in)",
        "test_reference_cases::test_a_domain_reading_a_view_is_no_definition",
        "the engine builds the Space where the view reads no choice",
    ),
)

BY_ID = {row.id: row for row in ROWS}


class Reaching(Implementation, Protocol):
    """An implementation declaring the later milestones it has reached."""

    @property
    def milestones(self) -> frozenset[str]: ...


def reached(implementation: Implementation) -> frozenset[str]:
    """The later milestones ``implementation`` declares it has reached (its ``milestones``;
    none where it declares none). A milestone not in ``MILESTONES[1:]`` is an error."""
    found = frozenset(getattr(implementation, "milestones", frozenset()))
    unknown = sorted(found - set(MILESTONES[1:]))
    if unknown:
        raise ValueError(f"{implementation.name}: no later milestones {unknown}")
    return found


#: a test function, returned marked as it was given
F = TypeVar("F", bound=Callable[..., object])


def expected(row_id: str) -> Callable[[F], F]:
    """Mark a test as the witness of the row ``row_id``: a strict expected failure raising
    ``Disagreement``, its reason the row's milestone and change; a plain test where the
    harness's default implementation has reached the row's milestone (``reached``)."""
    row = BY_ID[row_id]

    def mark(test: F) -> F:
        if row.milestone in reached(default()):
            return test
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
