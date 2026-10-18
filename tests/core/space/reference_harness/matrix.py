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

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
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
    #: a qualification: how the engine answers it now, or what it comes with
    note: str = ""


_CASES = "test_reference_cases"
_RULES = "test_reference_rules"
_SCOPES = "test_reference_scopes"

ROWS: tuple[Row, ...] = (
    Row(
        "outside-domain-forcing",
        "lane 2",
        "a named value whose own check is false is refused with its own reason; forcing never "
        "reads it",
        f"{_CASES}::test_a_value_outside_its_domain_a_cores_rule_reads_is_refused_with_its_own_reason",
        "own-reason-forcing",
        "the engine refuses it as inactive with no finding: the core's trial reads the false own "
        "check and forcing selects the other core",
    ),
    # -- rules between choices (cases 5 to 8)
    Row(
        "pair-first",
        "lane 2",
        "forcing runs in frozen rounds with one final check: two forced choices excluding each "
        "other are both refused, the first naming the second, not both forced",
        f"{_RULES}::test_the_first_of_two_forced_choices_excluding_each_other_is_refused_naming_the_other",
        "conflict",
    ),
    Row(
        "pair-second",
        "lane 2",
        "forcing runs in frozen rounds with one final check: two forced choices excluding each "
        "other are both refused, the second naming the first, not both forced",
        f"{_RULES}::test_the_second_of_two_forced_choices_excluding_each_other_is_refused_naming_the_other",
        "conflict",
    ),
    Row(
        "pair-validity",
        "lane 2",
        "a conflict refuses the configuration as a piece of its own, naming its participants, "
        "in place of the rule the forced values break",
        f"{_RULES}::test_two_forced_choices_excluding_each_other_refuse_the_configuration",
        "conflict",
    ),
    Row(
        "four-participants",
        "lane 2",
        "the four-choice conflict refuses every participant, each naming the other three",
        f"{_RULES}::test_the_four_choice_conflict_refuses_every_participant_naming_the_others",
        "conflict",
    ),
    Row(
        "four-validity",
        "lane 2",
        "the four-choice conflict refuses the configuration as a conflict and a choice with no "
        "possible value, not as the first rule rank-order forcing breaks",
        f"{_RULES}::test_the_four_choice_conflict_refuses_the_configuration",
        "conflict",
    ),
    Row(
        "four-orders",
        "lane 2",
        "the refused set and each participant's peers are the same in every order the choices "
        "and rules are declared",
        f"{_RULES}::test_the_four_choice_conflict_is_the_same_in_every_declaration_order",
        "conflict-order",
    ),
    # -- views, nested scopes, look-ahead (case 9, the composing cores' two)
    Row(
        "inactive-under-refused-selector",
        "lane 2",
        "a choice under an inner selector admitted, under a refused outer selector, on another "
        "case is inactive in the configuration a change would publish: the change is refused, "
        "reaching it as inactive",
        f"{_SCOPES}::test_a_choice_on_an_inner_selectors_other_case_is_inactive_under_a_refused_outer",
        "inactive-under-refused-selector",
        "classified beside the two-cores row (outside-domain-forcing): where the outer case is "
        "assumed, the inner selector's admitted case decides the choice's presence; the "
        "engine reads the choice under a refused guard",
    ),
    Row(
        "case-trial-reads-unchecked",
        "lane 2",
        "a case's trial reads a stored choice of the case, kept unchecked only because its "
        "presence waits, as its value, so a case whose rule refuses that value is not "
        "possible: the selector lists only the other cases",
        f"{_CASES}::test_a_case_whose_rule_refuses_a_kept_choice_of_it_is_not_possible",
        "trial-reads-unchecked",
        "classified beside the two-cores row (outside-domain-forcing): the engine lists the "
        "case as possible (more than the model, never a wrong forced value); with two cases "
        "it forces the other one, as the model does; a walk meets it through a rule of the "
        "root reading the case's choice too",
    ),
    Row(
        "case-trial-reads-waiting",
        "lane 2",
        "a case's trial never reads as its value a stored choice of the case whose own check "
        "waits on more than its presence (a fact not given), so a commit of it under a "
        "two-case selector is accepted and kept unchecked, not refused as inactive",
        f"{_CASES}::test_a_choice_waiting_on_a_fact_under_two_cases_is_kept_unchecked",
        "trial-reads-waiting",
        "beside case-trial-reads-unchecked, outside-domain-forcing and "
        "inactive-under-refused-selector, one fault: a change's trial iteration reaches the "
        "least fixed point from nothing readable and takes no refusal from a pass that read a "
        "candidate before its check could conclude; with three cases the engine lists the "
        "case, as the model does",
    ),
    Row(
        "nested-conflict-choice",
        "lane 2",
        "a committed choice in a conflict keeps its value, read refused through both selectors "
        "the conflict refuses",
        f"{_SCOPES}::test_a_choice_three_deep_in_a_conflict_keeps_its_value_refused_through_its_guards",
        "conflict",
    ),
    Row(
        "nested-conflict-outer",
        "lane 2",
        "a selector forced only because its other case is refused takes part in the conflict "
        "with the choice three deep, refused naming the others",
        f"{_SCOPES}::test_the_outer_selector_in_a_conflict_with_a_choice_three_deep_is_refused",
        "conflict",
    ),
    Row(
        "nested-conflict-inner",
        "lane 2",
        "a selector with no possible case takes part in the conflict with the choice three "
        "deep, refused naming the others",
        f"{_SCOPES}::test_the_inner_selector_in_a_conflict_with_a_choice_three_deep_is_refused",
        "conflict",
    ),
    Row(
        "nested-conflict-validity",
        "lane 2",
        "a conflict with a choice three deep refuses the configuration as a conflict, naming "
        "the forced selector it refuses (the inner selector, under it, adds nothing of its own)",
        f"{_SCOPES}::test_a_conflict_with_a_choice_three_deep_refuses_the_configuration",
        "conflict",
    ),
    # -- a domain reading an input refused by forcing, or one that does not apply
    Row(
        "forcing-refused-input-commit",
        "lane 2",
        "forcing never reads a refused value as one: a choice whose domain input forcing "
        "refuses (a selector with no possible case) is accepted and kept unchecked, waiting on "
        "the input",
        f"{_CASES}::test_a_commit_whose_domain_input_forcing_refuses_is_kept_unchecked",
        note="the engine refuses the commit, naming the choice with no finding of its own",
    ),
    Row(
        "forcing-refused-input-new-facts",
        "lane 2",
        "forcing never reads a refused value as one: new facts that leave a kept choice's "
        "domain input no possible case are accepted, the choice kept unchecked",
        f"{_CASES}::test_new_facts_that_leave_a_kept_choices_domain_input_no_case_keep_it_unchecked",
        note="the engine refuses the change, reaching the kept choice with no finding of its own",
    ),
    Row(
        "forcing-refused-input-state",
        "lane 2",
        "forcing never reads a refused value as one: a kept choice whose domain input forcing "
        "refuses is unchecked, waiting on the input",
        f"{_CASES}::test_a_choice_whose_domain_input_forcing_refuses_waits_on_it",
        note="asked once the commit is accepted; the engine refuses the commit",
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
