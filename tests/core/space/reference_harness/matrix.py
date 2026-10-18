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
_RULES = "test_reference_rules"
_SCOPES = "test_reference_scopes"

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
        "commit-waits-on-fact",
        "lane 0",
        "a choice committed while its domain waits on a fact not given yet is accepted and kept "
        "unchecked, waiting on the fact, not refused",
        f"{_CASES}::test_a_choice_committed_while_its_domain_waits_on_a_fact_is_kept_unchecked",
        "commit-waits",
    ),
    Row(
        "outside-domain-under-open-selector",
        "lane 0",
        "a value outside its domain under a selector still open is refused at its own commit "
        "with its own membership finding, not through the selector's open state",
        f"{_CASES}::test_a_value_outside_its_domain_under_an_open_selector_is_refused_at_commit",
        "own-reason",
    ),
    Row(
        "outside-domain-forcing",
        "lane 2",
        "a named value whose own check is false is refused with its own reason; forcing never "
        "reads it",
        f"{_CASES}::test_a_value_outside_its_domain_a_cores_rule_reads_is_refused_with_its_own_reason",
        "own-reason-forcing",
        "lane 0 refuses it as inactive with no finding: the core's trial reads the false own "
        "check and forcing selects the other core",
    ),
    Row(
        "outside-domain-under-false-guard",
        "lane 2",
        "a value outside its domain under a guard that is false (another case selected, or a "
        "guard's test false) is refused with its own reason only, not as inactive",
        f"{_CASES}::test_a_value_outside_its_domain_under_a_false_guard_is_refused_with_its_own_reason",
        "own-reason-false-guard",
        "unclassified: pending a ruling on whether a false guard leaves a value outside its "
        "domain inactive, as the engine reads it, or refused by its own check, as the model "
        "reads it; no lane names it, and the forcing row's fix would bring it",
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
    # -- rules between choices (cases 5 to 8)
    Row(
        "linked-possible",
        "lane 1",
        "a value case is tried against the constraints that read the choice, so a case a "
        "linking rule makes false is not possible",
        f"{_RULES}::test_a_linking_rule_leaves_the_other_choice_the_cases_it_holds_for",
        "value-case-trial",
    ),
    Row(
        "linked-forced",
        "lane 1",
        "a value case is tried against the constraints that read the choice, so the one case "
        "a linking rule leaves is forced",
        f"{_RULES}::test_a_linking_rule_forces_the_open_choice",
        "value-case-trial",
    ),
    Row(
        "linked-domain-commit-first",
        "lane 1",
        "a linking rule is a constraint, not a domain reading another choice: a commit before "
        "the choice it reads is accepted",
        f"{_RULES}::test_a_linking_rule_in_a_domain_takes_a_commit_before_the_choice_it_reads",
        "linking-in-domain",
    ),
    Row(
        "linked-domain-commit",
        "lane 1",
        "a linking rule is a constraint, not a domain reading another choice: a commit it makes "
        "false is accepted, the configuration refused naming the rule",
        f"{_RULES}::test_a_linking_rule_in_a_domain_never_refuses_a_commit",
        "linking-in-domain",
    ),
    Row(
        "linked-domain-forced",
        "lane 1",
        "a linking rule is a constraint, not a domain reading another choice: the one case it "
        "leaves is forced",
        f"{_RULES}::test_a_linking_rule_in_a_domain_forces_the_open_choice",
        "linking-in-domain",
    ),
    Row(
        "requirement-unknown-fact",
        "lane 1",
        "an unknown fact blocks only the cases a constraint reads it for, so a choice whose "
        "requirement waits on a fact lists its cases",
        f"{_RULES}::test_a_case_needing_an_unknown_capability_is_possible",
        "unknown-fact",
    ),
    Row(
        "requirement-commit",
        "lane 1",
        "a requirement is a constraint: committing a case it refuses is accepted, not refused "
        "at commit",
        f"{_RULES}::test_committing_a_case_needing_a_missing_capability_is_accepted",
        "requirement-at-commit",
    ),
    Row(
        "requirement-validity",
        "lane 1",
        "a requirement is a constraint: a committed case it refuses leaves the configuration "
        "refused, naming it",
        f"{_RULES}::test_committing_a_case_needing_a_missing_capability_refuses_the_configuration",
        "requirement-at-commit",
    ),
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
        "pair-committed-peer",
        "lane 1",
        "a value case is tried against the constraints that read the choice, so a choice whose "
        "one value a rule with a committed choice refuses has no possible value (a plain "
        "refusal: no forced value, no conflict), not forced to the value the rule refuses",
        f"{_RULES}::test_the_other_choice_of_a_rule_with_a_committed_one_has_no_possible_value",
        "value-case-trial",
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
    Row(
        "retained-check-forced-core",
        "lane 0",
        "a change checks every retained choice against the configuration it would publish, "
        "never an admission its base made: a retained value the successor's forced core puts "
        "outside its domain refuses the change, as a cold build does",
        f"{_RULES}::test_a_retained_choice_checked_against_the_successors_forced_core",
        "stale-forced",
    ),
    Row(
        "stale-forced-core",
        "lane 0",
        "a change checks retained and named choices against the configuration it would "
        "publish, never a value its base forced: accepted where only the base's forced core "
        "refused it",
        f"{_RULES}::test_a_change_only_the_bases_forced_core_would_refuse_is_accepted",
        "stale-forced",
    ),
    # -- views, nested scopes, look-ahead (case 9, the composing cores' two)
    Row(
        "view-of-refused-space",
        "lane 1",
        "a view refuses when its Space is refused, every view, not only one with obligations",
        f"{_SCOPES}::test_a_view_refuses_when_its_space_is_refused",
        "view-of-refused-space",
    ),
    Row(
        "cycle-storage-view",
        "lane 1",
        "a view refuses when its Space is refused, so a constraint on a storage view of its "
        "own Space is a dependency cycle the engine reports",
        f"{_SCOPES}::test_a_constraint_on_a_storage_view_of_its_own_space_is_a_cycle",
        "view-cycle",
    ),
    Row(
        "cycle-by-name",
        "lane 1",
        "a view refuses when its Space is refused, so a constraint reading views of its own "
        "Space by name is a dependency cycle the engine reports",
        f"{_SCOPES}::test_a_constraint_reading_views_of_its_own_space_by_name_is_a_cycle",
        "view-cycle",
    ),
    Row(
        "cycle-own-stages",
        "lane 1",
        "a view refuses when its Space is refused, so a parent's constraint reading its "
        "children's stages through its own view is a dependency cycle the engine reports",
        f"{_SCOPES}::test_a_parent_reading_its_childrens_stages_through_its_own_view_is_a_cycle",
        "view-cycle",
    ),
    Row(
        "nested-outer-forced",
        "lane 1",
        "a selector's case is tried against the constraints reading the selector, so a "
        "parent's rule refusing one kernel forces the other",
        f"{_SCOPES}::test_forcing_selects_the_outer_kernel",
        "selector-case-trial",
    ),
    Row(
        "nested-inner-forced",
        "lane 1",
        "a selector's case is tried against the constraints reading the selector, so the core "
        "inside the forced kernel is forced too",
        f"{_SCOPES}::test_forcing_selects_the_core_inside_the_forced_kernel",
        "selector-case-trial",
    ),
    Row(
        "nested-possible",
        "lane 1",
        "a selector's case is tried against the constraints reading the selector, so a choice "
        "three deep under the forced selectors lists its possible cases",
        f"{_SCOPES}::test_a_choice_three_deep_under_forced_selectors_lists_its_possible_cases",
        "selector-case-trial",
    ),
    Row(
        "nested-commit-forced",
        "lane 1",
        "a selector's case is tried against the constraints reading the selector, so a commit "
        "three deep under the forced selectors is accepted, checked",
        f"{_SCOPES}::test_a_commit_three_deep_under_forced_selectors_is_accepted",
        "selector-case-trial",
        "lane 0 accepts it kept unchecked; lane 1 forces the selectors",
    ),
    Row(
        "nested-readable",
        "lane 1",
        "a selector's case is tried against the constraints reading the selector, so a commit "
        "three deep under the forced selectors is readable",
        f"{_SCOPES}::test_a_commit_three_deep_under_forced_selectors_is_readable",
        "selector-case-trial",
        "lane 0 accepts it kept unchecked; lane 1 forces the selectors",
    ),
    Row(
        "nested-commit-under-open-selectors",
        "lane 0",
        "a choice committed three deep under two open selectors is accepted and kept "
        "unchecked, waiting on both, not refused",
        f"{_SCOPES}::test_a_commit_three_deep_under_two_open_selectors_is_kept_unchecked",
        "commit-waits",
    ),
    Row(
        "nested-conflict-commit",
        "lane 1",
        "a value inside its domain three deep, whose core the forced selectors refuse, is "
        "accepted, admitted under the refused core",
        f"{_SCOPES}::test_a_commit_three_deep_in_a_conflict_with_the_forced_selectors_is_accepted",
        "selector-case-trial",
        "lane 0 accepts it kept unchecked; lane 1 forces the outer selector and refuses the "
        "inner one",
    ),
    Row(
        "inactive-under-refused-selector",
        "E2",
        "a choice under an inner selector admitted, under a refused outer selector, on another "
        "case is inactive in the configuration a change would publish: the change is refused, "
        "reaching it as inactive",
        f"{_SCOPES}::test_a_choice_on_an_inner_selectors_other_case_is_inactive_under_a_refused_outer",
        "inactive-under-refused-selector",
        "unclassified: pending a ruling on whether the inner selector's admitted case or the "
        "refused outer selector decides the choice's presence; the model reads it absent, "
        "lane 0's engine under a refused guard (at the base the commits are refused first)",
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
    Row(
        "look-ahead-parent-rule",
        "lane 1",
        "a value case is tried against the constraints that read it, so a parent's rule over "
        "the capabilities makes the case impossible and forces the other",
        f"{_SCOPES}::test_a_parent_rule_over_the_capabilities_makes_the_case_impossible",
        "value-case-trial",
    ),
    Row(
        "look-ahead-inactive",
        "lane 1",
        "a value case is tried against the constraints that read it, so the selector under the "
        "impossible case is inactive, not open",
        f"{_SCOPES}::test_the_selector_under_the_impossible_case_is_inactive",
        "value-case-trial",
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
