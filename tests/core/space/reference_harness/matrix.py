# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The acceptance matrix: every disagreement between the engine and the reference model
expected at this commit, by the milestone that removes it.

The model states the semantics that replace forcing (assignment, refusal at the origin,
dormant choices); the engine at this commit is the one before them. Every disagreement is
therefore the engine core's: the checklist its lane works down, landing with the model.

Each row names the change that removes it, in one sentence, and its **witness**: the one
test that shows it, marked with :func:`expected` (a strict expected failure raising
:class:`~.compare.Disagreement`, so an engine error or a harness fault still fails). A
milestone is accepted when its rows' witnesses pass (each strict expected failure then
fails) and it removes the rows. A row's **kind** is the mechanism of the disagreement its
witness shows on today's engine (a question asked beyond a change today's engine refuses
shows that refusal), shared by the rows it explains, for a walk to match against.

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

#: in order: the line as this harness lands, then the engine core (the rewrite to the
#: semantics the model states, landing with it)
MILESTONES = ("base", "engine core")


@dataclass(frozen=True)
class Row:
    """One expected disagreement."""

    id: str
    milestone: str
    #: the change that removes it, in one sentence: what the milestone's engine must do
    change: str
    #: the test showing it, ``module::function``
    witness: str
    #: the mechanism, shared by the rows it explains: what a walk can match it by
    kind: str
    #: a qualification: how the engine answers it now
    note: str = ""


_CASES = "test_reference_cases"
_RULES = "test_reference_rules"
_SCOPES = "test_reference_scopes"
_CORE = "engine core"

ROWS: tuple[Row, ...] = (
    Row(
        "outside-domain-unselected",
        _CORE,
        "a named value is checked against its domain wherever its choice sits: one outside it "
        "under a case not selected is refused by its type, not as inactive",
        f"{_CASES}::test_a_value_outside_its_domain_under_an_unselected_case_is_refused_by_its_type",
        "inactive-refused",
    ),
    Row(
        "commit-unselected",
        _CORE,
        "a value set under a case not selected is accepted and kept out of play, not refused as "
        "inactive",
        f"{_CASES}::test_a_value_inside_its_domain_under_an_unselected_case_is_accepted_out_of_play",
        "inactive-refused",
    ),
    Row(
        "no-case-waits",
        _CORE,
        "a selector no case of which the facts allow stays open: a value set under it waits on "
        "it, not refused through it",
        f"{_CASES}::test_a_value_under_a_selector_no_case_of_which_holds_still_waits_on_it",
        "no-viable-case",
    ),
    Row(
        "no-case-rule",
        _CORE,
        "a rule of a case under a selector no case of which holds waits on the selector, not "
        "false through the selector's refusal",
        f"{_CASES}::test_a_rule_of_a_case_no_rule_of_which_holds_waits_on_its_selector",
        "no-viable-case",
    ),
    Row(
        "no-case-validity",
        _CORE,
        "a selector with no possible case is no piece of validity: the configuration is open, its "
        "cases' rules waiting on it",
        f"{_CASES}::test_a_selector_no_case_of_which_holds_leaves_the_configuration_open",
        "no-viable-case",
    ),
    Row(
        "select-facts-break",
        _CORE,
        "selecting a case brings its rules in: one false over the facts is newly false and "
        "refuses the switch, naming it with nothing standing",
        f"{_CASES}::test_selecting_a_case_whose_rule_the_facts_break_is_refused_naming_the_rule",
        "change-breaks-rule",
    ),
    Row(
        "outside-domain-trial",
        _CORE,
        "a value outside its domain is refused by its type alone, never read by a case's trial "
        "and refused as inactive once forcing selects the other case",
        f"{_CASES}::test_a_value_outside_its_domain_a_cores_rule_reads_is_refused_by_its_type",
        "forced-case",
    ),
    Row(
        "case-trial-commit",
        _CORE,
        "a value set under an open selector is accepted and waits on it: no case's trial reads "
        "it, so a rule refusing it never selects the other case and refuses it as inactive",
        f"{_CASES}::test_a_value_a_root_rule_refuses_under_an_open_selector_is_accepted_waiting",
        "forced-case",
    ),
    Row(
        "case-trial-select",
        _CORE,
        "selecting the case of a waiting choice a root rule refuses is refused, naming the rule "
        "and the choice standing",
        f"{_CASES}::test_selecting_the_case_of_a_choice_a_root_rule_refuses_names_the_choice_standing",
        "forced-case",
        "today the commit on the way is refused as inactive",
    ),
    Row(
        "guard-open",
        _CORE,
        "nothing gives a choice a value: a choice one of whose values brings a false rule into "
        "play stays open, not forced to the other",
        f"{_CASES}::test_a_guards_choice_whose_value_brings_a_false_rule_into_play_is_open",
        "forced-value",
    ),
    Row(
        "guard-possible",
        _CORE,
        "possible values are answered per value: false refused by the rule it brings into play, "
        "true possible, opening what it brings into presence",
        f"{_CASES}::test_a_guards_value_bringing_a_false_rule_into_play_is_not_possible",
        "viable-per-choice",
        "today the choice is forced, so not listed",
    ),
    Row(
        "guard-off",
        _CORE,
        "turning a guard off is accepted and puts the stored choices under it dormant, not "
        "refused as inactive",
        f"{_CASES}::test_turning_a_guard_off_puts_the_choices_under_it_dormant",
        "inactive-refused",
    ),
    Row(
        "empty-list-state",
        _CORE,
        "a choice whose list is empty is open (its domain is the piece that refuses), not refused",
        f"{_CASES}::test_a_choice_whose_list_is_empty_is_open",
        "no-viable-case",
    ),
    Row(
        "empty-list-code",
        _CORE,
        "an empty list's piece carries the code domain-empty, not decision-no-viable-case",
        f"{_CASES}::test_an_empty_list_refuses_the_configuration",
        "provisional-code",
    ),
    Row(
        "switch-waiting-away",
        _CORE,
        "selecting the other case puts a waiting choice dormant and opens the new case's choices, "
        "reported; not refused as inactive",
        f"{_CASES}::test_selecting_the_other_case_puts_a_waiting_choice_dormant",
        "inactive-refused",
    ),
    Row(
        "select-stored-break",
        _CORE,
        "selecting a case whose rule a stored choice breaks is refused, naming the rule and that "
        "choice standing",
        f"{_CASES}::test_selecting_a_case_whose_rule_a_stored_choice_breaks_names_it_standing",
        "change-breaks-rule",
    ),
    Row(
        "selector-possible",
        _CORE,
        "each possible case of a selector names what selecting it moves: in play, dormant and "
        "opened",
        f"{_CASES}::test_the_possible_cases_of_a_selector_name_what_each_moves",
        "viable-per-choice",
    ),
    Row(
        "case-rule-commit",
        _CORE,
        "a choice set under an open selector whose case's rule refuses it is accepted, waiting: "
        "no trial selects the other case and refuses it as inactive",
        f"{_CASES}::test_setting_a_choice_its_cases_rule_refuses_under_an_open_selector_is_accepted",
        "forced-case",
    ),
    Row(
        "case-rule-possible",
        _CORE,
        "a case whose rule refuses a waiting choice of it is impossible, the choice standing; the "
        "other case possible, putting the choice dormant",
        f"{_CASES}::test_a_case_whose_rule_refuses_a_waiting_choice_of_it_is_not_possible",
        "forced-case",
        "today the commit on the way is refused as inactive",
    ),
    Row(
        "refused-input-commit",
        _CORE,
        "a value whose membership reads a refused input is not of its type: the commit is "
        "refused, blamed at the input, not kept",
        f"{_CASES}::test_a_value_whose_membership_reads_a_refused_input_is_refused_at_commit",
        "assignment-type-check",
    ),
    Row(
        "refused-input-facts",
        _CORE,
        "new facts making a stored value's membership input refuse are refused by that value's "
        "type check",
        f"{_CASES}::test_new_facts_that_refuse_a_stored_choices_membership_input_are_refused",
        "assignment-type-check",
    ),
    Row(
        "refused-input-validity",
        _CORE,
        "a listed domain whose list cannot be made through a refused input is a false piece, "
        "blamed at the input",
        f"{_CASES}::test_a_list_refused_through_its_input_refuses_the_configuration_at_the_input",
        "legacy-validity",
    ),
    Row(
        "guarded-input-commit",
        _CORE,
        "a membership input that does not apply for these facts (a guard over facts beyond the "
        "choice's own is false) refuses the assignment, naming it (domain-input-absent, § 16.17)",
        f"{_CASES}::test_a_commit_whose_membership_input_does_not_apply_is_refused_naming_it",
        "assignment-type-check",
    ),
    Row(
        "guarded-input-validity",
        _CORE,
        "an unset present choice whose membership input does not apply has a list that cannot "
        "be made: its domain piece is false through the input (§ 16.17)",
        f"{_CASES}::test_an_unset_choice_whose_membership_input_does_not_apply_refuses_the_"
        "configuration",
        "assignment-type-check",
        "today the list is made over the facts whatever the input's presence: valid",
    ),
    Row(
        "guarded-input-facts",
        _CORE,
        "new facts putting a stored value's membership input out of play are refused by that "
        "value's type check (domain-input-absent, § 16.17)",
        f"{_CASES}::test_new_facts_putting_a_stored_choices_membership_input_out_of_play_are_"
        "refused",
        "assignment-type-check",
    ),
    Row(
        "linked-open",
        _CORE,
        "a linking rule leaves the other choice open, not forced to the one value it allows",
        f"{_RULES}::test_a_linking_rule_leaves_the_other_choice_open",
        "forced-value",
    ),
    Row(
        "linked-refuses",
        _CORE,
        "a value breaking a linking rule is refused, naming the rule and the other choice standing",
        f"{_RULES}::test_a_linking_rule_refuses_a_value_naming_the_other_choice_standing",
        "change-breaks-rule",
    ),
    Row(
        "linked-possible",
        _CORE,
        "an impossible value carries its refusal: the rule, and the other choice standing",
        f"{_RULES}::test_a_linking_rules_possible_values_name_the_other_choice_standing",
        "viable-per-choice",
        "today's reasons are text",
    ),
    Row(
        "domain-reads-choice",
        _CORE,
        "a domain whose membership reads a choice is a definition error at compile "
        "(domain-reads-choice)",
        f"{_RULES}::test_a_domain_reading_another_choice_is_no_definition",
        "domain-reads-choice",
    ),
    Row(
        "required-open",
        _CORE,
        "a choice one of whose cases the facts rule out stays open, not forced to the other",
        f"{_RULES}::test_a_case_needing_a_capability_the_facts_lack_leaves_the_choice_open",
        "forced-value",
    ),
    Row(
        "required-possible",
        _CORE,
        "the possible values of a choice the facts narrow: the allowed case possible, the other "
        "refused by its rule, nothing standing",
        f"{_RULES}::test_a_case_needing_a_capability_the_facts_lack_is_not_possible",
        "viable-per-choice",
        "today the choice is forced, so not listed",
    ),
    Row(
        "required-commit",
        _CORE,
        "a value breaking a rule over itself and the facts is refused, naming the rule with "
        "nothing standing",
        f"{_RULES}::test_setting_a_case_needing_a_missing_capability_is_refused_by_the_facts_alone",
        "change-breaks-rule",
    ),
    Row(
        "required-facts",
        _CORE,
        "new facts making a rule newly false are refused, naming the rule and the stored choice "
        "standing",
        f"{_RULES}::test_new_facts_lacking_a_capability_a_stored_case_needs_are_refused",
        "change-breaks-rule",
    ),
    Row(
        "pair-left-alone",
        _CORE,
        "the canon's Pair: left alone at 6 is refused by right_not_slower, right's lanes standing",
        f"{_RULES}::test_the_pairs_left_alone_at_six_is_refused_naming_the_rule_and_right",
        "change-breaks-rule",
    ),
    Row(
        "pair-possible",
        _CORE,
        "possible values answer a stored choice too, each refused value with its rule and what "
        "stands in its way",
        f"{_RULES}::test_the_pairs_possible_values_of_left_beside_right_at_three",
        "viable-per-choice",
        "today a stored choice is not listed",
    ),
    Row(
        "memory-possible",
        _CORE,
        "the canon's Memory: block possible, ultra refused by ultra_needs_uram with nothing "
        "standing",
        f"{_RULES}::test_the_memorys_ultra_is_not_possible_without_uram",
        "viable-per-choice",
        "today the choice is forced, so not listed",
    ),
    Row(
        "memory-commit",
        _CORE,
        "the canon's Memory: ultra without UltraRAM is refused, naming ultra_needs_uram",
        f"{_RULES}::test_the_memorys_ultra_is_refused_without_uram_nothing_standing",
        "change-breaks-rule",
    ),
    Row(
        "origin-blame",
        _CORE,
        "a change making a derived value refuse on its own values is refused at that origin "
        "alone, not at the rule false through it",
        f"{_RULES}::test_a_refusal_through_a_refused_input_is_blamed_at_its_origin_only",
        "change-breaks-rule",
    ),
    Row(
        "switch-reaches-case-rule",
        _CORE,
        "a switch is refused by a rule of the new case false over the facts, naming it, not for "
        "the old case's choices it leaves out of play",
        f"{_RULES}::test_a_rule_in_a_case_that_never_names_its_selector_refuses_a_switch_to_it",
        "inactive-refused",
    ),
    Row(
        "switch-dormant-opened",
        _CORE,
        "a switch is accepted, putting the old case's choices dormant and opening the new case's, "
        "reported",
        f"{_RULES}::test_a_switch_puts_the_old_cases_choices_dormant_and_opens_the_new_ones",
        "inactive-refused",
    ),
    Row(
        "sum-two-standing",
        _CORE,
        "a new value breaking a sum is refused, naming the other two choices standing",
        f"{_RULES}::test_a_new_value_breaking_a_sum_names_the_other_two_standing",
        "change-breaks-rule",
    ),
    Row(
        "sum-one-standing",
        _CORE,
        "a new value breaking a sum with one other set in the same change names the last standing",
        f"{_RULES}::test_a_new_value_breaking_a_sum_with_one_other_set_names_the_last_standing",
        "change-breaks-rule",
    ),
    Row(
        "switch-away",
        _CORE,
        "switching a selector away is accepted, its case's choices dormant and the new case's "
        "opened, reported",
        f"{_RULES}::test_switching_away_puts_the_cases_choices_dormant",
        "inactive-refused",
    ),
    Row(
        "switched-away-dormant",
        _CORE,
        "a choice of the case switched away is dormant, keeping its value",
        f"{_RULES}::test_a_choice_of_the_case_switched_away_is_dormant",
        "inactive-refused",
        "today the switch on the way is refused as inactive",
    ),
    Row(
        "switch-back",
        _CORE,
        "switching back brings the dormant choices into play and puts the other case's dormant, "
        "reported",
        f"{_RULES}::test_switching_back_brings_the_dormant_choices_into_play",
        "inactive-refused",
        "today the switch on the way is refused as inactive",
    ),
    Row(
        "switch-back-refused",
        _CORE,
        "a switch back is refused by a rule a restored choice breaks, naming it and the restored "
        "choice and the other stored one standing",
        f"{_RULES}::test_a_switch_back_is_refused_by_a_restored_choice",
        "inactive-refused",
        "today the switch on the way is refused as inactive",
    ),
    Row(
        "switch-back-moved",
        _CORE,
        "a switch back setting the restored choice in the same change is accepted, the choice in "
        "play",
        f"{_RULES}::test_a_switch_back_with_the_restored_choice_moved_too_is_accepted",
        "inactive-refused",
        "today the switch on the way is refused as inactive",
    ),
    Row(
        "facts-dormant-outside",
        _CORE,
        "a facts change is refused by a dormant value outside its new domain: every stored value "
        "is type-checked",
        f"{_RULES}::test_a_facts_change_is_refused_by_a_dormant_value_outside_its_new_domain",
        "inactive-refused",
        "today the switch on the way is refused as inactive",
    ),
    Row(
        "facts-each-outside",
        _CORE,
        "a facts change names each stored value outside its new domain, in play or dormant",
        f"{_RULES}::test_a_facts_change_is_refused_by_each_value_outside_its_new_domain",
        "inactive-refused",
        "today the switch on the way is refused as inactive",
    ),
    Row(
        "facts-clear-standing",
        _CORE,
        "a facts change clearing the value that stands in its way is accepted",
        f"{_RULES}::test_a_facts_change_clearing_what_stands_in_its_way_is_accepted",
        "inactive-refused",
        "today the switch on the way is refused as inactive",
    ),
    Row(
        "waiting-valid",
        _CORE,
        "a waiting choice whose list has values is no piece of validity: valid, not open on its "
        "selector",
        f"{_RULES}::test_a_named_choice_under_an_open_selector_leaves_the_configuration_valid",
        "legacy-validity",
    ),
    Row(
        "waiting-in-play",
        _CORE,
        "selecting a waiting choice's case reports it brought into play",
        f"{_RULES}::test_selecting_a_waiting_choices_case_brings_it_into_play",
        "no-dormant-report",
    ),
    Row(
        "nested-open",
        _CORE,
        "a selector whose other case the facts refuse stays open, not forced to the case left",
        f"{_SCOPES}::test_a_selector_whose_other_case_the_facts_refuse_stays_open",
        "forced-value",
    ),
    Row(
        "nested-waits",
        _CORE,
        "a value set three deep waits on both open selectors, which nothing sets: not read as set",
        f"{_SCOPES}::test_a_commit_three_deep_under_two_open_selectors_waits_on_both",
        "forced-value",
    ),
    Row(
        "nested-opened",
        _CORE,
        "selecting the outer kernel reports the inner selector opened",
        f"{_SCOPES}::test_selecting_the_outer_kernel_leaves_a_choice_three_deep_waiting_on_the_inner",
        "no-dormant-report",
    ),
    Row(
        "nested-in-play",
        _CORE,
        "selecting the inner kernel reports the choice three deep brought into play",
        f"{_SCOPES}::test_selecting_both_kernels_brings_a_choice_three_deep_into_play",
        "no-dormant-report",
    ),
    Row(
        "nested-away",
        _CORE,
        "switching the outer kernel away is accepted, the inner selector and the choice three "
        "deep dormant, reported",
        f"{_SCOPES}::test_switching_the_outer_kernel_away_puts_the_inner_choices_dormant",
        "inactive-refused",
    ),
    Row(
        "nested-rule",
        _CORE,
        "a value three deep breaking its case's rule over the facts is refused, naming the rule",
        f"{_SCOPES}::test_a_commit_three_deep_breaking_its_rule_is_refused_naming_it",
        "change-breaks-rule",
    ),
    Row(
        "outer-facts",
        _CORE,
        "new facts under which no case of an open selector holds are accepted: nothing is refused "
        "through the selector",
        f"{_SCOPES}::test_new_facts_no_case_of_an_open_selector_holds_under_are_accepted",
        "no-viable-case",
    ),
    Row(
        "outer-select",
        _CORE,
        "selecting the outer case brings the inner selector into play, reported; the choice under "
        "its other case stays dormant, not refused as inactive",
        f"{_SCOPES}::test_selecting_the_outer_case_brings_the_inner_selector_into_play",
        "inactive-refused",
    ),
    Row(
        "look-opened",
        _CORE,
        "a possible value names the choices it opens: native opens the core selector it puts in "
        "scope",
        f"{_SCOPES}::test_a_case_is_not_tried_against_a_selector_it_puts_in_scope",
        "viable-per-choice",
    ),
    Row(
        "look-open",
        _CORE,
        "a selector no case of which the facts allow stays open, not refused",
        f"{_SCOPES}::test_the_selector_a_set_case_puts_in_scope_stays_open",
        "no-viable-case",
    ),
    Row(
        "look-validity",
        _CORE,
        "a selector with no possible case leaves the configuration open, its cases' rules waiting "
        "on it, not refused",
        f"{_SCOPES}::test_the_selector_a_set_case_puts_in_scope_leaves_the_configuration_open",
        "no-viable-case",
    ),
    Row(
        "ruled-possible",
        _CORE,
        "native is impossible by the parent's rule over the facts, nothing standing; other "
        "possible",
        f"{_SCOPES}::test_a_parent_rule_over_the_capabilities_makes_the_case_impossible",
        "viable-per-choice",
        "today the choice is forced, so not listed",
    ),
    Row(
        "guard-input-facts",
        _CORE,
        "new facts making a guard's input refuse are refused at that input's origin",
        f"{_SCOPES}::test_new_facts_refusing_a_guards_input_are_refused_at_the_input",
        "change-breaks-rule",
    ),
    Row(
        "guard-code",
        _CORE,
        "a guard refused through its input is one piece carrying guard-refused beside its input's "
        "codes",
        f"{_SCOPES}::test_a_guard_refused_through_its_input_refuses_the_configuration_at_the_guard",
        "provisional-code",
    ),
    Row(
        "guard-absent",
        _CORE,
        "a scope under a guard refused through its input is out of play: an unset choice in it is "
        "absent, not refused",
        f"{_SCOPES}::test_an_unset_choice_under_a_guard_refused_through_its_input_is_absent",
        "refused-guard",
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
        lines.append(f"| {row.id} | {row.milestone} | {change} | `{row.witness}` | {row.kind} |")
    return "\n".join(lines)
