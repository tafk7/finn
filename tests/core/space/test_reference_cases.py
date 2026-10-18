# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model, case by case: one test per question.

Each case is a small synthetic description, built in the model and in the engine
(``core.space.reference_harness``); each test brings one configuration to its question on
both sides and compares the answers. A question the engine answers differently at this
commit is a strict expected failure, the witness of its row in the acceptance matrix
(``reference_harness.matrix``), which names the milestone that removes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Refuse,
    Scope,
    Selector,
    Space,
    When,
    computed,
    listed,
    membership,
)
from core.space.reference_harness import Case, Pair, Spelling, expected

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")


def divisors(extent: int) -> tuple[int, ...]:
    return tuple(value for value in range(1, extent + 1) if extent % value == 0)


# -- 1. typed membership ---------------------------------------------------------------------------
#
# Two cores share a width: ``packed`` takes at most 64 lanes, its ``pe`` a divisor of the
# width; ``stub`` takes an even width. At 129 neither core is possible: the selector is
# refused. A value outside its domain is refused by its own check, whatever its guard says.

CORES = Selector(
    "compute",
    (
        Scope(
            "packed",
            choices=(Choice("pe", computed(lambda read: divisors(read["width"]), ("width",))),),
            constraints=(
                Constraint("fits", ("width",), lambda read: read["width"] <= 64, "packed-width"),
            ),
        ),
        Scope(
            "stub",
            choices=(Choice("rows", listed(1, 2)),),
            constraints=(
                Constraint("even", ("width",), lambda read: read["width"] % 2 == 0, "stub-width"),
            ),
        ),
    ),
)
TYPED = Case(
    "typed membership",
    Space(("width",), Scope("", choices=(Choice("x", listed(1, 2, 3)),), selectors=(CORES,))),
)


def test_a_value_outside_its_domain_is_refused_at_commit() -> None:
    TYPED.start({"width": 8}).check_change(Change(commit={"x": 4}))


def test_a_value_outside_its_domain_under_a_refused_selector_is_refused_with_its_own_finding() -> (
    None
):
    TYPED.start({"width": 129}).check_change(Change(commit={"compute.packed.pe": 2}))


def test_a_retained_value_new_facts_put_outside_its_domain_is_refused_with_its_own_finding() -> (
    None
):
    # At 9 the stub core refuses an odd width, so packed is forced and 3 divides 9; at 71
    # neither core is possible and 3 does not divide 71.
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3})
    point.check_change(Change(facts={"width": 71}))


def test_a_selector_left_no_case_refuses_the_configuration() -> None:
    TYPED.start({"width": 129}).check_validity()


def test_a_value_inside_its_domain_under_a_refused_selector_is_accepted() -> None:
    # 3 divides 129: its own check holds, so it is admitted, never unchecked (unchecked is
    # for an open selector only), and the configuration reads refused through the selector.
    TYPED.start({"width": 129}).check_change(Change(commit={"compute.packed.pe": 3}))


def test_a_retained_value_inside_its_domain_under_a_selector_new_facts_refuse_is_kept() -> None:
    # At 99 neither core is possible (past 64 lanes, odd), and 3 divides 99: kept, admitted,
    # so nothing is kept unchecked.
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3})
    point.check_change(Change(facts={"width": 99}))


def refused_under() -> Pair:
    """``pe`` admitted at 3, then new facts leave its selector no possible case."""
    return TYPED.start({"width": 9}, {"compute.packed.pe": 3}).step(Change(facts={"width": 99}))


def test_an_admitted_value_under_a_refused_selector_reads_refused_through_it() -> None:
    # Published, its state is refused with the selector's no-possible-case.
    refused_under().check_state("compute.packed.pe")


def test_an_admitted_value_under_a_refused_selector_is_queried_refused_through_it() -> None:
    refused_under().check_answer("compute.packed.pe")


def test_an_open_choice_under_a_refused_selector_reads_refused_through_it() -> None:
    # Blamed at the selector: its own lack of a possible value adds nothing.
    refused_under().check_state("compute.stub.rows")


def test_a_rule_under_a_refused_selector_reads_refused_through_it() -> None:
    # Not evaluated; queried directly, it is refused with the selector's reasons.
    refused_under().check_rule("compute.packed.fits")


def test_a_value_outside_its_domain_under_an_open_selector_is_refused_at_commit() -> None:
    # At 12 both cores are possible: the selector is open, and 5 does not divide 12. Its own
    # check is read with its case assumed: refused, named, with its own reason only.
    TYPED.start({"width": 12}).check_change(Change(commit={"compute.packed.pe": 5}))


def test_a_value_outside_its_domain_under_a_false_guard_is_refused_as_inactive() -> None:
    # The stub core is selected: ``pe`` does not exist, so its own check (5 does not divide
    # 12) is not evaluated, and the value is refused as inactive.
    TYPED.start({"width": 12}, {"compute": "stub"}).check_change(
        Change(commit={"compute.packed.pe": 5})
    )


def test_a_value_inside_its_domain_under_a_false_guard_is_refused_as_inactive() -> None:
    TYPED.start({"width": 12}, {"compute": "stub"}).check_change(
        Change(commit={"compute.packed.pe": 3})
    )


# Two cores both possible, the first's rule reading its own ``pe`` (whether it divides the
# width): a trial of that core reads ``pe``, so a ``pe`` outside its domain must not be read
# there, or forcing selects the other core and ``pe`` reads inactive.

TWO_CORES = Case(
    "two cores, a rule reading a core's choice",
    Space(
        ("width",),
        Scope(
            "",
            selectors=(
                Selector(
                    "compute",
                    (
                        Scope(
                            "packed",
                            choices=(
                                Choice(
                                    "pe", computed(lambda read: divisors(read["width"]), ("width",))
                                ),
                            ),
                            constraints=(
                                Constraint(
                                    "divides",
                                    ("width", "compute.packed.pe"),
                                    lambda read: read["width"] % read["compute.packed.pe"] == 0,
                                    "lanes-divide",
                                ),
                            ),
                        ),
                        Scope("closed", choices=(Choice("n", listed(1, 2, 3)),)),
                    ),
                ),
            ),
        ),
    ),
)


@expected("outside-domain-forcing")
def test_a_value_outside_its_domain_a_cores_rule_reads_is_refused_with_its_own_reason() -> None:
    # At 4 both cores are possible, and 3 does not divide 4.
    TWO_CORES.start({"width": 4}).check_change(Change(commit={"compute.packed.pe": 3}))


def test_a_refused_selector_over_an_admitted_value_refuses_the_configuration() -> None:
    refused_under().check_validity()


# The same cores with a rule of the root reading packed's ``pe``: once the selector has no
# possible case, the admitted ``pe`` reads refused through it, so the rule is false, naming
# the selector's no-possible-case (the reasons of the refused input it reads).

ROOT_RULE = Case(
    "a root rule reading a choice under the cores",
    Space(
        ("width",),
        Scope(
            "",
            selectors=(CORES,),
            constraints=(
                Constraint(
                    "small",
                    ("compute.packed.pe",),
                    lambda read: read["compute.packed.pe"] <= 8,
                    "pe-small",
                ),
            ),
        ),
    ),
)


def test_a_root_rule_reading_a_choice_admitted_under_a_refused_selector_is_false_through_it() -> (
    None
):
    point = ROOT_RULE.start({"width": 9}, {"compute.packed.pe": 3})
    point.step(Change(facts={"width": 99})).check_rule("small")


# -- 2. guards -----------------------------------------------------------------------------------
#
# ``core`` selects ``big`` (whose rule needs a budget of 10) or ``small``; ``extra`` is
# present when the fact ``on`` holds, and its rule needs a budget of 20; ``spare`` is
# present when the choice ``want`` holds, ``costly`` when it does not, and ``costly``'s
# rule needs a budget of 20 too.

GUARDS = Case(
    "guards",
    Space(
        ("budget", "on"),
        Scope(
            "",
            choices=(Choice("want", listed(False, True)),),
            selectors=(
                Selector(
                    "core",
                    (
                        Scope(
                            "big",
                            choices=(Choice("n", listed(1, 2)),),
                            constraints=(
                                Constraint(
                                    "limit",
                                    ("budget",),
                                    lambda read: read["budget"] >= 10,
                                    "big-budget",
                                ),
                            ),
                        ),
                        Scope("small", choices=(Choice("m", listed(1, 2)),)),
                    ),
                ),
            ),
            children=(
                Scope(
                    "extra",
                    when=When(("on",), lambda read: bool(read["on"])),
                    choices=(Choice("size", listed(1, 2, 3)),),
                    constraints=(
                        Constraint(
                            "cap", ("budget",), lambda read: read["budget"] >= 20, "extra-budget"
                        ),
                    ),
                ),
                Scope(
                    "spare",
                    when=When(("want",), lambda read: bool(read["want"])),
                    choices=(Choice("size", listed(1, 2)),),
                ),
                Scope(
                    "costly",
                    when=When(("want",), lambda read: not read["want"]),
                    constraints=(
                        Constraint(
                            "cap", ("budget",), lambda read: read["budget"] >= 20, "costly-budget"
                        ),
                    ),
                ),
            ),
        ),
    ),
    Spelling(types={"on": bool}),
)


def test_a_rule_under_an_unselected_case_does_not_apply() -> None:
    GUARDS.start({"budget": 5}, {"core": "small"}).check_rule("core.big.limit")


def test_a_rule_under_an_unselected_case_counts_as_true() -> None:
    point = GUARDS.start({"budget": 5, "on": False}, {"core": "small", "want": True})
    point.check_validity()


def test_a_rule_under_a_false_guard_does_not_apply() -> None:
    GUARDS.start({"budget": 5, "on": False}).check_rule("extra.cap")


def test_a_rule_under_an_unknown_guard_reads_unknown() -> None:
    GUARDS.start({"budget": 5}).check_rule("extra.cap")


def test_a_rule_that_holds_under_an_unknown_guard_reads_unknown_not_true() -> None:
    GUARDS.start({"budget": 25}).check_rule("extra.cap")


def test_a_rule_under_an_unknown_guard_leaves_the_configuration_undecided() -> None:
    GUARDS.start({"budget": 5}, {"core": "small", "want": True}).check_validity()


def test_a_choice_under_an_unknown_guard_is_open_waiting_on_it() -> None:
    GUARDS.start({"budget": 5}).check_state("extra.size")


@expected("value-case-trial")
def test_a_guards_choice_takes_no_case_that_makes_a_rule_under_it_false() -> None:
    # At a budget of 5 ``costly``'s rule is false wherever it applies, and it applies when
    # ``want`` is false: so ``want`` has one possible case, and it is forced.
    GUARDS.start({"budget": 5}).check_state("want")


# -- 3. exhaustive and empty domains ---------------------------------------------------------------
#
# A listed domain lists all its values, and every listed value is checked by its
# membership: ``x`` lists 1, 2 and 3 and rejects 2; ``y`` lists 1 and 2 and rejects 2. ``w``
# is known by membership only. ``d`` lists the divisors of ``n``: none at 0.

DOMAINS = Case(
    "exhaustive domains",
    Space(
        ("n",),
        Scope(
            "",
            choices=(
                Choice("x", listed(1, 2, 3, member=lambda value, read: value != 2)),
                Choice("y", listed(1, 2, member=lambda value, read: value != 2)),
                Choice("w", membership(lambda value, read: value >= 2)),
                Choice("d", computed(lambda read: divisors(read["n"]), ("n",))),
            ),
        ),
    ),
)
EMPTY = Case("an empty domain", Space((), Scope("", choices=(Choice("z", listed()),))))


@expected("listed-value-not-member-possible")
def test_a_listed_value_the_domain_rejects_is_not_possible() -> None:
    DOMAINS.start({"n": 6}).check_possible("x")


@expected("listed-value-not-member-forced")
def test_a_listed_value_the_domain_rejects_leaves_the_other_forced() -> None:
    DOMAINS.start({"n": 6}).check_state("y")


def test_a_listed_value_the_domain_rejects_is_refused_at_commit() -> None:
    DOMAINS.start({"n": 6}).check_change(Change(commit={"x": 2}))


def test_a_choice_known_by_membership_only_is_open_with_no_list() -> None:
    DOMAINS.start({"n": 6}).check_state("w")


def test_a_value_outside_a_membership_only_domain_is_refused_at_commit() -> None:
    DOMAINS.start({"n": 6}).check_change(Change(commit={"w": 1}))


@expected("empty-list-choice")
def test_an_empty_list_refuses_its_choice() -> None:
    EMPTY.start().check_state("z")


@expected("empty-computed-list-choice")
def test_an_empty_computed_list_refuses_its_choice() -> None:
    DOMAINS.start({"n": 0}).check_state("d")


@expected("empty-list-validity")
def test_an_empty_list_refuses_the_configuration() -> None:
    EMPTY.start().check_validity()


def test_a_value_under_an_empty_list_is_outside_its_domain() -> None:
    EMPTY.start().check_change(Change(commit={"z": 1}))


# -- 4. unchecked readability ----------------------------------------------------------------------
#
# The two cores again, their width a choice ``lanes`` (9 or 12): at 9 the stub core
# refuses, so packed is forced; at 12 both are possible and the selector is open. ``doubled``
# is a value computed from packed's ``pe``. One Space: a change that leaves a stored choice
# inactive is refused.

UNCHECKED = Case(
    "unchecked readability",
    Space(
        (),
        Scope(
            "",
            choices=(Choice("lanes", listed(9, 12)),),
            selectors=(
                Selector(
                    "compute",
                    (
                        Scope(
                            "packed",
                            choices=(Choice("pe", listed(1, 3)),),
                            constraints=(
                                Constraint(
                                    "fits",
                                    ("lanes",),
                                    lambda read: read["lanes"] <= 64,
                                    "packed-width",
                                ),
                            ),
                        ),
                        Scope(
                            "stub",
                            choices=(Choice("rows", listed(1, 2)),),
                            constraints=(
                                Constraint(
                                    "even",
                                    ("lanes",),
                                    lambda read: read["lanes"] % 2 == 0,
                                    "stub-width",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            derived=(
                Derived(
                    "doubled", ("compute.packed.pe",), lambda read: 2 * read["compute.packed.pe"]
                ),
            ),
        ),
    ),
)


def opened() -> Pair:
    """``pe`` committed while packed is forced, then the selector opened by ``lanes``."""
    point = UNCHECKED.start({}, {"lanes": 9}).step(Change(commit={"compute.packed.pe": 3}))
    return point.step(Change(commit={"lanes": 12}))


def test_a_change_that_opens_a_selector_keeps_a_choice_under_it_unchecked() -> None:
    point = UNCHECKED.start({}, {"lanes": 9}).step(Change(commit={"compute.packed.pe": 3}))
    point.check_change(Change(commit={"lanes": 12}))


def test_an_unchecked_choice_waits_on_its_selector() -> None:
    opened().check_state("compute.packed.pe")


def test_an_unchecked_choice_reads_unresolved() -> None:
    opened().check_answer("compute.packed.pe")


def test_a_value_computed_from_an_unchecked_choice_is_unresolved() -> None:
    opened().check_answer("doubled")


def test_an_unchecked_choice_leaves_the_configuration_undecided() -> None:
    opened().check_validity()


def test_an_unchecked_choice_is_checked_when_its_selector_settles() -> None:
    opened().step(Change(commit={"compute": "packed"})).check_state("compute.packed.pe")


def test_a_selector_settling_elsewhere_refuses_the_change() -> None:
    opened().check_change(Change(commit={"compute": "stub"}))


def test_a_choice_committed_under_an_open_selector_is_kept_unchecked() -> None:
    UNCHECKED.start({}, {"lanes": 12}).check_change(Change(commit={"compute.packed.pe": 3}))


def test_new_facts_that_open_a_selector_keep_a_choice_under_it_unchecked() -> None:
    # At 12 both cores are possible: the selector is open, and ``pe`` (3 divides 12) waits.
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3})
    point.check_change(Change(facts={"width": 12}))


# A domain computed from a fact the configuration does not give yet (met by the seeded walks):
# a value committed to it waits on the fact, as one under an open selector waits on the selector.

WAITING = Case(
    "a domain waiting on a fact",
    Space(
        ("width",),
        Scope(
            "",
            choices=(Choice("pe", computed(lambda read: divisors(read["width"]), ("width",))),),
        ),
    ),
)


def test_a_choice_committed_while_its_domain_waits_on_a_fact_is_kept_unchecked() -> None:
    WAITING.start().check_change(Change(commit={"pe": 3}))


# A stored choice of a selector's case, its own check true, that a rule of the case refuses
# (met by the seeded walks): once the selector is cleared the choice waits only on its
# presence. The case's trial settles that presence, so it reads the choice as its value.

CASE_TRIAL = Case(
    "a case's trial and a choice of the case kept unchecked",
    Space(
        ("limit",),
        Scope(
            "",
            selectors=(
                Selector(
                    "core",
                    (
                        Scope("plain"),
                        Scope(
                            "wide",
                            choices=(Choice("lanes", listed(1, 2)),),
                            constraints=(
                                Constraint(
                                    "small",
                                    ("core.wide.lanes", "limit"),
                                    lambda read: read["core.wide.lanes"] <= read["limit"],
                                    "too-many-lanes",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


def test_clearing_a_selector_whose_case_a_kept_choice_rules_out_is_refused_as_inactive() -> None:
    # The trial of ``wide`` reads ``lanes`` as 2, which ``small`` refuses at a limit of 1:
    # ``wide`` is not possible, ``plain`` is forced, and the clear is refused, leaving the
    # retained ``lanes`` inactive.
    point = CASE_TRIAL.start({"limit": 1}, {"core": "wide"}).step(
        Change(commit={"core.wide.lanes": 2})
    )
    point.check_change(Change(clear=("core",)))


def test_committing_a_choice_its_cases_rule_refuses_under_two_cases_is_refused_as_inactive() -> (
    None
):
    # The selector is open (the limit is a fact, both cases possible). The trial of ``wide``
    # reads the committed ``lanes`` as 2, which ``small`` refuses at a limit of 1: ``plain``
    # is forced, and the commit is refused, ``lanes`` inactive where it would publish.
    CASE_TRIAL.start({"limit": 1}).check_change(Change(commit={"core.wide.lanes": 2}))


# The same with a third case: the commit is accepted, the choice kept unchecked waiting on its
# selector, and the case whose rule refuses its value is not possible, the other two are.

THREE_CASES = Case(
    "a case's trial and a choice of the case kept unchecked, three cases",
    Space(
        ("limit",),
        Scope(
            "",
            selectors=(
                Selector(
                    "core",
                    (
                        Scope("plain"),
                        Scope("other"),
                        Scope(
                            "wide",
                            choices=(Choice("lanes", listed(1, 2)),),
                            constraints=(
                                Constraint(
                                    "small",
                                    ("core.wide.lanes", "limit"),
                                    lambda read: read["core.wide.lanes"] <= read["limit"],
                                    "too-many-lanes",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


def committed_wide() -> Pair:
    """``lanes`` committed at 2 under the open selector, at a limit of 1."""
    return THREE_CASES.start({"limit": 1}).step(Change(commit={"core.wide.lanes": 2}))


def test_committing_a_choice_its_cases_rule_refuses_under_three_cases_is_accepted() -> None:
    THREE_CASES.start({"limit": 1}).check_change(Change(commit={"core.wide.lanes": 2}))


def test_a_choice_its_cases_rule_refuses_stays_unchecked_under_three_cases() -> None:
    committed_wide().check_state("core.wide.lanes")


@expected("case-trial-reads-unchecked")
def test_a_case_whose_rule_refuses_a_kept_choice_of_it_is_not_possible() -> None:
    # The trial of ``wide`` settles the presence ``lanes`` waits on, so it reads 2, which
    # ``small`` refuses: ``plain`` and ``other`` are possible, ``wide`` is not.
    committed_wide().check_possible("core")


def test_a_choice_its_cases_rule_refuses_leaves_the_configuration_undecided() -> None:
    committed_wide().check_validity()


# A choice of a case whose domain is computed from a fact not given yet: committed under the
# open selector it waits on the fact as well as on its presence, so no trial reads it as its
# value, and a rule of its case reading it stays unknown there (met by the seeded walks).


def waiting_cases(*names: str) -> Case:
    return Case(
        f"a case's trial and a choice of the case waiting on a fact, cases {', '.join(names)}",
        Space(
            ("width",),
            Scope(
                "",
                choices=(Choice("lanes", listed(1, 2, 3)),),
                selectors=(
                    Selector(
                        "core",
                        (
                            *(Scope(name) for name in names),
                            Scope(
                                "wide",
                                choices=(
                                    Choice(
                                        "pe",
                                        computed(lambda read: divisors(read["width"]), ("width",)),
                                    ),
                                ),
                                constraints=(
                                    Constraint(
                                        "enough",
                                        ("lanes", "core.wide.pe"),
                                        lambda read: read["lanes"] <= read["core.wide.pe"],
                                        "too-few-pe",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


WAITING_TWO = waiting_cases("plain")
WAITING_THREE = waiting_cases("plain", "other")


@expected("case-trial-reads-waiting")
def test_a_choice_waiting_on_a_fact_under_two_cases_is_kept_unchecked() -> None:
    # ``pe`` of 1 waits on the width and on ``core``: ``enough`` (3 lanes need 3) is unknown
    # in the trial of ``wide``, so both cases stay possible and the commit is accepted.
    WAITING_TWO.start({}, {"lanes": 3}).check_change(Change(commit={"core.wide.pe": 1}))


def test_a_choice_waiting_on_a_fact_under_three_cases_is_kept_unchecked() -> None:
    WAITING_THREE.start({}, {"lanes": 3}).check_change(Change(commit={"core.wide.pe": 1}))


# A domain reading a derived value that refuses on its facts: ``width`` refuses an odd ``w``
# (``bad-width``), and ``lanes``'s domain is the divisors of ``width`` among 1, 2, 4 and 8.
# Only membership refuses a value, so a refused input of the domain never refuses a commit:
# the choice is kept, never read as a value, and the configuration is refused through the
# input, blamed there once. A fact not given leaves the choice waiting on it.


def even_width(read: Mapping[str, Any]) -> object:
    return Refuse("bad-width") if read["w"] % 2 else read["w"]


REFUSED_INPUT = Case(
    "a domain reading a derived value that refuses",
    Space(
        ("w",),
        Scope(
            "",
            choices=(
                Choice(
                    "lanes",
                    computed(
                        lambda read: tuple(d for d in (1, 2, 4, 8) if read["width"] % d == 0),
                        ("width",),
                    ),
                ),
            ),
            derived=(Derived("width", ("w",), even_width),),
        ),
    ),
)


def test_a_derived_value_that_refuses_reads_refused_with_its_code() -> None:
    REFUSED_INPUT.start({"w": 7}).check_answer("width")


@expected("refused-input-commit")
def test_a_commit_whose_domain_input_is_refused_is_accepted() -> None:
    # Nothing is waited on: no commitment can settle a refusal on the facts.
    REFUSED_INPUT.start({"w": 7}).check_change(Change(commit={"lanes": 1}))


@expected("refused-input-new-facts")
def test_new_facts_that_refuse_a_kept_choices_domain_input_are_accepted() -> None:
    REFUSED_INPUT.start({"w": 8}, {"lanes": 4}).check_change(Change(facts={"w": 7}))


def committed_over_odd_width() -> Pair:
    """``lanes`` committed at 1 over an odd width, the commit compared on the way (the
    questions below are asked once both sides accept it)."""
    pair = REFUSED_INPUT.start({"w": 7}).check_change(Change(commit={"lanes": 1}))
    assert pair is not None
    return pair


@expected("refused-input-state")
def test_a_choice_whose_domain_input_is_refused_reads_refused_through_it() -> None:
    # Kept, never read as a value: refused with the input's own finding.
    committed_over_odd_width().check_state("lanes")


@expected("refused-input-answer")
def test_a_choice_whose_domain_input_is_refused_is_queried_refused_through_it() -> None:
    committed_over_odd_width().check_answer("lanes")


@expected("refused-input-validity")
def test_a_choice_whose_domain_input_is_refused_refuses_the_configuration_through_it() -> None:
    # Blamed once, at the input: the one false piece's reason is ``width``'s refusal.
    committed_over_odd_width().check_validity()


def test_a_commit_inside_a_domain_over_an_even_width_is_accepted() -> None:
    REFUSED_INPUT.start({"w": 8}).check_change(Change(commit={"lanes": 4}))


def test_a_commit_outside_a_domain_over_an_even_width_is_refused() -> None:
    REFUSED_INPUT.start({"w": 8}).check_change(Change(commit={"lanes": 3}))


def test_a_choice_inside_a_domain_over_an_even_width_is_readable() -> None:
    REFUSED_INPUT.start({"w": 8}, {"lanes": 4}).check_state("lanes")


def test_a_choice_inside_a_domain_over_an_even_width_leaves_the_configuration_valid() -> None:
    REFUSED_INPUT.start({"w": 8}, {"lanes": 4}).check_validity()


def test_a_commit_whose_domain_input_waits_on_a_fact_is_kept_unchecked() -> None:
    REFUSED_INPUT.start().check_change(Change(commit={"lanes": 1}))


def test_a_choice_whose_domain_input_waits_on_a_fact_waits_on_it() -> None:
    REFUSED_INPUT.start({}, {"lanes": 1}).check_state("lanes")


# A domain reading a selector that forcing refuses: two cores, ``packed`` needing at most 64
# lanes and ``stub`` an even width, and ``lanes`` listing 1 and 2 under ``packed``, 1 under
# ``stub``. At 129 neither core is possible: the domain's input is refused by forcing, so it
# reads unresolved inside the own check. The commit is accepted and the choice kept
# unchecked, waiting on the selector; the configuration is refused through the selector.

FORCING_REFUSED_INPUT = Case(
    "a domain reading a selector forcing refuses",
    Space(
        ("width",),
        Scope(
            "",
            choices=(
                Choice(
                    "lanes",
                    computed(
                        lambda read: (1, 2) if read["compute"] == "packed" else (1,),
                        ("compute",),
                    ),
                ),
            ),
            selectors=(
                Selector(
                    "compute",
                    (
                        Scope(
                            "packed",
                            constraints=(
                                Constraint(
                                    "fits",
                                    ("width",),
                                    lambda read: read["width"] <= 64,
                                    "packed-width",
                                ),
                            ),
                        ),
                        Scope(
                            "stub",
                            constraints=(
                                Constraint(
                                    "even",
                                    ("width",),
                                    lambda read: read["width"] % 2 == 0,
                                    "stub-width",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


@expected("forcing-refused-input-commit")
def test_a_commit_whose_domain_input_forcing_refuses_is_kept_unchecked() -> None:
    FORCING_REFUSED_INPUT.start({"width": 129}).check_change(Change(commit={"lanes": 2}))


@expected("forcing-refused-input-new-facts")
def test_new_facts_that_leave_a_kept_choices_domain_input_no_case_keep_it_unchecked() -> None:
    # At 9 the stub core refuses an odd width, so packed is forced and 2 is in the domain.
    point = FORCING_REFUSED_INPUT.start({"width": 9}, {"lanes": 2})
    point.check_change(Change(facts={"width": 129}))


@expected("forcing-refused-input-state")
def test_a_choice_whose_domain_input_forcing_refuses_waits_on_it() -> None:
    pair = FORCING_REFUSED_INPUT.start({"width": 129}).check_change(Change(commit={"lanes": 2}))
    assert pair is not None  # asked once both sides accept the commit
    pair.check_state("lanes")


def test_a_domain_input_forcing_refuses_leaves_its_open_choice_waiting() -> None:
    FORCING_REFUSED_INPUT.start({"width": 129}).check_state("lanes")


# A domain reading a value that does not apply: ``size``, a value of the scope ``extra``,
# present when the fact ``on`` holds, and ``lanes`` over its divisors among 1, 2 and 4. With
# ``on`` false the input is inapplicable: membership cannot be evaluated, so the commit is
# accepted and the choice kept unchecked, waiting on the input, never refused as inactive
# (``lanes`` itself is present).

INAPPLICABLE_INPUT = Case(
    "a domain reading a value that does not apply",
    Space(
        ("on", "n"),
        Scope(
            "",
            choices=(
                Choice(
                    "lanes",
                    computed(
                        lambda read: tuple(d for d in (1, 2, 4) if read["extra.size"] % d == 0),
                        ("extra.size",),
                    ),
                ),
            ),
            children=(
                Scope(
                    "extra",
                    when=When(("on",), lambda read: bool(read["on"])),
                    derived=(Derived("size", ("n",), lambda read: read["n"]),),
                ),
            ),
        ),
    ),
    Spelling(types={"on": bool}),
)


@expected("inapplicable-input-commit")
def test_a_commit_whose_domain_input_does_not_apply_is_kept_unchecked() -> None:
    INAPPLICABLE_INPUT.start({"on": False, "n": 4}).check_change(Change(commit={"lanes": 2}))


@expected("inapplicable-input-new-facts")
def test_new_facts_under_which_a_kept_choices_domain_input_does_not_apply_keep_it() -> None:
    point = INAPPLICABLE_INPUT.start({"on": True, "n": 4}, {"lanes": 2})
    point.check_change(Change(facts={"on": False}))


@expected("inapplicable-input-state")
def test_a_choice_whose_domain_input_does_not_apply_waits_on_it() -> None:
    pair = INAPPLICABLE_INPUT.start({"on": False, "n": 4}).check_change(Change(commit={"lanes": 2}))
    assert pair is not None  # asked once both sides accept the commit
    pair.check_state("lanes")


def test_a_choice_whose_domain_input_applies_is_readable() -> None:
    INAPPLICABLE_INPUT.start({"on": True, "n": 4}, {"lanes": 2}).check_state("lanes")


def test_a_domain_input_that_does_not_apply_leaves_its_open_choice_waiting() -> None:
    INAPPLICABLE_INPUT.start({"on": False, "n": 4}).check_state("lanes")
