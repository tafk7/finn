# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model, case by case: one test per question.

Each case is a small synthetic description, built in the model and in the engine
(``core.space.reference_harness``); each test brings one configuration to its question on
both sides and compares the answers. A question the engine answers differently is a
failure, unless a pending milestone's row in the acceptance matrix
(``reference_harness.matrix``) names the test as its witness.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
from core.space.reference_harness import Case, Pair, Spelling

# Each test runs with the readers index used for every cache, then for none: the same answers.


def divisors(extent: int) -> tuple[int, ...]:
    return tuple(value for value in range(1, extent + 1) if extent % value == 0)


# -- 1. typed membership ---------------------------------------------------------------------------
#
# Two cores share a width: ``packed`` takes at most 64 lanes, its ``pe`` a divisor of the
# width; ``stub`` takes an even width. A value is checked against its domain, a type check
# over the facts, wherever it sits; a rule of a case counts once the case is selected. At 129
# neither core's rule holds: the selector stays open, and whatever is under it waits.

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
PE = "compute.packed.pe"


def test_a_value_outside_its_domain_is_refused_at_commit() -> None:
    TYPED.start({"width": 8}).check_change(Change(commit={"x": 4}))


def test_a_value_outside_its_domain_under_an_open_selector_is_refused_by_its_type() -> None:
    # At 12 both cores' rules hold; 5 does not divide 12.
    TYPED.start({"width": 12}).check_change(Change(commit={PE: 5}))


def test_a_value_outside_its_domain_under_a_selector_no_case_of_which_holds_is_refused() -> None:
    # At 129 neither core's rule holds, which leaves the selector open: 2 does not divide 129.
    TYPED.start({"width": 129}).check_change(Change(commit={PE: 2}))


def test_a_value_outside_its_domain_under_an_unselected_case_is_refused_by_its_type() -> None:
    # The stub core is selected: ``pe`` is not in play, and its type is still checked.
    TYPED.start({"width": 12}, {"compute": "stub"}).check_change(Change(commit={PE: 5}))


def test_a_value_inside_its_domain_under_an_unselected_case_is_accepted_out_of_play() -> None:
    TYPED.start({"width": 12}, {"compute": "stub"}).check_change(Change(commit={PE: 3}))


def test_a_value_under_an_open_selector_waits_on_it() -> None:
    TYPED.start({"width": 12}, {PE: 3}).check_state(PE)


def test_a_stored_value_new_facts_put_outside_its_domain_refuses_the_change() -> None:
    # 3 divides 9, not 71: every stored value is checked against the new facts.
    TYPED.start({"width": 9}, {PE: 3}).check_change(Change(facts={"width": 71}))


def test_new_facts_no_core_of_which_holds_leave_the_selector_open_and_are_accepted() -> None:
    # 3 divides 99, which neither core takes: no piece is newly false, since neither case is
    # selected.
    TYPED.start({"width": 9}, {PE: 3}).check_change(Change(facts={"width": 99}))


def no_core() -> Pair:
    """``pe`` set at 3 under the open selector, then a width neither core takes."""
    return TYPED.start({"width": 9}, {PE: 3}).step(Change(facts={"width": 99}))


def test_a_value_under_a_selector_no_case_of_which_holds_still_waits_on_it() -> None:
    no_core().check_state(PE)


def test_a_rule_of_a_case_no_rule_of_which_holds_waits_on_its_selector() -> None:
    no_core().check_rule("compute.packed.fits")


def test_a_selector_no_case_of_which_holds_leaves_the_configuration_open() -> None:
    no_core().check_validity()


def test_selecting_a_case_whose_rule_the_facts_break_is_refused_naming_the_rule() -> None:
    no_core().check_change(Change(commit={"compute": "packed"}))


# Two cores both possible, the first's rule reading its own ``pe`` (whether it divides the
# width): a value outside ``pe``'s domain is refused by its type alone, whatever a rule reads.

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
                                    ("width", PE),
                                    lambda read: read["width"] % read[PE] == 0,
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


def test_a_value_outside_its_domain_a_cores_rule_reads_is_refused_by_its_type() -> None:
    # At 4 both cores are possible, and 3 does not divide 4.
    TWO_CORES.start({"width": 4}).check_change(Change(commit={PE: 3}))


# The same cores with a rule of the root reading packed's ``pe``: the rule waits while ``pe``
# waits on its selector, and selecting ``packed`` with ``pe`` over 8 makes it newly false, ``pe``
# standing in its way.

ROOT_RULE = Case(
    "a root rule reading a choice under the cores",
    Space(
        ("width",),
        Scope(
            "",
            selectors=(CORES,),
            constraints=(Constraint("small", (PE,), lambda read: read[PE] <= 8, "pe-small"),),
        ),
    ),
)


def test_a_value_a_root_rule_refuses_under_an_open_selector_is_accepted_waiting() -> None:
    ROOT_RULE.start({"width": 12}).check_change(Change(commit={PE: 12}))


def test_a_root_rule_reading_a_waiting_choice_waits_on_its_selector() -> None:
    ROOT_RULE.start({"width": 12}, {PE: 4}).check_rule("small")


def test_selecting_the_case_of_a_choice_a_root_rule_refuses_names_the_choice_standing() -> None:
    waiting = ROOT_RULE.start({"width": 12}).reach(Change(commit={PE: 12}))
    waiting.check_change(Change(commit={"compute": "packed"}))


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


def test_a_rule_under_an_unknown_guard_leaves_the_configuration_open() -> None:
    GUARDS.start({"budget": 5}, {"core": "small", "want": True}).check_validity()


def test_a_choice_under_an_unknown_guard_is_open_waiting_on_it() -> None:
    GUARDS.start({"budget": 5}).check_state("extra.size")


def test_a_guards_choice_whose_value_brings_a_false_rule_into_play_is_open() -> None:
    # At a budget of 5 ``costly``'s rule is false wherever it applies, and it applies when
    # ``want`` is false; nothing is given a value, so ``want`` stays open.
    GUARDS.start({"budget": 5}).check_state("want")


def test_a_guards_value_bringing_a_false_rule_into_play_is_not_possible() -> None:
    # False brings ``costly`` into play, whose rule refuses (``want`` decides its existence,
    # so it does not stand in its way); true opens ``spare.size``.
    GUARDS.start({"budget": 5}).check_possible("want")


def test_turning_a_guard_off_puts_the_choices_under_it_dormant() -> None:
    GUARDS.start({"budget": 25}, {"want": True, "spare.size": 2}).check_change(
        Change(commit={"want": False})
    )


# -- 3. exhaustive and empty domains ---------------------------------------------------------------
#
# A listed domain lists all its values, and every listed value is checked by its
# membership: ``x`` lists 1, 2 and 3 and rejects 2. ``w`` is known by membership only.
# ``z`` lists no value.

DOMAINS = Case(
    "exhaustive domains",
    Space(
        ("n",),
        Scope(
            "",
            choices=(
                Choice("x", listed(1, 2, 3, member=lambda value, read: value != 2)),
                Choice("w", membership(lambda value, read: value >= 2)),
            ),
        ),
    ),
)
EMPTY = Case("an empty domain", Space((), Scope("", choices=(Choice("z", listed()),))))


def test_a_listed_value_the_domain_rejects_is_not_among_the_possible_values() -> None:
    DOMAINS.start({"n": 6}).check_possible("x")


def test_a_listed_value_the_domain_rejects_is_refused_at_commit() -> None:
    DOMAINS.start({"n": 6}).check_change(Change(commit={"x": 2}))


def test_a_choice_known_by_membership_only_is_open() -> None:
    DOMAINS.start({"n": 6}).check_state("w")


def test_a_choice_known_by_membership_only_lists_no_possible_values() -> None:
    DOMAINS.start({"n": 6}).check_possible("w")


def test_a_value_outside_a_membership_only_domain_is_refused_at_commit() -> None:
    DOMAINS.start({"n": 6}).check_change(Change(commit={"w": 1}))


# A membership-only domain is never a piece of validity (§ 4, § 16.12; reading 13): with the
# fact its membership reads missing, or with its input a derived value refusing on the facts
# (a value committed there would be refused by assignment), the configuration is valid.

MEMBERSHIP_ONLY = Case(
    "a membership-only domain over a fact",
    Space(
        ("f",),
        Scope(
            "",
            choices=(Choice("c", membership(lambda value, read: 0 <= value <= read["f"], ("f",))),),
        ),
    ),
)
MEMBERSHIP_REFUSED = Case(
    "a membership-only domain over a derived value that refuses",
    Space(
        ("f",),
        Scope(
            "",
            choices=(Choice("c", membership(lambda value, read: value <= read["d"], ("d",))),),
            derived=(
                Derived("d", ("f",), lambda read: Refuse("big") if read["f"] > 2 else read["f"]),
            ),
        ),
    ),
)


def test_a_membership_only_domain_over_a_missing_fact_leaves_the_configuration_valid() -> None:
    MEMBERSHIP_ONLY.start({}).check_validity()


def test_a_membership_only_domain_over_a_refused_input_leaves_the_configuration_valid() -> None:
    MEMBERSHIP_REFUSED.start({"f": 5}).check_validity()


def test_a_choice_whose_list_is_empty_is_open() -> None:
    EMPTY.start().check_state("z")


def test_an_empty_list_refuses_the_configuration() -> None:
    EMPTY.start().check_validity()


def test_a_value_under_an_empty_list_is_outside_its_domain() -> None:
    EMPTY.start().check_change(Change(commit={"z": 1}))


# -- 4. waiting choices --------------------------------------------------------------------------
#
# The two cores again, their width a choice ``lanes`` (9 or 12): the stub core refuses 9.
# ``doubled`` is a value computed from packed's ``pe``. A choice set under the open selector
# waits on it; selecting its case brings it into play, selecting the other puts it dormant.

WAITING_CORE = Case(
    "a choice waiting on its selector",
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
            derived=(Derived("doubled", (PE,), lambda read: 2 * read[PE]),),
        ),
    ),
)


def waiting() -> Pair:
    """``pe`` set at 3 under the open selector, ``lanes`` at 12."""
    return WAITING_CORE.start({}, {"lanes": 12, PE: 3})


def test_a_choice_set_under_an_open_selector_is_accepted_waiting() -> None:
    WAITING_CORE.start({}, {"lanes": 12}).check_change(Change(commit={PE: 3}))


def test_a_waiting_choice_waits_on_its_selector() -> None:
    waiting().check_state(PE)


def test_a_waiting_choice_reads_unresolved() -> None:
    waiting().check_answer(PE)


def test_a_value_computed_from_a_waiting_choice_is_unresolved() -> None:
    waiting().check_answer("doubled")


def test_a_waiting_choice_alone_leaves_the_configuration_valid() -> None:
    # Its list has values and no rule reads it: nothing is unknown.
    waiting().check_validity()


def test_a_waiting_choice_is_set_once_its_case_is_selected() -> None:
    waiting().step(Change(commit={"compute": "packed"})).check_state(PE)


def test_selecting_the_other_case_puts_a_waiting_choice_dormant() -> None:
    waiting().check_change(Change(commit={"compute": "stub"}))


def test_selecting_a_case_whose_rule_a_stored_choice_breaks_names_it_standing() -> None:
    # The stub core refuses 9 lanes: ``even`` reads ``lanes``, stored and not named.
    WAITING_CORE.start({}, {"lanes": 9}).check_change(Change(commit={"compute": "stub"}))


def test_the_possible_cases_of_a_selector_name_what_each_moves() -> None:
    WAITING_CORE.start({}, {"lanes": 12, PE: 3}).check_possible("compute")


# A domain computed from a fact the configuration does not give yet: a value set to it
# waits on the fact, as one under an open selector waits on the selector.

ON_A_FACT = Case(
    "a domain waiting on a fact",
    Space(
        ("width",),
        Scope(
            "",
            choices=(Choice("pe", computed(lambda read: divisors(read["width"]), ("width",))),),
        ),
    ),
)


def test_a_choice_set_while_its_domain_waits_on_a_fact_is_accepted_waiting() -> None:
    ON_A_FACT.start().check_change(Change(commit={"pe": 3}))


def test_a_choice_whose_domain_waits_on_a_fact_waits_on_it() -> None:
    ON_A_FACT.start({}, {"pe": 3}).check_state("pe")


def test_the_fact_a_waiting_choice_lacks_checks_its_type_when_it_comes() -> None:
    ON_A_FACT.start({}, {"pe": 3}).check_change(Change(facts={"width": 8}))


# A rule over such a choice: the choice waits on the fact, never on itself (reading 14), so
# what reads it waits on the fact too, not on the choice.

FACT_RULE = Case(
    "a rule over a choice waiting on a fact",
    Space(
        ("width",),
        Scope(
            "",
            choices=(Choice("pe", computed(lambda read: divisors(read["width"]), ("width",))),),
            constraints=(Constraint("few", ("pe",), lambda read: read["pe"] <= 4, "many-pe"),),
        ),
    ),
)


def test_a_rule_over_a_choice_waiting_on_a_fact_waits_on_the_fact() -> None:
    FACT_RULE.start({}, {"pe": 3}).check_rule("few")


# A stored choice of a selector's case that a rule of the case refuses: set under the open
# selector, it waits, and the rule waits with it. Selecting its case makes the rule newly false.

CASE_RULE = Case(
    "a case's rule and a waiting choice of the case",
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


def test_setting_a_choice_its_cases_rule_refuses_under_an_open_selector_is_accepted() -> None:
    CASE_RULE.start({"limit": 1}).check_change(Change(commit={"core.wide.lanes": 2}))


def test_a_case_whose_rule_refuses_a_waiting_choice_of_it_is_not_possible() -> None:
    # ``plain`` puts ``lanes`` dormant; ``wide`` brings it into play, where ``small`` refuses
    # it, ``lanes`` standing in the way.
    CASE_RULE.start({"limit": 1}).reach(Change(commit={"core.wide.lanes": 2})).check_possible(
        "core"
    )


def test_clearing_a_selector_leaves_its_cases_choice_waiting() -> None:
    point = CASE_RULE.start({"limit": 2}, {"core": "wide", "core.wide.lanes": 2})
    point.step(Change(clear=("core",))).check_state("core.wide.lanes")


# -- 5. a domain reading a derived value over facts ------------------------------------------------
#
# ``width`` refuses an odd ``w`` (``bad-width``), and ``lanes``'s domain is the divisors of
# ``width`` among 1, 2, 4 and 8. Membership reads facts only, through ``width``: a value
# whose membership reads a refused input is not of its type, and its list cannot be made.


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


def test_a_value_whose_membership_reads_a_refused_input_is_refused_at_commit() -> None:
    REFUSED_INPUT.start({"w": 7}).check_change(Change(commit={"lanes": 1}))


def test_new_facts_that_refuse_a_stored_choices_membership_input_are_refused() -> None:
    REFUSED_INPUT.start({"w": 8}, {"lanes": 4}).check_change(Change(facts={"w": 7}))


def test_a_list_refused_through_its_input_refuses_the_configuration_at_the_input() -> None:
    REFUSED_INPUT.start({"w": 7}).check_validity()


def test_a_commit_inside_a_domain_over_an_even_width_is_accepted() -> None:
    REFUSED_INPUT.start({"w": 8}).check_change(Change(commit={"lanes": 4}))


def test_a_commit_outside_a_domain_over_an_even_width_is_refused() -> None:
    REFUSED_INPUT.start({"w": 8}).check_change(Change(commit={"lanes": 3}))


def test_a_choice_inside_a_domain_over_an_even_width_is_set() -> None:
    REFUSED_INPUT.start({"w": 8}, {"lanes": 4}).check_state("lanes")


def test_a_choice_inside_a_domain_over_an_even_width_leaves_the_configuration_valid() -> None:
    REFUSED_INPUT.start({"w": 8}, {"lanes": 4}).check_validity()


def test_a_commit_whose_membership_input_waits_on_a_fact_is_accepted_waiting() -> None:
    REFUSED_INPUT.start().check_change(Change(commit={"lanes": 1}))


def test_a_choice_whose_membership_input_waits_on_a_fact_waits_on_it() -> None:
    REFUSED_INPUT.start({}, {"lanes": 1}).check_state("lanes")


# A domain reading a derived value over facts in a guarded scope: ``size``, a value of
# ``extra``, present when the fact ``on`` holds, and ``lanes`` over its divisors among 1, 2
# and 4. With ``on`` false the input does not exist for these facts: it is refused
# (``domain-input-absent``, § 16.17), not unresolved.

GUARDED_INPUT = Case(
    "a domain reading a value in a guarded scope",
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


def test_a_commit_whose_membership_input_does_not_apply_is_refused_naming_it() -> None:
    GUARDED_INPUT.start({"on": False, "n": 4}).check_change(Change(commit={"lanes": 2}))


def test_an_unset_choice_whose_membership_input_does_not_apply_refuses_the_configuration() -> None:
    GUARDED_INPUT.start({"on": False, "n": 4}).check_validity()


def test_a_choice_whose_membership_input_is_in_play_is_set() -> None:
    GUARDED_INPUT.start({"on": True, "n": 4}, {"lanes": 2}).check_state("lanes")


def test_new_facts_putting_a_stored_choices_membership_input_out_of_play_are_refused() -> None:
    point = GUARDED_INPUT.start({"on": True, "n": 4}, {"lanes": 2})
    point.check_change(Change(facts={"on": False}))
