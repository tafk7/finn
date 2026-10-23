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
    Members,
    Refuse,
    Scope,
    Selector,
    Space,
    View,
    When,
    computed,
    listed,
    membership,
    nothing,
)
from core.space.reference_harness import Case, Pair, Spelling, check_definition, expected


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


def test_a_stored_value_new_facts_put_outside_its_domain_is_dropped() -> None:
    # 3 divides 9, not 71: every stored value is checked against the new facts, and one
    # outside is dropped and reported, never refusing the change (§ 16.24).
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
# present when the choice ``on`` holds, and its rule needs a budget of 20; ``spare`` is
# present when the choice ``want`` holds, ``costly`` when it does not, and ``costly``'s
# rule needs a budget of 20 too. (``on`` was a fact, left out for an unknown guard; facts
# are always known now, SZ100 (5), so a guard waits on an open choice.)

GUARDS = Case(
    "guards",
    Space(
        ("budget",),
        Scope(
            "",
            choices=(Choice("want", listed(False, True)), Choice("on", listed(False, True))),
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
)


def test_a_rule_under_an_unselected_case_does_not_apply() -> None:
    GUARDS.start({"budget": 5}, {"core": "small"}).check_rule("core.big.limit")


def test_a_rule_under_an_unselected_case_counts_as_true() -> None:
    point = GUARDS.start({"budget": 5}, {"core": "small", "want": True, "on": False})
    point.check_validity()


def test_a_rule_under_a_false_guard_does_not_apply() -> None:
    GUARDS.start({"budget": 5}, {"on": False}).check_rule("extra.cap")


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


# A membership-only domain is never a piece of validity (§ 4, § 16.12; reading 13): with its
# input a derived value refusing on the facts (a value committed there would be refused by
# assignment), the configuration is valid. (Its twin over a missing fact went with SZ100
# (5): facts are always known.)

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


def test_new_facts_that_refuse_a_stored_choices_membership_input_drop_it() -> None:
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


def test_new_facts_putting_a_stored_choices_membership_input_out_of_play_drop_it() -> None:
    point = GUARDED_INPUT.start({"on": True, "n": 4}, {"lanes": 2})
    point.check_change(Change(facts={"on": False}))


# -- 6. Members: standing is what the origin's evaluation read (SZ100 (6)) ----------------------
#
# RUNNING § 2.2's House: the annex ``g`` (present while ``x >= 2``) and each room of ``s``
# state a derived value ``e`` of their own choice, exported to the root's ``m``. The
# scullery refuses ``q = 3``; the annex's ``e`` reads its shed's ``z``, inapplicable while
# the shed (present while ``g.c >= 2``) is absent. ``small`` asks the rooms' sum to be at
# most 4; ``fitted`` reads ``x``, the annex's guard's read, beside ``m`` (§ 16.22).


def rooms(r: Mapping[str, Any]) -> int:
    return sum(value for _, value in r["m"])


def house(*rules: Constraint) -> Case:
    shed = Scope(
        "shed",
        when=When(("g.c",), lambda read: read["g.c"] >= 2),
        choices=(Choice("z", listed(1, 2)),),
    )
    annex = Scope(
        "g",
        when=When(("x",), lambda read: read["x"] >= 2),
        choices=(Choice("c", listed(1, 2, 3)),),
        derived=(Derived("e", ("g.c", "g.shed.z"), lambda read: read["g.c"] + read["g.shed.z"]),),
        children=(shed,),
    )
    kitchen = Scope(
        "kitchen",
        choices=(Choice("p", listed(1, 2, 3)),),
        derived=(Derived("e", ("s.kitchen.p",), lambda read: 2 * read["s.kitchen.p"]),),
    )
    scullery = Scope(
        "scullery",
        choices=(Choice("q", listed(1, 2, 3)),),
        derived=(
            Derived(
                "e",
                ("s.scullery.q",),
                lambda read: Refuse("wet") if read["s.scullery.q"] == 3 else 1,
            ),
        ),
    )
    return Case(
        "the house and its rooms",
        Space(
            (),
            Scope(
                "",
                choices=(Choice("x", listed(1, 2, 3)),),
                selectors=(Selector("s", (kitchen, scullery)),),
                children=(annex,),
                members=(Members("m", ("g.e", "s.kitchen.e", "s.scullery.e")),),
                constraints=rules,
            ),
        ),
    )


HOUSE = house()
SMALL = house(Constraint("small", ("m",), lambda read: rooms(read) <= 4, "big"))
FITTED = house(Constraint("fitted", ("x", "m"), lambda read: read["x"] + rooms(read) <= 7, "big"))


def test_members_skips_an_entry_whose_scope_is_absent() -> None:
    HOUSE.start({}, {"x": 1, "g.c": 3, "s": "kitchen", "s.kitchen.p": 2}).check_answer("m")


def test_members_skips_a_present_entry_answering_inapplicable() -> None:
    pair = HOUSE.start({}, {"x": 2, "g.c": 1, "s": "kitchen", "s.kitchen.p": 2})
    pair.check_answer("g.e")
    pair.check_answer("m")


def test_members_answers_its_entries_with_a_value_by_entry() -> None:
    stored = {"x": 2, "g.c": 2, "g.shed.z": 1, "s": "kitchen", "s.kitchen.p": 1}
    HOUSE.start({}, stored).check_answer("m")


def test_members_refuses_through_a_refused_entry_even_while_another_waits() -> None:
    pair = HOUSE.start({}, {"x": 2, "s": "scullery", "s.scullery.q": 3})
    pair.check_answer("g.e")
    pair.check_answer("m")


def test_members_waits_on_an_entry_that_waits() -> None:
    HOUSE.start({}, {"x": 2, "s": "scullery", "s.scullery.q": 2}).check_answer("m")


def test_a_dormant_entrys_choice_never_stands() -> None:
    pair = SMALL.start({}, {"x": 1, "g.c": 3, "s": "kitchen"})
    pair.check_change(Change(commit={"s.kitchen.p": 3}))


def test_a_present_entrys_choice_stands() -> None:
    pair = SMALL.start({}, {"x": 2, "g.c": 2, "g.shed.z": 1, "s": "kitchen"})
    pair.check_change(Change(commit={"s.kitchen.p": 2}))
    pair.check_possible("s.kitchen.p")


def test_a_choice_read_beside_members_stands_though_it_decides_an_entrys_existence() -> None:
    stored = {"x": 2, "g.c": 1, "g.shed.z": 1, "s": "kitchen", "s.kitchen.p": 2}
    FITTED.start({}, stored).check_change(Change(commit={"s.kitchen.p": 3, "g.c": 2}))
    raised = FITTED.start({}, {**stored, "g.c": 2, "s.kitchen.p": 1})
    raised.check_change(Change(commit={"x": 3}))
    raised.check_possible("x")


# -- 7. views are read out (SZ100 (3)) ------------------------------------------------------------
#
# A stage presents its storage as a view; the parent's rule reads it. Only a view reads a
# view, so the model refuses the definition (``view-read-in``); the public API builds it
# and answers, until T4b.


def storage_stage(name: str) -> Scope:
    return Scope(
        name,
        choices=(Choice("depth", listed(1, 2, 4)),),
        views=(View("storage", (f"{name}.depth",), lambda read: 8 * read[f"{name}.depth"]),),
    )


READ_IN = Space(
    (),
    Scope(
        "",
        children=(storage_stage("first"), storage_stage("second")),
        constraints=(
            Constraint(
                "fits",
                ("first.storage", "second.storage"),
                lambda read: read["first.storage"] + read["second.storage"] <= 40,
                "too-deep",
            ),
        ),
    ),
)


@expected("view-read-in")
def test_a_rule_reading_a_view_is_no_definition() -> None:
    check_definition("a rule reading its children's views", READ_IN)


# -- 8. an optional fact left out is None (SZ100 (5)) ---------------------------------------------
#
# ``width`` is optional: left out, it reads None, a known value, and ``lanes`` lists 1 and 2
# then; given None, it is the fact left out (one design space).

OPTIONAL = Case(
    "an optional fact left out",
    Space(
        ("width",),
        Scope(
            "",
            derived=(
                Derived(
                    "lanes",
                    ("width",),
                    lambda read: 2 if read["width"] is None else read["width"] // 4,
                ),
            ),
        ),
        optional=("width",),
    ),
)


def test_an_optional_fact_left_out_reads_none() -> None:
    OPTIONAL.start().check_answer("lanes")


def test_an_optional_fact_given_reads_its_value() -> None:
    OPTIONAL.start({"width": 8}).check_answer("lanes")


# -- 9. a refused guard counts once (reading 8) ---------------------------------------------------
#
# ``heated`` refuses at gate 1, and ``plant``, present when it holds, holds one selector,
# ``heating``, of two cases. Under the refused guard the model counts one guard piece, owned by
# ``plant``: the cases under it add nothing. Written as the selector's own guard
# (``Decision(..., when=)``, ``Spelling.guarded_selectors``), the guard and the cases' guards
# are nodes of one Space, and the engine counts the piece at the highest refused guard of its
# chain, one (it counted 1 + 2 before T4a); written as a guarded child, the cases are another
# Space's, and it counts one.


def heated(read: Mapping[str, Any]) -> object:
    return Refuse("gate-refused") if read["gate"] == 1 else True


HEATED_PLANT = Space(
    ("gate",),
    Scope(
        "",
        derived=(Derived("heated", ("gate",), heated),),
        children=(
            Scope(
                "plant",
                when=When(("heated",), lambda read: bool(read["heated"])),
                selectors=(
                    Selector(
                        "heating",
                        (
                            Scope("boiler", choices=(Choice("stages", listed(1, 2, 3)),)),
                            Scope("heat_pump", choices=(Choice("cop", listed(2, 3, 4)),)),
                        ),
                    ),
                ),
            ),
        ),
    ),
)
GUARDED_SELECTOR = Case(
    "a chosen selector under a refused when",
    HEATED_PLANT,
    Spelling(types={"heated": bool}, guarded_selectors=frozenset({"plant"})),
)
GUARDED_CHILD = Case(
    "a selector in a child under a refused when", HEATED_PLANT, Spelling(types={"heated": bool})
)


def test_a_chosen_selector_under_a_refused_when_is_one_piece() -> None:
    GUARDED_SELECTOR.start({"gate": 1}).check_validity()


def test_a_selector_in_a_child_under_a_refused_when_is_one_piece() -> None:
    GUARDED_CHILD.start({"gate": 1}).check_validity()


def test_a_chosen_selector_under_a_refused_when_puts_its_cases_out_of_play() -> None:
    point = GUARDED_SELECTOR.start({"gate": 1})
    point.check_state("plant.heating")
    point.check_state("plant.heating.boiler.stages")


def test_a_chosen_selector_under_a_when_that_holds_is_set_as_any_selector() -> None:
    point = GUARDED_SELECTOR.start({"gate": 2}, {"plant.heating": "boiler"})
    point.check_state("plant.heating.boiler.stages")
    point.check_validity()


# A refused guard below an unknown one: ``s`` is open, its case ``a`` holds ``g``, present when
# ``d`` is at least 1, and ``d`` refuses an odd ``f``, read by nothing else. The model leaves
# ``g``'s piece unknown while ``a``'s presence is (reading 8): validity is open, waiting on
# ``s``. So does the engine (it answered valid before T4a, SZ111).


def even(read: Mapping[str, Any]) -> object:
    return Refuse("d-odd") if read["f"] % 2 else read["f"]


IN_AN_OPEN_CASE = Case(
    "a refused when in a case of an open selector",
    Space(
        ("f",),
        Scope(
            "",
            derived=(Derived("d", ("f",), even),),
            selectors=(
                Selector(
                    "s",
                    (
                        Scope(
                            "a",
                            choices=(Choice("p", listed(1, 2)),),
                            children=(
                                Scope(
                                    "g",
                                    when=When(("d",), lambda read: read["d"] >= 1),
                                    choices=(Choice("q", listed(1, 2)),),
                                ),
                            ),
                        ),
                        Scope("b", choices=(Choice("r", listed(1, 2)),)),
                    ),
                ),
            ),
        ),
    ),
)


def test_a_refused_when_in_a_case_of_an_open_selector_leaves_validity_open() -> None:
    IN_AN_OPEN_CASE.start({"f": 1}).check_validity()


def test_a_refused_when_in_an_unselected_case_does_not_apply() -> None:
    IN_AN_OPEN_CASE.start({"f": 1}, {"s": "b"}).check_validity()


# -- 10. a choice its facts alone rule out is in no list (SZ113) -----------------------------------
#
# ``wide`` is present when the fact ``mode`` is 1; ``gated`` when ``checked``, a derived value
# over ``mode`` refusing at 2; ``opted`` when the choice ``pe`` is 2. Under the facts alone
# (every choice unknown) ``wide`` is false at any other mode, so ``wide.lanes`` is
# pruned: in no list, unless it is stored (SZ119 (5)). A refused guard and a guard over a choice
# are not false, so ``gated.x`` and ``opted.y`` stay listed. The engine prunes as the model does
# (T4c, its row ``facts-pruning`` removed).


def checked(read: Mapping[str, Any]) -> object:
    return Refuse("mode-bad") if read["mode"] == 2 else 1


PRUNED = Case(
    "a choice its facts rule out",
    Space(
        ("mode",),
        Scope(
            "",
            choices=(Choice("pe", listed(1, 2)),),
            derived=(Derived("checked", ("mode",), checked),),
            children=(
                Scope(
                    "wide",
                    when=When(("mode",), lambda read: read["mode"] == 1),
                    choices=(Choice("lanes", listed(2, 4)),),
                ),
                Scope(
                    "gated",
                    when=When(("checked",), lambda read: read["checked"] == 1),
                    choices=(Choice("x", listed(1, 2)),),
                ),
                Scope(
                    "opted",
                    when=When(("pe",), lambda read: read["pe"] == 2),
                    choices=(Choice("y", listed(1, 2)),),
                ),
            ),
        ),
    ),
)


def test_a_choice_the_facts_rule_out_is_in_no_list() -> None:
    pair = PRUNED.start({"mode": 0})
    assert pair.solution().pruned() == ("wide.lanes",)
    pair.check_listed()


def test_a_change_of_facts_prunes_a_choice() -> None:
    pair = PRUNED.start({"mode": 1})
    assert pair.solution().pruned() == ()
    pruned = pair.reach(Change(facts={"mode": 0}))
    assert pruned.solution().pruned() == ("wide.lanes",)
    assert "wide.lanes" not in pruned.solution().states()
    pruned.check_listed()
    pruned.check_state("wide.lanes")


def test_a_change_of_facts_unprunes_a_choice_which_comes_back_open() -> None:
    pair = PRUNED.start({"mode": 0})
    assert pair.solution().pruned() == ("wide.lanes",)
    unpruned = pair.reach(Change(facts={"mode": 1}))
    assert unpruned.solution().pruned() == ()
    unpruned.check_listed()
    unpruned.check_state("wide.lanes")


def test_a_stored_choice_is_never_pruned_by_a_change_of_facts() -> None:
    pair = PRUNED.start({"mode": 1}, {"wide.lanes": 4})
    narrow = pair.reach(Change(facts={"mode": 0}))
    assert narrow.solution().pruned() == ()
    assert "wide.lanes" in narrow.solution().dormant()
    narrow.check_listed()
    narrow.check_state("wide.lanes")


def test_a_stored_choice_is_never_pruned_by_a_change_of_choices() -> None:
    pair = PRUNED.start({"mode": 0})
    assert pair.solution().pruned() == ("wide.lanes",)
    stored = pair.reach(Change(commit={"wide.lanes": 2}))
    assert stored.solution().pruned() == ()
    assert "wide.lanes" in stored.solution().dormant()
    stored.check_listed()
    stored.check_state("wide.lanes")


def test_a_guard_the_facts_refuse_or_a_choice_decides_is_no_pruning() -> None:
    pair = PRUNED.start({"mode": 2})
    assert pair.solution().pruned() == ("wide.lanes",)
    assert {"gated.x", "opted.y"} <= set(pair.solution().listed())
    pair.check_listed()
    pair.check_state("gated.x")
    pair.check_state("opted.y")


# A selector picked by a derived value over facts alone (lane 3's ``input_chain``) is the rule's
# widest reach: under the facts alone it names one case, so the choices of every other case are
# pruned. ``adapter`` is picked by ``chain`` over ``mode`` (0: ``fifo``, 1: ``reg``, 2 refuses);
# ``other`` by ``side`` over the choice ``pe``, which waits under the facts alone, so its cases stay
# listed, as ``adapter``'s do where ``chain`` refuses.


def chain_by_mode(read: Mapping[str, Any]) -> object:
    return Refuse("chain-refused") if read["mode"] == 2 else ("fifo", "reg")[read["mode"]]


PICKED = Case(
    "a selector picked by facts",
    Space(
        ("mode",),
        Scope(
            "",
            choices=(Choice("pe", listed(1, 2)),),
            derived=(
                Derived("chain", ("mode",), chain_by_mode),
                Derived("side", ("pe",), lambda read: ("u", "v")[read["pe"] - 1]),
            ),
            selectors=(
                Selector(
                    "adapter",
                    (
                        Scope("fifo", choices=(Choice("depth", listed(2, 4)),)),
                        Scope("reg", choices=(Choice("stages", listed(1, 2)),)),
                        nothing("direct"),
                    ),
                    by="chain",
                ),
                Selector(
                    "other",
                    (
                        Scope("u", choices=(Choice("a", listed(1, 2)),)),
                        Scope("v", choices=(Choice("b", listed(1, 2)),)),
                    ),
                    by="side",
                ),
            ),
        ),
    ),
)


def test_a_selector_picked_by_facts_prunes_the_cases_it_does_not_pick() -> None:
    pair = PICKED.start({"mode": 0})
    assert pair.solution().pruned() == ("adapter.reg.stages",)
    assert {"adapter.fifo.depth", "other.u.a", "other.v.b"} <= set(pair.solution().listed())
    pair.check_listed()
    pair.check_state("adapter.reg.stages")
    pair.check_state("adapter.fifo.depth")


def test_a_selector_picked_by_facts_that_refuse_prunes_no_case() -> None:
    pair = PICKED.start({"mode": 2})
    assert pair.solution().pruned() == ()
    pair.check_listed()
    pair.check_state("adapter.reg.stages")


def test_an_optional_fact_given_none_is_the_one_left_out() -> None:
    OPTIONAL.check_one_space({"width": None}, {})


# A guard over a derived value over facts alone is evaluated, never read as a presence (lane
# 4's ``offered``, computed from the channel's sides; E2's ``valued``, over ``contents``):
# ``offered`` reads the optional fact ``offer``, ``mover`` is present where it holds. Left out
# (None), the guard reads false and ``mover.burst`` is pruned, still answered by name; given,
# it is listed. Set by name where pruned, it is stored, dormant and listed (SZ119 (5)).

OFFERED = Case(
    "a derived guard over an optional fact",
    Space(
        ("offer",),
        Scope(
            "",
            derived=(Derived("offered", ("offer",), lambda read: read["offer"] is not None),),
            children=(
                Scope(
                    "mover",
                    when=When(("offered",), lambda read: read["offered"]),
                    choices=(Choice("burst", listed(1, 2)),),
                ),
            ),
        ),
        optional=("offer",),
    ),
    Spelling(types={"offered": bool}),
)


def test_a_derived_guard_over_facts_alone_prunes_where_it_reads_false() -> None:
    pair = OFFERED.start()
    assert pair.solution().pruned() == ("mover.burst",)
    pair.check_listed()
    pair.check_state("mover.burst")
    pair.check_possible("mover.burst")
    stored = pair.reach(Change(commit={"mover.burst": 2}))
    assert stored.solution().pruned() == ()
    assert "mover.burst" in stored.solution().dormant()
    stored.check_listed()
    stored.check_state("mover.burst")
    offered = OFFERED.start({"offer": 3})
    assert offered.solution().pruned() == ()
    offered.check_listed()
    offered.check_state("mover.burst")


def test_unsetting_an_optional_fact_prunes_what_its_derived_guard_held() -> None:
    pair = OFFERED.start({"offer": 3})
    unset = pair.reach(Change(unset=("offer",)))
    assert unset.solution().pruned() == ("mover.burst",)
    unset.check_listed()
    unset.check_state("mover.burst")
