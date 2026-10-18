# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model, case by case: one test per question.

Each case is a small synthetic description, built in the model and in the engine
(``core.space.reference_harness``); each test brings one configuration to its question on
both sides and compares the answers. A question the engine answers differently at this
commit is a strict expected failure, the witness of its row in the acceptance matrix
(``reference_harness.matrix``), which names the milestone that removes it.

Where the semantics have been ruled since the model was written and the model does not
answer the ruling yet, the test states the ruled answer (``ruled=``).
"""

from __future__ import annotations

import pytest

from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Scope,
    Selector,
    Space,
    When,
    computed,
    listed,
    membership,
)
from core.space.reference_harness import (
    Case,
    ChangeForm,
    Pair,
    Refusal,
    Spelling,
    expected,
)

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


@expected("commit-under-refused-selector")
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


@expected("open-under-refused-selector")
def test_an_open_choice_under_a_refused_selector() -> None:
    refused_under().check_state("compute.stub.rows")


def test_a_rule_under_a_refused_selector_reads_refused_through_it() -> None:
    # Not evaluated; queried directly, it is refused with the selector's reasons.
    refused_under().check_rule("compute.packed.fits")


@expected("outside-domain-under-open-selector")
def test_a_value_outside_its_domain_under_an_open_selector_is_refused_at_commit() -> None:
    # At 12 both cores are possible: the selector is open, and 5 does not divide 12.
    # Ruled: refused, named, with its own reason. (The model forces the other core to keep
    # the value out of its domain, and refuses it as inactive.)
    own = frozenset({("domain-membership", "compute.packed.pe")})
    TYPED.start({"width": 12}).check_change(
        Change(commit={"compute.packed.pe": 5}),
        ruled=ChangeForm(False, refusals=frozenset({Refusal("compute.packed.pe", "named", own)})),
    )


def test_a_refused_selector_over_an_admitted_value_refuses_the_configuration() -> None:
    refused_under().check_validity()


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


@expected("commit-under-open-selector")
def test_a_choice_committed_under_an_open_selector_is_kept_unchecked() -> None:
    UNCHECKED.start({}, {"lanes": 12}).check_change(Change(commit={"compute.packed.pe": 3}))


@expected("facts-open-selector")
def test_new_facts_that_open_a_selector_keep_a_choice_under_it_unchecked() -> None:
    # At 12 both cores are possible: the selector is open, and ``pe`` (3 divides 12) waits.
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3})
    point.check_change(Change(facts={"width": 12}))
