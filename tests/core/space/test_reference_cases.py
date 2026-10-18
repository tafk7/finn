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
    Scope,
    Selector,
    Space,
    When,
    computed,
    listed,
)
from core.space.reference_harness import (
    Case,
    ChangeForm,
    RuleForm,
    Spelling,
    StateForm,
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
    # 3 divides 129: its own check holds, and the configuration reads refused through the
    # selector. Ruled: accepted and readable, never unchecked (unchecked is for an open
    # selector only).
    TYPED.start({"width": 129}).check_change(
        Change(commit={"compute.packed.pe": 3}), ruled=ChangeForm(True)
    )


def test_a_retained_value_inside_its_domain_under_a_selector_new_facts_refuse_is_kept() -> None:
    # At 99 neither core is possible (past 64 lanes, odd), and 3 divides 99. Ruled: kept,
    # readable, so nothing is kept unchecked.
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3})
    point.check_change(Change(facts={"width": 99}), ruled=ChangeForm(True))


@expected("readable-under-refused-selector")
def test_a_value_inside_its_domain_under_a_refused_selector_is_readable() -> None:
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3}).step(Change(facts={"width": 99}))
    point.check_state("compute.packed.pe", ruled=StateForm("readable", 3))


def test_a_rule_under_a_refused_selector_reads_refused_through_it() -> None:
    # Ruled: not evaluated; read, it is refused with the selector's reasons.
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3}).step(Change(facts={"width": 99}))
    refused = frozenset({("decision-no-viable-case", "compute")})
    point.check_rule("compute.packed.fits", ruled=RuleForm("false", codes=refused))


def test_a_refused_selector_over_a_readable_value_refuses_the_configuration() -> None:
    point = TYPED.start({"width": 9}, {"compute.packed.pe": 3}).step(Change(facts={"width": 99}))
    point.check_validity()


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
