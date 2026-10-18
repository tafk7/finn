# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model: rules between choices, one test per question.

Cases 5 to 8 of the reference cases (cases 1 to 4 are ``test_reference_cases``): a linking
rule, a requirement, forcing conflicts and the one trial rule. Each rule is a constraint in
the model; the engine spells it as it is authored at this commit (``Spelling`` beside each
case), so the acceptance matrix shows what a later milestone changes in each spelling.
"""

from __future__ import annotations

import itertools

import pytest

from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Scope,
    Selector,
    Space,
    computed,
    listed,
)
from core.space.reference_harness import (
    Case,
    Pair,
    Requirement,
    Spelling,
    expected,
)

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")


def reach(pair: Pair, edit: Change) -> Pair:
    """``edit`` compared, then its successor: a change the model accepts and the engine
    refuses is a disagreement, so a question about its configuration cannot be asked."""
    __tracebackhide__ = True
    successor = pair.check_change(edit)
    assert successor is not None, "the model refuses the change: the test is written wrongly"
    return successor


# -- 5. a linking rule as a constraint -------------------------------------------------------------
#
# ``b <= a``, over two choices of one Space: a constraint of the Space (in its admission group
# today). Beside it, the same bound as ``b``'s domain, computed from ``a``: not a rule but
# ``b``'s type check, its own check (unchecked while ``a`` is open, false refusing a named
# commit or a reached retained ``b``).

LINKED_SPACE = Space(
    (),
    Scope(
        "",
        choices=(Choice("a", listed(1, 2, 3)), Choice("b", listed(1, 2, 3))),
        constraints=(
            Constraint("linked", ("a", "b"), lambda read: read["b"] <= read["a"], "b-above-a"),
        ),
    ),
)
LINKED = Case("a linking rule", LINKED_SPACE)
LINKED_DOMAIN = Case(
    "a domain reading another choice",
    Space(
        (),
        Scope(
            "",
            choices=(
                Choice("a", listed(1, 2, 3)),
                Choice(
                    "b",
                    computed(lambda read: tuple(v for v in (1, 2, 3) if v <= read["a"]), ("a",)),
                ),
            ),
        ),
    ),
)


def test_a_linking_rule_leaves_the_other_choice_the_cases_it_holds_for() -> None:
    LINKED.start({}, {"b": 2}).check_state("a")


def test_a_linking_rule_never_refuses_a_commit() -> None:
    LINKED.start({}, {"b": 2}).check_change(Change(commit={"a": 1}))


def test_a_linking_rule_refuses_the_configuration_naming_itself() -> None:
    reach(LINKED.start({}, {"b": 2}), Change(commit={"a": 1})).check_validity()


def test_a_linking_rule_forces_the_open_choice() -> None:
    LINKED.start({}, {"a": 1}).check_state("b")


def test_a_domain_reading_another_choice_takes_a_commit_before_the_choice_it_reads() -> None:
    LINKED_DOMAIN.start().check_change(Change(commit={"b": 2}))


def test_a_domain_reading_another_choice_refuses_a_commit_that_puts_a_kept_choice_outside() -> None:
    # ``b`` of 2 is kept unchecked while ``a`` is open; ``a`` of 1 leaves 2 outside ``b``'s
    # domain: the change is refused, reaching ``b``.
    LINKED_DOMAIN.start({}, {"b": 2}).check_change(Change(commit={"a": 1}))


def test_a_domain_reading_another_choice_forces_its_one_value_left() -> None:
    LINKED_DOMAIN.start({}, {"a": 1}).check_state("b")


# -- 6. a requirement as a constraint --------------------------------------------------------------
#
# ``style`` ``big`` needs a capability, the fact ``has_big``: a constraint over the choice
# and the fact, spelled ``requires(has_big, cases=("big",))`` on the domain today.

REQUIRED = Case(
    "a requirement",
    Space(
        ("has_big",),
        Scope(
            "",
            choices=(Choice("style", listed("small", "big")),),
            constraints=(
                Constraint(
                    "big_needs_support",
                    ("style", "has_big"),
                    lambda read: read["style"] != "big" or bool(read["has_big"]),
                    "big-unsupported",
                ),
            ),
        ),
    ),
    Spelling(
        rules={"big_needs_support": Requirement("style", "has_big", ("big",))},
        types={"has_big": bool},
    ),
)


def test_a_case_needing_a_capability_the_facts_lack_is_not_possible() -> None:
    REQUIRED.start({"has_big": False}).check_state("style")


def test_a_case_needing_an_unknown_capability_is_possible() -> None:
    REQUIRED.start().check_state("style")


def test_committing_a_case_needing_a_missing_capability_is_accepted() -> None:
    REQUIRED.start({"has_big": False}).check_change(Change(commit={"style": "big"}))


def test_committing_a_case_needing_a_missing_capability_refuses_the_configuration() -> None:
    reach(REQUIRED.start({"has_big": False}), Change(commit={"style": "big"})).check_validity()


# -- 7. forcing conflicts --------------------------------------------------------------------------
#
# The pair: ``x`` and ``y`` each have one value, and a rule excludes both. The four-choice
# example: ``x + y <= 1``, ``d == y``, ``y + d + z <= 2``, with ``x``, ``y``, ``z`` of one
# value and ``d`` of two; every choice takes part.

PAIR = Case(
    "an excluding pair",
    Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1)), Choice("y", listed(1))),
            constraints=(
                Constraint(
                    "exclusive", ("x", "y"), lambda read: read["x"] + read["y"] <= 1, "both"
                ),
            ),
        ),
    ),
)

FOUR_CHOICES = (
    Choice("x", listed(1)),
    Choice("y", listed(1)),
    Choice("z", listed(1)),
    Choice("d", listed(0, 1)),
)
FOUR_RULES = (
    Constraint("pair", ("x", "y"), lambda read: read["x"] + read["y"] <= 1, "pair"),
    Constraint("copy", ("d", "y"), lambda read: read["d"] == read["y"], "copy"),
    Constraint(
        "triple", ("y", "d", "z"), lambda read: read["y"] + read["d"] + read["z"] <= 2, "triple"
    ),
)


def four(choices: tuple[int, ...], rules: tuple[int, ...]) -> Case:
    """The four-choice example, its choices and rules declared in the given order."""
    return Case(
        f"four choices {choices} {rules}",
        Space(
            (),
            Scope(
                "",
                choices=tuple(FOUR_CHOICES[index] for index in choices),
                constraints=tuple(FOUR_RULES[index] for index in rules),
            ),
        ),
    )


FOUR = four((0, 1, 2, 3), (0, 1, 2))


@expected("pair-first")
def test_the_first_of_two_forced_choices_excluding_each_other_is_refused_naming_the_other() -> None:
    PAIR.start().check_state("x")


@expected("pair-second")
def test_the_second_of_two_forced_choices_excluding_each_other_is_refused_naming_the_other() -> (
    None
):
    PAIR.start().check_state("y")


@expected("pair-validity")
def test_two_forced_choices_excluding_each_other_refuse_the_configuration() -> None:
    PAIR.start().check_validity()


def test_committing_a_participant_of_a_conflict_is_accepted() -> None:
    PAIR.start().check_change(Change(commit={"x": 1}))


def test_a_committed_participant_keeps_its_value() -> None:
    reach(PAIR.start(), Change(commit={"x": 1})).check_state("x")


def test_the_other_choice_of_a_rule_with_a_committed_one_has_no_possible_value() -> None:
    reach(PAIR.start(), Change(commit={"x": 1})).check_state("y")


@expected("four-participants")
def test_the_four_choice_conflict_refuses_every_participant_naming_the_others() -> None:
    point = FOUR.start()
    for name in ("x", "y", "z", "d"):
        point.check_state(name)


@expected("four-validity")
def test_the_four_choice_conflict_refuses_the_configuration() -> None:
    FOUR.start().check_validity()


@expected("four-orders")
def test_the_four_choice_conflict_is_the_same_in_every_declaration_order() -> None:
    # The 24 orders of the choices, each with one of the 6 orders of the rules in turn.
    rule_orders = list(itertools.permutations(range(3)))
    for position, choices in enumerate(itertools.permutations(range(4))):
        point = four(choices, rule_orders[position % len(rule_orders)]).start()
        for name in ("x", "y", "z", "d"):
            point.check_state(name)


# -- 8. the one trial rule -------------------------------------------------------------------------
#
# ``lanes`` forces the core: two leave both cores possible, three force ``packed`` (the stub
# takes an even width), 100 and 128 force ``stub`` (packed takes at most 64). ``rows`` is 2
# under the stub, 1 under packed. ``depth`` takes at most ``rows``: through the forced core
# (``depth``), or from ``lanes`` directly, a committed choice (``span``).

FIT = Case(
    "the one trial rule",
    Space(
        (),
        Scope(
            "",
            choices=(
                Choice("lanes", listed(2, 3, 100, 128)),
                Choice(
                    "depth",
                    listed(1, 2, member=lambda value, read: value <= read["rows"], reads=("rows",)),
                ),
                Choice(
                    "span",
                    listed(
                        1,
                        2,
                        member=lambda value, read: value <= (1 if read["lanes"] % 2 else 2),
                        reads=("lanes",),
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
                                    ("lanes",),
                                    lambda read: read["lanes"] <= 64,
                                    "packed-width",
                                ),
                            ),
                        ),
                        Scope(
                            "stub",
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
                Derived("rows", ("compute",), lambda read: 2 if read["compute"] == "stub" else 1),
            ),
        ),
    ),
)


def test_a_retained_choice_false_on_the_commitments_refuses_the_change() -> None:
    # ``span`` of 2 holds at 128 lanes, and not at 3.
    FIT.start({}, {"lanes": 128, "span": 2}).check_change(Change(commit={"lanes": 3}))


def test_a_retained_choice_false_on_the_commitments_is_refused_as_a_cold_build_is() -> None:
    FIT.start().check_change(Change(commit={"lanes": 3, "span": 2}))


def test_a_retained_choice_checked_against_the_successors_forced_core() -> None:
    # ``depth`` of 2 holds under the stub that 128 lanes force; 3 lanes leave only packed,
    # under which it does not hold. Forcing takes packed, so the change is refused, reaching
    # ``depth`` with its own reason: the configuration it would publish, as a cold build of
    # the same choices reads it, not the admission ``depth`` had under the stub.
    FIT.start({}, {"lanes": 128, "depth": 2}).check_change(Change(commit={"lanes": 3}))


def test_a_retained_choice_checked_against_the_successors_forced_core_as_a_cold_build() -> None:
    FIT.start().check_change(Change(commit={"lanes": 3, "depth": 2}))


def test_a_change_only_the_bases_forced_core_would_refuse_is_accepted() -> None:
    # At 3 lanes packed is forced and ``depth`` of 2 is outside its domain; the change to
    # 128 lanes forces the stub, under which it holds.
    FIT.start({}, {"lanes": 3}).check_change(Change(commit={"lanes": 128, "depth": 2}))


def test_a_change_only_the_bases_forced_core_would_refuse_is_accepted_as_a_cold_build() -> None:
    FIT.start().check_change(Change(commit={"lanes": 128, "depth": 2}))
