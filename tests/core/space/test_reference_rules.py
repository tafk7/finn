# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model: rules between choices, one test per question.

A linking rule and a requirement (cases 5 and 6 of the reference cases; cases 1 to 4 are
``test_reference_cases``), the canon's two examples, and the hand cases of the reference
model's rewrite: a change is assignment, refused where it breaks a rule it reaches, naming
the rule at its origin and the stored choices standing in its way; a selector switched
away puts its case's choices dormant. Each rule is a constraint in the model; the engine
spells it as it is authored at this commit (``Spelling`` beside each case).
"""

from __future__ import annotations

from typing import Any

import pytest

from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Reads,
    Refuse,
    Scope,
    Selector,
    Space,
    View,
    computed,
    listed,
)
from core.space.reference_harness import (
    Case,
    Pair,
    Requirement,
    Spelling,
    check_definition,
    expected,
)

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")


def divisors(extent: int) -> tuple[int, ...]:
    return tuple(value for value in range(1, extent + 1) if extent % value == 0)


# -- 5. a linking rule as a constraint -------------------------------------------------------------
#
# ``b <= a``, over two choices of one Space: a constraint of the Space. The same bound as
# ``b``'s domain, computed from ``a``, is no definition: a domain's membership reads facts
# only, and a relation to another choice is a constraint.

LINKED = Case(
    "a linking rule",
    Space(
        (),
        Scope(
            "",
            choices=(Choice("a", listed(1, 2, 3)), Choice("b", listed(1, 2, 3))),
            constraints=(
                Constraint("linked", ("a", "b"), lambda read: read["b"] <= read["a"], "b-above-a"),
            ),
        ),
    ),
)


@expected("linked-open")
def test_a_linking_rule_leaves_the_other_choice_open() -> None:
    LINKED.start({}, {"a": 1}).check_state("b")


@expected("linked-refuses")
def test_a_linking_rule_refuses_a_value_naming_the_other_choice_standing() -> None:
    LINKED.start({}, {"b": 2}).check_change(Change(commit={"a": 1}))


def test_a_linking_rule_accepts_both_choices_moved_in_one_change() -> None:
    LINKED.start({}, {"a": 3, "b": 2}).check_change(Change(commit={"a": 1, "b": 1}))


def test_a_linking_rule_accepts_a_value_once_the_other_choice_is_cleared() -> None:
    LINKED.start({}, {"b": 2}).check_change(Change(commit={"a": 1}, clear=("b",)))


@expected("linked-possible")
def test_a_linking_rules_possible_values_name_the_other_choice_standing() -> None:
    LINKED.start({}, {"b": 2}).check_possible("a")


@expected("domain-reads-choice")
def test_a_domain_reading_another_choice_is_no_definition() -> None:
    check_definition(
        "a domain reading another choice",
        Space(
            (),
            Scope(
                "",
                choices=(
                    Choice("a", listed(1, 2, 3)),
                    Choice(
                        "b",
                        computed(
                            lambda read: tuple(v for v in (1, 2, 3) if v <= read["a"]), ("a",)
                        ),
                    ),
                ),
            ),
        ),
    )


# -- 6. a requirement as a constraint --------------------------------------------------------------
#
# ``style`` ``big`` needs a capability, the fact ``has_big``: a constraint over the choice
# and the fact, spelled ``requires(has_big, cases=("big",))`` on the domain in the canon.

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


@expected("required-open")
def test_a_case_needing_a_capability_the_facts_lack_leaves_the_choice_open() -> None:
    REQUIRED.start({"has_big": False}).check_state("style")


@expected("required-possible")
def test_a_case_needing_a_capability_the_facts_lack_is_not_possible() -> None:
    REQUIRED.start({"has_big": False}).check_possible("style")


def test_a_case_needing_an_unknown_capability_is_possible() -> None:
    REQUIRED.start().check_possible("style")


@expected("required-commit")
def test_setting_a_case_needing_a_missing_capability_is_refused_by_the_facts_alone() -> None:
    REQUIRED.start({"has_big": False}).check_change(Change(commit={"style": "big"}))


@expected("required-facts")
def test_new_facts_lacking_a_capability_a_stored_case_needs_are_refused() -> None:
    REQUIRED.start({"has_big": True}, {"style": "big"}).check_change(
        Change(facts={"has_big": False})
    )


# -- 7. the canon's examples (DESIGN § 3 and § 4.6) ------------------------------------------------
#
# ``Pair``: two ``Tiles`` of one extent, each with ``lanes`` dividing it and a budget of four
# cycles, and a rule that the right is not slower than the left. ``Memory``: a style
# ``ultra`` needs UltraRAM, a fact.


def tiles(name: str) -> Scope:
    lanes, cycles = f"{name}.lanes", f"{name}.cycles"
    return Scope(
        name,
        choices=(Choice("lanes", computed(lambda r: divisors(r["extent"]), ("extent",))),),
        derived=(Derived("cycles", ("extent", lanes), lambda r: r["extent"] // r[lanes]),),
        constraints=(
            Constraint("within_budget", (cycles,), lambda r: r[cycles] <= 4, "over-budget"),
        ),
    )


PAIR = Case(
    "the canon's Pair",
    Space(
        ("extent",),
        Scope(
            "",
            children=(tiles("left"), tiles("right")),
            constraints=(
                Constraint(
                    "right_not_slower",
                    ("right.cycles", "left.cycles"),
                    lambda r: r["right.cycles"] <= r["left.cycles"],
                    "right-slower",
                ),
            ),
            views=(
                View(
                    "timing",
                    ("left.cycles", "right.cycles"),
                    lambda r: (r["left.cycles"], r["right.cycles"]),
                ),
            ),
        ),
    ),
    Spelling(types={"timing": tuple}),
)
BOTH_AT_THREE = {"left.lanes": 3, "right.lanes": 3}


def test_the_pair_with_nothing_set_is_open() -> None:
    PAIR.start({"extent": 12}).check_validity()


def test_the_pair_with_both_lanes_set_is_valid() -> None:
    PAIR.start({"extent": 12}, BOTH_AT_THREE).check_validity()


def test_the_pairs_timing_view_answers() -> None:
    PAIR.start({"extent": 12}, BOTH_AT_THREE).check_answer("timing")


@expected("pair-left-alone")
def test_the_pairs_left_alone_at_six_is_refused_naming_the_rule_and_right() -> None:
    PAIR.start({"extent": 12}, BOTH_AT_THREE).check_change(Change(commit={"left.lanes": 6}))


def test_the_pairs_lanes_both_at_six_are_accepted() -> None:
    PAIR.start({"extent": 12}, BOTH_AT_THREE).check_change(
        Change(commit={"left.lanes": 6, "right.lanes": 6})
    )


@expected("pair-possible")
def test_the_pairs_possible_values_of_left_beside_right_at_three() -> None:
    # 1, 2: left's own budget, nothing standing; 3: the value held; 4, 6, 12:
    # ``right_not_slower``, with ``right.lanes`` standing.
    PAIR.start({"extent": 12}, BOTH_AT_THREE).check_possible("left.lanes")


MEMORY = Case(
    "the canon's Memory",
    Space(
        ("uram",),
        Scope(
            "",
            choices=(Choice("style", listed("block", "ultra")),),
            constraints=(
                Constraint(
                    "ultra_needs_uram",
                    ("style", "uram"),
                    lambda r: r["style"] != "ultra" or bool(r["uram"]),
                    "uram-absent",
                ),
            ),
        ),
    ),
    Spelling(types={"uram": bool}),
)


@expected("memory-possible")
def test_the_memorys_ultra_is_not_possible_without_uram() -> None:
    MEMORY.start({"uram": False}).check_possible("style")


@expected("memory-commit")
def test_the_memorys_ultra_is_refused_without_uram_nothing_standing() -> None:
    MEMORY.start({"uram": False}).check_change(Change(commit={"style": "ultra"}))


# -- 8. the reference model's hand cases (the semantics note, § 12) --------------------------------
#
# A refusal through a refused input is blamed at its origin: ``x = 3`` makes ``width``
# refuse on its own values, and ``fits`` is false only through it.


def odd_width(r: Reads) -> Any:
    return Refuse("odd") if (r["x"] * r["y"]) % 2 else r["x"] * r["y"]


ODD_PRODUCT = Case(
    "a refusal through a refused input",
    Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2, 3)), Choice("y", listed(1, 2))),
            derived=(Derived("width", ("x", "y"), odd_width),),
            constraints=(Constraint("fits", ("width",), lambda r: r["width"] <= 4, "too-wide"),),
            views=(View("shown", ("width",), lambda r: r["width"]),),
        ),
    ),
)


@expected("origin-blame")
def test_a_refusal_through_a_refused_input_is_blamed_at_its_origin_only() -> None:
    ODD_PRODUCT.start({}, {"x": 2, "y": 1}).check_change(Change(commit={"x": 3}))


# ``compute``: the ``packed`` core, or ``dsp58``, which a rule over facts alone allows only
# on a part with DSP58; ``dsp58``'s rule never names its selector.

DSP58 = Case(
    "a rule in a case that never names its selector",
    Space(
        ("has_dsp58",),
        Scope(
            "",
            selectors=(
                Selector(
                    "compute",
                    (
                        Scope("packed", choices=(Choice("pe", listed(1, 2)),)),
                        Scope(
                            "dsp58",
                            choices=(Choice("pe", listed(1, 2, 4)),),
                            constraints=(
                                Constraint(
                                    "supported",
                                    ("has_dsp58",),
                                    lambda r: bool(r["has_dsp58"]),
                                    "no-dsp58",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
    Spelling(types={"has_dsp58": bool}),
)
ON_PACKED = {"compute": "packed", "compute.packed.pe": 2}


@expected("switch-reaches-case-rule")
def test_a_rule_in_a_case_that_never_names_its_selector_is_reached_by_switching() -> None:
    DSP58.start({"has_dsp58": False}, ON_PACKED).check_change(Change(commit={"compute": "dsp58"}))


@expected("switch-dormant-opened")
def test_a_switch_puts_the_old_cases_choices_dormant_and_opens_the_new_ones() -> None:
    DSP58.start({"has_dsp58": True}, ON_PACKED).check_change(Change(commit={"compute": "dsp58"}))


# ``a + b + c <= 10``.

BUDGET = Case(
    "a budget over three choices",
    Space(
        (),
        Scope(
            "",
            choices=tuple(Choice(name, listed(1, 2, 3, 4, 5, 6)) for name in ("a", "b", "c")),
            constraints=(
                Constraint(
                    "budget", ("a", "b", "c"), lambda r: r["a"] + r["b"] + r["c"] <= 10, "over"
                ),
            ),
        ),
    ),
)
SPENT = {"a": 1, "b": 4, "c": 4}


@expected("sum-two-standing")
def test_a_new_value_breaking_a_sum_names_the_other_two_standing() -> None:
    BUDGET.start({}, SPENT).check_change(Change(commit={"a": 5}))


@expected("sum-one-standing")
def test_a_new_value_breaking_a_sum_with_one_other_set_names_the_last_standing() -> None:
    BUDGET.start({}, SPENT).check_change(Change(commit={"a": 5, "b": 3}))


def test_clearing_what_stands_in_a_sums_way_is_accepted() -> None:
    BUDGET.start({}, SPENT).check_change(Change(clear=("b", "c")))


def test_a_value_once_what_stood_in_its_way_is_cleared_is_accepted() -> None:
    BUDGET.start({}, SPENT).step(Change(clear=("b", "c"))).check_change(Change(commit={"a": 5}))


# A selector switched away and back: the old case's choices go dormant and come back.

PE2, SIMD2, LANES2 = "compute.packed.pe", "compute.packed.simd", "compute.serial.lanes"
TWO_KINDS = Case(
    "a selector switched away and back",
    Space(
        (),
        Scope(
            "",
            selectors=(
                Selector(
                    "compute",
                    (
                        Scope(
                            "packed",
                            choices=(Choice("pe", listed(1, 2, 4)), Choice("simd", listed(1, 2))),
                        ),
                        Scope("serial", choices=(Choice("lanes", listed(1, 2)),)),
                    ),
                ),
            ),
        ),
    ),
)
ON_PACKED_TWO = {"compute": "packed", PE2: 2, SIMD2: 1}


@expected("switch-away")
def test_switching_away_puts_the_cases_choices_dormant() -> None:
    TWO_KINDS.start({}, ON_PACKED_TWO).check_change(Change(commit={"compute": "serial"}))


def away() -> Pair:
    """``pe`` and ``simd`` set on ``packed``, then ``serial`` selected (compared on the way)."""
    return TWO_KINDS.start({}, ON_PACKED_TWO).reach(Change(commit={"compute": "serial"}))


@expected("switched-away-dormant")
def test_a_choice_of_the_case_switched_away_is_dormant() -> None:
    away().check_state(PE2)


@expected("switch-back")
def test_switching_back_brings_the_dormant_choices_into_play() -> None:
    away().step(Change(commit={LANES2: 2})).check_change(Change(commit={"compute": "packed"}))


def test_clearing_a_selector_leaves_its_cases_choices_waiting() -> None:
    TWO_KINDS.start({}, ON_PACKED_TWO).step(Change(clear=("compute",))).check_state(PE2)


# A switch back refused by a restored choice: ``pe = 4`` is dormant while ``w`` moves to 6;
# switching back brings ``pe`` into play, where ``pe_fits`` is false.

FITTED = Case(
    "a switch back refused by a restored choice",
    Space(
        (),
        Scope(
            "",
            choices=(Choice("w", listed(4, 6)),),
            selectors=(
                Selector(
                    "compute",
                    (
                        Scope(
                            "packed",
                            choices=(Choice("pe", listed(1, 2, 4)),),
                            constraints=(
                                Constraint(
                                    "pe_fits",
                                    (PE2, "w"),
                                    lambda r: r["w"] % r[PE2] == 0,
                                    "pe-misfit",
                                ),
                            ),
                        ),
                        Scope("serial"),
                    ),
                ),
            ),
        ),
    ),
)


def moved() -> Pair:
    """``pe = 4`` on ``packed`` at ``w = 4``, then ``serial`` (compared on the way), then
    ``w = 6``."""
    base = FITTED.start({}, {"w": 4, "compute": "packed", PE2: 4})
    return base.reach(Change(commit={"compute": "serial"})).step(Change(commit={"w": 6}))


@expected("switch-back-refused")
def test_a_switch_back_is_refused_by_a_restored_choice() -> None:
    moved().check_change(Change(commit={"compute": "packed"}))


@expected("switch-back-moved")
def test_a_switch_back_with_the_restored_choice_moved_too_is_accepted() -> None:
    moved().check_change(Change(commit={"compute": "packed", PE2: 2}))


# A facts change refused by a stored value outside its new domain, a dormant one included.

SIZED = Case(
    "a facts change and the stored values' types",
    Space(
        ("n",),
        Scope(
            "",
            choices=(Choice("lanes", computed(lambda r: divisors(r["n"]), ("n",))),),
            selectors=(
                Selector(
                    "mode",
                    (
                        Scope(
                            "a",
                            choices=(
                                Choice("depth", computed(lambda r: divisors(r["n"]), ("n",))),
                            ),
                        ),
                        Scope("b"),
                    ),
                ),
            ),
        ),
    ),
)
DORMANT_DEPTH = {"lanes": 4, "mode": "a", "mode.a.depth": 8}


def sized() -> Pair:
    """``lanes`` 4 and ``mode.a.depth`` 8 at ``n = 8``, then ``mode`` switched to ``b``
    (compared on the way)."""
    return SIZED.start({"n": 8}, DORMANT_DEPTH).reach(Change(commit={"mode": "b"}))


@expected("facts-dormant-outside")
def test_a_facts_change_is_refused_by_a_dormant_value_outside_its_new_domain() -> None:
    sized().check_change(Change(facts={"n": 12}))


@expected("facts-each-outside")
def test_a_facts_change_is_refused_by_each_value_outside_its_new_domain() -> None:
    sized().check_change(Change(facts={"n": 6}))


@expected("facts-clear-standing")
def test_a_facts_change_clearing_what_stands_in_its_way_is_accepted() -> None:
    sized().check_change(Change(facts={"n": 12}, clear=("mode.a.depth",)))


# A named choice under an open selector waits on it.

OPEN_MODE = Case(
    "a named choice under an open selector",
    Space(
        (),
        Scope(
            "",
            selectors=(
                Selector(
                    "mode",
                    (
                        Scope("a", choices=(Choice("p", listed(1, 2)),)),
                        Scope("b", choices=(Choice("q", listed(5, 6)),)),
                    ),
                ),
            ),
        ),
    ),
)


def test_a_named_choice_under_an_open_selector_is_accepted() -> None:
    OPEN_MODE.start().check_change(Change(commit={"mode.a.p": 2}))


def test_a_named_choice_under_an_open_selector_waits() -> None:
    OPEN_MODE.start({}, {"mode.a.p": 2}).check_state("mode.a.p")


@expected("waiting-valid")
def test_a_named_choice_under_an_open_selector_leaves_the_configuration_valid() -> None:
    OPEN_MODE.start({}, {"mode.a.p": 2}).check_validity()


@expected("waiting-in-play")
def test_selecting_a_waiting_choices_case_brings_it_into_play() -> None:
    OPEN_MODE.start({}, {"mode.a.p": 2}).check_change(Change(commit={"mode": "a"}))
