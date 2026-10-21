# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model: rules between choices, one test per question.

A linking rule and a requirement (cases 5 and 6 of the reference cases; cases 1 to 4 are
``test_reference_cases``), the canon's two examples, and the hand cases of the reference
model's rewrite: a change is assignment, refused where it makes a rule newly false, naming
the rule at its origin and the stored choices standing in its way; a selector switched
away puts its case's choices dormant; a change of facts is a new design space, never
refused, dropping what falls outside and reporting the rules it breaks, a batch its facts
then its choices, and a bind the same instantiation from nothing. Each rule is a constraint
in the model; the engine spells it as it is authored (``Spelling`` beside each case).
"""

from __future__ import annotations

from typing import Any

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
)


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


def test_a_linking_rule_leaves_the_other_choice_open() -> None:
    LINKED.start({}, {"a": 1}).check_state("b")


def test_a_linking_rule_refuses_a_value_naming_the_other_choice_standing() -> None:
    LINKED.start({}, {"b": 2}).check_change(Change(commit={"a": 1}))


def test_a_linking_rule_accepts_both_choices_moved_in_one_change() -> None:
    LINKED.start({}, {"a": 3, "b": 2}).check_change(Change(commit={"a": 1, "b": 1}))


def test_a_linking_rule_accepts_a_value_once_the_other_choice_is_cleared() -> None:
    LINKED.start({}, {"b": 2}).check_change(Change(commit={"a": 1}, clear=("b",)))


def test_a_linking_rules_possible_values_name_the_other_choice_standing() -> None:
    LINKED.start({}, {"b": 2}).check_possible("a")


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


def test_a_case_needing_a_capability_the_facts_lack_leaves_the_choice_open() -> None:
    REQUIRED.start({"has_big": False}).check_state("style")


def test_a_case_needing_a_capability_the_facts_lack_is_not_possible() -> None:
    REQUIRED.start({"has_big": False}).check_possible("style")


def test_setting_a_case_needing_a_missing_capability_is_refused_by_the_facts_alone() -> None:
    REQUIRED.start({"has_big": False}).check_change(Change(commit={"style": "big"}))


def test_new_facts_lacking_a_capability_a_stored_case_needs_are_reported() -> None:
    # Accepted, reading refused at the rule, which the report names with ``style`` standing
    # (§ 16.24).
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


def test_the_pairs_left_alone_at_six_is_refused_naming_the_rule_and_right() -> None:
    PAIR.start({"extent": 12}, BOTH_AT_THREE).check_change(Change(commit={"left.lanes": 6}))


def test_the_pairs_lanes_both_at_six_are_accepted() -> None:
    PAIR.start({"extent": 12}, BOTH_AT_THREE).check_change(
        Change(commit={"left.lanes": 6, "right.lanes": 6})
    )


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


def test_the_memorys_ultra_is_not_possible_without_uram() -> None:
    MEMORY.start({"uram": False}).check_possible("style")


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


def test_a_rule_in_a_case_that_never_names_its_selector_refuses_a_switch_to_it() -> None:
    DSP58.start({"has_dsp58": False}, ON_PACKED).check_change(Change(commit={"compute": "dsp58"}))


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


def test_a_new_value_breaking_a_sum_names_the_other_two_standing() -> None:
    BUDGET.start({}, SPENT).check_change(Change(commit={"a": 5}))


def test_a_new_value_breaking_a_sum_with_one_other_set_names_the_last_standing() -> None:
    BUDGET.start({}, SPENT).check_change(Change(commit={"a": 5, "b": 3}))


def test_clearing_what_stands_in_a_sums_way_is_accepted() -> None:
    BUDGET.start({}, SPENT).check_change(Change(clear=("b", "c")))


def test_a_value_once_what_stood_in_its_way_is_cleared_is_accepted() -> None:
    BUDGET.start({}, SPENT).step(Change(clear=("b", "c"))).check_change(Change(commit={"a": 5}))


# A root rule reading the selector ``s`` as a value and ``s.a.p``, whose existence ``s``
# decides: ``s.a.p + x <= 3`` once ``s`` is ``a``. A choice read as a value stands, even when
# it also decides whether something else the rule reads exists (the semantics note, § 16.22).

SELECTOR_READ = Case(
    "a root rule reading a selector as a value",
    Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2, 3)),),
            selectors=(
                Selector("s", (Scope("a", choices=(Choice("p", listed(1, 2, 3)),)), Scope("b"))),
            ),
            constraints=(
                Constraint(
                    "fits",
                    ("s", "s.a.p", "x"),
                    lambda r: r["s"] != "a" or r["s.a.p"] + r["x"] <= 3,
                    "over",
                ),
            ),
        ),
    ),
)


def test_a_selector_a_rule_reads_as_a_value_stands_beside_the_choice_it_decides() -> None:
    SELECTOR_READ.start({}, {"s": "a", "s.a.p": 1, "x": 1}).check_change(Change(commit={"x": 3}))


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


def test_switching_away_puts_the_cases_choices_dormant() -> None:
    TWO_KINDS.start({}, ON_PACKED_TWO).check_change(Change(commit={"compute": "serial"}))


def away() -> Pair:
    """``pe`` and ``simd`` set on ``packed``, then ``serial`` selected (compared on the way)."""
    return TWO_KINDS.start({}, ON_PACKED_TWO).reach(Change(commit={"compute": "serial"}))


def test_a_choice_of_the_case_switched_away_is_dormant() -> None:
    away().check_state(PE2)


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


def test_a_switch_back_is_refused_by_a_restored_choice() -> None:
    moved().check_change(Change(commit={"compute": "packed"}))


def test_a_switch_back_with_the_restored_choice_moved_too_is_accepted() -> None:
    moved().check_change(Change(commit={"compute": "packed", PE2: 2}))


# A facts change dropping each stored value outside its new domain, a dormant one included
# (§ 16.24): new facts are a new design space, never refused.

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


def test_a_facts_change_drops_a_dormant_value_outside_its_new_domain() -> None:
    sized().check_change(Change(facts={"n": 12}))


def test_a_facts_change_drops_each_value_outside_its_new_domain() -> None:
    sized().check_change(Change(facts={"n": 6}))


def test_a_facts_change_clearing_a_value_outside_its_new_domain_is_accepted() -> None:
    # The cleared ``mode.a.depth`` is named: left out without being reported (§ 16.25).
    sized().check_change(Change(facts={"n": 12}, clear=("mode.a.depth",)))


# A batch is its facts, then its choices as a change of choices on their design space
# (§ 16.25); a bind is the same instantiation from nothing (SZ90). ``k`` asks ``c`` to be at
# most the fact ``f``, ``l`` the same of ``x``.

BOUNDED = Case(
    "a batch of facts and choices",
    Space(
        ("f",),
        Scope(
            "",
            choices=(Choice("c", listed(1, 2, 3, 4)), Choice("x", listed(1, 2, 3, 4))),
            constraints=(
                Constraint("k", ("c", "f"), lambda r: r["c"] <= r["f"], "c-over"),
                Constraint("l", ("x", "f"), lambda r: r["x"] <= r["f"], "x-over"),
            ),
        ),
    ),
)


def broken_by_facts() -> Pair:
    """``c = 3`` at ``f = 4``, then ``f = 2``: accepted, ``k`` reported with ``c`` standing,
    and the design space reads refused (compared on the way)."""
    return BOUNDED.start({"f": 4}, {"c": 3}).reach(Change(facts={"f": 2}))


def test_a_design_space_new_facts_left_reading_refused_reads_refused_through_the_rule() -> None:
    pair = broken_by_facts()
    pair.check_validity()
    pair.check_rule("k")


def test_a_batch_whose_choices_add_back_a_pair_its_facts_removed_is_refused() -> None:
    # The receiver holds ``k`` false through ``c = 3`` at ``f = 2``. ``f = 3`` removes the
    # pair; ``c = 4`` beside it adds it back against the facts' design space: refused, ``c``
    # named, so nothing standing.
    broken_by_facts().check_change(Change(facts={"f": 3}, commit={"c": 4}))


def test_a_batch_breaking_a_rule_through_a_choice_it_sets_is_its_facts_then_its_choices() -> None:
    # The equivalence: ``f = 2`` makes ``k`` false with ``c = 3``; the batch sets ``c = 4``,
    # ``k`` false through the same origin: accepted, ``k`` reported with ``c`` standing, the
    # same design space as the facts, then the choices.
    start = BOUNDED.start({"f": 4}, {"c": 3})
    batch = start.check_change(Change(facts={"f": 2}, commit={"c": 4}))
    steps = start.reach(Change(facts={"f": 2})).check_change(Change(commit={"c": 4}))
    assert batch is not None and steps is not None
    assert batch.model == steps.model and batch.engine.stored() == steps.engine.stored()
    batch.check_validity()
    batch.check_rule("k")


def test_a_bind_keeps_saved_choices_that_break_a_rule_and_drops_one_outside_its_domain() -> None:
    # Under ``f = 2``: ``c = 3`` breaks ``k`` and is kept, reported, ``c`` standing; ``x = 5``
    # is not of its type and is dropped. The design space reads refused through ``k``.
    bound = BOUNDED.start({"f": 2}).check_bind({"c": 3, "x": 5})
    bound.check_validity()
    bound.check_state("c")
    bound.check_state("x")


# A bind reads against its facts with nothing stored, whatever the design space it is asked
# of stores. ``x`` lists 1 to ``f``; ``k`` asks ``c`` to be at most ``f``; ``mode`` selects
# ``a`` (``p``, ``r``) or ``b`` (``q``).

SELECTED = Case(
    "a bind that moves, drops and breaks",
    Space(
        ("f",),
        Scope(
            "",
            choices=(
                Choice("c", listed(1, 2, 3, 4)),
                Choice("x", computed(lambda r: tuple(range(1, r["f"] + 1)), ("f",))),
            ),
            constraints=(Constraint("k", ("c", "f"), lambda r: r["c"] <= r["f"], "c-over"),),
            selectors=(
                Selector(
                    "mode",
                    (
                        Scope("a", choices=(Choice("p", listed(1, 2)), Choice("r", listed(1, 2)))),
                        Scope("b", choices=(Choice("q", listed(5, 6)),)),
                    ),
                ),
            ),
        ),
    ),
)
SAVED = {"c": 3, "x": 4, "mode": "a", "mode.a.p": 1, "mode.b.q": 5}


def test_a_bind_from_a_design_space_with_stored_choices_reads_against_nothing_stored() -> None:
    # Under ``f = 2``, from nothing stored: ``x = 4`` dropped, ``k`` broken with ``c``
    # standing, ``mode.a.p`` brought into play, ``mode.b.q`` put dormant, ``mode.a.r``
    # opened. From a design space storing ``mode = a``, ``mode.a.p = 2``, ``mode.a.r = 1``
    # and ``c = 1`` (where ``p`` and ``r`` are present, ``q`` absent), the same: the
    # receiver's choices don't matter.
    nothing, _ = SELECTED.start({"f": 2}).engine.bind(SAVED)
    assert nothing.dormant == {"mode.b.q"} and nothing.in_play == {"mode.a.p"}
    assert nothing.opened == {"mode.a.r"} and {d.subject for d in nothing.dropped} == {"x"}
    assert {(rule.subject, rule.standing) for rule in nothing.broken} == {("k", frozenset({"c"}))}
    held = SELECTED.start({"f": 2}, {"mode": "a", "mode.a.p": 2, "mode.a.r": 1, "c": 1})
    form, _ = held.engine.bind(SAVED)
    assert form == nothing
    bound = held.check_bind(SAVED)
    bound.check_validity()
    bound.check_state("mode.b.q")


# New facts emptying a present choice's listed domain (reading 9, settled by § 16.24):
# accepted, the empty-domain piece false and reported, a value stored there dropped first.

COUNTED = Case(
    "a domain the facts empty",
    Space(
        ("n",), Scope("", choices=(Choice("x", computed(lambda r: tuple(range(r["n"])), ("n",))),))
    ),
)


def test_new_facts_emptying_a_domain_are_accepted_and_report_its_empty_list() -> None:
    COUNTED.start({"n": 2}).check_change(Change(facts={"n": 0}))


def test_new_facts_emptying_a_stored_choices_domain_drop_it_and_report_the_empty_list() -> None:
    emptied = COUNTED.start({"n": 2}, {"x": 1}).reach(Change(facts={"n": 0}))
    emptied.check_validity()
    emptied.check_state("x")


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


def test_a_named_choice_under_an_open_selector_leaves_the_configuration_valid() -> None:
    OPEN_MODE.start({}, {"mode.a.p": 2}).check_validity()


def test_selecting_a_waiting_choices_case_brings_it_into_play() -> None:
    OPEN_MODE.start({}, {"mode.a.p": 2}).check_change(Change(commit={"mode": "a"}))
