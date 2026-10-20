# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model: views, nested scopes and guards, one test per
question.

Case 9 of the reference cases (views and validity, and a constraint reading a view of its
own Space, a cycle, in three shapes), and two of how a Space holds cores: selectors nested
three deep, and a value choice whose case puts a selector no case of which holds in scope.
"""

from __future__ import annotations

from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Refuse,
    Scope,
    Selector,
    Space,
    View,
    When,
    listed,
)
from core.space.reference_harness import (
    ByName,
    Case,
    Spelling,
    check_definition,
)

# -- 9. views and validity -------------------------------------------------------------------------
#
# ``title`` presents the fact ``name``, ``doubled`` the choice ``x``. The Space refuses a
# ``name`` of "bad", and an ``x`` above ``limit``.

VIEWED = Case(
    "views",
    Space(
        ("limit", "name"),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2)),),
            constraints=(
                Constraint(
                    "under", ("x", "limit"), lambda read: read["x"] <= read["limit"], "over"
                ),
                Constraint("named", ("name",), lambda read: read["name"] != "bad", "bad-name"),
            ),
            views=(
                View("doubled", ("x",), lambda read: 2 * read["x"]),
                View("title", ("name",), lambda read: f"unit {read['name']}"),
            ),
        ),
    ),
    Spelling(types={"name": str, "title": str}),
)


def test_a_view_refuses_when_its_space_is_refused() -> None:
    VIEWED.start({"limit": 2, "name": "bad"}).check_answer("title")


def test_a_view_answers_while_its_space_is_open() -> None:
    VIEWED.start({"name": "unit"}).check_answer("title")


def test_a_view_of_an_open_choice_waits_on_it() -> None:
    VIEWED.start({"limit": 2, "name": "unit"}).check_answer("doubled")


# A constraint reading a view of its own Space closes a loop through the Space's validity:
# the model refuses the definition. Three shapes: a constraint on a storage view; a rule
# reading views by name; a parent reading its children's stages through a view of its own.


def stage(name: str) -> Scope:
    return Scope(
        name,
        choices=(Choice("depth", listed(1, 2, 4)),),
        views=(View("storage", (f"{name}.depth",), lambda read: 8 * read[f"{name}.depth"]),),
    )


STORAGE = Space(
    (),
    Scope(
        "",
        choices=(Choice("depth", listed(1, 2, 4)),),
        views=(View("storage", ("depth",), lambda read: 8 * read["depth"]),),
        constraints=(
            Constraint("fits", ("storage",), lambda read: read["storage"] <= 16, "too-deep"),
        ),
    ),
)
TENSORS = ("activation", "weight")
BY_NAME = Space(
    (),
    Scope(
        "",
        choices=(Choice("rows", listed(1, 2)), Choice("cols", listed(1, 2))),
        views=(
            View("activation", ("rows",), lambda read: 8 * read["rows"]),
            View("weight", ("cols",), lambda read: 8 * read["cols"]),
        ),
        constraints=(
            Constraint(
                "carried", TENSORS, lambda read: sum(read[name] for name in TENSORS) <= 24, "wide"
            ),
        ),
    ),
)
STAGES = Space(
    (),
    Scope(
        "",
        children=(stage("first"), stage("second")),
        views=(
            View(
                "stages",
                ("first.storage", "second.storage"),
                lambda read: (read["first.storage"], read["second.storage"]),
            ),
        ),
        constraints=(
            Constraint(
                "compatible", ("stages",), lambda read: sum(read["stages"]) <= 40, "too-deep"
            ),
        ),
    ),
)


CHILDREN = Space(
    (),
    Scope(
        "",
        children=(stage("first"), stage("second")),
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


def test_a_parent_reading_its_childrens_views_is_not_a_cycle() -> None:
    check_definition("children's views", CHILDREN)


def test_a_constraint_on_a_storage_view_of_its_own_space_is_a_cycle() -> None:
    check_definition("a constraint on a storage view", STORAGE)


def test_a_constraint_reading_views_of_its_own_space_by_name_is_a_cycle() -> None:
    check_definition("views read by name", BY_NAME, Spelling(rules={"carried": ByName()}))


def test_a_parent_reading_its_childrens_stages_through_its_own_view_is_a_cycle() -> None:
    check_definition("stages through a parent's view", STAGES, Spelling(types={"stages": tuple}))


# -- nested selectors: a core in a kernel in a kernel ----------------------------------------------
#
# ``kernel`` selects ``inner`` (a kernel whose ``compute`` selects a core, ``packed`` or
# ``serial``) or ``plain``. ``plain`` needs the fact ``plain_ok``, ``serial`` the fact
# ``serial_ok``; ``packed``'s ``pe`` divides ``width``. Keys three deep.

PE = "kernel.inner.compute.packed.pe"
COMPUTE = "kernel.inner.compute"
FITS = "kernel.inner.compute.packed.pe_fits"

NESTED = Case(
    "nested selectors",
    Space(
        ("plain_ok", "serial_ok", "width"),
        Scope(
            "",
            selectors=(
                Selector(
                    "kernel",
                    (
                        Scope(
                            "inner",
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
                                                    (PE, "width"),
                                                    lambda read: read["width"] % read[PE] == 0,
                                                    "pe-misfit",
                                                ),
                                            ),
                                        ),
                                        Scope(
                                            "serial",
                                            choices=(Choice("lanes", listed(1)),),
                                            constraints=(
                                                Constraint(
                                                    "serial_ok",
                                                    ("serial_ok",),
                                                    lambda read: bool(read["serial_ok"]),
                                                    "no-serial",
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        Scope("plain"),
                    ),
                ),
            ),
            constraints=(
                Constraint(
                    "plain_allowed",
                    ("kernel", "plain_ok"),
                    lambda read: read["kernel"] != "plain" or bool(read["plain_ok"]),
                    "no-plain",
                ),
            ),
        ),
    ),
    Spelling(types={"plain_ok": bool, "serial_ok": bool}),
)
OPEN = {"plain_ok": True, "serial_ok": True, "width": 4}
CLOSED = {"plain_ok": False, "serial_ok": False, "width": 6}
INNER_PACKED = {"kernel": "inner", COMPUTE: "packed"}


def test_a_rule_three_deep_under_two_open_selectors_waits_on_both() -> None:
    NESTED.start(OPEN).check_rule(FITS)


def test_a_rule_three_deep_under_one_open_selector_waits_on_it() -> None:
    NESTED.start(OPEN, {"kernel": "inner"}).check_rule(FITS)


def test_a_rule_three_deep_under_an_unselected_case_does_not_apply() -> None:
    NESTED.start(OPEN, {"kernel": "plain"}).check_rule(FITS)


def test_an_unset_choice_three_deep_under_an_unselected_case_is_absent() -> None:
    NESTED.start(OPEN, {"kernel": "plain"}).check_state(PE)


def test_a_selector_whose_other_case_the_facts_refuse_stays_open() -> None:
    NESTED.start(CLOSED).check_state("kernel")


def test_a_commit_three_deep_under_two_open_selectors_is_accepted() -> None:
    NESTED.start(CLOSED).check_change(Change(commit={PE: 2}))


def test_a_commit_three_deep_under_two_open_selectors_waits_on_both() -> None:
    NESTED.start(CLOSED, {PE: 2}).check_state(PE)


def test_selecting_the_outer_kernel_leaves_a_choice_three_deep_waiting_on_the_inner() -> None:
    NESTED.start(OPEN, {PE: 2}).check_change(Change(commit={"kernel": "inner"}))


def test_selecting_both_kernels_brings_a_choice_three_deep_into_play() -> None:
    NESTED.start(OPEN, {PE: 2, "kernel": "inner"}).check_change(Change(commit={COMPUTE: "packed"}))


def test_switching_the_outer_kernel_away_puts_the_inner_choices_dormant() -> None:
    NESTED.start(OPEN, {**INNER_PACKED, PE: 2}).check_change(Change(commit={"kernel": "plain"}))


def test_a_commit_three_deep_breaking_its_rule_is_refused_naming_it() -> None:
    NESTED.start({**OPEN, "width": 6}, INNER_PACKED).check_change(Change(commit={PE: 4}))


def test_validity_waits_on_a_fact_a_rule_three_deep_reads() -> None:
    NESTED.start({"plain_ok": True, "serial_ok": True}, {**INNER_PACKED, PE: 4}).check_validity()


# An inner selector inside a case of an outer one: ``inner`` set to ``y`` while ``outer`` is
# open (it waits on it), and a choice under its other case ``x`` set too. New facts then
# leave neither of ``outer``'s cases' rules holding: no case is selected, so the change is
# accepted and ``inner`` still waits. The choice under ``x`` is dormant throughout, waiting on
# nothing: ``inner`` holds ``y``, and presence is Kleene's AND over the path (§ 16.21).
# Selecting ``a`` brings ``inner`` into play.

OUTER = Case(
    "an inner selector under an open outer one",
    Space(
        ("on",),
        Scope(
            "",
            selectors=(
                Selector(
                    "outer",
                    (
                        Scope(
                            "a",
                            selectors=(
                                Selector(
                                    "inner",
                                    (Scope("x", choices=(Choice("c", listed(1, 2)),)), Scope("y")),
                                ),
                            ),
                            constraints=(
                                Constraint("needs", ("on",), lambda read: read["on"] >= 1, "off-a"),
                            ),
                        ),
                        Scope(
                            "b",
                            constraints=(
                                Constraint("needs", ("on",), lambda read: read["on"] >= 2, "off-b"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)
INNER_SET = {"outer.a.inner": "y", "outer.a.inner.x.c": 1}


def test_new_facts_no_case_of_an_open_selector_holds_under_are_accepted() -> None:
    OUTER.start({"on": 2}, INNER_SET).check_change(Change(facts={"on": 0}))


def test_selecting_the_outer_case_brings_the_inner_selector_into_play() -> None:
    OUTER.start({"on": 2}, INNER_SET).check_change(Change(commit={"outer": "a"}))


def test_a_choice_under_the_inner_case_not_selected_is_dormant_while_the_outer_is_open() -> None:
    """``x``'s presence is unknown (``outer`` open) and false (``inner`` holds ``y``): Kleene's
    AND over the path is false (§ 16.21), so its stored choice is dormant, waiting on nothing."""
    OUTER.start({"on": 2}, INNER_SET).check_state("outer.a.inner.x.c")


# A false ``when`` under an open selector: ``g``, in case ``a`` of ``s``, is present when
# ``k`` is at least 3. With ``k`` at 0, ``g``'s presence is unknown (``s`` open) and false (its
# ``when``): Kleene's AND over the path is false (§ 16.21), so ``g`` is absent and its unset
# choice absent, waiting on nothing. With ``k`` a choice, setting it to 0 puts a stored choice
# of ``g`` dormant (waiting before, absent after; § 16.5).


def false_when(over_choice: bool) -> Space:
    guarded = Scope(
        "g",
        choices=(Choice("x", listed(1)),),
        when=When(("k",), lambda read: read["k"] >= 3),
    )
    return Space(
        () if over_choice else ("k",),
        Scope(
            "",
            choices=(Choice("k", listed(0, 5)),) if over_choice else (),
            selectors=(Selector("s", (Scope("a", children=(guarded,)), Scope("b"))),),
        ),
    )


FALSE_WHEN = Case("a false when over a fact under an open selector", false_when(False))
FALSE_WHEN_CHOICE = Case("a when over a choice under an open selector", false_when(True))


def test_a_choice_under_a_false_when_beneath_an_open_selector_is_absent() -> None:
    FALSE_WHEN.start({"k": 0}).check_state("s.a.g.x")


def test_turning_a_when_off_beneath_an_open_selector_puts_its_choice_dormant() -> None:
    FALSE_WHEN_CHOICE.start({}, {"k": 5, "s.a.g.x": 1}).check_change(Change(commit={"k": 0}))


# -- a case putting a selector in scope whose every case the facts refuse --------------------------
#
# ``realization`` is ``native`` or ``other``; ``native`` puts a scope in place whose selector
# chooses a core, ``fast`` or ``slow``, each needing its capability. With neither
# capability neither core's rule holds; with ``native_needs_core``, ``native`` itself needs one.


def realized(with_rule: bool) -> Space:
    cores = Selector(
        "compute",
        (
            Scope(
                "fast",
                constraints=(
                    Constraint(
                        "needs_fast", ("fast_ok",), lambda read: bool(read["fast_ok"]), "no-fast"
                    ),
                ),
            ),
            Scope(
                "slow",
                constraints=(
                    Constraint(
                        "needs_slow", ("slow_ok",), lambda read: bool(read["slow_ok"]), "no-slow"
                    ),
                ),
            ),
        ),
    )
    rules = (
        Constraint(
            "native_needs_core",
            ("realization", "fast_ok", "slow_ok"),
            lambda read: (
                read["realization"] != "native" or bool(read["fast_ok"] or read["slow_ok"])
            ),
            "no-core",
        ),
    )
    return Space(
        ("fast_ok", "slow_ok"),
        Scope(
            "",
            choices=(Choice("realization", listed("native", "other")),),
            children=(
                Scope(
                    "native_impl",
                    when=When(("realization",), lambda read: read["realization"] == "native"),
                    selectors=(cores,),
                ),
            ),
            constraints=rules if with_rule else (),
        ),
    )


CAPABILITIES = Spelling(types={"fast_ok": bool, "slow_ok": bool})
LOOK = Case("no look-ahead", realized(with_rule=False), CAPABILITIES)
RULED = Case("a parent rule over the capabilities", realized(with_rule=True), CAPABILITIES)
NO_CORES = {"fast_ok": False, "slow_ok": False}


def test_a_case_is_not_tried_against_a_selector_it_puts_in_scope() -> None:
    LOOK.start(NO_CORES).check_possible("realization")


def test_a_selector_a_case_puts_in_scope_waits_while_the_case_is_open() -> None:
    LOOK.start(NO_CORES).check_state("native_impl.compute")


def test_the_selector_a_set_case_puts_in_scope_stays_open() -> None:
    LOOK.start(NO_CORES, {"realization": "native"}).check_state("native_impl.compute")


def test_the_selector_a_set_case_puts_in_scope_leaves_the_configuration_open() -> None:
    LOOK.start(NO_CORES, {"realization": "native"}).check_validity()


def test_a_parent_rule_over_the_capabilities_makes_the_case_impossible() -> None:
    RULED.start(NO_CORES).check_possible("realization")


# -- a guard refused through its input ------------------------------------------------------------
#
# ``g`` is present while the derived ``gate`` exceeds 2; ``gate`` refuses an odd fact ``w``.
# A root rule reads ``g.n``. Refused through its input, the guard refuses its scope once, at
# the guard; the scope is not in play, so ``g.n`` is dormant and ``g.m`` absent.

GATED = Case(
    "a guard refused through its input",
    Space(
        ("w",),
        Scope(
            "",
            derived=(Derived("gate", ("w",), lambda r: Refuse("odd") if r["w"] % 2 else r["w"]),),
            children=(
                Scope(
                    "g",
                    when=When(("gate",), lambda r: r["gate"] > 2),
                    choices=(Choice("n", listed(1, 2)), Choice("m", listed(1, 2))),
                ),
            ),
            constraints=(Constraint("uses", ("g.n",), lambda r: r["g.n"] >= 1, "unused"),),
        ),
    ),
)


def test_new_facts_refusing_a_guards_input_are_reported_at_the_input() -> None:
    GATED.start({"w": 4}, {"g.n": 1}).check_change(Change(facts={"w": 3}))


def test_a_guard_refused_through_its_input_refuses_the_configuration_at_the_guard() -> None:
    GATED.start({"w": 3}).check_validity()


def test_an_unset_choice_under_a_guard_refused_through_its_input_is_absent() -> None:
    GATED.start({"w": 3}).check_state("g.m")


def test_a_rule_reading_a_choice_under_a_refused_guard_is_false_through_it() -> None:
    GATED.start({"w": 3}).check_rule("uses")


# The same guard over a scope holding a rule of its own, over the fact ``w``: refused through
# its input, the scope is not in play, so its rule does not apply when asked (§ 16.8); the
# guard's one piece refuses the configuration.

GATED_RULE = Case(
    "a rule under a guard refused through its input",
    Space(
        ("w",),
        Scope(
            "",
            derived=(Derived("gate", ("w",), lambda r: Refuse("odd") if r["w"] % 2 else r["w"]),),
            children=(
                Scope(
                    "g",
                    when=When(("gate",), lambda r: r["gate"] > 2),
                    constraints=(Constraint("fits", ("w",), lambda r: r["w"] < 9, "wide"),),
                ),
            ),
        ),
    ),
)


def test_a_rule_under_a_guard_refused_through_its_input_does_not_apply() -> None:
    GATED_RULE.start({"w": 3}).check_rule("g.fits")
