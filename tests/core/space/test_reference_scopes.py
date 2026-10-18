# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine against the reference model: views, nested scopes and look-ahead, one test
per question.

Case 9 of the reference cases (views and validity, and a constraint reading a view of its
own Space, a cycle, in three shapes), and two of how a Space holds cores: selectors nested
three deep, and a value choice whose case puts a selector with no possible case in scope.
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
    View,
    When,
    listed,
)
from core.space.reference_harness import (
    ByName,
    Case,
    Pair,
    Spelling,
    check_definition,
    expected,
)

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")


def reach(pair: Pair, edit: Change) -> Pair:
    """``edit`` compared, then its successor (a refusal by the engine is a disagreement)."""
    __tracebackhide__ = True
    successor = pair.check_change(edit)
    assert successor is not None, "the model refuses the change: the test is written wrongly"
    return successor


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


@expected("view-of-refused-space")
def test_a_view_refuses_when_its_space_is_refused() -> None:
    VIEWED.start({"limit": 2, "name": "bad"}).check_answer("title")


def test_a_view_answers_while_its_space_is_undecided() -> None:
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


@expected("cycle-storage-view")
def test_a_constraint_on_a_storage_view_of_its_own_space_is_a_cycle() -> None:
    check_definition("a constraint on a storage view", STORAGE)


@expected("cycle-by-name")
def test_a_constraint_reading_views_of_its_own_space_by_name_is_a_cycle() -> None:
    check_definition("views read by name", BY_NAME, Spelling(rules={"carried": ByName()}))


@expected("cycle-own-stages")
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


def test_a_rule_three_deep_under_two_open_selectors_waits_on_both() -> None:
    NESTED.start(OPEN).check_rule(FITS)


def test_a_rule_three_deep_under_one_open_selector_waits_on_it() -> None:
    NESTED.start(OPEN, {"kernel": "inner"}).check_rule(FITS)


def test_a_rule_three_deep_under_an_unselected_case_does_not_apply() -> None:
    NESTED.start(OPEN, {"kernel": "plain"}).check_rule(FITS)


def test_a_choice_three_deep_under_an_unselected_case_is_inactive() -> None:
    NESTED.start(OPEN, {"kernel": "plain"}).check_state(PE)


@expected("nested-outer-forced")
def test_forcing_selects_the_outer_kernel() -> None:
    NESTED.start(CLOSED).check_state("kernel")


@expected("nested-inner-forced")
def test_forcing_selects_the_core_inside_the_forced_kernel() -> None:
    NESTED.start(CLOSED).check_state(COMPUTE)


@expected("nested-possible")
def test_a_choice_three_deep_under_forced_selectors_lists_its_possible_cases() -> None:
    NESTED.start(CLOSED).check_state(PE)


@expected("nested-commit-forced")
def test_a_commit_three_deep_under_forced_selectors_is_accepted() -> None:
    NESTED.start(CLOSED).check_change(Change(commit={PE: 2}))


@expected("nested-readable")
def test_a_commit_three_deep_under_forced_selectors_is_readable() -> None:
    reach(NESTED.start(CLOSED), Change(commit={PE: 2})).check_state(PE)


def test_validity_counts_a_rule_three_deep() -> None:
    point = NESTED.start({**OPEN, "width": 6}, {"kernel": "inner", COMPUTE: "packed"})
    reach(point, Change(commit={PE: 4})).check_validity()


@expected("nested-commit-under-open-selectors")
def test_a_commit_three_deep_under_two_open_selectors_is_kept_unchecked() -> None:
    NESTED.start(OPEN).check_change(Change(commit={PE: 2}))


@expected("nested-conflict-commit")
def test_a_commit_three_deep_in_a_conflict_with_the_forced_selectors_is_accepted() -> None:
    # 4 does not divide 6, so ``packed`` is refused; ``serial`` is refused by its fact, so
    # ``compute`` has no possible case; ``kernel`` was forced to ``inner`` only because
    # ``plain`` is refused. Both selectors are refused, and ``pe`` keeps its value.
    NESTED.start(CLOSED).check_change(Change(commit={PE: 4}))


@expected("nested-conflict-choice")
def test_a_choice_three_deep_in_a_conflict_keeps_its_value_refused_through_its_guards() -> None:
    reach(NESTED.start(CLOSED), Change(commit={PE: 4})).check_state(PE)


@expected("nested-conflict-outer")
def test_the_outer_selector_in_a_conflict_with_a_choice_three_deep_is_refused() -> None:
    reach(NESTED.start(CLOSED), Change(commit={PE: 4})).check_state("kernel")


@expected("nested-conflict-inner")
def test_the_inner_selector_in_a_conflict_with_a_choice_three_deep_is_refused() -> None:
    reach(NESTED.start(CLOSED), Change(commit={PE: 4})).check_state(COMPUTE)


@expected("nested-conflict-validity")
def test_a_conflict_with_a_choice_three_deep_refuses_the_configuration() -> None:
    reach(NESTED.start(CLOSED), Change(commit={PE: 4})).check_validity()


# -- a look-ahead against a choice with no possible value ------------------------------------------
#
# ``realization`` is ``native`` or ``other``; ``native`` puts a scope in place whose selector
# chooses a core, ``fast`` or ``slow``, each needing its capability. With neither
# capability the selector has no possible case once ``native`` is chosen.


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


def test_a_case_is_not_tried_against_another_choice_having_no_possible_value() -> None:
    LOOK.start(NO_CORES).check_state("realization")


@expected("look-ahead-conflict")
def test_the_selector_a_committed_case_puts_in_scope_has_no_possible_value() -> None:
    LOOK.start(NO_CORES, {"realization": "native"}).check_state("native_impl.compute")


@expected("look-ahead-conflict-validity")
def test_the_selector_a_committed_case_puts_in_scope_refuses_the_configuration() -> None:
    LOOK.start(NO_CORES, {"realization": "native"}).check_validity()


@expected("look-ahead-parent-rule")
def test_a_parent_rule_over_the_capabilities_makes_the_case_impossible() -> None:
    RULED.start(NO_CORES).check_state("realization")


@expected("look-ahead-inactive")
def test_the_selector_under_the_impossible_case_is_inactive() -> None:
    RULED.start(NO_CORES).check_state("native_impl.compute")
