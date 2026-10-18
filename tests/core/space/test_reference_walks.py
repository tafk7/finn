# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded random walks: the engine against the reference model after every step.

Each seed draws a small description and a walk of changes over it
(``core.space.reference_harness.walks``). After every step both sides answer every
question; a disagreement whose kind a row of the acceptance matrix still names is
expected, any other fails the walk with its seed, step, description and both answers.
Each walk also checks the engine's refinement over commit-only steps and its walked
configuration against one built from nothing, on both index paths.

The gate runs a fixed set of seeds; ``python -m core.space.reference_harness walks`` (from
``tests``) runs more and prints what they met.
"""

from __future__ import annotations

import random

import pytest

from core.space.reference import (
    Change,
    Choice,
    Configuration,
    Constraint,
    Inactive,
    Rejected,
    Scope,
    Selector,
    Space,
    Unchecked,
    When,
    change,
    listed,
)
from core.space.reference_harness import MILESTONES, ROWS
from core.space.reference_harness.compare import state_form
from core.space.reference_harness.forms import (
    AnswerForm,
    ChangeForm,
    Form,
    Refusal,
    StateForm,
    ValidityForm,
)
from core.space.reference_harness.walks import (
    PRIORITY,
    Context,
    Question,
    Reach,
    added_unknown,
    change_mechanisms,
    classify,
    expected_kinds,
    generate,
    moved_down,
    table,
    unmet,
    violations,
    walk,
)

SEEDS = range(24)
STEPS = 30


@pytest.mark.parametrize("seed", SEEDS)
def test_a_seeded_walk_meets_no_disagreement_the_matrix_does_not_name(
    seed: int, index_path: str
) -> None:
    tally = walk(seed, STEPS, index_path)
    assert tally.failure is None, tally.failure
    assert tally.steps == STEPS
    assert tally.compared == tally.accepted + 1 == tally.cold_checks


def test_a_walk_is_the_same_for_the_same_seed() -> None:
    first, second = walk(5, 12), walk(5, 12)
    assert (first.steps, first.accepted, first.questions, first.kinds) == (
        second.steps,
        second.accepted,
        second.questions,
        second.kinds,
    )


def test_a_kind_no_row_names_fails_the_walk_with_its_seed_step_and_description() -> None:
    # Seed 28's third step (the fact f0 set to 1) leaves two forced choices excluding each
    # other: the model refuses c1 in a conflict, the engine forces it.
    assert walk(28, 3).failure is None
    failure = walk(28, 3, kinds=expected_kinds() - {"conflict"}).failure
    assert failure is not None
    assert failure.startswith("seed 28, index path, step 2: a disagreement no row names")
    assert "conflict" in failure and "description:" in failure and "| engine" in failure


def test_a_case_trial_reading_a_kept_choice_is_told_and_fails_unexplained() -> None:
    # Seed 7's third step: a choice of one of three cases committed under the open selector,
    # kept unchecked, that a rule of its case refuses. Its row explains it; without the row's
    # kind the walk fails there.
    explained = walk(7, 3)
    assert explained.failure is None
    assert explained.kinds["trial-reads-unchecked"]
    failure = walk(7, 3, kinds=expected_kinds() - {"trial-reads-unchecked"}).failure
    assert failure is not None
    assert failure.startswith("seed 7, index path, step 3: a disagreement no row names")
    assert "trial-reads-unchecked" in failure and "the state of s" in failure


def test_a_case_trial_reading_a_kept_choice_through_a_root_rule_is_told_as_the_same_kind() -> None:
    # Seed 12's eleventh step: a choice of one of three cases committed under the open
    # selector, kept unchecked, that a rule of the root refuses. The case's trial reads it as
    # its value through that rule as through one of the case's own: the same row's kind, told
    # ahead of the selector's case trial; without either kind the walk fails there.
    explained = walk(12, 11)
    assert explained.failure is None
    assert explained.kinds["trial-reads-unchecked"]
    failure = walk(
        12, 11, kinds=expected_kinds() - {"trial-reads-unchecked", "selector-case-trial"}
    ).failure
    assert failure is not None
    assert failure.startswith("seed 12, index path, step 11: a disagreement no row names")
    assert "trial-reads-unchecked, selector-case-trial" in failure
    assert "the state of s" in failure


def test_a_root_rule_refusing_a_kept_choice_of_a_case_is_told_as_a_trial_reading_it() -> None:
    space = Space(
        ("limit",),
        Scope(
            "",
            selectors=(
                Selector(
                    "core",
                    (
                        Scope("plain"),
                        Scope("other"),
                        Scope("wide", choices=(Choice("lanes", listed(1, 2)),)),
                    ),
                ),
            ),
            constraints=(
                Constraint(
                    "small",
                    ("core.wide.lanes", "limit"),
                    lambda read: read["core.wide.lanes"] <= read["limit"],
                    "too-many-lanes",
                ),
            ),
        ),
    )
    solution = Configuration(space, {"limit": 1}, {"core.wide.lanes": 2}).solve()
    assert isinstance(solution.state("core.wide.lanes"), Unchecked)
    assert solution.possible("core") == ("plain", "other")
    context = Context(solution, Reach(solution.index))
    assert context.found["trial-reads-unchecked"] == {"core"}
    engine = StateForm("open", possible=("plain", "other", "wide"))
    kinds = classify([context], ("state", "core"), state_form(solution, "core"), engine)
    assert kinds[:2] == ("trial-reads-unchecked", "selector-case-trial")


def test_a_value_case_making_a_kept_choice_present_is_told_as_a_trial_reading_it() -> None:
    # ``big`` is present when ``n`` is 3; its ``k``, committed while ``n`` is open, is kept
    # unchecked waiting on ``n``. The trial of 3 settles that presence and reads ``k`` as 3,
    # which ``under`` refuses: 3 is not possible. The engine, reading ``k`` unresolved there,
    # lists it.
    space = Space(
        (),
        Scope(
            "",
            choices=(Choice("n", listed(1, 2, 3)),),
            children=(
                Scope(
                    "big",
                    when=When(("n",), lambda read: read["n"] >= 3),
                    choices=(Choice("k", listed(3)),),
                    constraints=(
                        Constraint(
                            "under", ("big.k", "n"), lambda read: read["big.k"] < read["n"], "k"
                        ),
                    ),
                ),
            ),
        ),
    )
    solution = Configuration(space, {}, {"big.k": 3}).solve()
    assert isinstance(solution.state("big.k"), Unchecked)
    assert solution.possible("n") == (1, 2)
    context = Context(solution, Reach(solution.index))
    assert context.found["trial-reads-unchecked"] == {"n"}
    engine = StateForm("open", possible=(1, 2, 3))
    kinds = classify([context], ("state", "n"), state_form(solution, "n"), engine)
    assert kinds[:2] == ("trial-reads-unchecked", "value-case-trial")


def test_an_open_choice_on_an_inner_selectors_other_case_under_a_refused_outer_is_told() -> None:
    # Neither of ``s``'s cases is possible at ``f`` 1; ``t`` is admitted at ``y`` under it, so
    # ``m``, of ``t``'s case ``x``, is inactive, where the engine reads it under ``s``.
    inner = Selector("t", (Scope("x", choices=(Choice("m", listed(1, 2)),)), Scope("y")))
    space = Space(
        ("f",),
        Scope(
            "",
            selectors=(
                Selector(
                    "s",
                    (
                        Scope(
                            "a",
                            selectors=(inner,),
                            constraints=(
                                Constraint("ra", ("f",), lambda read: read["f"] > 5, "a"),
                            ),
                        ),
                        Scope(
                            "b",
                            constraints=(
                                Constraint("rb", ("f",), lambda read: read["f"] > 9, "b"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    solution = Configuration(space, {"f": 1}, {"s.a.t": "y"}).solve()
    assert isinstance(solution.state("s.a.t.x.m"), Inactive)
    context = Context(solution, Reach(solution.index))
    assert context.found["inactive-under-refused-selector"] == {"s.a.t.x.m"}


def test_a_case_a_named_value_outside_its_domain_no_longer_rules_out_is_told() -> None:
    # Neither case of ``s`` is possible: ``a``'s rule refuses the admitted ``c`` of 2 at a
    # limit of 1, and the root's rule refuses ``b``. Committing ``c`` at 9 (outside its
    # domain): forcing never reads it, so ``a`` is possible again and forced, and the retained
    # ``e`` of ``b`` is reached as inactive. An engine whose forcing reads 9 refuses ``c``
    # alone.
    space = Space(
        ("limit", "f"),
        Scope(
            "",
            selectors=(
                Selector(
                    "s",
                    (
                        Scope(
                            "a",
                            choices=(Choice("c", listed(1, 2, 3)),),
                            constraints=(
                                Constraint(
                                    "fits",
                                    ("s.a.c", "limit"),
                                    lambda read: read["s.a.c"] <= read["limit"],
                                    "too-big",
                                ),
                            ),
                        ),
                        Scope("b", choices=(Choice("e", listed(1, 2)),)),
                    ),
                ),
            ),
            constraints=(
                Constraint(
                    "no-b", ("s", "f"), lambda read: read["s"] != "b" or read["f"] >= 5, "no-b"
                ),
            ),
        ),
    )
    stored = {"s.a.c": 2, "s.b.e": 1}
    base = Configuration(space, {"limit": 1, "f": 1}, stored)
    result = change(base, Change(commit={"s.a.c": 9}))
    assert isinstance(result, Rejected)
    assert {(item.choice, item.kind) for item in result.refusals} == {
        ("s.a.c", "named"),
        ("s.b.e", "reached"),
    }
    published = Configuration(space, {"limit": 1, "f": 1}, {**stored, "s.a.c": 9})
    engine = ChangeForm(
        False,
        refusals=frozenset(
            {Refusal("s.a.c", "named", frozenset({("domain-membership", "s.a.c")}))}
        ),
    )
    found = change_mechanisms(base.solve(), published, published.solve(), result, engine)
    assert found["own-reason-forcing"] == {"s.a.c", "s.b.e"}


def test_a_case_trial_reading_a_choice_waiting_on_a_fact_is_told_and_fails_unexplained() -> None:
    # Seed 46's step 24: the engine refuses as inactive choices of a case the model keeps
    # unchecked, waiting on a fact.
    explained = walk(46, 24)
    assert explained.failure is None
    assert explained.kinds["trial-reads-waiting"]
    failure = walk(46, 24, kinds=expected_kinds() - {"trial-reads-waiting"}).failure
    assert failure is not None
    assert failure.startswith("seed 46, index path, step 24: a disagreement no row names")
    assert "trial-reads-waiting" in failure


def test_the_kinds_expected_without_a_milestones_rows_are_the_other_rows_kinds() -> None:
    # A milestone's engine is walked as if it had removed its rows: only the others' kinds.
    for milestone in MILESTONES:
        kept = {row.kind for row in ROWS if row.kind and row.milestone != milestone}
        assert expected_kinds((milestone,)) == kept
    assert expected_kinds(()) == {row.kind for row in ROWS if row.kind}
    assert expected_kinds(MILESTONES) == frozenset()
    with pytest.raises(ValueError, match="no such milestone"):
        expected_kinds(("lane 9",))


def test_a_walk_tells_only_kinds_a_row_names() -> None:
    # A kind no row names would only ever fail a walk; the classifier keeps none.
    assert set(PRIORITY) <= expected_kinds()


def test_the_summary_names_the_kinds_no_walk_met() -> None:
    tallies = [walk(28, 3)]
    assert "conflict" not in unmet(tallies)
    assert "kinds a row names that no walk met: " + ", ".join(unmet(tallies)) in table(tallies)


def test_descriptions_hold_two_to_six_choices_and_skip_cycles() -> None:
    rng = random.Random(7)
    cycles = 0
    for _ in range(40):
        generated, engine = generate(rng)
        cycles += generated.cycles
        assert 2 <= sum(info.cases is None for info in engine.index.choices.values()) <= 6
        assert generated.text.startswith("facts ")
    assert cycles  # a rule reading a view of its own Space is drawn now and then, and skipped


# -- the refinement property's order --------------------------------------------------------------


def test_an_answer_moving_down_the_information_order_is_named() -> None:
    assert (
        moved_down(StateForm("readable", 1), StateForm("unchecked", 1)) == "readable -> unchecked"
    )
    assert moved_down(StateForm("forced", 2), StateForm("open")) == "forced 2 -> open None"
    assert moved_down(StateForm("refused"), StateForm("open")) == "refused -> open"
    assert moved_down(AnswerForm("value", 3), AnswerForm("unresolved")) is not None
    assert moved_down(ValidityForm("valid"), ValidityForm("undecided")) == "valid -> undecided"
    assert moved_down(ValidityForm("refused"), ValidityForm("valid")) == "refused -> valid"


def test_an_answer_moving_up_or_into_a_refusal_is_not() -> None:
    assert moved_down(StateForm("open"), StateForm("forced", 1)) is None
    assert moved_down(StateForm("forced", 1), StateForm("refused")) is None
    assert moved_down(StateForm("unchecked", 1), StateForm("readable", 1)) is None
    assert moved_down(ValidityForm("undecided"), ValidityForm("valid")) is None
    # the committed choice: its forced value committed is no move; another value is
    assert moved_down(StateForm("forced", 1), StateForm("readable", 1), committed=True) is None
    assert moved_down(StateForm("forced", 1), StateForm("readable", 2), committed=True)


def test_nothing_moves_down_in_a_configuration_refused_after_the_step() -> None:
    before: dict[Question, Form] = {
        ("state", "x"): StateForm("readable", 1),
        ("validity", ""): ValidityForm("valid"),
    }
    refused: dict[Question, Form] = {
        ("state", "x"): StateForm("unchecked", 1),
        ("validity", ""): ValidityForm("refused"),
    }
    undecided: dict[Question, Form] = {**refused, ("validity", ""): ValidityForm("undecided")}
    assert violations(before, refused, ()) == {}
    assert set(violations(before, undecided, ())) == {("state", "x"), ("validity", "")}


def test_a_commitment_kept_unchecked_may_leave_validity_undecided_waiting_on_it() -> None:
    # The commitment adds a piece, its own check, unknown while its guard waits: validity
    # goes from valid to undecided through it, waiting on what it waits on.
    before: dict[Question, Form] = {
        ("state", "x"): StateForm("open", blockers=frozenset({"g"})),
        ("state", "y"): StateForm("readable", 1),
        ("validity", ""): ValidityForm("valid"),
    }
    after: dict[Question, Form] = {
        ("state", "x"): StateForm("unchecked", 2, blockers=frozenset({"g"})),
        ("state", "y"): StateForm("readable", 1),
        ("validity", ""): ValidityForm("undecided", blockers=frozenset({"g"})),
    }
    assert added_unknown(after, ("x",)) == {"g"}
    assert violations(before, after, ("x",)) == {}
    # waiting on anything else, it is a piece that existed before the step moving down
    other = {**after, ("validity", ""): ValidityForm("undecided", blockers=frozenset({"y"}))}
    assert set(violations(before, other, ("x",))) == {("validity", "")}
    # and a commitment checked adds no unknown piece
    checked = {**after, ("state", "x"): StateForm("readable", 2)}
    assert added_unknown(checked, ("x",)) == frozenset()
    assert set(violations(before, checked, ("x",))) == {("validity", "")}


def test_a_commitment_kept_unchecked_never_excuses_a_piece_that_existed_before() -> None:
    before: dict[Question, Form] = {
        ("state", "x"): StateForm("open", blockers=frozenset({"g"})),
        ("state", "y"): StateForm("readable", 1),
        ("validity", ""): ValidityForm("valid"),
    }
    after: dict[Question, Form] = {
        ("state", "x"): StateForm("unchecked", 2, blockers=frozenset({"g"})),
        ("state", "y"): StateForm("unchecked", 1, blockers=frozenset({"g"})),
        ("validity", ""): ValidityForm("undecided", blockers=frozenset({"g"})),
    }
    assert set(violations(before, after, ("x",))) == {("state", "y")}
