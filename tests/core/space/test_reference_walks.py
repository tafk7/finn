# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded random walks: the reference model's four invariants at every step, and the engine
compared with it (``core.space.reference_harness.walks``).

Each seed draws a small description and a walk of changes over it. At every step the
model's invariants hold strictly (history-independence and the bind, additions moving
answers only up the information order, no accepted change leaving a rule it reaches false
at its origin, every refusal naming exactly its origins and their standing choices), and
the engine's own checks hold (its walked configuration against one built from nothing).
While a milestone with rows is pending the engine's disagreements with the model are
counted; with them taken as removed, each fails its walk. Every invariant's checker is
also shown catching a deliberately wrong answer on a small description.

The gate runs a fixed set of seeds; ``python -m core.space.reference_harness walks`` (from
``tests``) runs more and prints what they met.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from core.space import reference
from core.space.reference import (
    Accepted,
    Change,
    Choice,
    Configuration,
    Constraint,
    Derived,
    Reason,
    ReasonKind,
    Refuse,
    Rejected,
    RuleRefusal,
    Scope,
    Selector,
    Space,
    View,
    When,
    change,
    listed,
)
from core.space.reference_harness.forms import (
    AnswerForm,
    ChangeForm,
    Form,
    OutcomeForm,
    PossibleForm,
    RuleForm,
    StateForm,
    ValidityForm,
)
from core.space.reference_harness.walks import (
    UNCLASSIFIED,
    VIEW_SPACE_REFUSED,
    Question,
    Structure,
    below,
    bind_problems,
    check_change,
    classify,
    differences,
    generate,
    model_forms,
    order_problems,
    pending,
    questions,
    table,
    view_space_refused,
    walk,
)

SEEDS = range(24)
STEPS = 30


@pytest.mark.parametrize("seed", SEEDS)
def test_a_seeded_walk_keeps_the_four_invariants_and_the_engines_own_checks(
    seed: int, index_path: str
) -> None:
    tally = walk(seed, STEPS, index_path)
    assert tally.failure is None, tally.failure
    assert tally.steps == STEPS and tally.checks["assignment"] == STEPS
    # every configuration the model reaches: the start and each accepted step
    assert tally.checks["1 history"] == tally.checks["1 bind"] == tally.accepted + 1
    assert tally.engine_checks["cold"] == tally.compared
    assert tally.engine_errors == 0


def test_the_gates_walks_exercise_every_invariant() -> None:
    tallies = [walk(seed, STEPS) for seed in range(6)]
    checks = sum((tally.checks for tally in tallies), start=type(tallies[0].checks)())
    for name in ("2 additions", "3 accepted", "4 refusals", "4 standing"):
        assert checks[name], (name, checks)
    assert sum(tally.model_only for tally in tallies)  # a fact unset, on the model alone


def test_a_walk_is_the_same_for_the_same_seed() -> None:
    first, second = walk(5, 12), walk(5, 12)
    assert (first.steps, first.accepted, first.checks, first.by_question, first.by_kind) == (
        second.steps,
        second.accepted,
        second.checks,
        second.by_question,
        second.by_kind,
    )


def test_descriptions_hold_two_to_six_choices_and_skip_cycles() -> None:
    rng = random.Random(7)
    skipped = 0
    for _ in range(40):
        generated, engine = generate(rng)
        skipped += generated.skipped["cycle"]
        assert 2 <= sum(info.cases is None for info in engine.index.choices.values()) <= 6
        assert generated.text.startswith("facts ")
    assert skipped  # a rule reading a view of its own Space is drawn now and then, and skipped


# -- the engine comparison: counted while a milestone is pending, strict without -------------------


def test_disagreements_are_counted_while_the_engine_core_has_rows() -> None:
    assert pending() == ("engine core",)
    tally = walk(0, 6)
    assert tally.failure is None, tally.failure
    assert sum(tally.by_question.values()) == sum(tally.by_kind.values()) > 0


def test_without_the_engine_cores_rows_every_disagreement_fails_its_walk() -> None:
    assert pending(("engine core",)) == ()
    tally = walk(0, 6, without=("engine core",))
    assert tally.failed == "disagreement" and tally.failure is not None
    first = tally.failure.splitlines()[0]
    assert first == "seed 0, index path, step 0: the engine disagrees (state, told as forced-value)"
    assert "description:" in tally.failure and "| engine" in tally.failure
    with pytest.raises(ValueError, match="no such milestone"):
        pending(("lane 9",))


def test_a_disagreement_is_told_by_one_simple_rule_or_left_unclassified() -> None:
    assert classify(StateForm("open"), StateForm("forced", 1)) == "forced-value"
    assert classify(StateForm("absent"), StateForm("refused")) == "refused-guard"
    assert classify(PossibleForm(((1, OutcomeForm("possible")),)), PossibleForm(None)) == (
        "viable-per-choice"
    )
    assert classify(ChangeForm(True, dormant=frozenset({"p"})), ChangeForm(True)) == (
        "no-dormant-report"
    )
    assert classify(AnswerForm("unresolved"), AnswerForm("value", 1)) == "forced-value"
    assert classify(StateForm("set", 1), StateForm("set", 2)) == UNCLASSIFIED


def test_the_summary_names_the_mode_the_checks_and_the_exception() -> None:
    tallies = [walk(0, 3)]
    text = table(tallies)
    checks = ", ".join(f"{name} {count}" for name, count in sorted(tallies[0].checks.items()))
    assert f"invariant checks: {checks}; binds refused as expected 0" in text
    assert "engine disagreements (counted (pending: engine core)) by question:" in text
    assert f"excused (§ 16.14): {VIEW_SPACE_REFUSED} 0" in text
    assert text.splitlines()[2].startswith("| 0 | index | 3 |")
    assert "strict (no milestone with rows pending)" in table(tallies, ("engine core",))


def test_a_broken_standing_list_in_the_model_fails_a_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    # The walk end to end: a model naming one standing choice too few fails at the step.
    standing: Callable[..., tuple[str, ...]] = reference.model._standing
    monkeypatch.setattr(reference.model, "_standing", lambda *args: standing(*args)[1:])
    tally = walk(11, 3)
    assert tally.failed == "4 standing" and tally.failure is not None
    assert tally.failure.startswith("seed 11, index path, step 3: invariant 4 standing: clearing")


# -- invariant 1: history-independence and the bind ------------------------------------------------


def linked() -> Space:
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("a", listed(1, 2, 3)), Choice("b", listed(1, 2, 3))),
            constraints=(Constraint("linked", ("a", "b"), lambda r: r["b"] <= r["a"], "b-above"),),
        ),
    )


def test_history_independence_names_each_answer_given_otherwise() -> None:
    walked = Configuration(linked(), {}, {"a": 2, "b": 1})
    asked = questions(walked.index)
    forms = model_forms(walked.solve(), asked)
    cold = Configuration(linked(), {}, {"a": 2, "b": 1})
    assert differences(forms, model_forms(cold.solve(), asked)) == []
    other = model_forms(Configuration(linked(), {}, {"a": 3, "b": 1}).solve(), asked)
    assert ("state", "a") in differences(forms, other)


def guarded_by_a_stored_choice() -> Space:
    """``d`` refuses an odd ``x``; ``g``, in case ``a`` of ``s``, is guarded by ``d``. Setting
    ``x`` to 1 while ``s`` is open is accepted (nothing in play rules on ``d``); selecting
    ``a`` then brings the guard into play, refused through ``d``, without reaching ``d``."""
    g = Scope("g", when=When(("d",), lambda r: r["d"] > 0), choices=(Choice("n", listed(1)),))
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2)),),
            selectors=(Selector("s", (Scope("a", children=(g,)), Scope("b"))),),
            derived=(Derived("d", ("x",), lambda r: Refuse("odd") if r["x"] % 2 else r["x"]),),
        ),
    )


def test_a_configuration_not_refused_binds_again_with_the_same_answers() -> None:
    walked = Configuration(linked(), {}, {"a": 2, "b": 1})
    structure = Structure(walked.index)
    bind = change(Configuration(linked()), Change(commit={"a": 2, "b": 1}))
    assert bind_problems(structure, walked, bind) == ([], False)
    refused = Rejected((RuleRefusal("linked", "constraint", (), ()),), None)
    problems, expected = bind_problems(structure, walked, refused)
    assert problems and not expected


def test_a_refused_walk_binds_refused_by_the_origin_its_stored_choice_reaches() -> None:
    start = Configuration(guarded_by_a_stored_choice())
    first = change(start, Change(commit={"x": 1}))
    assert isinstance(first, Accepted)
    walked = change(first.configuration, Change(commit={"s": "a"}))
    assert isinstance(walked, Accepted)
    assert walked.validity.verdict is reference.Verdict.FALSE
    structure = Structure(walked.configuration.index)
    bind = change(Configuration(guarded_by_a_stored_choice()), Change(commit={"x": 1, "s": "a"}))
    assert isinstance(bind, Rejected)
    assert bind_problems(structure, walked.configuration, bind) == ([], True)
    fake = Accepted(walked.configuration, (), (), (), walked.validity)
    problems, _ = bind_problems(structure, walked.configuration, fake)
    assert problems == ["1 bind: accepted, but refused by [('d', 'derived')]"]


# -- invariant 2: the information order ------------------------------------------------------------

UP: list[tuple[Form, Form]] = [
    (StateForm("open"), StateForm("absent")),
    (StateForm("open", blockers=frozenset({"s"})), StateForm("open")),
    (StateForm("waiting", 1, frozenset({"s"})), StateForm("set", 1)),
    (StateForm("waiting", 1, frozenset({"s"})), StateForm("dormant", 1)),
    (StateForm("set", 1), StateForm("set", 1)),
    (StateForm("absent"), StateForm("absent")),
    (AnswerForm("unresolved"), AnswerForm("refused")),
    (
        AnswerForm("refused", codes=frozenset({("a", "x")})),
        AnswerForm("refused", codes=frozenset({("a", "x"), ("b", "y")})),
    ),
    (RuleForm("unknown"), RuleForm("false")),
    (ValidityForm("open"), ValidityForm("refused")),
    (ValidityForm("open"), ValidityForm("valid")),
]

DOWN: list[tuple[Form, Form, str]] = [
    (StateForm("set", 1), StateForm("waiting", 1), "set -> waiting"),
    (StateForm("dormant", 1), StateForm("set", 1), "dormant -> set"),
    (StateForm("absent"), StateForm("open"), "absent -> open"),
    (StateForm("waiting", 1), StateForm("set", 2), "waiting 1 -> set 2"),
    (AnswerForm("value", 1), AnswerForm("unresolved"), "value -> unresolved"),
    (AnswerForm("value", 1), AnswerForm("refused"), "value -> refused"),
    (AnswerForm("value", 1), AnswerForm("value", True), "value 1 -> True"),
    (
        AnswerForm("refused", codes=frozenset({("a", "x")})),
        AnswerForm("refused"),
        "refused, losing [('a', 'x')]",
    ),
    (RuleForm("true"), RuleForm("false"), "true -> false"),
    (RuleForm("inapplicable"), RuleForm("unknown"), "inapplicable -> unknown"),
    (ValidityForm("valid"), ValidityForm("open"), "valid -> open"),
    (ValidityForm("refused"), ValidityForm("open"), "refused -> open"),
]


@pytest.mark.parametrize(("before", "after"), UP)
def test_an_answer_moving_up_the_information_order_is_no_move(before: Form, after: Form) -> None:
    assert below(before, after) is None


@pytest.mark.parametrize(("before", "after", "move"), DOWN)
def test_an_answer_moving_down_the_information_order_is_named(
    before: Form, after: Form, move: str
) -> None:
    assert below(before, after) == move


def test_a_choice_the_step_sets_holds_its_value_and_kept_out_of_play_turns_dormant() -> None:
    assert below(StateForm("open"), StateForm("set", 2), (2,)) is None
    assert (
        below(StateForm("open", blockers=frozenset({"s"})), StateForm("waiting", 2), (2,)) is None
    )
    assert below(StateForm("absent"), StateForm("dormant", 2), (2,)) is None
    assert below(StateForm("open"), StateForm("set", 3), (2,)) == "set to 2, but set 3"
    assert below(StateForm("absent"), StateForm("set", 2), (2,)) == "absent, set: set"


def views_of_a_space_refused_by_an_addition() -> Space:
    """The probe of § 16.14: ``g``, in case ``a`` of the open ``s``, is guarded by ``d``,
    which refuses the fact ``f`` when odd; the root's view ``v`` reads ``f``, and the root's
    rule ``uses`` reads ``v``."""
    g = Scope("g", when=When(("d",), lambda r: r["d"] > 0), choices=(Choice("n", listed(1, 2)),))
    return Space(
        ("f",),
        Scope(
            "",
            selectors=(Selector("s", (Scope("a", children=(g,)), Scope("b"))),),
            derived=(Derived("d", ("f",), lambda r: Refuse("odd") if r["f"] % 2 else r["f"]),),
            views=(View("v", ("f",), lambda r: r["f"]),),
            children=(
                Scope(
                    "k",
                    views=(View("w", ("f",), lambda r: r["f"]),),
                    selectors=(Selector("t", (Scope("x", children=(g,)), Scope("y"))),),
                ),
            ),
            constraints=(Constraint("uses", ("k.w",), lambda r: r["k.w"] > 0, "unused"),),
        ),
    )


def test_a_view_whose_space_an_addition_leaves_refused_is_told_as_the_one_exception() -> None:
    # § 16.14: selecting ``a`` is accepted (``d`` refuses on the facts alone and is not
    # reached), the root reads refused, so its view ``v`` moves from a value to refused.
    # In the child ``k`` the same: selecting ``x`` leaves ``k`` refused, its view ``w``
    # refused, and the root's rule ``uses`` reading it moves from true to false.
    start = Configuration(views_of_a_space_refused_by_an_addition(), {"f": 3})
    structure = Structure(start.index)
    asked = questions(start.index)
    before = model_forms(start.solve(), asked)
    for selector, moved in (("s", ("answer", "v")), ("k.t", ("rule", "uses"))):
        edit = Change(commit={selector: "a" if selector == "s" else "x"})
        result = change(start, edit)
        assert isinstance(result, Accepted)
        after_solution = result.configuration.solve()
        after = model_forms(after_solution, asked)

        def excuse(question: Question) -> bool:
            return view_space_refused(
                structure, after_solution, question, before[question], after[question]
            )

        found, told = order_problems(before, after, edit.commit, excuse)
        assert found == {} and moved in told
        strict, _ = order_problems(before, after, edit.commit)
        assert moved in strict  # without the exception, a move down


def test_the_exception_covers_nothing_but_a_view_whose_space_reads_refused() -> None:
    start = Configuration(views_of_a_space_refused_by_an_addition(), {"f": 3})
    solution = start.solve()
    structure = Structure(start.index)
    # the root is not refused here: no exception
    assert not view_space_refused(
        structure, solution, ("answer", "v"), AnswerForm("value", 3), AnswerForm("refused")
    )
    # a choice's answer is never one
    assert not view_space_refused(
        structure, solution, ("answer", "s"), AnswerForm("value", "a"), AnswerForm("refused")
    )


# -- invariants 3 and 4: no reached origin left false; refusals named exactly ----------------------


def summed() -> Space:
    return Space(
        (),
        Scope(
            "",
            choices=tuple(Choice(name, listed(0, 1, 2, 3)) for name in ("a", "b", "c")),
            constraints=(
                Constraint("sum", ("a", "b", "c"), lambda r: r["a"] + r["b"] + r["c"] <= 4, "big"),
            ),
        ),
    )


SUM = Configuration(summed(), {}, {"b": 2, "c": 2})
NEW_A = Change(commit={"a": 1})
BIG = (Reason(ReasonKind.CONSTRAINT, "sum", "", "big"),)


def test_the_models_refusal_names_the_rule_and_both_standing() -> None:
    result = change(SUM, NEW_A)
    assert isinstance(result, Rejected)
    assert result.refusals == (RuleRefusal("sum", "constraint", BIG, ("b", "c")),)
    assert check_change(Structure(SUM.index), SUM, NEW_A, result).problems == []


def test_a_refusal_missing_a_standing_choice_is_caught_by_clearing_it() -> None:
    fake = Rejected((RuleRefusal("sum", "constraint", BIG, ("b",)),), None)
    problems = check_change(Structure(SUM.index), SUM, NEW_A, fake).problems
    assert problems == ["4 standing: clearing c clears sum, named standing ['b']"]


def test_a_refusal_naming_another_origin_or_none_is_caught() -> None:
    structure = Structure(SUM.index)
    none = Rejected((), None)
    assert check_change(structure, SUM, NEW_A, none).problems[0].startswith("4 refusals: named []")
    published = Configuration(summed(), {}, {"a": 1, "b": 2, "c": 2})
    fake = Accepted(published, (), (), (), published.solve().validity())
    assert check_change(structure, SUM, NEW_A, fake).problems == [
        "3 accepted: the change is accepted, but leaves false at their origin rules it reaches: "
        "[('sum', 'constraint')]"
    ]


SMALL = (Reason(ReasonKind.CONSTRAINT, "s.a.small", "s.a", "big"),)
HIGH = (Reason(ReasonKind.CONSTRAINT, "low", "", "high"),)


def case_rule_over_the_root() -> Space:
    """A rule of case ``a`` reads the root's ``x`` alone: selecting ``a`` reaches it."""
    rule = Constraint("small", ("x",), lambda r: r["x"] <= 1, "big")
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2)),),
            selectors=(Selector("s", (Scope("a", constraints=(rule,)), Scope("b"))),),
        ),
    )


def test_an_accepted_switch_leaving_a_case_rule_false_is_caught_through_presence() -> None:
    base = Configuration(case_rule_over_the_root(), {}, {"x": 2})
    edit = Change(commit={"s": "a"})
    result = change(base, edit)
    assert isinstance(result, Rejected)
    assert result.refusals == (RuleRefusal("s.a.small", "constraint", SMALL, ("x",)),)
    structure = Structure(base.index)
    assert check_change(structure, base, edit, result).problems == []
    published = Configuration(case_rule_over_the_root(), {}, {"x": 2, "s": "a"})
    fake = Accepted(published, (), (), (), published.solve().validity())
    assert check_change(structure, base, edit, fake).problems[0].startswith("3 accepted:")


def choice_under_a_case_read_by_the_root() -> Space:
    """The root's rule reads ``s.a.p``: ``s`` decides whether that read exists."""
    return Space(
        (),
        Scope(
            "",
            selectors=(
                Selector("s", (Scope("a", choices=(Choice("p", listed(1, 2, 3)),)), Scope("b"))),
            ),
            constraints=(Constraint("low", ("s.a.p",), lambda r: r["s.a.p"] <= 2, "high"),),
        ),
    )


def test_a_choice_deciding_only_whether_a_read_exists_never_stands() -> None:
    # Clearing ``s`` would make ``low`` wait, but ``s`` is not among its reads: not standing
    # (§ 16.4), and the check by clearing does not ask it to be.
    base = Configuration(choice_under_a_case_read_by_the_root(), {}, {"s": "a", "s.a.p": 1})
    edit = Change(commit={"s.a.p": 3})
    result = change(base, edit)
    assert isinstance(result, Rejected)
    assert result.refusals == (RuleRefusal("low", "constraint", HIGH, ()),)
    assert check_change(Structure(base.index), base, edit, result).problems == []
    fake = Rejected((RuleRefusal("low", "constraint", (), ("s",)),), None)
    problems = check_change(Structure(base.index), base, edit, fake).problems
    assert problems == ["4 standing: low names ['s'], not standing by any reading"]


def test_an_assignment_refusal_is_checked_against_the_type_alone() -> None:
    edit = Change(commit={"a": 9})
    result = change(SUM, edit)
    assert check_change(Structure(SUM.index), SUM, edit, result).problems == []
    fake = Rejected((RuleRefusal("sum", "constraint", BIG, ("b", "c")),), None)
    problems = check_change(Structure(SUM.index), SUM, edit, fake).problems
    assert problems == ["assignment: values not of their type ['a'], refused as such []"]
