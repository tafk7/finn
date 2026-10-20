# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded random walks: the reference model's four invariants at every step, and the engine
compared with it (``core.space.reference_harness.walks``).

Each seed draws a small description and a walk of changes over it. At every step the
model's invariants hold strictly (history-independence and the bind publishing it with
nothing broken, additions moving answers only up the information order, no accepted change
adding a (false piece, origin) pair, every refusal naming exactly its origins and their
standing choices), and the engine's own checks hold (its walked configuration against one
built from nothing). No milestone with rows is pending, so each disagreement with the model
fails its walk; while a later one is, they are counted, and with its rows taken as removed,
each fails its walk. Every invariant's checker is
also shown catching a deliberately wrong answer on a small description.

The gate runs a fixed set of seeds; ``python -m core.space.reference_harness walks`` (from
``tests``) runs more and prints what they met.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import replace

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
    computed,
    listed,
)
from core.space.reference_harness import Row, walks
from core.space.reference_harness.engine import Point
from core.space.reference_harness.forms import (
    AnswerForm,
    Form,
    RuleForm,
    StateForm,
    ValidityForm,
)
from core.space.reference_harness.walks import (
    Pair,
    Question,
    Structure,
    below,
    bind_problems,
    check_change,
    check_facts,
    counted_pairs,
    differences,
    generate,
    masking_problems,
    model_forms,
    order_problems,
    pending,
    questions,
    table,
    walk,
)

SEEDS = range(24)
STEPS = 30


@pytest.mark.parametrize("seed", SEEDS)
def test_a_seeded_walk_keeps_the_four_invariants_and_the_engines_own_checks(
    seed: int,
) -> None:
    tally = walk(seed, STEPS)
    assert tally.failure is None, tally.failure
    # each step is a change of choices (its assignment checked) or a facts step; a batch's
    # choices are checked as a change of choices too
    checks = tally.checks
    assert tally.steps == STEPS
    assert checks["assignment"] + checks["facts"] - checks["facts batch"] == STEPS
    # every configuration the model reaches: the start and each accepted step
    assert checks["1 history"] == checks["1 bind"] == tally.accepted + 1
    # the engine compared at each, and its bind of each one's own choices against the model's
    assert tally.engine_checks["cold"] == tally.engine_checks["bind"] == tally.compared
    assert tally.engine_errors == 0


def test_the_gates_walks_exercise_every_invariant() -> None:
    tallies = [walk(seed, STEPS) for seed in range(6)]
    checks = sum((tally.checks for tally in tallies), start=type(tallies[0].checks)())
    for name in (
        "2 additions",
        "3 accepted",
        "4 refusals",
        "4 standing",
        "facts",
        "facts dropped",
        "facts report",
        "facts batch",
    ):
        assert checks[name], (name, checks)
    assert sum(tally.model_only for tally in tallies)  # a fact unset, on the model alone
    # a design space a facts step left reading refused binds again, on both sides
    assert sum(tally.binds_broken for tally in tallies)


def test_a_walk_is_the_same_for_the_same_seed() -> None:
    first, second = walk(5, 12), walk(5, 12)
    assert (first.steps, first.accepted, first.checks, first.by_question) == (
        second.steps,
        second.accepted,
        second.checks,
        second.by_question,
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
#
# The engine agrees with the model on every walk, so a disagreement is made here: the engine
# answers every choice's state as absent (the model's start holds an open choice), and a
# milestone with one row is put in the matrix while one is pending.


@pytest.fixture
def engine_disagrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine answers every choice's state as absent: consistently, so its own checks
    (walked against built from nothing) still hold."""
    asked = walks.engine_forms

    def absent(point: Point, questions: Iterable[Question]) -> dict[Question, Form]:
        forms = asked(point, questions)
        return {q: StateForm("absent") if q[0] == "state" else form for q, form in forms.items()}

    monkeypatch.setattr(walks, "engine_forms", absent)


def pend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later milestone, ``lane 9``, with one row: pending."""
    row = Row("a-row", "lane 9", "the change", "test_reference_cases::test_it")
    monkeypatch.setattr(walks, "MILESTONES", ("base", "lane 9"))
    monkeypatch.setattr(walks, "ROWS", (row,))


@pytest.mark.usefixtures("engine_disagrees")
def test_disagreements_are_counted_while_a_milestone_has_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pend(monkeypatch)
    assert pending() == ("lane 9",)
    tally = walk(0, 6)
    assert tally.failure is None, tally.failure
    assert tally.by_question["state"] > 0 and tally.engine_errors == 0


@pytest.mark.usefixtures("engine_disagrees")
def test_with_no_milestone_pending_every_disagreement_fails_its_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(walks, "ROWS", ())
    assert pending() == ()
    strict = walk(0, 6)
    pend(monkeypatch)
    assert pending(("lane 9",)) == ()
    without = walk(0, 6, without=("lane 9",))
    for tally in (strict, without):
        assert tally.failed == "disagreement" and tally.failure is not None
        first = tally.failure.splitlines()[0]
        assert first == "seed 0, step 0: the engine disagrees (state)"
        assert "description:" in tally.failure and "| engine" in tally.failure
    with pytest.raises(ValueError, match="no such milestone"):
        pending(("lane 10",))


def test_the_summary_names_the_mode_and_the_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    tallies = [walk(0, 3)]
    pend(monkeypatch)
    text = table(tallies)
    checks = ", ".join(f"{name} {count}" for name, count in sorted(tallies[0].checks.items()))
    assert (
        f"invariant checks: {checks}; binds reading refused, reporting rules a clear revealed "
        "(§ 16.16) 0, or new facts broke (§ 16.24) 0" in text
    )
    assert "engine disagreements (counted (pending: lane 9)) by question:" in text
    assert "excused" not in text
    assert text.splitlines()[2].startswith("| 0 | 3 |")
    assert "strict (no milestone with rows pending)" in table(tallies, ("lane 9",))


def test_a_broken_standing_list_in_the_model_fails_a_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    # The walk end to end: a model naming one standing choice too few fails at the step. The
    # engine, unbroken, disagrees with it sooner (its possible values name the standing
    # choices too): counted, under a milestone pending, so the invariant is what fails.
    standing: Callable[..., tuple[str, ...]] = reference.model._standing
    monkeypatch.setattr(reference.model, "_standing", lambda *args: standing(*args)[1:])
    pend(monkeypatch)
    tally = walk(11, 3)
    assert tally.failed == "4 standing" and tally.failure is not None
    assert tally.failure.startswith("seed 11, step 3: invariant 4 standing: clearing")


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
    ``a`` then makes the guard's piece false through ``d``, a new pair."""
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


ODD = (Reason(ReasonKind.DERIVED_BODY, "d", "", "odd"),)


def test_a_configuration_binds_again_with_nothing_broken() -> None:
    walked = Configuration(linked(), {}, {"a": 2, "b": 1})
    bound = reference.bind(linked(), {}, {"a": 2, "b": 1})
    assert bind_problems(walked, bound) == ([], frozenset())
    broke = replace(bound, broken=(RuleRefusal("linked", "constraint", (), ()),))
    assert bind_problems(walked, broke) == (
        [
            "1 bind: reports [('linked', 'constraint')]; what a walk publishes binds again "
            "with nothing broken (FN3), but for rules a clear revealed ([])"
        ],
        frozenset({"linked"}),
    )
    moved = reference.bind(linked(), {}, {"a": 3, "b": 1})
    assert bind_problems(walked, moved)[0] == [
        "1 bind: the bind of a design space's own choices drops or moves one"
    ]


def test_a_walk_never_reaches_a_configuration_whose_bind_is_refused() -> None:
    # § 16.15: the switch that would leave the guard false through ``d`` is refused, with
    # ``x`` standing, so a walk never holds the configuration; built directly, it is legal
    # and reads refused, and its bind is refused, which invariant 1 names.
    start = Configuration(guarded_by_a_stored_choice())
    first = change(start, Change(commit={"x": 1}))
    assert isinstance(first, Accepted)
    edit = Change(commit={"s": "a"})
    switch = change(first.configuration, edit)
    assert isinstance(switch, Rejected)
    assert switch.refusals == (RuleRefusal("d", "derived", ODD, ("x",)),)
    structure = Structure(start.index)
    assert check_change(structure, first.configuration, edit, switch).problems == []
    built = Configuration(guarded_by_a_stored_choice(), {}, {"x": 1, "s": "a"})
    assert built.solve().validity().verdict is reference.Verdict.FALSE
    bound = reference.bind(guarded_by_a_stored_choice(), {}, {"x": 1, "s": "a"})
    problems, _ = bind_problems(built, bound)
    assert problems == [
        "1 bind: reports [('d', 'derived')]; what a walk publishes binds again with nothing "
        "broken (FN3), but for rules a clear revealed ([])"
    ]


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


def test_an_addition_that_would_leave_a_space_refused_is_refused() -> None:
    # § 16.15 removes § 16.14's exception: selecting ``a`` (or ``k``'s ``x``) makes the guard's
    # piece false through ``d``, over the facts alone: a new pair, refused with nothing
    # standing. Were it accepted, the root's view ``v`` (or the rule ``uses``, through ``k``'s
    # view ``w``) would move down the information order, and invariants 2 and 3 name it.
    start = Configuration(views_of_a_space_refused_by_an_addition(), {"f": 3})
    structure = Structure(start.index)
    asked = questions(start.index)
    before = model_forms(start.solve(), asked)
    for selector, case, moved in (("s", "a", ("answer", "v")), ("k.t", "x", ("rule", "uses"))):
        edit = Change(commit={selector: case})
        result = change(start, edit)
        assert isinstance(result, Rejected)
        assert result.refusals == (RuleRefusal("d", "derived", ODD, ()),)
        assert check_change(structure, start, edit, result).problems == []
        published = Configuration(start.space, {"f": 3}, {selector: case})
        fake = Accepted(published, (), (), (), published.solve().validity())
        problems = check_change(structure, start, edit, fake).problems
        assert len(problems) == 1 and problems[0].startswith("3 accepted: ")
        assert moved in order_problems(before, model_forms(published.solve(), asked), edit.commit)


# -- invariants 3 and 4: no pair added; refusals named exactly ------------------------------------


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
        "3 accepted: the change is accepted, but adds (false piece, origin) pairs: "
        "[(('constraint', 'sum'), 'sum')]"
    ]


SMALL = (Reason(ReasonKind.CONSTRAINT, "s.a.small", "s.a", "big"),)
HIGH = (Reason(ReasonKind.CONSTRAINT, "low", "", "high"),)


def case_rule_over_the_root() -> Space:
    """A rule of case ``a`` reads the root's ``x`` alone: selecting ``a`` brings it in."""
    rule = Constraint("small", ("x",), lambda r: r["x"] <= 1, "big")
    return Space(
        (),
        Scope(
            "",
            choices=(Choice("x", listed(1, 2)),),
            selectors=(Selector("s", (Scope("a", constraints=(rule,)), Scope("b"))),),
        ),
    )


def test_an_accepted_switch_leaving_a_case_rule_newly_false_is_caught() -> None:
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


def short() -> Space:
    """``d`` is ``x + z``, refusing an odd sum; ``r`` asks ``d`` to be at least ``y``."""
    return Space(
        (),
        Scope(
            "",
            choices=tuple(Choice(name, listed(1, 2, 3, 4)) for name in ("x", "y", "z")),
            derived=(
                Derived(
                    "d",
                    ("x", "z"),
                    lambda r: Refuse("odd") if (r["x"] + r["z"]) % 2 else r["x"] + r["z"],
                ),
            ),
            constraints=(Constraint("r", ("d", "y"), lambda r: r["d"] >= r["y"], "short"),),
        ),
    )


def test_the_counted_pairs_name_each_false_piece_with_the_origins_it_is_false_through() -> None:
    structure = Structure(Configuration(short()).index)
    broken = Configuration(short(), {}, {"x": 1, "y": 4, "z": 1}).solve()
    assert counted_pairs(structure, broken) == {(("constraint", "r"), "r"): "constraint"}
    odd = Configuration(short(), {}, {"x": 2, "y": 4, "z": 1}).solve()
    assert counted_pairs(structure, odd) == {(("constraint", "r"), "d"): "derived"}
    # a broken configuration: a change keeping its pair is accepted, one adding a pair not
    base = Configuration(short(), {}, {"x": 1, "y": 4, "z": 1})
    for edit, kind in ((Change(commit={"y": 3}), Accepted), (Change(commit={"x": 2}), Rejected)):
        result = change(base, edit)
        assert isinstance(result, kind)
        assert check_change(structure, base, edit, result).problems == []


# -- clears and unsets (§ 16.16): never refused, adding a pair only by masking ---------------------


def viewed_child() -> Space:
    """The child ``c`` caps ``x``; its view ``v`` answers ``y``; the root's ``high`` asks
    ``v`` to be at least 2: while ``c`` is refused, ``high`` is false through ``v`` alone."""
    child = Scope(
        "c",
        choices=(Choice("x", listed(1, 2, 3)), Choice("y", listed(1, 2, 3))),
        constraints=(Constraint("small", ("c.x",), lambda r: r["c.x"] <= 1, "big"),),
        views=(View("v", ("c.y",), lambda r: r["c.y"]),),
    )
    return Space(
        (),
        Scope(
            "",
            children=(child,),
            constraints=(Constraint("high", ("c.v",), lambda r: r["c.v"] >= 2, "low"),),
        ),
    )


def test_a_clear_revealing_a_masked_rule_is_accepted_and_its_bind_refusal_counted() -> None:
    masked = Configuration(viewed_child(), {}, {"c.x": 3, "c.y": 1})
    edit = Change(clear=("c.x",))
    result = change(masked, edit)
    assert isinstance(result, Accepted)
    checked = check_change(Structure(masked.index), masked, edit, result)
    assert (checked.problems, checked.revealed, checked.runs["3 clears"]) == ([], {"high"}, 1)
    # refused, a clear-only change is a problem whatever it adds
    fake = Rejected((RuleRefusal("high", "constraint", (), ()),), None)
    assert check_change(Structure(masked.index), masked, edit, fake).problems == [
        "3 clears: a change made only of clears and unsets is refused (§ 16.16)"
    ]
    # its bind reports the revealed rule: expected after the clear, and only then
    revealed = result.configuration
    bound = reference.bind(viewed_child(), {}, revealed.stored)
    assert [item.origin for item in bound.broken] == ["high"]
    assert bind_problems(revealed, bound, {"high"}) == ([], frozenset({"high"}))
    assert bind_problems(revealed, bound)[0]


def test_a_clear_only_step_adding_a_pair_any_other_way_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No clear adds such a pair under the model's rules; the checker is shown it by hand. The
    # sum read no view and no fact, so the pair it adds is no masking revealed, and the walk
    # fails on the problem (as on every problem check_change returns).
    before = Configuration(summed(), {}, {"b": 2, "c": 2})
    edit = Change(clear=("c",))
    pair: Pair = (("constraint", "sum"), "sum")
    assert masking_problems(
        Structure(before.index), before.solve(), edit, {pair: "constraint"}
    ) == [
        "3 clears: the clear-only change adds (('constraint', 'sum'), 'sum'), not by revealing a "
        "rule masked by a view's Space or an unset fact (§ 16.16)"
    ]
    monkeypatch.setattr(walks, "added_pairs", lambda *args: {pair: "constraint"})
    result = change(before, edit)
    problems = check_change(Structure(before.index), before, edit, result).problems
    assert len(problems) == 1 and problems[0].startswith("3 clears: the clear-only change adds")


# -- facts steps (§ 16.24): never refused, checked against a bind of their result -----------------


def bounded() -> Space:
    """``c`` and ``x`` from 1 to 4, ``k`` asks ``c`` to be at most the fact ``f``; ``n``
    divides ``f``."""
    return Space(
        ("f",),
        Scope(
            "",
            choices=(
                Choice("c", listed(1, 2, 3, 4)),
                Choice("x", listed(1, 2, 3, 4)),
                Choice(
                    "n", computed(lambda r: tuple(d for d in (1, 2, 4) if r["f"] % d == 0), ("f",))
                ),
            ),
            constraints=(Constraint("k", ("c", "f"), lambda r: r["c"] <= r["f"], "c-over"),),
        ),
    )


BOUNDED = Configuration(bounded(), {"f": 4}, {"c": 3, "n": 4})
FALLS = Change(facts={"f": 2})
K_OVER = RuleRefusal("k", "constraint", (Reason(ReasonKind.CONSTRAINT, "k", "", "c-over"),), ("c",))


def test_a_facts_step_publishes_the_bind_of_its_stored_choices_and_reports_its_new_rules() -> None:
    # ``f = 2``: ``n = 4`` no longer divides it and is dropped; ``k`` turns false with ``c``
    # standing. Every check holds, and each one runs.
    result = change(BOUNDED, FALLS)
    assert isinstance(result, Accepted)
    assert [item.choice for item in result.dropped] == ["n"] and result.broken == (K_OVER,)
    checked = check_facts(Structure(BOUNDED.index), BOUNDED, FALLS, result)
    assert checked.problems == []
    assert (checked.runs["facts"], checked.runs["facts dropped"], checked.runs["facts report"]) == (
        1,
        1,
        1,
    )
    assert check_change(Structure(BOUNDED.index), BOUNDED, FALLS, result).runs == checked.runs


def test_a_facts_step_refused_or_publishing_otherwise_than_its_bind_is_caught() -> None:
    structure = Structure(BOUNDED.index)
    refused = Rejected((K_OVER,), None)
    assert check_facts(structure, BOUNDED, FALLS, refused).problems == [
        "facts: a change of facts alone is refused (§ 16.24)"
    ]
    result = change(BOUNDED, FALLS)
    assert isinstance(result, Accepted)
    # ``n`` left out, but not reported dropped
    unreported = replace(result, dropped=())
    problems = check_facts(structure, BOUNDED, FALLS, unreported).problems
    assert problems[:2] == [
        "facts: the change of facts publishes otherwise than the bind of its stored choices under "
        "the new facts",
        "facts: dropped [], the values not of their type under the new facts: ['n']",
    ]


def test_a_facts_steps_report_missing_a_rule_or_a_standing_choice_is_caught() -> None:
    structure = Structure(BOUNDED.index)
    result = change(BOUNDED, FALLS)
    assert isinstance(result, Accepted)
    silent = replace(result, broken=())
    assert check_facts(structure, BOUNDED, FALLS, silent).problems == [
        "facts report: names [], the origins of the pairs new against the receiver: "
        "[('k', 'constraint')]"
    ]
    alone = replace(result, broken=(replace(K_OVER, standing=()),))
    assert check_facts(structure, BOUNDED, FALLS, alone).problems == [
        "facts report: clearing c clears k, named standing []"
    ]


def test_a_batch_is_held_to_its_facts_then_its_choices() -> None:
    # ``x = 2`` beside the facts adds no pair against them: accepted, ``k`` reported; the
    # choices are a change of choices from the facts' design space (its assignment checked).
    structure = Structure(BOUNDED.index)
    edit = Change(facts={"f": 2}, commit={"x": 2})
    result = change(BOUNDED, edit)
    assert isinstance(result, Accepted) and result.broken == (K_OVER,)
    checked = check_facts(structure, BOUNDED, edit, result)
    assert checked.problems == []
    assert (checked.runs["facts batch"], checked.runs["assignment"]) == (1, 1)
    refused = Rejected((K_OVER,), None)
    assert check_facts(structure, BOUNDED, edit, refused).problems == [
        "facts batch: the batch and its facts then its choices answer otherwise"
    ]
    moved = replace(result, configuration=Configuration(bounded(), {"f": 2}, {"c": 3}))
    assert check_facts(structure, BOUNDED, edit, moved).problems[0] == (
        "facts batch: the batch publishes or drops otherwise than its facts then its choices"
    )


def test_a_design_space_new_facts_left_reading_refused_binds_and_reports_the_rule() -> None:
    # After ``f = 2`` the walk holds ``k`` false through ``c``: the bind of its stored set
    # keeps every value, drops none and reports ``k``, which invariant 1 allows once a facts
    # step broke it, and names otherwise.
    result = change(BOUNDED, FALLS)
    assert isinstance(result, Accepted)
    walked = result.configuration
    bound = reference.bind(walked.space, walked.facts, walked.stored)
    assert (bound.configuration.stored, bound.dropped, bound.broken) == ({"c": 3}, (), (K_OVER,))
    assert bind_problems(walked, bound, broken={"k"}) == ([], frozenset({"k"}))
    assert bind_problems(walked, bound)[0] == [
        "1 bind: reports [('k', 'constraint')]; what a walk publishes binds again with nothing "
        "broken (FN3), but for rules a clear revealed ([])"
    ]
    assert bind_problems(walked, bound, broken={"other"})[0][0].endswith(
        "but for rules a clear revealed ([]) or new facts broke (['other'])"
    )
