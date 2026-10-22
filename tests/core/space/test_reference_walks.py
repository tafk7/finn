# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded random walks: the reference model's four invariants at every step, and the engine
compared with it (``core.space.reference_harness.walks``).

Each seed draws a small description and a walk of changes over it. At every step the
model's invariants hold strictly (history-independence and the bind publishing it with
nothing broken, additions moving answers only up the information order, no accepted change
adding a (false piece, origin) pair, every refusal naming exactly its origins and their
standing choices), and the engine's own checks hold (its walked configuration against one
built from nothing). Each seed walks twice (DESIGN § 6): the strict run draws neither of the
rulings' causes the engine has not reached (no view read in, every fact given, no unset),
and each disagreement fails it; the full run draws both, attributes each disagreement by its
cause to T4b or T4c (or T4a, whose shape a walk is given by hand) and counts it under that
pending milestone, and fails on an unattributed one, or on one whose milestone is taken as
removed. Every invariant's checker is also shown
catching a deliberately wrong answer on a small description.

The gate runs a fixed set of seeds; ``python -m core.space.reference_harness walks`` (from
``tests``) runs more and prints what they met.
"""

from __future__ import annotations

import functools
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
from core.space.reference_harness import Case, walks
from core.space.reference_harness.compare import agrees, answer_form, change_form, possible_form
from core.space.reference_harness.engine import Engine, Point, Spelling, build
from core.space.reference_harness.forms import (
    AnswerForm,
    DefinitionForm,
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
    model_forms,
    order_problems,
    pending,
    questions,
    refused_below,
    table,
    walk,
)

SEEDS = range(24)
STEPS = 30


@functools.cache
def gate_walk(seed: int, strict: bool) -> walks.Tally:
    """The gate's walk of ``seed``, run once per session (never under a monkeypatch)."""
    return walk(seed, STEPS, strict=strict)


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "full"])
@pytest.mark.parametrize("seed", SEEDS)
def test_a_seeded_walk_keeps_the_four_invariants_and_the_engines_own_checks(
    seed: int, strict: bool
) -> None:
    tally = gate_walk(seed, strict)
    assert tally.failure is None, tally.failure
    if strict:
        assert not tally.by_milestone and not tally.model_only
    # each step is a change of choices (its assignment checked) or a facts step; a batch's
    # choices are checked as a change of choices too
    checks = tally.checks
    assert tally.steps == STEPS
    assert checks["assignment"] + checks["facts"] - checks["facts batch"] == STEPS
    # every configuration the model reaches: the start and each accepted step
    assert checks["1 history"] == checks["1 bind"] == tally.accepted + 1
    # the engine compared at each, and its bind of each one's own choices against the model's,
    # from nothing stored and from the walked design space
    assert tally.engine_checks["cold"] == tally.engine_checks["bind"] == tally.compared
    assert tally.engine_checks["bind held"] == tally.compared
    assert tally.engine_errors == 0
    # today's engine, every permanent check run (the cold build through today's ``make``)
    assert tally.implementation == "today"
    assert tally.engine_checks["cold change"] == tally.steps - tally.model_only


def test_the_full_run_meets_both_causes_and_counts_each_under_its_milestone() -> None:
    tallies = [gate_walk(seed, False) for seed in SEEDS]
    counted = sum((tally.by_milestone for tally in tallies), start=type(tallies[0].checks)())
    assert counted["T4b"] and counted["T4c"], counted
    assert set(counted) <= {"T4b", "T4c"}


def test_the_gates_walks_exercise_every_invariant() -> None:
    tallies = [gate_walk(seed, False) for seed in range(6)]
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
    first, second = walk(22, 12), walk(22, 12)
    assert (first.steps, first.accepted, first.checks, first.by_milestone) == (
        second.steps,
        second.accepted,
        second.checks,
        second.by_milestone,
    )


def test_descriptions_hold_two_to_six_choices_and_skip_views_read_in() -> None:
    rng = random.Random(7)
    skipped = 0
    left_out = 0
    for _ in range(40):
        generated, engine = generate(rng)
        skipped += generated.skipped["view-read-in"]
        assert len(generated.read_in) <= generated.skipped["view-read-in"]
        assert 2 <= sum(info.cases is None for info in engine.index.choices.values()) <= 6
        assert generated.text.startswith("facts ")
        assert set(engine.index.facts) - set(generated.facts) <= engine.index.optional
        left_out += len(set(engine.index.facts) - set(generated.facts))
    # a rule reading a view is drawn now and then (SZ100 (3)), checked against the engine's
    # definition and skipped; an optional fact is left out now and then (SZ100 (5))
    assert skipped and left_out
    for _ in range(40):
        generated, engine = generate(rng, strict=True)
        assert not generated.skipped["view-read-in"] and not generated.read_in
        assert set(generated.facts) == set(engine.index.facts)


# -- the engine comparison: attributed by cause in the full run, failing in the strict one -------
#
# Today's engine differs from the model where the rulings reach (T4b, T4c). Elsewhere it
# agrees, so a disagreement with no cause is made here: the engine answers every choice's
# state as absent (the model's start holds an open choice).


@pytest.fixture
def engine_disagrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine answers every choice's state as absent: consistently, so its own checks
    (walked against built from nothing) still hold."""
    asked = walks.engine_forms

    def absent(point: Point, questions: Iterable[Question]) -> dict[Question, Form]:
        forms = asked(point, questions)
        return {q: StateForm("absent") if q[0] == "state" else form for q, form in forms.items()}

    monkeypatch.setattr(walks, "engine_forms", absent)


def first_line(tally: walks.Tally) -> str:
    assert tally.failed == "disagreement" and tally.failure is not None
    assert "description:" in tally.failure and "| engine" in tally.failure
    return tally.failure.splitlines()[0]


@pytest.mark.usefixtures("engine_disagrees")
def test_a_disagreement_with_no_cause_fails_the_full_run_and_any_fails_the_strict_run() -> None:
    # seed 1 declares no optional fact: nothing reaches one left out
    assert first_line(walk(1, 6)) == "seed 1, step 0: the engine disagrees (state; no cause)"
    strict = walk(1, 6, strict=True)
    assert first_line(strict) == "seed 1, step 0: the engine disagrees (state; strict run)"


def test_a_view_read_in_is_counted_under_t4b_until_it_is_taken_as_removed() -> None:
    # seed 0 draws a rule reading a view before its description: the model refuses it as
    # view-read-in, the engine builds it
    tally = walk(0, 3)
    assert tally.failure is None, tally.failure
    assert (tally.by_milestone["T4b"], tally.by_question["definition"]) == (1, 1)
    assert first_line(walk(0, 3, without=("T4b",))) == (
        "seed 0, step 0: the engine disagrees (definition; T4b not pending)"
    )


def test_an_engine_refusing_a_view_read_in_as_the_model_does_is_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # once T4b's engine answers view-read-in, seed 0's drawn description agrees: nothing counted
    def refusing(*_: object, **__: object) -> DefinitionForm:
        return DefinitionForm("view-read-in")

    monkeypatch.setattr(walks, "definition", refusing)
    tally = walk(0, 3)
    assert tally.failure is None, tally.failure
    assert (tally.by_milestone["T4b"], tally.by_question["definition"]) == (0, 0)


def test_a_question_reaching_an_optional_fact_left_out_is_counted_under_t4c() -> None:
    # seed 7 leaves its optional f1 out: the engine reads it unresolved, the model None, so
    # f1's own answer (its subject the fact itself) and a rule reading it disagree, each
    # counted under T4c
    tally = walk(7, 3)
    assert tally.failure is None, tally.failure
    assert tally.by_milestone == {"T4c": 2}
    assert tally.by_question == {"answer": 1, "rule": 1}
    removed = walk(7, 3, without=("T4c",))
    assert first_line(removed) == "seed 7, step 0: the engine disagrees (answer; T4c not pending)"
    assert removed.failure is not None and "walk: the answer of f1" in removed.failure


# T4a (reading 8): no drawn description holds a guard below a refused guard (nothing is drawn
# below the guarded scope), so a walk is given one: a chosen selector under a refused when,
# written as the selector's own guard (the matrix's witness's shape). Today's engine counts a
# piece for each case's guard beside the selector's.

HEATED = Space(
    ("gate",),
    Scope(
        "",
        derived=(
            Derived("heated", ("gate",), lambda r: Refuse("off") if r["gate"] == 1 else True),
        ),
        children=(
            Scope(
                "plant",
                when=When(("heated",), lambda r: bool(r["heated"])),
                selectors=(
                    Selector(
                        "heating",
                        (
                            Scope("boiler", choices=(Choice("stages", listed(1, 2, 3)),)),
                            Scope("heat_pump", choices=(Choice("cop", listed(2, 3, 4)),)),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


@pytest.fixture
def guard_below_a_refused_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every walk draws ``HEATED`` at gate 1, its selector carrying ``plant``'s guard."""
    spelling = Spelling(types={"heated": bool}, guarded_selectors=frozenset({"plant"}))

    def drawn(*_: object, **__: object) -> tuple[walks.Generated, Engine]:
        return walks.Generated(HEATED, "HEATED", {"gate": 1}), build(HEATED, spelling)

    monkeypatch.setattr(walks, "generate", drawn)


@pytest.mark.usefixtures("guard_below_a_refused_guard")
def test_a_guard_refused_below_a_refused_guard_is_counted_under_t4a() -> None:
    # by the model's presence states, the cases are T4a's scopes while plant's guard refuses
    configuration = Configuration(HEATED, {"gate": 1})
    found = refused_below(configuration.index, configuration.solve())
    assert found == {"plant.heating.boiler", "plant.heating.heat_pump"}
    assert not refused_below(configuration.index, Configuration(HEATED, {"gate": 2}).solve())
    structure = Structure(configuration.index)
    assert structure.reaches_scope({"plant.heating.boiler.stages"}, found)
    assert not structure.reaches_scope({"heated"}, found)
    # validity, and possible values and changes carrying its findings, differ: each counted
    tally = walk(0, 20)
    assert tally.failure is None, tally.failure
    assert set(tally.by_milestone) == {"T4a"} and tally.by_question["validity"]
    assert first_line(walk(0, 20, without=("T4a",))) == (
        "seed 0, step 0: the engine disagrees (validity; T4a not pending)"
    )
    assert first_line(walk(0, 20, strict=True)) == (
        "seed 0, step 0: the engine disagrees (validity; strict run)"
    )


@pytest.mark.usefixtures("guard_below_a_refused_guard")
def test_the_cause_of_a_question_reaching_a_guard_below_a_refused_guard_is_t4a() -> None:
    # The walk's own attribution, asked directly: at gate 1 validity (no subject) and an
    # answer under ``plant`` reach the cases' guards, refused below plant's; ``heated``,
    # outside ``plant``, reaches none. At gate 2 the guard holds: nothing has a cause.
    walked = walks._Walk(0, 0, False, pending())
    refused, holding = Configuration(HEATED, {"gate": 1}), Configuration(HEATED, {"gate": 2})
    assert walked.cause(None, refused) == "T4a"
    assert walked.cause("plant.heating.boiler.stages", refused) == "T4a"
    assert walked.cause("heated", refused) is None
    for subject in (None, "plant.heating.boiler.stages", "heated"):
        assert walked.cause(subject, holding) is None


@pytest.mark.usefixtures("guard_below_a_refused_guard")
def test_switching_a_selector_under_a_refused_when_is_attributed_t4a() -> None:
    # A case stored under the refused guard: switching to the other case adds no pair in the
    # model (one guard piece, plant's, before and after). Today's engine counts each case's
    # guard apart, so the switch adds one: it refuses the change, and lists the case as
    # impossible. Both questions reach the whole description, attributed T4a, as the walk
    # asks them (a change against its predecessor and the published facts).
    spelling = Spelling(types={"heated": bool}, guarded_selectors=frozenset({"plant"}))
    pair = Case("switch", HEATED, spelling).start({"gate": 1}, {"plant.heating": "boiler"})
    switch = Change(commit={"plant.heating": "heat_pump"})
    expected = change_form(pair.model, switch, change(pair.model, switch))
    found, _ = pair.engine.change(switch)
    assert expected.accepted and not found.accepted
    viable = possible_form(pair.solution(), "plant.heating")
    assert not agrees(viable, pair.engine.possible("plant.heating"))
    walked = walks._Walk(0, 0, False, pending())
    published = Configuration(HEATED, {"gate": 1})
    assert walked.cause(None, pair.model, published) == "T4a"  # the change
    assert walked.cause(None, pair.model) == "T4a"  # the possible values


# A refused guard below an unknown one (reading 8's second half): ``s`` open, its case ``a``
# holding ``g``, present when ``d`` is at least 1, and ``d`` refusing an odd ``f``.

IN_AN_OPEN_CASE = Space(
    ("f",),
    Scope(
        "",
        derived=(Derived("d", ("f",), lambda r: Refuse("d-odd") if r["f"] % 2 else r["f"]),),
        selectors=(
            Selector(
                "s",
                (
                    Scope(
                        "a",
                        choices=(Choice("p", listed(1, 2)),),
                        children=(
                            Scope(
                                "g",
                                when=When(("d",), lambda r: r["d"] >= 1),
                                choices=(Choice("q", listed(1, 2)),),
                            ),
                        ),
                    ),
                    Scope("b", choices=(Choice("r", listed(1, 2)),)),
                ),
            ),
        ),
    ),
)


def test_the_cause_of_a_question_reaching_a_refused_guard_under_an_open_selector_is_t4a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def drawn(*_: object, **__: object) -> tuple[walks.Generated, Engine]:
        generated = walks.Generated(IN_AN_OPEN_CASE, "IN_AN_OPEN_CASE", {"f": 1})
        return generated, build(IN_AN_OPEN_CASE)

    monkeypatch.setattr(walks, "generate", drawn)
    walked = walks._Walk(0, 0, False, pending())
    open_ = Configuration(IN_AN_OPEN_CASE, {"f": 1})
    assert refused_below(open_.index, open_.solve()) == {"s.a.g"}
    # ``s`` open: validity, and a subject under ``g``, reach ``g``'s guard
    assert walked.cause(None, open_) == "T4a"
    assert walked.cause("s.a.g.q", open_) == "T4a"
    # a subject outside ``a`` reaches none
    for subject in ("s.b.r", "d"):
        assert walked.cause(subject, open_) is None
    # ``s`` set to ``b``: ``g`` is absent, nothing has a cause
    other = Configuration(IN_AN_OPEN_CASE, {"f": 1}, {"s": "b"})
    assert not refused_below(other.index, other.solve())
    assert walked.cause(None, other) is None


# Reading 26, invariant 2's one excused move: under ``plant``'s refused guard, setting the
# selector finds the other case absent (absent dominates refused in presence's AND), so the
# value of its choice moves from refused to inapplicable. An answer refused on its own never
# does while its scope stays present.


def test_an_answer_refused_through_its_scopes_presence_may_become_inapplicable() -> None:
    before = Configuration(HEATED, {"gate": 1}).solve()
    after = Configuration(HEATED, {"gate": 1}, {"plant.heating": "boiler"}).solve()
    asked = questions(before.index)
    old, new = model_forms(before, asked), model_forms(after, asked)
    cop = ("answer", "plant.heating.heat_pump.cop")
    assert (old[cop], new[cop]) == (
        AnswerForm("refused", codes=old[cop].codes),  # type: ignore[union-attr]
        AnswerForm("inapplicable"),
    )
    out = walks.refused_then_absent(before, after)
    assert out == {"plant.heating.heat_pump.cop"}
    committed = {"plant.heating": "boiler"}
    assert order_problems(old, new, committed) == {cop: "refused -> inapplicable"}
    assert order_problems(old, new, committed, out) == {}


def test_an_answer_refused_on_its_own_moving_to_inapplicable_is_still_named() -> None:
    # the same move, of a node whose scope is not out by reading 26: named
    old = {("answer", "d"): AnswerForm("refused", codes=frozenset({("d-odd", "d")}))}
    new = {("answer", "d"): AnswerForm("inapplicable")}
    assert order_problems(old, new, {}, frozenset({"other"})) == {
        ("answer", "d"): "refused -> inapplicable"
    }
    # and only refused -> inapplicable is excused for a node that is: refused -> a value is not
    valued = {("answer", "d"): AnswerForm("value", 2)}
    assert order_problems(old, valued, {}, frozenset({"d"})) == {
        ("answer", "d"): "refused -> value"
    }


def test_each_facts_answer_is_asked_at_each_configuration() -> None:
    # A fact's answer is always known (SZ100 (5)): the model answers its value, or None for
    # an optional one left out (seed 7's f1), and the engine is asked it through ``query``
    generated, engine = generate(random.Random(7))
    index = generated.description.validate()
    assert index.optional and set(index.optional) - set(generated.facts)
    asked = questions(index)
    assert {("answer", fact) for fact in index.facts} <= set(asked)
    configuration = Configuration(generated.description, generated.facts)
    solution, point = configuration.solve(), engine.open(dict(generated.facts))
    for fact in sorted(index.facts):
        expected = generated.facts.get(fact)
        assert answer_form(solution, fact) == AnswerForm("value", expected)
        found = point.answer(fact)
        if fact in generated.facts:
            assert found == AnswerForm("value", expected)
        else:  # left out: today's engine reads it unresolved (T4c's row)
            assert found.kind == "unresolved", found


def test_with_no_milestone_pending_every_disagreement_fails_its_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(walks, "ROWS", ())
    assert pending() == ()
    assert first_line(walk(7, 3)) == (
        "seed 7, step 0: the engine disagrees (answer; T4c not pending)"
    )
    with pytest.raises(ValueError, match="no such milestone"):
        pending(("lane 10",))


def test_the_summary_names_the_runs_the_milestones_and_the_checks() -> None:
    tallies = [walk(7, 3), walk(7, 3, strict=True)]
    text = table(tallies)
    checks = ", ".join(
        f"{name} {count}"
        for name, count in sorted((tallies[0].checks + tallies[1].checks).items())
        if count
    )
    assert (
        f"invariant checks: {checks}; binds reading refused, reporting rules new facts broke "
        "(§ 16.24) 0" in text
    )
    assert "walks 2 (full 1, strict 1)" in text
    assert "implementation: today" in text.splitlines()
    assert (
        "engine disagreements (full: attributed by cause, counted while pending (T4a, T4b, "
        "T4c), else failing; strict: every one failing): by milestone T4a 0, T4b 0, T4c 2; by "
        "question answer 1, rule 1"
    ) in text
    assert text.splitlines()[2].startswith("| 7 | full | 3 |")
    assert text.splitlines()[3].startswith("| 7 | strict | 3 |")
    removed = ("T4a", "T4b", "T4c")
    assert "every one failing (no milestone with rows pending)" in table(tallies, removed)


def test_a_broken_standing_list_in_the_model_fails_a_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    # The walk end to end: a model naming one standing choice too few fails at the step. The
    # engine, unbroken, would disagree with it sooner (its possible values name the standing
    # choices too), so the engine's answers are taken as agreeing: the invariant is what fails.
    standing: Callable[..., tuple[str, ...]] = reference.model._standing
    monkeypatch.setattr(reference.model, "_standing", lambda *args: standing(*args)[1:])
    monkeypatch.setattr(walks, "agrees", lambda *args: True)
    tally = walk(30, 4)
    assert tally.failed == "4 standing" and tally.failure is not None
    assert tally.failure.startswith("seed 30, step 4: invariant 4 standing: clearing")


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
            "with nothing broken (FN3), but for rules new facts broke ([])"
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
        "broken (FN3), but for rules new facts broke ([])"
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


def a_view_of_a_space_refused_by_an_addition() -> Space:
    """The probe of § 16.14: ``g``, in case ``a`` of the open ``s``, is guarded by ``d``,
    which refuses the fact ``f`` when odd; the root's view ``v`` reads ``f``. (Its second
    shape, a root rule reading a child's view, went with views read out: SZ100 (3).)"""
    g = Scope("g", when=When(("d",), lambda r: r["d"] > 0), choices=(Choice("n", listed(1, 2)),))
    return Space(
        ("f",),
        Scope(
            "",
            selectors=(Selector("s", (Scope("a", children=(g,)), Scope("b"))),),
            derived=(Derived("d", ("f",), lambda r: Refuse("odd") if r["f"] % 2 else r["f"]),),
            views=(View("v", ("f",), lambda r: r["f"]),),
        ),
    )


def test_an_addition_that_would_leave_a_space_refused_is_refused() -> None:
    # § 16.15 removes § 16.14's exception: selecting ``a`` makes the guard's piece false
    # through ``d``, over the facts alone: a new pair, refused with nothing standing. Were it
    # accepted, the root's view ``v`` would move down the information order, and invariants 2
    # and 3 name it.
    start = Configuration(a_view_of_a_space_refused_by_an_addition(), {"f": 3})
    structure = Structure(start.index)
    asked = questions(start.index)
    before = model_forms(start.solve(), asked)
    edit = Change(commit={"s": "a"})
    result = change(start, edit)
    assert isinstance(result, Rejected)
    assert result.refusals == (RuleRefusal("d", "derived", ODD, ()),)
    assert check_change(structure, start, edit, result).problems == []
    published = Configuration(start.space, {"f": 3}, {"s": "a"})
    fake = Accepted(published, (), (), (), published.solve().validity())
    problems = check_change(structure, start, edit, fake).problems
    assert len(problems) == 1 and problems[0].startswith("3 accepted: ")
    assert ("answer", "v") in order_problems(
        before, model_forms(published.solve(), asked), edit.commit
    )


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


# -- clears (§ 16.16): never refused, and adding no pair (SZ100 (3)) -------------------------------


def test_a_clear_only_step_adding_a_pair_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # No clear adds a pair under the model's rules: masking, the one shape that could, went
    # with views read out. The checker is shown one by hand, and the walk fails on the
    # problem (as on every problem check_change returns); a clear-only step refused fails too.
    before = Configuration(summed(), {}, {"b": 2, "c": 2})
    edit = Change(clear=("c",))
    result = change(before, edit)
    assert isinstance(result, Accepted)
    checked = check_change(Structure(before.index), before, edit, result)
    assert (checked.problems, checked.runs["3 clears"]) == ([], 1)
    fake = Rejected((RuleRefusal("sum", "constraint", (), ()),), None)
    assert check_change(Structure(before.index), before, edit, fake).problems == [
        "3 clears: a change made only of clears and unsets is refused (§ 16.16)"
    ]
    pair: Pair = (("constraint", "sum"), "sum")
    monkeypatch.setattr(walks, "added_pairs", lambda *args: {pair: "constraint"})
    assert check_change(Structure(before.index), before, edit, result).problems == [
        "3 clears: the clear-only change adds (false piece, origin) pairs: "
        "[(('constraint', 'sum'), 'sum')] (no masking: SZ100 (3))"
    ]


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
        "broken (FN3), but for rules new facts broke ([])"
    ]
    assert bind_problems(walked, bound, broken={"other"})[0][0].endswith(
        "but for rules new facts broke (['other'])"
    )
