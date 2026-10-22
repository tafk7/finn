# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness comparing the engine with its reference model, held to its own rules: the
model imports the standard library only, every row of the acceptance matrix has exactly
one witness and every expected failure a row, and the comparison reads blockers as a
subset and everything else exactly; and the comparison is live: an implementation with a
planted fault (``support.space_oracle.today.Planted``, DESIGN § 7) is reported by
question, in the hand cases and in walks 0-3. The walks' own checks are
``test_reference_walks``'s."""

from __future__ import annotations

import ast
import functools
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from support.space_oracle import Answers
from support.space_oracle.today import FAULTS, TODAY, Fault, Historied, Planted, PlantedAnswers

from core.space import reference
from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Members,
    Refuse,
    Scope,
    Selector,
    Space,
    When,
    computed,
    listed,
)
from core.space.reference_harness import (
    MILESTONES,
    ROWS,
    Case,
    Disagreement,
    DroppedForm,
    NotTranslatable,
    OutcomeForm,
    Pair,
    PossibleForm,
    RefusalForm,
    Row,
    RuleForm,
    Spelling,
    StateForm,
    ValidityForm,
    markdown,
    matrix,
)
from core.space.reference_harness.compare import (
    agrees,
    change_form,
    rule_form,
    validity_form,
)
from core.space.reference_harness.matrix import problems
from core.space.reference_harness.walks import Tally, walk
from finn.core.space import Available
from finn.core.space import Space as EngineSpace


def imported(path: Path) -> list[tuple[int, str, int]]:
    """Every import of a module's source: its line, the module and its relative level."""
    found: list[tuple[int, str, int]] = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.lineno, node.module or "", node.level))
    return found


def test_the_model_imports_the_standard_library_only() -> None:
    package = Path(reference.__file__).parent
    modules = sorted(package.glob("*.py"))
    assert len(modules) >= 4
    outside = [
        f"{path.name}:{line}: {name}"
        for path in modules
        for line, name, level in imported(path)
        if level == 0 and name.split(".")[0] not in sys.stdlib_module_names
    ]
    assert not outside, outside
    # A relative import stays inside the package: one level up would leave it.
    assert all(level <= 1 for path in modules for _, _, level in imported(path))


CASE_MODULES = tuple(
    Path(__file__).with_name(f"test_reference_{name}.py") for name in ("cases", "rules", "scopes")
)


def test_every_row_has_exactly_one_witness_and_every_expected_failure_a_row() -> None:
    assert all(path.exists() for path in CASE_MODULES)
    assert not problems(CASE_MODULES)


def test_the_pending_milestones_are_the_rulings_the_engine_has_not_reached() -> None:
    # The engine core's, the facts-instance and T4a's milestones are accepted, their rows
    # removed; the model holds SZ100's rulings before the engine (T4b: views read out; T4c:
    # an optional fact left out is None).
    assert MILESTONES == ("base", "T4b", "T4c")
    assert [(row.id, row.milestone) for row in ROWS] == [
        ("view-read-in", "T4b"),
        ("optional-fact-none", "T4c"),
    ]


# A later milestone with rows, put in the matrix where a test needs one.
LATER = ("lane 9", "lane 10")
WITNESS = "test_reference_cases::test_it"
LATER_ROWS = (
    Row("b-row", "lane 10", "the second change", "test_reference_cases::test_other"),
    Row("a-row", "lane 9", "the change", WITNESS, "today refused"),
)


@pytest.fixture
def later(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(matrix, "MILESTONES", ("base", *LATER))
    monkeypatch.setattr(matrix, "ROWS", LATER_ROWS)
    monkeypatch.setattr(matrix, "BY_ID", {row.id: row for row in LATER_ROWS})


@pytest.mark.usefixtures("later")
def test_the_matrix_check_finds_a_row_without_its_witness(tmp_path: Path) -> None:
    """A check that has never rejected anything is a check nobody has tested."""
    module = tmp_path / "test_reference_cases.py"
    module.write_text(
        '@expected("a-row")\ndef test_it() -> None: ...\n\n'
        "@pytest.mark.xfail(strict=True)\ndef test_other() -> None: ...\n\n"
        '@expected("c-row")\ndef test_third() -> None: ...\n'
    )
    assert problems([module]) == [
        "test_reference_cases::test_other: an expected failure with no row",
        "c-row: witnessed by ['test_reference_cases::test_third'], but no such row",
        "b-row: witnessed by [], names test_reference_cases::test_other",
    ]


@pytest.mark.usefixtures("later")
def test_a_witness_is_a_strict_expected_failure_raising_a_disagreement() -> None:
    @matrix.expected("a-row")
    def test_it() -> None: ...

    (mark,) = test_it.pytestmark  # type: ignore[attr-defined]
    assert mark.name == "xfail" and mark.kwargs == {
        "strict": True,
        "raises": Disagreement,
        "reason": "lane 9: the change (today refused)",
    }


@pytest.mark.usefixtures("later")
def test_the_matrix_prints_as_a_markdown_table_in_milestone_order() -> None:
    table = markdown().splitlines()
    assert table[0] == "| Row | Milestone | The change that removes it | Witness |"
    assert table[2:] == [
        f"| a-row | lane 9 | the change (today refused) | `{WITNESS}` |",
        "| b-row | lane 10 | the second change | `test_reference_cases::test_other` |",
    ]


# -- the comparison --------------------------------------------------------------------------------


def test_the_engine_may_name_fewer_blockers_than_the_model_never_others() -> None:
    model = ValidityForm("open", blockers=frozenset({"x", "y"}))
    assert agrees(model, ValidityForm("open", blockers=frozenset({"x"})))
    assert not agrees(model, ValidityForm("open", blockers=frozenset({"z"})))
    assert not agrees(model, ValidityForm("valid"))


def test_possible_values_compare_exactly_in_their_lists_order() -> None:
    one, two = (1, OutcomeForm("possible")), (2, OutcomeForm("possible"))
    assert agrees(PossibleForm((one, two)), PossibleForm((one, two)))
    assert not agrees(PossibleForm((one, two)), PossibleForm((two, one)))
    assert not agrees(PossibleForm((one,)), PossibleForm(None))
    moved = (1, OutcomeForm("possible", dormant=frozenset({"p"})))
    assert not agrees(PossibleForm((moved,)), PossibleForm((one,)))


def test_values_compare_by_type_as_well() -> None:
    assert not agrees(StateForm("set", True), StateForm("set", 1))
    assert not agrees(
        PossibleForm(((True, OutcomeForm("possible")),)),
        PossibleForm(((1, OutcomeForm("possible")),)),
    )


def gated() -> Space:
    """``g`` is present while the derived ``gate`` exceeds 2; ``gate`` refuses an odd ``w``.
    ``lanes``'s domain reads ``half``, which refuses an odd ``v``; ``z`` lists nothing."""
    return Space(
        ("w", "v"),
        Scope(
            "",
            choices=(
                Choice(
                    "lanes",
                    computed(lambda r: tuple(d for d in (1, 2) if r["half"] % d == 0), ("half",)),
                ),
                Choice("z", listed()),
            ),
            derived=(
                Derived("gate", ("w",), lambda r: Refuse("odd") if r["w"] % 2 else r["w"]),
                Derived("half", ("v",), lambda r: Refuse("odd-v") if r["v"] % 2 else r["v"]),
            ),
            children=(
                Scope(
                    "g",
                    when=When(("gate",), lambda r: r["gate"] > 2),
                    choices=(Choice("n", listed(1, 2)),),
                ),
            ),
            constraints=(Constraint("on-n", ("g.n",), lambda r: r["g.n"] > 0, "n"),),
        ),
    )


def test_a_refused_input_carries_the_codes_of_its_origin() -> None:
    # Whatever refuses what a node reads (a guard over a refusing input, a membership input
    # refusing), the node carries the codes of the origin, never none.
    configuration = reference.Configuration(gated(), {"w": 3, "v": 3}, {"g.n": 2})
    solution = configuration.solve()
    assert isinstance(solution.state("g.n"), reference.Dormant)
    assert rule_form(solution, "on-n") == RuleForm("false", frozenset({("odd", "gate")}))
    edit = Change(commit={"lanes": 1})
    result = reference.change(configuration, edit)
    assert change_form(configuration, edit, result).refusals == {
        RefusalForm("assignment", "lanes", frozenset({("odd-v", "half")}))
    }


def test_an_accepted_change_of_facts_carries_its_dropped_values_and_new_rules_as_sets() -> None:
    # ``v`` turns odd: ``lanes = 2`` is dropped, its type check through ``half``; the empty
    # list's piece is now false through ``half``, reported at its origin; ``w`` turns odd:
    # ``gate`` refuses, reported too. Both lists are sets: no side's order is compared.
    configuration = reference.Configuration(gated(), {"w": 4, "v": 4}, {"lanes": 2})
    edit = Change(facts={"w": 3, "v": 3})
    form = change_form(configuration, edit, reference.change(configuration, edit))
    assert form.accepted and form.refusals == frozenset()
    assert form.dropped == {DroppedForm("lanes", 2, frozenset({("odd-v", "half")}))}
    assert form.broken == {
        RefusalForm("rule", "gate", frozenset({("odd", "gate")})),
        RefusalForm("rule", "half", frozenset({("odd-v", "half")})),
    }
    assert not agrees(form, replace(form, dropped=frozenset()))


def ranged() -> Space:
    """``x`` below the fact ``n``; ``k`` asks ``y`` to be at most ``n``."""
    return Space(
        ("n",),
        Scope(
            "",
            choices=(
                Choice("x", computed(lambda r: tuple(range(r["n"])), ("n",))),
                Choice("y", listed(1, 2)),
            ),
            constraints=(Constraint("k", ("y", "n"), lambda r: r["y"] <= r["n"], "y-over"),),
        ),
    )


RANGED = Case("ranged", ranged())
K_OVER = RefusalForm("rule", "k", frozenset({("y-over", "k")}), frozenset({"y"}))


def test_the_engines_accepted_change_of_facts_carries_the_old_value_of_what_it_drops() -> None:
    # ``n = 1``: ``x = 2`` is dropped (its old value the receiver's), ``k`` turns false with
    # ``y`` standing; the engine's report in the model's form.
    edit = Change(facts={"n": 1})
    pair = RANGED.start({"n": 3}, {"x": 2, "y": 2})
    engine, _ = pair.engine.change(edit)
    expected = change_form(pair.model, edit, reference.change(pair.model, edit))
    assert engine == expected
    assert engine.dropped == {DroppedForm("x", 2, frozenset({("domain-membership", "x")}))}
    assert engine.broken == {K_OVER}


def test_a_bind_compares_from_any_design_space_with_its_facts() -> None:
    # Both sides read a bind against its facts with nothing stored, whatever the receiver
    # stores, so the harness asks it of any design space. The bind of a value outside its
    # domain drops it, its old value the saved one.
    bound = RANGED.start({"n": 1}).check_bind({"x": 2, "y": 2})
    assert bound.model.stored == {"y": 2} and bound.engine.stored() == {"y": 2}
    form, _ = RANGED.start({"n": 1}).engine.bind({"x": 2, "y": 2})
    assert form.dropped == {DroppedForm("x", 2, frozenset({("domain-membership", "x")}))}
    assert form.broken == {K_OVER}
    # From a design space storing choices of its own, the same report.
    held, _ = RANGED.start({"n": 1}, {"x": 0, "y": 1}).engine.bind({"x": 2, "y": 2})
    assert held == form
    RANGED.start({"n": 1}, {"x": 0, "y": 1}).check_bind({"x": 2, "y": 2})


def test_a_binds_outcomes_read_from_nothing_stored_whatever_the_receiver_holds() -> None:
    # The harness compares no outcome status (the model's bind has none), so the engine's
    # are read here against what the model's bind keeps and drops: each saved choice kept
    # is ``"changed"`` and each dropped ``"dropped"`` (requested), from a design space
    # storing nothing, the kept choices themselves, or other values.
    saved = {"x": 2, "y": 2}
    result = reference.bind(ranged(), {"n": 1}, saved)
    dropped = {item.choice for item in result.dropped}
    expected = tuple(
        (path, "changed" if path in result.configuration.stored else "dropped", True)
        for path in saved
    )
    assert dropped == {"x"} and expected == (("x", "dropped", True), ("y", "changed", True))
    nothing = RANGED.start({"n": 1}).engine
    _, holding = nothing.bind(saved)
    assert holding.stored() == dict(result.configuration.stored)
    other = RANGED.start({"n": 1}, {"x": 0, "y": 1}).engine
    for receiver in (nothing, holding, other):
        assert receiver.bind_outcomes(saved) == expected


def test_the_codes_of_an_empty_list_and_a_refused_guard() -> None:
    # ``domain-empty`` and ``guard-refused``, the engine core's codes too. ``on-n`` is false
    # through the guard: it carries the guard's origin, ``gate``.
    solution = reference.Configuration(gated(), {"w": 3, "v": 4}).solve()
    assert validity_form(solution) == ValidityForm(
        "refused",
        frozenset({("domain-empty", "z"), ("guard-refused", "g"), ("odd", "gate")}),
    )


def test_a_disagreement_names_the_case_the_question_and_both_answers() -> None:
    with pytest.raises(Disagreement) as caught:
        LISTED.start().check_state("x", ruled=StateForm("set", 2))
    assert isinstance(caught.value, AssertionError)
    text = str(caught.value)
    assert text.splitlines()[0] == "listed: the state of x"
    assert "! kind: model 'set' | engine 'open'" in text
    assert "! value: model 2 | engine None" in text
    assert "  blockers: model {} | engine {}" in text


# -- the translator --------------------------------------------------------------------------------

LISTED = Case("listed", Space((), Scope("", choices=(Choice("x", listed(1, 2)),))))

NESTED = Case(
    "nested",
    Space(
        ("width",),
        Scope(
            "",
            children=(
                Scope(
                    "unit",
                    selectors=(
                        Selector(
                            "compute",
                            (
                                Scope("packed", choices=(Choice("pe", listed(1, 2)),)),
                                Scope("stub", choices=(Choice("rows", listed(1, 2)),)),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        optional=("width",),
    ),
)


def test_every_choice_keeps_its_path_as_its_key() -> None:
    answers = NESTED.start({"width": 4}).engine.answers
    keys = {key for key, _ in answers.states()}
    assert keys == {"unit.compute", "unit.compute.packed.pe", "unit.compute.stub.rows"}


def test_a_selector_is_read_as_its_selected_case() -> None:
    NESTED.start({"width": 4}, {"unit.compute": "stub"}).check_answer("unit.compute")


def test_an_optional_fact_cannot_be_left_out_again_in_the_engine() -> None:
    # SZ100 (5): unset, an optional fact is None in the model; the engine cannot unsupply a
    # param it was given.
    with pytest.raises(NotTranslatable):
        NESTED.start({"width": 4}).check_change(Change(unset=("width",)))


def test_a_selector_carries_its_scopes_guard_only_where_the_scope_holds_it_alone() -> None:
    # ``Spelling.guarded_selectors``: a guarded scope holding one selector and nothing else
    # is written as the selector's own guard; any other scope cannot be.
    def described(*choices: Choice) -> Space:
        heating = Selector("heating", (Scope("boiler"), Scope("heat_pump")))
        plant = Scope(
            "plant",
            when=When(("on",), lambda r: bool(r["on"])),
            selectors=(heating,),
            choices=choices,
        )
        return Space(("on",), Scope("", children=(plant, Scope("hall"))))

    spelled = Spelling(guarded_selectors=frozenset({"plant"}))
    assert Case("alone", described(), spelled).engine.guards == {"plant.heating": "plant"}
    with pytest.raises(NotTranslatable, match="one selector alone"):
        Case("beside a choice", described(Choice("x", listed(1, 2))), spelled)
    with pytest.raises(NotTranslatable, match="not a child"):
        Case("unguarded", described(), Spelling(guarded_selectors=frozenset({"hall"})))


def test_a_members_entry_is_exported_by_a_child_its_scope_places() -> None:
    # The engine's Members ranges over the children its Space places, each exporting the
    # key once: an entry two scopes down, or two entries of one child, cannot be written.
    def described(*entries: str) -> Space:
        inner = Scope(
            "inner",
            choices=(Choice("c", listed(1, 2)),),
            derived=(Derived("e", ("unit.inner.c",), lambda r: r["unit.inner.c"]),),
        )
        unit = Scope(
            "unit",
            choices=(Choice("c", listed(1, 2)),),
            derived=(
                Derived("e", ("unit.c",), lambda r: r["unit.c"]),
                Derived("f", ("unit.c",), lambda r: r["unit.c"]),
            ),
            children=(inner,),
        )
        return Space((), Scope("", children=(unit,), members=(Members("m", entries),)))

    Case("one entry", described("unit.e")).start({}, {"unit.c": 2}).check_answer("m")
    for entries in (("unit.inner.e",), ("unit.e", "unit.f")):
        with pytest.raises(NotTranslatable):
            Case("not written", described(*entries))


# -- planted faults (DESIGN § 7) ---------------------------------------------------------------
#
# Today's engine with one answer altered (``today.FAULTS``): one standing choice dropped, one
# possible value refused, the root's validity swapped, or one member's value moved (the first
# member answering a value, by key). A harness comparing nothing would pass each.


def capped() -> Space:
    """``a`` and ``b`` sum to at most the fact ``n`` (``cap``); ``double``, ``n`` doubled, is
    the first member by key answering a value while ``a`` is open."""
    return Space(
        ("n",),
        Scope(
            "",
            choices=(Choice("a", listed(1, 2)), Choice("b", listed(1, 2))),
            derived=(Derived("double", ("n",), lambda r: 2 * r["n"]),),
            constraints=(
                Constraint("cap", ("a", "b", "n"), lambda r: r["a"] + r["b"] <= r["n"], "over"),
            ),
        ),
    )


CAPPED = Case("capped", capped())

#: each fault, the choices stored, and the hand case's question that reports it
PLANTED_QUESTIONS: tuple[tuple[Fault, dict[str, object], Callable[[Pair], object], str], ...] = (
    # ``b = 2`` breaks ``cap`` with ``a`` standing: in the list and in the refused change
    ("standing", {"a": 2}, lambda pair: pair.check_possible("b"), "the possible values of b"),
    (
        "standing",
        {"a": 2},
        lambda pair: pair.check_change(Change(commit={"b": 2})),
        "the change commit {'b': 2}",
    ),
    ("possible", {}, lambda pair: pair.check_possible("a"), "the possible values of a"),
    ("validity", {}, lambda pair: pair.check_validity(), "validity"),
    ("value", {}, lambda pair: pair.check_answer("double"), "the answer of double"),
)


def test_every_planted_fault_has_a_hand_case() -> None:
    assert {fault for fault, *_ in PLANTED_QUESTIONS} == set(FAULTS)


@pytest.mark.parametrize(
    ("fault", "stored", "ask", "question"),
    PLANTED_QUESTIONS,
    ids=[f"{fault}-{question}" for fault, _, _, question in PLANTED_QUESTIONS],
)
def test_a_planted_fault_is_reported_by_its_question_in_a_hand_case(
    fault: Fault, stored: dict[str, object], ask: Callable[[Pair], object], question: str
) -> None:
    ask(CAPPED.start({"n": 3}, stored))  # today's engine agrees
    with pytest.raises(Disagreement) as caught:
        ask(CAPPED.start({"n": 3}, stored, implementation=Planted(TODAY, fault)))
    assert (caught.value.case, caught.value.question) == ("capped", question)


def test_a_planted_value_on_a_stored_choice_is_seen() -> None:
    # A stored choice's answer is read from ``query``, as every other member's is, so a value
    # moved there alone is a disagreement (its state still reads ``stored``, which is right).
    planted = CAPPED.start({"n": 3}, {"a": 2}, implementation=Planted(TODAY, "value"))
    moved = planted.engine.answers.query("a")
    assert isinstance(moved, Available) and moved.value == ("planted-fault", 2)
    with pytest.raises(Disagreement) as caught:
        planted.check_answer("a")
    assert "planted-fault" in str(caught.value)
    planted.check_state("a")


#: walks 0-3 (the gate's 30 steps, strict) with each fault planted: the question each seed's
#: walk reports it by. The strict run fails on every disagreement; in the full run one
#: reaching an optional fact left out is counted under T4c, which could hide a fault. The
#: value fault moves the first member by key answering a value: the fact ``f0`` in every
#: one of these descriptions at the start, and each fact's answer is asked, so each walk
#: reports it there, at step 0 (``VALUE_LINE``).
PLANTED_WALKS: dict[Fault, dict[int, str]] = {
    "standing": {0: "facts change", 1: "facts change", 2: "possible", 3: "possible"},
    "possible": {0: "possible", 1: "possible", 2: "possible", 3: "possible"},
    "validity": {0: "validity", 1: "validity", 2: "validity", 3: "validity"},
    "value": {0: "answer", 1: "answer", 2: "answer", 3: "answer"},
}

FIRST_LINE = re.compile(r"seed (\d+), step \d+: the engine disagrees \((.+); strict run\)")
VALUE_LINE = re.compile(r"seed \d+, step 0: the engine disagrees \(answer; strict run\)")


@functools.cache
def planted_walk(fault: Fault, seed: int) -> Tally:
    return walk(seed, 30, strict=True, implementation=Planted(TODAY, fault))


@pytest.mark.parametrize("fault", FAULTS)
def test_a_planted_fault_is_reported_by_question_in_walks_0_to_3(fault: Fault) -> None:
    reported: dict[int, str] = {}
    for seed in range(4):
        tally = planted_walk(fault, seed)
        assert tally.implementation == f"today, planted {fault}"
        assert tally.engine_errors == 0
        if tally.failure is None:
            continue
        assert tally.failed == "disagreement", tally.failure
        lines = tally.failure.splitlines()
        found = FIRST_LINE.fullmatch(lines[0])
        assert found is not None and int(found[1]) == seed, lines[0]
        assert f"implementation: today, planted {fault}" in lines, tally.failure
        if fault == "value":
            assert VALUE_LINE.fullmatch(lines[0]) and lines[1] == "walk: the answer of f0"
        reported[seed] = found[2]
    assert reported == PLANTED_WALKS[fault]


class Renamed:
    """Today's answers behind a wrapper that alters nothing (``PlantedAnswers`` with no
    fault it knows): not today's implementation, and agreeing everywhere."""

    name = "today, renamed"

    def open(self, configuration: EngineSpace) -> Answers:
        return PlantedAnswers(TODAY.open(configuration), cast("Fault", "none"))

    def make(
        self,
        space_class: type[EngineSpace],
        facts: Mapping[str, object],
        stored: Mapping[str, object],
    ) -> Answers:
        return PlantedAnswers(TODAY.make(space_class, facts, stored), cast("Fault", "none"))


def test_the_check_against_a_cold_build_runs_for_every_implementation() -> None:
    # The cold build is the implementation's own (``Implementation.make``), so the check
    # runs for any implementation (here today's under another name, which agrees
    # everywhere), at every configuration and change, as the bind checks do.
    renamed = walk(0, 30, strict=True, implementation=Renamed())
    assert renamed.failure is None, renamed.failure
    assert renamed.implementation == "today, renamed"
    assert renamed.engine_checks["cold"] == renamed.compared
    assert renamed.engine_checks["cold change"] == renamed.steps - renamed.model_only
    assert renamed.engine_checks["bind"] == renamed.engine_checks["bind held"] == renamed.compared


COLD_LINE = re.compile(
    r"seed (\d+), step \d+: the walked engine configuration answers otherwise than one built "
    r"from nothing \(model column: built from nothing; engine column: walked\)"
)


@pytest.mark.parametrize("seed", range(4))
def test_history_is_reported_by_the_cold_build_in_walks_0_to_3(seed: int) -> None:
    # ``Historied``: what a change reaches answers one member otherwise than the same design
    # space made from nothing. Opened, it agrees with today's, so only the check against
    # the implementation's own cold build (``make``) can report it.
    tally = walk(seed, 30, strict=True, implementation=Historied(TODAY))
    assert tally.implementation == "today, history planted"
    assert tally.engine_errors == 0
    assert tally.failed == "engine", tally.failure
    assert tally.failure is not None
    lines = tally.failure.splitlines()
    found = COLD_LINE.fullmatch(lines[0])
    assert found is not None and int(found[1]) == seed, lines[0]
    assert lines[1].startswith("cold build: the "), tally.failure
