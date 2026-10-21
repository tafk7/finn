# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness comparing the engine with its reference model, held to its own rules: the
model imports the standard library only, every row of the acceptance matrix has exactly
one witness and every expected failure a row, and the comparison reads blockers as a
subset and everything else exactly. The walks' own checks are ``test_reference_walks``'s."""

from __future__ import annotations

import ast
import sys
from dataclasses import replace
from pathlib import Path

import pytest

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
    PossibleForm,
    RefusalForm,
    Row,
    RuleForm,
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
from finn.core.space import inspection


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


def test_the_engine_core_and_facts_instance_milestones_are_accepted_their_rows_removed() -> None:
    assert MILESTONES == ("base",) and ROWS == ()


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
    ),
)


def test_every_choice_keeps_its_path_as_its_key() -> None:
    point = NESTED.start({"width": 4}).engine.point
    keys = {item.key for item in inspection.decisions(point)}
    assert keys == {"unit.compute", "unit.compute.packed.pe", "unit.compute.stub.rows"}


def test_a_selector_is_read_as_its_selected_case() -> None:
    NESTED.start({"width": 4}, {"unit.compute": "stub"}).check_answer("unit.compute")


def test_a_fact_cannot_be_set_back_to_unknown_in_the_engine() -> None:
    with pytest.raises(NotTranslatable):
        NESTED.start({"width": 4}).check_change(Change(unset=("width",)))


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
