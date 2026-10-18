# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness comparing the engine with its reference model, held to its own rules: the
model imports the standard library only, every row of the acceptance matrix has exactly
one witness and every expected failure a row, and the comparison reads blockers as a
subset and everything else exactly."""

from __future__ import annotations

import ast
import dataclasses
import random
import sys
from pathlib import Path

import pytest

from core.space import reference
from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Refuse,
    Scope,
    Selector,
    Space,
    computed,
    listed,
)
from core.space.reference_harness import (
    MILESTONES,
    ROWS,
    Case,
    Disagreement,
    NotTranslatable,
    RuleForm,
    StateForm,
    ValidityForm,
    markdown,
)
from core.space.reference_harness.compare import agrees, rule_form
from core.space.reference_harness.matrix import problems
from core.space.reference_harness.walks import generate
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
    assert ROWS and all(path.exists() for path in CASE_MODULES)
    assert not problems(CASE_MODULES)


def test_the_matrix_check_finds_a_row_without_its_witness(tmp_path: Path) -> None:
    """A check that has never rejected anything is a check nobody has tested."""
    witness = ROWS[0].witness.split("::")[1]
    elsewhere = tmp_path / "test_reference_cases.py"
    elsewhere.write_text(f"@pytest.mark.xfail(strict=True)\ndef {witness}() -> None: ...\n")
    found = problems([elsewhere])
    assert f"test_reference_cases::{witness}: an expected failure with no row" in found
    assert any(item.startswith(f"{ROWS[0].id}: witnessed by []") for item in found)


def test_the_matrix_is_built_on_the_models_default_trial_reach() -> None:
    # Off: a value case's trial does not count other stored choices' own checks, so a forced
    # value may make a retained choice's own check false, and the change is refused. Every
    # case and every walk compares the model's default; flipping it moves answers, and the
    # rows with them.
    defaults = {item.name: item.default for item in dataclasses.fields(Space)}
    assert defaults["trials_count_retained_checks"] is False
    assert not generate(random.Random(0))[0].description.trials_count_retained_checks
    other = dataclasses.replace(Space((), Scope("")), trials_count_retained_checks=True)
    with pytest.raises(ValueError, match="default trial reach"):
        Case("the other reading", other)


def test_the_matrix_prints_as_a_markdown_table_in_milestone_order() -> None:
    table = markdown().splitlines()
    assert table[0].startswith("| Row | Milestone |") and len(table) == len(ROWS) + 2
    milestones = [line.split(" | ")[1] for line in table[2:]]
    assert milestones == sorted(milestones, key=MILESTONES.index)
    assert all(f"| {row.id} |" in markdown() for row in ROWS)


# -- the comparison --------------------------------------------------------------------------------


def test_the_engine_may_name_fewer_blockers_than_the_model_never_others() -> None:
    model = ValidityForm("undecided", blockers=frozenset({"x", "y"}))
    assert agrees(model, ValidityForm("undecided", blockers=frozenset({"x"})))
    assert not agrees(model, ValidityForm("undecided", blockers=frozenset({"z"})))
    assert not agrees(model, ValidityForm("valid"))


def test_the_possible_cases_of_a_choice_whose_answer_waits_are_not_compared() -> None:
    waiting = StateForm("open", possible=(1, 2), blockers=frozenset({"want"}))
    assert agrees(waiting, StateForm("open", blockers=frozenset({"want"})))
    assert not agrees(StateForm("open", possible=(1, 2)), StateForm("open", possible=(1,)))


def test_values_compare_by_type_as_well() -> None:
    assert not agrees(StateForm("forced", True), StateForm("forced", 1))


def test_a_rule_reading_a_refused_choice_carries_the_reasons_that_refuse_it() -> None:
    # Whatever refuses the choice it reads: its own check (a value outside its domain), its
    # guard (admitted under a selector with no possible case) or its domain's input (refused
    # on facts), the rule is false with those reasons' codes, never with none.
    cores = Selector(
        "compute",
        (
            Scope(
                "packed",
                choices=(Choice("pe", listed(1, 2, 3)),),
                constraints=(Constraint("fits", ("width",), lambda r: r["width"] <= 8, "wide"),),
            ),
            Scope(
                "stub",
                constraints=(Constraint("even", ("width",), lambda r: r["width"] % 2 == 0, "odd"),),
            ),
        ),
    )
    space = Space(
        ("width", "w"),
        Scope(
            "",
            choices=(
                Choice("x", listed(1, 2)),
                Choice(
                    "lanes",
                    computed(lambda r: tuple(d for d in (1, 2) if r["half"] % d == 0), ("half",)),
                ),
            ),
            selectors=(cores,),
            derived=(Derived("half", ("w",), lambda r: Refuse("odd-w") if r["w"] % 2 else r["w"]),),
            constraints=(
                Constraint("on-x", ("x",), lambda r: r["x"] > 0, "x"),
                Constraint(
                    "on-pe", ("compute.packed.pe",), lambda r: r["compute.packed.pe"] > 0, "pe"
                ),
                Constraint("on-lanes", ("lanes",), lambda r: r["lanes"] > 0, "lanes"),
            ),
        ),
    )
    stored = {"x": 3, "compute.packed.pe": 3, "lanes": 1}
    solution = reference.Configuration(space, {"width": 9, "w": 7}, stored).solve()
    assert isinstance(solution.state("x"), reference.RefusedChoice)
    assert isinstance(solution.state("compute.packed.pe"), reference.GuardRefused)
    assert isinstance(solution.state("lanes"), reference.InputRefused)
    assert rule_form(solution, "on-x") == RuleForm("false", frozenset({("domain-membership", "x")}))
    assert rule_form(solution, "on-pe") == RuleForm(
        "false", frozenset({("decision-no-viable-case", "compute")})
    )
    assert rule_form(solution, "on-lanes") == RuleForm("false", frozenset({("odd-w", "half")}))


def test_a_disagreement_names_the_case_the_question_and_both_answers() -> None:
    with pytest.raises(Disagreement) as caught:
        LISTED.start().check_state("x", ruled=StateForm("forced", 2))
    assert isinstance(caught.value, AssertionError)
    text = str(caught.value)
    assert text.splitlines()[0] == "listed: the state of x"
    assert "! kind: model 'forced' | engine 'open'" in text
    assert "! value: model 2 | engine None" in text
    assert "  codes: model {} | engine {}" in text


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


@pytest.mark.usefixtures("index_path")
def test_every_choice_keeps_its_path_as_its_key() -> None:
    point = NESTED.start({"width": 4}).engine.point
    keys = {item.key for item in inspection.decisions(point)}
    assert keys == {"unit.compute", "unit.compute.packed.pe", "unit.compute.stub.rows"}


@pytest.mark.usefixtures("index_path")
def test_a_selector_is_read_as_its_selected_case() -> None:
    NESTED.start({"width": 4}, {"unit.compute": "stub"}).check_answer("unit.compute")


def test_a_fact_cannot_be_set_back_to_unknown_in_the_engine() -> None:
    with pytest.raises(NotTranslatable):
        NESTED.start({"width": 4}).check_change(Change(unset=("width",)))
