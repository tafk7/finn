# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness comparing the engine with its reference model, held to its own rules: the
model imports the standard library only, every row of the acceptance matrix has exactly
one witness and every expected failure a row, and the comparison reads blockers as a
subset and everything else exactly."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from core.space import reference, test_reference_cases, test_reference_rules
from core.space.reference import Change, Choice, Scope, Selector, Space, listed
from core.space.reference_harness import (
    MILESTONES,
    ROWS,
    Case,
    Disagreement,
    NotTranslatable,
    StateForm,
    ValidityForm,
    markdown,
)
from core.space.reference_harness.compare import agrees
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


def test_every_row_has_exactly_one_witness_and_every_expected_failure_a_row() -> None:
    assert ROWS
    assert not problems([test_reference_cases, test_reference_rules])


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
