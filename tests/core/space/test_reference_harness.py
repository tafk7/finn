# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness comparing the engine with its reference model, held to its own rules: the
model imports the standard library only, every row of the acceptance matrix has exactly
one witness and every expected failure a row, the walks tell disagreements only by the
kinds the rows name, and the comparison reads blockers as a subset and everything else
exactly. The walks' own checks are ``test_reference_walks``'s."""

from __future__ import annotations

import ast
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
    When,
    computed,
    listed,
)
from core.space.reference_harness import (
    MILESTONES,
    ROWS,
    Case,
    Disagreement,
    NotTranslatable,
    OutcomeForm,
    PossibleForm,
    RefusalForm,
    RuleForm,
    StateForm,
    ValidityForm,
    markdown,
)
from core.space.reference_harness.compare import (
    agrees,
    change_form,
    rule_form,
    validity_form,
)
from core.space.reference_harness.matrix import problems
from core.space.reference_harness.walks import CLASSIFIED, UNCLASSIFIED, pending
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


def test_the_walks_tell_disagreements_only_by_kinds_the_rows_name() -> None:
    # A walk tallies each disagreement under a row's kind, or as unclassified: never a kind
    # no row names. The walks count while a milestone with rows is pending, which is now.
    assert set(CLASSIFIED) <= {row.kind for row in ROWS}
    assert UNCLASSIFIED not in {row.kind for row in ROWS}
    assert pending() and all(any(row.milestone == m for row in ROWS) for m in pending())


def test_the_matrix_prints_as_a_markdown_table_in_milestone_order() -> None:
    table = markdown().splitlines()
    assert table[0].startswith("| Row | Milestone |") and len(table) == len(ROWS) + 2
    milestones = [line.split(" | ")[1] for line in table[2:]]
    assert milestones == sorted(milestones, key=MILESTONES.index)
    assert all(f"| {row.id} |" in markdown() for row in ROWS)


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


def test_the_provisional_codes_mark_an_empty_list_and_a_refused_guard() -> None:
    # ``domain-empty`` and ``guard-refused``: the engine core adopts them, or both rename.
    # ``on-n`` is false through the guard: it carries the guard's origin, ``gate``.
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
