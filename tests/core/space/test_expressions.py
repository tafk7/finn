# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded arithmetic lowers into the same typed, guarded dependency runtime."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from finn.core.space import (
    Available,
    Const,
    Decision,
    Inapplicable,
    Param,
    Space,
    Subspace,
    View,
    compile_space,
    derived,
    divisors_of,
    inspection,
)
from finn.core.space.declarations import ValueRef
from finn.core.space.errors import DefinitionError, EvaluationError
from finn.core.space.expressions import Expr


def test_integer_and_reflected_operators_preserve_python_integer_results() -> None:
    class Arithmetic(Space):
        left = Param(int)
        right = Param(int)
        added = left + right
        subtracted = left - right
        multiplied = left * right
        divided = left // right
        remainder = left % right
        negated = -left
        reflected_add = 2 + left
        reflected_sub = 2 - left
        reflected_mul = 2 * left
        reflected_div = 20 // right
        reflected_mod = 20 % right

    point = Arithmetic({Arithmetic.left: -7, Arithmetic.right: 3})
    assert (point.added, point.subtracted, point.multiplied) == (-4, -10, -21)
    assert (point.divided, point.remainder, point.negated) == (-3, 2, 7)
    assert (point.reflected_add, point.reflected_sub, point.reflected_mul) == (-5, 9, -14)
    assert (point.reflected_div, point.reflected_mod) == (6, 2)


def test_an_inferred_derived_integer_remains_an_ordinary_callback() -> None:
    calls: list[str] = []

    class Family(Space):
        extent = Param(int)

        @derived
        def doubled(*, extent: int) -> int:
            calls.append("doubled")
            return extent * 2

        result = doubled + 3

    model = compile_space(Family)
    assert calls == []
    point = model.bind({Family.extent: 4})
    assert point.result == 11
    assert calls == ["doubled"]
    assert point.result == 11
    assert calls == ["doubled"]
    dependencies = inspection.dependencies(model, Family.result)
    assert any(item.key == "doubled" for item in dependencies)


def test_literal_expressions_evaluate_lazily_and_preserve_owned_dependencies() -> None:
    class Family(Space):
        base = Const(4)
        folded = (base + 3) * 2
        anonymous = Const(8) // 2

    model = compile_space(Family)
    metadata = {item.key: item for item in inspection.members(model)}
    assert metadata["folded"].kind == "derived"
    assert metadata["anonymous"].kind == "derived"
    point = model.bind()
    assert (point.folded, point.anonymous) == (14, 4)
    assert inspection.dependencies(model, Family.folded)
    evidence = inspection.explain(point, Family.folded)
    assert evidence.result == Available(14)
    assert {node.declaration.owner for node in evidence.nodes} == {"base", "folded"}
    assert len(evidence.nodes) > 1


def test_anonymous_expressions_work_in_aliases_domains_and_child_bindings() -> None:
    class Child(Space):
        width = Param(int)
        physical = View(width * 2)

    class Root(Space):
        extent = Param(int)
        factor = Decision(int, domain=divisors_of(extent * 2))
        child = Subspace(Child, width=extent + 1)

        @derived(value=extent * 3)
        def result(*, value: int) -> int:
            return value + 1

    point = Root({Root.extent: 5})
    assert point.result == 16
    assert point.field(Root.factor).candidates() == Available((1, 2, 5, 10))
    assert point.child.physical() == 12


def test_arithmetic_errors_are_deferred_until_guarded_expression_is_demanded() -> None:
    class Family(Space):
        enabled = Param(bool)
        physical = View(Const(1) // 0, when=enabled)

    model = compile_space(Family)
    inactive = model.bind({Family.enabled: False})
    assert isinstance(inactive.physical.inspect().accepted_result, Inapplicable)
    evidence = inspection.explain(inactive, Family.physical)
    assert not any(".$expr." in node.declaration.key for node in evidence.nodes)
    with pytest.raises(EvaluationError) as error:
        model.bind({Family.enabled: True}).physical()
    assert error.value.owner == "physical"
    assert isinstance(error.value.__cause__, ZeroDivisionError)


def test_repeated_placements_keep_expression_values_and_guards_independent() -> None:
    class Child(Space):
        size = Param(int)
        result = 12 // size

    class Root(Space):
        disabled = Const(False)
        first = Subspace(Child, size=2)
        second = Subspace(Child, size=3)
        unused = Subspace(Child, size=0, when=disabled)

    point = Root()
    assert (point.first.result, point.second.result) == (6, 4)
    assert isinstance(point.unused.query(Child.result), Inapplicable)
    assert point.query(Root.first.ref(Child.result)) == Available(6)


def test_expression_truthiness_and_non_integer_operands_are_rejected() -> None:
    value = Param(int)
    expression = value + 1
    with pytest.raises(TypeError, match="truth value"):
        bool(expression)
    for bad in (True, 1.5, "1", Param(bool), Param(str)):
        with pytest.raises(DefinitionError, match="int"):
            Expr("add", value, cast(int, bad))


def test_inferred_non_integer_operands_fail_before_their_callbacks_run() -> None:
    calls: list[str] = []

    class Family(Space):
        @derived
        def text() -> str:
            calls.append("text")
            return "1"

        invalid = cast(ValueRef[int], text) + 1

    with pytest.raises(DefinitionError, match="int value semantics"):
        compile_space(Family)
    assert calls == []


def test_expression_dags_and_deep_chains_are_linked_iteratively() -> None:
    source = Param(int)
    value: ValueRef[int] = source
    for _ in range(1_500):
        value = value + 1
    family = cast(type[Space], type("DeepExpression", (Space,), {"source": source, "value": value}))
    model = compile_space(family)
    assert model.bind({source: 2}).query(value) == Available(1_502)

    source = Param(int)
    value = source
    for _ in range(20):
        value = value + value
    shared = cast(
        type[Space], type("SharedExpression", (Space,), {"source": source, "value": value})
    )
    shared_model = compile_space(shared)
    assert inspection.statistics(shared_model).nodes <= 42
    assert shared_model.bind({source: 3}).query(value) == Available(3 * 2**20)


def test_a_compiled_expression_keeps_its_operator_after_declaration_mutation() -> None:
    class Family(Space):
        source = Param(int)
        value = source + 1

    old = compile_space(Family)
    Family.value.operator = "mul"
    new = compile_space(Family)
    assert old.bind({Family.source: 3}).value == 4
    assert new is old
    assert new.bind({Family.source: 3}).value == 4


def test_shared_expression_prefix_is_not_duplicated_across_consumers() -> None:
    def family(consumers: int) -> type[Space]:
        source = Param(int)
        prefix: ValueRef[int] = source
        for _ in range(200):
            prefix = prefix + 1
        disabled = Const(False)
        members: dict[str, object] = {"source": source, "disabled": disabled}
        for index in range(consumers):
            members[f"consumer{index}"] = View(prefix, when=disabled if index == 0 else None)
        return cast(type[Space], type("SharedConsumers", (Space,), members))

    small_type, large_type = family(2), family(40)
    small = compile_space(small_type)
    large = compile_space(large_type)
    small_counts = inspection.statistics(small)
    large_counts = inspection.statistics(large)
    # Only the additional public view declarations and their output edges grow;
    # the entire 200-operation anonymous prefix stays shared across consumers.
    assert large_counts.nodes - small_counts.nodes == 38
    assert large_counts.potential_edges - small_counts.potential_edges == 38
    source = cast(ValueRef[int], getattr(large_type, "source"))
    first = cast(View[int], getattr(large_type, "consumer0"))
    last = cast(View[int], getattr(large_type, "consumer39"))
    point = large.bind({source: 2})
    assert isinstance(point.inspect(first).accepted_result, Inapplicable)
    assert point.inspect(last).accepted_result == Available(202)
    # The source owner names the first mention; the actual demand still starts
    # at the later enabled consumer. No first-consumer guard contaminates it.
    evidence = inspection.explain(point, last)
    assert evidence.query.owner == "consumer39"
    assert any(
        node.declaration.generated and node.declaration.owner == "consumer0"
        for node in evidence.nodes
    )
    assert not any(node.declaration.key == "consumer0" for node in evidence.nodes)


def test_strict_expression_types(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    assert mypy is not None
    root = Path(__file__).resolve().parents[3]
    fixtures = Path(__file__).with_name("typing")
    source = (fixtures / "expressions_negative.py.txt").read_text()
    negative = tmp_path / "expressions_negative.py"
    negative.write_text(source)
    environment = dict(os.environ, MYPYPATH=f"{root / 'src'}:{root / 'tests'}")
    result = subprocess.run(
        [
            mypy,
            "--strict",
            "--explicit-package-bases",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "cache"),
            str(fixtures / "expressions_positive.py"),
            str(negative),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    expected = {line for line, value in enumerate(source.splitlines(), 1) if "# E" in value}
    actual = {
        int(line) for line in re.findall(r"expressions_negative\.py:(\d+): error:", result.stdout)
    }
    assert result.returncode == 1, result.stdout + result.stderr
    assert actual == expected, result.stdout + result.stderr
    assert "expressions_positive.py:" not in result.stdout, result.stdout
