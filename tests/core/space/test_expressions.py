# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded arithmetic lowers into the same typed, guarded dependency runtime."""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import (
    Available,
    Const,
    Decision,
    Inapplicable,
    Param,
    Space,
    View,
    composite,
    default_semantics,
    derived,
    design_space,
    divisors_of,
    inspection,
)
from finn.core.space.declarations import ValueRef
from finn.core.space.errors import DefinitionError, EvaluationError
from finn.core.space.expressions import Expr


def test_integer_and_reflected_operators_preserve_python_integer_results() -> None:
    class Arithmetic(Space):
        left: int = Param()
        right: int = Param()
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

    point = design_space(Arithmetic(left=-7, right=3))
    assert (point.added, point.subtracted, point.multiplied) == (-4, -10, -21)
    assert (point.divided, point.remainder, point.negated) == (-3, 2, 7)
    assert (point.reflected_add, point.reflected_sub, point.reflected_mul) == (-5, 9, -14)
    assert (point.reflected_div, point.reflected_mod) == (6, 2)


def test_an_inferred_derived_integer_remains_an_ordinary_callback() -> None:
    calls: list[str] = []

    class Family(Space):
        extent: int = Param()

        @derived
        def doubled(*, extent: int) -> int:
            calls.append("doubled")
            return extent * 2

        result = doubled + 3

    model = inspection.model(Family)
    assert calls == []
    point = design_space(Family(extent=4))
    assert inspection.model(point) is model
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

    model = inspection.model(Family)
    metadata = {item.key: item for item in inspection.members(model)}
    assert metadata["folded"].kind == "derived"
    assert metadata["anonymous"].kind == "derived"
    point = design_space(Family())
    assert (point.folded, point.anonymous) == (14, 4)
    assert inspection.dependencies(model, Family.folded)
    evidence = inspection.explain(point, Family.folded)
    assert evidence.result == Available(14)
    assert {node.declaration.owner for node in evidence.nodes} == {"base", "folded"}
    assert len(evidence.nodes) > 1


def test_anonymous_expressions_work_in_aliases_domains_and_child_bindings() -> None:
    class Child(Space):
        width: int = Param()
        physical = View(width * 2)

    class Root(Space):
        extent: int = Param()
        factor: int = Decision(domain=divisors_of(extent * 2))
        child = Child(width=extent + 1)
        # Arithmetic over a node's member reference is an expression too, here
        # supplying another node's formal.
        other = Child(width=child.width * 2 - extent)

        @derived(value=extent * 3)
        def result(*, value: int) -> int:
            return value + 1

    point = design_space(Root(extent=5))
    assert point.result == 16
    assert point.field(Root.factor).candidates() == Available((1, 2, 5, 10))
    assert point.child.physical == 12
    assert (point.other.width, point.other.physical) == (7, 14)


def test_arithmetic_errors_are_deferred_until_guarded_expression_is_demanded() -> None:
    class Family(Space):
        enabled: bool = Param()
        physical = View(Const(1) // 0, when=enabled)

    inactive = design_space(Family(enabled=False))
    assert isinstance(inactive.inspect(Family.physical).accepted_result, Inapplicable)
    evidence = inspection.explain(inactive, Family.physical)
    assert not any(".$expr." in node.declaration.key for node in evidence.nodes)
    with pytest.raises(EvaluationError) as error:
        design_space(Family(enabled=True)).physical
    assert error.value.owner == "physical"
    assert isinstance(error.value.__cause__, ZeroDivisionError)


def test_repeated_placements_keep_expression_values_and_guards_independent() -> None:
    class Child(Space):
        size: int = Param()
        result = 12 // size

    class Root(Space):
        disabled = Const(False)
        first = Child(size=2)
        second = Child(size=3)
        unused = Child(size=0, when=disabled)
        # References are typed as their values, so this is statically an int;
        # at runtime it is an expression over the two placements' members.
        total = first.result + second.result

    assert isinstance(vars(Root)["total"], Expr)
    point = design_space(Root())
    assert (point.first.result, point.second.result) == (6, 4)
    assert isinstance(point.unused.query(Child.result), Inapplicable)
    assert point.query(Root.first.result) == Available(6)
    assert point.total == 10
    assert point.query(Root.total) == Available(10)


def test_expression_truthiness_and_non_integer_operands_are_rejected() -> None:
    value: int = Param(semantics=default_semantics(int))
    expression = value + 1
    with pytest.raises(TypeError, match="truth value"):
        bool(expression)
    for bad in (True, 1.5, "1", Const(True), Const("1")):
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
        design_space(Family())
    assert calls == []


def test_a_reference_to_a_non_integer_member_is_refused_as_an_operand() -> None:
    class Child(Space):
        label: str = Param()

    child = Child(label="x")
    with pytest.raises(DefinitionError, match="int value semantics"):
        child.label + 1  # type: ignore[operator]


class Source(Space):
    """A typed base for families whose expressions are built as data."""

    source: int = Param()


def test_expression_dags_and_deep_chains_are_linked_iteratively() -> None:
    value: int = Source.source
    for _ in range(1_500):
        value = value + 1
    family = composite("DeepExpression", {"value": value}, base=Source)
    assert design_space(family(source=2)).query(value) == Available(1_502)

    value = Source.source
    for _ in range(20):
        value = value + value
    shared = composite("SharedExpression", {"value": value}, base=Source)
    assert inspection.statistics(shared).nodes <= 42
    assert design_space(shared(source=3)).query(value) == Available(3 * 2**20)


def test_a_compiled_expression_keeps_its_operator_after_declaration_mutation() -> None:
    class Family(Space):
        source: int = Param()
        value = source + 1

    old = design_space(Family(source=3))
    cast(Expr, Family.value).operator = "mul"
    new = design_space(Family(source=3))
    assert old.value == 4
    assert inspection.model(new) is inspection.model(old)
    assert new.value == 4


def test_shared_expression_prefix_is_not_duplicated_across_consumers() -> None:
    def family(consumers: int) -> type[Source]:
        prefix: int = Source.source
        for _ in range(200):
            prefix = prefix + 1
        disabled = Const(False)
        members: dict[str, object] = {"disabled": disabled}
        for index in range(consumers):
            members[f"consumer{index}"] = View(prefix, when=disabled if index == 0 else None)
        return composite("SharedConsumers", members, base=Source)

    small_type, large_type = family(2), family(40)
    small_counts = inspection.statistics(small_type)
    large_counts = inspection.statistics(large_type)
    # Only the additional public view declarations and their output edges grow;
    # the entire 200-operation anonymous prefix stays shared across consumers.
    assert large_counts.nodes - small_counts.nodes == 38
    assert large_counts.potential_edges - small_counts.potential_edges == 38
    first = cast(View[int], getattr(large_type, "consumer0"))
    last = cast(View[int], getattr(large_type, "consumer39"))
    point = design_space(large_type(source=2))
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
