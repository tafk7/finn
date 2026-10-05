# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiler ordering, ownership, and immutable definition invariants."""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import Space, default_semantics, design_space, inspection
from finn.core.space._nodes import node_record
from finn.core.space.compiler import compile_model
from finn.core.space.declarations import (
    Const,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    Param,
    View,
    constraint,
    derived,
    view,
)
from finn.core.space.domains import divisors_of, domain
from finn.core.space.errors import DefinitionError, RequestError
from finn.core.space.results import Available
from finn.core.space.semantics import ValueSemantics


def test_compile_links_forward_dependencies_without_executing_callbacks() -> None:
    calls: list[str] = []

    class Example(Space):
        @derived
        def cycles(*, extent: int, lanes: int) -> int:
            calls.append("cycles")
            return extent // lanes

        extent: int = Param()
        lanes: int = Decision(domain=divisors_of(extent))
        label = Const("flat space")

        @constraint
        def supported(*, extent: int) -> bool:
            calls.append("supported")
            return extent > 0

        admitted = ConstraintGroup(supported)
        result = View(cycles, requires=(admitted,))

    model = compile_model(Example)
    assert calls == []
    nodes = model.linked.nodes
    cycles = nodes[model.resolve(0, Example.cycles)]
    assert {argument.name for argument in cycles.arguments} == {"extent", "lanes"}
    position = {index: offset for offset, index in enumerate(model.linked.order)}
    assert all(
        position[dependency] < position[node.index]
        for node in nodes
        for dependency in node.dependencies
    )
    assert model.linked.parameters == (model.resolve(0, Example.extent),)
    assert model.linked.decisions == (model.resolve(0, Example.lanes),)
    assert nodes[model.resolve(0, Example.result)].semantics is not None


def test_placement_literals_are_frozen_once_per_declaration_and_reads_stay_detached() -> None:
    snapshots: list[list[int]] = []

    def snapshot(value: list[int]) -> list[int]:
        snapshots.append(list(value))
        return list(value)

    semantics: ValueSemantics[list[int]] = ValueSemantics(
        list,
        "integer list",
        lambda value: type(value) is list,
        lambda left, right: left == right,
        snapshot,
    )
    literal = [1]

    class Leaf(Space):
        value: list[int] = Param(semantics=semantics)

    class Branch(Space):
        leaf = Leaf(value=literal)

    # A literal binding is frozen once, where its node is declared.
    assert snapshots == [[1]]

    class Root(Space):
        left = Branch()
        right = Branch()

    model = compile_model(Root)
    assert snapshots == [[1]]  # compiling reuses the declaration's frozen literal
    literal.append(2)
    point = design_space(Root())
    point.left.leaf.value.append(99)
    assert point.left.leaf.value == point.right.leaf.value == [1]
    assert compile_model(Root) is model
    assert inspection.model(point) is model

    # A fresh declaration freezes the literal's current value; no cache lets
    # it reuse, or disturb, the earlier declaration's frozen literal.
    class Fresh(Space):
        leaf = Leaf(value=literal)

    assert design_space(Fresh()).leaf.value == [1, 2]
    assert design_space(Branch()).leaf.value == [1]
    assert point.left.leaf.value == [1]


def test_self_invocation_is_preserved_for_nested_functions_and_view_outputs() -> None:
    class Child(Space):
        source: int = Param()

        @derived
        def scalar(self) -> int:
            raise AssertionError("preparation must not execute self methods")

        @constraint
        def supported(self) -> bool:
            raise AssertionError("preparation must not execute self methods")

        @view(requires=(supported,))
        def physical(self) -> int:
            raise AssertionError("preparation must not execute self methods")

        @derived
        def explicit(*, source: int) -> int:
            raise AssertionError("preparation must not execute explicit providers")

    class Parent(Space):
        child = Child(source=4)

    model = compile_model(Parent)
    nodes = model.linked.nodes
    child_scope = model.linked.scopes[0].children[node_record(Parent.child)]
    scalar = nodes[model.resolve(child_scope, Child.scalar)]
    supported = nodes[model.resolve(child_scope, Child.supported)]
    physical = nodes[model.resolve(child_scope, Child.physical)]
    assert physical.output is not None
    for node in (scalar, supported, nodes[physical.output]):
        assert node.call_style == "self"
        assert node.scope == child_scope
        assert node.arguments == ()
    assert set(physical.dependencies) == {supported.index, physical.output}
    explicit = nodes[model.resolve(child_scope, Child.explicit)]
    assert explicit.call_style == "explicit"
    assert len(explicit.arguments) == 1
    assert explicit.arguments[0].node == model.resolve(child_scope, Child.source)


def test_function_and_value_views_have_an_explicit_raw_output() -> None:
    class Example(Space):
        size: int = Param()

        @derived
        def doubled(*, size: int) -> int:
            return size * 2

        value_view = View(doubled)

        @view
        def function_view(*, size: int) -> int:
            return size * 2

    model = compile_model(Example)
    for declaration in (Example.value_view, Example.function_view):
        node = model.linked.nodes[model.resolve(0, declaration)]
        assert node.kind == "view"
        assert node.output is not None
        assert model.linked.nodes[node.output].kind == "derived"


def test_a_transitive_triangle_is_acyclic_and_cycles_name_only_their_members() -> None:
    class Triangle(Space):
        first: int = Param()

        @derived
        def second(*, first: int) -> int:
            return first

        @derived
        def third(*, first: int, second: int) -> int:
            return first + second

    compile_model(Triangle)

    class Cyclic(Space):
        @derived
        def first(*, second: int) -> int:
            return second

        @derived
        def second(*, first: int) -> int:
            return first

        @derived
        def user(*, first: int) -> int:
            return first

        @derived
        def self_cycle(*, self_cycle: int) -> int:
            return self_cycle

        unrelated = Const(1)

    with pytest.raises(DefinitionError) as error:
        compile_model(Cyclic)
    assert [dict(finding.details)["members"] for finding in error.value.findings] == [
        ("first", "second"),
        ("self_cycle",),
    ]
    assert all(finding.code == "cyclic-dependency" for finding in error.value.findings)


def test_twenty_thousand_dependencies_compile_and_evaluate_iteratively() -> None:
    def step(*, previous: int) -> int:
        return previous + 1

    members: dict[str, object] = {"value0": Const(0)}
    for number in range(1, 20_001):
        members[f"value{number}"] = Derived(
            step, aliases={"previous": members[f"value{number - 1}"]}
        )
    space_type = cast(type[Space], type("Deep", (Space,), members))
    model = compile_model(space_type)
    assert len(model.linked.order) == 20_001
    assert len(model.linked.nodes[-1].dependencies) == 1
    assert model.linked.order[-1] == model.resolve(0, members["value20000"])
    assert design_space(space_type()).query(cast(Derived[int], members["value20000"])) == Available(
        20_000
    )


def test_compiled_handles_do_not_follow_later_class_rebinding() -> None:
    class Example(Space):
        value: int = Param()

    original = Example.value
    old = compile_model(Example)
    replacement = cast(Param[int], Param(semantics=default_semantics(int)))
    replacement.__set_name__(Example, "value")
    with pytest.raises(DefinitionError, match="finalized"):
        Example.value = replacement  # type: ignore[assignment]
    new = compile_model(Example)

    old_semantics = old.linked.nodes[old.resolve(0, original)].semantics
    assert old_semantics is not None and old_semantics.type_token is int
    assert new is old
    with pytest.raises(RequestError, match="compiled scope"):
        old.resolve(0, replacement)
    assert new.resolve(0, original) == old.resolve(0, original)
    with pytest.raises(RequestError, match="scope"):
        old.resolve(1, original)


def test_inherited_dependencies_bind_to_overrides_without_mutating_base() -> None:
    class Base(Space):
        size = Const(4)

        @derived
        def doubled(*, size: int) -> int:
            return size * 2

    class Child(Base):
        size = Const(8)

    original = compile_model(Base)
    child = compile_model(Child)
    assert child.resolve(0, Base.size) == child.resolve(0, Child.size)
    original_node = original.linked.nodes[original.resolve(0, Base.size)]
    child_node = child.linked.nodes[child.resolve(0, Child.size)]
    assert (original_node.value, child_node.value) == (4, 8)
    assert child.linked.nodes[child.resolve(0, Base.doubled)].arguments[0].node == child_node.index


def test_compile_snapshots_values_domains_and_callback_references() -> None:
    values = [[1], [2]]

    def first(*, value: int) -> int:
        return value + 1

    def second(*, value: int) -> int:
        return value + 2

    class Example(Space):
        value: int = Param()
        constant = Const([1])
        choice: list[int] = Decision(values=values)
        calculated = Derived(first)

    old = compile_model(Example)
    Example.constant.value.append(2)
    finite_values = cast(Decision[list[int]], Example.choice).domain._finite_values
    assert finite_values is not None
    finite_values[0].append(3)
    Example.calculated.function = second
    new = compile_model(Example)
    old_constant = old.linked.nodes[old.resolve(0, Example.constant)]
    old_choice = old.linked.nodes[old.resolve(0, Example.choice)]
    old_function = old.linked.nodes[old.resolve(0, Example.calculated)]
    assert old_constant.value == [1]
    assert old_choice.domain is not None
    assert old_choice.domain._finite_values == ([1], [2])
    assert old_function.function is first
    assert new is old
    assert new.linked.nodes[new.resolve(0, Example.calculated)].function is first


def test_foreign_value_and_obligation_references_are_definition_errors() -> None:
    class Other(Space):
        value = Const(1)

        @constraint
        def valid(*, value: int) -> bool:
            return value > 0

    class ForeignView(Space):
        result = View(Other.value)

    class ForeignConstraint(Space):
        own = Const(1)
        result = View(own, requires=(Other.valid,))

    class WrongKind(Space):
        own = Const(1)
        result = View(own, requires=(cast(Constraint, own),))

    for space_type in (ForeignView, ForeignConstraint, WrongKind):
        with pytest.raises(DefinitionError):
            compile_model(space_type)


def test_domain_binding_is_validated_without_invocation() -> None:
    calls: list[str] = []

    def membership(*, candidate: int, maximum: int) -> bool:
        calls.append("membership")
        return candidate <= maximum

    class Example(Space):
        maximum: int = Param()
        choice: int = Decision(domain=domain(accepts=membership, maximum=maximum))

    model = compile_model(Example)
    assert calls == []
    assert model.linked.nodes[model.resolve(0, Example.choice)].domain_arguments[
        0
    ].node == model.resolve(0, Example.maximum)

    class WrongSignature(Space):
        maximum: int = Param()
        choice: int = Decision(domain=domain(accepts=membership, misspelled=maximum))

    with pytest.raises(DefinitionError, match="domain membership signature"):
        compile_model(WrongSignature)

    class ForeignDomain(Space):
        choice: int = Decision(domain=domain(accepts=membership, maximum=Example.maximum))

    with pytest.raises(DefinitionError, match="not a member"):
        compile_model(ForeignDomain)


@pytest.mark.parametrize("name", ["query", "field", "with_choices", "root", "_state"])
def test_reserved_names_are_not_declarations(name: str) -> None:
    with pytest.raises(DefinitionError, match="reserved"):
        compile_model(type("Reserved", (Space,), {name: Const(1)}))


def test_incompatible_inferred_derived_override_is_rejected() -> None:
    class Base(Space):
        @derived
        def output() -> int:
            return 1

    def output() -> str:
        return "different"

    changed = cast(type[Space], type("Changed", (Base,), {"output": Derived(output)}))

    with pytest.raises(DefinitionError, match="semantics"):
        compile_model(changed)


def test_missing_binding_names_fail_before_any_callback() -> None:
    def invalid(*, unknown: int) -> int:
        raise AssertionError("a compiler must never discover dependencies by execution")

    class Example(Space):
        result = Derived(invalid)

    with pytest.raises(DefinitionError, match="no declaration"):
        compile_model(Example)


def test_generic_alias_adapter_tokens_keep_identity_for_input_and_output_annotations() -> None:
    token = tuple[int, ...]
    vector: ValueSemantics[tuple[int, ...]] = ValueSemantics(
        token,
        "integer vector",
        lambda value: type(value) is tuple and all(type(item) is int for item in value),
        lambda left, right: left == right,
        lambda value: value,
    )

    class Example(Space):
        source: tuple[int, ...] = Param(semantics=vector)

        @derived(semantics=vector)
        def result(*, source: tuple[int, ...]) -> tuple[int, ...]:
            return source + (3,)

    model = compile_model(Example)
    point = design_space(Example(source=(1, 2)))
    assert inspection.model(point) is model  # plain root values stay runtime inputs
    assert point.result == (1, 2, 3)
    semantics = model.linked.nodes[model.resolve(0, Example.result)].semantics
    assert semantics is not None and semantics.type_token is token
