# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Definition structure, signature checks, and canonical linked type validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, cast

import pytest

from finn.core.space import (
    Const,
    Decision,
    DefinitionError,
    Derived,
    Param,
    QueryResult,
    RequestError,
    Space,
    ValueRef,
    ValueSemantics,
    View,
    ViewKey,
    constraint,
    derived,
    design_space,
    inspection,
    view,
)
from finn.core.space._nodes import NodeDecision, NodeDecl, node_record, unwrap
from finn.core.space._signatures import BoundArgument, validate_argument
from finn.core.space.collection import collect_space
from finn.core.space.compiler import compile_model
from finn.core.space.declarations import Declaration


def decl(value: object) -> Declaration:
    """A class-level member as its declaration: statically, members are typed as values."""
    assert isinstance(value, Declaration)
    return value


def ref(value: object) -> ValueRef[object]:
    assert isinstance(value, ValueRef)
    return value


def record_of(node: object) -> NodeDecl:
    """The record behind a class-level node declaration."""
    record = node_record(node)
    assert record is not None
    return record


def decision_record(choice: object) -> NodeDecision:
    """The record behind a class-level Decision over nodes."""
    record = unwrap(choice)
    assert isinstance(record, NodeDecision)
    return record


class Base(Space):
    extent: int = Param()

    @derived
    def doubled(*, extent: int) -> int:
        raise AssertionError("collection must not execute author callbacks")

    @derived(source=extent)
    def aliased(*, source: int) -> int:
        raise AssertionError("collection must not execute author callbacks")


class Override(Base):
    extent: int = Param()


def test_effective_collection_resolves_inherited_names_and_explicit_aliases() -> None:
    collected = collect_space(Override)
    assert collected.members["extent"] is decl(Override.extent)
    assert collected.functions["doubled"].dependencies[0].source is decl(Override.extent)
    assert collected.functions["aliased"].dependencies[0].source is decl(Override.extent)
    assert collected.aliases[decl(Base.extent)] == "extent"
    assert collected.functions["doubled"].semantics.type_token is int
    assert ref(Base.doubled).semantics is None


def test_forward_members_and_postponed_class_annotations() -> None:
    class Example(Space):
        class Quantity:
            pass

        @derived
        def result(*, later: Quantity) -> Quantity:
            raise AssertionError("must not run")

        later: Quantity = Param()

    collected = collect_space(Example)
    assert collected.functions["result"].dependencies[0].source is decl(Example.later)
    assert collected.functions["result"].semantics.type_token is Example.Quantity


def test_value_and_function_views_are_distinct_declarations_with_same_type() -> None:
    class Example(Space):
        value = Const(5)

        @constraint
        def supported(*, value: int) -> bool:
            raise AssertionError("must not run")

        @view(requires=(supported,))
        def function(*, value: int) -> int:
            raise AssertionError("must not run")

        detached = View(value, requires=(supported,))

    collected = collect_space(Example)
    assert collected.functions["function"].semantics.type_token is int
    assert collected.semantics[Example.detached].type_token is int
    assert Example.function.source is None
    assert Example.detached.source is Example.value


def test_node_member_alias_and_typed_view_exports() -> None:
    # Exports are ViewKeys to views only. A value export is refused as the wrong
    # kind; the value is exported through a View.
    width = ViewKey("width", int)
    physical = ViewKey("physical", int)

    class Child(Space):
        extent: int = Param()

        @view
        def result(*, extent: int) -> int:
            return extent

        width_view = View(extent)
        exports = {width: width_view, physical: result}

    class ValueExport(Space):
        extent: int = Param()
        exports = {width: extent}  # type: ignore[dict-item]

    with pytest.raises(DefinitionError, match="export width has the wrong kind"):
        collect_space(ValueExport)

    # The parent declares the formal and binds it to the child.
    class Parent(Space):
        extent: int = Param()
        child = Child(extent=extent)

        @derived(size=child.extent)
        def total(*, size: int) -> int:
            return size * 2

    collected = collect_space(Child)
    assert collected.exports[width] is Child.width_view
    assert collected.exports[physical] is Child.result
    dependency = collect_space(Parent).functions["total"].dependencies[0].source
    assert dependency.semantics is not None and dependency.semantics.type_token is int
    # The child's member, by reference (statically a reference is typed as its value).
    assert dependency == cast(object, Parent.child.extent)
    assert inspection.reference(dependency).path == ("child",)
    assert set(vars(Parent)) >= {"extent", "child", "total"}
    assert design_space(Parent(extent=4)).total == 8

    class Exposing(Space):
        child = Child(extent=Param())

    with pytest.raises(DefinitionError, match="inline Param cannot supply"):
        design_space(Exposing())


def test_explicit_answer_semantics_preserve_the_value_type() -> None:
    class Example(Space):
        value: int = Param()

        @derived(input=value, semantics=ValueSemantics.immutable_nominal(int))
        def forwarded(*, input: int) -> QueryResult[int]:
            raise AssertionError("collection must not evaluate")

    collected = collect_space(Example)
    assert collected.functions["forwarded"].dependencies[0].source is decl(Example.value)
    assert collected.functions["forwarded"].semantics.type_token is int


@pytest.mark.parametrize(
    ("function_source", "message"),
    [
        ("def result(value: int = 1) -> int: return value", "cannot have defaults"),
        ("def result(value: int, /) -> int: return value", "positional-only"),
        ("def result(*value: int) -> int: return 1", "variadic"),
        ("def result(**value: int) -> int: return 1", "variadic"),
        ("def result(self: int) -> int: return self", "receiver annotation"),
        ("def result(cls: int) -> int: return cls", "cls"),
        ("def result(unknown: int) -> int: return unknown", "no declaration"),
        ("def result(value) -> int: return value", "annotation is required"),
        ("def result(value: int): return value", "return annotation"),
        ("def result(value: str) -> str: return value", "cannot consume"),
    ],
)
def test_bad_signatures_fail_without_invocation(function_source: str, message: str) -> None:
    namespace: dict[str, object] = {}
    exec(function_source, namespace)
    function = cast(Callable[..., object], namespace["result"])
    example = type(
        "BadSignature",
        (Space,),
        {"__annotations__": {"value": int}, "value": Param(), "result": derived(function)},
    )
    with pytest.raises(DefinitionError, match=message):
        collect_space(example)


def test_self_receivers_are_collected_without_executing_or_inventing_dependencies() -> None:
    class Parent(Space):
        value: int = Param()

    class Example(Parent):
        @derived
        def implicit(self) -> int:
            raise AssertionError("collection must not execute a self method")

        @derived
        def concrete(self: Example) -> int:
            raise AssertionError("collection must not execute a self method")

        @derived
        def base(self: Parent) -> int:
            raise AssertionError("collection must not execute a self method")

        @derived
        def generic(self: Space) -> int:
            raise AssertionError("collection must not execute a self method")

        @derived
        def self_type(self: Self) -> int:
            raise AssertionError("collection must not execute a self method")

    collected = collect_space(Example)
    for name in ("implicit", "concrete", "base", "generic", "self_type"):
        assert collected.functions[name].dependencies == ()
        assert collected.functions[name].semantics.type_token is int


def test_self_receiver_and_explicit_argument_aliases_are_ambiguous() -> None:
    class Ambiguous(Space):
        value: int = Param()

        @derived(value=value)
        def result(self) -> int:
            raise AssertionError("ambiguous declarations must not be evaluated")

    with pytest.raises(DefinitionError, match="self.*alias|alias.*self"):
        collect_space(Ambiguous)


def test_extra_alias_and_answer_without_semantics_are_rejected() -> None:
    class Extra(Space):
        value: int = Param()

        @derived(typo=value)
        def result(*, value: int) -> int:
            return value

    with pytest.raises(DefinitionError, match="unknown arguments.*typo"):
        collect_space(Extra)

    class NoSemantics(Space):
        @derived
        def result() -> QueryResult[int]:
            raise AssertionError("must not run")

    with pytest.raises(DefinitionError, match="QueryResult.*explicit semantics"):
        collect_space(NoSemantics)


def test_incompatible_override_collision_and_reserved_names() -> None:
    different = type("Different", (Base,), {"__annotations__": {"extent": str}, "extent": Param()})
    with pytest.raises(DefinitionError, match="Different.extent.*semantics"):
        collect_space(different)
    changed_kind = type("ChangedKind", (Base,), {"extent": Const(2)})
    with pytest.raises(DefinitionError, match="declaration kind"):
        collect_space(changed_kind)
    hidden = type("Hidden", (Base,), {"extent": 3})
    with pytest.raises(DefinitionError, match="hides an inherited declaration"):
        collect_space(hidden)

    class Other(Space):
        extent: int = Param()

    collision = type("Collision", (Base, Other), {})
    with pytest.raises(DefinitionError, match="conflicting inherited"):
        collect_space(collision)
    reserved = type("Reserved", (Space,), {"__annotations__": {"query": int}, "query": Param()})
    with pytest.raises(DefinitionError, match="reserved configuration name"):
        collect_space(reserved)


def test_duplicate_declaration_reuse_is_attributable() -> None:
    value: int = Param()
    # Python 3.11 wraps a __set_name__ failure in RuntimeError; 3.12 raises it directly.
    with pytest.raises((RuntimeError, DefinitionError)) as exc:
        type(
            "Duplicate",
            (Space,),
            {"__annotations__": {"one": int, "two": int}, "one": value, "two": value},
        )
    error = exc.value.__cause__ if isinstance(exc.value, RuntimeError) else exc.value
    assert isinstance(error, DefinitionError)
    assert "Duplicate.two" in str(error)


def test_unbound_late_declarations_and_bad_exports_are_rejected() -> None:
    class Example(Space):
        pass

    setattr(Example, "late", Param())
    with pytest.raises(DefinitionError, match="not bound at class creation"):
        collect_space(Example)
    wrong = type(
        "WrongExport",
        (Space,),
        {"__annotations__": {"value": int}, "value": Param()},
    )
    wrong_member = cast(Declaration, vars(wrong)["value"])
    setattr(wrong, "exports", {ViewKey("physical", int): wrong_member})
    with pytest.raises(DefinitionError, match="wrong kind"):
        collect_space(wrong)


def test_definition_constants_snapshot_and_decision_domain_contract() -> None:
    source = [1, 2]
    constant = Const(source)
    source.append(3)
    assert constant.value == [1, 2]
    with pytest.raises(DefinitionError, match="exactly one"):
        Decision()  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="truth value"):
        bool(Param(default=1))


def test_inferred_derived_override_cannot_change_value_type() -> None:
    @derived
    def text(*, extent: int) -> str:
        return str(extent)

    changed = type("Changed", (Base,), {"doubled": text})
    with pytest.raises(DefinitionError, match="override changes value semantics"):
        collect_space(changed)


def test_binding_forms_and_local_edit_ownership() -> None:
    # A reference input is supplied by a node, and a root keeps its values as
    # runtime parameters.
    class Child(Space):
        size: int = Param()
        internal: str = Decision(values=("auto", "block"))

    class Holder(Space):
        held: Child = Param()

    class Parent(Space):
        supplied: int = Decision(values=(1, 2))
        literal = Child(size=4)
        alias = Child(size=supplied)
        local = Child(size=Decision(values=(4, 8)))
        holder = Holder(held=Child(size=2))

    # How each supply links: a literal is a constant, a reference an alias, a
    # fresh Decision a decision, a node at a reference input its placement.
    linked = compile_model(Parent).linked
    kinds = [
        linked.nodes[compile_model(Parent).resolve(0, node.size)].kind
        for node in (Parent.literal, Parent.alias, Parent.local)
    ]
    assert kinds == ["const", "alias", "decision"]
    assert linked.scopes[0].named_children["holder"] in {
        scope.parent for scope in linked.scopes if scope.name == "holder.held"
    }
    root = compile_model(Child).linked
    assert root.nodes[root.keys["size"]].kind == "param"
    model = compile_model(Parent)
    local = model.decision(0, Parent.local.size)
    assert model.linked.nodes[local].kind == "decision"
    assert model.linked.nodes[local].key == "local.size"
    assert model.linked.nodes[model.decision(0, Parent.alias.internal)].kind == "decision"
    for node in (Parent.literal, Parent.alias):
        with pytest.raises(RequestError, match="not an owned Decision"):
            model.decision(0, node.size)

    # A formal is read like any member, and only an owned fresh Decision is
    # editable through it.
    point = design_space(Parent())
    assert point.with_choices({Parent.local.size: 8}).local.size == 8
    with pytest.raises(RequestError, match="not an owned Decision"):
        point.with_choices({Parent.alias.size: 1})

    class Reader(Space):
        supplier: int = Decision(values=(1, 2))
        child = Child(size=supplier)

        @derived(value=child.size)
        def doubled(*, value: int) -> int:
            return value * 2

    reader = design_space(Reader()).with_choices(supplier=2)
    assert reader.doubled == 4


def test_child_omissions_and_incompatible_bindings_are_not_implicit_exposure() -> None:
    class Child(Space):
        width: int = Param()
        optional_width: int = Param(required=False)

    # An optional formal may stay unsupplied: it is not implicitly exposed.
    assert inspection.declaration(Child(width=8)).unsupplied == ()
    assert inspection.declaration(Child()).unsupplied == ("width",)

    class Holder(Space):
        child = Child(width=8)

    linked = compile_model(Holder).linked
    assert linked.nodes[linked.keys["child.optional_width"]].kind == "present"

    # A required formal nobody supplies is legal at the call (an assignment may
    # still supply it), and fails when a Space class containing the node is prepared.
    class Parent(Space):
        child = Child(optional_width=3)

    with pytest.raises(DefinitionError, match=r"child\.width is not supplied"):
        design_space(Parent())
    with pytest.raises(DefinitionError, match="unknown members"):
        Child(width=8, optional_width=9, typo=1)  # type: ignore[call-arg]
    with pytest.raises(DefinitionError, match="incompatible value semantics"):

        class Wrong(Space):
            label: str = Param()
            child = Child(width=label, optional_width=9)  # type: ignore[arg-type]


def test_guards_are_collected_separately_and_respect_inherited_overrides() -> None:
    class Child(Space):
        value: int = Param()

    class Guarded(Space):
        enabled: bool = Param()
        selected: int = Decision(values=(1, 2), when=enabled)

        @derived(when=enabled)
        def result(*, selected: int) -> int:
            raise AssertionError("must not execute")

        @constraint(when=enabled)
        def support(*, selected: int) -> bool:
            raise AssertionError("must not execute")

        @view(when=enabled)
        def authored(*, selected: int) -> int:
            raise AssertionError("must not execute")

        physical = View(result, requires=(support,), when=enabled)
        child = Child(value=1, when=enabled)
        # A singleton structural choice is an ordinary Decision over nodes.
        implementation: Child = Decision({"only": Child(value=1)}, when=enabled)

    class OverrideGuard(Guarded):
        enabled: bool = Param()

    effective = collect_space(OverrideGuard)
    guarded: tuple[Declaration, ...] = (
        decl(Guarded.selected),
        decl(Guarded.result),
        decl(Guarded.support),
        decl(Guarded.authored),
        decl(Guarded.physical),
        record_of(Guarded.child),
        decision_record(Guarded.implementation),
    )
    assert all(
        effective.guards[declaration] is decl(OverrideGuard.enabled) for declaration in guarded
    )
    assert [arg.name for arg in effective.functions["result"].dependencies] == ["selected"]
    assert "when" not in cast("Derived[object]", Guarded.result).aliases
    assert "when" not in inspection.declaration(Guarded.child).bindings


def test_guards_on_fresh_local_choices_and_alternatives_use_the_placement_scope() -> None:
    class Child(Space):
        value: int = Param()

    class Parent(Space):
        enabled: bool = Param()
        child = Child(value=Decision(values=(1,), when=enabled))
        choice: Child = Decision({"one": Child(value=1, when=enabled)})

    effective = collect_space(Parent)
    decision = inspection.declaration(Parent.child).bindings["value"]
    assert isinstance(decision, Decision)
    assert effective.guards[decision] is decl(Parent.enabled)
    candidate = decision_record(Parent.choice).candidates["one"]
    assert candidate is not None
    assert effective.guards[candidate] is decl(Parent.enabled)


def test_nonboolean_and_foreign_guards_fail_during_collection() -> None:
    class Wrong(Space):
        count: int = Param()
        choice: int = Decision(values=(1,), when=cast(ValueRef[bool], count))

    with pytest.raises(DefinitionError, match="Wrong.choice guard.*Boolean"):
        collect_space(Wrong)

    class Foreign(Space):
        enabled: bool = Param()

    class Unrelated(Space):
        choice: int = Decision(values=(1,), when=Foreign.enabled)

    with pytest.raises(DefinitionError, match="Unrelated.choice guard.*effective scope"):
        collect_space(Unrelated)


def test_when_is_an_authoring_control_argument() -> None:
    class Wrong(Space):
        active: bool = Param()

        @derived(when=active)
        def value(*, when: bool) -> int:
            return 1

    with pytest.raises(DefinitionError, match="reserved authoring control argument"):
        collect_space(Wrong)


def test_linked_child_types_are_inferred_without_mutating_declarations() -> None:
    boolean_view = ViewKey("admitted", bool)

    class Child(Space):
        size: int = Param()

        @derived
        def enabled(*, size: int) -> bool:
            raise AssertionError("must not execute")

        admitted = View(enabled)
        exports = {boolean_view: admitted}

    class Parent(Space):
        child = Child(size=1)

        @derived(when=child.enabled)
        def value() -> int:
            raise AssertionError("must not execute")

        copied = View(child.admitted)
        exports = {boolean_view: copied}

    model = compile_model(Parent)
    semantics = model.linked.nodes[model.resolve(0, Parent.copied)].semantics
    assert semantics is not None and semantics.type_token is bool
    assert model.linked.nodes[model.resolve(0, Parent.value)].guard is not None
    assert ref(Child.enabled).semantics is None
    assert Child.admitted.semantics is None
    assert Parent.copied.semantics is None

    class Wrong(Space):
        child = Child(size=1)

        @derived(flag=child.enabled)
        def value(*, flag: str) -> int:
            raise AssertionError("must not execute")

    with pytest.raises(DefinitionError, match="cannot consume bool"):
        compile_model(Wrong)


def test_collection_does_not_descend_into_deep_child_hierarchies() -> None:
    current: type[Space] = Space
    for index in range(1500):
        current = type(f"Nested{index}", (Space,), {"child": current()})
    effective = collect_space(current)
    assert tuple(effective.members) == ("child",)


def test_linker_argument_validation_checks_required_input_types() -> None:
    source = cast(ValueRef[object], Param(default=0))
    integer = cast(ValueSemantics[object], ValueSemantics.immutable_nominal(int))
    validate_argument(BoundArgument("item", source, int), integer, owner="reader.item")
    with pytest.raises(DefinitionError, match="cannot consume int"):
        validate_argument(BoundArgument("item", source, str), integer, owner="reader.item")


def test_generic_list_annotations_validate_their_nominal_origin() -> None:
    class Vector(Space):
        values: list[int] = Param()

        @derived
        def total(*, values: list[int]) -> int:
            return sum(values)

    effective = collect_space(Vector)
    assert effective.functions["total"].dependencies[0].source is decl(Vector.values)
