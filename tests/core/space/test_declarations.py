# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""DS1 structure and signature checks; no evaluator is mocked here."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast, get_type_hints
from typing_extensions import Self

import pytest

from finn.core.space import (
    QueryResult,
    Const,
    Decision,
    DefinitionError,
    MissingInput,
    NotApplicable,
    Param,
    Space,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    ValueSemantics,
    View,
    ViewKey,
    constraint,
    derived,
    full_result,
    optional,
    view,
)
from finn.core.space.collection import (
    BoundArgument,
    collect_placement,
    collect_space,
    resolve_decision_ref,
    validate_argument,
)
from finn.core.space.declarations import Declaration, DecisionRef


class Base(Space):
    extent = Param(int)

    @derived
    def doubled(*, extent: int) -> int:
        raise AssertionError("collection must not execute author callbacks")

    @derived(source=extent)
    def aliased(*, source: int) -> int:
        raise AssertionError("collection must not execute author callbacks")


class Override(Base):
    extent = Param(int)


def test_effective_collection_resolves_inherited_names_and_explicit_aliases() -> None:
    collected = collect_space(Override)
    assert collected.members["extent"].declaration is Override.extent
    assert collected.functions["doubled"].dependencies[0].source is Override.extent
    assert collected.functions["aliased"].dependencies[0].source is Override.extent
    assert collected.aliases[Base.extent] == "extent"
    assert collected.functions["doubled"].semantics.type_token is int
    assert Base.doubled.semantics is None


def test_forward_members_and_postponed_class_annotations() -> None:
    class Example(Space):
        class Quantity:
            pass

        @derived
        def result(*, later: Quantity) -> Quantity:
            raise AssertionError("must not run")

        later = Param(Quantity)

    collected = collect_space(Example)
    assert collected.functions["result"].dependencies[0].source is Example.later
    assert collected.functions["result"].semantics.type_token is Example.Quantity


def test_value_and_function_views_are_distinct_declarations_with_same_type() -> None:
    class Example(Space):
        value = Const(5)

        @constraint
        def supported(*, value: int) -> bool:
            raise AssertionError("must not run")

        @view(constraints=(supported,))
        def function(*, value: int) -> int:
            raise AssertionError("must not run")

        detached = View(value, constraints=(supported,))

    collected = collect_space(Example)
    assert collected.functions["function"].semantics.type_token is int
    assert collected.semantics[Example.detached].type_token is int
    assert Example.function.source is None
    assert Example.detached.source is Example.value
    assert Example.function.requires == Example.detached.requires == ()


def test_nested_alias_and_typed_exports() -> None:
    width = ValueKey("width", int)
    physical = ViewKey("physical", int)

    class Child(Space):
        extent = Param(int)

        @view
        def result(*, extent: int) -> int:
            return extent

        exports = {width: extent, physical: result}

    class Parent(Space):
        child = Subspace(Child, extent=Param(int))

        @derived(size=child.ref(Child.extent))
        def total(*, size: int) -> int:
            return size * 2

    collected = collect_space(Child)
    assert collected.exports[width] is Child.extent
    assert collected.exports[physical] is Child.result
    assert collect_space(Parent).functions["total"].dependencies[0].source.semantics is not None
    assert set(vars(Parent)) >= {"child", "total"}
    assert "extent" not in vars(Parent)


def test_explicit_answer_semantics_and_dependency_modes() -> None:
    class Example(Space):
        value = Param(int, required=False)

        @derived(input=optional(value))
        def missing(*, input: int | MissingInput | NotApplicable) -> bool:
            return isinstance(input, MissingInput)

        @derived(input=full_result(value), semantics=ValueSemantics.immutable_nominal(int))
        def forwarded(*, input: QueryResult[int]) -> QueryResult[int]:
            return input

    collected = collect_space(Example)
    assert collected.functions["missing"].dependencies[0].mode == "optional"
    assert collected.functions["forwarded"].dependencies[0].mode == "result"
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
    example = type("BadSignature", (Space,), {"value": Param(int), "result": derived(function)})
    with pytest.raises(DefinitionError, match=message):
        collect_space(example)


def test_self_receivers_are_collected_without_executing_or_inventing_dependencies() -> None:
    class Parent(Space):
        value = Param(int)

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
        value = Param(int)

        @derived(value=value)
        def result(self) -> int:
            raise AssertionError("ambiguous declarations must not be evaluated")

    with pytest.raises(DefinitionError, match="self.*alias|alias.*self"):
        collect_space(Ambiguous)


def test_extra_alias_and_answer_without_semantics_are_rejected() -> None:
    class Extra(Space):
        value = Param(int)

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
    different = type("Different", (Base,), {"extent": Param(str)})
    with pytest.raises(DefinitionError, match="Different.extent.*semantics"):
        collect_space(different)
    changed_kind = type("ChangedKind", (Base,), {"extent": Const(2)})
    with pytest.raises(DefinitionError, match="declaration kind"):
        collect_space(changed_kind)
    hidden = type("Hidden", (Base,), {"extent": 3})
    with pytest.raises(DefinitionError, match="hides an inherited declaration"):
        collect_space(hidden)

    class Other(Space):
        extent = Param(int)

    collision = type("Collision", (Base, Other), {})
    with pytest.raises(DefinitionError, match="conflicting inherited"):
        collect_space(collision)
    reserved = type("Reserved", (Space,), {"query": Param(int)})
    with pytest.raises(DefinitionError, match="reserved configuration name"):
        collect_space(reserved)


def test_duplicate_declaration_reuse_is_attributable() -> None:
    value = Param(int)
    with pytest.raises(RuntimeError) as exc:
        type("Duplicate", (Space,), {"one": value, "two": value})
    assert isinstance(exc.value.__cause__, DefinitionError)
    assert "Duplicate.two" in str(exc.value.__cause__)


def test_unbound_late_declarations_and_bad_exports_are_rejected() -> None:
    class Example(Space):
        pass

    setattr(Example, "late", Param(int))
    with pytest.raises(DefinitionError, match="not bound at class creation"):
        collect_space(Example)
    wrong = type(
        "WrongExport",
        (Space,),
        {"value": Param(int)},
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
        Decision(int)
    with pytest.raises(TypeError, match="truth value"):
        bool(Param(int))


def test_inferred_derived_override_cannot_change_value_type() -> None:
    @derived
    def text(*, extent: int) -> str:
        return str(extent)

    changed = type("Changed", (Base,), {"doubled": text})
    with pytest.raises(DefinitionError, match="override changes value semantics"):
        collect_space(changed)


def test_four_binding_forms_and_local_edit_ownership() -> None:
    class Child(Space):
        size = Param(int)
        internal = Decision(str, values=("auto", "block"))

    class Parent(Space):
        supplied = Decision(int, values=(1, 2))
        literal = Subspace(Child, size=4)
        alias = Subspace(Child, size=supplied)
        exposed = Subspace(Child, size=Param(int))
        local = Subspace(Child, size=Decision(int, values=(4, 8)))

    plans = [
        collect_placement(placement)
        for placement in (Parent.literal, Parent.alias, Parent.exposed, Parent.local)
    ]
    assert [plan.bindings["size"].kind for plan in plans] == [
        "literal",
        "reference",
        "exposed-param",
        "local-decision",
    ]
    local = cast(DecisionRef[object], Parent.local.decision_ref(Child.size))
    assert resolve_decision_ref(local) is Parent.local.bindings["size"]
    internal = cast(DecisionRef[object], Parent.alias.decision_ref(Child.internal))
    assert resolve_decision_ref(internal) is Child.internal
    for placement in (Parent.literal, Parent.alias, Parent.exposed):
        with pytest.raises(DefinitionError, match="not a locally owned Decision"):
            resolve_decision_ref(cast(DecisionRef[object], placement.decision_ref(Child.size)))


def test_child_omissions_and_incompatible_bindings_are_not_implicit_exposure() -> None:
    class Child(Space):
        width = Param(int)
        optional_width = Param(int, required=False)

    with pytest.raises(DefinitionError, match="missing child parameter"):
        collect_placement(Subspace(Child, width=8))
    with pytest.raises(DefinitionError, match="unknown child parameter"):
        collect_placement(Subspace(Child, width=8, optional_width=9, typo=1))
    with pytest.raises(DefinitionError, match="incompatible value semantics"):
        collect_placement(Subspace(Child, width=Param(str), optional_width=9))


def test_guards_are_collected_separately_and_respect_inherited_overrides() -> None:
    class Child(Space):
        value = Param(int)

    class Guarded(Space):
        enabled = Param(bool)
        selected = Decision(int, values=(1, 2), when=enabled)

        @derived(when=enabled)
        def result(*, selected: int) -> int:
            raise AssertionError("must not execute")

        @constraint(when=enabled)
        def support(*, selected: int) -> bool:
            raise AssertionError("must not execute")

        @view(when=enabled)
        def authored(*, selected: int) -> int:
            raise AssertionError("must not execute")

        physical = View(result, constraints=(support,), when=enabled)
        child = Subspace(Child, value=1, when=enabled)
        implementation = SubspaceChoice({"only": Subspace(Child, value=1)}, when=enabled)

    class OverrideGuard(Guarded):
        enabled = Param(bool)

    effective = collect_space(OverrideGuard)
    guarded = (
        Guarded.selected,
        Guarded.result,
        Guarded.support,
        Guarded.authored,
        Guarded.physical,
        Guarded.child,
        Guarded.implementation,
    )
    assert all(effective.guards[declaration] is OverrideGuard.enabled for declaration in guarded)
    assert [arg.name for arg in effective.functions["result"].dependencies] == ["selected"]
    assert "when" not in Guarded.result.aliases
    assert "when" not in Guarded.child.bindings


def test_guards_on_fresh_local_choices_and_alternatives_use_the_placement_scope() -> None:
    class Child(Space):
        value = Param(int)

    class Parent(Space):
        enabled = Param(bool)
        child = Subspace(Child, value=Decision(int, values=(1,), when=enabled))
        choice = SubspaceChoice(
            {
                "one": Subspace(Child, value=1, when=enabled),
            }
        )

    effective = collect_space(Parent)
    decision = cast(Decision[object], Parent.child.bindings["value"])
    assert effective.guards[decision] is Parent.enabled
    assert effective.guards[Parent.choice.alternatives["one"]] is Parent.enabled


def test_nonboolean_and_foreign_guards_fail_during_collection() -> None:
    class Wrong(Space):
        count = Param(int)
        choice = Decision(int, values=(1,), when=cast(ValueRef[bool], count))

    with pytest.raises(DefinitionError, match="Wrong.choice guard.*Boolean"):
        collect_space(Wrong)

    class Foreign(Space):
        enabled = Param(bool)

    class Unrelated(Space):
        choice = Decision(int, values=(1,), when=Foreign.enabled)

    with pytest.raises(DefinitionError, match="Unrelated.choice guard.*effective scope"):
        collect_space(Unrelated)


def test_when_is_an_authoring_control_argument() -> None:
    class Wrong(Space):
        active = Param(bool)

        @derived(when=active)
        def value(*, when: bool) -> int:
            return 1

    with pytest.raises(DefinitionError, match="reserved authoring control argument"):
        collect_space(Wrong)


def test_cached_child_records_supply_inferred_types_without_mutating_declarations() -> None:
    boolean_view = ViewKey("admitted", bool)

    class Child(Space):
        size = Param(int)

        @derived
        def enabled(*, size: int) -> bool:
            raise AssertionError("must not execute")

        admitted = View(enabled)
        exports = {boolean_view: admitted}

    child_record = collect_space(Child)

    class Parent(Space):
        child = Subspace(Child, size=1)

        @derived(when=child.ref(Child.enabled))
        def value() -> int:
            raise AssertionError("must not execute")

        copied = View(child.accepted(Child.admitted))
        exports = {boolean_view: copied}

    effective = collect_space(Parent, known_spaces={Child: child_record})
    assert effective.semantics[Parent.copied].type_token is bool
    assert effective.guards[Parent.value] is Parent.value.when
    assert Child.enabled.semantics is None
    assert Child.admitted.semantics is None
    assert Parent.copied.semantics is None

    class Wrong(Space):
        child = Subspace(Child, size=1)

        @derived(flag=child.ref(Child.enabled))
        def value(*, flag: str) -> int:
            raise AssertionError("must not execute")

    with pytest.raises(DefinitionError, match="cannot consume bool"):
        collect_space(Wrong, known_spaces={Child: child_record})


def test_collection_does_not_descend_into_deep_child_hierarchies() -> None:
    current: type[Space] = Space
    for index in range(1500):
        current = type(f"Nested{index}", (Space,), {"child": Subspace(current)})
    effective = collect_space(current)
    assert tuple(effective.members) == ("child",)


def test_linker_argument_validation_shares_dependency_mode_policy() -> None:
    def annotations(integer: QueryResult[int], text: QueryResult[str]) -> None:
        pass

    hints: dict[str, object] = get_type_hints(annotations)
    source = cast(ValueRef[object], Param(int))
    integer = cast(ValueSemantics[object], ValueSemantics.immutable_nominal(int))
    validate_argument(BoundArgument("item", source, "required", int), integer, owner="reader.item")
    validate_argument(
        BoundArgument("item", source, "result", hints["integer"]), integer, owner="reader.item"
    )
    validate_argument(
        BoundArgument("item", source, "optional", int | MissingInput | NotApplicable),
        integer,
        owner="reader.item",
    )
    with pytest.raises(DefinitionError, match="full_result dependency requires"):
        validate_argument(
            BoundArgument("item", source, "result", int), integer, owner="reader.item"
        )
    with pytest.raises(DefinitionError, match="cannot consume int"):
        validate_argument(
            BoundArgument("item", source, "result", hints["text"]), integer, owner="reader.item"
        )
    with pytest.raises(DefinitionError, match="cannot consume int"):
        validate_argument(
            BoundArgument("item", source, "optional", str | float | MissingInput | NotApplicable),
            integer,
            owner="reader.item",
        )


def test_generic_list_annotations_validate_their_nominal_origin() -> None:
    class Vector(Space):
        values: Param[list[int]] = Param(list)

        @derived
        def total(*, values: list[int]) -> int:
            return sum(values)

    effective = collect_space(Vector)
    assert effective.functions["total"].dependencies[0].source is Vector.values
