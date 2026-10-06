# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from weakref import ReferenceType, ref

import pytest

from finn.core.space import (
    Available,
    Decision,
    Param,
    Space,
    Unresolved,
    ValueSemantics,
    View,
    default_semantics,
    derived,
    design_space,
    divisors_of,
    domain,
    require_value,
)
from finn.core.space.compiler import compile_model
from finn.core.space.errors import (
    ConfigurationError,
    DefinitionError,
    RequestError,
    ValueUnavailableError,
)
from finn.core.space.occurrence import state


def test_concurrent_configures_share_one_preparation() -> None:
    class Example(Space):
        value: int = Param()

    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda value: design_space(Example(value=value)), range(16)))
    model = compile_model(Example)
    explicit = design_space(Example(value=20))
    assert all(type(instance) is Example for instance in (*instances, explicit))
    assert all(state(instance).model is model for instance in (*instances, explicit))
    assert len({id(state(instance)) for instance in instances}) == len(instances)
    assert sorted(instance.value for instance in instances) == list(range(16))


def test_failed_preparation_does_not_poison_cache_and_subclasses_do_not_borrow_it() -> None:
    class Broken(Space):
        @derived
        def doubled(*, value: int) -> int:
            return value * 2

    with pytest.raises(DefinitionError, match="no declaration"):
        compile_model(Broken)
    value = cast(Param[int], Param(semantics=default_semantics(int)))
    value.__set_name__(Broken, "value")
    setattr(Broken, "value", value)
    # The formal is added after class creation, so it is not part of the static signature.
    assert design_space(Broken(value=3)).doubled == 6  # type: ignore[call-arg]

    class Child(Broken):
        extra: int = Param()

    parent_model = compile_model(Broken)
    child_model = compile_model(Child)
    assert child_model is not parent_model
    assert design_space(Child(value=3, extra=4)).extra == 4  # type: ignore[call-arg]


def test_prepared_structure_and_configuration_fields_are_immutable() -> None:
    class Example(Space):
        value: int = Param()
        choice: int = Decision(values=(1, 2))

    node = Example(value=3)
    instance = design_space(node)
    with pytest.raises(DefinitionError, match="finalized"):
        Example.choice = Decision(values=(3, 4))
    with pytest.raises(DefinitionError, match="finalized"):
        del Example.value
    with pytest.raises(AttributeError, match="immutable configuration field"):
        instance.value = 4
    with pytest.raises(AttributeError, match="immutable configuration field"):
        del instance.value
    # design_space() froze the root declaration: its formals can no longer be assigned.
    with pytest.raises(DefinitionError, match="is frozen"):
        node.value = 4
    setattr(instance, "note", "ordinary metadata")
    assert getattr(instance, "note") == "ordinary metadata"
    delattr(instance, "note")
    assert not hasattr(instance, "note")


def test_prepared_inherited_declarations_and_their_owners_are_immutable() -> None:
    class Base(Space):
        value: int = Param()

    class Example(Base):
        pass

    instance = design_space(Example(value=1))
    reference = Example.value
    with pytest.raises(DefinitionError, match="finalized"):
        Example.value = 99
    with pytest.raises(DefinitionError, match="finalized"):
        Base.value = 99
    assert instance.value == 1
    assert instance.query(reference) == Available(1)

    class Variant(Base):
        value: int = Param(required=False)

    assert design_space(Variant(value=2)).value == 2


def test_custom_instance_initialization_is_rejected_at_preparation() -> None:
    class Stateful(Space):
        value: int = Param()

        def __init__(self, value: int) -> None:
            self.saved = value

    with pytest.raises(DefinitionError, match="custom instance __init__"):
        compile_model(Stateful)


def test_nested_custom_instance_initialization_is_rejected() -> None:
    class Child(Space):
        value: int = Param()

        def __init__(self, **parameters: object) -> None:
            self.saved = parameters

    class Parent(Space):
        child = Child(value=1)

    with pytest.raises(DefinitionError, match="custom instance __init__"):
        compile_model(Parent)

    class ChoiceParent(Space):
        child: Child = Decision({"only": Child(value=1)})

    with pytest.raises(DefinitionError, match="custom instance __init__"):
        compile_model(ChoiceParent)


def test_structural_choices_cannot_be_shadowed_on_instances() -> None:
    class Example(Space):
        implementation: Space = Decision({"a": Space(), "b": Space()})
        singleton: Space = Decision({"only": Space()})

    instance = design_space(Example()).with_choices(implementation="a")
    with pytest.raises(AttributeError, match="immutable configuration field"):
        instance.implementation = "b"  # type: ignore[assignment]
    with pytest.raises(AttributeError, match="immutable configuration field"):
        instance.singleton = "only"  # type: ignore[assignment]


@pytest.mark.parametrize("method", ["with_choices", "try_with_choices"])
@pytest.mark.parametrize("name", ["self", "point"])
def test_choice_keywords_do_not_collide_with_receiver_arguments(method: str, name: str) -> None:
    class Example(Space):
        self: int = Decision(values=(1, 2))
        point: int = Decision(values=(1, 2))

    instance = design_space(Example())
    result = getattr(instance, method)(**{name: 1})
    revised = result if method == "with_choices" else result.instance
    assert getattr(revised, name) == 1


def test_call_keywords_do_not_steal_formal_names_and_request_errors_precede_snapshots() -> None:
    snapshots: list[int] = []

    def snapshot(value: int) -> int:
        snapshots.append(value)
        return value

    counted = ValueSemantics(
        int,
        "integer",
        lambda value: type(value) is int,
        lambda left, right: left == right,
        snapshot,
    )

    class Example(Space):
        parameters: int = Param(semantics=counted)
        choice: int = Decision(values=(1, 2), semantics=counted)

    point = design_space(Example(parameters=3))
    assert point.parameters == 3
    snapshots.clear()
    # A call on a Space class binds formals by keyword only; a positional mapping is
    # refused before any value is snapshotted.
    with pytest.raises(DefinitionError, match="by keyword"):
        Example({Example.parameters: 3}, parameters=4)  # type: ignore[arg-type, call-arg]
    assert snapshots == []
    # A choice requested twice is refused before any value is snapshotted.
    with pytest.raises(RequestError, match="duplicate change"):
        point.try_with_choices({Example.choice: 1}, choice=2)
    assert snapshots == []


def test_dynamic_prepared_space_class_and_bound_metadata_are_collectable() -> None:
    def create() -> tuple[ReferenceType[object], ReferenceType[object], ReferenceType[object]]:
        class Example(Space):
            value: int = Param()

        instance = design_space(Example(value=1))
        bound = instance.field(Example.value)
        compile_model(Example)
        return ref(Example), ref(instance), ref(bound)

    space_type_ref, instance_ref, bound_ref = create()
    gc.collect()
    assert space_type_ref() is None
    assert instance_ref() is None
    assert bound_ref() is None


def test_replacement_revalidates_retained_choices_and_supports_atomic_clear() -> None:
    class Example(Space):
        extent: int = Decision(values=(8, 12))
        lanes: int = Decision(domain=divisors_of(extent))

    base = design_space(Example())
    configured = base.with_choices(extent=12, lanes=3)
    refused = configured.try_with_choices(extent=8)
    assert not refused.accepted and refused.instance is configured
    assert [(item.owner, item.requested) for item in refused.outcomes] == [
        ("extent", True),
        ("lanes", False),
    ]
    with pytest.raises(ConfigurationError) as caught:
        configured.with_choices(extent=8)
    assert caught.value.report == refused
    revised = configured.with_choices(extent=8, lanes=4)
    assert revised.extent == 8 and revised.lanes == 4
    assert configured.extent == 12 and configured.lanes == 3
    cleared = revised.with_choices(
        revised.field(Example.extent).clear(), revised.field(Example.lanes).clear()
    )
    assert isinstance(cleared.query(Example.extent), Unresolved)
    assert isinstance(cleared.query(Example.lanes), Unresolved)


def test_require_value_preserves_result_and_view_context() -> None:
    class Example(Space):
        choice: int = Decision(values=(0, 1))
        result = View(choice)

    base = design_space(Example())
    unresolved = base.query(Example.choice)
    with pytest.raises(ValueUnavailableError) as caught:
        require_value(unresolved)
    assert caught.value.result is unresolved
    with pytest.raises(ValueUnavailableError) as call_error:
        _ = base.result
    assert call_error.value.result == base.query(Example.result)
    with pytest.raises(ValueUnavailableError):
        base.field(Example.choice).get()
    assert base.field(Example.choice).query() == unresolved
    assessment = base.inspect(Example.result)
    with pytest.raises(ValueUnavailableError) as view_error:
        assessment.require_value()
    assert view_error.value.context is assessment
    configured = base.with_choices(choice=0)
    assert require_value(configured.query(Example.choice)) == 0
    assert configured.result == 0
    assert configured.inspect(Example.result).require_value() == 0
    assert configured.field(Example.choice).get() == 0
    assert configured.field(Example.choice).query() == Available(0)


def test_child_replacement_returns_child_and_revalidates_the_whole_root() -> None:
    class Child(Space):
        local: int = Decision(values=(1, 2))

    class Root(Space):
        extent: int = Decision(values=(2, 4))
        sibling: int = Decision(domain=divisors_of(extent))
        child = Child()

    root = design_space(Root()).with_choices(extent=4, sibling=4)
    child = root.child.with_choices(local=1)
    refused = child.try_with_choices(child.root.field(Root.extent).change(2))
    assert not refused.accepted and refused.instance is child
    revised = child.with_choices(
        child.root.field(Root.extent).change(2), child.root.field(Root.sibling).change(2)
    )
    assert type(revised) is Child
    revised_root = revised.root
    assert isinstance(revised_root, Root)
    assert revised_root.extent == 2 and revised_root.sibling == 2
    assert revised.local == 1


def test_replacement_reuses_frozen_facts_keeps_views_lazy_and_starts_a_fresh_cache() -> None:
    calls: list[int] = []

    class Example(Space):
        source: list[object] = Param()
        choice: int = Decision(values=(1, 2))

        @derived
        def output(*, source: list[object], choice: int) -> int:
            calls.append(choice)
            return len(source) + choice

        result = View(output)

    source: list[object] = [1]
    first = design_space(Example(source=source)).with_choices(choice=1)
    source.append(2)
    assert calls == []
    assert first.result == 2
    revised = first.with_choices(choice=2)
    assert calls == [1]
    assert state(revised).parameters is state(first).parameters
    assert state(revised).cache == {}
    assert revised.result == 3
    assert calls == [1, 2]


def test_all_replacement_request_errors_precede_domain_callbacks() -> None:
    calls: list[int] = []

    def accepts(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Example(Space):
        first: int = Decision(values=(1, 2))
        second: int = Decision(domain=domain(accepts=accepts))

    base = design_space(Example())
    valid = base.field(Example.second).change(1)
    with pytest.raises(RequestError):
        base.try_with_choices(valid, object())  # type: ignore[arg-type]
    assert calls == []
