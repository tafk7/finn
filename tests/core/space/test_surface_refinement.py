# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from weakref import ReferenceType, ref

import pytest

from finn.core.space import (
    UNSUPPLIED,
    Available,
    Decision,
    Param,
    Space,
    Unresolved,
    ValueSemantics,
    View,
    configure,
    derived,
    divisors_of,
    domain,
    require_value,
)
from finn.core.space.compiler import compile_space
from finn.core.space.errors import (
    ConfigurationError,
    DefinitionError,
    RequestError,
    ValueUnavailableError,
)
from finn.core.space.occurrence import state


def test_concurrent_configures_share_one_preparation() -> None:
    class Family(Space):
        value: Param[int] = Param(int)

    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda value: configure(Family(value=value)), range(16)))
    model = compile_space(Family)
    explicit = configure(Family(value=20))
    assert all(type(instance) is Family for instance in (*instances, explicit))
    assert all(state(instance).model is model for instance in (*instances, explicit))
    assert len({id(state(instance)) for instance in instances}) == len(instances)
    assert sorted(instance.value for instance in instances) == list(range(16))


def test_failed_preparation_does_not_poison_cache_and_subclasses_do_not_borrow_it() -> None:
    class Broken(Space):
        @derived
        def doubled(*, value: int) -> int:
            return value * 2

    with pytest.raises(DefinitionError, match="no declaration"):
        compile_space(Broken)
    value = Param(int)
    value.__set_name__(Broken, "value")
    setattr(Broken, "value", value)
    # The formal is added after class creation, so it is not part of the static signature.
    assert configure(Broken(value=3)).doubled == 6  # type: ignore[call-arg]

    class Child(Broken):
        extra: Param[int] = Param(int)

    parent_model = compile_space(Broken)
    child_model = compile_space(Child)
    assert child_model is not parent_model
    assert configure(Child(value=3, extra=4)).extra == 4  # type: ignore[call-arg]


def test_prepared_structure_and_configuration_fields_are_immutable() -> None:
    class Family(Space):
        value: Param[int] = Param(int)
        choice = Decision(int, values=(1, 2))

    node = Family(value=3)
    instance = configure(node)
    with pytest.raises(DefinitionError, match="finalized"):
        Family.choice = Decision(int, values=(3, 4))
    with pytest.raises(DefinitionError, match="finalized"):
        del Family.value
    with pytest.raises(AttributeError, match="immutable configuration field"):
        instance.value = 4
    with pytest.raises(AttributeError, match="immutable configuration field"):
        del instance.value
    with pytest.raises(AttributeError, match="node declaration is immutable"):
        node.value = 4
    setattr(instance, "note", "ordinary metadata")
    assert getattr(instance, "note") == "ordinary metadata"
    delattr(instance, "note")
    assert not hasattr(instance, "note")


def test_prepared_inherited_declarations_and_their_owners_are_immutable() -> None:
    class Base(Space):
        value: Param[int] = Param(int)

    class Family(Base):
        pass

    instance = configure(Family(value=1))
    reference = Family.value
    with pytest.raises(DefinitionError, match="finalized"):
        Family.value = 99  # type: ignore[assignment]
    with pytest.raises(DefinitionError, match="finalized"):
        Base.value = 99  # type: ignore[assignment]
    assert instance.value == 1
    assert instance.query(reference) == Available(1)

    class Variant(Base):
        value: Param[int] = Param(int, default=UNSUPPLIED)

    assert configure(Variant(value=2)).value == 2


def test_custom_instance_initialization_is_rejected_at_preparation() -> None:
    class Stateful(Space):
        value: Param[int] = Param(int)

        def __init__(self, value: int) -> None:
            self.saved = value

    with pytest.raises(DefinitionError, match="custom instance __init__"):
        compile_space(Stateful)


def test_nested_custom_instance_initialization_is_rejected() -> None:
    class Child(Space):
        value: Param[int] = Param(int)

        def __init__(self, **parameters: object) -> None:
            self.saved = parameters

    class Parent(Space):
        child = Child(value=1)

    with pytest.raises(DefinitionError, match="custom instance __init__"):
        compile_space(Parent)

    class ChoiceParent(Space):
        child = Decision(values={"only": Child(value=1)})

    with pytest.raises(DefinitionError, match="custom instance __init__"):
        compile_space(ChoiceParent)


def test_structural_choices_cannot_be_shadowed_on_instances() -> None:
    class Family(Space):
        implementation = Decision(values={"a": Space(), "b": Space()})
        singleton = Decision(values={"only": Space()})

    instance = configure(Family()).with_choices(implementation="a")
    with pytest.raises(AttributeError, match="immutable configuration field"):
        instance.implementation = "b"  # type: ignore[assignment]
    with pytest.raises(AttributeError, match="immutable configuration field"):
        instance.singleton = "only"  # type: ignore[assignment]


@pytest.mark.parametrize("method", ["with_choices", "try_with_choices"])
@pytest.mark.parametrize("name", ["self", "point"])
def test_choice_keywords_do_not_collide_with_receiver_arguments(method: str, name: str) -> None:
    class Family(Space):
        self = Decision(int, values=(1, 2))
        point = Decision(int, values=(1, 2))

    instance = configure(Family())
    result = getattr(instance, method)(**{name: 1})
    revised = result if method == "with_choices" else result.instance
    assert getattr(revised, name) == 1


def test_family_keywords_do_not_steal_formal_names_and_request_errors_precede_snapshots() -> None:
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

    class Family(Space):
        parameters: Param[int] = Param(counted)
        choice = Decision(counted, values=(1, 2))

    point = configure(Family(parameters=3))
    assert point.parameters == 3
    snapshots.clear()
    # A family call binds formals by keyword only; the positional mapping of the
    # old constructor is refused before any value is snapshotted.
    with pytest.raises(DefinitionError, match="by keyword"):
        Family({Family.parameters: 3}, parameters=4)  # type: ignore[arg-type, call-arg]
    assert snapshots == []
    # A choice requested twice is refused before any value is snapshotted.
    with pytest.raises(RequestError, match="duplicate change"):
        point.try_with_choices({Family.choice: 1}, choice=2)
    assert snapshots == []


def test_dynamic_prepared_family_and_bound_metadata_are_collectable() -> None:
    def create() -> tuple[ReferenceType[object], ReferenceType[object], ReferenceType[object]]:
        class Family(Space):
            value: Param[int] = Param(int)

        instance = configure(Family(value=1))
        bound = instance.field(Family.value)
        compile_space(Family)
        return ref(Family), ref(instance), ref(bound)

    family_ref, instance_ref, bound_ref = create()
    gc.collect()
    assert family_ref() is None
    assert instance_ref() is None
    assert bound_ref() is None


def test_replacement_revalidates_retained_choices_and_supports_atomic_clear() -> None:
    class Family(Space):
        extent = Decision(int, values=(8, 12))
        lanes = Decision(int, domain=divisors_of(extent))

    base = configure(Family())
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
        revised.field(Family.extent).clear(), revised.field(Family.lanes).clear()
    )
    assert isinstance(cleared.query(Family.extent), Unresolved)
    assert isinstance(cleared.query(Family.lanes), Unresolved)


def test_require_value_preserves_result_and_view_context() -> None:
    class Family(Space):
        choice = Decision(int, values=(0, 1))
        result = View(choice)

    base = configure(Family())
    unresolved = base.query(Family.choice)
    with pytest.raises(ValueUnavailableError) as caught:
        require_value(unresolved)
    assert caught.value.result is unresolved
    with pytest.raises(ValueUnavailableError) as call_error:
        base.result()
    assert call_error.value.result == base.result.query()
    with pytest.raises(ValueUnavailableError):
        base.field(Family.choice).get()
    assert base.field(Family.choice).query() == unresolved
    assessment = base.result.inspect()
    with pytest.raises(ValueUnavailableError) as view_error:
        assessment.require_value()
    assert view_error.value.context is assessment
    configured = base.with_choices(choice=0)
    assert require_value(configured.query(Family.choice)) == 0
    assert configured.result() == 0
    assert configured.result.inspect().require_value() == 0
    assert configured.field(Family.choice).get() == 0
    assert configured.field(Family.choice).query() == Available(0)


def test_child_replacement_returns_child_and_revalidates_the_whole_root() -> None:
    class Child(Space):
        local = Decision(int, values=(1, 2))

    class Root(Space):
        extent = Decision(int, values=(2, 4))
        sibling = Decision(int, domain=divisors_of(extent))
        child = Child()

    root = configure(Root()).with_choices(extent=4, sibling=4)
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

    class Family(Space):
        source: Param[list[object]] = Param(list)
        choice = Decision(int, values=(1, 2))

        @derived
        def output(*, source: list[object], choice: int) -> int:
            calls.append(choice)
            return len(source) + choice

        result = View(output)

    source: list[object] = [1]
    first = configure(Family(source=source)).with_choices(choice=1)
    source.append(2)
    assert calls == []
    assert first.result() == 2
    revised = first.with_choices(choice=2)
    assert calls == [1]
    assert state(revised).parameters is state(first).parameters
    assert state(revised).cache == {}
    assert revised.result() == 3
    assert calls == [1, 2]


def test_all_replacement_request_errors_precede_domain_callbacks() -> None:
    calls: list[int] = []

    def accepts(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Family(Space):
        first = Decision(int, values=(1, 2))
        second = Decision(int, domain=domain(accepts=accepts))

    base = configure(Family())
    valid = base.field(Family.second).change(1)
    with pytest.raises(RequestError):
        base.try_with_choices(valid, object())  # type: ignore[arg-type]
    assert calls == []
