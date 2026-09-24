# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
from weakref import ReferenceType, ref

import pytest

from finn.kernels.space import (
    Decision,
    Param,
    Space,
    Subspace,
    Unresolved,
    ValueSemantics,
    View,
    compile_space,
    derived,
    divisors_of,
    domain,
    require_value,
)
from finn.kernels.space.errors import (
    ConfigurationError,
    DefinitionError,
    RequestError,
    ValueUnavailableError,
)
from finn.kernels.space.occurrence import state


def test_constructor_and_explicit_bind_share_one_concurrent_preparation() -> None:
    class Family(Space):
        value = Param(int)

    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda value: Family(value=value), range(16)))
    model = compile_space(Family)
    explicit = model.bind(value=20)
    assert all(type(instance) is Family for instance in (*instances, explicit))
    assert all(state(instance).model is model for instance in (*instances, explicit))
    assert len({id(state(instance).snapshot) for instance in instances}) == len(instances)


def test_failed_preparation_does_not_poison_cache_and_subclasses_do_not_borrow_it() -> None:
    class Broken(Space):
        @derived
        def doubled(*, value: int) -> int:
            return value * 2

    with pytest.raises(DefinitionError, match="no declaration"):
        compile_space(Broken)
    value = Param(int)
    value.__set_name__(Broken, "value")
    Broken.value = value
    assert Broken(value=3).doubled == 6

    class Child(Broken):
        extra = Param(int)

    parent_model = compile_space(Broken)
    child_model = compile_space(Child)
    assert child_model is not parent_model
    assert Child(value=3, extra=4).extra == 4


def test_prepared_structure_and_configuration_fields_are_immutable() -> None:
    class Family(Space):
        value = Param(int)
        choice = Decision(int, values=(1, 2))

    instance = Family(value=3)
    with pytest.raises(DefinitionError, match="finalized"):
        Family.choice = Decision(int, values=(3, 4))
    with pytest.raises(DefinitionError, match="finalized"):
        del Family.value
    with pytest.raises(AttributeError, match="immutable configuration field"):
        instance.value = 4
    setattr(instance, "note", "ordinary metadata")
    assert getattr(instance, "note") == "ordinary metadata"


def test_custom_instance_initialization_is_rejected_at_preparation() -> None:
    class Stateful(Space):
        value = Param(int)

        def __init__(self, value: int) -> None:
            self.saved = value

    with pytest.raises(DefinitionError, match="custom instance __init__"):
        compile_space(Stateful)


def test_constructor_keywords_do_not_steal_parameter_names_and_duplicates_precede_snapshots() -> (
    None
):
    snapshots: list[int] = []

    def snapshot(value: int) -> int:
        snapshots.append(value)
        return value

    class Family(Space):
        parameters: Param[int] = Param(
            ValueSemantics(
                int,
                "integer",
                lambda value: type(value) is int,
                lambda left, right: left == right,
                snapshot,
            )
        )

    assert Family(parameters=3).parameters == 3
    snapshots.clear()
    with pytest.raises(RequestError, match="more than once"):
        Family({Family.parameters: 3}, parameters=4)
    assert snapshots == []


def test_dynamic_prepared_family_and_bound_metadata_are_collectable() -> None:
    def create() -> tuple[ReferenceType[object], ReferenceType[object], ReferenceType[object]]:
        class Family(Space):
            value = Param(int)

        instance = Family(value=1)
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

    base = Family()
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

    base = Family()
    unresolved = base.query(Family.choice)
    with pytest.raises(ValueUnavailableError) as caught:
        require_value(unresolved)
    assert caught.value.result is unresolved
    assessment = base.result()
    with pytest.raises(ValueUnavailableError) as view_error:
        assessment.require_value()
    assert view_error.value.context is assessment
    configured = base.with_choices(choice=0)
    assert require_value(configured.query(Family.choice)) == 0
    assert configured.result().require_value() == 0


def test_child_replacement_returns_child_and_revalidates_the_whole_root() -> None:
    class Child(Space):
        local = Decision(int, values=(1, 2))

    class Root(Space):
        extent = Decision(int, values=(2, 4))
        sibling = Decision(int, domain=divisors_of(extent))
        child = Subspace(Child)

    root = Root().with_choices(extent=4, sibling=4)
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
        source = Param(list)
        choice = Decision(int, values=(1, 2))

        @derived
        def output(*, source: list[object], choice: int) -> int:
            calls.append(choice)
            return len(source) + choice

        result = View(output)

    source: list[object] = [1]
    first = Family(source=source).with_choices(choice=1)
    source.append(2)
    assert calls == []
    assert first.result().require_value() == 2
    revised = first.with_choices(choice=2)
    assert calls == [1]
    assert state(revised).snapshot.parameters is state(first).snapshot.parameters
    assert state(revised).snapshot.cache == {}
    assert revised.result().require_value() == 3
    assert calls == [1, 2]


def test_all_replacement_request_errors_precede_domain_callbacks() -> None:
    calls: list[int] = []

    def accepts(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Family(Space):
        first = Decision(int, values=(1, 2))
        second = Decision(int, domain=domain(accepts=accepts))

    base = Family()
    valid = base.field(Family.second).change(1)
    with pytest.raises(RequestError):
        base.try_with_choices(valid, object())  # type: ignore[arg-type]
    assert calls == []
