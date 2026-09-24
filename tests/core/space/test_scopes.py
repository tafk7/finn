# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped binding, frozen references, composition, and frontend depth limits."""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import (
    Const,
    Available,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Subspace,
    SubspaceChoice,
    Unresolved,
    ValueKey,
    View,
    ViewKey,
    compile_space,
    constraint,
    derived,
    divisors_of,
)
from finn.core.space.declarations import ScopedValueRef
from finn.core.space.errors import ConfigurationError, DefinitionError

PHYSICAL = ViewKey("physical", int)
WIDTH = ValueKey("width", int)


def test_composite_keeps_narrow_fields_available_and_reuses_accepted_children() -> None:
    class Interface(Space):
        width = Param(int)
        lanes = Decision(int, values=(1, 2))

        @derived
        def complete(*, width: int, lanes: int) -> int:
            return width * lanes

        @constraint
        def supported(*, width: int) -> bool:
            return width > 0

        physical = View(complete, constraints=(supported,))
        exports = {WIDTH: width, PHYSICAL: physical}

    class Refused(Space):
        width = Const(-1)

        @constraint
        def supported() -> bool:
            return False

        physical = View(width, constraints=(supported,))
        exports = {WIDTH: width, PHYSICAL: physical}

    class Composite(Space):
        enabled = Const(False)
        activation = Subspace(Interface, width=8)
        weights = Subspace(Interface, width=4)
        optional = Subspace(Interface, width=16, when=enabled)
        implementation = SubspaceChoice(
            {"normal": Subspace(Interface, width=32), "refused": Subspace(Refused)},
            exports=(WIDTH, PHYSICAL),
        )

        @derived(activation=activation.ref(Interface.width), weights=weights.ref(Interface.width))
        def narrow(*, activation: int, weights: int) -> int:
            return activation + weights

        physical = View(implementation.accepted(PHYSICAL))

    base = Composite()
    assert base.narrow == 12
    assert isinstance(base.activation.physical.inspect().accepted_result, Unresolved)
    assert isinstance(base.weights.physical.inspect().accepted_result, Unresolved)
    assert isinstance(base.optional.query(Interface.lanes), Inapplicable)
    assert isinstance(base.optional.field(Interface.lanes).state, Inapplicable)
    selected = base.implementation.select("refused")
    successor = cast(Composite, selected.instance.root)
    direct = selected.alternative("refused").inspect(Refused.physical).accepted_result
    assert isinstance(direct, Rejected)
    assert successor.physical.inspect().accepted_result == direct
    assert successor.query(Composite.implementation.ref(WIDTH)) == Available(-1)
    assert isinstance(base.physical.inspect().accepted_result, Unresolved)


def test_local_decision_domain_uses_parent_suppliers_and_exposure_is_explicit() -> None:
    class Child(Space):
        value = Param(int)
        physical = View(value)

    class Root(Space):
        extent = Param(int)
        owned = Subspace(Child, value=Decision(int, domain=divisors_of(extent)))
        exposed = Subspace(Child, value=Param(int, required=False))

    model = compile_space(Root)
    base = model.bind({Root.extent: 12})
    assert base.field(Root.owned.decision_ref(Child.value)).candidates() == Available(
        (1, 2, 3, 4, 6, 12)
    )
    chosen = base.with_choices(base.field(Root.owned.decision_ref(Child.value)).change(3))
    assert chosen.owned.physical() == 3
    assert isinstance(chosen.exposed.query(Child.value), Unresolved)
    with pytest.raises(ConfigurationError):
        base.with_choices(base.field(Root.owned.decision_ref(Child.value)).change(5))
    supplied = model.bind({Root.extent: 12, Root.exposed.ref(Child.value): 7})
    assert supplied.exposed.value == 7

    with pytest.raises(DefinitionError, match="missing child parameter"):
        compile_space(type("Missing", (Space,), {"child": Subspace(Child)}))


def test_nested_handles_and_named_aliases_keep_frozen_interpretations() -> None:
    class Leaf(Space):
        value = Const(5)
        physical = View(value)

    class Middle(Space):
        inner = Subspace(Leaf)

    class Root(Space):
        outer = Subspace(Middle)
        accepted = outer.ref(Middle.inner.accepted(Leaf.physical))

    original_placement = Middle.inner
    original_accepted = Root.accepted
    model = compile_space(Root)
    base = model.bind()
    assert base.query(Root.outer.ref(Middle.inner.ref(Leaf.value))) == Available(5)
    assert base.query(original_accepted) == Available(5)

    replacement = Subspace(Leaf)
    replacement.__set_name__(Middle, "inner")
    with pytest.raises(DefinitionError, match="finalized"):
        Middle.inner = replacement
    assert base.query(Root.outer.ref(original_placement.ref(Leaf.value))) == Available(5)
    assert base.query(Root.outer.ref(Middle.inner.ref(Leaf.value))) == Available(5)

    # A declared alias is interpreted by its frozen compiled entry. Mutating
    # the source wrapper later cannot retarget that old compiled reference.
    handle = cast(ScopedValueRef[int], original_accepted)
    handle.member = replacement.accepted(Leaf.physical)
    assert base.query(original_accepted) == Available(5)


def test_two_thousand_guarded_scopes_compile_and_query_iteratively() -> None:
    class Leaf(Space):
        value = Const(9)

    family: type[Space] = Leaf
    for depth in range(2_000):
        enabled = Const(True)
        family = cast(
            type[Space],
            type(
                f"Layer{depth}",
                (Space,),
                {"enabled": enabled, "inner": Subspace(family, when=enabled)},
            ),
        )
    model = compile_space(family)
    assert len(model.linked.scopes) == 2_001
    leaf = model.bind()
    for _ in range(2_000):
        leaf = cast(Space, getattr(leaf, "inner"))
    assert leaf.query(Leaf.value) == Available(9)


def test_recursive_structure_is_rejected_before_occurrence_allocation() -> None:
    class Recursive(Space):
        value = Const(1)

    repeated = Subspace(Recursive)
    repeated.__set_name__(Recursive, "again")
    setattr(Recursive, "again", repeated)
    with pytest.raises(DefinitionError, match="recursive Space placement"):
        compile_space(Recursive)


def test_choice_exports_validate_all_cases_before_selection() -> None:
    class Complete(Space):
        value = Const(1)
        physical = View(value)
        exports = {PHYSICAL: physical}

    class Missing(Space):
        value = Const(2)

    class Root(Space):
        implementation = SubspaceChoice(
            {"complete": Subspace(Complete), "missing": Subspace(Missing)},
            exports=(PHYSICAL,),
        )

    with pytest.raises(DefinitionError, match="missing choice export physical"):
        compile_space(Root)


def test_inferred_nested_output_must_match_consumer_annotation() -> None:
    class Leaf(Space):
        @derived
        def value() -> int:
            return 1

        physical = View(value)

    class Middle(Space):
        inner = Subspace(Leaf)

    class Root(Space):
        outer = Subspace(Middle)

        @derived(value=outer.ref(Middle.inner.accepted(Leaf.physical)))
        def wrong(*, value: str) -> str:
            return value

    with pytest.raises(DefinitionError, match="cannot consume"):
        compile_space(Root)


def test_false_choice_guard_does_not_demand_selector_or_case_condition() -> None:
    class Leaf(Space):
        value = Const(1)

    class Root(Space):
        disabled = Const(False)
        unknown = Decision(bool, values=(False, True))
        choice = SubspaceChoice(
            {"a": Subspace(Leaf, when=unknown), "b": Subspace(Leaf)},
            when=disabled,
        )

    point = Root()
    assert isinstance(point.choice.alternative("a").query(Leaf.value), Inapplicable)
    assert isinstance(point.choice.alternative("b").query(Leaf.value), Inapplicable)
