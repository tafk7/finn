# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped binding, frozen references, composition, and frontend depth limits."""

from __future__ import annotations

from typing import cast

import pytest

from finn.kernels.space._next import (
    Const,
    Decided,
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
from finn.kernels.space._next.declarations import ScopedValueRef
from finn.kernels.space._next.errors import DefinitionError, RefinementError, RequestError

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

    base = Composite.start()
    assert base.narrow == 12
    assert isinstance(base.activation.physical().accepted_answer, Unresolved)
    assert isinstance(base.weights.physical().accepted_answer, Unresolved)
    assert isinstance(base.optional.answer(Interface.lanes), Inapplicable)
    assert isinstance(base.optional.decision_state(Interface.lanes), Inapplicable)
    selected = base.implementation.select("refused")
    successor = cast(Composite, selected.occurrence.root)
    direct = selected.alternative("refused").assess(Refused.physical).accepted_answer
    assert isinstance(direct, Rejected)
    assert successor.physical().accepted_answer == direct
    assert successor.answer(Composite.implementation.ref(WIDTH)) == Decided(-1)
    assert isinstance(base.physical().accepted_answer, Unresolved)


def test_local_decision_domain_uses_parent_suppliers_and_exposure_is_explicit() -> None:
    class Child(Space):
        value = Param(int)
        physical = View(value)

    class Root(Space):
        extent = Param(int)
        owned = Subspace(Child, value=Decision(int, domain=divisors_of(extent)))
        exposed = Subspace(Child, value=Param(int, required=False))

    model = compile_space(Root)
    base = model.start({Root.extent: 12})
    assert base.candidates(Root.owned.decision_ref(Child.value)) == Decided((1, 2, 3, 4, 6, 12))
    chosen = base.assign(Root.owned.decision_ref(Child.value), 3)
    assert chosen.owned.physical().accepted_answer == Decided(3)
    assert isinstance(chosen.exposed.answer(Child.value), Unresolved)
    with pytest.raises(RefinementError):
        base.assign(Root.owned.decision_ref(Child.value), 5)
    supplied = model.start({Root.extent: 12, Root.exposed.ref(Child.value): 7})
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
    base = model.start()
    assert base.answer(Root.outer.ref(Middle.inner.ref(Leaf.value))) == Decided(5)
    assert base.answer(original_accepted) == Decided(5)

    replacement = Subspace(Leaf)
    replacement.__set_name__(Middle, "inner")
    Middle.inner = replacement
    assert base.answer(Root.outer.ref(original_placement.ref(Leaf.value))) == Decided(5)
    with pytest.raises(RequestError, match="placement"):
        base.answer(Root.outer.ref(Middle.inner.ref(Leaf.value)))

    # A declared alias is interpreted by its frozen compiled entry. Mutating
    # the source wrapper later cannot retarget that old compiled reference.
    handle = cast(ScopedValueRef[int], original_accepted)
    handle.member = replacement.accepted(Leaf.physical)
    assert base.answer(original_accepted) == Decided(5)


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
    leaf = model.start()
    for _ in range(2_000):
        leaf = cast(Space, getattr(leaf, "inner"))
    assert leaf.answer(Leaf.value) == Decided(9)


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

    point = Root.start()
    assert isinstance(point.choice.alternative("a").answer(Leaf.value), Inapplicable)
    assert isinstance(point.choice.alternative("b").answer(Leaf.value), Inapplicable)
