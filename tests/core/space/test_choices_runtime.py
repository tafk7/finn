# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped and selected evaluation through the public Space language."""

from __future__ import annotations

from types import MappingProxyType
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
    View,
    ViewKey,
    compile_space,
    constraint,
    derived,
    divisors_of,
    refinement,
    view,
)
from finn.core.space.errors import EvaluationError, ConfigurationError, RequestError
from finn.core.space import inspection

PHYSICAL = ViewKey("physical", int)


class Tile(Space):
    extent = Param(int)
    lanes = Decision(int, domain=divisors_of(extent))

    @derived
    def cycles(*, extent: int, lanes: int) -> int:
        return extent // lanes

    physical = View(cycles)


def test_two_child_placements_have_independent_choices_and_immutable_roots() -> None:
    class Pair(Space):
        extent = Param(int)
        first = Subspace(Tile, extent=extent)
        second = Subspace(Tile, extent=extent)

    base = Pair({Pair.extent: 12})
    first = base.first.with_choices(lanes=3)
    assert isinstance(first, Tile)
    assert first.physical() == 4
    successor = cast(Pair, first.root)
    assert successor.first.lanes == 3
    assert isinstance(successor.second.query(Tile.lanes), Unresolved)
    assert isinstance(base.first.query(Tile.lanes), Unresolved)
    second = successor.second.with_choices(lanes=4)
    final = cast(Pair, second.root)
    assert final.first.physical() == 4
    assert final.second.physical() == 3
    assert isinstance(successor.second.query(Tile.lanes), Unresolved)
    assert final.query(Pair.first.ref(Tile.extent)) == Available(12)


def test_false_outer_scope_suppresses_inner_commitments_and_callbacks() -> None:
    calls: list[str] = []

    class Guarded(Space):
        inner = Decision(bool, values=(True, False))
        lanes = Decision(int, values=(1, 2), when=inner)

        @derived(when=inner)
        def raw() -> int:
            calls.append("inactive body")
            raise AssertionError("inactive body was evaluated")

        physical = View(raw, when=inner)

    class Outer(Space):
        enabled = Const(False)
        child = Subspace(Guarded, when=enabled)

    point = Outer()
    assert isinstance(point.child.query(Guarded.lanes), Inapplicable)
    assert isinstance(point.child.field(Guarded.lanes).state, Inapplicable)
    assert isinstance(point.child.query(Guarded.raw), Inapplicable)
    assert isinstance(point.child.physical.inspect().accepted_result, Inapplicable)
    assert calls == []
    with pytest.raises(ConfigurationError):
        point.child.with_choices(lanes=1)
    assert calls == []


def test_selected_view_preserves_direct_refusal_and_skips_other_alternatives() -> None:
    calls: list[str] = []

    class Refused(Space):
        output = Const(4)

        @constraint
        def supported() -> bool:
            return False

        physical = View(output, constraints=(supported,))
        exports = {PHYSICAL: physical}

    class Explodes(Space):
        @view
        def physical() -> int:
            calls.append("unselected")
            raise AssertionError("unselected alternative was evaluated")

        exports = {PHYSICAL: physical}

    class Root(Space):
        implementation = SubspaceChoice(
            {"refused": Subspace(Refused), "explodes": Subspace(Explodes)},
            exports=(PHYSICAL,),
        )
        accepted = implementation.accepted(PHYSICAL)
        physical = View(accepted)

    base = Root()
    assert isinstance(base.query(Root.accepted), Unresolved)
    selected = base.implementation.select("refused")
    point = cast(Root, selected.instance.root)
    child = selected.alternative("refused")
    direct = child.inspect(Refused.physical).accepted_result
    assert isinstance(direct, Rejected)
    assert point.query(Root.accepted) == direct
    assert point.physical.inspect().accepted_result == direct
    assert selected.select("refused") is selected
    selected.select("explodes")
    assert isinstance(
        selected.alternative("explodes").inspect(Explodes.physical).accepted_result, Inapplicable
    )
    assert isinstance(base.query(Root.accepted), Unresolved)
    assert calls == []


def test_singleton_choice_needs_no_commitment_and_respects_its_outer_guard() -> None:
    class Only(Space):
        output = Const(7)
        physical = View(output)
        exports = {PHYSICAL: physical}

    class Root(Space):
        enabled = Param(bool)
        implementation = SubspaceChoice({"only": Subspace(Only)}, exports=(PHYSICAL,), when=enabled)
        accepted = implementation.accepted(PHYSICAL)

    active = Root({Root.enabled: True})
    assert active.query(Root.accepted) == Available(7)
    selected = active.implementation
    assert selected.alternatives == ("only",)
    assert selected.select("only") is selected
    inactive = Root({Root.enabled: False})
    assert isinstance(inactive.query(Root.accepted), Inapplicable)
    with pytest.raises(ConfigurationError):
        inactive.implementation.select("only")


def test_nested_choice_selection_retains_its_owning_scope() -> None:
    class A(Space):
        value = Const(1)

    class B(Space):
        value = Const(2)

    class Family(Space):
        implementation = SubspaceChoice({"a": Subspace(A), "b": Subspace(B)})

    class Root(Space):
        first = Subspace(Family)
        second = Subspace(Family)

    base = Root()
    selected = base.first.implementation.select("a")
    assert isinstance(selected.instance, Family)
    assert selected.alternative("a").query(A.value) == Available(1)
    successor = cast(Root, selected.instance.root)
    assert isinstance(successor.second.implementation.alternative("a").query(A.value), Unresolved)
    assert isinstance(base.first.implementation.alternative("a").query(A.value), Unresolved)


def test_choice_metadata_and_handles_retain_the_compiled_definition() -> None:
    class A(Space):
        value = Const(1)

    class B(Space):
        value = Const(2)

    class Root(Space):
        implementation = SubspaceChoice({"a": Subspace(A), "b": Subspace(B)})

    model = compile_space(Root)
    base = model.bind()
    saved = base.implementation
    Root.implementation.alternatives = MappingProxyType({"renamed": Subspace(A)})
    assert saved.alternatives == ("a", "b")
    chosen = saved.select("b")
    assert chosen.alternative("b").query(B.value) == Available(2)
    with pytest.raises(RequestError, match="unknown choice case"):
        saved.select("renamed")
    with pytest.raises(RequestError, match="unknown choice case"):
        saved.alternative("missing")


def test_function_view_failure_names_its_authored_owner() -> None:
    class Broken(Space):
        @view
        def physical() -> int:
            raise ZeroDivisionError("broken calculation")

    with pytest.raises(EvaluationError) as raised:
        Broken().physical()
    assert raised.value.owner == "physical"
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


def test_exposed_inputs_local_decisions_and_supplier_aliases_keep_distinct_rights() -> None:
    class Child(Space):
        width = Param(int)
        physical = View(width)

    class Root(Space):
        supplier = Decision(int, values=(2, 4))
        aliased = Subspace(Child, width=supplier)
        local = Subspace(Child, width=Decision(int, values=(3, 6)))
        exposed = Subspace(Child, width=Param(int))

    model = compile_space(Root)
    base = model.bind({Root.exposed.ref(Child.width): 9})
    assert base.exposed.physical() == 9
    chosen = base.with_choices(supplier=4)
    assert chosen.aliased.width == 4
    with pytest.raises(RequestError):
        refinement.change(chosen.aliased, cast(Decision[int], Child.width), 2)
    local = chosen.with_choices(chosen.field(Root.local.decision_ref(Child.width)).change(6))
    assert local.local.width == 6
    assert local.aliased.width == 4
    assert isinstance(chosen.local.query(Child.width), Unresolved)


def test_wide_selected_outputs_keep_frozen_case_order_and_exact_targets() -> None:
    calls: list[int] = []

    class Leaf(Space):
        value = Param(int)

        @view
        def physical(*, value: int) -> int:
            calls.append(value)
            return value

        exports = {PHYSICAL: physical}

    class Root(Space):
        implementation = SubspaceChoice(
            {f"case{index}": Subspace(Leaf, value=index) for index in range(128)},
            exports=(PHYSICAL,),
        )

    model = compile_space(Root)
    output = Root.implementation.accepted(PHYSICAL)
    metadata = inspection.choices(model)[0]
    assert tuple(case.name for case in metadata.cases) == tuple(
        f"case{index}" for index in range(128)
    )
    assert (
        len([node for node in inspection.dependencies(model, output) if node.kind == "view"]) == 128
    )
    assert calls == []
    Root.implementation.alternatives = MappingProxyType({"changed": Subspace(Leaf, value=-1)})
    base = model.bind()
    assert metadata.selector is not None
    for index in (0, 64, 127):
        point = base.with_choices(base.field(metadata.selector).change(f"case{index}"))
        assert point.query(output) == Available(index)
    assert calls == [0, 64, 127]
