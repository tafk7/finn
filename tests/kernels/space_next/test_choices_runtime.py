# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped and selected evaluation through the supported candidate language."""

from __future__ import annotations

from types import MappingProxyType
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
    View,
    ViewKey,
    compile_space,
    constraint,
    derived,
    divisors_of,
    view,
)
from finn.kernels.space._next.errors import EvaluationError, RefinementError, RequestError

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

    base = Pair.start({Pair.extent: 12})
    first = base.first.assign(Tile.lanes, 3)
    assert isinstance(first, Tile)
    assert first.physical().accepted_answer == Decided(4)
    successor = cast(Pair, first.root)
    assert successor.first.lanes == 3
    assert isinstance(successor.second.answer(Tile.lanes), Unresolved)
    assert isinstance(base.first.answer(Tile.lanes), Unresolved)
    second = successor.second.assign(Tile.lanes, 4)
    final = cast(Pair, second.root)
    assert final.first.physical().accepted_answer == Decided(4)
    assert final.second.physical().accepted_answer == Decided(3)
    assert isinstance(successor.second.answer(Tile.lanes), Unresolved)
    assert final.answer(Pair.first.ref(Tile.extent)) == Decided(12)


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

    point = Outer.start()
    assert isinstance(point.child.answer(Guarded.lanes), Inapplicable)
    assert isinstance(point.child.decision_state(Guarded.lanes), Inapplicable)
    assert isinstance(point.child.answer(Guarded.raw), Inapplicable)
    assert isinstance(point.child.physical().accepted_answer, Inapplicable)
    assert calls == []
    with pytest.raises(RefinementError):
        point.child.assign(Guarded.lanes, 1)
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

    base = Root.start()
    assert isinstance(base.answer(Root.accepted), Unresolved)
    selected = base.implementation.select("refused")
    point = cast(Root, selected.occurrence.root)
    child = selected.alternative("refused")
    direct = child.assess(Refused.physical).accepted_answer
    assert isinstance(direct, Rejected)
    assert point.answer(Root.accepted) == direct
    assert point.physical().accepted_answer == direct
    assert selected.select("refused") is selected
    with pytest.raises(RefinementError):
        selected.select("explodes")
    assert isinstance(
        selected.alternative("explodes").assess(Explodes.physical).accepted_answer, Inapplicable
    )
    assert isinstance(base.answer(Root.accepted), Unresolved)
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

    active = Root.start({Root.enabled: True})
    assert active.answer(Root.accepted) == Decided(7)
    selected = active.implementation
    assert selected.alternatives == ("only",)
    assert selected.select("only") is selected
    inactive = Root.start({Root.enabled: False})
    assert isinstance(inactive.answer(Root.accepted), Inapplicable)
    with pytest.raises(RefinementError):
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

    base = Root.start()
    selected = base.first.implementation.select("a")
    assert isinstance(selected.occurrence, Family)
    assert selected.alternative("a").answer(A.value) == Decided(1)
    successor = cast(Root, selected.occurrence.root)
    assert isinstance(successor.second.implementation.alternative("a").answer(A.value), Unresolved)
    assert isinstance(base.first.implementation.alternative("a").answer(A.value), Unresolved)


def test_choice_metadata_and_handles_retain_the_compiled_definition() -> None:
    class A(Space):
        value = Const(1)

    class B(Space):
        value = Const(2)

    class Root(Space):
        implementation = SubspaceChoice({"a": Subspace(A), "b": Subspace(B)})

    model = compile_space(Root)
    base = model.start()
    saved = base.implementation
    Root.implementation.alternatives = MappingProxyType({"renamed": Subspace(A)})
    assert saved.alternatives == ("a", "b")
    chosen = saved.select("b")
    assert chosen.alternative("b").answer(B.value) == Decided(2)
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
        Broken.start().physical()
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
    base = model.start({Root.exposed.ref(Child.width): 9})
    assert base.exposed.physical().accepted_answer == Decided(9)
    chosen = base.assign(Root.supplier, 4)
    assert chosen.aliased.width == 4
    with pytest.raises(RequestError):
        chosen.aliased.assign(cast(Decision[int], Child.width), 2)
    local = chosen.assign(Root.local.decision_ref(Child.width), 6)
    assert local.local.width == 6
    assert local.aliased.width == 4
    assert isinstance(chosen.local.answer(Child.width), Unresolved)
