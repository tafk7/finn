# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped and selected evaluation through the public Space language."""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import (
    Available,
    BoundDecision,
    Const,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    View,
    ViewKey,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    view,
)
from finn.core.space.errors import (
    ConfigurationError,
    DefinitionError,
    EvaluationError,
    RequestError,
)

PHYSICAL = ViewKey("physical", int)


class Tile(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))

    @derived
    def cycles(*, extent: int, lanes: int) -> int:
        return extent // lanes

    physical = View(cycles)


def test_two_child_placements_have_independent_choices_and_immutable_roots() -> None:
    class Pair(Space):
        extent: int = Param()
        first = Tile(extent=extent)
        second = Tile(extent=extent)

    base = design_space(Pair(extent=12))
    first = base.first.with_choices(lanes=3)
    assert isinstance(first, Tile)
    assert first.physical == 4
    successor = cast(Pair, first.root)
    assert successor.first.lanes == 3
    assert isinstance(successor.second.query(Tile.lanes), Unresolved)
    assert isinstance(base.first.query(Tile.lanes), Unresolved)
    second = successor.second.with_choices(lanes=4)
    final = cast(Pair, second.root)
    assert final.first.physical == 4
    assert final.second.physical == 3
    assert isinstance(successor.second.query(Tile.lanes), Unresolved)
    assert final.query(Pair.first.extent) == Available(12)


def test_false_outer_scope_suppresses_inner_commitments_and_callbacks() -> None:
    calls: list[str] = []

    class Guarded(Space):
        inner: bool = Decision(values=(True, False))
        lanes: int = Decision(values=(1, 2), when=inner)

        @derived(when=inner)
        def raw() -> int:
            calls.append("inactive body")
            raise AssertionError("inactive body was evaluated")

        physical = View(raw, when=inner)

    class Outer(Space):
        enabled = Const(False)
        child = Guarded(when=enabled)

    point = design_space(Outer())
    assert isinstance(point.child.query(Guarded.lanes), Inapplicable)
    assert isinstance(point.child.field(Guarded.lanes).state, Inapplicable)
    assert isinstance(point.child.query(Guarded.raw), Inapplicable)
    assert isinstance(point.child.inspect(Guarded.physical).accepted_result, Inapplicable)
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

        physical = View(output, requires=(supported,))
        exports = {PHYSICAL: physical}

    class Explodes(Space):
        @view
        def physical() -> int:
            calls.append("unselected")
            raise AssertionError("unselected alternative was evaluated")

        exports = {PHYSICAL: physical}

    class Root(Space):
        # A Decision over nodes replaces SubspaceChoice; ``implementation.physical``
        # reads the selected candidate's member by name (was accepted(PHYSICAL)).
        implementation: Refused | Explodes = Decision(
            {"refused": Refused(), "explodes": Explodes()}
        )
        accepted = View(implementation.physical)
        physical = View(accepted)

    base = design_space(Root())
    assert isinstance(base.query(Root.accepted), Unresolved)
    point = base.with_choices(implementation="refused")
    child = point.implementation
    assert isinstance(child, Refused)
    direct = child.inspect(Refused.physical).accepted_result
    assert isinstance(direct, Rejected)
    assert point.query(Root.accepted) == direct
    assert point.inspect(Root.physical).accepted_result == direct
    assert point.with_choices(implementation="refused") is point
    point.with_choices(implementation="explodes")
    unselected = inspection.candidate(point, Root.implementation, "explodes")
    assert isinstance(unselected, Explodes)
    assert isinstance(unselected.inspect(Explodes.physical).accepted_result, Inapplicable)
    assert isinstance(base.query(Root.accepted), Unresolved)
    assert calls == []


def test_singleton_choice_is_forced_and_respects_its_outer_guard() -> None:
    # A singleton choice is an ordinary Decision whose one viable case reads as that
    # case (forced), never committed: its state stays unassigned. Committing it is
    # still a choice like any other.
    class Only(Space):
        output = Const(7)
        physical = View(output)
        exports = {PHYSICAL: physical}

    class Root(Space):
        enabled: bool = Param()
        implementation: Only = Decision({"only": Only()}, when=enabled)
        accepted = View(implementation.physical)

    active = design_space(Root(enabled=True))
    assert active.query(Root.accepted) == Available(7)
    state = active.field(Root.implementation).state
    assert isinstance(state, Available) and state.value.status == "unassigned"
    assert [(item.key, item.value) for item in inspection.forced(active)] == [
        ("implementation", "only")
    ]
    (choice,) = inspection.choices(active)
    assert [case.name for case in choice.cases] == ["only"]
    chosen = active.with_choices(implementation="only")
    assert chosen.query(Root.accepted) == Available(7)
    assert chosen.with_choices(implementation="only") is chosen
    inactive = design_space(Root(enabled=False))
    assert isinstance(inactive.query(Root.accepted), Inapplicable)
    with pytest.raises(ConfigurationError):
        inactive.with_choices(implementation="only")


def test_nested_choice_selection_retains_its_owning_scope() -> None:
    class A(Space):
        value = Const(1)

    class B(Space):
        value = Const(2)

    class Family(Space):
        implementation: A | B = Decision({"a": A(), "b": B()})

    class Root(Space):
        first = Family()
        second = Family()

    def alternative(point: Family, case: str) -> Space:
        candidate = inspection.candidate(point, Family.implementation, case)
        assert candidate is not None
        return candidate

    base = design_space(Root())
    selected = base.first.with_choices(implementation="a")
    assert isinstance(selected, Family)
    assert alternative(selected, "a").query(A.value) == Available(1)
    assert isinstance(selected.implementation, A)
    successor = cast(Root, selected.root)
    assert isinstance(alternative(successor.second, "a").query(A.value), Unresolved)
    assert isinstance(alternative(base.first, "a").query(A.value), Unresolved)


def test_choice_metadata_and_handles_retain_the_compiled_definition() -> None:
    class A(Space):
        value = Const(1)

    class B(Space):
        value = Const(2)

    class Root(Space):
        implementation: A | B = Decision({"a": A(), "b": B()})

    base = design_space(Root())
    (saved,) = inspection.choices(base)
    # The compiled choice cannot be retargeted: neither the Decision nor the family.
    with pytest.raises(AttributeError, match="immutable"):
        setattr(Root.implementation, "candidates", {"renamed": A()})
    with pytest.raises(DefinitionError, match="finalized"):
        Root.implementation = Decision({"renamed": A()})
    assert [case.name for case in saved.cases] == ["a", "b"]
    chosen = base.with_choices({saved.selector: "b"})
    b = inspection.candidate(chosen, Root.implementation, "b")
    assert b is not None and b.query(B.value) == Available(2)
    with pytest.raises(ConfigurationError):
        base.with_choices(implementation="renamed")
    with pytest.raises(RequestError, match="unknown candidate"):
        inspection.candidate(base, Root.implementation, "missing")


def test_function_view_failure_names_its_authored_owner() -> None:
    class Broken(Space):
        @view
        def physical() -> int:
            raise ZeroDivisionError("broken calculation")

    with pytest.raises(EvaluationError) as raised:
        design_space(Broken()).physical
    assert raised.value.owner == "physical"
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


def test_exposed_inputs_local_decisions_and_supplier_aliases_keep_distinct_rights() -> None:
    class Child(Space):
        width: int = Param()
        physical = View(width)

    class Root(Space):
        supplier: int = Decision(values=(2, 4))
        # An inline exposed Param is gone: the formal is declared here and bound by name.
        exposed_width: int = Param()
        aliased = Child(width=supplier)
        local = Child(width=Decision(values=(3, 6)))
        exposed = Child(width=exposed_width)

    base = design_space(Root(exposed_width=9))
    assert base.exposed.physical == 9
    chosen = base.with_choices(supplier=4)
    assert chosen.aliased.width == 4
    with pytest.raises(RequestError):
        inspection.decision_handle(chosen.aliased, Child.width)
    with pytest.raises(RequestError, match="not independently editable"):
        chosen.with_choices({Root.aliased.width: 2})
    local = chosen.with_choices({Root.local.width: 6})
    assert local.local.width == 6
    assert local.aliased.width == 4
    assert isinstance(chosen.local.query(Child.width), Unresolved)
    assert isinstance(chosen.local.field(Child.width), BoundDecision)


def test_wide_selected_outputs_keep_frozen_case_order_and_exact_targets() -> None:
    calls: list[int] = []

    class Leaf(Space):
        value: int = Param()

        @view
        def physical(*, value: int) -> int:
            calls.append(value)
            return value

        exports = {PHYSICAL: physical}

    class Root(Space):
        implementation: Leaf = Decision({f"case{index}": Leaf(value=index) for index in range(128)})
        physical = View(implementation.physical)

    model = inspection.model(Root)
    output = Root.physical
    (metadata,) = inspection.choices(model)
    assert tuple(case.name for case in metadata.cases) == tuple(
        f"case{index}" for index in range(128)
    )
    # The linked selection reads each case's view: one exact target per case.
    selection = Root.implementation.physical
    assert (
        len([node for node in inspection.dependencies(model, selection) if node.kind == "view"])
        == 128
    )
    assert calls == []
    with pytest.raises(AttributeError, match="immutable"):
        setattr(Root.implementation, "candidates", {"changed": Leaf(value=-1)})
    base = design_space(Root())
    for index in (0, 64, 127):
        point = base.with_choices({metadata.selector: f"case{index}"})
        assert point.query(output) == Available(index)
    assert calls == [0, 64, 127]
