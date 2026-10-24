# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Nested formals are supplied where their node is declared; a Decision is its class's.

A reusable Space class exposes its child's formals as its own and binds them by
name (``port = Port(dtype=dtype, lanes=lanes)``); whoever places it supplies
them. A Decision bound into a child is declared on the class that binds it, which
owns the choice.
"""

from __future__ import annotations

import pytest
from support.refusals import refused

from core.space._choices_support import committed
from finn.core.space import (
    Available,
    BoundDecision,
    Decision,
    Inapplicable,
    Param,
    Space,
    Unresolved,
    View,
    composite,
    derived,
    design_space,
    divisors_of,
    inspection,
)
from finn.core.space.errors import ConfigurationError, DefinitionError, RequestError


class Port(Space):
    dtype: str = Param()
    lanes: int = Param()

    @derived
    def description(*, dtype: str, lanes: int) -> tuple[str, int]:
        return dtype, lanes

    physical = View(description)


class Reusable(Space):
    """Its port's formals are its own: whoever places a Reusable supplies them."""

    count: int = Param()
    dtype: str = Param()
    lanes: int = Param()
    port = Port(dtype=dtype, lanes=lanes)


def test_outer_params_and_decisions_supply_interface_slots_without_new_choices() -> None:
    class Parent(Space):
        dtype: str = Param()
        lanes: int = Decision(values=(1, 2, 4))
        kernel = Reusable(count=1, dtype=dtype, lanes=lanes)

    base = design_space(Parent(dtype="INT8"))
    assert base.kernel.port.dtype == "INT8"
    assert isinstance(base.kernel.port.query(Port.lanes), Unresolved)
    chosen = base.with_choices(lanes=2)
    assert chosen.kernel.port.physical == ("INT8", 2)
    assert [item.key for item in inspection.decisions(chosen)] == ["lanes"]
    assert committed(chosen) == {"lanes": 2}
    # A formal supplied by another member is not a choice of its own.
    with pytest.raises(RequestError, match="not independently editable"):
        chosen.with_choices({Parent.kernel.port.lanes: 1})
    evidence = inspection.explain(chosen.kernel.port, Port.description)
    assert any(
        node.declaration.key == "dtype" and node.result == Available("INT8")
        for node in evidence.nodes
    )
    assert any(
        node.declaration.key == "lanes" and node.decision_state is not None
        for node in evidence.nodes
    )


def test_a_decision_bound_into_a_child_reads_the_enclosing_class_for_its_domain_and_guard() -> None:
    class Parent(Space):
        extent: int = Param()
        enabled: bool = Param()
        # Declared on the enclosing class, which owns the choice; bound into the child.
        lanes: int = Decision(domain=divisors_of(extent), when=enabled)
        kernel = Port(dtype="INT4", lanes=lanes)

        @derived(lanes=kernel.lanes)
        def folded(*, lanes: int) -> int:
            return lanes * 2

    base = design_space(Parent(extent=12, enabled=True))
    field = base.field(Parent.lanes)
    assert isinstance(field, BoundDecision)
    assert field.candidates() == Available((1, 2, 3, 4, 6, 12))
    chosen = base.with_choices({Parent.lanes: 3})
    assert chosen.folded == 6
    assert chosen.kernel.lanes == 3
    assert [item.key for item in inspection.decisions(chosen)] == ["lanes"]
    with pytest.raises(ConfigurationError):
        base.with_choices({Parent.lanes: 5})
    inactive = design_space(Parent(extent=12, enabled=False))
    inactive_field = inactive.field(Parent.lanes)
    assert isinstance(inactive_field, BoundDecision)
    assert isinstance(inactive_field.state, Inapplicable)
    assert isinstance(inactive.kernel.query(Port.lanes), Inapplicable)


def test_a_decision_bound_into_two_nodes_is_one_decision_of_its_class() -> None:
    class Shared(Space):
        use_first: bool = Decision(values=(False, True))
        lanes: int = Decision(values=(1, 2, 4))
        first = Port(dtype="INT4", lanes=lanes, when=use_first)
        second = Port(dtype="INT8", lanes=lanes)

    (info,) = [item for item in inspection.decisions(Shared) if item.key != "use_first"]
    assert (info.key, info.scope) == ("lanes", "")
    base = design_space(Shared())
    point = base.with_choices(lanes=4, use_first=False)
    assert point.second.physical == ("INT8", 4)
    assert isinstance(point.first.query(Port.lanes), Inapplicable)
    assert point.query(info.reference) == Available(4)
    both = point.with_choices(lanes=2, use_first=True)
    assert both.first.physical == ("INT4", 2)
    assert both.second.physical == ("INT8", 2)
    # A formal a member supplies is no choice of its own.
    with pytest.raises(RequestError, match="not independently editable"):
        both.with_choices({Shared.second.lanes: 1})
    assert committed(both) == {"lanes": 2, "use_first": True}


def test_unbound_exposure_is_a_formal_declared_on_the_enclosing_space_class() -> None:
    # The formal is declared here and bound to the child by name.
    class Parent(Space):
        lanes: int = Param()
        kernel = Reusable(count=1, dtype="INT8", lanes=lanes)

    supplied = design_space(Parent(lanes=3))
    assert supplied.kernel.port.physical == ("INT8", 3)


def test_a_childs_formal_nothing_supplies_is_reported_and_exposed_by_name() -> None:
    class Middle(Space):
        kernel = Reusable(count=1)  # dtype and lanes left unsupplied

    with refused("formal-unsupplied", error=DefinitionError, message=r"kernel\.dtype"):
        design_space(Middle())

    # Exposing by name: the enclosing Space class declares the formal and binds it.
    class Named(Space):
        dtype: str = Param()
        kernel = Reusable(count=1, dtype=dtype, lanes=2)

    class Top(Space):
        named = Named(dtype="INT5")

    assert design_space(Top()).named.kernel.port.physical == ("INT5", 2)


def test_assignment_targets_are_checked() -> None:
    class Parent(Space):
        child = Reusable(count=2)
        child.dtype = "INT8"
        child.lanes = 1

    assert design_space(Parent()).child.count == 2

    with refused("formal-assigned-twice", error=DefinitionError):

        class DuplicateDirect(Space):
            child = Reusable(count=1)
            child.count = 2

    with refused("assignment-not-own-formal", error=DefinitionError):

        class Nested(Space):
            child = Reusable(count=1)
            child.port.dtype = "INT8"

    with refused("member-belongs-to-class", error=DefinitionError):

        class NonParameter(Space):
            port = Port(dtype="INT8", lanes=2)
            port.description = ("x", 2)

    with refused(
        "literal-unrecognized", error=DefinitionError, message="expected value of nominal type int"
    ):
        Reusable().count = "two"  # type: ignore[assignment]
    with refused("members-unknown", error=DefinitionError, message="unknown members"):
        Reusable(count=1, width=2)  # type: ignore[call-arg]


def test_assignment_after_freezing_is_refused() -> None:
    class Parent(Space):
        first: str = Param()
        second: str = Param()
        kernel = Reusable(count=1)
        kernel.dtype = first
        kernel.lanes = 1

    reference = Parent.kernel.port.dtype
    old = design_space(Parent(first="INT3", second="INT7"))
    # Preparing Parent froze its node declarations: the model cannot drift.
    with refused(
        "declaration-frozen", error=DefinitionError, message="is frozen .Parent was prepared"
    ) as caught:
        Parent.kernel.lanes = 2
    assert "assigned at test_nested_parameter_bindings.py:" in str(caught.error)
    point = design_space(Parent(first="INT3", second="INT7"))
    assert old.query(reference) == point.query(reference) == Available("INT3")


def test_configure_freezes_its_root() -> None:
    root = Reusable()
    root.count = 3
    root.dtype = "INT8"
    root.lanes = 2
    assert design_space(root).port.physical == ("INT8", 2)
    with refused("declaration-frozen", error=DefinitionError):
        root.count = 4


def test_a_graph_built_as_data_binds_formals_and_keeps_their_types() -> None:
    # Nodes are plain values, joined by assignment, named by composite.
    child = Reusable(count=1)
    child.dtype = "INT8"
    child.lanes = 2
    space_type = composite("Parent", {"child": child})
    point = design_space(space_type())
    placed = getattr(point, "child")
    assert isinstance(placed, Reusable)
    assert placed.port.physical == ("INT8", 2)
    with refused("formal-unsupplied", error=DefinitionError, message=r"child\.dtype"):
        design_space(composite("Unbound", {"child": Reusable(count=1)})())


def test_nested_decision_reference_can_traverse_concrete_reference_layers() -> None:
    class Leaf(Space):
        value: int = Decision(values=(1, 2))

    class Middle(Space):
        leaf = Leaf()

    class Outer(Space):
        middle = Middle()

    reference = Outer.middle.leaf.value
    base = design_space(Outer())
    point = base.with_choices({reference: 2})
    assert point.middle.leaf.value == 2
    through_root = inspection.decision_handle(point, reference)
    through_leaf = inspection.decision_handle(point.middle.leaf, Leaf.value)
    assert through_root == through_leaf
    nested = point.middle.leaf.field(Leaf.value)
    direct = point.field(through_root)
    assert isinstance(nested, BoundDecision) and isinstance(direct, BoundDecision)
    assert nested.state == direct.state
