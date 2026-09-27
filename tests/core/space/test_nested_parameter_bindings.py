# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A reusable interface leaves nested formals unsupplied; enclosing nodes supply them.

The nested ``bindings={...}`` map is gone. Its roles are played by assignment:
a nested formal left unsupplied is assigned through a path by an enclosing
family (``kernel.port.dtype = dtype``), which binds it for that placement only;
a formal that should be exposed is declared on the enclosing family and bound
by name; and a fresh inline ``Decision`` at a node call (also on a node the
caller supplies to a reference input) owns its choice.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from finn.core.space import (
    UNSUPPLIED,
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
    composite,
    configure,
    derived,
    divisors_of,
    inspection,
    selections,
)
from finn.core.space.errors import ConfigurationError, DefinitionError, RequestError


class Port(Space):
    dtype: Param[str] = Param(str)
    lanes: Param[int] = Param(int)

    @derived
    def description(*, dtype: str, lanes: int) -> tuple[str, int]:
        return dtype, lanes

    physical = View(description)


class Reusable(Space):
    """Its port's formals stay unsupplied: whoever places a Reusable supplies them."""

    count: Param[int] = Param(int)
    port = Port()


class Kernel(Space):
    """The caller supplies the port node itself, with its own bindings."""

    count: Param[int] = Param(int)
    port: Param[Port] = Param(Port)


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.code for finding in result.findings}


def test_outer_params_and_decisions_supply_interface_slots_without_new_choices() -> None:
    class Parent(Space):
        dtype: Param[str] = Param(str)
        lanes = Decision(int, values=(1, 2, 4))
        kernel = Reusable(count=1)
        kernel.port.dtype = dtype  # for this placement of Reusable only
        kernel.port.lanes = lanes

    base = configure(Parent(dtype="INT8"))
    assert base.kernel.port.dtype == "INT8"
    assert isinstance(base.kernel.port.query(Port.lanes), Unresolved)
    chosen = base.with_choices(lanes=2)
    assert chosen.kernel.port.physical() == ("INT8", 2)
    assert [item.key for item in inspection.decisions(chosen)] == ["lanes"]
    assert len(selections.capture(chosen).entries) == 1
    # A formal supplied by an assignment is not a choice of its own.
    with pytest.raises(RequestError, match="not independently editable"):
        chosen.with_choices({Parent.kernel.port.lanes: 1})
    evidence = inspection.explain(chosen.kernel.port, Port.description)
    assert any(
        node.declaration.key == "dtype" and node.input_presence == "supplied"
        for node in evidence.nodes
    )
    assert any(
        node.declaration.key == "lanes" and node.decision_state is not None
        for node in evidence.nodes
    )


def test_fresh_nested_decision_uses_outer_domain_and_guard_sources() -> None:
    class Parent(Space):
        extent: Param[int] = Param(int)
        enabled: Param[bool] = Param(bool)
        # A fresh Decision at the call of the node supplied to a family-typed
        # formal: its domain and guard read the enclosing family.
        kernel = Kernel(
            count=9,
            port=Port(dtype="INT4", lanes=Decision(int, domain=divisors_of(extent), when=enabled)),
        )

        @derived(lanes=kernel.port.lanes)
        def folded(*, lanes: int) -> int:
            return lanes * 2

    base = configure(Parent(extent=12, enabled=True))
    field = base.kernel.port.field(Port.lanes)
    assert isinstance(field, BoundDecision)  # the fresh Decision is owned here
    assert field.candidates() == Available((1, 2, 3, 4, 6, 12))
    chosen = base.with_choices({Parent.kernel.port.lanes: 3})
    assert chosen.folded == 6
    assert chosen.kernel.port.lanes == 3
    assert [item.key for item in inspection.decisions(chosen)] == ["kernel.port.lanes"]
    with pytest.raises(ConfigurationError):
        base.with_choices({Parent.kernel.port.lanes: 5})
    inactive = configure(Parent(extent=12, enabled=False))
    inactive_field = inactive.kernel.port.field(Port.lanes)
    assert isinstance(inactive_field, BoundDecision)
    assert isinstance(inactive_field.state, Inapplicable)
    assert isinstance(inactive.kernel.port.query(Port.lanes), Inapplicable)


def test_repeated_placements_keep_fresh_choices_independent() -> None:
    class Pair(Space):
        first = Kernel(count=1, port=Port(dtype="INT4", lanes=Decision(int, values=(1, 2))))
        second = Kernel(count=1, port=Port(dtype="INT8", lanes=Decision(int, values=(2, 4))))

    base = configure(Pair())
    chosen = base.with_choices({Pair.first.port.lanes: 2})
    assert chosen.first.port.physical() == ("INT4", 2)
    assert isinstance(chosen.second.port.query(Port.lanes), Unresolved)
    assert isinstance(base.first.port.query(Port.lanes), Unresolved)
    keys = [item.key for item in inspection.decisions(base)]
    assert keys == ["first.port.lanes", "second.port.lanes"]


def test_an_unnamed_decision_shared_by_two_nodes_is_a_definition_error() -> None:
    lanes = Decision(int, values=(1, 2, 4))  # unnamed: bound at calls, never a class attribute

    class Shared(Space):
        first = Kernel(count=1, port=Port(dtype="INT4", lanes=lanes))
        second = Kernel(count=2, port=Port(dtype="INT8", lanes=lanes))

    with pytest.raises(DefinitionError, match="a shared decision must be named") as caught:
        configure(Shared())
    assert "Port.lanes" in str(caught.value) and 'name="..."' in str(caught.value)


def test_a_named_decision_shared_by_two_nodes_is_one_decision_of_their_common_scope() -> None:
    lanes = Decision(int, values=(1, 2, 4), name="lanes")

    class Shared(Space):
        use_first = Decision(bool, values=(False, True))
        first = Kernel(count=1, port=Port(dtype="INT4", lanes=lanes), when=use_first)
        second = Kernel(count=2, port=Port(dtype="INT8", lanes=lanes))

    (info,) = [item for item in inspection.decisions(Shared) if item.key != "use_first"]
    # Owned by the lowest scope containing both uses, keyed by that scope and its name.
    assert (info.key, info.scope) == ("lanes", "")
    base = configure(Shared())
    # Either use edits the one decision; it applies whenever its owner does.
    point = base.with_choices({Shared.second.port.lanes: 4}, use_first=False)
    assert point.second.port.physical() == ("INT8", 4)
    assert isinstance(point.first.port.query(Port.lanes), Inapplicable)
    assert point.query(info.reference) == Available(4)
    both = point.with_choices({Shared.first.port.lanes: 2}, use_first=True)
    assert both.first.port.physical() == ("INT4", 2)
    assert both.second.port.physical() == ("INT8", 2)
    assert both.with_choices(lanes=1).second.port.physical() == ("INT8", 1)
    assert len(selections.capture(both).entries) == 2


def test_a_decision_name_is_checked_where_it_becomes_a_key() -> None:
    # A class attribute takes its attribute name; a different name= is refused.
    with pytest.raises(DefinitionError, match="a class attribute takes its attribute name"):
        composite("Renamed", {"lanes": Decision(int, values=(1,), name="other")})
    # A named shared decision is a member of its owner: it may not shadow one.
    lanes = Decision(int, values=(1, 2), name="width")

    class Clash(Space):
        width = Const(4)
        first = Kernel(count=1, port=Port(dtype="INT4", lanes=lanes))
        second = Kernel(count=2, port=Port(dtype="INT8", lanes=lanes))

    with pytest.raises(DefinitionError, match="is named like a member"):
        configure(Clash())


def test_unbound_exposure_is_a_formal_declared_on_the_enclosing_family() -> None:
    # An exposed inline Param is gone: declare the formal here, bind it by name.
    class Parent(Space):
        lanes: Param[int] = Param(int, default=UNSUPPLIED)
        kernel = Reusable(count=1)
        kernel.port.dtype = "INT8"
        kernel.port.lanes = lanes

    omitted = configure(Parent())
    assert omitted.kernel.port.dtype == "INT8"
    assert isinstance(omitted.kernel.port.query(Port.lanes), Unresolved)
    supplied = configure(Parent(lanes=3))
    assert supplied.kernel.port.physical() == ("INT8", 3)


def test_reexposed_nested_slot_can_be_bound_again_by_an_outer_placement() -> None:
    class Middle(Space):
        kernel = Reusable(count=1)  # the port's formals stay unsupplied through Middle

    class Outer(Space):
        lanes = Decision(int, values=(2, 4))
        middle = Middle()
        middle.kernel.port.dtype = "INT3"
        middle.kernel.port.lanes = lanes

    point = configure(Outer())
    chosen = point.with_choices(lanes=4)
    assert chosen.middle.kernel.port.physical() == ("INT3", 4)
    with pytest.raises(DefinitionError, match=r"kernel\.port\.dtype is not supplied"):
        configure(Middle())

    # Re-exposing by name: the enclosing family declares the formal and binds it.
    class Named(Space):
        dtype: Param[str] = Param(str)
        kernel = Reusable(count=1)
        kernel.port.dtype = dtype
        kernel.port.lanes = 2

    class Top(Space):
        named = Named(dtype="INT5")

    assert configure(Top()).named.kernel.port.physical() == ("INT5", 2)


@pytest.mark.parametrize("kind", ["literal", "alias", "decision"])
def test_an_outer_assignment_cannot_override_an_internal_binding(kind: str) -> None:
    class Internal(Space):
        local = Const(2)
        port = Port(
            dtype="INT8",
            lanes=2
            if kind == "literal"
            else local
            if kind == "alias"
            else Decision(int, values=(1, 2)),
        )

    with pytest.raises(DefinitionError, match="already supplied"):

        class Parent(Space):
            child = Internal()
            child.port.lanes = 4


def test_an_outer_assignment_beside_an_inner_one_is_refused_when_prepared() -> None:
    # A formal has one supplier. Two bodies assigning it through a path meet only
    # when the outer family is prepared.
    class Middle(Space):
        kernel = Reusable(count=1)
        kernel.port.dtype = "INT8"
        kernel.port.lanes = 2

    class Outer(Space):
        middle = Middle()
        middle.kernel.port.dtype = "INT4"

    assert configure(Middle()).kernel.port.physical() == ("INT8", 2)
    with pytest.raises(DefinitionError, match=r"middle\.kernel\.port\.dtype .*already supplied"):
        configure(Outer())


def test_assignment_targets_are_checked() -> None:
    class Parent(Space):
        child = Reusable(count=2)
        child.port.dtype = "INT8"
        child.port.lanes = 1

    assert configure(Parent()).child.count == 2

    with pytest.raises(DefinitionError, match="already supplied"):

        class DuplicateDirect(Space):
            child = Reusable(count=1)
            child.count = 2

    with pytest.raises(DefinitionError, match="already supplied"):

        class DuplicateNested(Space):
            child = Reusable(count=1)
            child.port.dtype = "INT8"
            child.port.dtype = "INT4"

    with pytest.raises(DefinitionError, match="has no formal 'description'"):

        class NonParameter(Space):
            child = Reusable(count=1)
            child.port.description = ("x", 2)

    with pytest.raises(DefinitionError, match="expected value of nominal type int"):
        Reusable().count = "two"  # type: ignore[assignment]
    with pytest.raises(DefinitionError, match="unknown formals"):
        Reusable(count=1, width=2)  # type: ignore[call-arg]


def test_assignment_after_freezing_is_refused() -> None:
    class Parent(Space):
        first: Param[str] = Param(str)
        second: Param[str] = Param(str)
        kernel = Reusable(count=1)
        kernel.port.dtype = first
        kernel.port.lanes = 1

    reference = Parent.kernel.port.dtype
    old = configure(Parent(first="INT3", second="INT7"))
    # Preparing Parent froze its node declarations: the model cannot drift.
    with pytest.raises(DefinitionError, match="is frozen .Parent was prepared") as caught:
        Parent.kernel.port.lanes = 2
    assert "assigned at test_nested_parameter_bindings.py:" in str(caught.value)
    point = configure(Parent(first="INT3", second="INT7"))
    assert old.query(reference) == point.query(reference) == Available("INT3")
    assert inspection.declaration(Parent.kernel).frozen == "Parent was prepared"
    with pytest.raises(TypeError):
        inspection.declaration(Parent.kernel).bindings["count"] = 2  # type: ignore[index]


def test_configure_freezes_its_root() -> None:
    root = Reusable()
    root.count = 3
    root.port.dtype = "INT8"
    root.port.lanes = 2
    assert configure(root).port.physical() == ("INT8", 2)
    with pytest.raises(DefinitionError, match="is frozen"):
        root.count = 4


def test_a_graph_built_as_data_binds_nested_formals_and_keeps_their_types() -> None:
    # ScopeBuilder is gone: nodes are plain values, joined by assignment, named by composite.
    child = Reusable(count=1)
    child.port.dtype = "INT8"
    child.port.lanes = 2
    declaration = inspection.declaration(child)
    assert dict(declaration.bindings) == {"count": 1}
    assert dict(declaration.nested) == {"port.dtype": "INT8", "port.lanes": 2}
    assert declaration.unsupplied == () and declaration.frozen is None
    family = composite("Parent", {"child": child})
    point = configure(family())
    placed = getattr(point, "child")
    assert isinstance(placed, Reusable)
    assert placed.port.physical() == ("INT8", 2)
    with pytest.raises(DefinitionError, match=r"child\.port\.dtype is not supplied"):
        configure(composite("Unbound", {"child": Reusable(count=1)})())


def test_nested_decision_reference_can_traverse_concrete_reference_layers() -> None:
    class Leaf(Space):
        value = Decision(int, values=(1, 2))

    class Middle(Space):
        leaf = Leaf()

    class Outer(Space):
        middle = Middle()

    reference = Outer.middle.leaf.value
    base = configure(Outer())
    point = base.with_choices({reference: 2})
    assert point.middle.leaf.value == 2
    through_root = inspection.decision_handle(point, reference)
    through_leaf = inspection.decision_handle(point.middle.leaf, Leaf.value)
    assert through_root == through_leaf
    nested = point.middle.leaf.field(Leaf.value)
    direct = point.field(through_root)
    assert isinstance(nested, BoundDecision) and isinstance(direct, BoundDecision)
    assert nested.state == direct.state


def test_nested_binding_typing_rejects_a_supplier_of_the_wrong_type(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    assert mypy is not None
    source = """from finn.core.space import Param, Space
class Port(Space):
    width: Param[int] = Param(int)
class Child(Space):
    port = Port()
class Parent(Space):
    child = Child()
    child.port.width = 4
    # Assignment is typed by Param.__set__: the value type is checked exactly.
    child.port.width = "wrong"  # E
    fresh = Port(width="wrong")  # E
"""
    fixture = tmp_path / "nested_binding_types.py"
    fixture.write_text(source)
    root = Path(__file__).resolve().parents[3]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            mypy,
            "--strict",
            "--explicit-package-bases",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "cache"),
            str(fixture),
        ],
        cwd=root,
        env=dict(environment, MYPYPATH=f"{root / 'src'}:{root / 'tests'}"),
        capture_output=True,
        text=True,
        check=False,
    )
    expected = {line for line, text in enumerate(source.splitlines(), 1) if "# E" in text}
    actual = {
        int(line) for line in re.findall(r"nested_binding_types\.py:(\d+): error:", result.stdout)
    }
    assert result.returncode == 1, result.stdout + result.stderr
    assert actual == expected, result.stdout + result.stderr
