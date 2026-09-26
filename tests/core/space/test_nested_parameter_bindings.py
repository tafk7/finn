# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A reusable interface leaves nested formals open; enclosing nodes supply them.

The nested ``bindings={...}`` map is gone. Its roles are now played by graph
primitives: a nested formal left ``OPEN`` is supplied by a ``Bind`` edge of an
enclosing family; a formal that should be exposed is declared on the enclosing
family and bound by name; and a fresh inline ``Decision`` at a node call (also
on a node the caller supplies to a family-typed formal) owns its choice.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from finn.core.space import (
    OPEN,
    UNSUPPLIED,
    Available,
    Bind,
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
    """Its port's formals stay open: whoever places a Reusable supplies them."""

    count: Param[int] = Param(int)
    port = Port(dtype=OPEN, lanes=OPEN)


class Kernel(Space):
    """The caller supplies the port node itself, with its own bindings."""

    count: Param[int] = Param(int)
    port: Port = Param(Port)


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.code for finding in result.findings}


def test_outer_params_and_decisions_supply_interface_slots_without_new_choices() -> None:
    class Parent(Space):
        dtype: Param[str] = Param(str)
        lanes = Decision(int, values=(1, 2, 4))
        kernel = Reusable(count=1)
        kernel_dtype = Bind(kernel.port.dtype, dtype)
        kernel_lanes = Bind(kernel.port.lanes, lanes)

    base = configure(Parent(dtype="INT8"))
    assert base.kernel.port.dtype == "INT8"
    assert isinstance(base.kernel.port.query(Port.lanes), Unresolved)
    chosen = base.with_choices(lanes=2)
    assert chosen.kernel.port.physical() == ("INT8", 2)
    assert [item.key for item in inspection.decisions(chosen)] == ["lanes"]
    assert len(selections.capture(chosen).entries) == 1
    # A formal supplied by an edge is not a choice of its own.
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


def test_a_fresh_decision_shared_by_two_nodes_is_one_decision_of_their_common_scope() -> None:
    lanes = Decision(int, values=(1, 2, 4))  # fresh: bound at calls, never a class attribute

    class Shared(Space):
        use_first = Decision(bool, values=(False, True))
        first = Kernel(count=1, port=Port(dtype="INT4", lanes=lanes), when=use_first)
        second = Kernel(count=2, port=Port(dtype="INT8", lanes=lanes))

    (info,) = [item for item in inspection.decisions(Shared) if item.key != "use_first"]
    # Owned by the lowest scope containing both uses, keyed by its first use.
    assert (info.key, info.scope) == ("first.port.lanes", "")
    base = configure(Shared())
    # Either use edits the one decision; it applies whenever any use does.
    point = base.with_choices({Shared.second.port.lanes: 4}, use_first=False)
    assert point.second.port.physical() == ("INT8", 4)
    assert isinstance(point.first.port.query(Port.lanes), Inapplicable)
    assert point.query(info.reference) == Available(4)
    both = point.with_choices({Shared.first.port.lanes: 2}, use_first=True)
    assert both.first.port.physical() == ("INT4", 2)
    assert both.second.port.physical() == ("INT8", 2)
    assert len(selections.capture(both).entries) == 2


def test_unbound_exposure_is_a_formal_declared_on_the_enclosing_family() -> None:
    # An exposed inline Param is gone: declare the formal here, bind it by name.
    class Parent(Space):
        lanes: Param[int] = Param(int, default=UNSUPPLIED)
        kernel = Reusable(count=1)
        kernel_dtype = Bind(kernel.port.dtype, "INT8")
        kernel_lanes = Bind(kernel.port.lanes, lanes)

    omitted = configure(Parent())
    assert omitted.kernel.port.dtype == "INT8"
    assert isinstance(omitted.kernel.port.query(Port.lanes), Unresolved)
    supplied = configure(Parent(lanes=3))
    assert supplied.kernel.port.physical() == ("INT8", 3)


def test_reexposed_nested_slot_can_be_bound_again_by_an_outer_placement() -> None:
    class Middle(Space):
        kernel = Reusable(count=1)  # the port's formals stay open through Middle

    class Outer(Space):
        lanes = Decision(int, values=(2, 4))
        middle = Middle()
        dtype_edge = Bind(middle.kernel.port.dtype, "INT3")
        lanes_edge = Bind(middle.kernel.port.lanes, lanes)

    point = configure(Outer())
    chosen = point.with_choices(lanes=4)
    assert chosen.middle.kernel.port.physical() == ("INT3", 4)
    with pytest.raises(DefinitionError, match="no Bind supplying them"):
        configure(Middle())

    # Re-exposing by name: the enclosing family declares the formal and binds it.
    class Named(Space):
        dtype: Param[str] = Param(str)
        kernel = Reusable(count=1)
        dtype_edge = Bind(kernel.port.dtype, dtype)
        lanes_edge = Bind(kernel.port.lanes, 2)

    class Top(Space):
        named = Named(dtype="INT5")

    assert configure(Top()).named.kernel.port.physical() == ("INT5", 2)


@pytest.mark.parametrize("kind", ["literal", "alias", "decision"])
def test_an_outer_bind_cannot_override_an_internal_binding(kind: str) -> None:
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

    class Parent(Space):
        child = Internal()
        override = Bind(child.port.lanes, 4)

    with pytest.raises(DefinitionError, match="already supplied"):
        configure(Parent())


def test_an_outer_bind_beside_an_inner_bind_is_refused_where_it_is_read() -> None:
    # An open formal is supplied by whichever of its Binds is present, like Present.
    class Middle(Space):
        kernel = Reusable(count=1)
        dtype_edge = Bind(kernel.port.dtype, "INT8")
        lanes_edge = Bind(kernel.port.lanes, 2)

    class Outer(Space):
        middle = Middle()
        again = Bind(middle.kernel.port.dtype, "INT4")

    assert configure(Middle()).kernel.port.physical() == ("INT8", 2)
    answer = configure(Outer()).middle.kernel.port.query(Port.dtype)
    assert codes(answer) == {"multiple-suppliers"}


def test_bind_targets_are_checked() -> None:
    class Parent(Space):
        child = Reusable(count=2)
        dtype_edge = Bind(child.port.dtype, "INT8")
        lanes_edge = Bind(child.port.lanes, 1)

    assert configure(Parent()).child.count == 2

    class DuplicateDirect(Space):
        child = Reusable(count=1)
        again = Bind(child.count, 2)

    with pytest.raises(DefinitionError, match="already supplied"):
        configure(DuplicateDirect())

    class DuplicateNested(Space):
        child = Reusable(count=1)
        first = Bind(child.port.dtype, "INT8")
        second = Bind(child.port.dtype, "INT4")
        lanes_edge = Bind(child.port.lanes, 1)

    assert codes(configure(DuplicateNested()).child.port.query(Port.dtype)) == {
        "multiple-suppliers"
    }

    class Foreign(Space):
        value: Param[str] = Param(str)

    with pytest.raises(DefinitionError, match="through a reference"):
        Bind(Foreign.value, "INT8")

    unplaced = Reusable(count=1)

    class ForeignTarget(Space):
        child = Reusable(count=1)
        edge = Bind(unplaced.port.dtype, "INT8")

    with pytest.raises(DefinitionError, match="is not placed"):
        configure(ForeignTarget())

    class NonParameter(Space):
        child = Reusable(count=1)
        edge = Bind(child.port.description, ("x", 2))

    with pytest.raises(DefinitionError, match="the target is not a formal"):
        configure(NonParameter())

    with pytest.raises(DefinitionError, match="unknown formals"):
        Reusable(count=1, width=2)  # type: ignore[call-arg]


def test_compiled_nested_binding_keeps_original_supplier_after_source_changes() -> None:
    class Parent(Space):
        first: Param[str] = Param(str)
        second: Param[str] = Param(str)
        kernel = Reusable(count=1)
        dtype_edge = Bind(kernel.port.dtype, first)
        lanes_edge = Bind(kernel.port.lanes, 1)

    reference = Parent.kernel.port.dtype
    old = configure(Parent(first="INT3", second="INT7"))
    Parent.dtype_edge.source = Parent.second
    new = configure(Parent(first="INT3", second="INT7"))
    assert old.query(reference) == Available("INT3")
    assert new.query(reference) == Available("INT3")
    with pytest.raises(TypeError):
        inspection.declaration(Parent.kernel).bindings["count"] = 2  # type: ignore[index]


def test_a_graph_built_as_data_binds_nested_formals_and_keeps_their_types() -> None:
    # ScopeBuilder is gone: nodes and edges are plain values that composite names.
    child = Reusable(count=1)
    edges = {"dtype": Bind(child.port.dtype, "INT8"), "lanes": Bind(child.port.lanes, 2)}
    declaration = inspection.declaration(child)
    assert dict(declaration.bindings) == {"count": 1} and declaration.open == ()
    family = composite("Parent", {"child": child, **edges})
    point = configure(family())
    placed = getattr(point, "child")
    assert isinstance(placed, Reusable)
    assert placed.port.physical() == ("INT8", 2)
    with pytest.raises(DefinitionError, match="Bind supplying them"):
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
    source = """from finn.core.space import OPEN, Bind, Param, Space
class Port(Space):
    width: Param[int] = Param(int)
class Child(Space):
    port = Port(width=OPEN)
class Parent(Space):
    child = Child()
    good = Bind(child.port.width, 4)
    # mypy infers Bind's T from both arguments (a join, here object), so an
    # unannotated Bind cannot reject the source; an explicit Bind[int] can.
    joined = Bind(child.port.width, "wrong")
    bad = Bind[int](child.port.width, "wrong")  # E
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
