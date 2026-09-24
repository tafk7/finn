# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable interfaces expose typed slots that enclosing placements may bind."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from finn.core.space import (
    Const,
    Available,
    Decision,
    Inapplicable,
    Param,
    ScopeBuilder,
    Space,
    Subspace,
    Unresolved,
    View,
    compile_space,
    derived,
    divisors_of,
)
from finn.core.space import inspection, selections
from finn.core.space.errors import DefinitionError, RequestError


class Port(Space):
    dtype = Param(str)
    lanes = Param(int)

    @derived
    def description(*, dtype: str, lanes: int) -> tuple[str, int]:
        return dtype, lanes

    physical = View(description)


class Reusable(Space):
    count = Param(int)
    port = Subspace(Port, dtype=Param(str), lanes=Param(int, required=False))


def test_outer_params_and_decisions_supply_interface_slots_without_new_choices() -> None:
    class Parent(Space):
        dtype = Param(str)
        lanes = Decision(int, values=(1, 2, 4))
        kernel = Subspace(
            Reusable,
            count=1,
            bindings={Reusable.port.ref(Port.dtype): dtype, Reusable.port.ref(Port.lanes): lanes},
        )

    base = Parent({Parent.dtype: "INT8"})
    assert base.kernel.port.dtype == "INT8"
    assert isinstance(base.kernel.port.query(Port.lanes), Unresolved)
    chosen = base.with_choices(lanes=2)
    assert chosen.kernel.port.physical().accepted_result == Available(("INT8", 2))
    assert [item.key for item in inspection.decisions(chosen)] == ["lanes"]
    assert len(selections.capture(chosen).entries) == 1
    with pytest.raises(RequestError, match="Param alias"):
        chosen.with_choices(
            chosen.field(Parent.kernel.decision_ref(Reusable.port.ref(Port.lanes))).change(1)
        )
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
        extent = Param(int)
        enabled = Param(bool)
        kernel = Subspace(
            Reusable,
            count=9,
            bindings={
                Reusable.port.ref(Port.dtype): "INT4",
                Reusable.port.ref(Port.lanes): Decision(
                    int, domain=divisors_of(extent), when=enabled
                ),
            },
        )

        @derived(lanes=kernel.decision_ref(Reusable.port.ref(Port.lanes)))
        def folded(*, lanes: int) -> int:
            return lanes * 2

    base = Parent({Parent.extent: 12, Parent.enabled: True})
    handle = Parent.kernel.decision_ref(Reusable.port.ref(Port.lanes))
    assert base.field(handle).candidates() == Available((1, 2, 3, 4, 6, 12))
    chosen = base.with_choices(base.field(handle).change(3))
    assert chosen.folded == 6
    assert chosen.kernel.port.lanes == 3
    assert [item.key for item in inspection.decisions(chosen)] == ["kernel.port.lanes"]
    inactive = Parent({Parent.extent: 12, Parent.enabled: False})
    assert isinstance(inactive.field(handle).state, Inapplicable)
    assert isinstance(inactive.kernel.port.query(Port.lanes), Inapplicable)


def test_repeated_placements_keep_mapped_choices_independent() -> None:
    class Pair(Space):
        first = Subspace(
            Reusable,
            count=1,
            bindings={
                Reusable.port.ref(Port.dtype): "INT4",
                Reusable.port.ref(Port.lanes): Decision(int, values=(1, 2)),
            },
        )
        second = Subspace(
            Reusable,
            count=1,
            bindings={
                Reusable.port.ref(Port.dtype): "INT8",
                Reusable.port.ref(Port.lanes): Decision(int, values=(2, 4)),
            },
        )

    base = Pair()
    chosen = base.with_choices(
        base.field(Pair.first.decision_ref(Reusable.port.ref(Port.lanes))).change(2)
    )
    assert chosen.first.port.physical().accepted_result == Available(("INT4", 2))
    assert isinstance(chosen.second.port.query(Port.lanes), Unresolved)
    assert isinstance(base.first.port.query(Port.lanes), Unresolved)


def test_unbound_deliberate_exposure_remains_a_scoped_root_parameter() -> None:
    class Parent(Space):
        kernel = Subspace(Reusable, count=1, bindings={Reusable.port.ref(Port.dtype): "INT8"})

    model = compile_space(Parent)
    omitted = model.bind()
    assert omitted.kernel.port.dtype == "INT8"
    assert isinstance(omitted.kernel.port.query(Port.lanes), Unresolved)
    supplied = model.bind({Parent.kernel.ref(Reusable.port.ref(Port.lanes)): 3})
    assert supplied.kernel.port.physical().accepted_result == Available(("INT8", 3))


def test_reexposed_nested_slot_can_be_bound_again_by_an_outer_placement() -> None:
    class Middle(Space):
        kernel = Subspace(
            Reusable,
            count=1,
            bindings={
                Reusable.port.ref(Port.dtype): Param(str),
                Reusable.port.ref(Port.lanes): Param(int),
            },
        )

    class Outer(Space):
        middle = Subspace(
            Middle,
            bindings={
                Middle.kernel.ref(Reusable.port.ref(Port.dtype)): "INT3",
                Middle.kernel.ref(Reusable.port.ref(Port.lanes)): Decision(int, values=(2, 4)),
            },
        )

    point = Outer()
    decision = Outer.middle.decision_ref(Middle.kernel.ref(Reusable.port.ref(Port.lanes)))
    chosen = point.with_choices(point.field(decision).change(4))
    assert chosen.middle.kernel.port.physical().accepted_result == Available(("INT3", 4))


@pytest.mark.parametrize("kind", ["literal", "alias", "decision"])
def test_nested_maps_cannot_override_internal_bindings(kind: str) -> None:
    class Internal(Space):
        local = Const(2)
        port = Subspace(
            Port,
            dtype="INT8",
            lanes=2
            if kind == "literal"
            else local
            if kind == "alias"
            else Decision(int, values=(1, 2)),
        )

    class Parent(Space):
        child = Subspace(Internal, bindings={Internal.port.ref(Port.lanes): 4})

    with pytest.raises(DefinitionError, match="not deliberately exposed"):
        compile_space(Parent)


def test_outer_mapping_cannot_replace_an_inner_mapping_to_a_literal() -> None:
    class Middle(Space):
        kernel = Subspace(
            Reusable,
            count=1,
            bindings={
                Reusable.port.ref(Port.dtype): "INT8",
                Reusable.port.ref(Port.lanes): Param(int),
            },
        )

    class Outer(Space):
        middle = Subspace(
            Middle, bindings={Middle.kernel.ref(Reusable.port.ref(Port.dtype)): "INT4"}
        )

    with pytest.raises(DefinitionError, match="not deliberately exposed"):
        compile_space(Outer)


def test_direct_parameter_keys_and_duplicate_or_foreign_targets_are_checked() -> None:
    class Parent(Space):
        child = Subspace(
            Reusable, bindings={Reusable.count: 2, Reusable.port.ref(Port.dtype): "INT8"}
        )

    assert Parent().child.count == 2

    class DuplicateDirect(Space):
        child = Subspace(Reusable, count=1, bindings={Reusable.count: 2})

    with pytest.raises(DefinitionError, match="duplicate named and mapped"):
        compile_space(DuplicateDirect)

    class DuplicateNested(Space):
        child = Subspace(
            Reusable,
            count=1,
            bindings={Reusable.port.ref(Port.dtype): "INT8", Reusable.port.ref(Port.dtype): "INT4"},
        )

    with pytest.raises(DefinitionError, match="duplicate parameter"):
        compile_space(DuplicateNested)

    class Foreign(Space):
        value = Param(str)

    class ForeignTarget(Space):
        child = Subspace(Reusable, count=1, bindings={Foreign.value: "INT8"})

    with pytest.raises(DefinitionError, match="not a member"):
        compile_space(ForeignTarget)

    class NonParameter(Space):
        child = Subspace(
            Reusable, count=1, bindings={Reusable.port.ref(Port.description): ("x", 2)}
        )

    with pytest.raises(DefinitionError, match="not a Param"):
        compile_space(NonParameter)


def test_compiled_nested_binding_keeps_original_supplier_after_source_map_changes() -> None:
    class Parent(Space):
        first = Param(str)
        second = Param(str)
        kernel = Subspace(Reusable, count=1, bindings={Reusable.port.ref(Port.dtype): first})

    model = compile_space(Parent)
    reference = Parent.kernel.ref(Reusable.port.ref(Port.dtype))
    replacement = Subspace(
        Reusable, count=1, bindings={Reusable.port.ref(Port.dtype): Parent.second}
    )
    Parent.kernel.parameter_bindings = replacement.parameter_bindings
    old = model.bind({Parent.first: "INT3", Parent.second: "INT7"})
    new = compile_space(Parent).bind({Parent.first: "INT3", Parent.second: "INT7"})
    assert old.query(reference) == Available("INT3")
    assert new.query(reference) == Available("INT3")


def test_nested_extension_binding_preserves_scope_and_slot_type() -> None:
    builder = ScopeBuilder(Reusable)
    builder.bind(Reusable.count, 1)
    builder.binding(Reusable.port.ref(Port.dtype)).to("INT8")
    builder.binding(Reusable.port.ref(Port.lanes)).to(2)
    placement = builder.place()
    assert len(placement.parameter_bindings) == 2

    class Parent(Space):
        child = placement

    assert Parent().child.port.physical().accepted_result == Available(("INT8", 2))


def test_nested_decision_reference_can_traverse_concrete_reference_layers() -> None:
    class Leaf(Space):
        value = Decision(int, values=(1, 2))

    class Middle(Space):
        leaf = Subspace(Leaf)

    class Outer(Space):
        middle = Subspace(Middle)

    reference = Outer.middle.decision_ref(Middle.leaf.ref(Leaf.value))
    base = Outer()
    point = base.with_choices(base.field(reference).change(2))
    assert point.middle.leaf.value == 2
    nested = Outer.middle.decision_ref(Middle.leaf.decision_ref(Leaf.value))
    assert point.field(nested).state == point.field(reference).state


def test_nested_binder_typing_rejects_a_supplier_of_the_wrong_type(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    assert mypy is not None
    source = """from finn.core.space import Param, Space, Subspace, ScopeBuilder
class Port(Space):
    width = Param(int)
class Child(Space):
    port = Subspace(Port, width=Param(int))
builder = ScopeBuilder(Child)
builder.binding(Child.port.ref(Port.width)).to(4)
builder.binding(Child.port.ref(Port.width)).to("wrong")  # E
"""
    fixture = tmp_path / "nested_binding_types.py"
    fixture.write_text(source)
    root = Path(__file__).resolve().parents[3]
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
        env=dict(os.environ, MYPYPATH=f"{root / 'src'}:{root / 'tests'}"),
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
