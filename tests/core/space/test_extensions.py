# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Neutral extension bundles are ordinary declarations composed into a Space class.

Declarations and nodes are plain Python values, and ``composite(name, members,
base=B, exports=...)`` names them as a new Space class exactly as a class body
would. The typed surface is the base Space class: formals a caller binds are
declared on ``B``; the members composite adds are reached by name (or through
inspection handles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.core.space import (
    Available,
    Const,
    Decision,
    DefinitionError,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueSemantics,
    View,
    ViewKey,
    composite,
    constraint,
    default_semantics,
    derived,
    design_space,
    inspection,
)


@dataclass(frozen=True)
class Encoding:
    bits: int
    integer: bool = True


@dataclass(frozen=True)
class StreamValue:
    dtype: Encoding
    lanes: int
    bits: int


ENCODING = ValueSemantics.immutable_nominal(Encoding)
STREAM = ViewKey("stream", StreamValue)
BITS = ViewKey("bits", int)


class StreamShape(Space):
    dtype: Encoding = Param(semantics=ENCODING)
    lanes: int = Param()

    @derived
    def element_bits(*, dtype: Encoding) -> int:
        return dtype.bits

    @derived
    def bits(*, element_bits: int, lanes: int) -> int:
        return element_bits * lanes


class AdmissionFormals(StreamShape):
    """The formals the admitted stream adds, declared so node calls stay typed."""

    maximum_bits: int = Param()


def _admitted(*, dtype: Encoding, maximum_bits: int) -> bool:
    return dtype.integer and 0 < dtype.bits <= maximum_bits


def _positive_lanes(*, lanes: int, minimum: int) -> bool:
    return lanes >= minimum


def _stream(*, dtype: Encoding, lanes: int, bits: int) -> StreamValue:
    return StreamValue(dtype, lanes, bits)


def admitted_stream_type() -> type[AdmissionFormals]:
    """Admission checks and the accepted stream, built as data and named by composite."""
    minimum = Const(1)
    admission = constraint(
        dtype=AdmissionFormals.dtype, maximum_bits=AdmissionFormals.maximum_bits
    )(_admitted)
    geometry = constraint(lanes=AdmissionFormals.lanes, minimum=minimum)(_positive_lanes)
    value = derived(
        dtype=AdmissionFormals.dtype, lanes=AdmissionFormals.lanes, bits=AdmissionFormals.bits
    )(_stream)
    stream = View(value, requires=(admission, geometry))
    bits = View(AdmissionFormals.bits)
    return composite(
        "AdmittedStream",
        {
            "minimum_lanes": minimum,
            "admitted": admission,
            "positive_lanes": geometry,
            "stream_value": value,
            "stream": stream,
            "bits_view": bits,
        },
        base=AdmissionFormals,
        exports={STREAM: stream, BITS: bits},
    )


ADMITTED_STREAM = admitted_stream_type()


def stream_shape(
    *,
    dtype: Encoding,
    lanes: int,
    maximum_bits: int,
) -> AdmissionFormals:
    # Suppliers are typed as their values: a reference (``outer.lanes``) is an int.
    """A fresh node of the admitted stream Space class for each call."""
    return ADMITTED_STREAM(dtype=dtype, lanes=lanes, maximum_bits=maximum_bits)


def stream_of(node: Space) -> StreamValue:
    """A reference to the ``stream`` view composite added: reached by name, not by the
    base's type. Like every reference through a node it is typed as its value."""
    return cast(StreamValue, getattr(node, "stream"))


def test_stream_shape_places_independent_choices_and_keeps_narrow_fields_available() -> None:
    class Pair(Space):
        limit: int = Param(required=False)
        # The formals the children take are declared here.
        left_dtype: Encoding = Param(semantics=ENCODING)
        right_dtype: Encoding = Param(semantics=ENCODING)
        left = stream_shape(
            dtype=left_dtype,
            lanes=Decision(values=(1, 2, 4)),
            maximum_bits=limit,
        )
        right = stream_shape(
            dtype=right_dtype,
            lanes=Decision(values=(1, 2, 4)),
            maximum_bits=limit,
        )

        @constraint(a=left.bits, b=right.bits)
        def balanced(*, a: int, b: int) -> bool:
            return a == b

    # The bundle's own members stay inside its Space class.
    assert "maximum_bits" not in vars(Pair) and "minimum_lanes" not in vars(Pair)
    missing = design_space(Pair(left_dtype=Encoding(3), right_dtype=Encoding(6)))
    first = missing.with_choices({Pair.left.lanes: 2})
    assert first.left.bits == 6
    assert first.right.element_bits == 6
    assert isinstance(first.right.query(StreamShape.lanes), Unresolved)
    assert isinstance(first.query(stream_of(Pair.left)), Unresolved)
    assert isinstance(missing.left.query(StreamShape.lanes), Unresolved)

    admitted = design_space(Pair(left_dtype=Encoding(3), right_dtype=Encoding(6), limit=4))
    selected = admitted.with_choices({Pair.left.lanes: 2, Pair.right.lanes: 1})
    assert selected.left.bits == selected.right.bits == 6
    assert selected.inspect(Pair.balanced).verdict is True
    assert selected.query(stream_of(Pair.left)) == Available(StreamValue(Encoding(3), 2, 6))
    refused = selected.query(stream_of(Pair.right))
    assert isinstance(refused, Rejected)
    assert any("right" in finding.owner for finding in refused.findings)


def test_composite_is_pure_and_nodes_share_only_the_space_class() -> None:
    calls: list[int] = []

    class Scaled(Space):
        value: int = Param()

    def compute(*, value: int, choice: int) -> int:
        calls.append(value)
        return value * choice

    choice: int = Decision(values=(1, 2))
    output = derived(compute)
    exported = View(output)
    key = ViewKey("output", int)
    members = {"choice": choice, "output": output, "exported": exported}
    template = composite(
        "Template", members, base=Scaled, annotations={"choice": int}, exports={key: exported}
    )
    assert calls == []
    # A declaration belongs to one Space class: the same members cannot be named twice.
    with pytest.raises(DefinitionError, match="already belongs"):
        composite("Again", members, base=Scaled)

    class Parent(Space):
        first = template(value=3)
        second = template(value=3)

    first, second = inspection.declaration(Parent.first), inspection.declaration(Parent.second)
    assert first.space_type is second.space_type is template
    assert first.bindings == second.bindings == {"value": 3}
    point = design_space(Parent())
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    assert set(handles) == {"first.choice", "second.choice"}
    selected = point.with_choices({handles["first.choice"]: 2})
    assert selected.query(getattr(Parent.first, "exported")) == Available(6)
    assert isinstance(selected.second.query(choice), Unresolved)
    assert isinstance(point.first.query(choice), Unresolved)
    assert calls == [3]


def test_a_compiled_composite_rejects_structural_changes() -> None:
    calls: list[int] = []

    def snapshot(value: int) -> int:
        calls.append(value)
        return value

    semantics = ValueSemantics(
        int, "integer", lambda value: type(value) is int, int.__eq__, snapshot
    )
    value = Const(1)
    space_type = composite("Sealed", {"value": value, "physical": View(value)})
    assert design_space(space_type()).query(value) == Available(1)
    # There is no builder to seal: the compiled Space class itself refuses changes.
    with pytest.raises(DefinitionError, match="finalized"):
        setattr(space_type, "too_late", Const(1, semantics=semantics))
    with pytest.raises(DefinitionError, match="finalized"):
        setattr(space_type, "too_late", Decision(values=(1,), semantics=semantics))
    with pytest.raises(DefinitionError, match="finalized"):
        delattr(space_type, "value")
    with pytest.raises(DefinitionError, match="finalized"):
        setattr(space_type, "exports", {})
    with pytest.raises(DefinitionError, match="finalized"):
        setattr(space_type, "too_late", Const(1))
    assert calls == [1, 1]


def test_duplicate_names_owned_declarations_and_foreign_exports_are_definition_errors() -> None:
    with pytest.raises(DefinitionError, match="override changes value semantics"):
        composite("Shadow", {"dtype": Param(semantics=default_semantics(int))}, base=StreamShape)
    with pytest.raises(DefinitionError, match="already belongs"):
        composite("Renamed", {"renamed": StreamShape.dtype}, base=StreamShape)
    local = Const(3)
    with pytest.raises(DefinitionError, match="already belongs"):
        composite("Twice", {"constant": local, "other_name": local})
    with pytest.raises(DefinitionError, match="not a member"):
        composite("Foreign", {}, exports={ViewKey("foreign", int): View(StreamShape.bits)})
    view = View(Const(3))
    with pytest.raises(DefinitionError, match="duplicate export"):
        composite(
            "Duplicate",
            {"complete": view},
            exports={ViewKey("local", int): view, ViewKey("local", int): view},
        )
    constant = Const(4)
    with pytest.raises(DefinitionError, match="wrong kind"):  # exports are views only
        composite("Kind", {"constant": constant}, exports={ViewKey("constant", int): constant})
    with pytest.raises(DefinitionError, match="name segment"):
        composite("Invalid", {"invalid.name": Const(1)})
    with pytest.raises(DefinitionError, match="name segment"):
        composite("invalid.name", {})


def test_missing_bindings_and_incompatible_exports_fail_without_descriptor_runtime_errors() -> None:
    # A formal left unsupplied is refused where it would have to be supplied: when
    # the Space class placing the node is prepared.
    holder = composite("Holder", {"shape": StreamShape(dtype=Encoding(3))})
    with pytest.raises(DefinitionError, match=r"shape\.lanes is not supplied"):
        design_space(holder())
    integer = Const(3)
    view = View(integer)
    members = {"value": integer, "complete": view}
    with pytest.raises(DefinitionError, match="incompatible semantics"):
        composite("Wrong", members, exports={ViewKey("value", str): view})
    # The refused Space class still owns its declarations: they cannot be renamed silently.
    with pytest.raises(DefinitionError, match="already belongs"):
        composite("Retry", members, exports={ViewKey("value", int): view})


def test_external_references_must_be_explicit_formal_bindings() -> None:
    class Parent(Space):
        limit: int = Param()

    hidden = constraint(dtype=StreamShape.dtype, maximum_bits=Parent.limit)(_admitted)
    with pytest.raises(DefinitionError, match="not declared in this effective scope"):
        composite("Hidden", {"hidden_parent": hidden}, base=StreamShape)


def test_literal_bindings_snapshot_at_the_node_call() -> None:
    class Vector(Space):
        values: list[int] = Param()

    source = [1, 2]
    node = Vector(values=source)
    source.append(9)
    exposed = inspection.declaration(node).bindings["values"]
    assert exposed == [1, 2]
    assert isinstance(exposed, list)
    exposed.append(8)

    class Parent(Space):
        vector = node

    assert design_space(Parent()).vector.values == [1, 2]
