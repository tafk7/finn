# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Native scalar/list encodings and declaration-owned structured codecs."""

from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch
from finn.custom_op.dataflow import custom_op
from qonnx.custom_op import registry
from finn.dataflow.ops.space import DataflowSpace
from typing import Any

import pytest
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.dataflow._engine import Decided
from finn.dataflow.ops.native import (
    AttributeCodec,
    NativeAttribute,
    choice_subset,
    operation_choice_schema,
    read_attributes,
    resolve_choice_subset,
    encode_choice_value,
    decode_choice_value,
)
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.space import Decision, Subspace, Space
from finn.dataflow.ops.schema import Attribute
from finn.dataflow.space.declarations import AuthoringError
from finn.dataflow.space.compiler import compile_space


class Colour(Enum):
    RED = "red"
    BLUE = "blue"


class Count(Enum):
    ONE = 1
    TWO = 2


class Fraction(Enum):
    QUARTER = 0.25


FRACTION_CODEC = AttributeCodec(
    "test.fraction",
    1,
    lambda value: value.value.hex(),
    lambda value: Fraction(float.fromhex(value)),
    "s",
)


@dataclass(frozen=True)
class Tile:
    rows: int
    cols: int


def _tile(value: Any) -> Tile:
    if not isinstance(value, list) or len(value) != 2 or any(type(i) is not int for i in value):
        raise ValueError("expected two integer extents")
    return Tile(*value)


TILE_CODEC = AttributeCodec("test.tile", 1, lambda value: [value.rows, value.cols], _tile, "ints")


class NativeSpace(DataflowSpace):
    family = "test.native"
    flag = Decision(bool, values=(False, True))
    count = Decision(int, values=(-2, 4))
    fraction = Decision(float, values=(0.5, 0.1))
    label = Decision(str, values=("", "hello"))
    colour = Decision(Colour, values=tuple(Colour))
    enum_count = Decision(Count, values=tuple(Count))
    enum_fraction = Decision(Fraction, values=tuple(Fraction), canonical=FRACTION_CODEC)
    shape = Decision(tuple, values=((), (2, 4)))
    tile = Decision(Tile, values=(Tile(2, 4),), canonical=TILE_CODEC)

    def selected_dataflow(self):
        return None


def _model(operation=NativeSpace):
    node = helper.make_node("TestOp", [], [], domain=DATAFLOW_DOMAIN, name="native")
    model = ModelWrapper(helper.make_model(helper.make_graph([node], "native", [], [])))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _op(model, space_type=NativeSpace):
    definition = space_type

    class TestOp(DataflowOp):
        space_type = definition

    with (
        patch.dict(registry._OP_REGISTRY, {}, clear=True),
        patch.dict(custom_op, {"TestOp": TestOp}),
    ):
        return model.get_customop_wrapper(model.graph.node[0])


@pytest.mark.parametrize(
    "declaration,value,kind,encoded",
    [
        (NativeSpace.flag, True, "i", 1),
        (NativeSpace.flag, False, "i", 0),
        (NativeSpace.count, -2, "i", -2),
        (NativeSpace.fraction, 0.5, "f", 0.5),
        (NativeSpace.label, "", "s", ""),
        (NativeSpace.colour, Colour.BLUE, "s", "blue"),
        (NativeSpace.enum_count, Count.TWO, "s", "2"),
        (NativeSpace.enum_fraction, Fraction.QUARTER, "s", (0.25).hex()),
        (NativeSpace.shape, (), "ints", ()),
        (NativeSpace.shape, (2, 4), "ints", (2, 4)),
        (NativeSpace.tile, Tile(2, 4), "ints", (2, 4)),
    ],
)
def test_native_values_round_trip_through_onnx(declaration, value, kind, encoded, tmp_path):
    model = _model()
    chosen = _op(model).space.assign(declaration, value)
    committed = _op(model).save_space(chosen)
    key = next(iter(committed.recorded()))
    assert read_attributes(model.graph.node[0])[key] == NativeAttribute(kind, encoded)
    path = tmp_path / "native.onnx"
    model.save(str(path))
    restored_model = ModelWrapper(str(path))
    restored = _op(restored_model).space
    assert restored.recorded()[key] == value
    assert type(restored.recorded()[key]) is type(value)
    assert not any("codec" in item.name for item in model.graph.node[0].attribute)


def test_captured_native_values_share_compiled_nominal_and_custom_codecs() -> None:
    model = _model()
    operation = _op(model).space
    for declaration, value in (
        (NativeSpace.colour, Colour.BLUE),
        (NativeSpace.enum_fraction, Fraction.QUARTER),
        (NativeSpace.tile, Tile(2, 4)),
    ):
        operation = operation.assign(declaration, value)
    paths = ("colour", "enum_fraction", "tile")
    schema = choice_subset(operation_choice_schema(NativeSpace), paths)
    resolved = resolve_choice_subset(operation, choice_subset(tuple(schema), paths))
    choices = tuple(
        (item, answer.value) for item, answer in resolved if isinstance(answer, Decided)
    )

    encoded = tuple(encode_choice_value(item, value) for item, value in choices)
    assert encoded == (
        "blue",
        (0.25).hex(),
        [2, 4],
    )
    decoded = tuple(decode_choice_value(item, value) for item, value in zip(schema, encoded))
    assert decoded == (
        Colour.BLUE,
        Fraction.QUARTER,
        Tile(2, 4),
    )
    assert tuple(type(item) for item in decoded) == (Colour, Fraction, Tile)
    with pytest.raises(AuthoringError, match="unique"):
        choice_subset(schema, ("colour", "colour"))


@pytest.mark.parametrize("encoded", ([1, True], [1.0, float("inf")], [1, 2.0]))
def test_native_tuple_codec_refuses_unrepresentable_values(encoded) -> None:
    schema = choice_subset(operation_choice_schema(NativeSpace), ("shape",))
    with pytest.raises((ValueError, TypeError), match="homogeneous|finite"):
        decode_choice_value(schema[0], encoded)


@pytest.mark.parametrize(
    "name,value",
    [
        ("flag", 2),
        ("flag", "true"),
        ("count", 4.0),
        ("fraction", 1),
        ("colour", "purple"),
        ("tile", "2,4"),
    ],
)
def test_native_decoding_does_not_coerce_alternate_representations(name, value):
    model = _model()
    _op(model).save_space()
    model.graph.node[0].attribute.append(helper.make_attribute(name, value))
    with pytest.raises(DataflowOpError, match="cannot decode"):
        _op(model).space


def test_float32_precision_loss_requires_an_explicit_codec():
    model = _model()
    chosen = _op(model).space.assign(NativeSpace.fraction, 0.1)
    with pytest.raises(AuthoringError, match="loses precision"):
        chosen.graph_effects()


def test_a_structured_decision_without_an_attribute_codec_is_refused_at_binding():
    class Uncoded(DataflowSpace):
        family = "test.uncoded"
        tile = Decision(Tile, values=(Tile(2, 4),))

    model = _model(Uncoded)
    with pytest.raises(AuthoringError, match="canonical=AttributeCodec"):
        _op(model, Uncoded).space


def test_stable_declaration_names_determine_native_attribute_names():
    class Child(Space):
        fold = Decision(int, values=(2,), name="PE")

    class Named(DataflowSpace):
        family = "test.named"
        child = Subspace(Child, name="selected_kernel")

        def selected_dataflow(self):
            return None

    model = _model(Named)
    root = _op(model, Named).space
    committed = _op(model, Named).save_space(root.child.assign(Child.fold, 2).root)
    assert committed.recorded() == {"selected_kernel.PE": 2}
    assert read_attributes(model.graph.node[0])["selected_kernel__PE"] == NativeAttribute("i", 2)


def test_deterministic_name_encoding_rejects_collisions():
    class Child(Space):
        axis = Decision(int, values=(2,))

    class Collision(DataflowSpace):
        family = "test.collision"
        child = Subspace(Child)
        flat = Decision(int, values=(2,), name="child__axis")

    model = _model(Collision)
    with pytest.raises(AuthoringError, match="collision"):
        compile_space(Collision, "unrelated_root")
    with pytest.raises(AuthoringError, match="collision"):
        _op(model, Collision).space


def test_source_and_decision_names_cannot_share_an_attribute():
    class Collision(DataflowSpace):
        family = "test.source_collision"
        source = Attribute(int, default=2, onnx="PE")
        fold = Decision(int, values=(2,), name="PE")

    model = _model(Collision)
    with pytest.raises(AuthoringError):
        _op(model, Collision).space
