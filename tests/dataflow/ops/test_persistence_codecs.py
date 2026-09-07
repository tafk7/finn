# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Native scalar/list encodings and declaration-owned structured codecs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.dataflow.ops.native import AttributeCodec, NativeAttribute, read_attributes
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


class NativeOp(DataflowOp):
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


def _model(operation=NativeOp):
    node = helper.make_node(operation.__name__, [], [], domain=DATAFLOW_DOMAIN, name="native")
    model = ModelWrapper(helper.make_model(helper.make_graph([node], "native", [], [])))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


@pytest.mark.parametrize(
    "declaration,value,kind,encoded",
    [
        (NativeOp.flag, True, "i", 1),
        (NativeOp.flag, False, "i", 0),
        (NativeOp.count, -2, "i", -2),
        (NativeOp.fraction, 0.5, "f", 0.5),
        (NativeOp.label, "", "s", ""),
        (NativeOp.colour, Colour.BLUE, "s", "blue"),
        (NativeOp.enum_count, Count.TWO, "s", "2"),
        (NativeOp.enum_fraction, Fraction.QUARTER, "s", (0.25).hex()),
        (NativeOp.shape, (), "ints", ()),
        (NativeOp.shape, (2, 4), "ints", (2, 4)),
        (NativeOp.tile, Tile(2, 4), "ints", (2, 4)),
    ],
)
def test_native_values_round_trip_through_onnx(declaration, value, kind, encoded, tmp_path):
    model = _model()
    chosen = NativeOp(model.graph.node[0]).bind(model, None).assign(declaration, value)
    committed = chosen.commit(model)
    key = next(iter(committed.recorded()))
    assert read_attributes(model.graph.node[0])[key] == NativeAttribute(kind, encoded)
    path = tmp_path / "native.onnx"
    model.save(str(path))
    restored_model = ModelWrapper(str(path))
    restored = NativeOp(restored_model.graph.node[0]).bind(restored_model, None)
    assert restored.recorded()[key] == value
    assert type(restored.recorded()[key]) is type(value)
    assert not any("codec" in item.name for item in model.graph.node[0].attribute)


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
    NativeOp(model.graph.node[0]).bind(model, None).commit(model)
    model.graph.node[0].attribute.append(helper.make_attribute(name, value))
    with pytest.raises(DataflowOpError, match="cannot decode"):
        NativeOp(model.graph.node[0]).bind(model, None)


def test_float32_precision_loss_requires_an_explicit_codec():
    model = _model()
    chosen = NativeOp(model.graph.node[0]).bind(model, None).assign(NativeOp.fraction, 0.1)
    with pytest.raises(AuthoringError, match="loses precision"):
        chosen.graph_effects()


def test_a_structured_decision_without_an_attribute_codec_is_refused_at_binding():
    class Uncoded(DataflowOp):
        family = "test.uncoded"
        tile = Decision(Tile, values=(Tile(2, 4),))

    model = _model(Uncoded)
    with pytest.raises(AuthoringError, match="canonical=AttributeCodec"):
        Uncoded(model.graph.node[0]).bind(model, None)


def test_stable_declaration_names_determine_native_attribute_names():
    class Child(Space):
        fold = Decision(int, values=(2,), name="PE")

    class Named(DataflowOp):
        family = "test.named"
        child = Subspace(Child, name="selected_design")

        def selected_dataflow(self):
            return None

    model = _model(Named)
    root = Named(model.graph.node[0]).bind(model, None)
    committed = root.child.assign(Child.fold, 2).root.commit(model)
    assert committed.recorded() == {"selected_design.PE": 2}
    assert read_attributes(model.graph.node[0])["selected_design__PE"] == NativeAttribute("i", 2)


def test_deterministic_name_encoding_rejects_collisions():
    class Child(Space):
        axis = Decision(int, values=(2,))

    class Collision(DataflowOp):
        family = "test.collision"
        child = Subspace(Child)
        flat = Decision(int, values=(2,), name="child__axis")

    model = _model(Collision)
    with pytest.raises(AuthoringError, match="collision"):
        compile_space(Collision, "unrelated_root")
    with pytest.raises(AuthoringError, match="collision"):
        Collision(model.graph.node[0]).bind(model, None)


def test_source_and_decision_names_cannot_share_an_attribute():
    class Collision(DataflowOp):
        family = "test.source_collision"
        source = Attribute(int, default=2, onnx="PE")
        fold = Decision(int, values=(2,), name="PE")

    model = _model(Collision)
    with pytest.raises(AuthoringError):
        Collision(model.graph.node[0]).bind(model, None)
