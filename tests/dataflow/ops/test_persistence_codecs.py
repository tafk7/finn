# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every codec accepts exactly what it emits, and nothing that merely coerces.

A decoder looser than its encoder is not a convenience.  JSON has one numeric
tower and one truth value that is also an integer, so ``1``, ``true`` and
``"1"`` all coerce to something plausible -- and a design that reloads from a
document nobody wrote is the failure persistence exists to prevent.  The tests
here are therefore about *refusal*: the round trip is the easy half.

The second half is the contributor-facing contract.  A Decision whose value is
not one of the structural kinds must declare its own versioned codec, and the
layer must refuse it rather than invent an encoding -- so a synthetic
``TileShape`` stands in for the tiling, scheduling and policy values that are
coming, and proves the refusal, the round trip, the byte equality and the
version guard without waiting for one of them to exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.space import PersistentCodec
from finn.dataflow.space.declarations import AuthoringError, Decision
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.dataflow.ops.persistence import apply_graph_effects, assign_dataflow_scope_ids
from finn.dataflow.ops.schema import InputTensor, OutputTensor
from finn.dataflow.ops.state import (
    BOOL_CODEC,
    FLOAT_CODEC,
    INT_CODEC,
    SELECTOR_CODEC,
    STATE_ATTRIBUTE,
    STRING_CODEC,
    DecodeError,
    decode_state,
    encode_state,
    enum_codec,
    parse_codec_tag,
    structural_codec,
)

# -- the structural codecs -----------------------------------------------------


class Colour(Enum):
    RED = "red"
    BLUE = "blue"


class Count(Enum):
    ONE = 1
    TWO = 2


class Mixed(Enum):
    NAME = "one"
    NUMBER = 2


@pytest.mark.parametrize(
    ("codec", "value"),
    [
        (BOOL_CODEC, True),
        (BOOL_CODEC, False),
        (INT_CODEC, 0),
        (INT_CODEC, -7),
        (STRING_CODEC, ""),
        (STRING_CODEC, "dot_product"),
        (SELECTOR_CODEC, "supplied"),
        (FLOAT_CODEC, 4.0),
        (FLOAT_CODEC, 0.1),
        (FLOAT_CODEC, -1.5e-9),
    ],
    ids=str,
)
def test_a_structural_value_survives_the_json_boundary(
    codec: PersistentCodec[Any], value: object
) -> None:
    """Through real JSON, because that is where a float loses its last bits."""

    written = json.dumps({"value": codec.encode(value)})
    assert codec.decode(json.loads(written)["value"]) == value


@pytest.mark.parametrize(
    ("codec", "refused"),
    [
        # ``True`` is an ``int`` in Python and ``1`` is truthy in most readers;
        # both would round-trip "successfully" into the wrong value.
        (BOOL_CODEC, 1),
        (BOOL_CODEC, 0),
        (BOOL_CODEC, "true"),
        (INT_CODEC, True),
        (INT_CODEC, 2.0),
        (INT_CODEC, "2"),
        (STRING_CODEC, 2),
        (STRING_CODEC, True),
        (SELECTOR_CODEC, 0),
        # A bare JSON number is not an exact double and is indistinguishable
        # from an integer; only the tagged form is accepted.
        (FLOAT_CODEC, 4.0),
        (FLOAT_CODEC, 4),
        (FLOAT_CODEC, "0x1.0p+2"),
        (FLOAT_CODEC, {"float_hex": 4}),
        (FLOAT_CODEC, {"float_hex": "0x1.0p+2", "extra": 1}),
    ],
    ids=str,
)
def test_an_alternate_representation_is_refused_not_coerced(
    codec: PersistentCodec[Any], refused: Any
) -> None:
    with pytest.raises(DecodeError):
        codec.decode(refused)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_float_has_no_canonical_form(value: float) -> None:
    with pytest.raises(AuthoringError, match="finite"):
        FLOAT_CODEC.encode(value)


@pytest.mark.parametrize(
    ("codec", "wrong"),
    [(BOOL_CODEC, 1), (INT_CODEC, True), (STRING_CODEC, 1), (FLOAT_CODEC, 1)],
    ids=str,
)
def test_an_encoder_is_as_strict_as_its_decoder(codec: PersistentCodec[Any], wrong: Any) -> None:
    """Symmetry in the other direction: what cannot be read cannot be written."""

    with pytest.raises(AuthoringError):
        codec.encode(wrong)


def test_an_enum_persists_through_its_member_value_and_only_that_type() -> None:
    codec = enum_codec(Colour)

    assert codec.encode(Colour.RED) == "red"
    assert codec.decode("blue") is Colour.BLUE
    with pytest.raises(DecodeError):
        codec.decode(0)
    with pytest.raises(DecodeError):
        codec.decode(True)


def test_an_int_valued_enum_refuses_a_bool() -> None:
    codec = enum_codec(Count)

    assert codec.decode(2) is Count.TWO
    with pytest.raises(DecodeError):
        codec.decode(True)
    with pytest.raises(DecodeError):
        codec.decode("2")


def test_an_enum_with_no_single_canonical_kind_is_an_authoring_error() -> None:
    """Refused where it is declared, not where one member happens to be saved."""

    with pytest.raises(AuthoringError, match="canonical=PersistentCodec"):
        enum_codec(Mixed)


def test_a_codec_tag_carries_an_identity_and_a_positive_version() -> None:
    assert parse_codec_tag("dataflow.int@1") == ("dataflow.int", 1)
    assert parse_codec_tag("test.tile@12") == ("test.tile", 12)
    for malformed in ("dataflow.int", "@1", "dataflow.int@", "dataflow.int@0", "x@one", "x@1.0"):
        with pytest.raises(DecodeError):
            parse_codec_tag(malformed)


def test_only_the_structural_kinds_have_a_default_codec() -> None:
    assert structural_codec(int) is INT_CODEC
    assert structural_codec(bool) is BOOL_CODEC
    assert structural_codec(TileShape) is None
    assert structural_codec(object()) is None


@pytest.mark.parametrize("member", ["family", "family_version", "problem_fingerprint"], ids=str)
def test_an_empty_identity_member_is_refused(member: str) -> None:
    """An empty family is not "unspecified"; it is a document nobody can check."""

    document = json.loads(
        encode_state(
            family="test.op",
            family_version="1",
            problem_fingerprint="abc",
            commitment_stage="dataflow",
            assignments={},
        )
    )
    document[member] = ""

    with pytest.raises(DecodeError, match="non-empty string"):
        decode_state(json.dumps(document))


# -- a Decision the structural codecs do not cover ------------------------------


@dataclass(frozen=True, slots=True)
class TileShape:
    """A synthetic structured value: immutable, equal by value, not a scalar."""

    rows: int
    cols: int


TILES = (TileShape(1, 1), TileShape(2, 4))

TILE_CODEC: PersistentCodec[Any] = PersistentCodec(
    "test.tile",
    1,
    lambda value: [value.rows, value.cols],
    lambda value: TileShape(int(cast(Any, value)[0]), int(cast(Any, value)[1])),
)


class _TiledOp(DataflowOp):
    """An operation with one structured Decision and nothing else to say."""

    family = "test.tiled"

    activation = InputTensor(index=0)
    result = OutputTensor(index=0)

    tile = Decision(TileShape, values=TILES, canonical=TILE_CODEC)

    def selected_dataflow(self) -> None:
        return None


class _UncodedOp(DataflowOp):
    """The same Decision with no codec declared: an authoring error to persist."""

    family = "test.uncoded"

    activation = InputTensor(index=0)
    result = OutputTensor(index=0)

    tile = Decision(TileShape, values=TILES)

    def selected_dataflow(self) -> None:
        return None


def _model(op_type: str) -> ModelWrapper:
    node = helper.make_node(
        op_type, ["activation"], ["result"], domain=DATAFLOW_DOMAIN, name="tiled0"
    )
    graph = helper.make_graph(
        [node],
        "tiled",
        [helper.make_tensor_value_info("activation", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 4])],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("result", DataType["INT8"])
    model.set_initializer("activation", np.zeros((2, 4), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _bind(operation_type: type[DataflowOp], model: ModelWrapper) -> Any:
    return operation_type(model.graph.node[0], 1).bind(model, object())


def _tiled(tile: TileShape = TileShape(2, 4)) -> tuple[ModelWrapper, Any]:
    model = _model("_TiledOp")
    chosen = _bind(_TiledOp, model).assign(_TiledOp.tile, tile).root
    apply_graph_effects(model, chosen.graph_effects())
    return model, chosen


def test_a_structured_decision_without_a_codec_is_refused() -> None:
    """And refused when the model is walked, not when a value first needs writing."""

    model = _model("_UncodedOp")
    operation = _bind(_UncodedOp, model)

    with pytest.raises(AuthoringError, match="canonical=PersistentCodec"):
        operation.graph_effects()


def test_a_declared_codec_round_trips_a_structured_value() -> None:
    model, _chosen = _tiled()

    restored = _bind(_TiledOp, model)

    assert dict(restored.recorded()) == {"tile": TileShape(2, 4)}


def test_the_document_records_the_declared_identity_and_version() -> None:
    model, _chosen = _tiled()

    document = json.loads(_state_bytes(model))

    assert document["assignments"]["tile"] == {"codec": "test.tile@1", "value": [2, 4]}


def test_equal_values_produce_equal_bytes() -> None:
    """Canonical means byte-equal, or a content-addressed cache is noise."""

    first, _ = _tiled()
    second, _ = _tiled()

    assert _state_bytes(first) == _state_bytes(second)
    assert _state_bytes(_tiled(TileShape(1, 1))[0]) != _state_bytes(first)


def test_a_changed_codec_version_refuses_hydration() -> None:
    model, _chosen = _tiled()
    _patch_state(model, lambda document: _retag(document, "test.tile@2"))

    with pytest.raises(DataflowOpError, match="changed encoding is not reinterpreted"):
        _bind(_TiledOp, model)


def test_a_value_the_declared_codec_cannot_read_is_a_refusal() -> None:
    model, _chosen = _tiled()
    _patch_state(model, lambda document: _revalue(document, "not a tile"))

    with pytest.raises(DataflowOpError, match="cannot decode"):
        _bind(_TiledOp, model)


def _state_bytes(model: ModelWrapper) -> bytes:
    for item in model.graph.node[0].attribute:
        if item.name == STATE_ATTRIBUTE:
            return bytes(item.s)
    raise AssertionError("the node carries no dataflow state")


def _patch_state(model: ModelWrapper, edit: Any) -> None:
    for item in model.graph.node[0].attribute:
        if item.name == STATE_ATTRIBUTE:
            item.s = json.dumps(edit(json.loads(item.s.decode("utf-8")))).encode("utf-8")


def _retag(document: dict[str, Any], tag: str) -> dict[str, Any]:
    document["assignments"]["tile"]["codec"] = tag
    return document


def _revalue(document: dict[str, Any], value: object) -> dict[str, Any]:
    document["assignments"]["tile"]["value"] = value
    return document
