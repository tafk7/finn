# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned durable logical records; no native handles, hashes or Python imports.

The closed tag table is a data schema. Reload validates with the original value
constructors, including overlap and marker refusals. Bytes identify the exact
descriptor spelling, not semantic equivalence. Unknown tags/fields are rejected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from qonnx.core.ranges import TensorRange
from typing import cast

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow_next.logical import Edge, LogicalPlan, Node, Port, Source, Transfer
from finn.dataflow_next.maps import Piece, PositionMap
from finn.dataflow_next.queries import BeatRoute, MapExport, SourceWordBound, WordScope
from finn.dataflow_next.reindex import Reader
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.tensor import ScalarEncoding, Tensor
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    LevelDone,
    LevelEnd,
    Loop,
    Repetition,
    Selected,
    Traversal,
)

SCHEMA = "logical-descriptors-1"
_VALUES: dict[str, Callable[..., object]] = {
    cls.__name__: cls
    for cls in (
        LogicalPlan,
        Edge,
        Node,
        Port,
        Source,
        Transfer,
        Piece,
        PositionMap,
        Reader,
        Index,
        Tensor,
        BeatSequence,
        Grouped,
        LevelDone,
        LevelEnd,
        Loop,
        Selected,
        Traversal,
        BeatRoute,
        WordScope,
        MapExport,
        SourceWordBound,
    )
}


def _encode(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        number = value
        return (
            number
            if math.isfinite(number)
            else {"float": "nan" if math.isnan(number) else ("+inf" if number > 0 else "-inf")}
        )
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if type(value) is Repetition:
        return {"type": "Repetition", "value": value.value}
    if type(value) is ScalarEncoding:
        span = value.value_range
        return {
            "type": "ScalarEncoding",
            "dtype": value.dtype.name,
            "least": _encode(span.least),
            "greatest": _encode(span.greatest),
            "nan": span.nan,
        }
    if type(value) is Schedule:
        return {
            "type": "Schedule",
            "extents": _encode(value.extents),
            "factors": _encode(value.factors),
        }
    if (
        is_dataclass(value)
        and not isinstance(value, type)
        and _VALUES.get(type(value).__name__) is type(value)
    ):
        return {
            "type": type(value).__name__,
            **{field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    raise TypeError(f"not a durable logical descriptor: {type(value).__name__}")


def _exact(value: dict[str, object], names: set[str]) -> None:
    if set(value) != names:
        raise ValueError(f"descriptor fields {set(value)} must be {names}")


def _decode(value: object) -> object:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("a descriptor contains only JSON values")
    record = cast(dict[str, object], value)
    if "float" in record:
        _exact(record, {"float"})
        floats = {"+inf": math.inf, "-inf": -math.inf, "nan": math.nan}
        name = record["float"]
        if not isinstance(name, str) or name not in floats:
            raise ValueError("unknown float representation")
        return floats[name]
    tag = record.get("type")
    if not isinstance(tag, str):
        raise ValueError("a descriptor requires its type tag")
    if tag == "Repetition":
        _exact(record, {"type", "value"})
        return Repetition(record["value"])
    if tag == "ScalarEncoding":
        _exact(record, {"type", "dtype", "least", "greatest", "nan"})
        name, nan = record["dtype"], record["nan"]
        low, high = _decode(record["least"]), _decode(record["greatest"])
        if (
            not isinstance(name, str)
            or type(nan) is not bool
            or not all(type(v) in (int, float) for v in (low, high))
        ):
            raise ValueError("invalid encoding record")
        return ScalarEncoding(
            resolve_qonnx_datatype_name(name),
            TensorRange(cast(float, low), cast(float, high), nan),
        )
    if tag == "Schedule":
        _exact(record, {"type", "extents", "factors"})
        extents, factors = _decode(record["extents"]), _decode(record["factors"])
        for pairs in (extents, factors):
            if not isinstance(pairs, tuple) or not all(
                isinstance(pair, tuple)
                and len(pair) == 2
                and isinstance(pair[0], Index)
                and type(pair[1]) is int
                for pair in pairs
            ):
                raise ValueError("invalid schedule record")
            if len({pair[0] for pair in pairs}) != len(pairs):
                raise ValueError("duplicate schedule index")
        return Schedule(
            dict(cast(tuple[tuple[Index, int], ...], extents)),
            dict(cast(tuple[tuple[Index, int], ...], factors)),
        )
    if tag not in _VALUES:
        raise ValueError(f"unknown descriptor type {tag}")
    cls = _VALUES[tag]
    _exact(record, {"type", *(field.name for field in fields(cast(type, cls)))})
    return cls(**{name: _decode(item) for name, item in record.items() if name != "type"})


def dump(value: object) -> bytes:
    """Exact, portable descriptor bytes. Reject arbitrary objects/native handles."""
    return json.dumps(
        {"schema": SCHEMA, "descriptor": _encode(value)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load(data: bytes) -> object:
    """Validate and rebuild descriptor data; schema changes never import Python names."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("duplicate JSON key")
        return result

    def constant(name: str) -> object:
        raise ValueError(f"non-JSON constant {name}")

    record = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(record, dict):
        raise ValueError("a durable record is an object")
    _exact(record, {"schema", "descriptor"})
    if record["schema"] != SCHEMA:
        raise ValueError("unsupported descriptor schema")
    return _decode(record["descriptor"])
