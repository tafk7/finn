# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Value semantics and optional codecs for scalar datatypes and immutable arrays."""

from __future__ import annotations

from typing import cast

from finn.kernels.datatypes.values import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    canonical_qonnx_datatype,
    decode_datatype,
    encode_datatype,
    is_qonnx_datatype,
)
from finn.core.space import ValueSemantics
from finn.core.space.codecs import JSONValue, ValueCodec

QONNX_DATATYPE_VALUE_SEMANTICS: ValueSemantics[QONNXDataType] = ValueSemantics(
    type_token=QONNX_DATATYPE_TOKEN,
    name="QONNXDataType",
    recognizes=is_qonnx_datatype,
    equal=lambda left, right: bool(left == right),
    snapshot=canonical_qonnx_datatype,
)
QONNX_DATATYPE_SEMANTICS = cast(ValueSemantics[object], QONNX_DATATYPE_VALUE_SEMANTICS)


def _encode_datatype(value: QONNXDataType) -> JSONValue:
    return {key: name for key, name in encode_datatype(value).items()}


QONNX_DATATYPE_CODEC: ValueCodec[QONNXDataType] = ValueCodec(
    "finn.kernels.qonnx_datatype", 1, _encode_datatype, decode_datatype
)

IntegerVector = tuple[int, ...]
INTEGER_VECTOR: ValueSemantics[IntegerVector] = ValueSemantics(
    IntegerVector,
    "integer vector",
    lambda value: type(value) is tuple and all(type(item) is int for item in value),
    lambda left, right: left == right,
    lambda value: value,
)

IntegerMatrix = tuple[tuple[int, ...], ...]
INTEGER_MATRIX: ValueSemantics[IntegerMatrix] = ValueSemantics(
    IntegerMatrix,
    "integer matrix",
    lambda value: (
        type(value) is tuple
        and all(type(row) is tuple and all(type(item) is int for item in row) for row in value)
    ),
    lambda left, right: left == right,
    lambda value: value,
)

ThresholdTable = tuple[tuple[tuple[int, ...], ...], ...]


def _is_table(value: object) -> bool:
    return type(value) is tuple and all(
        type(table) is tuple
        and all(type(row) is tuple and all(type(item) is int for item in row) for row in table)
        for table in value
    )


THRESHOLD_TABLE: ValueSemantics[ThresholdTable] = ValueSemantics(
    ThresholdTable,
    "integer threshold table",
    _is_table,
    lambda left, right: left == right,
    lambda value: value,
)


__all__ = [
    "INTEGER_MATRIX",
    "INTEGER_VECTOR",
    "IntegerMatrix",
    "IntegerVector",
    "QONNX_DATATYPE_CODEC",
    "QONNX_DATATYPE_SEMANTICS",
    "QONNX_DATATYPE_VALUE_SEMANTICS",
    "THRESHOLD_TABLE",
    "ThresholdTable",
]
