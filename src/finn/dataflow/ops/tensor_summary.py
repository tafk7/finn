# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The declaration-owned canonical encoding of QONNX tensor summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy  # type: ignore[import-not-found]
from onnx import numpy_helper  # type: ignore[import-not-found]

from qonnx.analysis.tensor_value_summary import (  # type: ignore[import-not-found]
    TensorValueSummary,
    summarize_tensor_values,
)

from finn.dataflow.space.declarations import AuthoringError, CanonicalValue, CanonicalValueCodec


def _extremum(value: int | float | None) -> CanonicalValue:
    if value is None:
        return {"kind": "none"}
    if type(value) is int:
        encoded: dict[str, object] = {"kind": "integer", "value": value}
        # Equal int/float extrema must canonicalize equally. Keep the integer
        # intact and include its exact floating view when one exists. This
        # also gives every finite floating extremum a float.hex encoding.
        try:
            floating = float(value)
        except OverflowError:
            return encoded
        if math.isfinite(floating) and int(floating) == value:
            encoded["float_hex"] = floating.hex()
        return encoded
    if type(value) is float:
        if math.isnan(value):
            raise AuthoringError("a summary's NaN-only extrema must be None")
        if math.isinf(value):
            return {"kind": "+inf" if value > 0 else "-inf"}
        # Python equality identifies integral float extrema with integers and
        # -0.0 with 0.0. Both use the integer-plus-exact-float form above.
        if value.is_integer():
            return _extremum(int(value))
        return {"kind": "float", "hex": value.hex()}
    raise AuthoringError(f"unsupported summary extremum {type(value).__name__}")


def _encode_summary(summary: TensorValueSummary) -> CanonicalValue:
    return {
        "content_digest": summary.content_digest,
        "element_count": summary.element_count,
        "minimum": _extremum(summary.minimum),
        "maximum": _extremum(summary.maximum),
        "is_integral": summary.is_integral,
        "contains_zero": summary.contains_zero,
    }


TENSOR_VALUE_SUMMARY_CODEC: CanonicalValueCodec[TensorValueSummary] = CanonicalValueCodec(
    "finn.dataflow.tensor_value_summary", 1, _encode_summary
)


@dataclass(frozen=True, slots=True)
class FrozenInitializer:
    """Detached initializer payload captured with its QONNX content identity."""

    carrier_dtype: int
    numpy_dtype: str
    shape: tuple[int, ...]
    contiguous_bytes: bytes
    summary: TensorValueSummary

    @classmethod
    def from_tensor_proto(cls, tensor: Any) -> FrozenInitializer:
        original = numpy_helper.to_array(tensor)
        original_shape = tuple(int(extent) for extent in original.shape)
        array = numpy.ascontiguousarray(original).reshape(original_shape)
        summary = summarize_tensor_values(array)
        return cls(
            int(tensor.data_type),
            array.dtype.str,
            original_shape,
            array.tobytes(),
            summary,
        )

    def __post_init__(self) -> None:
        if type(self.carrier_dtype) is not int or self.carrier_dtype <= 0:
            raise ValueError("carrier_dtype must be a positive ONNX TensorProto enum")
        dtype = numpy.dtype(self.numpy_dtype)
        shape = tuple(self.shape)
        if any(type(extent) is not int or extent < 0 for extent in shape):
            raise ValueError("initializer shape extents must be non-negative integers")
        expected = int(numpy.prod(shape, dtype=numpy.int64)) * dtype.itemsize
        if len(self.contiguous_bytes) != expected:
            raise ValueError("initializer byte length does not match dtype and shape")
        array = numpy.frombuffer(self.contiguous_bytes, dtype=dtype).reshape(shape)
        if int(numpy_helper.from_array(array).data_type) != self.carrier_dtype:
            raise ValueError("initializer NumPy dtype does not match its ONNX carrier dtype")
        if summarize_tensor_values(array) != self.summary:
            raise ValueError("initializer payload does not match its value summary")
        object.__setattr__(self, "numpy_dtype", dtype.str)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "contiguous_bytes", bytes(self.contiguous_bytes))

    def array_copy(self) -> Any:
        return (
            numpy.frombuffer(self.contiguous_bytes, dtype=numpy.dtype(self.numpy_dtype))
            .reshape(self.shape)
            .copy()
        )


def encode_frozen_initializer(value: FrozenInitializer) -> CanonicalValue:
    return {
        "carrier_dtype": value.carrier_dtype,
        "numpy_dtype": value.numpy_dtype,
        "shape": list(value.shape),
        "contiguous_bytes": value.contiguous_bytes.hex(),
    }


def decode_frozen_initializer(value: object) -> FrozenInitializer:
    if not isinstance(value, Mapping) or set(value) != {
        "carrier_dtype",
        "numpy_dtype",
        "shape",
        "contiguous_bytes",
    }:
        raise ValueError("unsupported frozen-initializer payload")
    carrier = value["carrier_dtype"]
    dtype = value["numpy_dtype"]
    shape = value["shape"]
    payload = value["contiguous_bytes"]
    if type(carrier) is not int or type(dtype) is not str or type(payload) is not str:
        raise TypeError("frozen-initializer scalar fields have invalid types")
    if not isinstance(shape, (tuple, list)) or any(type(extent) is not int for extent in shape):
        raise TypeError("frozen-initializer shape must contain integers")
    try:
        raw = bytes.fromhex(payload)
    except ValueError as error:
        raise ValueError("frozen-initializer bytes are not hexadecimal") from error
    array = numpy.frombuffer(raw, dtype=numpy.dtype(dtype)).reshape(tuple(shape))
    frozen = FrozenInitializer.from_tensor_proto(numpy_helper.from_array(array))
    if frozen.carrier_dtype != carrier:
        raise ValueError("frozen-initializer carrier differs from its payload")
    return frozen


__all__ = [
    "FrozenInitializer",
    "TENSOR_VALUE_SUMMARY_CODEC",
    "decode_frozen_initializer",
    "encode_frozen_initializer",
]
