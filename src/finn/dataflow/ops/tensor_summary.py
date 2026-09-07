# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The declaration-owned canonical encoding of QONNX tensor summaries."""

from __future__ import annotations

import math

from qonnx.analysis.tensor_value_summary import TensorValueSummary  # type: ignore[import-not-found]

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

__all__ = ["TENSOR_VALUE_SUMMARY_CODEC"]
