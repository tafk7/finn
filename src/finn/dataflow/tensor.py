# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The fact a channel carries: a tensor of one element.

A ``Tensor`` is a row-major shape of positive extents and the element
``ScalarEncoding`` every position holds. Both ends of a channel read it; each
end presents its own traversal of it (``finn.dataflow.traversal``). Its
identity is the channel that carries it, so it holds no name.

An element is a datatype and the range of its values, by default the
datatype's. A producer that knows its values (a value owner) states a tighter
range; one element ``fits`` another when its values are values of the other,
which is what a channel checks between a producer and its consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
    qonnx_datatype_width,
)


@dataclass(frozen=True, init=False)
class ScalarEncoding:
    """Positive-width QONNX storage encoding and the range of its values.

    ``dtype`` is the datatype value itself (qonnx's are interned and frozen, so
    equality, hashing and the repr are the datatype's). ``value_range`` is the
    values' (minimum, maximum): the datatype's own unless stated tighter, and
    None for an encoding that is not an ordinary integer. Normalized, so
    ``INT8`` and ``INT8`` over ``(-128, 127)`` are one value.
    """

    dtype: QONNXDataType
    value_range: tuple[int, int] | None

    def __init__(self, dtype: QONNXDataType, value_range: tuple[int, int] | None = None) -> None:
        canonical = canonical_qonnx_datatype(dtype)
        if qonnx_datatype_width(canonical) < 1:
            raise ValueError("a scalar storage encoding must have positive width")
        try:
            full: tuple[int, int] | None = ordinary_integer_bounds(canonical)
        except DatatypeError:
            full = None
        if value_range is not None:
            if full is None:
                raise ValueError(f"{canonical.name} carries its datatype alone, not a range")
            low, high = value_range
            if not (type(low) is int and type(high) is int and full[0] <= low <= high <= full[1]):
                raise ValueError(f"{list(value_range)} is not a range of {canonical.name} values")
        object.__setattr__(self, "dtype", canonical)
        object.__setattr__(
            self, "value_range", full if value_range is None else (value_range[0], value_range[1])
        )

    def fits(self, other: ScalarEncoding) -> bool:
        """Its values are values of ``other``: one datatype, the range within ``other``'s."""
        if self.dtype != other.dtype:
            return False
        mine, theirs = self.value_range, other.value_range
        if mine is None or theirs is None:
            return mine == theirs
        return theirs[0] <= mine[0] and mine[1] <= theirs[1]

    def __str__(self) -> str:
        """The datatype's name, and the range when it is tighter than the datatype's."""
        if self.value_range is None or self.value_range == ordinary_integer_bounds(self.dtype):
            return self.dtype.name
        return f"{self.dtype.name} over {list(self.value_range)}"

    @property
    def bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @property
    def signed(self) -> bool:
        return self.dtype.signed()


@dataclass(frozen=True)
class Tensor:
    """A row-major tensor: positive ``shape`` extents, every position an ``element``."""

    shape: tuple[int, ...]
    element: ScalarEncoding

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not shape or any(type(extent) is not int or extent < 1 for extent in shape):
            raise ValueError("a tensor has at least one axis, each of positive extent")
        if not isinstance(self.element, ScalarEncoding):
            raise TypeError("a tensor's element is a ScalarEncoding")
        object.__setattr__(self, "shape", shape)

    @property
    def size(self) -> int:
        return prod(self.shape)


__all__ = ["ScalarEncoding", "Tensor"]
