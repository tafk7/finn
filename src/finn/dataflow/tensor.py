# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The fact a channel carries: a tensor whose elements share one encoding.

A ``Tensor`` is a row-major shape of positive extents and the encoding
``ScalarEncoding`` of every element. Both sides of a channel read it; each
port presents its own traversal of it (``finn.dataflow.traversal``). Its
identity is the channel that carries it, so it holds no name.

An encoding is a datatype and the range of its values (qonnx's ``TensorRange``,
``qonnx.core.ranges``), by default the datatype's full range. They are one fact, as
they are on the graph: a tensor's range is part of its datatype annotation, for any
numeric datatype (exact dyadic endpoints, possibly unbounded, and whether it may hold
NaN). A producer that knows its values (a value owner, or a kernel stating its output)
states a tighter range; one encoding ``fits`` another when its values are values of the
other, which is what a channel checks between each stream and its tensor
(``finn.dataflow.streams.misfit``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import prod

from qonnx.core import ranges
from qonnx.core.ranges import TensorRange, holds

from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    qonnx_datatype_width,
)


@cache
def full(dtype: QONNXDataType) -> TensorRange:
    """``dtype``'s full range (``qonnx.core.ranges.full``), read once per datatype."""
    return ranges.full(dtype)


def within(inner: TensorRange, outer: TensorRange) -> bool:
    """Every value ``inner`` admits is one ``outer`` admits: its ends within ``outer``'s,
    and NaN only where ``outer`` admits it."""
    return (
        outer.least <= inner.least
        and inner.greatest <= outer.greatest
        and (outer.nan or not inner.nan)
    )


@dataclass(frozen=True, init=False)
class ScalarEncoding:
    """Positive-width QONNX storage encoding and the range of its values.

    ``dtype`` is the datatype value itself (qonnx's are interned and frozen, so
    equality, hashing and the repr are the datatype's). ``value_range`` is the
    values' range, a ``TensorRange``: the datatype's full range (``qonnx.core.ranges.
    full``: an integer's bounds, a float's unbounded with NaN) unless stated tighter,
    and one the datatype holds (``holds``). Normalized, so ``INT8`` and ``INT8`` over
    ``[-128, 127]`` are one value.
    """

    dtype: QONNXDataType
    value_range: TensorRange

    def __init__(self, dtype: QONNXDataType, value_range: TensorRange | None = None) -> None:
        canonical = canonical_qonnx_datatype(dtype)
        if qonnx_datatype_width(canonical) < 1:
            raise ValueError("a scalar storage encoding must have positive width")
        if value_range is None:
            value_range = full(canonical)
        elif not isinstance(value_range, TensorRange):
            raise TypeError(f"a range is a TensorRange, not {value_range!r}")
        elif not holds(canonical, value_range):
            raise ValueError(f"{value_range} is not a range of {canonical.name} values")
        object.__setattr__(self, "dtype", canonical)
        object.__setattr__(self, "value_range", value_range)

    def fits(self, other: ScalarEncoding) -> bool:
        """Its values are values of ``other``: one datatype, the range within ``other``'s
        (and NaN only where ``other`` may hold it)."""
        return self.dtype == other.dtype and within(self.value_range, other.value_range)

    @property
    def stated(self) -> bool:
        """Whether its range says more than its datatype's (is tighter than ``full``)."""
        return self.value_range != full(self.dtype)

    def __str__(self) -> str:
        """The datatype's name, and the range when it is tighter than the datatype's."""
        return f"{self.dtype.name} over {self.value_range}" if self.stated else self.dtype.name

    @property
    def bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @property
    def signed(self) -> bool:
        return self.dtype.signed()


@dataclass(frozen=True)
class Tensor:
    """A row-major tensor: positive ``shape`` extents, ``element`` the encoding of every element."""

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


__all__ = ["ScalarEncoding", "Tensor", "full", "within"]
