# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The fact a stream carries: a tensor of one scalar storage encoding.

A ``Tensor`` is a row-major shape of positive extents and the element
``ScalarEncoding`` every position holds. Both ends of a stream read it; each
end presents its own traversal of it (``finn.dataflow.traversal``). Its
identity is the stream that carries it, so it holds no name.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from finn.core.space import Rejected, reject
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    qonnx_datatype_width,
    resolve_qonnx_datatype_name,
)


@dataclass(frozen=True, init=False)
class ScalarEncoding:
    """Positive-width QONNX storage encoding, detached by canonical identity."""

    datatype_name: str

    def __init__(self, dtype: QONNXDataType) -> None:
        canonical = canonical_qonnx_datatype(dtype)
        if qonnx_datatype_width(canonical) < 1:
            raise ValueError("a scalar storage encoding must have positive width")
        object.__setattr__(self, "datatype_name", canonical.name)

    @classmethod
    def admit(cls, dtype: QONNXDataType) -> ScalarEncoding | Rejected:
        """The encoding, or a ``dtype-storage`` refusal for a zero-width dtype."""
        try:
            return cls(dtype)
        except ValueError as error:
            return reject("dtype-storage", str(error))

    @property
    def dtype(self) -> QONNXDataType:
        return resolve_qonnx_datatype_name(self.datatype_name)

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
