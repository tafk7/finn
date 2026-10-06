# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Value semantics for scalar datatypes and immutable arrays."""

from __future__ import annotations

import hashlib
import sys
from array import array
from collections.abc import Callable, Sequence
from itertools import chain
from math import prod
from typing import cast

import numpy as np
import numpy.typing as npt

from finn.core.space import ValueSemantics
from finn.dataflow.datatypes import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    canonical_qonnx_datatype,
    is_qonnx_datatype,
)

QONNX_DATATYPE_VALUE_SEMANTICS: ValueSemantics[QONNXDataType] = ValueSemantics(
    type_token=QONNX_DATATYPE_TOKEN,
    name="QONNXDataType",
    recognizes=is_qonnx_datatype,
    # One instance per canonical name: two datatypes are equal when they are the same.
    equal=lambda left, right: left is right,
    snapshot=canonical_qonnx_datatype,
)

IntegerVector = tuple[int, ...]
INTEGER_VECTOR: ValueSemantics[IntegerVector] = ValueSemantics(
    IntegerVector,
    "integer vector",
    lambda value: type(value) is tuple and all(type(item) is int for item in value),
    lambda left, right: left == right,
    lambda value: value,
)


def _shape(value: object) -> tuple[int, ...] | None:
    """The shape of a nonempty rectangular nest of tuples with int leaves, rank at
    least one; ``None`` for anything else."""

    def shape(item: object) -> tuple[int, ...] | None:
        if type(item) is int:
            return ()
        if type(item) is not tuple or not item:
            return None
        inner = {shape(element) for element in cast(tuple[object, ...], item)}
        if len(inner) != 1 or None in inner:
            return None
        (common,) = inner
        return None if common is None else (len(cast(tuple[object, ...], item)), *common)

    found = shape(value)
    return found if found else None


INTEGER_DIGEST = b"finn.integer-tensor/1\0"
"""The domain of an integer tensor's digest: bump it when the preimage changes."""


def integer_bytes(integers: Sequence[int]) -> bytes:
    """Row-major integers as ``integer_digest`` reads them: eight little-endian
    two's-complement bytes each, or, when one needs more than 64 bits, their decimal
    text (each form tagged, so the two never meet)."""
    try:
        data = array("q", integers)
    except OverflowError:
        return b"text:" + ",".join(map(str, integers)).encode()
    if sys.byteorder != "little":
        data.byteswap()
    return b"q:" + data.tobytes()


def integer_digest(shape: tuple[int, ...], data: bytes) -> str:
    """The digest of an integer tensor: its shape and its integers as ``integer_bytes``
    writes them. A producer that holds them in another form (an array) writes the same
    bytes."""
    hasher = hashlib.sha256(INTEGER_DIGEST)
    hasher.update(repr(tuple(shape)).encode())
    hasher.update(data)
    return hasher.hexdigest()


class IntegerTensorValue:
    """An integer tensor as a value: its ``shape``, the least and greatest of its
    integers (``range``) and a ``digest`` of them, stated when it is made; the integers
    themselves, row-major, loaded when first read (``integers``). Two are equal when
    their digests are: a stored operand's identity is its shape and integers, however
    they are stored (a float32 and an int8 initializer of the same integers are one
    value).

    Its producer states it. Nested int tuples make one by a single walk (``of``); a
    producer that holds the integers in another form states the facts and how to load
    them (a KernelOp's initializer, an array: loaded only when a memory image is
    packed). Loading checks the stated shape and range. A memory image reads them as
    ``row_major``: a read-only int64 array when the stated range fits in 64 bits, so
    the integers of an array are never made Python ints. Immutable; a copy is the value
    itself.
    """

    __slots__ = ("shape", "range", "digest", "_load", "_integers", "_array")

    shape: tuple[int, ...]
    range: tuple[int, int]
    digest: str
    _load: Callable[[], Sequence[int]]
    _integers: tuple[int, ...] | None
    _array: npt.NDArray[np.int64] | None

    def __init__(
        self,
        shape: tuple[int, ...],
        range: tuple[int, int],
        digest: str,
        load: Callable[[], Sequence[int]],
    ) -> None:
        shape = tuple(shape)
        if not shape or any(type(extent) is not int or extent < 1 for extent in shape):
            raise TypeError("an integer tensor has positive extents, rank at least one")
        least, greatest = range
        if type(least) is not int or type(greatest) is not int or least > greatest:
            raise TypeError("an integer tensor's range is its least and greatest integer")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "range", (least, greatest))
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "_load", load)
        object.__setattr__(self, "_integers", None)
        object.__setattr__(self, "_array", None)

    @classmethod
    def of(cls, values: object) -> IntegerTensorValue:
        """The value of nested int tuples: a nonempty rectangular nest, rank at least
        one, walked once."""
        shape = _shape(values)
        if shape is None:
            raise TypeError("an integer tensor is a nonempty rectangular nest of int tuples")
        flat = cast(tuple[object, ...], values)
        for _ in shape[1:]:
            flat = tuple(chain.from_iterable(cast(tuple[tuple[object, ...], ...], flat)))
        return cls._made(shape, cast(tuple[int, ...], flat))

    @classmethod
    def flat(cls, shape: tuple[int, ...], integers: Sequence[int]) -> IntegerTensorValue:
        """The value of ``shape`` whose row-major integers are ``integers``."""
        found = tuple(integers)
        if len(found) != prod(shape) or any(type(value) is not int for value in found):
            raise TypeError(f"an integer tensor of shape {tuple(shape)} holds {prod(shape)} ints")
        return cls._made(shape, found)

    @classmethod
    def _made(cls, shape: tuple[int, ...], found: tuple[int, ...]) -> IntegerTensorValue:
        made = cls(
            shape, (min(found), max(found)), integer_digest(shape, integer_bytes(found)), tuple
        )
        object.__setattr__(made, "_integers", found)
        return made

    @property
    def integers(self) -> tuple[int, ...]:
        """Every integer, row-major: loaded on first read, checked against the stated
        shape and range."""
        found = self._integers
        if found is None:
            if self._fits:
                found = tuple(self._int64().tolist())
            else:
                found = tuple(self._load())
                if len(found) != prod(self.shape) or (min(found), max(found)) != self.range:
                    raise self._misstated()
            object.__setattr__(self, "_integers", found)
        return found

    @property
    def row_major(self) -> Sequence[int] | npt.NDArray[np.int64]:
        """Every integer, row-major, as a memory image packs them: a read-only int64
        array when the stated range fits in 64 bits, else ``integers``. Loaded on first
        read and checked as ``integers`` is."""
        return self._int64() if self._fits else self.integers

    @property
    def _fits(self) -> bool:
        least, greatest = self.range
        return -(2**63) <= least and greatest < 2**63

    def _int64(self) -> npt.NDArray[np.int64]:
        found = self._array
        if found is None:
            loaded = self._integers if self._integers is not None else self._load()
            try:
                found = np.array(loaded, dtype=np.int64)
            except OverflowError:
                raise self._misstated() from None
            if found.shape != (prod(self.shape),):
                raise self._misstated()
            if (int(found.min()), int(found.max())) != self.range:
                raise self._misstated()
            found.flags.writeable = False
            object.__setattr__(self, "_array", found)
        return found

    def _misstated(self) -> ValueError:
        return ValueError(
            f"an integer tensor stated as {self.shape} over {list(self.range)} "
            "loaded other integers"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"an integer tensor value is immutable; cannot set {name}")

    def __eq__(self, other: object) -> bool:
        if type(other) is not IntegerTensorValue:
            return NotImplemented
        return self.digest == other.digest

    def __hash__(self) -> int:
        return hash(self.digest)

    def __repr__(self) -> str:
        return (
            f"IntegerTensorValue(shape={self.shape}, range={self.range}, digest={self.digest[:16]})"
        )

    def __reduce__(self) -> tuple[object, ...]:
        return (IntegerTensorValue.flat, (self.shape, self.integers))

    def __copy__(self) -> IntegerTensorValue:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> IntegerTensorValue:
        return self


IntegerTensor = IntegerTensorValue | tuple[object, ...]
"""What an integer tensor input takes: the value, or nested int tuples that
``INTEGER_TENSOR`` snapshots into one."""


def integers(values: object) -> tuple[int, ...]:
    """Every integer of an integer tensor, row-major."""
    return _value(values).integers


def row_major(values: object) -> Sequence[int] | npt.NDArray[np.int64]:
    """Every integer of an integer tensor, row-major, as a memory image packs them
    (``IntegerTensorValue.row_major``)."""
    return _value(values).row_major


def integer_shape(values: object) -> tuple[int, ...] | None:
    """The shape of an integer tensor (a value's, stated); ``None`` for anything that is
    no integer tensor."""
    if type(values) is IntegerTensorValue:
        return values.shape
    return _shape(values)


def integer_range(values: object) -> tuple[int, int]:
    """The least and the greatest integer of an integer tensor (a value's, stated)."""
    return _value(values).range


def _value(values: object) -> IntegerTensorValue:
    return values if type(values) is IntegerTensorValue else IntegerTensorValue.of(values)


def _recognized(value: object) -> bool:
    return type(value) is IntegerTensorValue or (type(value) is tuple and _shape(value) is not None)


def _snapshot(value: IntegerTensor) -> IntegerTensor:
    return _value(value)


INTEGER_TENSOR: ValueSemantics[IntegerTensor] = ValueSemantics(
    IntegerTensor,
    "integer tensor",
    _recognized,
    lambda left, right: left is right or _value(left) == _value(right),
    _snapshot,
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
    "INTEGER_DIGEST",
    "INTEGER_TENSOR",
    "INTEGER_VECTOR",
    "IntegerTensor",
    "IntegerTensorValue",
    "IntegerVector",
    "QONNX_DATATYPE_VALUE_SEMANTICS",
    "THRESHOLD_TABLE",
    "ThresholdTable",
    "integer_bytes",
    "integer_digest",
    "integer_range",
    "integer_shape",
    "integers",
    "row_major",
]
