# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Value semantics for scalar datatypes and immutable arrays."""

from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import cast

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

IntegerTensor = tuple[object, ...]


def _shape(value: object) -> tuple[int, ...] | None:
    """The shape of a nonempty rectangular nest of tuples with int leaves, rank at
    least one; ``None`` for anything else."""

    def shape(item: object) -> tuple[int, ...] | None:
        if type(item) is int:
            return ()
        if type(item) not in (tuple, IntegerTensorValue) or not item:
            return None
        inner = {shape(element) for element in cast(tuple[object, ...], item)}
        if len(inner) != 1 or None in inner:
            return None
        (common,) = inner
        return None if common is None else (len(cast(tuple[object, ...], item)), *common)

    found = shape(value)
    return found if found else None


class IntegerTensorValue(tuple[object, ...]):
    """An integer tensor as a value: the nested tuples, its shape checked once when
    it is made, its integers and their range computed once, when first read.

    ``INTEGER_TENSOR`` snapshots a tensor into one, so every later recognition is
    a type check: a weights value is walked once per value, not once per
    configuration of every point that holds it. Immutable like the tuples it
    wraps; a copy is the value itself.
    """

    shape: tuple[int, ...]

    def __new__(cls, values: object) -> IntegerTensorValue:
        shape = _shape(values)
        if shape is None:
            raise TypeError("an integer tensor is a nonempty rectangular nest of int tuples")
        made = super().__new__(cls, cast(tuple[object, ...], values))
        made.__dict__["shape"] = shape
        return made

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"an integer tensor value is immutable; cannot set {name}")

    def __reduce__(self) -> tuple[type[IntegerTensorValue], tuple[tuple[object, ...]]]:
        return (IntegerTensorValue, (tuple(self),))

    def __copy__(self) -> IntegerTensorValue:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> IntegerTensorValue:
        return self

    @cached_property
    def integers(self) -> tuple[int, ...]:
        """Every integer, in order: the leading axes flattened, one level at a time."""
        values: tuple[object, ...] = self
        for _ in self.shape[1:]:
            values = tuple(chain.from_iterable(cast(tuple[tuple[object, ...], ...], values)))
        return cast(tuple[int, ...], tuple(values))

    @cached_property
    def range(self) -> tuple[int, int]:
        """The least and the greatest integer."""
        values = self.integers
        return min(values), max(values)


def _integers(values: object) -> tuple[int, ...]:
    if type(values) is int:
        return (values,)
    assert isinstance(values, tuple)
    return tuple(leaf for item in values for leaf in _integers(item))


def integers(values: object) -> tuple[int, ...]:
    """Every integer of a nested operand, in order (an integer tensor value's, once)."""
    if type(values) is IntegerTensorValue:
        return values.integers
    return _integers(values)


def integer_range(values: object) -> tuple[int, int]:
    """The least and the greatest integer of a nested operand (an integer tensor
    value's, once)."""
    if type(values) is IntegerTensorValue:
        return values.range
    found = _integers(values)
    return min(found), max(found)


def _recognized(value: object) -> bool:
    return type(value) is IntegerTensorValue or (type(value) is tuple and _shape(value) is not None)


def _snapshot(value: IntegerTensor) -> IntegerTensor:
    return value if type(value) is IntegerTensorValue else IntegerTensorValue(value)


INTEGER_TENSOR: ValueSemantics[IntegerTensor] = ValueSemantics(
    IntegerTensor,
    "integer tensor",
    _recognized,
    lambda left, right: left is right or left == right,
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
    "INTEGER_TENSOR",
    "INTEGER_VECTOR",
    "IntegerTensor",
    "IntegerTensorValue",
    "IntegerVector",
    "QONNX_DATATYPE_VALUE_SEMANTICS",
    "THRESHOLD_TABLE",
    "ThresholdTable",
    "integer_range",
    "integers",
]
