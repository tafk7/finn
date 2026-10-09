# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integer datatype policy: what a port's hardware or a kernel's arithmetic takes.

A policy is plain data. A port states the one its hardware takes (``admits``);
a kernel checks one in a constraint of its ``admission``. ``admit_element`` is
the same refusal for an element: the ``ScalarEncoding`` of a datatype and a
range, or why there is none.

Two encodings follow from a range of values alone. ``range_dtype`` is the
range's smallest integer encoding, ``UINT`` when no value is negative: a
result's (MatMul's, and each dot-product core's accumulator). ``stored_element``
is the element a known value's owner states for the channel that carries it,
narrowed from the type the graph gives it to what the value needs: weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.core.space import Rejected, reject
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    is_ordinary_integer,
    ordinary_integer_bounds,
    qonnx_datatype_width,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.tensor import ScalarEncoding


def set_index_dtype(sets: int) -> QONNXDataType:
    """FinnLib's set selector: ``SET_BITS = SETS > 2 ? $clog2(SETS) : 1`` unsigned bits."""
    return resolve_qonnx_datatype_name(f"UINT{(sets - 1).bit_length() if sets > 2 else 1}")


def _width(least: int, greatest: int, signed: bool) -> int:
    """The bits an encoding of this signedness needs for every integer of the range."""
    if signed:
        return max(greatest.bit_length(), (~least).bit_length() if least < 0 else 0) + 1
    if least < 0:
        raise ValueError(f"[{least}, {greatest}] has negative values: no UINT holds it")
    return max(1, greatest.bit_length())


def _integer_dtype(bits: int, signed: bool) -> QONNXDataType:
    return resolve_qonnx_datatype_name(f"{'INT' if signed else 'UINT'}{bits}")


def range_dtype(least: int, greatest: int) -> QONNXDataType:
    """The smallest integer encoding of every integer in ``[least, greatest]``: ``UINT<b>``
    (``BINARY`` for one bit) when no value is negative, otherwise ``INT<b>``."""
    if least > greatest:
        raise ValueError(f"[{least}, {greatest}] is an empty range")
    signed = least < 0
    return _integer_dtype(_width(least, greatest, signed), signed)


#: The narrowest weights FinnLib's dot-product cores read: two signed bits
#: (``dotp``'s ``WEIGHT_WIDTH >= 2``; ``dotp_8sx9_dsp58`` sign-extends any width).
STORED_MIN_BITS = 2


def stored_element(declared: QONNXDataType, value_range: tuple[int, int]) -> ScalarEncoding:
    """The element a known value's owner states for the channel carrying it: over the
    value's range, in the narrowest encoding of ``declared``'s signedness that holds it,
    at least ``STORED_MIN_BITS`` and at most ``declared``'s width.

    ``declared`` is the graph's type of the value (an ordinary integer type that holds
    the range); the value narrows it, as FINN's ``minimize_weight_bit_width`` does, so
    INT8-typed ternary weights are ``INT2 over [-1, 1]``: the source stores them at two
    bits a weight, and a core reads two (its lanes, ``NARROW_WEIGHTS``). It keeps the
    signedness, and two bits, where FINN takes ``UINT`` for non-negative weights and
    one bit for ``{-1, 0}``: the cores read signed weights of two bits or more, and the
    channel does not re-encode between its source and its consumer.
    """
    least, greatest = value_range
    declared = canonical_qonnx_datatype(declared)
    low, high = ordinary_integer_bounds(declared)
    if not low <= least <= greatest <= high:
        raise ValueError(f"[{least}, {greatest}] is not a range of {declared.name}")
    signed = declared.signed()
    width = qonnx_datatype_width(declared)
    bits = min(width, max(STORED_MIN_BITS, _width(least, greatest, signed)))
    return ScalarEncoding(_integer_dtype(bits, signed), value_range)


def values_within(inner: ScalarEncoding, outer: ScalarEncoding) -> bool:
    """Every value of ``inner`` is a value of ``outer``: ``fits``, or two integer encodings
    (a stored value's narrowed one in its graph type's, say) with ``inner``'s range
    within ``outer``'s."""
    if inner.fits(outer):
        return True
    mine, theirs = inner.value_range, outer.value_range
    if mine is None or theirs is None:
        return False
    return theirs[0] <= mine[0] and mine[1] <= theirs[1]


def admit_element(
    dtype: QONNXDataType, value_range: tuple[int, int] | None = None
) -> ScalarEncoding | Rejected:
    """The element, or a ``dtype-storage`` refusal (a zero width, a range it cannot hold)."""
    try:
        return ScalarEncoding(dtype, value_range)
    except ValueError as error:
        return reject("dtype-storage", str(error))


@dataclass(frozen=True)
class Integer:
    """Ordinary INT/UINT encodings, including canonical BINARY, with bit bounds."""

    min_bits: int = 1
    max_bits: int | None = None
    signed: bool | None = None

    def __post_init__(self) -> None:
        for bound in (self.min_bits, self.max_bits):
            if bound is not None and (type(bound) is not int or bound < 1):
                raise ValueError("integer bit bounds must be positive integers")
        if self.max_bits is not None and self.min_bits > self.max_bits:
            raise ValueError("minimum bit width exceeds maximum bit width")
        if self.signed is not None and type(self.signed) is not bool:
            raise TypeError("signed must be True, False or None")

    def check(self, dtype: QONNXDataType) -> bool | Rejected:
        """True when ``dtype`` is in the policy; otherwise the refusal says why."""
        if not is_ordinary_integer(dtype) or (
            self.signed is not None and dtype.name.startswith("INT") != self.signed
        ):
            expected = {None: "INT/UINT", True: "signed INT", False: "unsigned UINT"}[self.signed]
            return reject(
                "dtype-family",
                f"{dtype.name} is outside the {expected} datatype domain",
                values={"datatype": dtype.name, "expected": expected},
            )
        actual = qonnx_datatype_width(dtype)
        for limit, minimum in ((self.min_bits, True), (self.max_bits, False)):
            if limit is not None and ((actual < limit) if minimum else (actual > limit)):
                relation = "at least" if minimum else "at most"
                return reject(
                    "dtype-minimum-bits" if minimum else "dtype-maximum-bits",
                    f"{dtype.name} uses {actual} bits; this encoding requires {relation} {limit}",
                    values={"datatype": dtype.name, "actual_bits": actual, "bound": limit},
                )
        return True


__all__ = [
    "Integer",
    "admit_element",
    "range_dtype",
    "set_index_dtype",
    "stored_element",
    "values_within",
]
