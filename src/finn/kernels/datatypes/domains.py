# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integer datatype policy: shared predicates, concrete checks and decision domains.

A policy is plain data. Supplied dtypes are admitted by the ``IntegerScalar``
Space in ``datatypes.scalar``, whose bound Params receive a policy's bounds as
ordinary bindings; owned dtype choices use ``Integer.domain()``. Both apply the
same predicates, so a policy never needs to know the scope that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import re

from finn.dataflow.datatypes import QONNXDataType, qonnx_datatype_width
from finn.core.space import Domain, Rejected, ValueRef, domain, reject
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.datatypes import resolve_qonnx_datatype_name

# A literal bound, or a reference to one (references are typed as their values).
BitBound = int


def set_index_dtype(sets: int) -> QONNXDataType:
    """FinnLib's set selector: ``SET_BITS = SETS > 2 ? $clog2(SETS) : 1`` unsigned bits."""
    return resolve_qonnx_datatype_name(f"UINT{(sets - 1).bit_length() if sets > 2 else 1}")


def check_bit_bound(dtype: QONNXDataType, limit: int, *, minimum: bool) -> bool | Rejected:
    if type(limit) is not int or limit < 1:
        return reject("dtype-bound-invalid", "a storage-bit bound must be a positive integer")
    actual = qonnx_datatype_width(dtype)
    if (actual < limit) if minimum else (actual > limit):
        relation = "at least" if minimum else "at most"
        return reject(
            "dtype-minimum-bits" if minimum else "dtype-maximum-bits",
            f"{dtype.name} uses {actual} bits; this encoding requires {relation} {limit}",
            values={"datatype": dtype.name, "actual_bits": actual, "bound": limit},
        )
    return True


def check_integer_family(dtype: QONNXDataType, signed: bool | None) -> bool | Rejected:
    ordinary = dtype.name == "BINARY" or re.fullmatch(r"U?INT-?\d+", dtype.name) is not None
    if not ordinary or (signed is not None and dtype.name.startswith("INT") != signed):
        expected = {None: "INT/UINT", True: "signed INT", False: "unsigned UINT"}[signed]
        return reject(
            "dtype-family",
            f"{dtype.name} is outside the {expected} datatype domain",
            values={"datatype": dtype.name, "expected": expected},
        )
    return True


@dataclass(frozen=True)
class Integer:
    """Ordinary INT/UINT encodings, including canonical BINARY, with bit bounds."""

    min_bits: BitBound = 1
    max_bits: BitBound | None = None
    signed: bool | None = None

    def __post_init__(self) -> None:
        for bound in (self.min_bits, self.max_bits):
            if bound is not None and not isinstance(bound, ValueRef):
                if type(bound) is not int or bound < 1:
                    raise ValueError("integer bit bounds must be positive integers or ValueRefs")
            elif isinstance(bound, ValueRef):
                semantics = bound.semantics
                if semantics is not None and semantics.type_token is not int:
                    raise TypeError("integer bit bounds require integer ValueRefs")
        if type(self.min_bits) is int and type(self.max_bits) is int:
            if self.min_bits > self.max_bits:
                raise ValueError("minimum bit width exceeds maximum bit width")
        if self.signed is not None and type(self.signed) is not bool:
            raise TypeError("signed must be True, False or None")

    def check(self, dtype: QONNXDataType) -> bool | Rejected:
        """Check a concrete policy; referenced bounds are bindings, resolved by a Space."""
        if isinstance(self.min_bits, ValueRef) or isinstance(self.max_bits, ValueRef):
            raise TypeError("check requires concrete bit bounds; use IntegerScalar or domain()")
        family = check_integer_family(dtype, self.signed)
        if isinstance(family, Rejected):
            return family
        minimum = check_bit_bound(dtype, self.min_bits, minimum=True)
        if isinstance(minimum, Rejected):
            return minimum
        return (
            True if self.max_bits is None else check_bit_bound(dtype, self.max_bits, minimum=False)
        )

    def domain(self) -> Domain[QONNXDataType]:
        """Use the same policy for owned dtype choices; bounded policies enumerate widths."""
        dependencies = {
            name: value
            for name, value in (("minimum", self.min_bits), ("maximum", self.max_bits))
            if isinstance(value, ValueRef)
        }

        def resolved(bounds: dict[str, int]) -> Integer:
            low = bounds["minimum"] if isinstance(self.min_bits, ValueRef) else self.min_bits
            high = bounds["maximum"] if isinstance(self.max_bits, ValueRef) else self.max_bits
            return Integer(low, high, signed=self.signed)

        def accepts(*, candidate: QONNXDataType, **bounds: int) -> bool | Rejected:
            # Invalid supplied bounds are semantic refusals, as in IntegerScalar.
            try:
                policy = resolved(bounds)
            except ValueError as error:
                return reject("dtype-bound-invalid", str(error))
            return policy.check(candidate)

        def candidates(**bounds: int) -> Iterable[QONNXDataType] | Rejected:
            try:
                policy = resolved(bounds)
            except ValueError as error:
                return reject("dtype-bound-invalid", str(error))
            assert type(policy.min_bits) is int and type(policy.max_bits) is int
            prefixes = (
                ("INT", "UINT") if self.signed is None else (("INT",) if self.signed else ("UINT",))
            )
            return tuple(
                resolve_qonnx_datatype_name(f"{prefix}{bits}")
                for bits in range(policy.min_bits, policy.max_bits + 1)
                for prefix in prefixes
            )

        return domain(
            accepts=accepts,
            candidates=None if self.max_bits is None else candidates,
            semantics=QONNX_DATATYPE_VALUE_SEMANTICS,
            **dependencies,
        )


class SignedInteger(Integer):
    def __init__(self, min_bits: BitBound = 1, max_bits: BitBound | None = None) -> None:
        super().__init__(min_bits, max_bits, signed=True)


__all__ = [
    "BitBound",
    "Integer",
    "SignedInteger",
    "check_bit_bound",
    "check_integer_family",
    "set_index_dtype",
]
