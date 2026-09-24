# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatype admission declarations with explicit family and bit-bound dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from finn.kernels.datatypes.values import QONNXDataType, qonnx_datatype_width
from finn.kernels.space import Constraint, Rejected, ValueRef, constraint, reject

BitBound = int | ValueRef[int]


class DatatypeDomain(Protocol):
    def constraints(
        self, datatype: ValueRef[QONNXDataType]
    ) -> tuple[tuple[str, Constraint], ...]: ...


def _bit_bound(datatype: ValueRef[QONNXDataType], bound: BitBound, *, minimum: bool) -> Constraint:
    def check(dtype: QONNXDataType, limit: int) -> bool | Rejected:
        if type(limit) is not int or limit < 1:
            return reject("dtype-bound-invalid", "a storage-bit bound must be a positive integer")
        actual = qonnx_datatype_width(dtype)
        if (actual < limit) if minimum else (actual > limit):
            relation = "at least" if minimum else "at most"
            return reject(
                "dtype-minimum-bits" if minimum else "dtype-maximum-bits",
                f"{dtype.name} uses {actual} bits; this interface requires {relation} {limit}",
                values={"datatype": dtype.name, "actual_bits": actual, "bound": limit},
            )
        return True

    if isinstance(bound, ValueRef):

        @constraint(datatype=datatype, bound=bound)
        def dynamic(*, datatype: QONNXDataType, bound: int) -> bool | Rejected:
            return check(datatype, bound)

        return dynamic

    @constraint(datatype=datatype)
    def fixed(*, datatype: QONNXDataType) -> bool | Rejected:
        return check(datatype, bound)

    return fixed


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

    def constraints(self, datatype: ValueRef[QONNXDataType]) -> tuple[tuple[str, Constraint], ...]:
        @constraint(datatype=datatype)
        def family(*, datatype: QONNXDataType) -> bool | Rejected:
            name = datatype.name
            ordinary = name == "BINARY" or re.fullmatch(r"U?INT-?\d+", name) is not None
            is_signed = name.startswith("INT")
            if not ordinary or (self.signed is not None and is_signed != self.signed):
                expected = {None: "INT/UINT", True: "signed INT", False: "unsigned UINT"}[
                    self.signed
                ]
                return reject(
                    "dtype-family",
                    f"{name} is outside this interface's {expected} datatype domain",
                    values={"datatype": name, "expected": expected},
                )
            return True

        return (
            ("family", family),
            ("minimum_bits", _bit_bound(datatype, self.min_bits, minimum=True)),
            *(
                (("maximum_bits", _bit_bound(datatype, self.max_bits, minimum=False)),)
                if self.max_bits is not None
                else ()
            ),
        )


class SignedInteger(Integer):
    def __init__(self, min_bits: BitBound = 1, max_bits: BitBound | None = None) -> None:
        super().__init__(min_bits, max_bits, signed=True)


__all__ = ["DatatypeDomain", "Integer", "SignedInteger"]
