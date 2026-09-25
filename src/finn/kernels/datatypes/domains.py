# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatype admission declarations with explicit family and bit-bound dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
import re
from typing import Protocol

from finn.kernels.datatypes.values import QONNXDataType, qonnx_datatype_width
from finn.core.space import Constraint, Domain, Rejected, ValueRef, constraint, domain, reject
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import resolve_qonnx_datatype_name

BitBound = int | ValueRef[int]


class DatatypeDomain(Protocol):
    def rebind(self, bind: Callable[[str, BitBound], BitBound]) -> DatatypeDomain:
        """Place policy dependencies in a consumer's scope, without knowing that consumer."""
        ...

    def constraints(
        self, datatype: ValueRef[QONNXDataType]
    ) -> tuple[tuple[str, Constraint], ...]: ...


def _check_bound(dtype: QONNXDataType, limit: int, *, minimum: bool) -> bool | Rejected:
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


def _check_family(dtype: QONNXDataType, signed: bool | None) -> bool | Rejected:
    ordinary = dtype.name == "BINARY" or re.fullmatch(r"U?INT-?\d+", dtype.name) is not None
    if not ordinary or (signed is not None and dtype.name.startswith("INT") != signed):
        expected = {None: "INT/UINT", True: "signed INT", False: "unsigned UINT"}[signed]
        return reject(
            "dtype-family",
            f"{dtype.name} is outside the {expected} datatype domain",
            values={"datatype": dtype.name, "expected": expected},
        )
    return True


def _bit_bound(datatype: ValueRef[QONNXDataType], bound: BitBound, *, minimum: bool) -> Constraint:

    if isinstance(bound, ValueRef):

        @constraint(datatype=datatype, bound=bound)
        def dynamic(*, datatype: QONNXDataType, bound: int) -> bool | Rejected:
            return _check_bound(datatype, bound, minimum=minimum)

        return dynamic

    @constraint(datatype=datatype)
    def fixed(*, datatype: QONNXDataType) -> bool | Rejected:
        return _check_bound(datatype, bound, minimum=minimum)

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
            return _check_family(datatype, self.signed)

        return (
            ("family", family),
            ("minimum_bits", _bit_bound(datatype, self.min_bits, minimum=True)),
            *(
                (("maximum_bits", _bit_bound(datatype, self.max_bits, minimum=False)),)
                if self.max_bits is not None
                else ()
            ),
        )

    def rebind(self, bind: Callable[[str, BitBound], BitBound]) -> Integer:
        return Integer(
            bind("minimum_bits", self.min_bits),
            None if self.max_bits is None else bind("maximum_bits", self.max_bits),
            signed=self.signed,
        )

    def check(self, dtype: QONNXDataType) -> bool | Rejected:
        """Check a concrete policy; dynamic bounds are resolved by the Space adapters."""
        if isinstance(self.min_bits, ValueRef) or isinstance(self.max_bits, ValueRef):
            raise TypeError("check requires concrete bit bounds; use constraints() or domain()")
        family = _check_family(dtype, self.signed)
        if isinstance(family, Rejected):
            return family
        minimum = _check_bound(dtype, self.min_bits, minimum=True)
        if isinstance(minimum, Rejected):
            return minimum
        return True if self.max_bits is None else _check_bound(dtype, self.max_bits, minimum=False)

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
            # Invalid supplied bounds are semantic refusals, like constraints().
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


__all__ = ["DatatypeDomain", "Integer", "SignedInteger"]
