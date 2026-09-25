# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar encoding admission independent of pins, streams, and implementation language.

``Scalar`` and its subclasses are ordinary handwritten Spaces. A kernel places
one per operand, binding its dtype and the policy's bounds like any other child
formal; each admission rule is its own constraint, so a known family refusal
remains visible while an unrelated bound is unresolved. Callers read the raw
``element_bits`` fact independently and consume the accepted ``encoding`` view.
Subclassing is the extension mechanism: a subclass adds constraints and names
them in its ``admission`` group.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.core.space import (
    ConstraintGroup,
    Param,
    Rejected,
    Space,
    Subspace,
    ValueRef,
    View,
    constraint,
    default_semantics,
    derived,
    reject,
)
from finn.kernels.datatypes.domains import Integer, check_bit_bound, check_integer_family
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import (
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

    @property
    def dtype(self) -> QONNXDataType:
        return resolve_qonnx_datatype_name(self.datatype_name)

    @property
    def bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @property
    def signed(self) -> bool:
        return self.dtype.signed()


SCALAR_ENCODING = default_semantics(ScalarEncoding)


class Scalar(Space):
    """Any positive-width QONNX encoding; subclasses add admission constraints."""

    dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)

    @derived
    def element_bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @derived(semantics=SCALAR_ENCODING)
    def candidate(self) -> ScalarEncoding | Rejected:
        try:
            return ScalarEncoding(self.dtype)
        except ValueError as error:
            return reject("dtype-storage", str(error))

    admission = ConstraintGroup()
    encoding = View(candidate, constraints=(admission,))


class Signedness(Enum):
    ANY = None
    SIGNED = True
    UNSIGNED = False


class IntegerScalar(Scalar):
    """An ordinary INT/UINT encoding with a minimum storage width."""

    signedness = Param(Signedness)
    min_bits = Param(int)

    @constraint
    def family(self) -> bool | Rejected:
        return check_integer_family(self.dtype, self.signedness.value)

    @constraint
    def minimum_bits(self) -> bool | Rejected:
        return check_bit_bound(self.dtype, self.min_bits, minimum=True)

    admission = ConstraintGroup(family, minimum_bits)


class BoundedIntegerScalar(IntegerScalar):
    """An integer encoding that additionally fits a maximum storage width."""

    max_bits = Param(int)

    @constraint
    def maximum_bits(self) -> bool | Rejected:
        return check_bit_bound(self.dtype, self.max_bits, minimum=False)

    admission = ConstraintGroup(IntegerScalar.family, IntegerScalar.minimum_bits, maximum_bits)


def integer_scalar(
    dtype: ValueRef[QONNXDataType], policy: Integer, *, when: ValueRef[bool] | None = None
) -> Subspace[IntegerScalar]:
    """Place a policy's admission; referenced bounds become ordinary child bindings."""
    signedness = Signedness(policy.signed)
    if policy.max_bits is None:
        return Subspace(
            IntegerScalar,
            when=when,
            dtype=dtype,
            signedness=signedness,
            min_bits=policy.min_bits,
        )
    return Subspace(
        BoundedIntegerScalar,
        when=when,
        dtype=dtype,
        signedness=signedness,
        min_bits=policy.min_bits,
        max_bits=policy.max_bits,
    )


__all__ = [
    "BoundedIntegerScalar",
    "IntegerScalar",
    "SCALAR_ENCODING",
    "Scalar",
    "ScalarEncoding",
    "Signedness",
    "integer_scalar",
]
