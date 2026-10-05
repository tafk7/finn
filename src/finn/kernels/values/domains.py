# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integer datatype policy: what a port's hardware or a kernel's arithmetic takes.

A policy is plain data. A port states the one its hardware takes (``admits``);
a kernel checks one in a constraint of its ``admission``.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.core.space import Rejected, reject
from finn.dataflow.datatypes import (
    QONNXDataType,
    is_ordinary_integer,
    qonnx_datatype_width,
    resolve_qonnx_datatype_name,
)


def set_index_dtype(sets: int) -> QONNXDataType:
    """FinnLib's set selector: ``SET_BITS = SETS > 2 ? $clog2(SETS) : 1`` unsigned bits."""
    return resolve_qonnx_datatype_name(f"UINT{(sets - 1).bit_length() if sets > 2 else 1}")


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


__all__ = ["Integer", "set_index_dtype"]
