# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The complete numeric signature of one folded dot-product computation.

A physical Kernel's datatype coverage is asked about *all four* operand roles at
once, through this one value.

The reason is a defect this replaced.  Coverage used to be several independent
predicates, and the ones that existed inspected activation and weight only:

    covers_operand_types(activation, weight)

Nothing asked what the accumulator or the output were.  The only condition that
mentioned them checked that they were *equal to each other*, which an integer
dot product declaring a floating-point accumulator satisfies perfectly well.  So

    INT8 activation, INT8 weight, FLOAT16 accumulator, FLOAT16 output

was admitted with no refusals at all, lowered, and elaborated onto a multiplier
that cannot produce it.

That is not a predicate anyone got wrong; it is a shape that makes the omission
invisible.  Adding a role to a bundle is a change every Kernel's coverage
function must then account for, and forgetting one is a type error rather than
a silently unasked question.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.region import NumericElementType

__all__ = ["DotProductNumericTypes", "RoleVerdict"]


@dataclass(frozen=True)
class DotProductNumericTypes:
    """Every numeric role in one dot-product computation.

    Complete QONNX datatypes, never cached widths or signedness flags: those
    are *queries* on a datatype, and caching them here would rebuild the
    reduced family-and-width pair that would erase operation role.

    The threshold role belongs to the fused-threshold signature, when that
    computation is implemented; this one is the accumulator-output form.
    """

    activation: NumericElementType
    weight: NumericElementType
    accumulator: NumericElementType
    output: NumericElementType


@dataclass(frozen=True)
class RoleVerdict:
    """One numeric role's answer, with the reason if it is a refusal.

    Shared *vocabulary*, not a shared answer.  Every Kernel that multiplies
    reports per role rather than as one boolean, so a refusal can say which
    operand was wrong -- and two Kernels reporting in the same shape is what
    makes their answers comparable when one is checked against the other.

    What is deliberately **not** shared is the function that produces these.
    Each physical core owns its own predicate, because agreement between two
    cores is a fact to be demonstrated per release rather than a structure to
    be assumed once.
    """

    role: str
    supported: bool
    detail: str = ""
