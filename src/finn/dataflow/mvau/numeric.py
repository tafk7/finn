# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The complete numeric signature of one MVAU computation.

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

__all__ = ["MVAUNumericTypes"]


@dataclass(frozen=True)
class MVAUNumericTypes:
    """Every numeric role in one MVAU computation.

    Complete QONNX datatypes, never cached widths or signedness flags: those
    are *queries* on a datatype, and caching them here would rebuild the
    reduced family-and-width pair this migration removed.

    The threshold role belongs to the fused-threshold signature, when that
    computation is implemented; this one is the accumulator-output form.
    """

    activation: NumericElementType
    weight: NumericElementType
    accumulator: NumericElementType
    output: NumericElementType
