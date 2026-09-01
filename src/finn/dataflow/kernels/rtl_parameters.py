# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The three RTL parameter values every MVU-family core needs derived.

``VERSION``, ``SIGNED_ACTIVATIONS`` and ``SEGMENTLEN`` are not projected facts
and nobody chooses them: they follow from the target family, the activation
type, and the clock against a timing model.  They used to be computed inside
the elaborator, which meant a value reached an artifact without ever appearing
in the design point -- the disallowed fifth ownership category.

They live here, once, so that every Kernel that needs them *declares* them from
the same implementation.  A Kernel imports these; an elaborator reads the
declared property.  Nothing computes them a second time.
"""

from __future__ import annotations

from math import ceil, floor

from finn.dataflow.authoring.scope import unresolved
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.region import NumericElementType

#: The DSP generation each target family selects in the RTL.
DSP_VERSION = {
    DspBlock.DSP48E1: 1,
    DspBlock.DSP48E2: 2,
    DspBlock.DSP58: 3,
}

#: Per-DSP delay terms behind the segment-length derivation, in nanoseconds.
#: Named constants because they are a timing model, not magic numbers.
SEGMENT_BASE_DELAY_NS = 0.741
SEGMENT_STAGE_DELAY_NS = 0.605


def dsp_version(target: DspBlock) -> object:
    """``VERSION``: which DSP generation the core should instantiate."""

    return DSP_VERSION[target]


def signed_activations(activation: NumericElementType) -> object:
    """``SIGNED_ACTIVATIONS``: whether the activation operand carries a sign.

    Asked of the datatype directly.  This used to read a reduced family label
    and compare it to ``"int"``, which happened to give the right answer only
    because the label conflated "signed" with "integer"; the parameter is about
    signedness, so it asks about signedness.

    Reaching here at all means coverage already established the operand is a
    two's-complement integer, so ``signed()`` is the sign of that encoding
    rather than the looser "can represent negatives" it means for, say,
    ``BIPOLAR``.
    """

    return bool(activation.signed())


def segment_length(clock_period_ns: float, pumping: bool, simd: int) -> object:
    """``SEGMENTLEN``: the DSP cascade length the target clock can carry.

    Preserved verbatim from the elaborator it was taken out of, because the
    point of declaring it is to fix its *ownership*, not its value.  Do not
    replace it with ``SEGMENTLEN = 0``: zero means ``SEGLEN = CHAINLEN``, the
    longest cascade, which discards exactly the clock-driven shortening this
    computes and quietly loses timing coverage at fast clocks.
    """

    reference_clock = clock_period_ns / 2 if pumping else clock_period_ns
    if reference_clock <= SEGMENT_BASE_DELAY_NS:
        return unresolved(
            "mvau-segment-length-clock-infeasible",
            "the target clock period is below the covered RTL segment-delay bound",
            values={"reference_clock_ns": reference_clock},
        )
    covered_stages = floor((reference_clock - SEGMENT_BASE_DELAY_NS) / SEGMENT_STAGE_DELAY_NS + 1)
    longest_chain = ceil(simd / (6 if pumping else 3))
    return min(covered_stages, longest_chain)


__all__ = [
    "DSP_VERSION",
    "SEGMENT_BASE_DELAY_NS",
    "SEGMENT_STAGE_DELAY_NS",
    "dsp_version",
    "segment_length",
    "signed_activations",
]
