# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How many weights fit in one DSP A port, and when they do not fit at all.

Both MVU-family cores pack several weights into the single ``A`` operand of a
DSP slice and recover the products from disjoint slices of ``P``.  The packing
is what makes ``SIMD`` cheap, and it is also the thing that silently fails: if
the requested weight width leaves no room for the guard bits, the core's own
``sliceLanes()`` computes a negative slack and terminates elaboration.

This is a model of *that* calculation, and it exists because the coverage rule
it replaced was a guess.  ``DotpAxiKernel`` and ``MvuVvuAxiKernel`` both used to
say "DSP48E1 requires the narrow-weight promise", which is wrong in both
directions:

- it **admitted** 27-bit non-narrow weights on DSP48E2 and DSP58, which reach
  the generic core, compute ``bit_slack = -1``, and stop at ``mvu.sv:113``
  with "Cannot accommodate 27-bit non-narrow weights"; and
- it **refused** every non-narrow configuration on DSP48E1, including 8-bit
  weights, which pack perfectly well -- three lanes with a bit to spare -- and
  which baseline FINN builds routinely.

The real rule is the one the RTL computes, so it is computed here rather than
summarized.  Deriving it also makes the answer independent of which DSP family
the target happens to be, which is what the old rule was really reaching for.

**This is shared on purpose, and it is a different kind of sharing from the
datatype predicates.**  Those stay per Kernel because "which datatypes will
this core multiply" is a property of the core.  Lane packing is a property of
the *DSP slice and the packing algorithm both cores implement* -- the two
sources are line-for-line the same computation, which
``test_the_lane_calculation_matches_both_rtl_sources`` pins by reading them.
A second implementation here would be a second place to get it wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DspBlock(str, Enum):
    """Target DSP generation available to a multiply-accumulate Kernel."""

    __dataflow_identity_token__ = "finn.dataflow.mvau_problem.MVAUDspBlock"

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


__all__ = [
    "DspBlock",
    "LanePacking",
    "a_datapath_width",
    "b_datapath_width",
    "p_datapath_width",
    "pack_lanes",
]


def a_datapath_width(version: int) -> int:
    """``A_WIDTH``: the multiplier operand port the weights are packed into.

    ``25 + 2*(VERSION > 1)`` -- ``mvu.sv:94``, and ``a_width()`` in FinnLib's
    ``add_multi_pkg.sv``.
    """

    return 25 + 2 * (version > 1)


def b_datapath_width(version: int) -> int:
    """``B_WIDTH``: the port the activation is presented on.  ``mvu.sv:95``."""

    return 18 + 6 * (version > 2)


def p_datapath_width(version: int) -> int:
    """``P_WIDTH``: the accumulation path.  ``mvu.sv:96``."""

    return 58 if version == 3 else 48


@dataclass(frozen=True)
class LanePacking:
    """The outcome of ``sliceLanes()`` for one weight and activation width."""

    #: ``NUM_LANES``: how many weights share the A port.
    lanes: int
    #: ``bit_slack`` after reserving the minimum width for every lane.  The
    #: core refuses to elaborate when this is negative.
    slack: int

    @property
    def fits(self) -> bool:
        return self.slack >= 0


def pack_lanes(
    *, a_width: int, weight_width: int, activation_width: int, narrow_weights: bool
) -> LanePacking:
    """Reproduce ``mvu.sv``'s lane slicing, up to the point that can fail.

    Transcribed from ``mvu.sv:113`` and ``:132-135``, which FinnLib's
    ``dotp.sv:90`` and ``:109-111`` repeat verbatim::

        MIN_LANE_WIDTH = WEIGHT_WIDTH + ACTIVATION_WIDTH - 1
        NUM_LANES      = A_WIDTH == WEIGHT_WIDTH ? 1
                       : 1 + (A_WIDTH - !NARROW_WEIGHTS - WEIGHT_WIDTH) / MIN_LANE_WIDTH
        bit_slack      = A_WIDTH - !NARROW_WEIGHTS - WEIGHT_WIDTH
                       - (NUM_LANES - 1) * MIN_LANE_WIDTH

    The ``!NARROW_WEIGHTS`` term is the whole narrow-weight question: a weight
    that may take its type's minimum needs one extra bit to keep its sign from
    being consumed by the neighbouring lane.  A narrow weight promises it never
    takes that value, so the bit is free.

    Only the slack sign is modelled.  The distribution loop after it cannot
    fail, and reproducing it would be a second copy of arithmetic nothing here
    asks about.

    The caller must have established ``weight_width <= a_width`` already --
    ``mvu.sv:107`` refuses beyond that, and the subtraction above is unsigned
    in SystemVerilog, so a wider weight wraps rather than going negative.
    """

    if weight_width > a_width:
        raise ValueError(
            f"a {weight_width}-bit weight exceeds the {a_width}-bit A datapath; "
            "the width envelope must be checked before the packing"
        )
    sign_bit = 0 if narrow_weights else 1
    min_lane_width = weight_width + activation_width - 1
    lanes = (
        1 if a_width == weight_width else 1 + (a_width - sign_bit - weight_width) // min_lane_width
    )
    slack = a_width - sign_bit - weight_width - (lanes - 1) * min_lane_width
    return LanePacking(lanes, slack)
