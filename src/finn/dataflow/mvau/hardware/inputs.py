# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What each MVAU physical Kernel is allowed to read.

A physical Kernel reads nothing it was not handed.  These bundles are what the
decomposed assembly wires in, and they are deliberately narrow: the Region and
computation declarations it covers, the folding those were built from, and the
target facts its coverage depends on.

Notably absent is anything that would let a Kernel re-decide the logical
dataflow.  ``PE`` and ``SIMD`` arrive as handles to decisions the Region
declaration already owns, so a Kernel consumes the fold and cannot choose a
conflicting one.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.authoring.scope import Ref
from finn.dataflow.hardware import ComputationContract
from finn.dataflow.mvau_problem import MVAUDspBlock
from finn.dataflow.region import DataflowRegion, NumericElementType


@dataclass(frozen=True)
class DotProductHardwareInputs:
    """What a dot-product physical Kernel may read."""

    #: The Region declaration this Kernel covers, and what it must compute.
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    #: The folding the Region was built from, imported rather than chosen.
    pe: Ref[int]
    simd: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    narrow_weights: Ref[bool]
    #: Target facts.  Coverage is a question about a board, so it needs one.
    target_dsp_block: Ref[MVAUDspBlock]
    target_clock_period_ns: Ref[float]


@dataclass(frozen=True)
class ActivationReplayHardwareInputs:
    """What an activation-replay physical Kernel may read.

    Smaller than the dot product's because the buffer has no arithmetic and no
    target coverage: it is width, depth, and repetition, all of them restatements
    of the folding it has to feed.
    """

    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    pe: Ref[int]
    simd: Ref[int]
    activation_element_type: Ref[NumericElementType]


__all__ = ["ActivationReplayHardwareInputs", "DotProductHardwareInputs"]
