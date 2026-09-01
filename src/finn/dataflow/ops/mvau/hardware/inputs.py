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
from finn.dataflow.ops.mvau.problem import MVAUDspBlock
from finn.dataflow.network import DataflowNetwork
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


@dataclass(frozen=True)
class FusedMatrixVectorHardwareInputs:
    """What a Kernel covering *both* MVAU Regions may read.

    Not the union of the other two bundles, and the differences are the point.

    It carries a ``network`` handle, which neither of the others does. A Kernel
    that absorbs the edge between two Regions has to name the Network that edge
    lives in, so binding can check the connection really runs the way the
    Kernel claims rather than trusting an id.

    It carries the matrix geometry, which ``DotProductHardwareInputs`` does
    not. ``MW`` and ``MH`` size the replay this core contains; the decomposed
    dot product needs neither, because the replay it feeds from is a separate
    Kernel that derives its own ``LEN`` and ``REP``. That difference in the
    input bundle is the fusion, before a single parameter is declared.
    """

    replay_region: Ref[DataflowRegion]
    replay_computation: Ref[ComputationContract]
    compute_region: Ref[DataflowRegion]
    compute_computation: Ref[ComputationContract]
    #: The Network the absorbed activation edge runs in.
    network: Ref[DataflowNetwork]
    #: The geometry the internal replay is sized from.
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    #: The folding, imported rather than chosen -- as for every physical Kernel.
    pe: Ref[int]
    simd: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    narrow_weights: Ref[bool]
    target_dsp_block: Ref[MVAUDspBlock]
    target_clock_period_ns: Ref[float]


__all__ = [
    "ActivationReplayHardwareInputs",
    "DotProductHardwareInputs",
    "FusedMatrixVectorHardwareInputs",
]
