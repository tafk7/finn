# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The real two-Kernel Design: activation replay feeding a folded dot product.

```text
activation boundary -> replay -> activation_replay -> compute -> output boundary
                                                  weight boundary -^
```

The Design owns PE and SIMD once.  They change the replay Region, the
dot-product Region, and the beat contract on the edge between them, so no single
Kernel can own them and no export or equality constraint has to connect the two.
Both Kernels consume the same engine decision handles as ordinary Inputs.

DotpAxi keeps pumping and its own physical feasibility.  Each Kernel declares
exactly one Region.  The Design constructs neither, and calls no historical
Region or Network helper: the Network it publishes is generated from the exact
selected Regions and the topology declared here.
"""

from __future__ import annotations

from finn.dataflow.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.model.declarations import Subspace
from finn.dataflow.designs.design import (
    Boundary,
    Connection,
    Kernels,
    Sink,
)
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.ops.mvau.designs.base import SHARED_INPUTS, WeightedDotProductDesign


class DotProductDesign(WeightedDotProductDesign):
    """Matrix-vector arithmetic decomposed into replay and dot product.

    The matrix arrives from outside, always.  ``SuppliedDotProductDesign`` is
    the same composition with the weight path as a declared choice; this one is
    the simplest thing that works and stays the reference the composed hardware
    evidence is written against.
    """

    id = "dot_product"
    version = "1"

    repetitions = WeightedDotProductDesign.repetitions
    matrix_width = WeightedDotProductDesign.matrix_width
    matrix_height = WeightedDotProductDesign.matrix_height
    activation_type = WeightedDotProductDesign.activation_type
    weight_type = WeightedDotProductDesign.weight_type
    accumulator_type = WeightedDotProductDesign.accumulator_type
    output_type = WeightedDotProductDesign.output_type
    narrow_weights = WeightedDotProductDesign.narrow_weights
    target_dsp = WeightedDotProductDesign.target_dsp
    clock_period_ns = WeightedDotProductDesign.clock_period_ns
    pe = WeightedDotProductDesign.pe
    simd = WeightedDotProductDesign.simd

    replay = Kernels(
        Subspace(
            ReplayBufferKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            pe=pe,
            simd=simd,
        ),
        computation=ACTIVATION_REPLAY_COMPUTATION,
    )

    compute = Kernels(
        Subspace(
            DotpAxiKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            weight_type=weight_type,
            accumulator_type=accumulator_type,
            output_type=output_type,
            narrow_weights=narrow_weights,
            target_dsp=target_dsp,
            clock_period_ns=clock_period_ns,
            pe=pe,
            simd=simd,
        ),
        computation=DOT_PRODUCT_COMPUTATION,
    )

    activation_replay = Connection(
        replay.output("activation_out"),
        Sink(compute.input("activation")),
    )

    activation = Boundary(replay.input("activation_in"))
    weight = Boundary(compute.input("weight"))
    output = Boundary(compute.output("output"))


#: Every Input the Design consumes, for a caller assembling the bindings.
DESIGN_INPUTS = SHARED_INPUTS

__all__ = ["DESIGN_INPUTS", "DotProductDesign"]
