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
from finn.dataflow.design.region import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.declarations import Case, Decision, Input, divisors_of
from finn.dataflow.model.design import (
    Boundary,
    Connection,
    DataflowDesign,
    Kernels,
    Sink,
)
from finn.dataflow.model.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.model.replay_buffer import ReplayBufferKernel


class DotProductDesign(DataflowDesign):
    """Matrix-vector arithmetic decomposed into replay and dot product."""

    id = "dot_product"
    version = "1"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    narrow_weights = Input(bool)
    target_dsp = Input(DspBlock)
    clock_period_ns = Input(float)

    #: Owned here because each of them changes both Regions and their edge.
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))

    replay = Kernels(
        Case(
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
        Case(
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
DESIGN_INPUTS = (
    "repetitions",
    "matrix_width",
    "matrix_height",
    "activation_type",
    "weight_type",
    "accumulator_type",
    "output_type",
    "narrow_weights",
    "target_dsp",
    "clock_period_ns",
)

__all__ = ["DESIGN_INPUTS", "DotProductDesign"]
