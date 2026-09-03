# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What every MVAU Design agrees about before it disagrees about supply.

The folding and the operand facts are the same question whichever weight path a
Design takes, and stating them once has a consequence beyond tidiness: ``pe``
and ``simd`` are then *one* declaration object shared by every alternative, so
an operation persisting them names one Decision rather than one per Design.
Two declarations that had to be kept in step would be two ways for a saved
choice to land on the wrong one.
"""

from __future__ import annotations

from finn.dataflow.designs.design import DataflowDesign
from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.model.declarations import Decision, Input, divisors_of
from finn.dataflow.model.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


class WeightedDotProductDesign(DataflowDesign):
    """The shared operand facts and the two folding choices MVAU owns."""

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


#: Every Input the shared base consumes, for a caller assembling bindings.
SHARED_INPUTS = (
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

__all__ = ["SHARED_INPUTS", "WeightedDotProductDesign"]
