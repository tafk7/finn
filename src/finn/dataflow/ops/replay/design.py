# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One segment, one Kernel, two boundaries.

The smallest Design that is still a Design.  Its value here is negative: it has
no SubspaceChoice, so nothing in the operation layer may assume a selector
exists; it has one node, so nothing may assume an edge; and it has no weight
path, so nothing may assume a matrix.
"""

from __future__ import annotations

from finn.dataflow.designs.design import NetworkBoundary, DataflowDesign, KernelChoice
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.space.declarations import Decision, Input, Subspace, divisors_of
from finn.dataflow.space.dataflow_value_semantics import QONNX_DATATYPE_VALUE_SEMANTICS


class ActivationReplayDesign(DataflowDesign):
    """Present each activation row once per neuron fold, and nothing else."""

    id = "activation_replay"
    version = "1"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)

    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))

    replay = KernelChoice(
        Subspace(
            ReplayBufferKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            pe=pe,
            simd=simd,
        ),
    )

    activation = NetworkBoundary(replay.input("activation_in"))
    expanded = NetworkBoundary(replay.output("activation_out"))


DESIGN_INPUTS = ("repetitions", "matrix_width", "matrix_height", "activation_type")

__all__ = ["DESIGN_INPUTS", "ActivationReplayDesign"]
