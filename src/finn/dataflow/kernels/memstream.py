# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A local-state source that presents a parameter matrix as an ordered stream.

Decoupled supply is not "the same weights, arranged differently".  It is a
second Region in the Network -- a rank-zero source whose whole content is
available at its one schedule point -- feeding a real edge into the compute
Region's weight input.  Naming it that way is what makes the three supply modes
comparable: external streaming has a boundary where this has a node, and
embedded has neither.

The Region is exact about the order it produces, because the edge it feeds is
compared position by position against what the consumer requires.  Nothing here
knows how the matrix reaches the memory: that is physical, and this Kernel's
physical projection says so.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.kernels.kernel import (
    Kernel,
    PhysicallyUnsupported,
    RegionDeclaration,
)
from finn.dataflow.space.declarations import Input
from finn.dataflow.space.dataflow_value_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.ops.mvau.regions import construct_weight_stream_region


class MemstreamKernel(Kernel):
    """The decoupled weight supplier as one Region and, later, one module.

    It owns no Decision.  Depth, width and repetition are the Design's folding
    restated in the memory's vocabulary, exactly as the replay buffer's are, and
    a supplier that picked its own depth would be picking a fold.

    Its physical projection is explicitly unavailable at this phase.  A real
    memstream needs a data slot, an initializer written into it, a RAM style and
    the artifact stages that carry them; U3 owns the semantics and U6 owns all
    of that.  The Network placing this Kernel resolves regardless, which is the
    separation the phase exists to demonstrate.
    """

    id = "finn_rtl_memstream"
    version = "1"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    weight_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    pe = Input(int)
    simd = Input(int)

    region = RegionDeclaration(
        family="parameter.cyclic_delivery",
        version="1",
        construct=construct_weight_stream_region,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        weight_element_type=weight_type,
        pe=pe,
        simd=simd,
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        raise PhysicallyUnsupported(
            "memstream realization arrives in U6 with its data slot, loader and RAM style"
        )


__all__ = [
    "MemstreamKernel",
]
