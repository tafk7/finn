# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compose a real known-weight MatMul, source memory, FIFO and required adapters.

Run from the checkout: uv run python examples/untimed/compute.py
No kernel reference computation, cycle query, FIFO-sizing strategy or hardware run.
"""

from common import PLATFORM, configured, emit

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.composer import Facts, PartitionDeclaration, Port, Stated, channel
from finn.kernels.matmul import KnownWeights, MatMulKernel
from finn.kernels.partition import Partition
from finn.kernels.sides import BUILD, ROW_MAJOR, Outside, Side
from finn.kernels.values.semantics import IntegerTensorValue

DTYPE = resolve_qonnx_datatype_name("INT3")
INPUT = Tensor((2, 4), ScalarEncoding(DTYPE))
WEIGHTS = KnownWeights(
    IntegerTensorValue.of(((1, -1, 2, 0), (0, 2, -1, 1), (1, 0, 1, -2), (-1, 1, 0, 2))),
    DTYPE,
)

# Actual compute authoring is MatMulKernel -> PackedDotpKernel with included
# ControlModel and child Indexed/Scheduled Ports. Result/weight facts stay owned
# by MatMul; the composer reads Stated values rather than recreating them.
DECLARATION = PartitionDeclaration(
    {"matmul": Facts(MatMulKernel, {"x_tensor": INPUT, "weights": WEIGHTS, "platform": PLATFORM})},
    {
        "x": channel(
            Side.boundary("s_axis_0", outside=Outside(BUILD, order=ROW_MAJOR, passes=1)),
            Port("matmul", "x"),
            tensor=INPUT,
            platform=PLATFORM,
        ),
        "w": channel(
            Side.source(),
            Port("matmul", "w"),
            tensor=Stated("matmul", "carried_weights"),
            contents=Stated("matmul", "stored_weights"),
            platform=PLATFORM,
        ),
        "y": channel(
            Port("matmul", "y"),
            Side.boundary("m_axis_0"),
            tensor=Stated("matmul", "result_tensor"),
            platform=PLATFORM,
        ),
    },
)


def network() -> Partition:
    # An explicit build policy: existing packed hardware, fixed folding and FIFO
    # capacity. Channels' existing FirstValid policies select registered recipes.
    return configured(
        DECLARATION,
        {
            ("matmul", "compute"): "packed",
            ("matmul", "pe"): 2,
            ("matmul", "simd"): 2,
            ("matmul", "compute.packed.compute_pumping"): False,
            ("matmul", "compute.packed.reducer"): "tree",
            ("w", "source.ram_style"): "distributed",
            ("w", "source.pumped_memory"): False,
            ("x", "transport"): "fifo",
            ("x", "transport.fifo.depth"): 16,
        },
    )


if __name__ == "__main__":
    emit(network(), "compute")
