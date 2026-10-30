# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compose existing native VPC and FIFO kernels, then emit real RTL without cycles.

Run from the checkout: uv run python examples/untimed/native.py
Set FINN_RESOURCES_FINNLIB to your matching library checkout for offline emission.
"""

from common import PLATFORM, configured, emit

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.composer import Facts, PartitionDeclaration, Port, channel
from finn.kernels.fifo import FifoKernel
from finn.kernels.partition import Partition
from finn.kernels.sides import BUILD, Outside, Side
from finn.kernels.vpc import VpcKernel

TENSOR = Tensor((2, 6), ScalarEncoding(resolve_qonnx_datatype_name("INT8")))
TWO = BeatSequence(vector_major(TENSOR.shape, 2))
THREE = BeatSequence(vector_major(TENSOR.shape, 3))

# Real native authoring lives in src/finn/kernels/vpc.py and fifo.py. Each owns
# ordinary Param/Decision and PortModel children; the parent names real ports.
DECLARATION = PartitionDeclaration(
    {
        "convert": Facts(VpcKernel, {"input_tensor": TENSOR, "arriving": TWO, "leaving": THREE}),
        "buffer": Facts(
            FifoKernel,
            {
                "input_tensor": TENSOR,
                "arriving": THREE,
                "depth": 4,
                "platform": PLATFORM,
            },
        ),
    },
    {
        "x": channel(
            Side.boundary("s_axis_0", outside=Outside(BUILD)),
            Port("convert", "input"),
            tensor=TENSOR,
            platform=PLATFORM,
        ),
        "middle": channel(
            Port("convert", "output"), Port("buffer", "input"), tensor=TENSOR, platform=PLATFORM
        ),
        "y": channel(
            Port("buffer", "output"), Side.boundary("m_axis_0"), tensor=TENSOR, platform=PLATFORM
        ),
    },
)


def network() -> Partition:
    # Explicit physical capacity, not derived from a rate or claimed sufficient.
    return configured(DECLARATION, {("buffer", "ram_style"): "distributed"})


if __name__ == "__main__":
    emit(network(), "native")
