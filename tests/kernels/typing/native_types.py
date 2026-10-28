# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from typing import assert_type

from support.helpers import FULL_DSP48E2, in_order, stage_tensor

from finn.core.space import BoundValue, Rejected, ViewAssessment, design_space
from finn.kernels.fifo import FifoKernel, FifoStorage
from finn.kernels.port import transport
from finn.kernels.port_model import SequencePort
from finn.kernels.transport import ReadyValidTransport


def declare() -> None:
    # A call on a Space class is a node declaration typed as the class; configure keeps it.
    tensor = stage_tensor(8)
    node = FifoKernel(
        input_tensor=tensor, arriving=in_order(tensor), depth=4, platform=FULL_DSP48E2
    )
    assert_type(node, FifoKernel)
    assert_type(node.input, SequencePort)
    assert_type(node.input.width, int)
    assert_type(design_space(node), FifoKernel)


def check(fifo: FifoKernel) -> None:
    assert_type(fifo.storage, FifoStorage)
    found = transport(fifo.input.physical_spec(idle=False), fifo.input.stream, fifo.input.clocking)
    assert_type(found, ReadyValidTransport | Rejected)
    assert_type(fifo.inspect(FifoKernel.storage), ViewAssessment[FifoStorage])
    assert_type(fifo.field(FifoKernel.storage), BoundValue[FifoStorage])
