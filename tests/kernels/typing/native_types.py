# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from typing import assert_type

from finn.core.space import BoundValue, ViewAssessment, design_space
from finn.kernels.fifo import FifoKernel, FifoStorage
from finn.kernels.port import WordPort
from finn.kernels.transport import ReadyValidStream
from kernels.helpers import FULL_DSP48E2


def declare() -> None:
    # A call on a Space class is a node declaration typed as the class; configure keeps it.
    node = FifoKernel(word_bits=8, depth=4, platform=FULL_DSP48E2)
    assert_type(node, FifoKernel)
    assert_type(node.input, WordPort)
    assert_type(node.word_bits, int)
    assert_type(design_space(node), FifoKernel)


def check(fifo: FifoKernel) -> None:
    assert_type(fifo.storage, FifoStorage)
    assert_type(fifo.input.transport, ReadyValidStream)
    assert_type(fifo.inspect(FifoKernel.storage), ViewAssessment[FifoStorage])
    assert_type(fifo.field(FifoKernel.storage), BoundValue[FifoStorage])
