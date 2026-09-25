# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from typing_extensions import assert_type

from finn.core.space import BoundView
from finn.kernels.datatypes.scalar import IntegerScalar, ScalarEncoding
from finn.kernels.fifo import FifoKernel, FifoStorage
from finn.kernels.int_to_fp32 import IntToFp32Kernel
from finn.kernels.physical.stream import ReadyValidStream


def check(fifo: FifoKernel, converter: IntToFp32Kernel) -> None:
    assert_type(fifo.storage, BoundView[FifoStorage])
    assert_type(fifo.interfaces, BoundView[tuple[ReadyValidStream, ...]])
    assert_type(fifo.interfaces()[0], ReadyValidStream)
    assert_type(converter.input, IntegerScalar)
    assert_type(converter.input.encoding(), ScalarEncoding)
