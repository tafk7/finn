# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from typing_extensions import assert_type

from finn.core.space import BoundView, configure
from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.datatypes.scalar import IntegerScalar, ScalarEncoding
from finn.kernels.fifo import FifoKernel, FifoStorage
from finn.kernels.int_to_fp32 import IntToFp32Kernel
from finn.kernels.physical.stream import ReadyValidStream


def declare(dtype: QONNXDataType) -> None:
    # A family call is a node declaration typed as the family; configure keeps it.
    node = IntToFp32Kernel(input_dtype=dtype)
    assert_type(node, IntToFp32Kernel)
    assert_type(node.input, IntegerScalar)
    assert_type(node.input.dtype, QONNXDataType)
    assert_type(configure(node), IntToFp32Kernel)
    assert_type(configure(FifoKernel(word_bits=8, depth=4)), FifoKernel)


def check(fifo: FifoKernel, converter: IntToFp32Kernel) -> None:
    assert_type(fifo.storage, BoundView[FifoStorage])
    assert_type(fifo.interfaces, BoundView[tuple[ReadyValidStream, ...]])
    assert_type(fifo.interfaces()[0], ReadyValidStream)
    assert_type(converter.input, IntegerScalar)
    assert_type(converter.input.encoding(), ScalarEncoding)
