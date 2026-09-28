# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical components and the supported explicit MatMul assembly.

The public construction path needs no compiler node. Scalar datatype values
and canonical logical values come from :mod:`finn.dataflow`, below this package.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.kernels.configure import commit
    from finn.kernels.delivery import CyclicDelivery
    from finn.kernels.dotp import DotpAxiKernel
    from finn.kernels.eltwise import EltwiseKernel
    from finn.kernels.fifo import FifoKernel
    from finn.kernels.input_generator import InputGeneratorKernel
    from finn.kernels.int_to_fp32 import IntToFp32Kernel
    from finn.kernels.memstream_hls import MemStreamHlsKernel
    from finn.kernels.thresholding import ThresholdingAxiKernel
    from finn.kernels.matmul import MatMulAssembly, MatMulKernel, WeightDelivery, matmul_assembly
    from finn.kernels.streaming import (
        cyclic_stream_requirements,
        replay_buffer_requirements,
    )
    from finn.kernels.target import DspBlock

_LAZY_EXPORTS = {
    "CyclicDelivery": ("finn.kernels.delivery", "CyclicDelivery"),
    "commit": ("finn.kernels.configure", "commit"),
    "DotpAxiKernel": ("finn.kernels.dotp", "DotpAxiKernel"),
    "EltwiseKernel": ("finn.kernels.eltwise", "EltwiseKernel"),
    "FifoKernel": ("finn.kernels.fifo", "FifoKernel"),
    "InputGeneratorKernel": ("finn.kernels.input_generator", "InputGeneratorKernel"),
    "IntToFp32Kernel": ("finn.kernels.int_to_fp32", "IntToFp32Kernel"),
    "MemStreamHlsKernel": ("finn.kernels.memstream_hls", "MemStreamHlsKernel"),
    "ThresholdingAxiKernel": ("finn.kernels.thresholding", "ThresholdingAxiKernel"),
    "MatMulKernel": ("finn.kernels.matmul", "MatMulKernel"),
    "MatMulAssembly": ("finn.kernels.matmul", "MatMulAssembly"),
    "WeightDelivery": ("finn.kernels.matmul", "WeightDelivery"),
    "matmul_assembly": ("finn.kernels.matmul", "matmul_assembly"),
    "replay_buffer_requirements": ("finn.kernels.streaming", "replay_buffer_requirements"),
    "cyclic_stream_requirements": ("finn.kernels.streaming", "cyclic_stream_requirements"),
    "DspBlock": ("finn.kernels.target", "DspBlock"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "CyclicDelivery",
    "commit",
    "DotpAxiKernel",
    "EltwiseKernel",
    "FifoKernel",
    "InputGeneratorKernel",
    "IntToFp32Kernel",
    "MemStreamHlsKernel",
    "ThresholdingAxiKernel",
    "MatMulKernel",
    "MatMulAssembly",
    "WeightDelivery",
    "matmul_assembly",
    "replay_buffer_requirements",
    "cyclic_stream_requirements",
    "DspBlock",
]
