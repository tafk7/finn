# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical kernels: FinnLib modules on design spaces, and kernels built from them.

The public construction path needs no compiler node. Scalar datatype values
and canonical logical values come from :mod:`finn.dataflow`, below this package.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.kernels.configure import commit
    from finn.kernels.dotp import (
        DotpAxiKernel,
        Int8Dsp58DotpKernel,
        PackedDotpKernel,
    )
    from finn.kernels.eltwise import EltwiseKernel
    from finn.kernels.fifo import FifoKernel
    from finn.kernels.input_generator import InputGeneratorKernel
    from finn.kernels.matmul import MatMulKernel
    from finn.kernels.memstream import MemStreamKernel
    from finn.kernels.target import DspBlock
    from finn.kernels.thresholding import ThresholdingAxiKernel
    from finn.kernels.vpc import VpcKernel

_LAZY_EXPORTS = {
    "MemStreamKernel": ("finn.kernels.memstream", "MemStreamKernel"),
    "commit": ("finn.kernels.configure", "commit"),
    "DotpAxiKernel": ("finn.kernels.dotp", "DotpAxiKernel"),
    "Int8Dsp58DotpKernel": ("finn.kernels.dotp", "Int8Dsp58DotpKernel"),
    "PackedDotpKernel": ("finn.kernels.dotp", "PackedDotpKernel"),
    "EltwiseKernel": ("finn.kernels.eltwise", "EltwiseKernel"),
    "FifoKernel": ("finn.kernels.fifo", "FifoKernel"),
    "InputGeneratorKernel": ("finn.kernels.input_generator", "InputGeneratorKernel"),
    "ThresholdingAxiKernel": ("finn.kernels.thresholding", "ThresholdingAxiKernel"),
    "MatMulKernel": ("finn.kernels.matmul", "MatMulKernel"),
    "VpcKernel": ("finn.kernels.vpc", "VpcKernel"),
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
    "MemStreamKernel",
    "commit",
    "DotpAxiKernel",
    "Int8Dsp58DotpKernel",
    "PackedDotpKernel",
    "EltwiseKernel",
    "FifoKernel",
    "InputGeneratorKernel",
    "ThresholdingAxiKernel",
    "MatMulKernel",
    "VpcKernel",
    "DspBlock",
]
