# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Matrix model adapters, shared target values and assembly template resolution.

Physical dotp and MVAU construction are owned by ``finn.kernels``. This facade
serves the retained logical matrix models; its target and template exports refer
to the same objects used by the physical library.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.kernels.matmul.base import (
        AccumulationMode,
        ActivationMode,
        DspBlock,
        MvauComputationProfile,
        WeightedDotProductKernel,
        computation_profile,
    )
    from finn.dataflow.kernels.matmul.batch_interleaved import BatchInterleavedKernel
    from finn.dataflow.kernels.matmul.dot_product import DotProductKernel
    from finn.dataflow.kernels.matmul.supply import WeightSupply

_BASE_EXPORTS = (
    "AccumulationMode",
    "ActivationMode",
    "DspBlock",
    "MvauComputationProfile",
    "WeightedDotProductKernel",
    "computation_profile",
)
_LAZY_EXPORTS = {name: ("finn.dataflow.kernels.matmul.base", name) for name in _BASE_EXPORTS}
_LAZY_EXPORTS.update(
    {
        "BatchInterleavedKernel": (
            "finn.dataflow.kernels.matmul.batch_interleaved",
            "BatchInterleavedKernel",
        ),
        "DotProductKernel": ("finn.dataflow.kernels.matmul.dot_product", "DotProductKernel"),
        "WeightSupply": ("finn.dataflow.kernels.matmul.supply", "WeightSupply"),
        "template_root": ("finn.kernels.resources", "template_root"),
    }
)


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AccumulationMode",
    "ActivationMode",
    "BatchInterleavedKernel",
    "DotProductKernel",
    "DspBlock",
    "MvauComputationProfile",
    "WeightSupply",
    "WeightedDotProductKernel",
    "computation_profile",
    "template_root",
]
