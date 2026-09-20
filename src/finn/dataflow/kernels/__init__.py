# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable concrete Kernel implementations.

The common Kernel contract and view authoring API live in
:mod:`finn.dataflow.model`. This package contains only implementation leaves,
composites, family definitions and their resources.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.kernels.dotp_axi import (
        BatchInterleavedDotpAxiKernel,
        DotpAxiKernel,
        EmbeddedDotpAxiKernel,
    )
    from finn.dataflow.kernels.matmul.base import DspBlock
    from finn.dataflow.kernels.matmul.batch_interleaved import BatchInterleavedKernel
    from finn.dataflow.kernels.matmul.dot_product import DotProductKernel
    from finn.dataflow.kernels.matmul.supply import WeightSupply
    from finn.dataflow.kernels.memstream import MemstreamKernel
    from finn.dataflow.kernels.replay import ActivationReplayKernel
    from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel

_LAZY_EXPORTS = {
    "ActivationReplayKernel": ("finn.dataflow.kernels.replay", "ActivationReplayKernel"),
    "BatchInterleavedDotpAxiKernel": (
        "finn.dataflow.kernels.dotp_axi",
        "BatchInterleavedDotpAxiKernel",
    ),
    "BatchInterleavedKernel": (
        "finn.dataflow.kernels.matmul.batch_interleaved",
        "BatchInterleavedKernel",
    ),
    "DotProductKernel": (
        "finn.dataflow.kernels.matmul.dot_product",
        "DotProductKernel",
    ),
    "DotpAxiKernel": ("finn.dataflow.kernels.dotp_axi", "DotpAxiKernel"),
    "DspBlock": ("finn.dataflow.kernels.matmul.base", "DspBlock"),
    "EmbeddedDotpAxiKernel": (
        "finn.dataflow.kernels.dotp_axi",
        "EmbeddedDotpAxiKernel",
    ),
    "MemstreamKernel": ("finn.dataflow.kernels.memstream", "MemstreamKernel"),
    "ReplayBufferKernel": (
        "finn.dataflow.kernels.replay_buffer",
        "ReplayBufferKernel",
    ),
    "WeightSupply": ("finn.dataflow.kernels.matmul.supply", "WeightSupply"),
}


def __getattr__(name: str) -> object:
    """Load a concrete implementation only when it is named."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "ActivationReplayKernel",
    "BatchInterleavedDotpAxiKernel",
    "BatchInterleavedKernel",
    "DotProductKernel",
    "DotpAxiKernel",
    "DspBlock",
    "EmbeddedDotpAxiKernel",
    "MemstreamKernel",
    "ReplayBufferKernel",
    "WeightSupply",
]
