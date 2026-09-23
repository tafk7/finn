# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical components and the supported explicit MVAU assembly.

The public construction path needs no Region, Network or compiler node.
Logical modeling experiments remain available in their individual modules.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.kernels.dotp_axi_minimal import DotpAxiKernel
    from finn.dataflow.kernels.mvau import MVAU, MVAUAssembly, WeightDelivery, mvau_assembly
    from finn.dataflow.kernels.streaming import (
        cyclic_stream_requirements,
        replay_buffer_requirements,
    )
    from finn.dataflow.kernels.target import DspBlock

_LAZY_EXPORTS = {
    "DotpAxiKernel": ("finn.dataflow.kernels.dotp_axi_minimal", "DotpAxiKernel"),
    "MVAU": ("finn.dataflow.kernels.mvau", "MVAU"),
    "MVAUAssembly": ("finn.dataflow.kernels.mvau", "MVAUAssembly"),
    "WeightDelivery": ("finn.dataflow.kernels.mvau", "WeightDelivery"),
    "mvau_assembly": ("finn.dataflow.kernels.mvau", "mvau_assembly"),
    "replay_buffer_requirements": ("finn.dataflow.kernels.streaming", "replay_buffer_requirements"),
    "cyclic_stream_requirements": ("finn.dataflow.kernels.streaming", "cyclic_stream_requirements"),
    "DspBlock": ("finn.dataflow.kernels.target", "DspBlock"),
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
    "DotpAxiKernel",
    "MVAU",
    "MVAUAssembly",
    "WeightDelivery",
    "mvau_assembly",
    "replay_buffer_requirements",
    "cyclic_stream_requirements",
    "DspBlock",
]
