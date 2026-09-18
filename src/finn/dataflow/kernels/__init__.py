# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified domain authoring for leaf and composite Kernels.

A ``Kernel`` is an ordinary ``Space`` with stable identity and typed logical,
physical and relation capabilities. Leaves may use ``RegionDeclaration`` and
``ModuleParameter``; composites use ordinary child alternatives plus explicit
network topology. Both share the same compiler and occurrence runtime.

Artifact projection is downstream and one-way.  Nothing in
``finn.dataflow.artifacts`` imports this package.

A Kernel answers two questions separately -- ``kernel.dataflow`` for its
Region, ``kernel.physical`` for its detached build unit -- and only the second
crosses into artifact code.  ``ModuleBuildRequirements`` is that boundary: an
artifact function receives resolved identity, parameters, ABI and
contributions, and no handle back into the design space.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
    from finn.dataflow.kernels.kernel import (
        EdgeSink,
        Kernel,
        KernelChoice,
        LogicalView,
        ModuleBuildRequirements,
        ModuleParameter,
        NetworkBoundary,
        NetworkEdge,
        PhysicalView,
        PhysicallyUnsupported,
        RegionDeclaration,
        RelationView,
        kernel_dataflow,
        kernel_physical,
    )
    from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel

_LAZY_EXPORTS = {
    name: ("finn.dataflow.kernels.kernel", name)
    for name in (
        "EdgeSink",
        "Kernel",
        "KernelChoice",
        "LogicalView",
        "ModuleBuildRequirements",
        "ModuleParameter",
        "NetworkBoundary",
        "NetworkEdge",
        "PhysicalView",
        "PhysicallyUnsupported",
        "RegionDeclaration",
        "RelationView",
        "kernel_dataflow",
        "kernel_physical",
    )
}
_LAZY_EXPORTS.update(
    {name: ("finn.dataflow.kernels.dotp_axi", name) for name in ("DspBlock", "DotpAxiKernel")}
)
_LAZY_EXPORTS["ReplayBufferKernel"] = (
    "finn.dataflow.kernels.replay_buffer",
    "ReplayBufferKernel",
)


def __getattr__(name: str) -> object:
    """Load a Kernel implementation only when it is named."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    # the generic Kernel contract
    "Kernel",
    "KernelChoice",
    "LogicalView",
    "ModuleParameter",
    "NetworkBoundary",
    "NetworkEdge",
    "PhysicalView",
    "PhysicallyUnsupported",
    "RegionDeclaration",
    "RelationView",
    "EdgeSink",
    # the two projections, and the detached value the physical one produces
    "ModuleBuildRequirements",
    "kernel_dataflow",
    "kernel_physical",
    # reusable implementations
    "DotpAxiKernel",
    "DspBlock",
    "ReplayBufferKernel",
]
