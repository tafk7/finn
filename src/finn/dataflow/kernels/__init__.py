# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Kernel layer: one Region, its physics, and the reusable Kernels.

A ``Kernel`` is an ordinary ``Space`` that derives exactly one canonical
``DataflowRegion`` from semantic facts its enclosing Design supplies as typed
Inputs, and owns nothing but physical realization locally.  ``DotpAxiKernel``
and ``ReplayBufferKernel`` are the two reusable implementations; a Kernel that
belongs to one operation belongs with that operation instead.

Artifact projection is downstream and one-way -- ``artifacts`` reads a
configured Kernel and produces artifact-native values.  Nothing in
``finn.dataflow.artifacts`` imports this package.

The final detached ``KernelPhysicalResult`` handoff is U2 work; today the
artifact helpers still read the configured occurrence directly.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.kernels.artifacts import (
        kernel_source_derivation,
        portable_kernel_component,
        resolve_kernel_contributions,
    )
    from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
    from finn.dataflow.kernels.kernel import (
        Kernel,
        Parameter,
        Region,
        RegionRefused,
        configure_kernel,
    )
    from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel

_LAZY_EXPORTS = {
    name: ("finn.dataflow.kernels.kernel", name)
    for name in ("Kernel", "Parameter", "Region", "RegionRefused", "configure_kernel")
}
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.kernels.artifacts", name)
        for name in (
            "kernel_source_derivation",
            "portable_kernel_component",
            "resolve_kernel_contributions",
        )
    }
)
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
    "Parameter",
    "Region",
    "RegionRefused",
    "configure_kernel",
    # downstream artifact projection, one-way
    "kernel_source_derivation",
    "portable_kernel_component",
    "resolve_kernel_contributions",
    # reusable implementations
    "DotpAxiKernel",
    "DspBlock",
    "ReplayBufferKernel",
]
