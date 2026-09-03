# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Design layer: composition, topology, and one selected Network.

A ``DataflowDesign`` owns every decision that changes its selected logical
Regions or the Network they form.  It places named segments of candidate Kernel
``Subspace`` alternatives and declares explicit ``Connection`` and ``Boundary``
topology; the Network it publishes is generated from the exact selected Regions.

This package holds the *generic* mechanism only.  A Design that exists to
realize one operation is that operation's inventory and lives with it -- MVAU's
under ``finn.dataflow.ops.mvau.designs``.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.designs.design import (
        Boundary,
        Connection,
        DataflowDesign,
        Kernels,
        SelectedNetwork,
        Sink,
        design_dataflow,
    )

_LAZY_EXPORTS = {
    name: ("finn.dataflow.designs.design", name)
    for name in (
        "Boundary",
        "Connection",
        "DataflowDesign",
        "Kernels",
        "SelectedNetwork",
        "Sink",
        "design_dataflow",
    )
}


def __getattr__(name: str) -> object:
    """Load the Design declarations only when one is named."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "Boundary",
    "Connection",
    "DataflowDesign",
    "Kernels",
    "SelectedNetwork",
    "Sink",
    "design_dataflow",
]
