# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-domain values and authoring framework.

The package root is intentionally lightweight. Detached logical values live in
:mod:`finn.dataflow.model.logical`; importing them must not load the Space
runtime or Kernel authoring adapters.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.model.children import KernelChoice
    from finn.dataflow.model.identity import ImplementationIdentity, implementation_identity
    from finn.dataflow.model.kernel import Kernel
    from finn.dataflow.model.logical.authoring import (
        EdgeSink,
        KernelEndpoint,
        NetworkBoundary,
        NetworkEdge,
        RegionDeclaration,
    )
    from finn.dataflow.model.logical.view import LogicalView, kernel_dataflow
    from finn.dataflow.model.physical.authoring import ModuleParameter, PhysicallyUnsupported
    from finn.dataflow.model.physical.view import PhysicalView, kernel_physical
    from finn.dataflow.model.relations.view import RelationView

_LAZY_EXPORTS = {
    "Kernel": ("finn.dataflow.model.kernel", "Kernel"),
    "KernelChoice": ("finn.dataflow.model.children", "KernelChoice"),
    "LogicalView": ("finn.dataflow.model.logical.view", "LogicalView"),
    "RegionDeclaration": ("finn.dataflow.model.logical.authoring", "RegionDeclaration"),
    "KernelEndpoint": ("finn.dataflow.model.logical.authoring", "KernelEndpoint"),
    "EdgeSink": ("finn.dataflow.model.logical.authoring", "EdgeSink"),
    "NetworkEdge": ("finn.dataflow.model.logical.authoring", "NetworkEdge"),
    "NetworkBoundary": ("finn.dataflow.model.logical.authoring", "NetworkBoundary"),
    "ModuleParameter": ("finn.dataflow.model.physical.authoring", "ModuleParameter"),
    "PhysicallyUnsupported": (
        "finn.dataflow.model.physical.authoring",
        "PhysicallyUnsupported",
    ),
    "PhysicalView": ("finn.dataflow.model.physical.view", "PhysicalView"),
    "RelationView": ("finn.dataflow.model.relations.view", "RelationView"),
    "ImplementationIdentity": (
        "finn.dataflow.model.identity",
        "ImplementationIdentity",
    ),
    "implementation_identity": (
        "finn.dataflow.model.identity",
        "implementation_identity",
    ),
    "kernel_dataflow": ("finn.dataflow.model.logical.view", "kernel_dataflow"),
    "kernel_physical": ("finn.dataflow.model.physical.view", "kernel_physical"),
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
    "EdgeSink",
    "ImplementationIdentity",
    "Kernel",
    "KernelChoice",
    "KernelEndpoint",
    "LogicalView",
    "ModuleParameter",
    "NetworkBoundary",
    "NetworkEdge",
    "PhysicalView",
    "PhysicallyUnsupported",
    "RegionDeclaration",
    "RelationView",
    "implementation_identity",
    "kernel_dataflow",
    "kernel_physical",
]
