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
    from finn.parked.dataflow.model.children import KernelChoice
    from finn.parked.dataflow.model.identity import ImplementationIdentity, implementation_identity
    from finn.parked.dataflow.model.kernel import Kernel
    from finn.parked.dataflow.model.logical.authoring import (
        EdgeSink,
        KernelEndpoint,
        NetworkBoundary,
        NetworkEdge,
        RegionDeclaration,
    )
    from finn.parked.dataflow.model.logical.view import LogicalView, kernel_dataflow
    from finn.parked.dataflow.model.physical.authoring import ModuleParameter, PhysicallyUnsupported
    from finn.parked.dataflow.model.physical.view import PhysicalView, kernel_physical
    from finn.dataflow.model.logical.interface import PublicOperand, OperandExport, OperandTarget
    from finn.parked.dataflow.model.logical.interface_authoring import PublicOperandDeclaration

_LAZY_EXPORTS = {
    "Kernel": ("finn.parked.dataflow.model.kernel", "Kernel"),
    "KernelChoice": ("finn.parked.dataflow.model.children", "KernelChoice"),
    "LogicalView": ("finn.parked.dataflow.model.logical.view", "LogicalView"),
    "RegionDeclaration": ("finn.parked.dataflow.model.logical.authoring", "RegionDeclaration"),
    "KernelEndpoint": ("finn.parked.dataflow.model.logical.authoring", "KernelEndpoint"),
    "EdgeSink": ("finn.parked.dataflow.model.logical.authoring", "EdgeSink"),
    "NetworkEdge": ("finn.parked.dataflow.model.logical.authoring", "NetworkEdge"),
    "NetworkBoundary": ("finn.parked.dataflow.model.logical.authoring", "NetworkBoundary"),
    "ModuleParameter": ("finn.parked.dataflow.model.physical.authoring", "ModuleParameter"),
    "PhysicallyUnsupported": (
        "finn.parked.dataflow.model.physical.authoring",
        "PhysicallyUnsupported",
    ),
    "PhysicalView": ("finn.parked.dataflow.model.physical.view", "PhysicalView"),
    "PublicOperand": ("finn.dataflow.model.logical.interface", "PublicOperand"),
    "OperandExport": ("finn.dataflow.model.logical.interface", "OperandExport"),
    "OperandTarget": ("finn.dataflow.model.logical.interface", "OperandTarget"),
    "PublicOperandDeclaration": (
        "finn.parked.dataflow.model.logical.interface_authoring",
        "PublicOperandDeclaration",
    ),
    "ImplementationIdentity": (
        "finn.parked.dataflow.model.identity",
        "ImplementationIdentity",
    ),
    "implementation_identity": (
        "finn.parked.dataflow.model.identity",
        "implementation_identity",
    ),
    "kernel_dataflow": ("finn.parked.dataflow.model.logical.view", "kernel_dataflow"),
    "kernel_physical": ("finn.parked.dataflow.model.physical.view", "kernel_physical"),
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
    "PublicOperand",
    "OperandExport",
    "OperandTarget",
    "PublicOperandDeclaration",
    "implementation_identity",
    "kernel_dataflow",
    "kernel_physical",
]
