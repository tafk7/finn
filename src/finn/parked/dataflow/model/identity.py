# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable identity for Kernel-domain implementations."""

from __future__ import annotations

from dataclasses import dataclass


TYPE_IDENTITY_RELOCATIONS = {
    "finn.dataflow.model.identity.ImplementationIdentity": (
        "finn.kernels.space.capabilities.ImplementationIdentity"
    ),
    "finn.kernels.physical.layout.FieldPlacement": (
        "finn.dataflow.kernels.physical.FieldPlacement"
    ),
    "finn.kernels.physical.layout.PackedBeatLayout": (
        "finn.dataflow.kernels.physical.PackedBeatLayout"
    ),
    "finn.kernels.physical.layout.PeriodicLast": ("finn.dataflow.kernels.physical.PeriodicLast"),
    "finn.kernels.physical.layout.UnusedBitPolicy": (
        "finn.dataflow.kernels.physical.UnusedBitPolicy"
    ),
    "finn.kernels.physical.layout.UnusedBitRange": (
        "finn.dataflow.kernels.physical.UnusedBitRange"
    ),
    "finn.parked.dataflow.model.physical.interface.KernelRealizationFacts": (
        "finn.dataflow.kernels.physical.KernelRealizationFacts"
    ),
    "finn.parked.dataflow.model.physical.interface.KernelStreamBinding": (
        "finn.dataflow.kernels.physical.KernelStreamBinding"
    ),
    "finn.dataflow.kernels.matmul.base.AccumulationMode": (
        "finn.parked.dataflow.ops.mvau.computation.AccumulationMode"
    ),
    "finn.dataflow.kernels.matmul.base.ActivationMode": (
        "finn.parked.dataflow.ops.mvau.computation.ActivationMode"
    ),
    "finn.dataflow.kernels.matmul.base.MvauComputationProfile": (
        "finn.parked.dataflow.ops.mvau.computation.MvauComputationProfile"
    ),
    "finn.dataflow.kernels.matmul.base.DspBlock": "finn.dataflow.kernels.dotp_axi.DspBlock",
    "finn.dataflow.kernels.matmul.supply.WeightSupply": (
        "finn.parked.dataflow.ops.mvau.kernels.supply.WeightSupply"
    ),
}

MODULE_IDENTITY_RELOCATIONS = {
    "finn.dataflow.model.logical.composition": "finn.dataflow.model.composition",
    "finn.dataflow.datatypes": "finn.dataflow.model.datatypes",
    "finn.dataflow.model.logical.maps": "finn.dataflow.model.maps",
    "finn.dataflow.model.logical.network": "finn.dataflow.model.network",
    "finn.dataflow.model.logical.network_validation": "finn.dataflow.model.network_validation",
    "finn.dataflow.model.logical.presentation": "finn.dataflow.model.presentation",
    "finn.dataflow.model.logical.refs": "finn.dataflow.model.refs",
    "finn.dataflow.model.logical.region": "finn.dataflow.model.region",
    "finn.dataflow.model.logical.region_profiles": "finn.dataflow.model.region_profiles",
    "finn.dataflow.model.logical.region_validation": "finn.dataflow.model.region_validation",
    "finn.parked.dataflow.model.children": "finn.dataflow.kernels.kernel",
    "finn.parked.dataflow.model.kernel": "finn.dataflow.kernels.kernel",
    "finn.parked.dataflow.model.logical.authoring": "finn.dataflow.kernels.kernel",
    "finn.parked.dataflow.model.logical.view": "finn.dataflow.kernels.kernel",
    "finn.parked.dataflow.model.physical.authoring": "finn.dataflow.kernels.kernel",
    "finn.parked.dataflow.model.physical.view": "finn.dataflow.kernels.kernel",
    "finn.kernels.physical.structure": "finn.dataflow.kernels.physical_composition",
    "finn.parked.dataflow.model.physical.capture": "finn.parked.dataflow.ops.physical",
}


@dataclass(frozen=True, slots=True)
class ImplementationIdentity:
    family: str
    version: str

    def __post_init__(self) -> None:
        if not self.family or not self.version:
            raise ValueError("implementation family and version must be non-empty")


def implementation_identity(
    implementation: object | type[object],
) -> ImplementationIdentity:
    implementation_type = (
        implementation if isinstance(implementation, type) else type(implementation)
    )
    family = getattr(implementation_type, "id", "")
    version = getattr(implementation_type, "version", "")
    if not isinstance(family, str) or not isinstance(version, str):
        raise TypeError("implementation id and version must be strings")
    return ImplementationIdentity(family, version)


def comparison_type_identity(value: object | type[object]) -> str:
    """Return the pre-KP type identity used by existing comparison encodings."""

    value_type = value if isinstance(value, type) else type(value)
    identity = f"{value_type.__module__}.{value_type.__qualname__}"
    relocated = TYPE_IDENTITY_RELOCATIONS.get(identity)
    if relocated is not None:
        return relocated
    module = MODULE_IDENTITY_RELOCATIONS.get(value_type.__module__)
    return identity if module is None else f"{module}.{value_type.__qualname__}"


__all__ = [
    "MODULE_IDENTITY_RELOCATIONS",
    "TYPE_IDENTITY_RELOCATIONS",
    "ImplementationIdentity",
    "comparison_type_identity",
    "implementation_identity",
]
