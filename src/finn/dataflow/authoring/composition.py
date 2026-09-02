# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Restricted whole-Design physical-composition boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.kernels.kernel import KernelOrigin
from finn.dataflow.network import DataflowNetwork


@dataclass(frozen=True, slots=True)
class PhysicalCompositionProvenance:
    """Resolved inputs retained after a composer has consumed its realization."""

    family_id: str
    family_version: str
    problem_fingerprint: str
    assignments: tuple[tuple[str, object], ...]
    source_scope_id: str
    selected_design_id: str
    selected_design_version: str
    network: DataflowNetwork
    source_association: object
    facts: Mapping[str, object]
    kernel_origins: tuple[tuple[str, KernelOrigin], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))


@dataclass(frozen=True, slots=True)
class PhysicalCompositionContext(PhysicalCompositionProvenance):
    """Capability-restricted input supplied to one Design-owned composer."""

    realization: DesignRealization

    def provenance(self) -> PhysicalCompositionProvenance:
        """Drop configured Kernels before crossing the artifact handoff."""

        return PhysicalCompositionProvenance(
            family_id=self.family_id,
            family_version=self.family_version,
            problem_fingerprint=self.problem_fingerprint,
            assignments=self.assignments,
            source_scope_id=self.source_scope_id,
            selected_design_id=self.selected_design_id,
            selected_design_version=self.selected_design_version,
            network=self.network,
            source_association=self.source_association,
            facts=self.facts,
            kernel_origins=self.kernel_origins,
        )


__all__ = ["PhysicalCompositionContext", "PhysicalCompositionProvenance"]
