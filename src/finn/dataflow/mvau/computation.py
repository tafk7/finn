# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU-specific computation contracts kept separate from region structure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MVAUComputationProfile(str, Enum):
    """Evidenced source-computation profiles supported by MVAU bindings."""

    ACCUMULATOR_INTEGER = "accumulator_integer"
    BIPOLAR_XNOR_ACCUMULATOR = "bipolar_xnor_accumulator"
    FUSED_THRESHOLD = "fused_threshold"


@dataclass(frozen=True)
class MVAUBindingWitness:
    """Named implementation mechanisms that witness one MVAU binding claim.

    Feasibility is intentionally evaluated by separate constraints.  This value
    records the association and the mechanisms whose correctness the binding
    promises; it is not a structural-validation result.
    """

    binding_id: str
    region_declaration_id: str
    computation_profile: MVAUComputationProfile
    mechanisms: tuple[str, ...]


__all__ = ["MVAUBindingWitness", "MVAUComputationProfile"]
