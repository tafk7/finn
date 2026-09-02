# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable logical types and family identity for the canonical MVAU adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.dataflow.authoring.scope import Ref
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.parameters.cyclic.definition import CyclicTargetMemoryCapabilities
from finn.dataflow.region import BeatSequence, NumericElementType

MVAU_DATAFLOW_OP_FAMILY_ID = "finn.dataflow.mvau"
MVAU_DATAFLOW_OP_FAMILY_VERSION = "mvau-dataflow-op-v7"


class MVAUComputationProfile(str, Enum):
    """Source-level MVAU computation semantics recognized by the adapter."""

    __dataflow_identity_token__ = "finn.dataflow.mvau_problem.MVAUComputationProfile"

    ACCUMULATOR_INTEGER = "accumulator_integer"
    BIPOLAR_XNOR_ACCUMULATOR = "bipolar_xnor_accumulator"
    FUSED_THRESHOLD = "fused_threshold"


@dataclass(frozen=True)
class MVAUSourceDescription:
    """Source identities and shapes projected for one logical MVAU occurrence."""

    __dataflow_identity_token__ = "finn.dataflow.mvau_problem.MVAUSourceDescription"

    source_node_id: str
    activation_operand_id: str
    weight_operand_id: str
    output_operand_id: str
    leading_shape: tuple[int, ...]
    threshold_operand_id: str | None = None
    fused_source_node_ids: tuple[str, ...] = ()
    threshold_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "leading_shape", tuple(self.leading_shape))
        object.__setattr__(self, "fused_source_node_ids", tuple(self.fused_source_node_ids))
        if self.threshold_shape is not None:
            object.__setattr__(self, "threshold_shape", tuple(self.threshold_shape))


@dataclass(frozen=True)
class MVAUProblem:
    """Typed operation facts imported by MVAU Designs and Kernels."""

    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    threshold_element_type: Ref[NumericElementType]
    threshold_initializer_available: Ref[bool]
    computation_profile: Ref[MVAUComputationProfile]
    weight_initializer_available: Ref[bool]
    weight_initializer_fingerprint: Ref[str]
    threshold_initializer_fingerprint: Ref[str]
    source_description: Ref[MVAUSourceDescription]
    initializer_excludes_minimum: Ref[bool]
    runtime_weight_range_contract: Ref[bool]
    runtime_writable: Ref[bool]
    external_weight_sequence: Ref[BeatSequence]
    accumulator_type_analysis_owner: Ref[str]
    target_dsp_block: Ref[DspBlock]
    target_fpga_part: Ref[str]
    target_clock_period_ns: Ref[float]
    target_memory_capabilities: Ref[CyclicTargetMemoryCapabilities]


__all__ = [
    "MVAU_DATAFLOW_OP_FAMILY_ID",
    "MVAU_DATAFLOW_OP_FAMILY_VERSION",
    "MVAUComputationProfile",
    "MVAUProblem",
    "MVAUSourceDescription",
]
