# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical dataflow-region model.

This package intentionally contains only model-level concepts. It has no
dependency on FINN graphs, custom operations, the design-space engine, or
physical backends.
"""

from finn.dataflow.region import (
    BeatSequence,
    BeatType,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.region_profiles import (
    CanonicalExtentProfile,
    ProfileCertificationError,
    ProfileCertificationIssue,
    direct_output_availability,
    explicit_beat_sequence,
    lexicographic_occurrence_to_field,
)
from finn.dataflow.region_validation import (
    RegionValidationIssue,
    RegionValidationReport,
    is_structurally_well_formed,
    validate_region,
)
from finn.dataflow.network import (
    BoundaryContract,
    ChannelSpec,
    DataflowNetwork,
    DirectConnection,
    Edge,
    FanoutMode,
    NetworkNode,
    OrderedChannel,
    PassCorrespondence,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import (
    NetworkValidationIssue,
    NetworkValidationReport,
    is_network_structurally_well_formed,
    validate_network,
)

__all__ = [
    "BeatSequence",
    "BeatType",
    "BoundaryContract",
    "CanonicalExtentProfile",
    "ChannelSpec",
    "DataflowNetwork",
    "DirectConnection",
    "Edge",
    "FanoutMode",
    "DataflowRegion",
    "InputInterface",
    "LogicalSchedule",
    "NumericElementType",
    "NetworkNode",
    "NetworkValidationIssue",
    "NetworkValidationReport",
    "OrderedChannel",
    "Operand",
    "OutputInterface",
    "Port",
    "PassCorrespondence",
    "PositionMap",
    "ProfileCertificationError",
    "ProfileCertificationIssue",
    "RegionValidationIssue",
    "RegionValidationReport",
    "RegionEndpoint",
    "ScheduleLevel",
    "ScheduledInputRequirements",
    "ScheduledOutputAvailability",
    "SinkContract",
    "direct_output_availability",
    "explicit_beat_sequence",
    "is_structurally_well_formed",
    "is_network_structurally_well_formed",
    "lexicographic_occurrence_to_field",
    "validate_region",
    "validate_network",
]
