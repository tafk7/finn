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

__all__ = [
    "BeatSequence",
    "BeatType",
    "CanonicalExtentProfile",
    "DataflowRegion",
    "InputInterface",
    "LogicalSchedule",
    "NumericElementType",
    "Operand",
    "OutputInterface",
    "Port",
    "ProfileCertificationError",
    "ProfileCertificationIssue",
    "RegionValidationIssue",
    "RegionValidationReport",
    "ScheduleLevel",
    "ScheduledInputRequirements",
    "ScheduledOutputAvailability",
    "direct_output_availability",
    "explicit_beat_sequence",
    "is_structurally_well_formed",
    "lexicographic_occurrence_to_field",
    "validate_region",
]
