# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Design-engine value semantics for model-owned dataflow region values."""

from finn.dataflow._engine import ValueSemantics
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import RegionValidationReport

DATAFLOW_REGION_SEMANTICS = ValueSemantics.immutable_nominal(
    DataflowRegion,
    name="DataflowRegion",
)
REGION_VALIDATION_REPORT_SEMANTICS = ValueSemantics.immutable_nominal(
    RegionValidationReport,
    name="RegionValidationReport",
)

__all__ = [
    "DATAFLOW_REGION_SEMANTICS",
    "REGION_VALIDATION_REPORT_SEMANTICS",
]
