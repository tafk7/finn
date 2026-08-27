# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Design-engine value semantics for model-owned dataflow region values."""

from finn.dataflow._engine import ValueSemantics
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport
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
DATAFLOW_NETWORK_SEMANTICS = ValueSemantics.immutable_nominal(
    DataflowNetwork,
    name="DataflowNetwork",
)
NETWORK_VALIDATION_REPORT_SEMANTICS = ValueSemantics.immutable_nominal(
    NetworkValidationReport,
    name="NetworkValidationReport",
)

__all__ = [
    "DATAFLOW_REGION_SEMANTICS",
    "DATAFLOW_NETWORK_SEMANTICS",
    "NETWORK_VALIDATION_REPORT_SEMANTICS",
    "REGION_VALIDATION_REPORT_SEMANTICS",
]
