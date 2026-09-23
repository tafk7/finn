# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility facade for logical value semantics.

Datatype and logical-result semantics have lightweight owning modules. This
facade re-exports the same objects and declares the remaining Region, Network
and validation-report domains. No alternative tokens are created by the split.
"""

from finn.kernels._engine import ValueSemantics
from finn.dataflow.model.logical.network import DataflowNetwork, PositionMap
from finn.dataflow.model.logical.network_validation import NetworkValidationReport
from finn.dataflow.model.logical.region import DataflowRegion
from finn.dataflow.model.logical.region_validation import RegionValidationReport

from finn.kernels.datatypes.semantics import (
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_SEMANTICS,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)

from finn.dataflow.model.logical.result_semantics import (
    DATAFLOW_LOGICAL_RESULT_SEMANTICS,
)


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


POSITION_MAP_SEMANTICS = ValueSemantics.immutable_nominal(
    PositionMap,
    name="PositionMap",
)
NETWORK_VALIDATION_REPORT_SEMANTICS = ValueSemantics.immutable_nominal(
    NetworkValidationReport,
    name="NetworkValidationReport",
)

__all__ = [
    "DATAFLOW_LOGICAL_RESULT_SEMANTICS",
    "DATAFLOW_REGION_SEMANTICS",
    "QONNX_DATATYPE_CODEC",
    "DATAFLOW_NETWORK_SEMANTICS",
    "NETWORK_VALIDATION_REPORT_SEMANTICS",
    "POSITION_MAP_SEMANTICS",
    "QONNX_DATATYPE_SEMANTICS",
    "QONNX_DATATYPE_VALUE_SEMANTICS",
    "REGION_VALIDATION_REPORT_SEMANTICS",
]
