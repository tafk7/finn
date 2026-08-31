# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Production MVAU operation assembly over ``DataflowDesign`` objects."""

from __future__ import annotations

from finn.dataflow.design import DesignSpaceSpec, QualifiedPath
from finn.dataflow.mvau.associations import (
    BindingLocalStateDestination,
    CoordinateMappingKind,
    MVAUNetworkRef as NetworkRef,
    MVAUParameterTopology,
    MVAURegionRef as RegionRef,
    MVAUSourceAssociation,
    SemanticOperandDestination,
    SourceOperandAssociation,
    SourceOperandDestination,
)
from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau_problem import MVAUProblemPaths, MVAUSourceDescription

DataflowOpResult = RegionRef | NetworkRef


class MVAUDataflowOpPaths:
    """Stable v6 operation paths and projected problem aliases."""

    SOURCE_DESCRIPTION = MVAUProblemPaths.SOURCE_DESCRIPTION
    ACCUMULATOR_TYPE_ANALYSIS_OWNER = MVAUProblemPaths.ACCUMULATOR_TYPE_ANALYSIS_OWNER
    WEIGHT_INITIALIZER_FINGERPRINT = MVAUProblemPaths.WEIGHT_INITIALIZER_FINGERPRINT
    THRESHOLD_INITIALIZER_FINGERPRINT = MVAUProblemPaths.THRESHOLD_INITIALIZER_FINGERPRINT
    EXTERNAL_WEIGHT_SEQUENCE = MVAUProblemPaths.EXTERNAL_WEIGHT_SEQUENCE
    TARGET_FPGA_PART = MVAUProblemPaths.TARGET_FPGA_PART
    TARGET_CLOCK_PERIOD_NS = MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS
    EFFECTIVE_NARROW_WEIGHTS = MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS

    DESIGN = QualifiedPath("mvau.design")
    SOURCE_ASSOCIATION = QualifiedPath("semantic.mvau.op.source_association")
    NETWORK = QualifiedPath("semantic.mvau.op.network")
    NETWORK_VALIDATION = QualifiedPath("semantic.mvau.op.network_validation")
    RESULT = QualifiedPath("semantic.mvau.op.result")
    NETWORK_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.op.network_structurally_well_formed"
    )


MVAU_DATAFLOW_OP_SPEC = MVAU_DESIGN_INVENTORY.specification


def build_mvau_dataflow_op_spec() -> DesignSpaceSpec:
    """Return the reviewed v6 operation specification."""

    return MVAU_DATAFLOW_OP_SPEC


__all__ = [
    "BindingLocalStateDestination",
    "CoordinateMappingKind",
    "DataflowOpResult",
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAU_DESIGN_INVENTORY",
    "MVAUDataflowOpPaths",
    "MVAUParameterTopology",
    "MVAUSourceAssociation",
    "MVAUSourceDescription",
    "NetworkRef",
    "RegionRef",
    "SemanticOperandDestination",
    "SourceOperandAssociation",
    "SourceOperandDestination",
    "build_mvau_dataflow_op_spec",
]
