# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN dataflow-operation design-space assemblies."""

from finn.dataflow.ops.mvau import (
    MVAU_COMPUTE_SELECTION,
    MVAU_DATAFLOW_OP_SPEC,
    MVAU_SELECTIONS,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    BindingLocalStateDestination,
    CoordinateMappingKind,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    MVAUSourceDescription,
    NetworkRef,
    RegionRef,
    SemanticOperandDestination,
    SourceOperandAssociation,
    build_mvau_dataflow_op_spec,
)

__all__ = [
    "MVAU_COMPUTE_SELECTION",
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAU_SELECTIONS",
    "MVAU_WEIGHT_ADAPTER_SELECTION",
    "MVAU_WEIGHT_SUPPLY_SELECTION",
    "BindingLocalStateDestination",
    "CoordinateMappingKind",
    "MVAUDataflowOpPaths",
    "MVAUParameterTopology",
    "MVAUSourceAssociation",
    "MVAUSourceDescription",
    "NetworkRef",
    "RegionRef",
    "SemanticOperandDestination",
    "SourceOperandAssociation",
    "build_mvau_dataflow_op_spec",
]
