# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN dataflow-operation design-space assemblies."""

from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAU_DESIGN_INVENTORY,
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
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAU_DESIGN_INVENTORY",
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
