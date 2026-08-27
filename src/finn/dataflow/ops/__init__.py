# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN dataflow-operation design-space assemblies."""

from finn.dataflow.ops.mvau import (
    BindingLocalStateDestination,
    MVAU_DATAFLOW_OP_SPEC,
    MVAUConnectionTopology,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    MVAUSourceDescription,
    MVAUWeightDeliveryDeclaration,
    NetworkRef,
    RegionRef,
    SemanticOperandDestination,
    build_mvau_dataflow_op_spec,
)

__all__ = [
    "BindingLocalStateDestination",
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAUConnectionTopology",
    "MVAUDataflowOpPaths",
    "MVAUParameterTopology",
    "MVAUSourceAssociation",
    "MVAUSourceDescription",
    "MVAUWeightDeliveryDeclaration",
    "NetworkRef",
    "RegionRef",
    "SemanticOperandDestination",
    "build_mvau_dataflow_op_spec",
]
