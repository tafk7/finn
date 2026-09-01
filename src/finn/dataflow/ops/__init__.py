# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN dataflow-operation design-space assemblies."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        SemanticOperandDestination,
        SourceOperandAssociation,
        build_mvau_dataflow_op_spec,
    )


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module("finn.dataflow.ops.mvau"), name)
    globals()[name] = value
    return value


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
    "SemanticOperandDestination",
    "SourceOperandAssociation",
    "build_mvau_dataflow_op_spec",
]
