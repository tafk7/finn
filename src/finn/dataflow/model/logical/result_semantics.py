# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared value semantics for explicit logical Region/Network results."""

from finn.dataflow._engine import ValueSemantics
from finn.dataflow.model.logical.results import LogicalResult, RegionResult, NetworkResult


class _LogicalResultToken:
    """Identity token for the Region-or-Network result capability."""


_LogicalResultToken.__module__ = "finn.dataflow.model.logical.semantics"

DATAFLOW_LOGICAL_RESULT_SEMANTICS: ValueSemantics[LogicalResult] = ValueSemantics(
    type_token=_LogicalResultToken,
    name="LogicalResult",
    recognizes=lambda value: isinstance(value, (RegionResult, NetworkResult)),
    equal=lambda left, right: bool(left == right),
    snapshot=lambda value: value,
)
__all__ = ["DATAFLOW_LOGICAL_RESULT_SEMANTICS"]
