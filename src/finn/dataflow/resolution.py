# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-generic selected-dataflow references and resolution records."""

from __future__ import annotations

from dataclasses import dataclass
from finn.dataflow.design import DesignPoint, Engine, ValueSemantics
from finn.dataflow.network import DataflowNetwork


@dataclass(frozen=True)
class NetworkRef:
    """Selected network associated with one source-operation scope."""

    network_id: str
    network: DataflowNetwork
    source_association: object


DataflowOpResult = NetworkRef

DATAFLOW_OP_RESULT_SEMANTICS: ValueSemantics[object] = ValueSemantics(
    DataflowOpResult,
    "DataflowOpResult",
    lambda value: isinstance(value, NetworkRef),
    lambda left, right: left == right,
    lambda value: value,
)


@dataclass(frozen=True)
class ResolvedDataflowOp:
    """One re-created design point and its selected logical dataflow."""

    engine: Engine
    point: DesignPoint
    result: DataflowOpResult
    source_association: object
    source_scope_id: str


__all__ = [
    "DATAFLOW_OP_RESULT_SEMANTICS",
    "DataflowOpResult",
    "NetworkRef",
    "ResolvedDataflowOp",
]
