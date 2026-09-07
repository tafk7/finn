# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-level, in-memory source analysis and operation reconstruction.

Use one source_analysis context around model-wide QONNX inference/execution
passes. bind_operations and analyze_sources establish that context themselves.
Nested readers consult its single result; no result is serialized or retained
on the model. A later pass always observes current initializer contents.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from qonnx.analysis.tensor_value_summary import (  # type: ignore[import-not-found]
    TensorValueSummary,
    initializer_value_summaries,
)

if TYPE_CHECKING:
    from finn.dataflow.ops.base import DataflowOp
    from finn.dataflow.ops.source import SourceNode

_active: ContextVar[tuple[Any, Mapping[str, TensorValueSummary]] | None] = ContextVar(
    "dataflow_source_analysis", default=None
)


@contextmanager
def source_analysis(
    model: Any, *, fresh: bool = False
) -> Iterator[Mapping[str, TensorValueSummary]]:
    """Analyze every initializer once for this pass, including unused ones."""

    current = _active.get()
    if not fresh and current is not None and current[0] is model:
        yield current[1]
        return
    summaries = MappingProxyType(initializer_value_summaries(model))
    token = _active.set((model, summaries))
    try:
        yield summaries
    finally:
        _active.reset(token)


def _operations(model: Any, operations: Sequence[DataflowOp] | None) -> Sequence[DataflowOp]:
    if operations is not None:
        return operations
    from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp  # noqa: PLC0415

    result = []
    for node in model.graph.node:
        if node.domain == DATAFLOW_DOMAIN:
            operation = model.get_customop_wrapper(node)
            if isinstance(operation, DataflowOp):
                result.append(operation)
    return result


def bind_operations(
    model: Any, build: Any, *, operations: Sequence[DataflowOp] | None = None
) -> tuple[DataflowOp, ...]:
    """Reconstruct all selected operations against one initializer analysis."""

    with source_analysis(model) as summaries:
        return tuple(
            op._bind_with(model, op._build_values(build), summaries)
            for op in _operations(model, operations)
        )


def analyze_sources(model: Any) -> tuple[SourceNode, ...]:
    """Read the logical sources of the model without a build or hydration."""

    with source_analysis(model) as summaries:
        return tuple(
            op._read_source(model, op._live_node(model), summaries)
            for op in _operations(model, None)
        )


__all__ = ["analyze_sources", "bind_operations", "source_analysis"]
