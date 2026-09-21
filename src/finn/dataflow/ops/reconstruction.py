# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-level, in-memory source analysis and operation reconstruction.

Use source_analysis for read passes on one ModelWrapper. For inference, use
ops.inference's pass owners: an outer context cannot cover a wrapper that
ModelWrapper.transform copies or preprocesses before callbacks. bind_operations
and analyze_sources establish their own context. Nested readers consult its
single result; no result is serialized or retained on the model. A later pass
always observes current initializer contents.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from qonnx.analysis.tensor_value_summary import TensorValueSummary  # type: ignore[import-not-found]

from finn.dataflow.ops.tensor_summary import FrozenInitializer

if TYPE_CHECKING:
    from finn.dataflow.ops.base import DataflowOp
    from finn.dataflow.ops.source import SourceNode


@dataclass(frozen=True, slots=True)
class SourceAnalysis(Mapping[str, TensorValueSummary]):
    summaries: Mapping[str, TensorValueSummary]
    initializers: Mapping[str, FrozenInitializer]

    def __getitem__(self, key: str) -> TensorValueSummary:
        return self.summaries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.summaries)

    def __len__(self) -> int:
        return len(self.summaries)


_active: ContextVar[tuple[Any, SourceAnalysis] | None] = ContextVar(
    "dataflow_source_analysis", default=None
)


def initializer_facts(model: Any) -> SourceAnalysis:
    """Capture every initializer's summary and detached payload in one traversal."""

    summaries: dict[str, TensorValueSummary] = {}
    initializers: dict[str, FrozenInitializer] = {}
    for tensor in model.graph.initializer:
        # Conversion happens once inside this constructor. The resulting
        # summary is the same QONNX content identity used by source facts.
        frozen = FrozenInitializer.from_tensor_proto(tensor)
        summaries[tensor.name] = frozen.summary
        initializers[tensor.name] = frozen
    return SourceAnalysis(MappingProxyType(summaries), MappingProxyType(initializers))


@contextmanager
def source_analysis(model: Any, *, fresh: bool = False) -> Iterator[SourceAnalysis]:
    """Analyze every initializer once for this pass, including unused ones."""

    current = _active.get()
    if not fresh and current is not None and current[0] is model:
        yield current[1]
        return
    summaries = initializer_facts(model)
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
    model: Any,
    build: Any,
    *,
    operations: Sequence[DataflowOp] | None = None,
    graph_context: Any = None,
) -> tuple[DataflowOp, ...]:
    """Reconstruct all selected operations against one initializer analysis."""

    with source_analysis(model) as summaries:
        result = []
        for op in _operations(model, operations):
            context_read = None
            if graph_context is not None:
                from finn.dataflow.ops.graph_context import require_context_read  # noqa: PLC0415

                scope = op.recorded_scope_id()
                if not scope:
                    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

                    raise DataflowOpError(
                        f"{op.onnx_node.name!r} has no dataflow scope id; "
                        "run AssignDataflowScopeIds first"
                    )
                context_read = require_context_read(
                    graph_context,
                    model,
                    build,
                    consumer_scope_id=scope,
                )
            result.append(
                op._bind_with(
                    model,
                    op._build_values(build),
                    summaries,
                    context_read=context_read,
                )
            )
        return tuple(result)


def bind_sources_only(
    model: Any, build: Any, *, operations: Sequence[DataflowOp] | None = None
) -> tuple[DataflowOp, ...]:
    """Bind current source facts without hydrating persisted choices.

    This is useful for inspecting or repairing a source independently of its
    recorded selection. Ordinary bind/rebind remains strict.
    """

    with source_analysis(model) as summaries:
        return tuple(
            op._bind_with(model, op._build_values(build), summaries, recorded=False)
            for op in _operations(model, operations)
        )


def analyze_sources(model: Any) -> tuple[SourceNode, ...]:
    """Read the logical sources of the model without a build or hydration."""

    with source_analysis(model) as summaries:
        return tuple(
            op._read_source(model, op._live_node(model), summaries)
            for op in _operations(model, None)
        )


__all__ = [
    "SourceAnalysis",
    "analyze_sources",
    "bind_operations",
    "bind_sources_only",
    "initializer_facts",
    "source_analysis",
]
