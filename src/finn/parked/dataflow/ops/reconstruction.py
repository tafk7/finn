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
from typing import TYPE_CHECKING, Any, cast

from qonnx.analysis.tensor_value_summary import TensorValueSummary  # type: ignore[import-not-found]

from finn.parked.dataflow.ops.tensor_summary import FrozenInitializer

if TYPE_CHECKING:
    from finn.parked.dataflow.ops.base import DataflowOp
    from finn.parked.dataflow.ops.space import DataflowSpace
    from finn.parked.dataflow.ops.source import SourceNode


def build_space(
    space_type: type[DataflowSpace],
    model: Any,
    node: Any,
    *,
    build: Any = None,
    graph_context: Any = None,
    recorded: bool = True,
    opset_version: int = 1,
    fresh: bool = False,
    frozen_build: Mapping[Any, object] | None = None,
    context_read: Any = None,
) -> DataflowSpace:
    """Freeze one current graph invocation into the existing immutable runtime."""
    from finn.parked.dataflow.ops.space import (  # noqa: PLC0415
        _BoundNode,
        _attribute,
        _datatype_attribute,
        _plain_attribute,
        _build_value,
        _value_info_bytes,
        _annotation_bytes,
        _positional,
        source_declarations,
    )
    from finn.parked.dataflow.ops.schema import OpInput, OpOutput, Attribute, DatatypeAttribute, BuildFact  # noqa: PLC0415
    from finn.parked.dataflow.ops.source import read_source_node  # noqa: PLC0415
    from finn.parked.dataflow.ops.native import SCOPE_ID_ATTRIBUTE, hydrate  # noqa: PLC0415

    scope_attribute = _attribute(node, SCOPE_ID_ATTRIBUTE)
    scope = "" if scope_attribute is None else scope_attribute.s.decode("utf-8")
    inputs: list[tuple[int, str]] = []
    outputs: list[tuple[int, str]] = []
    optional: list[str] = []
    attributes: dict[str, object] = {}
    declarations = source_declarations(space_type)
    for name, declaration in declarations:
        if isinstance(declaration, (OpInput, OpOutput)):
            (outputs if declaration.output else inputs).append((declaration.index, name))
            if isinstance(declaration, OpInput) and declaration.optional:
                optional.append(name)
        elif isinstance(declaration, DatatypeAttribute):
            attributes[name] = _datatype_attribute(node, name, declaration)
        elif isinstance(declaration, Attribute):
            attributes[name] = _plain_attribute(node, name, declaration)
    if graph_context is not None:
        from finn.parked.dataflow.ops.graph_context import require_context_read  # noqa: PLC0415

        context_read = require_context_read(graph_context, model, build, consumer_scope_id=scope)
    with source_analysis(model, fresh=fresh) as analysis:
        source = read_source_node(
            model,
            node,
            inputs=_positional(inputs),
            outputs=_positional(outputs),
            optional_inputs=optional,
            attributes=attributes,
            summaries=analysis,
            initializers=analysis.initializers,
        )
        state = _BoundNode(
            node.SerializeToString(deterministic=True),
            scope,
            opset_version,
            next(
                (
                    int(item.version)
                    for item in model.model.opset_import
                    if item.domain == node.domain
                ),
                None,
            ),
            source.outputs,
            tuple(_value_info_bytes(model, item.tensor) for item in source.outputs),
            tuple(_annotation_bytes(model, item.tensor) for item in source.outputs),
            context_read,
        )
        values = (
            dict(frozen_build)
            if frozen_build is not None
            else {
                declaration: value
                for name, declaration in declarations
                if isinstance(declaration, BuildFact)
                if (value := _build_value(space_type, name, declaration, build)) is not None
            }
        )
        values.update(space_type._additional_problem_values(source, scope_id=scope))
        if context_read is not None:
            values[space_type.incoming_graph_context] = context_read.incoming
        for name, declaration in declarations:
            if isinstance(declaration, OpInput) and source.has(name):
                operand = source.operand(name)
                values[declaration] = operand
                if operand.tensor in analysis:
                    values[declaration.value_summary] = analysis[operand.tensor]
            elif isinstance(declaration, (Attribute, DatatypeAttribute)):
                values[declaration] = source.attributes[name]
        space = space_type._start_frozen(values, state)
    if recorded:
        space = hydrate(space)
    if getattr(space_type, "implementation_binding", None) is not None:
        from finn.parked.dataflow.ops.binding import validate_bindings  # noqa: PLC0415

        validate_bindings(space)
    return cast("DataflowSpace", space)


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
    from finn.parked.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp  # noqa: PLC0415

    result = []
    for node in model.graph.node:
        if node.domain == DATAFLOW_DOMAIN:
            from qonnx.custom_op.registry import getCustomOp  # type: ignore[import-not-found] # noqa: PLC0415

            operation = getCustomOp(node)
            if isinstance(operation, DataflowOp):
                result.append(operation)
    return result


def bind_operations(
    model: Any,
    build: Any = None,
    *,
    operations: Sequence[DataflowOp] | None = None,
    graph_context: Any = None,
) -> tuple[DataflowSpace, ...]:
    """Build source Spaces for a pass using one initializer analysis."""
    with source_analysis(model):
        return tuple(
            build_space(
                op.space_type,
                model,
                op._live_node(model),
                build=build,
                graph_context=graph_context,
                opset_version=op.onnx_opset_version,
            )
            for op in _operations(model, operations)
        )


def bind_sources_only(
    model: Any,
    build: Any = None,
    *,
    operations: Sequence[DataflowOp] | None = None,
) -> tuple[DataflowSpace, ...]:
    """Read current source facts without applying saved assignments."""
    with source_analysis(model):
        return tuple(
            build_space(
                op.space_type,
                model,
                op._live_node(model),
                build=build,
                recorded=False,
                opset_version=op.onnx_opset_version,
            )
            for op in _operations(model, operations)
        )


def analyze_sources(model: Any) -> tuple[SourceNode, ...]:
    return tuple(space.source for space in bind_sources_only(model))


__all__ = [
    "SourceAnalysis",
    "analyze_sources",
    "bind_operations",
    "bind_sources_only",
    "build_space",
    "initializer_facts",
    "source_analysis",
]
