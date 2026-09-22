# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh producer type queries for graph inputs and checked annotation updates.

Graph datatype annotations on DataflowOp outputs are derived caches. Reading a
consumer input therefore asks its producer's public contract, even when an old
annotation is present. This service owns no persistent cache and never mutates
the graph during a query.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from finn.dataflow._engine import Answer, Finding, FindingKind, QualifiedPath, Unresolved
from finn.dataflow.model.logical.datatypes import QONNXDataType

if TYPE_CHECKING:
    from finn.dataflow.ops.model_effects import ModelReadSet


_active_types: ContextVar[tuple[tuple[int, str], ...]] = ContextVar(
    "dataflow_producer_type_queries", default=()
)


def _unavailable(code: str, message: str) -> Unresolved:
    return Unresolved(
        (Finding(FindingKind.LIMITATION, code, QualifiedPath("op.producer_type"), message),)
    )


def producer_type(
    model: Any, tensor: str, build: object | None = None
) -> Answer[QONNXDataType] | None:
    return producer_type_facts(model, tensor, build)[0]


def producer_type_facts(
    model: Any, tensor: str, build: object | None = None
) -> tuple[Answer[QONNXDataType] | None, ModelReadSet | None]:
    """Resolve a Dataflow producer's current type, independently of annotations.

    ``None`` means this value has no Dataflow producer and its external annotation
    remains an input fact. An unavailable producer returns an Answer, never None
    or a carrier-type fallback. Recursion is scoped to this synchronous query.
    """

    from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError  # noqa: PLC0415
    from finn.dataflow.ops.space import source_declarations  # noqa: PLC0415
    from finn.dataflow.ops.schema import OpOutput  # noqa: PLC0415

    producers = tuple(node for node in model.graph.node if tensor in node.output)
    if not any(node.domain == DATAFLOW_DOMAIN for node in producers):
        return None, None
    if len(producers) != 1:
        raise DataflowOpError(
            f"graph tensor {tensor!r} requires exactly one producer, found {len(producers)}"
        )
    node = producers[0]
    active = _active_types.get()
    key = (id(model), tensor)
    if key in active:
        return _unavailable(
            "producer-type-cycle", f"producer type dependencies cycle through {tensor!r}"
        ), None
    token = _active_types.set((*active, key))
    try:
        operation = model.get_customop_wrapper(node)
        if not isinstance(operation, DataflowOp):
            return _unavailable(
                "producer-type-definition-missing",
                f"no DataflowOp definition interprets producer of {tensor!r}",
            ), None
        outputs = tuple(
            name
            for name, declaration in source_declarations(operation.space_type)
            if isinstance(declaration, OpOutput)
            and declaration.index < len(node.output)
            and node.output[declaration.index] == tensor
        )
        if len(outputs) != 1:
            return _unavailable(
                "producer-type-output-ambiguous",
                f"producer does not declare exactly one public output for {tensor!r}",
            ), None
        from finn.dataflow.ops.persistence import source_read_set  # noqa: PLC0415

        if build is not None:
            operation.set_context(build=build)
        current = operation.space
        answer = cast("Answer[QONNXDataType]", current.operand_type(outputs[0]))
        # Source type rules can consume semantic attributes and authenticated
        # initializer contents. Include their frozen reads recursively, without
        # promoting cached producer output annotations into dependencies.
        return answer, source_read_set(
            current,
            expected_attributes={},
            include_output_annotations=False,
            node_output_address=tensor,
        )
    finally:
        _active_types.reset(token)


__all__ = ["producer_type", "producer_type_facts"]
