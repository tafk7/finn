# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Reading one ONNX node as source facts, once, and freezing the result.

Every fact a DataflowOp's design space is allowed to depend on comes through
here: operand tensors, their shapes and QONNX datatypes, whether each carries an
initializer, and the node attributes the operation declares.  Reading them in
one place and freezing the result is what makes "an ordinary query does not
reread the live graph" a property of the layer rather than a habit each
operation has to keep.

Nothing here is MVAU-shaped.  An operation names its operands and its attribute
types; the extraction, the refusals and the frozen record are the same for all
of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from finn.dataflow.model.datatypes import QONNXDataType, canonical_qonnx_datatype

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onnx import NodeProto  # type: ignore[import-not-found]


class SourceError(ValueError):
    """One source node cannot be read as this operation's facts.

    A refusal about the *graph*, not about the design space: a missing operand,
    an unset shape, a datatype the model never annotated.  Kept distinct from
    ``AuthoringError`` because the caller's declarations are fine and their
    model is not, and from the engine's ``RequestError`` because nothing has
    reached an engine yet.
    """


@dataclass(frozen=True, slots=True)
class SourceOperand:
    """Frozen tensor metadata. Every present initializer contributes its digest."""

    id: str
    tensor: str
    shape: tuple[int, ...]
    datatype: QONNXDataType
    initializer: bool = False
    initializer_digest: str | None = None
    #: Whether the graph actually annotates this operand's shape.  ``False``
    #: for an output the operation has not written a shape for yet, which is
    #: the ordinary state *before* shape inference -- and shape inference is
    #: exactly the pass that asks the operation what the shape should be.
    #: Refusing there would make the operation require the annotation it exists
    #: to produce.
    annotated: bool = True

    @property
    def elements(self) -> int:
        total = 1
        for extent in self.shape:
            total *= extent
        return total


@dataclass(frozen=True, slots=True)
class SourceNode:
    """The frozen reading of one source node.

    Immutable and complete: once an operation holds one of these, the live
    ``ModelWrapper`` is not consulted again for anything the design space reads.
    A later graph edit therefore cannot change an answer under a caller; it can
    only make the occurrence stale, which is a different and detectable thing.
    """

    node_name: str
    op_type: str
    domain: str
    inputs: tuple[SourceOperand, ...]
    outputs: tuple[SourceOperand, ...]
    attributes: Mapping[str, object]

    def operand(self, operand_id: str) -> SourceOperand:
        for item in (*self.inputs, *self.outputs):
            if item.id == operand_id:
                return item
        raise SourceError(f"{self.node_name} has no operand {operand_id!r}")

    def has(self, operand_id: str) -> bool:
        return any(item.id == operand_id for item in (*self.inputs, *self.outputs))


def read_source_node(
    model: Any,
    node: NodeProto,
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
    summaries: Mapping[str, Any],
    optional_inputs: Sequence[str] = (),
    attributes: Mapping[str, object] | None = None,
) -> SourceNode:
    """Read one node using the model-level pass's initializer summaries.

    This reader never looks up initializer arrays or runs the bulk analysis.
    Output metadata is an observation and may be absent before inference.
    """

    if len(node.input) > len(inputs):
        raise SourceError(f"{node.name} has {len(node.input)} inputs; declares {len(inputs)}")
    if len(node.output) != len(outputs):
        raise SourceError(f"{node.name} has {len(node.output)} outputs; declares {len(outputs)}")

    def read(tensor: str, operand_id: str, *, output: bool) -> SourceOperand:
        shape = model.get_tensor_shape(tensor)
        if shape is None and not output:
            raise SourceError(f"{node.name} operand {operand_id!r} has no shape")
        datatype = model.get_tensor_datatype(tensor)
        if datatype is None:
            if not output:
                raise SourceError(f"{node.name} operand {operand_id!r} has no annotated datatype")
            # SourceOperand keeps the existing concrete datatype contract;
            # this placeholder is an observation only, never a Problem.
            from qonnx.core.datatype import DataType  # type: ignore[import-not-found] # noqa: PLC0415

            datatype = DataType["FLOAT32"]
        summary = None if output else summaries.get(tensor)
        return SourceOperand(
            operand_id,
            tensor,
            () if shape is None else tuple(int(extent) for extent in shape),
            canonical_qonnx_datatype(datatype),
            summary is not None,
            None if summary is None else summary.content_digest,
            shape is not None,
        )

    read_inputs: list[SourceOperand] = []
    for index, operand_id in enumerate(inputs):
        tensor = node.input[index] if index < len(node.input) else ""
        if not tensor:
            if operand_id in optional_inputs:
                continue
            raise SourceError(f"{node.name} requires operand {operand_id!r}")
        read_inputs.append(read(tensor, operand_id, output=False))
    read_outputs: list[SourceOperand] = []
    for tensor, operand_id in zip(node.output, outputs):
        if not tensor:
            raise SourceError(f"{node.name} requires output operand {operand_id!r}")
        read_outputs.append(read(tensor, operand_id, output=True))
    return SourceNode(
        node.name,
        node.op_type,
        node.domain,
        tuple(read_inputs),
        tuple(read_outputs),
        MappingProxyType(dict(attributes or {})),
    )


__all__ = ["SourceError", "SourceNode", "SourceOperand", "read_source_node"]
