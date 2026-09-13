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

from finn.dataflow.model.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.ops.tensor_summary import FrozenInitializer

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
    initializer_value: FrozenInitializer | None = None
    #: Whether the graph actually annotates this operand's shape.  ``False``
    #: for an output the operation has not written a shape for yet, which is
    #: the ordinary state *before* shape inference -- and shape inference is
    #: exactly the pass that asks the operation what the shape should be.
    #: Refusing there would make the operation require the annotation it exists
    #: to produce.
    annotated: bool = True
    #: The ONNX tensor carrier is independent of the QONNX logical datatype.
    carrier_dtype: int = 1
    #: Whether the QONNX logical datatype came from one explicit canonical
    #: finn_datatype annotation rather than ModelWrapper's carrier fallback.
    datatype_annotated: bool = True

    def __post_init__(self) -> None:
        if type(self.datatype_annotated) is not bool:
            raise ValueError("datatype_annotated must be bool")
        if type(self.carrier_dtype) is not int or self.carrier_dtype <= 0:
            raise ValueError("carrier_dtype must be a positive ONNX TensorProto enum")
        if self.initializer != (self.initializer_digest is not None):
            raise ValueError("initializer presence and digest must agree")
        if self.initializer != (self.initializer_value is not None):
            raise ValueError("initializer presence and frozen payload must agree")
        if (
            self.initializer_value is not None
            and self.initializer_digest != self.initializer_value.summary.content_digest
        ):
            raise ValueError("initializer digest and frozen payload must agree")
        if (
            self.initializer_value is not None
            and self.carrier_dtype != self.initializer_value.carrier_dtype
        ):
            raise ValueError("initializer and source operand carrier dtypes must agree")

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
    initializers: Mapping[str, FrozenInitializer] | None = None,
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
        datatype_annotations = [
            entry.value
            for annotation in model.graph.quantization_annotation
            if annotation.tensor_name == tensor
            for entry in annotation.quant_parameter_tensor_names
            if entry.key == "finn_datatype"
        ]
        if len(datatype_annotations) > 1:
            raise SourceError(
                f"{node.name} operand {operand_id!r} has duplicate logical datatype annotations"
            )
        datatype_annotated = len(datatype_annotations) == 1
        if datatype_annotated:
            try:
                datatype = resolve_qonnx_datatype_name(datatype_annotations[0])
            except Exception as error:
                raise SourceError(
                    f"{node.name} operand {operand_id!r} has an invalid logical datatype annotation"
                ) from error
            if datatype.name != datatype_annotations[0]:
                raise SourceError(
                    f"{node.name} operand {operand_id!r} has a noncanonical logical "
                    "datatype annotation"
                )
        else:
            datatype = model.get_tensor_datatype(tensor)
        if datatype is None:
            if not output:
                raise SourceError(f"{node.name} operand {operand_id!r} has no annotated datatype")
            # SourceOperand keeps the existing concrete datatype contract;
            # this placeholder is an observation only, never a Problem.
            from qonnx.core.datatype import DataType  # type: ignore[import-not-found] # noqa: PLC0415

            datatype = DataType["FLOAT32"]
        summary = None if output else summaries.get(tensor)
        initializer_value = None if output or initializers is None else initializers.get(tensor)
        value_info = model.get_tensor_valueinfo(tensor)
        carrier_dtype = (
            initializer_value.carrier_dtype
            if initializer_value is not None
            else int(value_info.type.tensor_type.elem_type)
            if value_info is not None
            else 1
            if output
            else 0
        )
        if carrier_dtype <= 0:
            raise SourceError(f"{node.name} operand {operand_id!r} has no ONNX carrier type")
        return SourceOperand(
            id=operand_id,
            tensor=tensor,
            shape=() if shape is None else tuple(int(extent) for extent in shape),
            datatype=canonical_qonnx_datatype(datatype),
            initializer=summary is not None,
            initializer_digest=None if summary is None else summary.content_digest,
            initializer_value=initializer_value,
            annotated=shape is not None,
            carrier_dtype=carrier_dtype,
            datatype_annotated=datatype_annotated,
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
