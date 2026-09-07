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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
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
    """One tensor an operation reads or writes, as the graph presents it.

    ``initializer`` records presence only.  The values themselves are the
    artifact layer's business, and an operation that folded them into its
    design space would make every query depend on megabytes of weights.

    ``initializer_digest`` is the one concession, and it is a scalar: an
    operand whose *values* change what gets built needs its identity to move
    when they do.  A digest computed once at read time does that without any
    array entering the design point -- which is the property that keeps a
    fingerprint cheap and an engine cache sound.
    """

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
    #: The scalars computed *from* initializer values at read time, by member
    #: name.  A name absent from this mapping has no answer here -- the operand
    #: was not supplied, carried no initializer, or the analysis declined --
    #: and reaches the design space as an absent Problem rather than a default.
    analyses: Mapping[str, object] = MappingProxyType({})

    def operand(self, operand_id: str) -> SourceOperand:
        for item in (*self.inputs, *self.outputs):
            if item.id == operand_id:
                return item
        raise SourceError(f"{self.node_name} has no operand {operand_id!r}")

    def has(self, operand_id: str) -> bool:
        return any(item.id == operand_id for item in (*self.inputs, *self.outputs))


def _tensor_facts(
    model: Any,
    tensor: str,
    operand_id: str,
    node_name: str,
    digest: bool,
    required: bool = True,
) -> tuple[tuple[int, ...], QONNXDataType, bool, str | None, bool]:
    shape = model.get_tensor_shape(tensor)
    if shape is None and required:
        raise SourceError(
            f"{node_name} operand {operand_id!r} has no shape; the graph must be "
            "shape-inferred before a dataflow operation reads it"
        )
    datatype = model.get_tensor_datatype(tensor)
    if datatype is None:
        raise SourceError(f"{node_name} operand {operand_id!r} has no annotated datatype")
    initializer = model.get_initializer(tensor)
    return (
        () if shape is None else tuple(int(extent) for extent in shape),
        canonical_qonnx_datatype(datatype),
        initializer is not None,
        _digest(initializer) if digest and initializer is not None else None,
        shape is not None,
    )


def _digest(values: Any) -> str:
    """A stable digest of one initializer, computed once and never stored whole.

    Shape and dtype travel with the bytes, because two arrays with the same
    buffer and different shapes are different weights.  ``ascontiguousarray``
    is not cosmetic: a transposed view has the same buffer as its base, and
    hashing that buffer would give two genuinely different matrices one
    identity.
    """

    import numpy  # type: ignore[import-not-found] # noqa: PLC0415 - heavy import

    contiguous = numpy.ascontiguousarray(values)
    payload = sha256()
    payload.update(str(contiguous.dtype).encode("utf-8"))
    payload.update(str(contiguous.shape).encode("utf-8"))
    payload.update(contiguous.tobytes())
    return payload.hexdigest()


def read_source_node(
    model: Any,
    node: NodeProto,
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
    optional_inputs: Sequence[str] = (),
    digest_inputs: Sequence[str] = (),
    attributes: Mapping[str, object] | None = None,
    analyses: Mapping[str, tuple[str, Callable[[Any, QONNXDataType], object]]] | None = None,
) -> SourceNode:
    """Read one node's operands and attributes into a frozen record.

    ``inputs`` and ``outputs`` are the operation's own operand names, in ONNX
    positional order.  An operand named in ``optional_inputs`` may be absent or
    empty -- ONNX spells "not supplied" as an empty string -- and is simply not
    present in the result, which is what lets applicability depend on it
    without anyone inventing a placeholder tensor.
    """

    optional = set(optional_inputs)
    digested = set(digest_inputs)
    if len(node.input) > len(inputs):
        raise SourceError(
            f"{node.name} has {len(node.input)} inputs; {node.op_type} declares {len(inputs)}"
        )
    if len(node.output) != len(outputs):
        raise SourceError(
            f"{node.name} has {len(node.output)} outputs; {node.op_type} declares {len(outputs)}"
        )

    read_inputs: list[SourceOperand] = []
    for index, operand_id in enumerate(inputs):
        tensor = node.input[index] if index < len(node.input) else ""
        if not tensor:
            if operand_id in optional:
                continue
            raise SourceError(f"{node.name} requires operand {operand_id!r}")
        shape, datatype, initializer, digest, annotated = _tensor_facts(
            model, tensor, operand_id, node.name, operand_id in digested
        )
        read_inputs.append(
            SourceOperand(operand_id, tensor, shape, datatype, initializer, digest, annotated)
        )

    read_outputs: list[SourceOperand] = []
    for index, operand_id in enumerate(outputs):
        tensor = node.output[index]
        if not tensor:
            raise SourceError(f"{node.name} requires output operand {operand_id!r}")
        # An output annotation is an *observation*: the operation derives what
        # it should be, and shape inference is the pass that asks.  Requiring
        # it here would make the operation demand the very thing it produces.
        shape, datatype, initializer, digest, annotated = _tensor_facts(
            model, tensor, operand_id, node.name, False, required=False
        )
        read_outputs.append(
            SourceOperand(operand_id, tensor, shape, datatype, initializer, digest, annotated)
        )

    return SourceNode(
        node.name,
        node.op_type,
        node.domain,
        tuple(read_inputs),
        tuple(read_outputs),
        MappingProxyType(dict(attributes or {})),
        MappingProxyType(_run_analyses(model, read_inputs, analyses or {})),
    )


def _run_analyses(
    model: Any,
    operands: Sequence[SourceOperand],
    analyses: Mapping[str, tuple[str, Callable[[Any, QONNXDataType], object]]],
) -> dict[str, object]:
    """Compute each declared initializer analysis, once, and keep only its result.

    The array is fetched here and dropped here.  An operand that is absent or
    carries no initializer, and an analysis that returns ``None``, are all the
    same outcome: no entry, and therefore an absent Problem -- never a zero or
    a ``False`` that a reader would happily compute with.
    """

    by_id = {operand.id: operand for operand in operands}
    results: dict[str, object] = {}
    for member_name, (operand_id, evaluate) in analyses.items():
        operand = by_id.get(operand_id)
        if operand is None or not operand.initializer:
            continue
        values = model.get_initializer(operand.tensor)
        if values is None:
            continue
        result = evaluate(values, operand.datatype)
        if result is not None:
            results[member_name] = result
    return results


__all__ = [
    "SourceError",
    "SourceNode",
    "SourceOperand",
    "read_source_node",
]
