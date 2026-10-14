# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the KernelOps read of a tensor: its datatype annotation with its range, its
shape, and the values of an initializer of integers; and what they write of one, its
element (``state``). A missing shape or annotation raises ``FactUnstated``; anything
that contradicts the graph raises ``KernelOpError``.

A tensor's range is part of its datatype fact (``qonnx.core.ranges``, stored on the
model by ``qonnx.core.tensor_ranges``): written with the datatype in one call
(``ModelWrapper.set_tensor_datatype``), so that a datatype stated without its range
clears it. A tensor that states none reads as its datatype's full range.

Inside ``reading(model)`` (a network's bind), the model's tensors are read from one index,
built once: its value infos, its initializers (each converted to an array once, shared
and read-only) and, through ``qonnx.core.tensor_ranges.one_pass``, its annotations.
Nothing is kept after it, and the model must not change inside it.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from onnx import numpy_helper
from qonnx.analysis.tensor_value_summary import (
    UnsupportedTensorValueError,
    summarize_tensor_values,
)
from qonnx.core import tensor_ranges
from qonnx.core.metadata import MetadataError

from finn.core.graph.errors import FactUnstated, KernelOpError
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    is_ordinary_integer,
    ordinary_integer_bounds,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.values.domains import stored_element
from finn.kernels.values.semantics import IntegerTensorValue, integer_bytes, integer_digest

if TYPE_CHECKING:
    from onnx import TensorProto, ValueInfoProto
    from qonnx.core.modelwrapper import ModelWrapper


class _Index:
    """A model's value infos and initializers by name, each initializer's array once."""

    def __init__(self, model: ModelWrapper) -> None:
        graph = model.graph
        self.model = model
        self.infos: dict[str, ValueInfoProto] = {}
        self.repeated: set[str] = set()
        for info in [*graph.input, *graph.output, *graph.value_info]:
            if info.name in self.infos:
                self.repeated.add(info.name)
            self.infos[info.name] = info
        self.protos: dict[str, TensorProto] = {item.name: item for item in graph.initializer}
        self.arrays: dict[str, npt.NDArray[Any]] = {}


_INDEX: ContextVar[_Index | None] = ContextVar("finn_graph_facts_index", default=None)


@contextmanager
def reading(model: ModelWrapper) -> Iterator[None]:
    """Read ``model``'s tensors from one index while inside (see the module docstring)."""
    with tensor_ranges.one_pass(model):
        token = _INDEX.set(_Index(model))
        try:
            yield
        finally:
            _INDEX.reset(token)


def _index(model: ModelWrapper) -> _Index | None:
    current = _INDEX.get()
    return current if current is not None and current.model is model else None


def initializer(model: ModelWrapper, tensor: str) -> npt.NDArray[Any] | None:
    """The values of the initializer ``tensor``, or None if it is none
    (``ModelWrapper.get_initializer``); inside ``reading``, one read-only array."""
    index = _index(model)
    if index is None:
        return model.get_initializer(tensor)
    if tensor not in index.arrays:
        proto = index.protos.get(tensor)
        if proto is None:
            return None
        found = numpy_helper.to_array(proto)
        found.setflags(write=False)
        index.arrays[tensor] = found
    return index.arrays[tensor]


def _shape(model: ModelWrapper, tensor: str) -> list[int] | None:
    """``ModelWrapper.get_tensor_shape``, by the index inside ``reading``."""
    index = _index(model)
    if index is None or tensor in index.repeated:
        return model.get_tensor_shape(tensor)
    info = index.infos.get(tensor)
    if info is not None:
        return [dim.dim_value for dim in info.type.tensor_type.shape.dim]
    stored = initializer(model, tensor)
    return None if stored is None else list(stored.shape)


def _stated_datatype(model: ModelWrapper, tensor: str) -> bool:
    """``ModelWrapper.has_tensor_datatype``, by qonnx's index inside ``reading``."""
    found = tensor_ranges.annotation(model, tensor)
    return found is not None and any(
        entry.key == "finn_datatype" for entry in found.quant_parameter_tensor_names
    )


def datatype(model: ModelWrapper, tensor: str, label: str) -> QONNXDataType:
    """The datatype annotation of ``tensor``. A tensor without one is refused, not read
    as FLOAT32 as qonnx would."""
    if not _stated_datatype(model, tensor):
        produced = model.find_producer(tensor) is not None
        raise FactUnstated(
            "datatype",
            f"{label}: {tensor} has no datatype annotation (absent is not FLOAT32"
            + ("; InferKernelTensors states each node's outputs in order)" if produced else ")"),
            tensor,
            label,
        )
    return canonical_qonnx_datatype(tensor_ranges.datatype(model, tensor))


def known_shape(model: ModelWrapper, tensor: str) -> tuple[int, ...] | None:
    """The shape of ``tensor``, or None if the graph has none (qonnx gives an empty
    list) or one with an unknown extent."""
    found = _shape(model, tensor)
    if not found or any(type(dim) is not int or dim < 1 for dim in found):
        return None
    return tuple(found)


def shape(model: ModelWrapper, tensor: str, label: str) -> tuple[int, ...]:
    """The shape of ``tensor``; one not known yet is refused."""
    found = known_shape(model, tensor)
    if found is None:
        raise FactUnstated(
            "shape", f"{label}: {tensor} has no shape yet (run InferKernelTensors)", tensor, label
        )
    return found


def element(model: ModelWrapper, tensor: str, label: str) -> ScalarEncoding:
    """The element of ``tensor``, not an initializer: its datatype over the range it
    states with it (``qonnx.core.tensor_ranges.stated``), else over its datatype's full range.
    A stored range that is malformed, or that its datatype does not hold, is refused."""
    dtype = datatype(model, tensor, label)
    try:
        found = tensor_ranges.stated(model, tensor)
        return ScalarEncoding(dtype, found)
    except (MetadataError, ValueError) as error:
        raise KernelOpError.refused(
            "range-invalid", "element", f"{label}: {tensor}: {error}", node=label, tensor=tensor
        ) from error


def edge_tensor(model: ModelWrapper, tensor: str, label: str) -> Tensor:
    """The tensor as a fact, which a kernel's port reads and a channel carries: the
    graph's shape, whole (a kernel that reads another view of it, MatMul's rows, states
    so), and its element, its datatype over its range (``element``: its producer's
    statement, as inference wrote it). For an initializer of integers, the datatype is
    narrowed to its values' range (``stored_element``: INT8-typed ternary weights become
    INT2 over [-1, 1])."""
    stored = initializer(model, tensor)
    if stored is None:
        found = element(model, tensor, label)
    else:
        dtype = datatype(model, tensor, label)
        found = (
            stored_element(dtype, (int(stored.min()), int(stored.max())))
            if is_ordinary_integer(dtype)
            else ScalarEncoding(dtype)
        )
    return Tensor(shape(model, tensor, label), found)


def state(model: ModelWrapper, tensor: str, stated: ScalarEncoding) -> None:
    """Write ``stated`` as ``tensor``'s datatype and range, one fact in one call
    (``set_tensor_datatype``): a range no tighter than the datatype's is stored as none,
    and clears one stored before."""
    model.set_tensor_datatype(tensor, stated.dtype, stated.value_range)


def admitted(model: ModelWrapper, tensor: str, dtype: QONNXDataType, label: str) -> None:
    """Check that the initializer ``tensor`` holds integers its datatype ``dtype``
    allows."""
    stored = initializer(model, tensor)
    if stored is None:
        raise KernelOpError.refused(
            "initializer-absent",
            "admitted",
            f"{label}: {tensor} is not an initializer",
            node=label,
            tensor=tensor,
        )
    try:
        summary = summarize_tensor_values(stored)
    except UnsupportedTensorValueError as error:
        raise KernelOpError.refused(
            "initializer-unsupported",
            "admitted",
            f"{label}: {tensor}: {error}",
            node=label,
            tensor=tensor,
        ) from error
    try:
        low, high = ordinary_integer_bounds(dtype)
    except DatatypeError:
        # Not an ordinary integer (FLOAT32 holds 0.5): no contradiction, the kernel refuses it.
        return
    if not summary.is_integral:
        raise KernelOpError.refused(
            "initializer-not-integer",
            "admitted",
            f"{label}: {tensor} holds values that are not integers",
            node=label,
            tensor=tensor,
        )
    if summary.minimum is None or summary.maximum is None:  # integral with no range: empty
        raise KernelOpError.refused(
            "initializer-empty",
            "admitted",
            f"{label}: {tensor} is empty: it holds no values",
            node=label,
            tensor=tensor,
        )
    observed = (int(summary.minimum), int(summary.maximum))
    if not low <= observed[0] <= observed[1] <= high:
        raise KernelOpError.refused(
            "initializer-out-of-datatype",
            "admitted",
            f"{label}: {tensor} is annotated {dtype.name} and holds values over {list(observed)}",
            node=label,
            tensor=tensor,
        )


def integer_tensor(values: npt.NDArray[Any]) -> IntegerTensorValue:
    """An initializer of integers (``admitted``) as an integer tensor value: its shape,
    range and digest read from the array, its integers loaded only when first read (a
    memory image packs them)."""
    found = np.asarray(values)
    shape = tuple(int(extent) for extent in found.shape)
    least, greatest = int(found.min()), int(found.max())
    if not -(2**63) <= least <= greatest < 2**63:
        return IntegerTensorValue.flat(shape, [int(value) for value in found.ravel()])
    stored = array("q")
    stored.frombytes(found.astype("=i8").tobytes())
    digest = integer_digest(shape, integer_bytes(stored))
    return IntegerTensorValue(shape, (least, greatest), digest, lambda: stored)


__all__ = [
    "admitted",
    "datatype",
    "edge_tensor",
    "element",
    "initializer",
    "integer_tensor",
    "known_shape",
    "reading",
    "shape",
    "state",
]
