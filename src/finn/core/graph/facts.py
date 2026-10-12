# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a graph states of a tensor, as the KernelOps read their facts.

A tensor's datatype is its annotation, and an unannotated tensor is refused rather than
read as FLOAT32. Its shape is the one the graph states, every extent known, and its
leading axes are rows. An initializer of integers is admitted by its value summary
against its annotation, and its digest keys the node's binding; as a channel's tensor
it is stated over its values' range, in the encoding they need. A fact the graph does
not state raises ``FactUnstated``; a contradiction raises ``KernelOpError``.
"""

from __future__ import annotations

from array import array
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from qonnx.analysis.tensor_value_summary import (
    UnsupportedTensorValueError,
    initializer_value_summary,
)

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
    from qonnx.core.modelwrapper import ModelWrapper


def datatype(model: ModelWrapper, tensor: str, label: str) -> QONNXDataType:
    """The annotation of ``tensor``; an unannotated tensor is refused (qonnx reads it
    as FLOAT32, which is not a statement)."""
    if not model.has_tensor_datatype(tensor):
        produced = model.find_producer(tensor) is not None
        raise FactUnstated(
            f"{label}: {tensor} has no datatype annotation (absent is not FLOAT32"
            + ("; InferKernelTensors states each node's outputs in order)" if produced else ")"),
            tensor,
        )
    return canonical_qonnx_datatype(model.get_tensor_datatype(tensor))


def known_shape(model: ModelWrapper, tensor: str) -> tuple[int, ...] | None:
    """The shape of ``tensor`` if the graph states it: None for none, or for one with an
    extent it does not state (qonnx reads both as a list, the first as empty)."""
    found = model.get_tensor_shape(tensor)
    if not found or any(type(dim) is not int or dim < 1 for dim in found):
        return None
    return tuple(found)


def shape(model: ModelWrapper, tensor: str, label: str) -> tuple[int, ...]:
    """The shape of ``tensor``; one not known yet is refused."""
    found = known_shape(model, tensor)
    if found is None:
        raise FactUnstated(f"{label}: {tensor} has no shape yet (run InferKernelTensors)", tensor)
    return found


def rows(dims: tuple[int, ...]) -> tuple[int, int]:
    """Leading axes are rows: (1, M, K) is M rows of K."""
    return prod(dims[:-1]), dims[-1]


def edge_tensor(model: ModelWrapper, tensor: str, label: str) -> Tensor:
    """The tensor a channel carries, as the graph states it: its shape as rows and its
    annotation; an initializer of integers (``admitted``) as its owner states it, over
    its values' range in the encoding they need (``stored_element``: INT8-typed ternary
    weights are ``INT2 over [-1, 1]``)."""
    dtype = datatype(model, tensor, label)
    stored = model.get_initializer(tensor)
    element = (
        stored_element(dtype, (int(stored.min()), int(stored.max())))
        if stored is not None and is_ordinary_integer(dtype)
        else ScalarEncoding(dtype)
    )
    return Tensor(rows(shape(model, tensor, label)), element)


def admitted(model: ModelWrapper, tensor: str, dtype: QONNXDataType, label: str) -> str:
    """An initializer holds integers its annotation admits, by its value summary; the
    summary's ``content_digest``, which keys the node's binding."""
    try:
        summary = initializer_value_summary(model, tensor)
    except UnsupportedTensorValueError as error:
        raise KernelOpError(f"{label}: {tensor}: {error}") from error
    if summary is None:
        raise KernelOpError(f"{label}: {tensor} is not an initializer")
    try:
        low, high = ordinary_integer_bounds(dtype)
    except DatatypeError:
        # Not an ordinary integer (FLOAT32 holds 0.5): no contradiction, the kernel refuses it.
        return str(summary.content_digest)
    if not summary.is_integral:
        raise KernelOpError(f"{label}: {tensor} holds values that are not integers")
    if summary.minimum is None or summary.maximum is None:  # integral with no range: empty
        raise KernelOpError(f"{label}: {tensor} is empty: it holds no values")
    observed = (int(summary.minimum), int(summary.maximum))
    if not low <= observed[0] <= observed[1] <= high:
        raise KernelOpError(
            f"{label}: {tensor} is annotated {dtype.name} and holds values over {list(observed)}"
        )
    return str(summary.content_digest)


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
    "integer_tensor",
    "known_shape",
    "rows",
    "shape",
]
