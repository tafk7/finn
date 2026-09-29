# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX graphs as Designs: the graph adapter, for MatMul.

``graph_design`` reads a QONNX model once and declares a ``Design``: one
stream per tensor its nodes exchange (a graph input or output is a boundary
port, ``in0_V``, ``out0_V``, in graph order) and one ``MatMulKernel`` per
``MatMul`` node, built from the node's facts. A node's weights are its second
operand: an initializer becomes the kernel's ``weights`` (stored ``(k, n)``, as
ONNX stores them), read from the model itself; without one, the weights are a
stream, and the kernel's ``memory`` is pinned to ``none``. Leading activation
axes are rows (``(1, M, K)`` is ``M`` rows of ``K``).

A MatMul's result is the kernel's exact integer type, which flows to the
stream it produces and to every consumer; the model's annotation of that
tensor must admit it (an unannotated ``FLOAT32`` admits anything). Every other
operator is refused.

``finn_model`` rewrites each MatMul node as FINN's ``MVAU``, its attributes the
kernel's ``finn_attributes`` from the configured design, and annotates each
result tensor with the kernel's type. Nothing is read from the graph but its
structure: the attributes are the kernel's.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

from onnx import helper  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.core.space import composite, design_space
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.composite import Design
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel, exact_result_dtype
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock


FINN_DOMAIN = "finn.custom_op.fpgadataflow"


class GraphError(ValueError):
    """The graph cannot be read as a Design: an operator, operand or type it does not take."""


@dataclass(frozen=True)
class GraphDesign:
    """A Design of a graph: the kernel node of each graph node, and what the graph pins."""

    point: Any
    kernels: tuple[tuple[str, str], ...]
    pinned: tuple[tuple[str, object], ...] = ()


def _name(prefix: str, text: str) -> str:
    return prefix + re.sub(r"\W", "_", text)


def _dtype(model: ModelWrapper, tensor: str) -> QONNXDataType:
    return canonical_qonnx_datatype(model.get_tensor_datatype(tensor))


def _admits(annotated: QONNXDataType, exact: QONNXDataType) -> bool:
    if annotated.name == "FLOAT32":
        return True
    try:
        low, high = ordinary_integer_bounds(annotated)
    except Exception:
        return False
    need_low, need_high = ordinary_integer_bounds(exact)
    return low <= need_low and need_high <= high


def _rows(shape: tuple[int, ...], tensor: str) -> tuple[int, int]:
    if len(shape) < 2:
        raise GraphError(f"{tensor}: a MatMul operand is a matrix (rows, columns)")
    return prod(shape[:-1]), shape[-1]


def graph_design(
    model: ModelWrapper, *, target_dsp: DspBlock, target_period_ns: float
) -> GraphDesign:
    """A Design of ``model``'s MatMul nodes on the streams between them."""
    graph = model.graph
    inputs = [item.name for item in graph.input]
    outputs = [item.name for item in graph.output]
    element: dict[str, QONNXDataType] = {name: _dtype(model, name) for name in inputs}
    shapes: dict[str, tuple[int, int]] = {}
    members: dict[str, object] = {}
    streams: dict[str, Stream] = {}
    kernels: list[tuple[str, str]] = []
    pins: dict[str, object] = {}

    def stream(tensor: str) -> Stream:
        if tensor not in streams:
            port = None
            if tensor in inputs:
                port = f"in{inputs.index(tensor)}_V"
            elif tensor in outputs:
                port = f"out{outputs.index(tensor)}_V"
            carried = Tensor(shapes[tensor], ScalarEncoding(element[tensor]))
            declared = Stream(tensor=carried, port=port) if port else Stream(tensor=carried)
            streams[tensor] = declared
            members[_name("t_", tensor)] = declared
        return streams[tensor]

    for index, node in enumerate(graph.node):
        label = node.name or f"{node.op_type}_{index}"
        if node.op_type != "MatMul":
            raise GraphError(f"{label}: {node.op_type} has no kernel yet; MatMul only")
        a, b = node.input
        (y,) = node.output
        if model.get_initializer(a) is not None:
            raise GraphError(f"{label}: constant activations are not streamed")
        if a not in element:
            raise GraphError(f"{label}: {a} is produced by no node before it")
        m, k = _rows(tuple(model.get_tensor_shape(a)), a)
        k_b, n = _rows(tuple(model.get_tensor_shape(b)), b)
        if k != k_b:
            raise GraphError(f"{label}: {a} has {k} columns, {b} {k_b} rows")
        shapes[a], shapes[y] = (m, k), (m, n)
        weights_dtype = _dtype(model, b)
        try:
            result = exact_result_dtype(k, element[a], weights_dtype)
        except (ValueError, TypeError) as error:
            raise GraphError(f"{label}: {error}") from error
        if not _admits(_dtype(model, y), result):
            raise GraphError(
                f"{label}: {y} is annotated {_dtype(model, y).name}, narrower than the "
                f"exact result {result.name}"
            )
        element[y] = result
        facts: dict[str, object] = dict(
            m=m,
            n=n,
            k=k,
            activation_dtype=element[a],
            weights_dtype=weights_dtype,
            target_dsp=target_dsp,
            target_period_ns=target_period_ns,
            x_stream=stream(a),
            y_stream=stream(y),
        )
        name = _name("mm_", label)
        initializer = model.get_initializer(b)
        if initializer is None:
            shapes[b], element[b] = (k, n), weights_dtype
            facts["w_stream"] = stream(b)
            pins[f"{name}.memory"] = "none"
        else:
            values = initializer.reshape(k, n)
            if (values != values.round()).any():
                raise GraphError(f"{label}: the weights {b} are not integers")
            facts["weights"] = tuple(tuple(int(v) for v in row) for row in values)
        members[name] = MatMulKernel(**facts)  # type: ignore[arg-type]
        kernels.append((label, name))
    family = composite("Graph", members, base=Design)
    point = design_space(family())
    return GraphDesign(
        commit(point, pins) if pins else point, tuple(kernels), tuple(sorted(pins.items()))
    )


def finn_model(model: ModelWrapper, point: Any, kernels: Sequence[tuple[str, str]]) -> ModelWrapper:
    """``model`` with each MatMul node an ``MVAU`` carrying its kernel's attributes.

    ``point`` is the configured Design; ``kernels`` pairs each graph node with
    its kernel node (``GraphDesign.kernels``).
    """
    result = _copy(model)
    kernel = dict(kernels)
    for index, node in enumerate(list(result.graph.node)):
        label = node.name or f"{node.op_type}_{index}"
        configured = getattr(point, kernel[label])
        attributes = dict(configured.finn_attributes)
        replacement = helper.make_node(
            "MVAU",
            list(node.input),
            list(node.output),
            name=node.name or label,
            domain=FINN_DOMAIN,
            **{
                key: list(value) if isinstance(value, tuple) else value
                for key, value in attributes.items()
            },
        )
        result.graph.node.remove(node)
        result.graph.node.insert(index, replacement)
        result.set_tensor_datatype(node.output[0], configured.result_type)
    return result


def _copy(model: ModelWrapper) -> ModelWrapper:
    return ModelWrapper(model.model.SerializeToString())


__all__ = ["FINN_DOMAIN", "GraphDesign", "GraphError", "finn_model", "graph_design"]
