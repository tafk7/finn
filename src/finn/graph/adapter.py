# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX graphs as kernels: the graph adapter, for MatMul.

``graph_design`` reads a QONNX model once and declares the root of its
hardware, a ``Kernel`` (``finn.graph``) with children: one stream per tensor
its nodes exchange (a graph input or output is a boundary port, ``in0_V``,
``out0_V``, in graph order), one ``MatMulKernel`` per ``MatMul`` node, built
from the node's facts, and each MatMul's weight stream. A node's weights are
its second operand: an initializer becomes the kernel's ``weights`` (stored
``(k, n)``, as ONNX stores them), read from the model itself, which its memory
streams into a weight stream of its own (``w_<node>``), and the kernel's
``memory`` is pinned to ``memstream``; without one, the weights are the stream
of that graph tensor, and ``memory`` is pinned to ``none``. Streaming known
weights from the host is not a memory choice: it is a rewrite of the graph that
lifts the initializer to a graph input. Each weight stream may buffer its words
in a FIFO (``<stream>.transport``). Leading activation axes are rows
(``(1, M, K)`` is ``M`` rows of ``K``).

Each MatMul's result tensor is inferred node by node from the kernel's facts
alone (``MatMulKernel.result_tensor``), before any stream exists, and so is a
stored weight stream's (``weight_tensor``); a node the kernel refuses is a
``GraphError``. The result is the kernel's exact integer type, which flows to
the stream it produces and to every consumer; the model's annotation of that
tensor must admit it (an unannotated ``FLOAT32`` admits anything). Every other
operator is refused. Provisional: the kernel's own ONNX operator will infer
this.

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

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.space import Available, composite, design_space
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.configure import commit, describe
from finn.kernels.matmul import MatMulKernel
from finn.kernels.streams import BufferedStream, Stream
from finn.kernels.target import DspBlock


FINN_DOMAIN = "finn.custom_op.fpgadataflow"


class GraphError(ValueError):
    """The graph cannot be read as kernels: an operator, operand or type it does not take."""


class Graph(Kernel):
    """A graph's hardware: its streams and the kernels on them."""

    id = "finn.graph"
    version = "1"


@dataclass(frozen=True)
class GraphDesign:
    """A graph's root, configured as far as the graph decides: the kernel node of each
    graph node, its weight stream node, and what the graph pins."""

    point: Any
    kernels: tuple[tuple[str, str], ...]
    pinned: tuple[tuple[str, object], ...] = ()
    weights: tuple[tuple[str, str], ...] = ()


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


def _inferred(label: str, facts: dict[str, object]) -> tuple[Tensor, Tensor]:
    """A MatMul node's result and weight tensors, from its facts alone: no stream is read."""
    try:
        point = design_space(MatMulKernel(**facts))  # type: ignore[arg-type]
    except (ValueError, TypeError) as error:
        raise GraphError(f"{label}: {error}") from error
    answers = [point.query(MatMulKernel.result_tensor), point.query(MatMulKernel.weight_tensor)]
    tensors = [answer.value for answer in answers if isinstance(answer, Available)]
    if len(tensors) != len(answers):
        raise GraphError(f"{label}: {describe(answers)}")
    return tensors[0], tensors[1]


def graph_design(
    model: ModelWrapper, *, target_dsp: DspBlock, target_period_ns: float
) -> GraphDesign:
    """The root of ``model``'s MatMul nodes on the streams between them."""
    graph = model.graph
    inputs = [item.name for item in graph.input]
    outputs = [item.name for item in graph.output]
    element: dict[str, QONNXDataType] = {name: _dtype(model, name) for name in inputs}
    shapes: dict[str, tuple[int, ...]] = {}
    members: dict[str, object] = {}
    streams: dict[str, Stream] = {}
    kernels: list[tuple[str, str]] = []
    weights: list[tuple[str, str]] = []
    pins: dict[str, object] = {}

    def stream(tensor: str, kind: type[Stream] = Stream) -> Stream:
        if tensor not in streams:
            port = None
            if tensor in inputs:
                port = f"in{inputs.index(tensor)}_V"
            elif tensor in outputs:
                port = f"out{outputs.index(tensor)}_V"
            carried = Tensor(shapes[tensor], ScalarEncoding(element[tensor]))
            declared = kind(tensor=carried, port=port) if port else kind(tensor=carried)
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
        shapes[a] = (m, k)
        weights_dtype = _dtype(model, b)
        facts: dict[str, object] = dict(
            m=m,
            n=n,
            k=k,
            activation_dtype=element[a],
            weights_dtype=weights_dtype,
            target_dsp=target_dsp,
            target_period_ns=target_period_ns,
        )
        initializer = model.get_initializer(b)
        if initializer is not None:
            values = initializer.reshape(k, n)
            if (values != values.round()).any():
                raise GraphError(f"{label}: the weights {b} are not integers")
            facts["weights"] = tuple(tuple(int(v) for v in row) for row in values)
        # The node's inference, by hand until the kernel has its ONNX operator.
        result, stored = _inferred(label, facts)
        exact = result.element.dtype
        if not _admits(_dtype(model, y), exact):
            raise GraphError(
                f"{label}: {y} is annotated {_dtype(model, y).name}, narrower than the "
                f"exact result {exact.name}"
            )
        shapes[y], element[y] = result.shape, exact
        name = _name("mm_", label)
        facts |= dict(x_stream=stream(a), y_stream=stream(y))
        if initializer is None:
            shapes[b], element[b] = (k, n), weights_dtype
            facts["w_stream"] = stream(b, BufferedStream)
            weights.append((name, _name("t_", b)))
            pins[f"{name}.memory"] = "none"
        else:
            # The memory's own stream into the core: an edge of this kernel alone.
            members[_name("w_", label)] = facts["w_stream"] = BufferedStream(tensor=stored)
            weights.append((name, _name("w_", label)))
            pins[f"{name}.memory"] = "memstream"
        members[name] = MatMulKernel(**facts)  # type: ignore[arg-type]
        kernels.append((label, name))
    family = composite("Graph", members, base=Graph)
    point = design_space(family())
    return GraphDesign(
        commit(point, pins) if pins else point,
        tuple(kernels),
        tuple(sorted(pins.items())),
        tuple(weights),
    )


def finn_model(model: ModelWrapper, point: Any, kernels: Sequence[tuple[str, str]]) -> ModelWrapper:
    """``model`` with each MatMul node an ``MVAU`` carrying its kernel's attributes.

    ``point`` is the configured root; ``kernels`` pairs each graph node with
    its kernel node (``GraphDesign.kernels``).
    """
    result = _copy(model)
    kernel = dict(kernels)
    for index, node in enumerate(list(result.graph.node)):
        label = node.name or f"{node.op_type}_{index}"
        configured = getattr(point, kernel[label])
        attributes: dict[str, Any] = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dict(configured.finn_attributes).items()
        }
        replacement = helper.make_node(
            "MVAU",
            list(node.input),
            list(node.output),
            name=node.name or label,
            domain=FINN_DOMAIN,
            **attributes,
        )
        result.graph.node.remove(node)
        result.graph.node.insert(index, replacement)
        result.set_tensor_datatype(node.output[0], configured.result_type)
    return result


def _copy(model: ModelWrapper) -> ModelWrapper:
    return ModelWrapper(model.model.SerializeToString())


__all__ = ["FINN_DOMAIN", "Graph", "GraphDesign", "GraphError", "finn_model", "graph_design"]
