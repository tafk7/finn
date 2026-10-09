# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The generated hardware against its KernelOps' oracle, at the ops' boundary.

PRINCIPLES 8: a kernel declares its hardware's design space, never its
computation; the KernelOp's ``execute_node`` is the op's computational
reference. ``check_parity`` checks that the hardware a model of KernelOp nodes
generates computes what those ops compute, on the same inputs:

1. **the point**: the model's shell root, its nodes' persisted choices
   replayed and the rest completed by ``completion`` (``Baseline()`` by
   default), as packaging builds it (``configured_root``): a chosen design
   point is the choices saved on the nodes;
2. **the oracle**: each graph output as the nodes' ``execute_node`` computes it,
   the model run by ``finn.core.onnx_exec.execute_onnx`` with its default executor
   (``Python()``) on the inputs as integers (int64, which hold every value of a type
   the kernels admit: a KernelOp's reference is exact on them, MatMul's integer
   product included); every output must be an integer the boundary's element holds,
   or the hardware cannot present it (``Disagreement``);
3. **the hardware**: the model as the body of one partition node (``as_partition``),
   run by ``execute_onnx`` with the XSim executor
   (``finn.core.executors.xsim.executor.XSim``, paced by ``pacing``) alone, the run
   requiring hardware: every graph input streamed at its boundary port in the order
   the port presents, and every output read back from its words. Each output is
   compared with the oracle's word for word, as its port presents both. Words that
   differ are decoded against the oracle as an order (``finn.harness.orders``): a
   boundary walked otherwise than declared is named by beat and index tuple.

So parity is two runs of the model compared: ``execute_onnx(partition, inputs,
executors=(XSim(pacing=...),), require_hardware=True)`` against
``execute_onnx(model, inputs)``.

A test may state ``reference`` in place of the oracle, an integer reference of
its own. ``finn.harness.reference`` checks a KernelOp's reference against ONNX;
here the reference is what the hardware must compute.

``boundary_inputs`` draws the inputs from each graph input's datatype: random
integers of its range, seeded, and its extremes, the first row (the last axis's
vector) all its least value and the second all its greatest; one row holds both.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.executors.xsim.executor import XSim
from finn.core.executors.xsim.pacing import STALLED, Pacing
from finn.core.executors.xsim.rtl import WordsDiffer
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.kernels.base import datatype, kernel_op
from finn.custom_op.kernels.base import shape as tensor_shape
from finn.custom_op.kernels.shell import member
from finn.custom_op.partition.kernel_partitions import PARTITION_DOMAIN, PARTITION_OP
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.traversal import pack
from finn.harness.orders import Stream, decode
from finn.kernels.explore import Completion
from finn.transformation.kernels.package import configured_root

Integers = npt.NDArray[np.int64]
Reference = Callable[[Mapping[str, Integers]], Mapping[str, Any]]
"""Integer inputs by graph input to outputs by graph output."""


class Disagreement(AssertionError):
    """The oracle's outputs are not values the hardware can present: not integers, or
    outside the boundary's element. An ``AssertionError``: parity fails."""


def graph_inputs(model: ModelWrapper) -> list[str]:
    """The graph's inputs that are no initializer, in order: the boundary's inputs."""
    initializers = {tensor.name for tensor in model.graph.initializer}
    return [item.name for item in model.graph.input if item.name not in initializers]


def boundary_inputs(model: ModelWrapper, seed: int) -> dict[str, Integers]:
    """Each graph input's values: random integers of its datatype, seeded, with its
    extremes in its first two rows (module docstring)."""
    rng = np.random.default_rng(seed)
    found = {}
    for name in graph_inputs(model):
        low, high = ordinary_integer_bounds(datatype(model, name, name))
        shape = tensor_shape(model, name, name)
        values = rng.integers(low, high + 1, size=shape, dtype=np.int64)
        rows = values.reshape(prod(shape[:-1]), shape[-1])
        if rows.shape[0] >= 2:
            rows[0], rows[1] = low, high
        else:
            rows[0, 0], rows[0, -1] = low, high
        found[name] = values
    return found


def oracle(model: ModelWrapper, inputs: Mapping[str, Integers]) -> dict[str, Any]:
    """Each graph output as the model's KernelOps compute it: every node's
    ``execute_node``, in graph order, on ``inputs`` as integers (int64) and the
    initializers as the model stores them."""
    found = executed(model, inputs)
    return {item.name: found[item.name] for item in model.graph.output}


def executed(model: ModelWrapper, inputs: Mapping[str, Integers]) -> dict[str, Any]:
    """Every tensor's values as the model's KernelOps compute them (``oracle``): the
    whole context, its initializers, inputs, and each node's outputs."""
    context: dict[str, Any] = {}
    for tensor in model.graph.initializer:
        context[tensor.name] = model.get_initializer(tensor.name)
    for name, values in inputs.items():
        context[name] = np.asarray(values, dtype=np.int64)
    for node in model.graph.node:
        kernel_op(model, node).execute_node(context, model.graph)
    return context


def _integers(name: str, values: Any) -> Integers:
    found = np.asarray(values)
    if found.dtype.kind == "f" and not np.array_equal(np.rint(found), found):
        raise Disagreement(f"{name}: the reference gives values that are not integers")
    return np.rint(found).astype(np.int64) if found.dtype.kind == "f" else found.astype(np.int64)


def as_partition(model: ModelWrapper, path: Path, name: str) -> ModelWrapper:
    """A parent graph of one partition node ``name`` whose body is ``model``, saved to
    ``path``: its graph inputs that are no initializer and its outputs, with their shapes
    and datatypes, are the node's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    parent = ModelWrapper(copy.deepcopy(model.model))
    graph = parent.graph
    boundary = graph_inputs(model)
    kept = [item for item in graph.input if item.name in boundary]
    for items in (graph.node, graph.initializer, graph.value_info, graph.input):
        del items[:]
    graph.input.extend(kept)
    node = graph.node.add()
    node.name, node.op_type, node.domain = name, PARTITION_OP, PARTITION_DOMAIN
    node.input.extend(boundary)
    node.output.extend(item.name for item in graph.output)
    parent.set_opset_import(PARTITION_DOMAIN, 1)
    parent.get_customop_wrapper(node).set_nodeattr("model", str(path))
    return parent


def check_parity(
    model: ModelWrapper,
    directory: Path,
    *,
    inputs: Mapping[str, Integers],
    reference: Reference | None = None,
    pacing: Pacing = STALLED,
    completion: Completion | None = None,
    label: str = "parity",
) -> None:
    """Check the hardware of ``model``'s KernelOp nodes against their oracle on
    ``inputs`` (or against ``reference``), simulating under ``directory``; see the
    module docstring. Raises ``WordsDiffer`` (an output word the hardware computed
    otherwise), ``SimulationFailed`` or ``Disagreement``."""
    point, boundary = configured_root(model, label, completion)
    expected = dict(execute_onnx(model, inputs) if reference is None else reference(inputs))
    outputs = {item.name for item in model.graph.output}
    if set(expected) != outputs:
        raise ValueError(f"{label}: the reference gives {sorted(expected)}, not {sorted(outputs)}")
    streams: dict[str, Stream] = {}
    checked: dict[str, tuple[int, ...]] = {}
    for tensor, _ in boundary:
        ends = getattr(point, member(tensor)).endpoints
        end = ends.source if ends.source_owner is None else ends.sink
        streams[tensor] = Stream(end.form, end.element.bits)
        if tensor in outputs:
            values = _integers(tensor, expected[tensor])
            low, high = ordinary_integer_bounds(end.element.dtype)
            if values.size and not low <= values.min() <= values.max() <= high:
                raise Disagreement(
                    f"{label}: {tensor}'s reference values span [{values.min()}, {values.max()}],"
                    f" which the boundary's {end.element} does not hold"
                )
            checked[tensor] = pack(end.form, values.ravel(), end.element.bits)
    parent = as_partition(model, directory / f"{label}.onnx", label)
    hardware = XSim(pacing=pacing, completion=completion, directory=directory)
    found = execute_onnx(parent, inputs, executors=(hardware,), require_hardware=True)
    # XSim's outputs are its words read back (an element presented twice with two values
    # is refused there), so packing them again gives back the words that arrived.
    received = {
        tensor: pack(stream.form, _integers(tensor, found[tensor]).ravel(), stream.bits)
        for tensor, stream in streams.items()
        if tensor in outputs
    }
    differing = [
        f"{port} word {index}: {got:x} != {want:x}"
        for tensor, port in boundary
        if tensor in outputs
        for index, (got, want) in enumerate(zip(received[tensor], checked[tensor], strict=True))
        if got != want
    ]
    if differing:
        decoded = decode(
            {tensor: streams[tensor] for tensor in outputs},
            received,
            inputs={tensor: streams[tensor] for tensor in inputs},
            values=inputs,
            reference=lambda read: (
                execute_onnx(model, read) if reference is None else reference(read)
            ),
        )
        why = decoded.message if decoded else "no order of the declared loops explains them"
        ports = dict(boundary)
        raise WordsDiffer(
            f"{label}: {why}\n{differing[0]}; {len(differing)} output words differ",
            {ports[tensor]: words for tensor, words in received.items()},
        )


__all__ = [
    "Disagreement",
    "Integers",
    "Reference",
    "as_partition",
    "boundary_inputs",
    "check_parity",
    "executed",
    "graph_inputs",
    "oracle",
]
