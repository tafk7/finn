# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The generated hardware against its KernelOps' oracle, at the ops' boundary.

PRINCIPLES 8: a kernel declares its hardware's design space, never its
computation; the KernelOp's ``execute_node`` is the op's computational
reference. ``check_parity`` checks that the hardware a model of KernelOp nodes
generates computes what those ops compute, on the same inputs:

1. **the cut**: the model cut as a build cuts it (``CutKernelPartition``, the build's
   one cut, ``partition_kernel_ops``): a parent graph of one partition node, its body
   saved under ``directory``, its tensors' channel choices moved into the body; the
   hardware checked is the partition a build ships;
2. **the point**: the body's shell root, its nodes' persisted choices and its tensors'
   channel choices replayed and the rest completed by ``completion`` (``Baseline()``
   by default), as packaging builds it (``configured_root``): a chosen design point is
   the choices saved on the nodes and the tensors;
3. **the oracle**: each graph output as the nodes' ``execute_node`` computes it, the
   model run by ``finn.core.onnx_exec.execute_onnx`` with its default executor
   (``Python()``) on the inputs as integers (int64, which hold every value of a type
   the kernels admit: a KernelOp's reference is exact on them, MatMul's integer
   product included); every output must be an integer the boundary's element holds,
   or the hardware cannot present it (``Disagreement``);
4. **the hardware**: the parent graph run by ``execute_onnx`` with the XSim executor
   (``finn.core.executors.xsim.executor.XSim``, paced by ``pacing``) alone, the run
   requiring hardware: every graph input streamed at its boundary port in the order
   the port presents, and every output read back from its words. Each output is
   compared with the oracle's word for word, as its port presents both. Words that
   differ are decoded against the oracle as an order (``finn.harness.orders``): a
   boundary walked otherwise than declared is named by beat and index tuple.

So parity is two runs of the model compared: ``execute_onnx(parent, inputs,
executors=(XSim(pacing=...),), require_hardware=True)``, ``parent`` the model as the
build cuts it, against ``execute_onnx(model, inputs)``.

A test may state ``reference`` in place of the oracle, an integer reference of
its own. ``finn.harness.reference`` checks a KernelOp's reference against ONNX;
here the reference is what the hardware must compute.

``boundary_inputs`` draws the inputs from each graph input's datatype: random
integers of its range, seeded, and its extremes, the first row (the last axis's
vector) all its least value and the second all its greatest; one row holds both.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.executors.xsim.executor import XSim
from finn.core.executors.xsim.pacing import STALLED, Pacing
from finn.core.executors.xsim.rtl import WordsDiffer
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.kernels.base import datatype, kernel_op
from finn.custom_op.kernels.base import shape as tensor_shape
from finn.custom_op.kernels.shell import configured_root
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.traversal import pack
from finn.harness.orders import Integers, Reference, Stream, decode
from finn.kernels.explore import Completion
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.package import free_side


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


def executed(model: ModelWrapper, inputs: Mapping[str, Integers]) -> dict[str, Any]:
    """Every tensor's values as the model's KernelOps compute them: every node's
    ``execute_node``, in graph order, on ``inputs`` as integers (int64) and the
    initializers as the model stores them; the whole context, its initializers, inputs,
    and each node's outputs."""
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
    """Check the hardware of ``model``'s KernelOp nodes, cut as a build cuts them, against
    their oracle on ``inputs`` (or against ``reference``), cutting and simulating under
    ``directory``; see the module docstring. Raises ``WordsDiffer`` (an output word the
    hardware computed otherwise), ``SimulationFailed`` or ``Disagreement``."""
    parent = model.transform(CutKernelPartition(directory))
    _, body, _ = partition_body(parent)
    point, boundary = configured_root(body, label, completion)
    expected = dict(execute_onnx(model, inputs) if reference is None else reference(inputs))
    outputs = {item.name for item in model.graph.output}
    if set(expected) != outputs:
        raise ValueError(f"{label}: the reference gives {sorted(expected)}, not {sorted(outputs)}")
    streams: dict[str, Stream] = {}
    checked: dict[str, tuple[int, ...]] = {}
    for tensor, _ in boundary:
        end = free_side(point, tensor)
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
    "boundary_inputs",
    "check_parity",
    "executed",
    "graph_inputs",
]
