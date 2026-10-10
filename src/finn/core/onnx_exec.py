# Copyright (c) 2022, Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import qonnx.analysis.topology as ta
from numpy.typing import NDArray
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.util.basic import (
    get_preferred_qonnx_opset,
    get_sanitize_quant_tensors,
    sanitize_quant_values,
)

from finn.core.executors import Context, Executor, Python, hardware_node, is_partition
from finn.custom_op.partition.kernel_partitions import body_file

#: The executors execute_onnx runs a model with when its caller names none.
DEFAULT_EXECUTORS: tuple[Executor, ...] = (Python(),)

RanBy = dict[str, Executor | None]
"""Which executor ran each node a run executed, by node name; None where qonnx's
execute_node ran it (a node no executor claims)."""


class HardwareRequired(RuntimeError):
    """A run that requires hardware reached a hardware node (a KernelOp, a partition
    node) that no hardware executor claims: it would otherwise have run in Python, and a
    Python result would pass for a hardware one."""


class InsidePartition(ValueError):
    """start_node is a node of a partition's body, and a hardware executor runs that
    partition: it simulates the partition whole, from its inputs, so the run cannot start
    there. The message names the partition and what to run instead."""


@dataclass(frozen=True)
class Running:
    """What the innermost execute_onnx running in this context was asked, as a node that
    executes a body of its own (a partition node) reads it: the executors, whether the
    run returns its full context, and the node of the running node's body the run starts
    or ends at (None: its body runs whole at that end).

    A node's execute_node has no parameter for them (qonnx's
    ``CustomOp.execute_node(context, graph)``), so the node, or the executor that runs
    it, reads them here (``running``)."""

    executors: tuple[Executor, ...] = DEFAULT_EXECUTORS
    full_context: bool = False
    start_node: NodeProto | None = None
    end_node: NodeProto | None = None


_running: ContextVar[Running] = ContextVar("finn.core.onnx_exec.running", default=Running())


def running() -> Running:
    """What the innermost execute_onnx running in this context was asked (``Running``),
    or the default outside one: what a node that executes a body of its own (a partition
    node) executes it with."""
    return _running.get()


def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, NDArray[Any]],
    return_full_exec_context: bool = False,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    executors: Sequence[Executor] = DEFAULT_EXECUTORS,
    require_hardware: bool = False,
    provenance: RanBy | None = None,
) -> Context:
    """Executes given ONNX ModelWrapper with given named inputs.
    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.
    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict.
    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.
    executors (finn.core.executors) run the nodes: for each node, the first
    executor that claims it runs it, and qonnx's execute_node runs a node none
    claims. A partition node's body runs under the same executors.

    require_hardware: every hardware node (a KernelOp, a partition node) must be run
    by a hardware executor (``Executor.hardware``, such as XSim); one that no hardware
    executor claims raises HardwareRequired before it runs, rather than running in
    Python. provenance, a dict, is given each node the run executes, by name, with
    the executor that ran it (None for qonnx's execute_node); a partition's body,
    run by its node, is not this run's.

    start_node and end_node may also be nodes of a partition node's body: the run then
    starts or ends at that partition node, and its body runs from or up to that node
    (``running``). Python runs the body so, its nodes from start_node to end_node. A
    hardware executor runs a partition whole: it refuses to start inside it
    (InsidePartition, naming the alternative: cut the graph there and build that
    partition), and ending inside it simulates the whole partition and observes, of its
    tensors, those up to end_node.

    The full context holds the tensors of a partition's body under the partition node's
    name, ``<node>_<tensor>`` (the body's outputs under the parent's names): Python's all
    of the body's context, a hardware executor's each link it observed between two
    kernels of the body, as its producer presents it. A caller starts inside a body from
    a context so named: input_dict may name a body's tensors so.
    """

    # validate that all provided input names exist in the model
    # this catches common bugs like using outdated tensor names
    valid_tensor_names = set(model.get_all_tensor_names())
    unknown = [name for name in input_dict if name not in valid_tensor_names]
    if unknown:
        # A body's tensor, as a full context names it (<node>_<tensor>).
        valid_tensor_names |= _body_tensors(model)
    for inp_name in unknown:
        if inp_name not in valid_tensor_names:
            graph_input_names = sorted(t.name for t in model.graph.input)
            raise ValueError(
                f"Provided input '{inp_name}' not found in model. "
                f"Valid graph inputs are: {graph_input_names}"
            )

    execution_context = _execution_context(model, input_dict)
    token = _running.set(Running(tuple(executors), return_full_exec_context))
    try:
        _execute_nodes(model, execution_context, start_node, end_node, require_hardware, provenance)
    finally:
        _running.reset(token)

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in model.graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict


def _execution_context(model: ModelWrapper, input_dict: Mapping[str, NDArray[Any]]) -> Context:
    """The model's execution context, checked as qonnx's execute_onnx checks it, with
    the given inputs in it."""
    if not model.check_all_tensor_shapes_specified():
        raise ValueError("Found unspecified tensor shapes, try infer_shapes")
    ret = model.analysis(ta.nodes_topologically_sorted)
    if ret["nodes_topologically_sorted"] is not True:
        raise ValueError(
            """Nodes must be
    topologically sorted."""
        )
    # every variable required by the graph has some buffer associated with it: graph
    # inputs (the input data as well as the trained parameters) and the graph
    # ValueInfo (intermediate tensors between layers)
    execution_context: Context = model.make_empty_exec_context()
    # fill in any inputs provided to this function
    for inp_name in input_dict.keys():
        if inp_name in execution_context:
            if execution_context[inp_name].shape == input_dict[inp_name].shape:
                execution_context[inp_name] = input_dict[inp_name]
            else:
                raise ValueError(
                    "Shape mismatch for provided input %s: found %s expected %s "
                    % (
                        inp_name,
                        str(execution_context[inp_name].shape),
                        str(input_dict[inp_name].shape),
                    )
                )
        else:
            # A body's tensor (<node>_<tensor>): its partition node passes it on, and
            # its body's run checks its shape.
            execution_context[inp_name] = input_dict[inp_name]
    return execution_context


def _body_tensors(model: ModelWrapper) -> set[str]:
    """Each tensor of a partition node's body, by the name a full context holds it under:
    ``<node>_<tensor>``."""
    return {
        f"{node.name}_{name}"
        for node in model.graph.node
        if is_partition(node)
        for name in ModelWrapper(body_file(node)).get_all_tensor_names()
    }


def window(
    model: ModelWrapper, start_node: NodeProto | None, end_node: NodeProto | None
) -> list[NodeProto]:
    """The nodes of ``model`` a run from ``start_node`` to ``end_node`` executes, in the
    graph's order (from the first, to the last, where None): both nodes of ``model``."""
    nodes = list(model.graph.node)
    start = 0 if start_node is None else model.get_node_index(start_node)
    end = len(nodes) - 1 if end_node is None else model.get_node_index(end_node)
    if start is None or end is None:
        raise ValueError("start_node and end_node are nodes of the model")
    return nodes[start : end + 1]


def _execute_nodes(
    model: ModelWrapper,
    execution_context: Context,
    start_node: NodeProto | None,
    end_node: NodeProto | None,
    require_hardware: bool,
    provenance: RanBy | None,
) -> None:
    """qonnx's node loop (qonnx.core.onnx_exec.execute_onnx) with the executors in
    front of its execute_node: the nodes from start_node to end_node, in the graph's
    order, each sanitized in place to its quantization annotation as qonnx sanitizes it.
    A partition node the run starts or ends inside runs with that node of its body in
    ``running()``."""
    run = running()
    executors = run.executors
    graph = model.graph
    opset_imports = model.get_opset_imports()
    start_ind, start_inside = _located(model, start_node, "start_node", 0)
    end_ind, end_inside = _located(model, end_node, "end_node", len(graph.node) - 1)
    if end_ind + 1 < start_ind:
        raise ValueError("Start/end nodes must define valid subgraph")
    for index, node in enumerate(graph.node[start_ind : end_ind + 1], start_ind):
        if get_sanitize_quant_tensors() != 0:
            # round input values to match quantization annotation
            sanitize_quant_values(model, node.input, execution_context)
        executor = next((e for e in executors if e.claims(node, model)), None)
        hardware = executor is not None and executor.hardware
        if require_hardware and hardware_node(node) and not hardware:
            raise HardwareRequired(_unclaimed(node, executor, executors))
        start = start_inside if index == start_ind else None
        end = end_inside if index == end_ind else None
        if start is not None and hardware:
            raise InsidePartition(_inside(node, start, executor))
        token = _running.set(replace(run, start_node=start, end_node=end))
        try:
            if executor is not None:
                executor.run(node, execution_context, model)
            else:
                opset_version = opset_imports.get(node.domain, get_preferred_qonnx_opset())
                # qonnx/legacy untyped
                execute_node(  # type: ignore[no-untyped-call]
                    node, execution_context, graph, opset_version, model=model
                )
        finally:
            _running.reset(token)
        if provenance is not None:
            provenance[node.name] = executor
        if get_sanitize_quant_tensors() != 0:
            # round output values to quantization annotation
            sanitize_quant_values(model, node.output, execution_context)


def _located(
    model: ModelWrapper, node: NodeProto | None, role: str, default: int
) -> tuple[int, NodeProto | None]:
    """The index of the node of ``model`` a run starts or ends at (``role``), and the
    node of its body when ``node`` is inside a partition node's body (the partition's
    index then); ``default`` when no node is given."""
    if node is None:
        return default, None
    index = model.get_node_index(node)
    if index is not None:
        return index, None
    for index, candidate in enumerate(model.graph.node):
        if is_partition(candidate):
            body = ModelWrapper(body_file(candidate))
            if body.get_node_index(node) is not None:
                return index, node
    raise ValueError(f"{role} {node.name!r} is no node of the model nor of a partition's body")


def _named(executor: Executor | None) -> str:
    return "qonnx's execute_node" if executor is None else type(executor).__name__


def _unclaimed(node: NodeProto, executor: Executor | None, executors: Sequence[Executor]) -> str:
    hardware = [type(e).__name__ for e in executors if e.hardware]
    kind = "partition node" if is_partition(node) else "KernelOp"
    alternative = (
        "XSim runs a partition whose body is all KernelOps"
        if is_partition(node)
        else "a KernelOp runs in hardware as a partition: cut the graph to one"
    )
    return (
        f"{node.name} ({kind} {node.op_type}): the run requires hardware, and "
        f"{_named(executor)} would run it; no hardware executor claims it (hardware "
        f"executors given: {', '.join(hardware) or 'none'}). {alternative}"
    )


def _inside(node: NodeProto, inside: NodeProto, executor: Executor | None) -> str:
    return (
        f"start_node {inside.name} is inside partition {node.name}, which "
        f"{_named(executor)} simulates whole, from its inputs: it cannot start there. Cut "
        f"the graph at {inside.name} and build that partition to simulate from it"
    )


def compare_execution(
    model_a: ModelWrapper,
    model_b: ModelWrapper,
    input_dict: Mapping[str, NDArray[Any]],
    compare_fxn: Callable[[Any, Any], Any] = lambda x, y: np.isclose(x, y, atol=1e-3).all(),
) -> Any:
    """Executes two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = list(execute_onnx(model_a, input_dict).items())[0][1]
    res_b = list(execute_onnx(model_b, input_dict).items())[0][1]
    return compare_fxn(res_a, res_b)
