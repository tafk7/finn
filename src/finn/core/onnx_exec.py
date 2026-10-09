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
from typing import Any

import numpy as np
import qonnx.analysis.topology as ta
from numpy.typing import NDArray
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.registry import getCustomOp
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
    """start_node or end_node is a node of a partition's body, where the run cannot start
    or stop: the message names the partition and what to run instead."""


# The executors of the innermost execute_onnx running in this context. A node's
# execute_node has no parameter for them (qonnx's CustomOp.execute_node(context,
# graph)), so an op that executes a model of its own, the partition node, reads them
# here (executing()) and passes them on.
_executing: ContextVar[tuple[Executor, ...]] = ContextVar(
    "finn.core.onnx_exec.executing", default=DEFAULT_EXECUTORS
)


def executing() -> tuple[Executor, ...]:
    """The executors of the innermost execute_onnx running in this context, or the
    default outside one: what a node that executes a body of its own (a partition
    node) executes it with."""
    return _executing.get()


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
    claims. A partition node's body runs under the same executors. The model's
    exec_mode metadata "rtlsim" (a stitched IP) bypasses them.

    require_hardware: every hardware node (a KernelOp, a partition node) must be run
    by a hardware executor (``Executor.hardware``, such as XSim); one that no hardware
    executor claims raises HardwareRequired before it runs, rather than running in
    Python. provenance, a dict, is given each node the run executes, by name, with
    the executor that ran it (None for qonnx's execute_node); a partition's body,
    run by its node, is not this run's.

    start_node and end_node may also be nodes of a partition node's body. A hardware
    executor runs a partition whole: it refuses to start inside it (InsidePartition,
    naming the alternative: cut the graph there and build that partition), and
    ending inside it runs the whole partition, whose outputs are what it observes (the
    tensors inside the body are not in the context). Under any other executor a run
    neither starts nor ends inside a body (InsidePartition).
    """

    # validate that all provided input names exist in the model
    # this catches common bugs like using outdated tensor names
    valid_tensor_names = set(model.get_all_tensor_names())
    for inp_name in input_dict.keys():
        if inp_name not in valid_tensor_names:
            graph_input_names = sorted(t.name for t in model.graph.input)
            raise ValueError(
                f"Provided input '{inp_name}' not found in model. "
                f"Valid graph inputs are: {graph_input_names}"
            )

    # check if model has an execution mode set
    # if None, execute model node by node with the executors
    # if set to "rtlsim" execute model using xsi
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
        execution_context = _execution_context(model, input_dict)
        token = _executing.set(tuple(executors))
        try:
            _execute_nodes(
                model,
                execution_context,
                start_node,
                end_node,
                tuple(executors),
                require_hardware,
                provenance,
            )
        finally:
            _executing.reset(token)
    elif model_exec_mode == "rtlsim":
        # use stitched IP for rtlsim
        # The legacy stitched-IP executor, loaded only when the model selects it.
        from finn.core.rtlsim_exec import rtlsim_exec  # noqa: PLC0415

        execution_context = _execution_context(model, input_dict)
        # qonnx/legacy untyped
        rtlsim_exec(model, execution_context)  # type: ignore[no-untyped-call]
    else:
        raise ValueError(
            """Metadata property "exec_mode" is set to an unknown value. Can be left
            unset or has to be set to "rtlsim" for execution using xsi!"""
        )

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
    return execution_context


def _execute_nodes(
    model: ModelWrapper,
    execution_context: Context,
    start_node: NodeProto | None,
    end_node: NodeProto | None,
    executors: tuple[Executor, ...],
    require_hardware: bool,
    provenance: RanBy | None,
) -> None:
    """qonnx's node loop (qonnx.core.onnx_exec.execute_onnx) with the executors in
    front of its execute_node: the nodes from start_node to end_node, in the graph's
    order, each sanitized in place to its quantization annotation as qonnx sanitizes it."""
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
        if index == start_ind and start_inside is not None:
            raise InsidePartition(_inside(node, start_inside, executor, "start_node"))
        if index == end_ind and end_inside is not None and not hardware:
            raise InsidePartition(_inside(node, end_inside, executor, "end_node"))
        if executor is not None:
            executor.run(node, execution_context, model)
        else:
            opset_version = opset_imports.get(node.domain, get_preferred_qonnx_opset())
            # qonnx/legacy untyped
            execute_node(  # type: ignore[no-untyped-call]
                node, execution_context, graph, opset_version, model=model
            )
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


def _inside(node: NodeProto, inside: NodeProto, executor: Executor | None, role: str) -> str:
    where = f"{role} {inside.name} is inside partition {node.name}"
    if executor is not None and executor.hardware:
        return (
            f"{where}, which {_named(executor)} simulates whole, from its inputs: it cannot "
            f"start there. Cut the graph at {inside.name} and build that partition to "
            "simulate from it"
        )
    point = "from" if role == "start_node" else "up to"
    return (
        f"{where}, which {_named(executor)} runs whole: running a partition's body {point} "
        f"one of its nodes is not supported. Give the partition node {node.name}, or run "
        f"its body (its model attribute) {point} {inside.name}"
    )


def execute_parent(
    parent_path: str,
    child_path: str,
    input_tensor_npy: NDArray[Any],
    return_full_ctx: bool = False,
) -> Any:
    """Execute parent model containing a single StreamingDataflowPartition by
    replacing it with the model at child_path and return result."""

    parent_model = ModelWrapper(parent_path)
    iname = parent_model.get_first_global_in()
    oname = parent_model.get_first_global_out()
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_op = getCustomOp(sdp_node)
    sdp_op.set_nodeattr("model", child_path)
    sdp_op.set_nodeattr("return_full_exec_context", 1 if return_full_ctx else 0)
    ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
    if return_full_ctx:
        return ret
    else:
        return ret[oname]


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
