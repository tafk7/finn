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

import copy
import numpy as np
import qonnx.analysis.topology as ta
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx as execute_onnx_base
from qonnx.custom_op.registry import getCustomOp

from finn.core.rtlsim_exec import rtlsim_exec


def execute_onnx(model, input_dict, return_full_exec_context=False, start_node=None, end_node=None):
    """Executes given ONNX ModelWrapper with given named inputs.
    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.
    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict.
    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.
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
    # if None, execute model node using the QONNX-provided execute_onnx impl
    # if set to "rtlsim" execute model using xsi
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
        return execute_onnx_base(model, input_dict, return_full_exec_context, start_node, end_node)
    elif model_exec_mode == "rtlsim":
        # check sanity of model and then use stitched IP for rtlsim
        if not model.check_all_tensor_shapes_specified():
            raise Exception("Found unspecified tensor shapes, try infer_shapes")
        ret = model.analysis(ta.nodes_topologically_sorted)
        assert (
            ret["nodes_topologically_sorted"] is True
        ), """Nodes must be
        topologically sorted."""

        graph = model.graph
        # first, we need to make sure that every variable required by the graph has
        # some buffer associated with it. this includes graph inputs (which includes
        # the input data as well as the trained parameters) and the graph ValueInfo
        # (intermediate tensors between layers)
        # this is provided by the execution_context, which is a dict of np.ndarray
        execution_context = model.make_empty_exec_context()
        # fill in any inputs provided to this function
        for inp_name in input_dict.keys():
            if inp_name in execution_context:
                expected_shape = execution_context[inp_name].shape
                provided_shape = input_dict[inp_name].shape
                same_shape = expected_shape == provided_shape
                runtime_batch_shape = (
                    len(expected_shape) > 0
                    and len(expected_shape) == len(provided_shape)
                    and expected_shape[1:] == provided_shape[1:]
                )
                if same_shape or runtime_batch_shape:
                    execution_context[inp_name] = input_dict[inp_name]
                else:
                    raise Exception(
                        "Shape mismatch for provided input %s: found %s expected %s "
                        % (
                            inp_name,
                            str(execution_context[inp_name].shape),
                            str(input_dict[inp_name].shape),
                        )
                    )

        # use stitched IP for rtlsim
        rtlsim_exec(model, execution_context)
    else:
        raise Exception(
            """Metadata property "exec_mode" is set to an unknown value. Can be left
            unset or has to be set to "rtlsim" for execution using xsi!"""
        )

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict


def execute_onnx_and_make_model(model, input_dict):
    """Executes given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values."""

    # retrieve the full execution context
    execution_context = execute_onnx(model, input_dict, True)
    new_model = copy.deepcopy(model)
    # create value_info entries and initializers for everything
    for i in execution_context.keys():
        new_model.set_initializer(i, execution_context[i])
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    return new_model


def compare_execution(
    model_a,
    model_b,
    input_dict,
    compare_fxn=lambda x, y: np.isclose(x, y, atol=1e-3).all(),
):
    """Executes two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = list(execute_onnx(model_a, input_dict).items())[0][1]
    res_b = list(execute_onnx(model_b, input_dict).items())[0][1]
    return compare_fxn(res_a, res_b)


def execute_parent(
    parent_model,
    child_path,
    input_tensor_npy,
    return_full_ctx=False,
    batch_child=False,
):
    """Execute a parent model (a ModelWrapper or a path) containing one
    StreamingDataflowPartition.

    With ``batch_child=True``, parent processing keeps its compiled batch-one
    shapes while the partition child receives all preprocessed frames at once.
    A full-context batched call returns one parent context per frame as a list.
    """

    if not isinstance(parent_model, ModelWrapper):
        parent_model = ModelWrapper(parent_model)
    iname = parent_model.get_first_global_in()
    oname = parent_model.get_first_global_out()
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_inst = getCustomOp(sdp_node)
    sdp_inst.set_nodeattr("model", child_path)
    sdp_inst.set_nodeattr("return_full_exec_context", 1 if return_full_ctx else 0)

    batch_size = input_tensor_npy.shape[0]
    if not batch_child or batch_size == 1:
        ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
        if return_full_ctx:
            return ret
        else:
            return ret[oname]

    sdp_index = parent_model.get_node_index(sdp_node)
    frame_contexts = []
    for batch_index in range(batch_size):
        frame_input = input_tensor_npy[batch_index : batch_index + 1]
        if sdp_index == 0:
            frame_context = parent_model.make_empty_exec_context()
            frame_context[iname] = frame_input
        else:
            frame_context = execute_onnx(
                parent_model,
                {iname: frame_input},
                True,
                end_node=parent_model.graph.node[sdp_index - 1],
            )
        frame_contexts.append(frame_context)

    # The partition only reads its declared inputs, so it does not need the full
    # parent execution contexts.
    partition_context = {
        name: np.concatenate([context[name] for context in frame_contexts], axis=0)
        for name in sdp_node.input
    }
    sdp_inst.execute_node(partition_context, parent_model.graph)

    for output_name in sdp_node.output:
        batched_output = partition_context[output_name]
        for batch_index, frame_context in enumerate(frame_contexts):
            frame_context[output_name] = batched_output[batch_index : batch_index + 1]

    if sdp_index + 1 < len(parent_model.graph.node):
        suffix_start = parent_model.graph.node[sdp_index + 1]
        for batch_index, frame_context in enumerate(frame_contexts):
            resume_context = {
                name: value for name, value in frame_context.items() if value is not None
            }
            frame_contexts[batch_index] = execute_onnx(
                parent_model,
                resume_context,
                True,
                start_node=suffix_start,
            )

    if return_full_ctx:
        return frame_contexts
    return np.concatenate([context[oname] for context in frame_contexts], axis=0)
