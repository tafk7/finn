############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright is held by AMD and is provided under BSD-3-Clause license.
#
# Note: This transform is inspired by the ExtractNormScaleBias
# transformation, but for Multithreshold nodes.
#
############################################################################

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueParameterTensors, SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps


class ExtractMultiThresholdScaleBias(Transformation):
    """Extract MultiThreshold out_scale and out_bias into separate nodes
    and reset attributes to 1.0 and 0.0 respectively."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            if node.op_type == "MultiThreshold":
                mt_node = node
                mt_inst = getCustomOp(mt_node)
                out_scale = mt_inst.get_nodeattr("out_scale")
                out_bias = mt_inst.get_nodeattr("out_bias")
                extract_scale = False
                extract_bias = False
                if out_scale != 1.0:
                    extract_scale = True
                if out_bias != 0.0:
                    extract_bias = True
                if (not extract_scale) and (not extract_bias):
                    continue
                act_shape = model.get_tensor_shape(mt_node.input[0])
                last_node = mt_node
                final_output = mt_node.output[0]
                if extract_scale:
                    scale_act_in_name = model.make_new_valueinfo_name()
                    scale_act_in = oh.make_tensor_value_info(
                        scale_act_in_name, TensorProto.FLOAT, act_shape
                    )
                    last_node.output[0] = scale_act_in_name
                    graph.value_info.append(scale_act_in)
                    scale_tensor_name = model.make_new_valueinfo_name()
                    model.set_initializer(scale_tensor_name, np.array(out_scale, dtype=np.float32))
                    scale_node = oh.make_node(
                        "Mul", [scale_act_in_name, scale_tensor_name], [final_output]
                    )
                    graph.node.append(scale_node)
                    # The graph holds a copy of the node make_node returned: keep the copy.
                    last_node = graph.node[-1]
                    mt_inst.set_nodeattr("out_scale", 1.0)
                if extract_bias:
                    bias_act_in_name = model.make_new_valueinfo_name()
                    bias_act_in = oh.make_tensor_value_info(
                        bias_act_in_name, TensorProto.FLOAT, act_shape
                    )
                    graph.value_info.append(bias_act_in)
                    bias_tensor_name = model.make_new_valueinfo_name()
                    model.set_initializer(bias_tensor_name, np.array(out_bias, dtype=np.float32))
                    bias_node = oh.make_node(
                        "Add", [bias_act_in_name, bias_tensor_name], [final_output]
                    )
                    last_node.output[0] = bias_act_in_name
                    graph.node.append(bias_node)
                    mt_inst.set_nodeattr("out_bias", 0.0)

                if extract_scale or extract_bias:
                    # The new nodes were appended: sort the graph again.
                    model = model.transform(SortGraph())
                    model = model.transform(RemoveIdentityOps())
                    model = model.transform(GiveUniqueParameterTensors())
                    model = model.transform(InferDataTypes())
                    graph_modified = True

        return model, graph_modified
