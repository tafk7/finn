############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
# Note: This transform is inspired by a transformation from Thomas Keller (ExpandNorms)
# and ExtractQuantScaleZeroPt from qonnx.
#
############################################################################

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueParameterTensors, SortGraph
from qonnx.transformation.remove import RemoveIdentityOps


class ExtractNormScaleBias(Transformation):
    """Extract LayerNormalization scale and bias into separate nodes
    and set initializers to 1 or 0 respectively."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        for node in graph.node:
            if node.op_type == "LayerNormalization":
                ln_node = node
                input_ln = node.input[0]
                scale_tensor = node.input[1]
                # The bias input is optional.
                if len(node.input) > 2:
                    bias_tensor = node.input[2]
                    bias = model.get_initializer(bias_tensor)
                else:
                    bias = None
                scale = model.get_initializer(scale_tensor)
                extract_scale = False
                extract_bias = False
                if (scale != 1).any():
                    extract_scale = True
                if bias is not None and np.any(bias):
                    extract_bias = True
                if (not extract_scale) and (not extract_bias):
                    continue
                act_shape = model.get_tensor_shape(input_ln)
                last_node = ln_node
                final_output = ln_node.output[0]
                if extract_scale:
                    scale_act_in_name = model.make_new_valueinfo_name()
                    scale_act_in = oh.make_tensor_value_info(
                        scale_act_in_name, TensorProto.FLOAT, act_shape
                    )
                    last_node.output[0] = scale_act_in_name
                    graph.value_info.append(scale_act_in)
                    scale_node = oh.make_node(
                        "Mul", [scale_act_in_name, scale_tensor], [final_output]
                    )
                    graph.node.append(scale_node)
                    # The graph holds a copy of the node make_node returned: keep the copy.
                    last_node = graph.node[-1]
                    new_scale_name = model.make_new_valueinfo_name()
                    model.set_initializer(new_scale_name, np.ones(act_shape[-1], dtype=np.float32))
                    ln_node.input[1] = new_scale_name
                if extract_bias:
                    bias_act_in_name = model.make_new_valueinfo_name()
                    bias_act_in = oh.make_tensor_value_info(
                        bias_act_in_name, TensorProto.FLOAT, act_shape
                    )
                    graph.value_info.append(bias_act_in)
                    bias_node = oh.make_node("Add", [bias_act_in_name, bias_tensor], [final_output])
                    last_node.output[0] = bias_act_in_name
                    graph.node.append(bias_node)
                    new_bias_name = model.make_new_valueinfo_name()
                    model.set_initializer(new_bias_name, np.zeros(act_shape[-1], dtype=np.float32))
                    ln_node.input[2] = new_bias_name

                if extract_scale or extract_bias:
                    # The new nodes were appended: sort the graph again.
                    model = model.transform(SortGraph())
                    model = model.transform(RemoveIdentityOps())
                    model = model.transform(GiveUniqueParameterTensors())
                    return model, True

        return model, False
