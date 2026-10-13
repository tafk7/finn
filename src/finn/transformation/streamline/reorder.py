# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
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

import numpy as np
import qonnx.core.data_layout as DataLayout
import warnings
from copy import deepcopy
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

from finn.util.basic import resolve_resize_param_input


class MoveAddPastMul(Transformation):
    """Move add operations past multiply operations on linear segments of the graph.
    The aim is to have them next to each other such that they can be collapsed into
    a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Mul"
                    and not model.is_join_node(consumer)
                ):
                    # have: (x) -> add(,B) -> (x+B) -> mul(,A) -> (xA+BA)
                    # want: (x) -> mul(,A) -> (xA) -> add(,BA) -> (xA+BA)
                    # Input 1 holds each op's constant parameter.
                    mul_weight_name = consumer.input[1]
                    add_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    B = model.get_initializer(add_weight_name)
                    if (A is None) or (B is None):
                        warnings.warn("Mul or add does not have constant params, skipping")
                        continue
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    BA = B * A

                    new_mul = oh.make_node(
                        "Mul",
                        [start_name, mul_weight_name],
                        [middle_name],
                        name=consumer.name,
                    )
                    new_add = oh.make_node(
                        "Add", [middle_name, add_weight_name], [end_name], name=n.name
                    )
                    graph.node.insert(node_ind, new_mul)
                    graph.node.insert(node_ind + 1, new_add)
                    model.set_initializer(add_weight_name, BA)
                    model.set_tensor_datatype(add_weight_name, DataType["FLOAT32"])
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        return model, graph_modified


class MoveScalarMulPastMatMul(Transformation):
    """Move scalar mul operations past matmul operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node):
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MatMul":
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    mm_out_shape = model.get_tensor_shape(end_name)
                    if n.output[0] == consumer.input[0]:
                        new_matmul_inps = [start_name, consumer.input[1]]
                    elif n.output[0] == consumer.input[1]:
                        new_matmul_inps = [consumer.input[0], start_name]
                    else:
                        raise Exception(
                            """Invalid pattern detected,
                            output of matmul is not connected to any of the consumers inputs."""
                        )

                    if all(x == 1 for x in A.shape):
                        new_matmul = oh.make_node(
                            "MatMul",
                            new_matmul_inps,
                            [middle_name],
                            name=consumer.name,
                        )
                        new_mul = oh.make_node(
                            "Mul",
                            [middle_name, mul_weight_name],
                            [end_name],
                            name=n.name,
                        )
                        graph.node.insert(node_ind, new_matmul)
                        graph.node.insert(node_ind + 1, new_mul)
                        model.set_tensor_shape(middle_name, mm_out_shape)
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
        model = model.transform(InferShapes())
        return model, graph_modified


class MoveScalarAddPastMatMul(Transformation):
    """Move scalar add operations past matmul operations. We want to have adds
    next to each other such that they can be collapsed into a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "MatMul"
                    and not model.is_join_node(consumer)
                ):
                    add_weight_name = n.input[1]
                    matmul_weight_name = consumer.input[1]
                    A = model.get_initializer(add_weight_name)
                    W = model.get_initializer(matmul_weight_name)
                    if (A is None) or (W is None):
                        warnings.warn("MatMul or Add params are not constant, skipping")
                        continue
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    mm_out_shape = model.get_tensor_shape(end_name)
                    if all(x == 1 for x in A.shape):
                        # A scalar add moves past the MatMul as its product with the weights.
                        Anew = np.dot(A * np.ones(W.shape[0], dtype=np.float32), W)
                        model.set_initializer(add_weight_name, Anew)
                        new_matmul = oh.make_node(
                            "MatMul",
                            [start_name, matmul_weight_name],
                            [middle_name],
                            name=consumer.name,
                        )
                        new_add = oh.make_node(
                            "Add",
                            [middle_name, add_weight_name],
                            [end_name],
                            name=n.name,
                        )
                        graph.node.insert(node_ind, new_matmul)
                        graph.node.insert(node_ind + 1, new_add)
                        model.set_tensor_shape(middle_name, mm_out_shape)
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveAddPastConv(Transformation):
    """Move scalar and channelwise add operations past conv operations. We want to have adds
    next to each other such that they can be collapsed into a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Conv"
                    and not model.is_join_node(consumer)
                ):
                    conv_node = consumer
                    add_node = n
                    add_weight_name = n.input[1]
                    conv_in_name = consumer.input[0]
                    conv_in_shape = model.get_tensor_shape(conv_in_name)
                    # The layout is assumed NCHW.
                    channels = conv_in_shape[1]
                    A = model.get_initializer(add_weight_name)
                    if A is None:
                        warnings.warn("Add param is not constant, skipping")
                        continue
                    start_name = n.input[0]
                    end_name = consumer.output[0]
                    conv_out_shape = model.get_tensor_shape(end_name)

                    using_padding = True
                    pads = list(get_by_name(consumer.attribute, "pads").ints)
                    if sum(pads) == 0:
                        using_padding = False
                    if (
                        all(x == 1 for x in A.shape) or A.shape == (1, channels, 1, 1)
                    ) and not using_padding:
                        # The add's effect past the Conv: the Conv executed on the constant.
                        conv_in_const = np.zeros(conv_in_shape, dtype=np.float32)
                        if A.shape == (1, channels, 1, 1):
                            for ch in range(channels):
                                conv_in_const[0][ch].fill(A[0][ch].item())
                        else:
                            conv_in_const.fill(A.item())
                        exec_ctx = model.make_empty_exec_context()
                        exec_ctx[conv_in_name] = conv_in_const
                        opset_imports = model.get_opset_imports()
                        opset_version = opset_imports[conv_node.domain]
                        execute_node(conv_node, exec_ctx, model.graph, opset_version)
                        Anew = exec_ctx[end_name]

                        # Without padding, every pixel of it is the same.
                        Anew = Anew[0, :, 0, 0].reshape(1, -1, 1, 1)
                        model.set_initializer(add_weight_name, Anew)
                        conv_node.input[0] = start_name
                        model.set_tensor_shape(start_name, conv_in_shape)
                        conv_node.output[0] = conv_in_name
                        model.set_tensor_shape(conv_in_name, conv_out_shape)
                        add_node.input[0] = conv_in_name
                        add_node.output[0] = end_name
                        graph.node.remove(add_node)
                        graph.node.insert(node_ind, add_node)
                        graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarMulPastConv(Transformation):
    """Move scalar mul operations past conv operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Conv"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn("Mul param is not constant, skipping")
                        continue
                    conv_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    conv_in_name = conv_node.input[0]
                    conv_in_shape = model.get_tensor_shape(conv_in_name)
                    conv_out_name = conv_node.output[0]
                    conv_out_shape = model.get_tensor_shape(conv_out_name)
                    if all(x == 1 for x in A.shape):
                        conv_node.input[0] = start_name
                        model.set_tensor_shape(start_name, conv_in_shape)
                        conv_node.output[0] = conv_in_name
                        model.set_tensor_shape(conv_in_name, conv_out_shape)
                        mul_node.input[0] = conv_in_name
                        mul_node.output[0] = conv_out_name
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarMulPastConvTranspose(Transformation):
    """Move scalar mul operations past ConvTranspose operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "ConvTranspose"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn("Mul param is not constant, skipping")
                        continue
                    conv_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    conv_in_name = conv_node.input[0]
                    conv_in_shape = model.get_tensor_shape(conv_in_name)
                    conv_out_name = conv_node.output[0]
                    conv_out_shape = model.get_tensor_shape(conv_out_name)
                    if all(x == 1 for x in A.shape):
                        conv_node.input[0] = start_name
                        model.set_tensor_shape(start_name, conv_in_shape)
                        conv_node.output[0] = conv_in_name
                        model.set_tensor_shape(conv_in_name, conv_out_shape)
                        mul_node.input[0] = conv_in_name
                        mul_node.output[0] = conv_out_name
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveMulPastDWConv(Transformation):
    """Move channelwise mul operations past depthwise conv operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Conv"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn(
                            """Mul weight tensor is not set. If it is a constant,
                                please use set_initializer to set the tensor."""
                        )
                        continue
                    conv_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    conv_in_name = conv_node.input[0]
                    conv_in_shape = model.get_tensor_shape(conv_in_name)
                    ifm_ch = conv_in_shape[1]
                    group_attribute = get_by_name(consumer.attribute, "group")
                    if group_attribute is None:
                        continue
                    group_attribute = group_attribute.i
                    conv_out_name = conv_node.output[0]
                    conv_out_shape = model.get_tensor_shape(conv_out_name)
                    if A.shape == (1, ifm_ch, 1, 1) and ifm_ch == group_attribute:
                        conv_node.input[0] = start_name
                        model.set_tensor_shape(start_name, conv_in_shape)
                        model.set_tensor_datatype(start_name, DataType["FLOAT32"])
                        conv_node.output[0] = conv_in_name
                        model.set_tensor_shape(conv_in_name, conv_out_shape)
                        model.set_tensor_datatype(conv_in_name, DataType["FLOAT32"])
                        mul_node.input[0] = conv_in_name
                        mul_node.output[0] = conv_out_name
                        model.set_tensor_datatype(conv_out_name, DataType["FLOAT32"])
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveMulPastMaxPool(Transformation):
    """Move non-negative scalar or channelwise mul operations past max pool operations.
    We want to have muls next to each other such that they can be collapsed into a
    single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "MaxPool"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn(
                            """Mul weight tensor is not set. If it is a constant,
                                please use set_initializer to set the tensor."""
                        )
                        continue
                    maxpool_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    maxpool_in_name = maxpool_node.input[0]
                    maxpool_in_shape = model.get_tensor_shape(maxpool_in_name)
                    ifm_ch = maxpool_in_shape[1]
                    maxpool_out_name = maxpool_node.output[0]
                    maxpool_out_shape = model.get_tensor_shape(maxpool_out_name)

                    kernel_shape = list(get_by_name(maxpool_node.attribute, "kernel_shape").ints)
                    if len(kernel_shape) != 2:
                        continue

                    # A negative factor does not commute with max.
                    if (A < 0).any():
                        continue

                    if all(x == 1 for x in A.shape) or A.shape == (1, ifm_ch, 1, 1):
                        maxpool_node.input[0] = start_name
                        model.set_tensor_shape(start_name, maxpool_in_shape)
                        model.set_tensor_datatype(start_name, DataType["FLOAT32"])
                        maxpool_node.output[0] = maxpool_in_name
                        model.set_tensor_shape(maxpool_in_name, maxpool_out_shape)
                        model.set_tensor_datatype(maxpool_in_name, DataType["FLOAT32"])
                        mul_node.input[0] = maxpool_in_name
                        mul_node.output[0] = maxpool_out_name
                        model.set_tensor_datatype(maxpool_out_name, DataType["FLOAT32"])
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarLinearPastInvariants(Transformation):
    """Move scalar linear operations (mul, add) past functions which are invariant
    to them. Specifically, matches and transforms the following patterns:
    f(x*C) -> f(x) * C
    f(x+C) -> f(x) + C
    where x is a dynamic input, C is a constant tensor.
    Known f which obey this property are: Reshape, Flatten, Transpose,
    GlobalAveragePool
    """

    SUPPORTED_INVARIANTS = {
        "GlobalAveragePool",
        "Reshape",
        "Transpose",
        "Flatten",
        "Slice",
        "Squeeze",
        "Unsqueeze",
    }

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            is_nearest_neighbor_resample = False
            if n.op_type == "Upsample" or n.op_type == "Resize":
                mode = get_by_name(n.attribute, "mode").s.decode("ascii")
                is_nearest_neighbor_resample = mode == "nearest"
            if n.op_type in self.SUPPORTED_INVARIANTS or is_nearest_neighbor_resample:
                in0 = n.input[0]
                if in0 is None:
                    continue
                prod0 = model.find_producer(in0)
                if prod0 is None:
                    continue

                if prod0.op_type in ["Mul", "Add", "Div"]:
                    if model.is_fork_node(prod0):
                        warnings.warn(
                            f"{self.__class__.__name__}:"
                            f" Skipping near match: {prod0.name} is a fork-node,"
                            f" try MoveLinearPastFork first"
                        )
                        continue
                    init0 = model.get_initializer(prod0.input[1])
                    if init0 is None:
                        continue
                    if np.prod(init0.shape) != 1:
                        continue
                    if model.is_fork_node(prod0):
                        model = model.transform(MoveOpPastFork(prod0.op_type))
                        return (model, True)
                    if len(init0.shape) > 0:
                        init0 = init0.flatten()[0]
                        model.set_initializer(prod0.input[1], init0)
                    old_prod0_in = prod0.input[0]
                    old_prod0_out = prod0.output[0]
                    scalar_op_odt = model.get_tensor_datatype(old_prod0_out)
                    old_n_out = n.output[0]
                    in_shape = model.get_tensor_shape(n.input[0])
                    out_shape = model.get_tensor_shape(n.output[0])
                    n.input[0] = old_prod0_in
                    n.output[0] = old_prod0_out
                    prod0.input[0] = old_prod0_out
                    prod0.output[0] = old_n_out
                    model.set_tensor_shape(n.input[0], in_shape)
                    model.set_tensor_shape(n.output[0], out_shape)
                    model.set_tensor_shape(prod0.output[0], out_shape)
                    model.set_tensor_datatype(prod0.output[0], scalar_op_odt)
                    model.set_tensor_datatype(n.output[0], DataType["FLOAT32"])
                    graph.node.remove(prod0)
                    graph.node.insert(node_ind - 1, prod0)
                    graph_modified = True
                else:
                    continue
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class MakeMaxPoolNHWC(Transformation):
    """Convert (MaxPool, NHWCTranspose) into (NHWCTranspose, MaxPoolNHWC)
    and (NCHWTranspose, MaxPool) into (MaxPoolNHWC, NCHWTranspose)."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MaxPool":
                consumer = model.find_consumer(n.output[0])
                producer = model.find_producer(n.input[0])
                if consumer is not None and consumer.op_type == "Transpose":
                    perms = list(get_by_name(consumer.attribute, "perm").ints)
                    if perms == [0, 2, 3, 1]:
                        ceil_mode = get_by_name(n.attribute, "ceil_mode")
                        if ceil_mode is not None:
                            ceil_mode = ceil_mode.i
                        else:
                            ceil_mode = 0  # default to ceil_mode=0 (equivalent to np.floor)
                        n.op_type = "MaxPoolNHWC"
                        n.domain = "qonnx.custom_op.general"
                        start_name = n.input[0]
                        mid_name = consumer.input[0]
                        end_name = consumer.output[0]
                        (b, c, hi, wi) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(mid_name)
                        consumer.input[0] = start_name
                        consumer.output[0] = mid_name
                        n.input[0] = mid_name
                        n.output[0] = end_name
                        model.set_tensor_shape(mid_name, (b, hi, wi, c))
                        model.set_tensor_shape(end_name, (b, ho, wo, c))
                        getCustomOp(n).set_nodeattr("ceil_mode", ceil_mode)
                        graph.node.remove(consumer)
                        graph.node.insert(node_ind - 1, consumer)
                        graph_modified = True
                elif producer is not None and producer.op_type == "Transpose":
                    perms = list(get_by_name(producer.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:
                        if model.is_fork_node(producer):
                            model = model.transform(MoveTransposePastFork())
                            return (model, True)
                        ceil_mode = get_by_name(n.attribute, "ceil_mode")
                        if ceil_mode is not None:
                            ceil_mode = ceil_mode.i
                        else:
                            ceil_mode = 0  # default to ceil_mode=0 (equivalent to np.floor)
                        n.op_type = "MaxPoolNHWC"
                        n.domain = "qonnx.custom_op.general"
                        start_name = producer.input[0]
                        mid_name = n.input[0]
                        end_name = n.output[0]
                        (b, hi, wi, c) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(end_name)
                        producer.input[0] = mid_name
                        producer.output[0] = end_name
                        n.input[0] = start_name
                        n.output[0] = mid_name
                        model.set_tensor_shape(mid_name, (b, ho, wo, c))
                        model.set_tensor_shape(end_name, (b, c, ho, wo))
                        getCustomOp(n).set_nodeattr("ceil_mode", ceil_mode)
                        graph.node.remove(producer)
                        graph.node.insert(node_ind, producer)
                        graph_modified = True
        return (model, graph_modified)


class MakeScaleResizeNHWC(Transformation):
    """
    Converts the inputs and outputs for all scales Resize and Upsample nodes
    from NCHW to NHWC.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Upsample" or n.op_type == "Resize":
                if model.get_tensor_layout(n.input[0]) != DataLayout.NCHW:
                    warnings.warn(
                        "%s: Input not NCHW. Can't operate transformation on node." % n.name
                    )
                    continue
                consumer = model.find_consumer(n.output[0])
                producer = model.find_producer(n.input[0])
                if n.op_type == "Upsample":
                    param_ind = 1
                    param_dtype = "float32"
                else:
                    param_ind, is_sizes = resolve_resize_param_input(model, n)
                    # Sizes are int64 and scales float32; each keeps its type.
                    param_dtype = "int64" if is_sizes else "float32"
                if producer is not None and producer.op_type == "Transpose":
                    perms = list(get_by_name(producer.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:
                        if model.is_fork_node(producer):
                            model = model.transform(MoveTransposePastFork())
                            return (model, True)
                        old_value = model.get_initializer(n.input[param_ind])
                        new_value = np.array(
                            [old_value[idx] for idx in (0, 2, 3, 1)],
                            dtype=np.dtype(param_dtype),
                        )
                        model.set_initializer(n.input[param_ind], new_value)
                        start_name = producer.input[0]
                        mid_name = n.input[0]
                        end_name = n.output[0]
                        (b, hi, wi, c) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(end_name)
                        producer.input[0] = mid_name
                        producer.output[0] = end_name
                        n.input[0] = start_name
                        n.output[0] = mid_name
                        model.set_tensor_shape(mid_name, (b, ho, wo, c))
                        model.set_tensor_shape(end_name, (b, c, ho, wo))
                        graph.node.remove(producer)
                        graph.node.insert(node_ind, producer)
                elif consumer is not None and consumer.op_type == "Transpose":
                    perms = list(get_by_name(consumer.attribute, "perm").ints)
                    if perms == [0, 2, 3, 1]:
                        old_value = model.get_initializer(n.input[param_ind])
                        new_value = np.array(
                            [old_value[idx] for idx in (0, 2, 3, 1)],
                            dtype=np.dtype(param_dtype),
                        )
                        model.set_initializer(n.input[param_ind], new_value)
                        start_name = n.input[0]
                        mid_name = consumer.input[0]
                        end_name = consumer.output[0]
                        (b, c, hi, wi) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(mid_name)
                        consumer.input[0] = start_name
                        consumer.output[0] = mid_name
                        n.input[0] = mid_name
                        n.output[0] = end_name
                        model.set_tensor_shape(mid_name, (b, hi, wi, c))
                        model.set_tensor_shape(end_name, (b, ho, wo, c))
                        graph.node.remove(consumer)
                        graph.node.insert(node_ind - 1, consumer)
        return (model, False)


class MoveOpPastFork(Transformation):
    """Move node operations past graph forks. Used when a node before a fork
    can be merged with nodes in the branches
    """

    def __init__(self, op_name_list):
        super().__init__()
        self.ops_to_move = op_name_list

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        nodes = [n for n in graph.node]
        node_ind = 0
        for n in nodes:
            node_ind += 1
            if (
                n.op_type in self.ops_to_move
                and model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                # An op's constant parameter, if any, is its input 1.
                if len(n.input) > 1:
                    op_init_param = model.get_initializer(n.input[1])
                else:
                    op_init_param = None

                # A fork whose branches all reach one node has nothing to move.
                consumers = model.find_consumers(n.output[0])
                assert len(consumers) > 1, "Must have >1 consumer"
                unique_consumer = True
                for consum_node in consumers[1:]:
                    if consumers[0] != consum_node:
                        unique_consumer = False
                        break

                if unique_consumer:
                    continue

                for consumer_node in consumers[1:]:
                    new_output_tensor_name = model.make_new_valueinfo_name()
                    if op_init_param is None:
                        new_inp_list = [n.input[0]]
                    else:
                        new_param_name = model.make_new_valueinfo_name()
                        new_inp_list = [n.input[0], new_param_name]
                        model.set_initializer(new_param_name, op_init_param)
                    new_node = deepcopy(n)
                    new_node.input[:] = new_inp_list
                    new_node.output[:] = [new_output_tensor_name]
                    graph.node.insert(node_ind, new_node)
                    node_ind += 1

                    graph.node.remove(consumer_node)
                    for idx, consumer_input in enumerate(consumer_node.input):
                        if consumer_input == n.output[0]:
                            consumer_node.input[idx] = new_output_tensor_name
                            break
                    else:
                        raise Exception("Consumer should have the current node output as input")

                    graph.node.insert(node_ind, consumer_node)

                graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveAddPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Add"])


class MoveMulPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Mul"])


class MoveLinearPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Add", "Mul"])


class MoveTransposePastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Transpose"])


def permute_shape(shape, perm):
    new_shape = np.zeros(len(shape))
    for i, p in enumerate(perm):
        new_shape[i] = shape[p]
    return [int(el) for el in new_shape]


class MoveScalarLinearPastSplit(Transformation):
    """
    Move scalar Mul and Add nodes past channel split operation.
    """

    def __init__(self):
        super().__init__()
        self.ops_to_move = ["Mul", "Add"]
        self.fork_ops = ["Split"]

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type in self.fork_ops:
                producer = model.find_producer(n.input[0])
                if producer is not None and producer.op_type in self.ops_to_move:
                    linear_param = model.get_initializer(producer.input[1])
                    if len(producer.input) != 2 or linear_param is None:
                        continue
                    if np.prod(linear_param.shape) != 1:
                        continue
                    split_outputs = n.output
                    for split_output_idx, old_split_output in enumerate(split_outputs):
                        new_mul_node = deepcopy(producer)
                        new_split_output = model.make_new_valueinfo_name()
                        model.set_tensor_datatype(
                            new_split_output, model.get_tensor_datatype(producer.input[0])
                        )

                        model.set_tensor_shape(
                            new_split_output, model.get_tensor_shape(old_split_output)
                        )

                        n.output[split_output_idx] = new_split_output
                        new_mul_node.input[0] = new_split_output
                        new_mul_node.output[0] = old_split_output

                        graph.node.insert(node_ind, new_mul_node)
                        node_ind += 1

                    n.input[0] = producer.input[0]
                    graph.node.remove(producer)
                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)


class MoveTransposePastSplit(Transformation):
    def __init__(self):
        super().__init__()
        self.ops_to_move = ["Transpose"]
        self.fork_ops = ["Split"]

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type in self.fork_ops:
                producer = model.find_producer(n.input[0])
                if producer is not None and producer.op_type in self.ops_to_move:
                    initial_perm = get_by_name(producer.attribute, "perm").ints
                    reverse_perm = np.argsort(initial_perm)
                    split_outputs = n.output
                    for split_output_idx, old_split_output in enumerate(split_outputs):
                        new_trans_node = deepcopy(producer)
                        new_split_output = model.make_new_valueinfo_name()
                        old_split_output_shape = model.get_tensor_shape(old_split_output)
                        model.set_tensor_datatype(
                            new_split_output, model.get_tensor_datatype(producer.input[0])
                        )

                        model.set_tensor_shape(
                            new_split_output, permute_shape(old_split_output_shape, reverse_perm)
                        )

                        n.output[split_output_idx] = new_split_output
                        new_trans_node.input[0] = new_split_output
                        new_trans_node.output[0] = old_split_output

                        graph.node.insert(node_ind, new_trans_node)
                        node_ind += 1

                    old_split_axis = get_by_name(n.attribute, "axis").i
                    get_by_name(n.attribute, "axis").i = initial_perm[old_split_axis]
                    n.input[0] = producer.input[0]
                    graph.node.remove(producer)
                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)


class MoveMaxPoolPastMultiThreshold(Transformation):
    """Move MaxPool nodes past MultiThreshold nodes on linear segments of the graph."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "MaxPool" and not model.is_fork_node(n):
                consumer = model.find_consumer(n.output[0])
                pads = get_by_name(n.attribute, "pads")
                has_padding = False
                if pads is not None:
                    pads = list(pads.ints)
                    has_padding = np.prod(pads) != 0
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    mt_out = consumer.output[0]
                    mt_odt = model.get_tensor_datatype(mt_out)
                    if mt_odt.signed() and has_padding:
                        warnings.warn("Skipping padded MaxPool + signed-output MultiThreshold")
                        continue
                    # Max commutes only with a monotonic MultiThreshold: thresholds
                    # non-decreasing, scale non-negative.
                    T = model.get_initializer(consumer.input[1])
                    T_sorted = np.sort(T, axis=1)
                    assert (
                        T == T_sorted
                    ).all(), "MultiThreshold must have non-decreasing thresholds"
                    mt_inst = getCustomOp(consumer)
                    if mt_inst.get_nodeattr("out_scale") < 0:
                        warnings.warn("Skipping MultiThreshold with negative out_scale")
                        continue

                    graph.node.remove(n)
                    graph.node.remove(consumer)

                    group_in = n.input[0]
                    group_middle = model.make_new_valueinfo_name()
                    group_out = consumer.output[0]

                    consumer.input[0] = group_in
                    consumer.output[0] = group_middle

                    n.input[0] = group_middle
                    n.output[0] = group_out

                    graph.node.insert(node_ind - 1, consumer)
                    graph.node.insert(node_ind, n)

                    graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveFlattenPastTopK(Transformation):
    """Move flatten node past a succeeding topk node, if the "axis" attribute in topk
    is set to -1 and the data layout before the flatten is NHWC with H=W=1"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Flatten":
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "TopK":
                    axis = get_by_name(consumer.attribute, "axis")
                    if axis is None or axis.i != -1:
                        continue
                    start_name = n.input[0]
                    data_layout = model.get_tensor_layout(start_name)
                    if data_layout != DataLayout.NHWC:
                        warnings.warn(
                            """Transformation can't be applied. The input
                            to flatten has to have DataLayout.NHWC"""
                        )
                        continue
                    (b, h, w, c) = model.get_tensor_shape(start_name)
                    if h != 1 or w != 1:
                        continue
                    k = model.get_tensor_shape(consumer.output[1])[-1]

                    middle_name = model.make_new_valueinfo_name()
                    topk_indices = oh.make_tensor_value_info(
                        middle_name, TensorProto.INT64, [b, h, w, k]
                    )
                    end_name = consumer.output[1]
                    graph.value_info.append(topk_indices)

                    graph.node.remove(n)
                    graph.node.remove(consumer)

                    consumer.input[0] = start_name
                    consumer.output[1] = middle_name
                    model.set_tensor_shape(consumer.output[0], (b, h, w, k))

                    n.input[0] = middle_name
                    n.output[0] = end_name

                    graph.node.insert(node_ind - 1, consumer)
                    graph.node.insert(node_ind, n)

                    graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveFlattenPastAffine(Transformation):
    """Moves a node that implements a (1, -1) reshape past a MatMul, Mul or Add node."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Flatten" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and (
                        consumer.op_type == "MatMul"
                        or consumer.op_type == "Mul"
                        or consumer.op_type == "Add"
                    )
                    and not model.is_join_node(consumer)
                ):
                    start_name = n.input[0]
                    datalayout = model.get_tensor_layout(start_name)
                    if datalayout == DataLayout.NHWC:
                        (b, h, w, c) = model.get_tensor_shape(start_name)
                        if h != 1 or w != 1:
                            warnings.warn(
                                """The Transformation can only be performed if
                            H=W=1."""
                            )
                            continue
                    else:
                        warnings.warn(
                            """The Transformation can only be performed on
                            operations that operate on data layout NHWC."""
                        )
                        continue
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    op_param_name = consumer.input[1]
                    A = model.get_initializer(op_param_name)
                    if A is None:
                        warnings.warn("Param is not constant, skipping")
                        continue
                    op_in_dt = model.get_tensor_datatype(consumer.input[0])
                    op_out_dt = model.get_tensor_datatype(consumer.output[0])
                    start_shape = model.get_tensor_shape(start_name)
                    dummy_in = np.random.uniform(low=0, high=1, size=(start_shape))

                    if consumer.op_type == "MatMul":
                        dummy_out = np.matmul(dummy_in, A)
                    elif consumer.op_type == "Mul":
                        dummy_out = dummy_in * A
                    elif consumer.op_type == "Add":
                        dummy_out = dummy_in + A

                    new_op = oh.make_node(
                        consumer.op_type,
                        [start_name, op_param_name],
                        [middle_name],
                        name=consumer.name,
                    )
                    new_flatten = oh.make_node("Flatten", [middle_name], [end_name])
                    graph.node.insert(node_ind, new_op)
                    graph.node.insert(node_ind + 1, new_flatten)
                    model.set_tensor_shape(middle_name, dummy_out.shape)
                    # Flatten keeps the datatype.
                    model.set_tensor_datatype(start_name, op_in_dt)
                    model.set_tensor_datatype(middle_name, op_out_dt)
                    model.set_tensor_datatype(end_name, op_out_dt)
                    model.set_tensor_layout(start_name, DataLayout.NHWC)
                    model.set_tensor_layout(middle_name, DataLayout.NHWC)
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
        return (model, graph_modified)


class MoveTransposePastScalarMul(Transformation):
    """Moves a Transpose node past a scalar Mul node"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Mul"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn("Mul param is not constant, skipping")
                        continue
                    transp_node = n
                    mul_node = consumer
                    start_name = transp_node.input[0]
                    middle_name = transp_node.output[0]
                    end_name = mul_node.output[0]
                    transp_in_shape = model.get_tensor_shape(start_name)
                    transp_out_shape = model.get_tensor_shape(middle_name)
                    transp_in_layout = model.get_tensor_layout(start_name)
                    transp_out_layout = model.get_tensor_layout(middle_name)
                    if transp_in_layout is None or transp_out_layout is None:
                        warnings.warn(
                            """Datalayout is not set for tensors.
                            Transformation can't be applied."""
                        )
                        continue
                    if all(x == 1 for x in A.shape):
                        mul_node.input[0] = start_name
                        model.set_tensor_shape(start_name, transp_in_shape)
                        model.set_tensor_layout(start_name, transp_in_layout)
                        mul_node.output[0] = middle_name
                        model.set_tensor_shape(middle_name, transp_in_shape)
                        model.set_tensor_layout(middle_name, transp_in_layout)
                        transp_node.input[0] = middle_name
                        transp_node.output[0] = end_name
                        model.set_tensor_shape(end_name, transp_out_shape)
                        model.set_tensor_layout(end_name, transp_out_layout)
                        graph.node.remove(transp_node)
                        graph.node.insert(node_ind, transp_node)
                        graph_modified = True

        if graph_modified is True:
            model = model.transform(InferDataLayouts())
            model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveIdenticalOpPastJoinOp(Transformation):
    """
    Move multiple identical operations on different branches past the common join node.
    It assumes the shape to be preserved by the join op in the default move_node() method
    """

    def __init__(self, identical_op_list, join_node_list):
        super().__init__()
        self.ops_to_move = identical_op_list
        self.join_node_op = join_node_list

    def move_node(self, model, n, producers):
        """
        Should be overwritten for some operations

        Returns:
            bool: whether moving the node was successful
        """
        identical_ops_inputs = [p.input[0] for p in producers]
        join_out = n.output[0]

        for i in range(len(n.input)):
            n.input[i] = identical_ops_inputs[i]

        new_join_output = model.make_new_valueinfo_name()

        n.output[0] = new_join_output
        producers[0].input[0] = new_join_output
        producers[0].output[0] = join_out

        for prod in producers[1:]:
            model.graph.node.remove(prod)

        return True

    def are_producers_identical(self, model, producers):
        """
        Checks only op_types
        Should be overwritten for additional checks
        """
        op_types = [prod.op_type for prod in producers]
        for op in op_types:
            if op != op_types[0]:
                return False
        return True

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type in self.join_node_op and model.is_join_node(n):
                inputs = n.input
                if None in inputs:
                    continue

                producers = [model.find_producer(inp) for inp in inputs]
                if producers[0].op_type not in self.ops_to_move:
                    continue
                identical_ops = self.are_producers_identical(model, producers)
                if not identical_ops:
                    warnings.warn("Producers not identical, skipping")
                    continue

                # A forking producer is moved past its fork first.
                for prod in producers:
                    if model.is_fork_node(prod) and not model.is_join_node(prod):
                        model = model.transform(MoveOpPastFork(self.ops_to_move))
                        return (model, True)
                graph_modified = self.move_node(model, n, producers)

        if graph_modified:
            # move_node left the new tensors' shapes to inference.
            model = model.transform(InferDataLayouts())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)


class MoveTransposePastJoinAdd(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Transpose"], ["Add"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        first_perm = get_by_name(producers[0].attribute, "perm").ints
        for producer in producers:
            if first_perm != get_by_name(producer.attribute, "perm").ints:
                False
        return True


class MoveMulPastJoinAdd(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Mul"], ["Add"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        first_mul = model.get_initializer(producers[0].input[1])
        if first_mul is None:
            return False
        for producer in producers:
            if first_mul != model.get_initializer(producer.input[1]):
                return False
        return True


class MoveAddPastJoinAdd(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Add"], ["Add"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        for producer in producers:
            if model.get_initializer(producer.input[1]) is None:
                return False
        return True

    def move_node(self, model, n, producers):
        """
        We use the base move_node method to move the first producer
        past the join node (and delete the rest)
        """
        add_inits = [model.get_initializer(producer.input[1]) for producer in producers]
        new_init = np.sum(add_inits)
        model.set_initializer(producers[0].input[1], new_init)
        super().move_node(model, n, producers)

        return True


class MoveMulPastJoinMul(MoveIdenticalOpPastJoinOp):
    """Move constant Mul operations past a join Mul node, folding the two
    constants into a single one: (x*A) * (y*B) -> (x*y) * (A*B).
    Unlike MoveMulPastJoinAdd, the two constants need not be equal since they
    are combined into their product."""

    def __init__(self):
        super().__init__(["Mul"], ["Mul"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        for producer in producers:
            if model.get_initializer(producer.input[1]) is None:
                return False
        return True

    def move_node(self, model, n, producers):
        """
        Fold the producer constants into their product on the surviving
        producer, then use the base move_node to move it past the join node
        (and delete the rest).
        """
        mul_inits = [model.get_initializer(producer.input[1]) for producer in producers]
        new_init = np.prod(mul_inits, axis=0)
        model.set_initializer(producers[0].input[1], new_init)
        super().move_node(model, n, producers)

        return True


class MoveTransposePastJoinConcat(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Transpose"], ["Concat"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        first_perm = get_by_name(producers[0].attribute, "perm").ints
        for producer in producers:
            if first_perm != get_by_name(producer.attribute, "perm").ints:
                False
        return True

    def move_node(self, model, n, producers):
        trans_inputs = [prod.input[0] for prod in producers]
        concat_out = n.output[0]
        for i in range(len(n.input)):
            n.input[i] = trans_inputs[i]

        new_concat_out = model.make_new_valueinfo_name()  # reuse tensor
        transpose_perm = get_by_name(producers[0].attribute, "perm").ints
        reverse_perm = np.argsort(transpose_perm)
        new_concat_out_shape = permute_shape(model.get_tensor_shape(concat_out), reverse_perm)
        new_concat_out_layout = model.get_tensor_layout(trans_inputs[0])
        model.set_tensor_shape(new_concat_out, new_concat_out_shape)
        if new_concat_out_layout:
            model.set_tensor_layout(new_concat_out, new_concat_out_layout)
        old_concat_axis = get_by_name(n.attribute, "axis").i
        get_by_name(n.attribute, "axis").i = transpose_perm[old_concat_axis]

        n.output[0] = new_concat_out
        producers[0].input[0] = new_concat_out
        producers[0].output[0] = concat_out

        for prod in producers[1:]:
            model.graph.node.remove(prod)

        return True


class MoveAffinePastJoinConcat(MoveIdenticalOpPastJoinOp):
    """
    Applies to scalar linear or channelwise affine ops with the same parameter value
    """

    def __init__(self, linear_ops=["Mul", "Add"]):
        super().__init__(linear_ops, ["Concat"])

    def are_producers_identical_scalar_ops(self, model, producers):
        first_param = model.get_initializer(producers[0].input[1])
        for producer in producers:
            producer_param = model.get_initializer(producer.input[1])
            if (first_param != producer_param).any() or np.prod(producer_param.shape) != 1:
                return False

        return True

    def are_producers_channelwise_ops(self, channel_dim, model, producers):
        for producer in producers:
            producer_input = producer.input[0]
            num_channels = model.get_tensor_shape(producer_input)[channel_dim]
            producer_param = model.get_initializer(producer.input[1])
            if (
                len(producer_param.shape) < channel_dim
                or producer_param.shape[channel_dim] != num_channels
            ):
                return False

        return True

    def move_node(self, model, n, producers):
        for producer in producers:
            producer_init = model.get_initializer(producer.input[1])
            if len(producer.input) != 2 or producer_init is None:
                warnings.warn("Producer found that is not single-input, skipping")
                return False

        channelwise_op = False
        identical_scalar_op = self.are_producers_identical_scalar_ops(model, producers)
        if not identical_scalar_op:
            channel_dim = get_by_name(n.attribute, "axis").i
            channelwise_op = self.are_producers_channelwise_ops(channel_dim, model, producers)
            if not channelwise_op:
                warnings.warn(
                    "Producers are neither identical scalar ops nor channelwise ops, skipping"
                )
                return False

        producers_inputs = [prod.input[0] for prod in producers]
        concat_out = n.output[0]
        for i in range(len(n.input)):
            n.input[i] = producers_inputs[i]
        new_concat_out = model.make_new_valueinfo_name()
        new_concat_out_layout = model.get_tensor_layout(producers_inputs[0])
        model.set_tensor_shape(new_concat_out, model.get_tensor_shape(concat_out))
        if new_concat_out_layout:
            model.set_tensor_layout(new_concat_out, new_concat_out_layout)
        model.set_tensor_datatype(new_concat_out, model.get_tensor_datatype(producers_inputs[0]))

        if channelwise_op:
            producers_params = [model.get_initializer(prod.input[1]) for prod in producers]
            new_mul_tensor = np.concatenate(producers_params, axis=channel_dim)
            model.set_initializer(producers[0].input[1], new_mul_tensor)

        n.output[0] = new_concat_out
        producers[0].input[0] = new_concat_out
        producers[0].output[0] = concat_out

        for prod in producers[1:]:
            model.graph.node.remove(prod)

        return True


class MoveMulPastJoinConcat(MoveAffinePastJoinConcat):
    def __init__(self):
        super().__init__(["Mul"])


class MoveAddPastJoinConcat(MoveAffinePastJoinConcat):
    def __init__(self):
        super().__init__(["Add"])
