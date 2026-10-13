# Copyright (c) 2020, Xilinx
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
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

from finn.util.basic import resolve_resize_param_input


class AbsorbScalarBiasIntoMultiThreshold(Transformation):
    """Absorb a scalar bias that follows a MultiThreshold (an Add) into its out_bias,
    and derive the output datatype again, with its range. A bias can widen the output
    datatype: it is absorbed only where the output grows by at most
    ``max_bitwidth_increase`` bits (by default none) over the MultiThreshold's own
    datatype before it, trading a wider output for one Add fewer."""

    def __init__(self, max_bitwidth_increase=0):
        super().__init__()
        self.max_bitwidth_increase = max_bitwidth_increase

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            if (
                node.op_type == "MultiThreshold"
                and not model.is_fork_node(node)
                and not model.is_join_node(node)
            ):
                consumer = model.find_consumer(node.output[0])
                # A Sub must be made an Add first.
                if consumer is not None and consumer.op_type == "Add":
                    bias = model.get_initializer(consumer.input[1])

                    if bias is None:
                        warnings.warn(
                            f"{self.__class__.__name__}: Bias not constant for"
                            f" {consumer.name}, skipping."
                        )
                        continue

                    thresholds = model.get_initializer(node.input[1])

                    if thresholds is None:
                        warnings.warn(
                            f"{self.__class__.__name__}: Thresholds not"
                            f" constant for {node.name}, skipping."
                        )
                        continue

                    # An attribute holds a scalar only.
                    if not (bias.ndim == 0 or all(x == 1 for x in bias.shape)):
                        warnings.warn(
                            f"{self.__class__.__name__}: Bias not scalar"
                            f" for {consumer.name}, skipping."
                        )
                        continue

                    threshold_op = getCustomOp(node)
                    old_odt = threshold_op.get_nodeattr("out_dtype")
                    bits = DataType[old_odt].bitwidth()

                    bias = bias.flatten()[0]
                    out_bias = threshold_op.get_nodeattr("out_bias") + bias
                    # The step count is in [0, num_steps]: the output range, shifted.
                    new_min = out_bias
                    new_max = out_bias + thresholds.shape[-1]

                    # A signed type that holds -(new_max + 1) holds new_max too.
                    if new_min >= 0:
                        odt = DataType.get_smallest_possible(new_max)
                    else:
                        odt = DataType.get_smallest_possible(min(new_min, -(new_max + 1)))

                    if not (odt.allowed(new_max) and odt.allowed(new_min)):
                        warnings.warn(
                            f"{self.__class__.__name__}: Cannot absorb bias"
                            f" from {consumer.name} into {node.name}: {bias}"
                        )
                        continue

                    bitwidth_increase = odt.bitwidth() - bits
                    if bitwidth_increase > self.max_bitwidth_increase:
                        warnings.warn(
                            f"{self.__class__.__name__}: Not absorbing bias from"
                            f" {consumer.name} into {node.name}: would grow"
                            f" output datatype from {old_odt} to {odt.name}"
                            f" (+{bitwidth_increase} bits >"
                            f" max_bitwidth_increase={self.max_bitwidth_increase})."
                            f" Increase max_bitwidth_increase to allow this"
                            f" absorption if the wider output type is acceptable."
                        )
                        continue

                    if bitwidth_increase > 0:
                        warnings.warn(
                            f"{self.__class__.__name__}: Output datatype for"
                            f" {node.name} growing from {old_odt} to {odt.name}"
                        )

                    threshold_op.set_nodeattr("out_bias", out_bias)
                    threshold_op.set_nodeattr("out_dtype", odt.name)
                    node.output[0] = consumer.output[0]
                    graph.node.remove(consumer)

                    graph_modified = True
                    break
        # The MultiThreshold states its new datatype with its range.
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        return model, graph_modified


class AbsorbAddIntoMultiThreshold(Transformation):
    """Absorb preceding Add ops into MultiThreshold by updating the threshold
    values. Only scalar/1D add vectors can be absorbed."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    add_weight_name = n.input[1]
                    threshold_name = consumer.input[1]
                    A = model.get_initializer(add_weight_name)
                    T = model.get_initializer(threshold_name)
                    assert A is not None, "Initializer for add weights is not set."
                    assert T is not None, "Initializer for thresholds is not set."
                    start_name = n.input[0]
                    is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)

                    shape = A.shape
                    layout = model.get_tensor_layout(add_weight_name)
                    # Without a layout annotation, the layout is guessed from the rank.
                    if layout is None and len(shape) < 5:
                        rank_to_layout = {0: None, 1: "C", 2: "NC", 3: "NWC", 4: "NCHW"}
                        layout = rank_to_layout[len(shape)]
                    if layout is not None and "C" in layout:
                        cdim = layout.index("C")
                    else:
                        cdim = 1
                        warnings.warn(
                            f"No layout annotations for {add_weight_name}:"
                            f" Assuming channel dimension at index {cdim}"
                        )

                    # At most per channel.
                    if is_scalar or A.shape[cdim] == A.size:
                        # In float64: ONNX gives thresholds no type, and float32 would
                        # round them.
                        Tnew = T.astype(np.float64) - A.reshape(-1, 1).astype(np.float64)
                        model.set_initializer(threshold_name, Tnew)
                        consumer.input[0] = start_name
                        graph.node.remove(n)
                        graph_modified = True
        return model, graph_modified


class AbsorbMulIntoMultiThreshold(Transformation):
    """Absorb preceding Mul ops into MultiThreshold by updating the threshold
    values. Only *positive* scalar/1D mul vectors can be absorbed."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                mul_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                assert A is not None, "Initializer for mul weights is not set."
                is_signed = (A < 0).any()
                is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    if not is_signed and (is_1d or is_scalar):
                        threshold_name = consumer.input[1]
                        T = model.get_initializer(threshold_name)
                        assert T is not None, "Initializer for thresholds is not set."
                        start_name = n.input[0]
                        # In float64: ONNX gives thresholds no type, and float32 would
                        # round them.
                        Tnew = T.astype(np.float64) / A.reshape(-1, 1).astype(np.float64)
                        model.set_initializer(threshold_name, Tnew)
                        consumer.input[0] = start_name
                        graph.node.remove(n)
                        graph_modified = True
        return (model, graph_modified)


class FactorOutMulSignMagnitude(Transformation):
    """Split multiply-by-constant nodes into two multiply-by-constant nodes,
    where the first node is a bipolar vector (of signs) and the second is a
    vector of magnitudes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_join_node(n):
                mul_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                assert A is not None, "Initializer for mul weights is not set."
                is_scalar = np.prod(A.shape) == 1
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                is_not_bipolar = model.get_tensor_datatype(mul_weight_name) != DataType["BIPOLAR"]
                is_signed = (A < 0).any()
                if is_signed and (is_scalar or is_1d) and is_not_bipolar:
                    start_name = n.input[0]
                    in_shape = model.get_tensor_shape(start_name)
                    middle_name = model.make_new_valueinfo_name()
                    model.set_tensor_shape(middle_name, in_shape)
                    sign_mul_param_name = model.make_new_valueinfo_name()
                    sgn = np.sign(A)
                    model.set_initializer(sign_mul_param_name, sgn)
                    model.set_tensor_datatype(sign_mul_param_name, DataType["BIPOLAR"])
                    model.set_initializer(mul_weight_name, np.abs(A))
                    new_mul = oh.make_node("Mul", [start_name, sign_mul_param_name], [middle_name])
                    n.input[0] = middle_name
                    graph.node.insert(node_ind - 1, new_mul)
                    graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoMatMul(Transformation):
    """Absorb bipolar or binary multiplications into the preceding matrix
    multiply."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul" and not model.is_fork_node(n):
                matmul_weight_name = n.input[1]
                W = model.get_initializer(matmul_weight_name)
                Wdt = model.get_tensor_datatype(matmul_weight_name)
                if W is None:
                    continue
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        continue
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    if is_1bit:
                        Wnew = A * W
                        assert (
                            Wnew.shape == W.shape
                        ), """Shape of new weights is not
                        the same as the shape of the weight matrix before."""
                        check_fxn = np.vectorize(lambda x: Wdt.allowed(x))
                        if check_fxn(Wnew).all():
                            model.set_initializer(matmul_weight_name, Wnew)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoConv(Transformation):
    """Absorb bipolar or binary multiplications into the preceding convolution."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv" and not model.is_fork_node(n):
                conv_weight_name = n.input[1]
                W = model.get_initializer(conv_weight_name)
                Wdt = model.get_tensor_datatype(conv_weight_name)
                if W is None:
                    continue
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        continue
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    is_scalar = np.prod(A.shape) == 1
                    actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                    is_1d = actual_ndims == 1
                    if is_1bit and (is_1d or is_scalar):
                        # The Mul scales each output channel.
                        Wnew = A.reshape(-1, 1, 1, 1) * W
                        assert (
                            Wnew.shape == W.shape
                        ), """Shape of new weights is not
                        the same as the shape of the conv weights before."""
                        check_fxn = np.vectorize(lambda x: Wdt.allowed(x))
                        if check_fxn(Wnew).all():
                            model.set_initializer(conv_weight_name, Wnew)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class AbsorbTransposeIntoMultiThreshold(Transformation):
    """For (NCHWTranspose -> MultiThreshold) move Transpose past MultiThreshold
    and set its data_layout mode to NHWC."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in model.graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Transpose" and not model.is_fork_node(n):
                perms = list(get_by_name(n.attribute, "perm").ints)
                if perms == [0, 3, 1, 2]:
                    mt_cand = model.find_consumer(n.output[0])
                    if mt_cand is not None and mt_cand.op_type == "MultiThreshold":
                        mt_cand_orig_output = mt_cand.output[0]
                        mt = getCustomOp(mt_cand)
                        mt.set_nodeattr("data_layout", "NHWC")
                        mt_cand.input[0] = n.input[0]
                        intermediate_tensor_name = model.make_new_valueinfo_name()
                        intermediate_tensor_shape = model.get_tensor_shape(n.input[0])
                        intermediate_tensor_finn_dtype = model.get_tensor_datatype(
                            mt_cand.output[0]
                        )
                        model.set_tensor_shape(intermediate_tensor_name, intermediate_tensor_shape)
                        model.set_tensor_layout(intermediate_tensor_name, DataLayout.NHWC)
                        model.set_tensor_datatype(
                            intermediate_tensor_name, intermediate_tensor_finn_dtype
                        )
                        mt_cand.output[0] = intermediate_tensor_name
                        graph.node.remove(n)
                        new_transpose = oh.make_node(
                            "Transpose",
                            [intermediate_tensor_name],
                            [mt_cand_orig_output],
                            perm=[0, 3, 1, 2],
                        )
                        graph.node.insert(node_ind + 1, new_transpose)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbTransposeIntoFlatten(Transformation):
    """Absorb transpose node into succeeding flatten node, if H=W=1 and the first
    dimension stays the same. Can also be applied if flatten is implemented implicitly
    by a reshape node with shape [1, -1] and the first input dimension is 1"""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Reshape" and (model.get_initializer(n.input[1]) == [1, -1]).all()
            ) or n.op_type == "Flatten":
                prod = model.find_producer(n.input[0])
                if (
                    prod is not None
                    and prod.op_type == "Transpose"
                    and get_by_name(prod.attribute, "perm").ints[0] == 0
                ):
                    data_layout = model.get_tensor_layout(prod.input[0])
                    if data_layout is None:
                        warnings.warn(
                            """Data layout for input tensor of Transpose node is not set.
                                To use AbsorbTransposeIntoFlatten transformation
                                please set tensor data layout."""
                        )
                        continue
                    elif data_layout == DataLayout.NCHW:
                        (b, c, h, w) = model.get_tensor_shape(prod.input[0])
                        # Only a transpose of a 1x1 image is a reshape.
                        if h != 1 or w != 1:
                            continue
                        # Flatten keeps the first axis and Reshape [1, -1] does not:
                        # they agree only for a batch of one.
                        if n.op_type == "Reshape" and b != 1:
                            continue
                    elif data_layout == DataLayout.NHWC:
                        (b, h, w, c) = model.get_tensor_shape(prod.input[0])
                        if h != 1 or w != 1:
                            continue
                        if n.op_type == "Reshape" and b != 1:
                            continue
                    node = oh.make_node("Flatten", [prod.input[0]], [n.output[0]])
                    graph.node.remove(n)
                    graph.node.remove(prod)
                    graph.node.insert(node_ind, node)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbScalarMulAddIntoTopK(Transformation):
    """Remove mul/add node prior to topk node if the op is scalar. Note that
    the TopK output probabilities will change, but the indices won't."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "TopK":
                prod = model.find_producer(n.input[0])
                if prod is not None and (prod.op_type in ["Mul", "Add"]):
                    prod_input = prod.input[0]
                    param_name = prod.input[1]
                    A = model.get_initializer(param_name)
                    if A is None:
                        warnings.warn("Param is not constant, skipping")
                        continue
                    is_scalar = all(x == 1 for x in A.shape)
                    is_scalar_pos_mul = is_scalar and (prod.op_type == "Mul") and A > 0
                    is_scalar_add = is_scalar and (prod.op_type == "Add")
                    if is_scalar_pos_mul or is_scalar_add:
                        # A positive scale or any shift keeps the order TopK selects by.
                        graph.node.remove(prod)
                        n.input[0] = prod_input
                        model.set_tensor_datatype(n.input[0], DataType["FLOAT32"])
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbConsecutiveTransposes(Transformation):
    """Remove (Transpose -> Transpose) patterns when the input and output
    of the pattern have the same layout."""

    def are_opposite_permutations(self, perms1, perms2):
        if len(perms1) != len(perms2):
            return False
        assert 0 <= max(perms2) < len(perms2), "invalid permutation"
        assert 0 <= max(perms1) < len(perms1), "invalid permutation"

        for i, p in enumerate(perms2):
            if perms1[p] != i:
                return False

        return True

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            if node.op_type == "Transpose":
                next_nodes = model.find_consumers(node.output[0])
                perms1 = list(get_by_name(node.attribute, "perm").ints)
                if len(next_nodes) == 0:
                    continue
                all_opposite_transposes = True
                for next_node in next_nodes:
                    if next_node is not None and next_node.op_type == "Transpose":
                        perms2 = list(get_by_name(next_node.attribute, "perm").ints)
                        if not self.are_opposite_permutations(perms1, perms2):
                            all_opposite_transposes = False
                            break
                    else:
                        all_opposite_transposes = False
                        break
                if not all_opposite_transposes:
                    continue
                source_tensor = node.input[0]
                for next_node in next_nodes:
                    nextnode_out = next_node.output[0]
                    assert nextnode_out not in [x.name for x in model.graph.output]
                    consumers = model.find_consumers(nextnode_out)
                    for cons in consumers:
                        for i, iname in enumerate(cons.input):
                            if iname == nextnode_out:
                                cons.input[i] = source_tensor
                    graph.node.remove(next_node)
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbTransposeIntoResize(Transformation):
    """For (NCHWTranspose -> Resize) move Transpose past Resize and
    change the Resize node's attributes accordingly."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Transpose" and not model.is_fork_node(node):
                perms = list(get_by_name(node.attribute, "perm").ints)
                if perms == [0, 3, 1, 2]:
                    mt_cand = model.find_consumer(node.output[0])
                    if mt_cand is not None and mt_cand.op_type == "Resize":
                        mode = get_by_name(mt_cand.attribute, "mode").s.decode("ascii")
                        if mode != "nearest":
                            continue
                        param_ind, is_sizes = resolve_resize_param_input(model, mt_cand)
                        if is_sizes:
                            # Sizes become scales, in the scales input (index 2).
                            sizes = model.get_initializer(mt_cand.input[param_ind])
                            ishape = model.get_tensor_shape(mt_cand.input[0])
                            ns, cs, hs, ws = sizes / np.asarray(ishape)
                            model.set_initializer(mt_cand.input[2], np.asarray([ns, cs, hs, ws]))
                            mt_cand.input.remove(mt_cand.input[param_ind])
                            param_ind = 2
                        scales = model.get_initializer(mt_cand.input[param_ind])
                        assert scales is not None
                        ns, cs, hs, ws = scales
                        model.set_initializer(
                            mt_cand.input[param_ind], np.asarray([ns, hs, ws, cs])
                        )
                        mt_cand.input[0] = node.input[0]
                        graph.node.remove(node)
                        is_last_node = mt_cand.output[0] in [x.name for x in model.graph.output]

                        new_tensor_name = model.make_new_valueinfo_name()
                        if is_last_node:
                            trans_input = new_tensor_name
                            trans_output = mt_cand.output[0]
                        else:
                            trans_input = mt_cand.output[0]
                            trans_output = new_tensor_name
                        n, c, hx, wx = model.get_tensor_shape(mt_cand.output[0])
                        model.set_tensor_shape(trans_input, (n, hx, wx, c))
                        model.set_tensor_shape(trans_output, (n, c, hx, wx))
                        new_transpose = oh.make_node(
                            "Transpose",
                            [trans_input],
                            [trans_output],
                            perm=[0, 3, 1, 2],
                        )
                        final_t_cands = model.find_consumers(mt_cand.output[0])
                        for final_t_cand in final_t_cands:
                            final_t_cand.input[0] = trans_output
                        mt_cand.output[0] = trans_input
                        graph.node.insert(node_ind + 1, new_transpose)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
