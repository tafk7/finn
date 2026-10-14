# Copyright (c) 2021, Xilinx
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


import math
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.transformation.qonnx.qonnx_activation_handlers import INT_QUANT_OP_TYPES


def _refused(owner, code, node, message):
    """The refusal of ``node`` by the transformation ``owner``."""
    details = (("node", node.name),)
    return Refusal([Finding(FindingKind.REJECTION, code, owner, message, details)])


def _least_behind(model, mt_node, consumer):
    """The least value of ``mt_node``'s output, ``out_bias + out_scale * [0, N]``, with
    the constant its ``consumer`` adds if that is an Add."""
    mt_inst = getCustomOp(mt_node)
    steps = model.get_initializer(mt_node.input[1]).shape[-1]
    out_bias = mt_inst.get_nodeattr("out_bias")
    least = out_bias + min(0.0, mt_inst.get_nodeattr("out_scale") * steps)
    if consumer.op_type == "Add":
        bias = model.get_initializer(consumer.input[1])
        if bias is not None:
            least += float(np.min(bias))
    return least


def _get_signed_from_upstream(model, trunc_node):
    """Whether the input of ``trunc_node`` is signed: its datatype's, else the first
    known datatype up the graph through sign-preserving nodes (a MultiThreshold's
    values with the bias behind it)."""
    node = trunc_node
    signed = None
    inp_dt = model.get_tensor_datatype(node.input[0])
    if inp_dt is not None and inp_dt != DataType["FLOAT32"]:
        signed = inp_dt.signed()
    # Datatype inference works top down, so the sign is looked for up the graph.
    curr_node = node
    if signed is None:
        while curr_node is not None:
            if model.is_join_node(curr_node):
                raise RuntimeError(
                    "Datatype Inference for the Trunc node only supports "
                    "linear nodes in the upstream path."
                )
            next_node = model.find_direct_predecessors(curr_node)
            if next_node is None:
                raise RuntimeError(
                    "Could not infere the Datatype for the Trunc node due to "
                    "missing upstream ndoes."
                )
            next_node = next_node[0]
            # A MultiThreshold's datatype is that of its own values, before the bias
            # the Add behind it adds.
            if next_node.op_type == "MultiThreshold":
                signed = _least_behind(model, next_node, curr_node) < 0
                break
            out_dt = model.get_tensor_datatype(next_node.output[0])
            if out_dt is not None and out_dt != DataType["FLOAT32"]:
                signed = out_dt.signed()
                break
            # Special cases where the node has an intrinsic datatype.
            if next_node.op_type == "BipolarQuant":
                signed = True
                break
            if next_node.op_type in INT_QUANT_OP_TYPES:
                q_inst = getCustomOp(next_node, onnx_opset_version=9)
                out_dt = q_inst.get_integer_datatype(model)
                if out_dt is not None and out_dt != DataType["FLOAT32"]:
                    signed = out_dt.signed()
                    break

            sign_preserving_ops = ["Add", "Mul", "AveragePool", "Pad"]
            if next_node.op_type not in sign_preserving_ops:
                raise RuntimeError(
                    f"Could not infere the Datatype for the Trunc node, "
                    f"because the sign of the input datatype could not be infered "
                    f"from upstream nodes. And traversal further up the graph was "
                    f"disallowed, since the next node type {next_node.op_type} "
                    f"is not in the list of "
                    f"sign preserving ops {sign_preserving_ops}."
                )
            curr_node = next_node

    if signed is None:
        raise RuntimeError(
            "Could not infere the Datatype for the Trunc node, "
            "because the sign of the input datatype could not be infered "
            "from upstream nodes."
        )

    return signed


class AvgPoolAndTruncToQuantAvgPool(Transformation):
    """Converts AveragePool -> Mul (scalar) -> Trunc, or AveragePool -> Trunc, to
    Div -> QuantAvgPool2d -> Mul."""

    def apply(self, model):
        opset_imports = model.get_opset_imports()
        if "qonnx.custom_op.general" in opset_imports:
            trunc_opset = opset_imports["qonnx.custom_op.general"]
        elif "onnx.brevitas" in opset_imports:
            trunc_opset = opset_imports["onnx.brevitas"]
        else:
            trunc_opset = 1  # Default to v1 if no opset found
        if trunc_opset == 1:
            model = model.transform(AvgPoolAndTruncv1ToQuantAvgPool())
            return model, False
        elif trunc_opset == 2:
            model = model.transform(AvgPoolAndTruncv2ToQuantAvgPool())
            return model, False
        else:
            raise NotImplementedError(
                f"AvgPoolAndTruncToQuantAvgPool not implemented for "
                f"Trunc opset version {trunc_opset}."
            )


class AvgPoolAndTruncv1ToQuantAvgPool(Transformation):
    """Converts AveragePool -> Mul (scalar) -> Trunc (v1) to Div -> QuantAvgPool2d -> Mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "AveragePool":
                mul_node = model.find_direct_successors(n)
                if mul_node is not None and len(mul_node) == 1 and mul_node[0].op_type == "Mul":
                    mul_node = mul_node[0]
                    t_node = model.find_direct_successors(mul_node)
                    if t_node is not None and len(t_node) == 1 and t_node[0].op_type == "Trunc":
                        t_node = t_node[0]
                        running_node_index = node_ind
                        k_s = get_by_name(n.attribute, "kernel_shape")
                        if k_s is None or len(k_s.ints) != 2 or len(set(k_s.ints)) != 1:
                            raise _refused(
                                type(self).__name__,
                                "avgpool-kernel",
                                n,
                                "FINN only supports average pooling with " "2D square kernels.",
                            )
                        k_s = k_s.ints[0]

                        pads = get_by_name(n.attribute, "pads")
                        if pads is None or len(set(pads.ints)) != 1 or pads.ints[0] != 0:
                            raise _refused(
                                type(self).__name__,
                                "avgpool-padding",
                                n,
                                "FINN dosn't support padding for average pooling.",
                            )

                        stride = get_by_name(n.attribute, "strides")
                        if stride is None or len(stride.ints) != 2 or len(set(stride.ints)) != 1:
                            raise _refused(
                                type(self).__name__,
                                "avgpool-stride",
                                n,
                                "FINN only supports 2D strides with equal values in "
                                "each direction.",
                            )
                        stride = stride.ints[0]

                        mul_val = model.get_initializer(mul_node.input[1])
                        if mul_val is None or len(mul_val.shape) != 0 or mul_val != k_s * k_s:
                            raise _refused(
                                type(self).__name__,
                                "avgpool-scale",
                                mul_node,
                                f"The Mul node after the AveragePool node must have "
                                f"static initialization at the second input, "
                                f"further the initialization must be of zero dimension "
                                f"and the value of the initialization must be "
                                f"the quadratic value of the kernel size, "
                                f"in this case {k_s * k_s}.",
                            )

                        rounding_mode = get_by_name(t_node.attribute, "rounding_mode")
                        normalized_mode_string = rounding_mode.s.upper()
                        if rounding_mode is None or normalized_mode_string != b"FLOOR":
                            raise _refused(
                                type(self).__name__,
                                "trunc-rounding-mode",
                                t_node,
                                "The Trunc node must have the rounding_mode " "set to 'FLOOR'.",
                            )
                        for inp in t_node.input[1:]:
                            if model.get_initializer(inp) is None:
                                raise _refused(
                                    type(self).__name__,
                                    "trunc-input-dynamic",
                                    t_node,
                                    f"All inputs of the Trunc node, "
                                    f"except the first, must be statically "
                                    f"initialized. However, {inp} is not.",
                                )
                        zero_pt = model.get_initializer(t_node.input[2])
                        if len(zero_pt.shape) != 0 or zero_pt != 0:
                            raise _refused(
                                type(self).__name__,
                                "trunc-zero-point",
                                t_node,
                                f"Finn only supports 0 as the zero point for "
                                f"the Trunc node, it currently is {zero_pt}.",
                            )
                        trunc_in_bits = model.get_initializer(t_node.input[3]).flatten()
                        trunc_out_bits = model.get_initializer(t_node.input[4]).flatten()
                        if len(trunc_in_bits.shape) != 1 or len(trunc_out_bits.shape) != 1:
                            raise _refused(
                                type(self).__name__,
                                "trunc-bitwidth",
                                t_node,
                                f"Finn only supports scalar bit widths "
                                f"for the Trunc node. The input bit width "
                                f"currently is: {trunc_in_bits}, "
                                f"while the output bit width is: {trunc_out_bits}.",
                            )
                        trunc_in_bits = int(trunc_in_bits[0])
                        trunc_out_bits = int(trunc_out_bits[0])

                        # The input width whose sum over the window the Trunc narrows.
                        ibits = math.floor(math.log(2**trunc_in_bits / (k_s * k_s), 2))
                        signed = _get_signed_from_upstream(model, t_node)
                        data_layout = "NCHW"

                        scale = model.get_initializer(t_node.input[1])
                        scale_div_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(scale_div_tensor)
                        model.set_initializer(scale_div_tensor.name, scale)

                        act_scale_div_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(act_scale_div_tensor)

                        scale_div_node = helper.make_node(
                            "Div",
                            [n.input[0], scale_div_tensor.name],
                            [act_scale_div_tensor.name],
                        )
                        graph.node.insert(running_node_index, scale_div_node)
                        running_node_index += 1

                        act_scale_mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(act_scale_mul_tensor)

                        QuantAvgPool2d_node = helper.make_node(
                            "QuantAvgPool2d",
                            [act_scale_div_tensor.name],
                            [act_scale_mul_tensor.name],
                            domain="qonnx.custom_op.general",
                            stride=stride,
                            kernel=k_s,
                            ibits=ibits,
                            obits=trunc_out_bits,
                            signed=int(signed),
                            data_layout=data_layout,
                        )
                        graph.node.insert(running_node_index, QuantAvgPool2d_node)
                        running_node_index += 1

                        scale_mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(scale_mul_tensor)
                        model.set_initializer(scale_mul_tensor.name, scale)

                        scale_mul_node = helper.make_node(
                            "Mul",
                            [act_scale_mul_tensor.name, scale_mul_tensor.name],
                            [t_node.output[0]],
                        )
                        graph.node.insert(running_node_index, scale_mul_node)
                        running_node_index += 1

                        graph.node.remove(n)
                        graph.node.remove(mul_node)
                        graph.node.remove(t_node)

                        model = model.transform(InferShapes())
                        model = model.transform(InferDataTypes())

                        return model, True

        return model, False


class AvgPoolAndTruncv2ToQuantAvgPool(Transformation):
    """Converts AveragePool -> Trunc (v2) to Div -> QuantAvgPool2d -> Mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for node in graph.node:
            node_ind += 1
            if node.op_type == "AveragePool":
                t_node = model.find_direct_successors(node)
                if t_node is not None and len(t_node) == 1 and t_node[0].op_type == "Trunc":
                    t_node = t_node[0]
                    running_node_index = node_ind
                    k_s = get_by_name(node.attribute, "kernel_shape")
                    if k_s is None or len(k_s.ints) != 2 or len(set(k_s.ints)) != 1:
                        raise _refused(
                            type(self).__name__,
                            "avgpool-kernel",
                            node,
                            "FINN only supports average pooling with " "2D square kernels.",
                        )
                    k_s = k_s.ints[0]

                    pads = get_by_name(node.attribute, "pads")
                    if pads is None or len(set(pads.ints)) != 1 or pads.ints[0] != 0:
                        raise _refused(
                            type(self).__name__,
                            "avgpool-padding",
                            node,
                            "FINN dosn't support padding for average pooling.",
                        )

                    stride = get_by_name(node.attribute, "strides")
                    if stride is None or len(stride.ints) != 2 or len(set(stride.ints)) != 1:
                        raise _refused(
                            type(self).__name__,
                            "avgpool-stride",
                            node,
                            "FINN only supports 2D strides with equal values in " "each direction.",
                        )
                    stride = stride.ints[0]

                    rounding_mode = get_by_name(t_node.attribute, "rounding_mode")
                    normalized_mode_string = rounding_mode.s.upper()
                    if rounding_mode is None or normalized_mode_string != b"FLOOR":
                        raise _refused(
                            type(self).__name__,
                            "trunc-rounding-mode",
                            t_node,
                            "The Trunc node must have the rounding_mode " "set to 'FLOOR'.",
                        )
                    for inp in t_node.input[1:]:
                        if model.get_initializer(inp) is None:
                            raise _refused(
                                type(self).__name__,
                                "trunc-input-dynamic",
                                t_node,
                                f"All inputs of the Trunc node, "
                                f"except the first, must be statically "
                                f"initialized. However, {inp} is not.",
                            )
                    zero_pt = model.get_initializer(t_node.input[2])
                    if len(zero_pt.shape) != 0 or zero_pt != 0:
                        raise _refused(
                            type(self).__name__,
                            "trunc-zero-point",
                            t_node,
                            f"Finn only supports 0 as the zero point for "
                            f"the Trunc node, it currently is {zero_pt}.",
                        )
                    scale = model.get_initializer(t_node.input[1]).flatten()
                    out_scale = model.get_initializer(t_node.input[4]).flatten()

                    trunc_in_bits = model.get_initializer(t_node.input[3]).flatten()
                    trunc_out_bits = model.get_initializer(t_node.input[5]).flatten()
                    if len(trunc_in_bits.shape) != 1 or len(trunc_out_bits.shape) != 1:
                        raise _refused(
                            type(self).__name__,
                            "trunc-bitwidth",
                            t_node,
                            f"Finn only supports scalar bit widths "
                            f"for the Trunc node. The input bit width "
                            f"currently is: {trunc_in_bits}, "
                            f"while the output bit width is: {trunc_out_bits}.",
                        )
                    trunc_in_bits = int(trunc_in_bits[0])
                    trunc_out_bits = int(trunc_out_bits[0])
                    if np.round(np.log2(out_scale / scale) != trunc_in_bits - trunc_out_bits):
                        raise _refused(
                            type(self).__name__,
                            "trunc-scale-bits",
                            t_node,
                            f"The scale values for the Trunc node are not "
                            f"compatible with the specified bit widths. "
                            f"Input scale: {scale}, output scale: {out_scale}, "
                            f"input bits: {trunc_in_bits}, output bits: {trunc_out_bits}.",
                        )

                    # The input width whose sum over the window the Trunc narrows.
                    ibits = math.floor(math.log(2**trunc_in_bits / (k_s * k_s), 2))
                    signed = _get_signed_from_upstream(model, t_node)
                    data_layout = "NCHW"

                    scale = model.get_initializer(t_node.input[1])
                    # Trunc v2's scale is per element of the window.
                    scale = (scale * k_s * k_s).astype(scale.dtype)
                    scale_div_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(scale_div_tensor)
                    model.set_initializer(scale_div_tensor.name, scale)

                    act_scale_div_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(act_scale_div_tensor)

                    scale_div_node = helper.make_node(
                        "Div",
                        [node.input[0], scale_div_tensor.name],
                        [act_scale_div_tensor.name],
                    )
                    graph.node.insert(running_node_index, scale_div_node)
                    running_node_index += 1

                    act_scale_mul_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(act_scale_mul_tensor)
                    QuantAvgPool2d_node = helper.make_node(
                        "QuantAvgPool2d",
                        [act_scale_div_tensor.name],
                        [act_scale_mul_tensor.name],
                        domain="qonnx.custom_op.general",
                        stride=stride,
                        kernel=k_s,
                        ibits=ibits,
                        obits=trunc_out_bits,
                        signed=int(signed),
                        data_layout=data_layout,
                    )
                    graph.node.insert(running_node_index, QuantAvgPool2d_node)
                    running_node_index += 1

                    scale_mul_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(scale_mul_tensor)
                    model.set_initializer(scale_mul_tensor.name, out_scale)

                    scale_mul_node = helper.make_node(
                        "Mul",
                        [act_scale_mul_tensor.name, scale_mul_tensor.name],
                        [t_node.output[0]],
                    )
                    graph.node.insert(running_node_index, scale_mul_node)
                    running_node_index += 1

                    graph.node.remove(node)
                    graph.node.remove(t_node)

                    model = model.transform(InferShapes())
                    model = model.transform(InferDataTypes())

                    return model, True

        return model, False
