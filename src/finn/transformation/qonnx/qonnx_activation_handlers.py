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
"""QONNX activation handlers: each replaces a quantized activation (a Quant or
BipolarQuant, after its activation node) by a MultiThreshold, followed by an Add and a
Mul where its bias and scale are not scalar. The thresholds, biases and scales are those
Brevitas's FINN export computes (``brevitas.export.onnx.finn.handler.act``)."""

import numpy as np
from abc import ABC, abstractmethod
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import int_type_holding

from finn.core.reporting.findings import Finding, FindingKind, Refusal

INT_QUANT_OP_TYPES = ("IntQuant", "Quant")
"""The integer quantizer's op types: qonnx registers IntQuant, and Quant as its other
name (Brevitas exports Quant). The conversion treats the two alike."""


def refused(code, node, message):
    """ConvertQuantActToMultiThreshold's refusal of the Quant ``node``."""
    details = (("node", node.name),)
    return Refusal(
        [Finding(FindingKind.REJECTION, code, "ConvertQuantActToMultiThreshold", message, details)]
    )


np_default_dtype = np.float32


def computed_datatype(out_bias, out_scale, steps):
    """The datatype of what a MultiThreshold with ``steps`` thresholds per channel
    computes, ``out_bias + out_scale * [0, steps]``: BIPOLAR for -1 and 1 alone, the
    smallest integer type holding an integer range, else FLOAT32."""
    ends = sorted((out_bias, out_bias + out_scale * steps))
    if steps == 1 and ends == [-1, 1]:
        return "BIPOLAR"
    if any(int(end) != end for end in ends):
        return "FLOAT32"
    return int_type_holding(int(ends[0]), int(ends[1])).name


class QuantActBaseHandler(ABC):
    """Converts a quantized activation of ``model``, its Quant node ``quant_node`` at
    ``quant_node_index``, from the QONNX dialect to FINN's."""

    def __init__(self, model: ModelWrapper, quant_node, quant_node_index: int):
        super().__init__()
        self._model = model
        self._q_node = quant_node
        self._q_index = quant_node_index

    @classmethod
    def valid_predecessor_op_types(self):
        """Defines which op types the preceding node is allowed to have for
        this type of activation.
        """
        raise NotImplementedError()

    @abstractmethod
    def _check_compatibility(self):
        """Refuse the QONNX settings FINN does not support."""
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_bias(self):
        """The activation bias, an Add behind the MultiThreshold."""
        raise NotImplementedError()

    @abstractmethod
    def _calculate_thresholds(self):
        """The MultiThreshold's thresholds."""
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_scale(self):
        """The activation scale, a Mul behind the bias's Add."""
        raise NotImplementedError()

    @abstractmethod
    def _remove_activation_node(self, multi_threshold_node):
        """Remove the activation node in front of the Quant node."""
        raise NotImplementedError()

    def _get_channel_axis(self, tensor_name):
        """The channel axis of ``tensor_name`` the Quant's scale applies along: its one
        non-unit axis, the scale's shape right-aligned against the tensor's as numpy
        broadcasts it; None for a per-tensor scale. Refuses a scale over several axes."""
        scale = np.asarray(self._model.get_initializer(self._q_node.input[1]))
        tensor_shape = self._model.get_tensor_shape(tensor_name)
        ndim = len(tensor_shape)
        # A scale of shape (C,) aligns with the last axis; (1, C, 1, 1) with axis 1.
        scale_shape = tuple(scale.shape)
        padded = (1,) * (ndim - len(scale_shape)) + scale_shape
        nonunit_axes = [axis for axis, dim in enumerate(padded) if dim != 1]
        if len(nonunit_axes) == 0:
            return None
        if len(nonunit_axes) > 1:
            raise refused(
                "quant-scale-multi-axis",
                self._q_node,
                f"Quant node {self._q_node.name} has a multi-axis scale of shape "
                f"{scale_shape}; only per-tensor or per-channel scales are supported.",
            )
        return nonunit_axes[0]

    def _channel_axis_to_data_layout(self, channel_axis, ndim):
        """The MultiThreshold's data layout for a channel axis, which must be axis 1
        (channels first) or the last (channels last)."""
        channels_first = {2: "NC", 3: "NCW", 4: "NCHW"}
        channels_last = {2: "NC", 3: "NWC", 4: "NHWC"}
        if ndim not in channels_first:
            raise refused(
                "multithreshold-rank",
                self._q_node,
                f"Unsupported tensor rank {ndim} for MultiThreshold data layout.",
            )
        if channel_axis == ndim - 1:
            return channels_last[ndim]
        if channel_axis == 1:
            return channels_first[ndim]
        raise refused(
            "multithreshold-channel-axis",
            self._q_node,
            f"Channel axis {channel_axis} (derived from the scale shape) is neither "
            f"axis 1 nor the last axis; MultiThreshold cannot represent this layout.",
        )

    def calculate_node_parameters(self):
        """The MultiThreshold's thresholds, and the bias and scale behind it."""
        return {
            "thresholds": self._calculate_thresholds(),
            "adder_bias": self._calculate_act_bias(),
            "mul_scale": self._calculate_act_scale(),
        }

    def replace_quant_node(self):
        """Replace the given QONNX style activation with a FINN style one."""

        self._check_compatibility()

        model = self._model
        graph = model.graph
        n = self._q_node
        running_node_index = self._q_index

        parameter_dict = self.calculate_node_parameters()
        thresholds = parameter_dict["thresholds"]
        steps = thresholds.shape[-1]
        adder_bias = parameter_dict["adder_bias"]
        mul_scale = parameter_dict["mul_scale"]

        thresh_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            thresholds.shape,
        )
        graph.value_info.append(thresh_tensor)
        model.set_initializer(thresh_tensor.name, thresholds)

        # A per-tensor scale implies no channel axis: the input's layout, if annotated.
        channel_axis = self._get_channel_axis(n.input[0])
        if channel_axis is not None:
            ndim = len(model.get_tensor_shape(n.input[0]))
            data_layout = self._channel_axis_to_data_layout(channel_axis, ndim)
        else:
            annotated = model.get_tensor_layout(n.input[0])
            data_layout = "".join(annotated) if annotated is not None else None

        outp_trans_node = helper.make_node(
            "MultiThreshold",
            [n.input[0], thresh_tensor.name],
            [n.output[0]],
            out_dtype="FLOAT32",
            domain="qonnx.custom_op.general",
        )
        graph.node.insert(running_node_index, outp_trans_node)
        running_node_index += 1

        mt_node = graph.node[running_node_index - 1]
        mt_inst = getCustomOp(mt_node)

        if data_layout is not None:
            mt_inst.set_nodeattr("data_layout", data_layout)

        # A BipolarQuant's scalar scale and bias are the MultiThreshold's attributes;
        # otherwise an Add and a Mul behind it. Either way the MultiThreshold's datatype
        # is that of what it computes itself; the Quant's is the last node's.
        bias_scalar = adder_bias.shape == (1,) or len(adder_bias.shape) == 0
        scale_scalar = mul_scale.shape == (1,) or len(mul_scale.shape) == 0
        if scale_scalar and bias_scalar and self._q_node.op_type == "BipolarQuant":
            mul_scale = np.atleast_1d(mul_scale)
            adder_bias = np.atleast_1d(adder_bias)

            # ONNX attributes take Python floats, not numpy's.
            out_scale = mul_scale[0].item()
            # The MultiThreshold scales, then biases: the other order from Brevitas's.
            out_bias = adder_bias[0].item() * out_scale
            mt_inst.set_nodeattr("out_scale", out_scale)
            mt_inst.set_nodeattr("out_bias", out_bias)
            mt_inst.set_nodeattr("out_dtype", computed_datatype(out_bias, out_scale, steps))
        else:
            mt_inst.set_nodeattr("out_dtype", computed_datatype(0, 1, steps))

            up_stream_node = mt_node

            zero_bias = False
            if bias_scalar:
                adder_bias = np.atleast_1d(adder_bias)
                adder_bias = adder_bias[0]
                add_shape = tuple()
                if adder_bias == 0.0:
                    zero_bias = True
            else:
                add_shape = adder_bias.shape

            if not zero_bias:
                add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    add_shape,
                )
                graph.value_info.append(add_tensor)
                model.set_initializer(add_tensor.name, adder_bias)

                output_shape = model.get_tensor_shape(n.output[0])
                act_add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    output_shape,
                )
                graph.value_info.append(act_add_tensor)

                add_node = helper.make_node(
                    "Add",
                    [act_add_tensor.name, add_tensor.name],
                    [n.output[0]],
                )
                graph.node.insert(running_node_index, add_node)
                running_node_index += 1
                add_node = graph.node[running_node_index - 1]

                up_stream_node.output[0] = act_add_tensor.name
                up_stream_node = add_node

            unity_scale = False
            if scale_scalar:
                mul_scale = np.atleast_1d(mul_scale)
                mul_scale = mul_scale[0]
                mul_shape = tuple()
                if mul_scale == 1.0:
                    unity_scale = True
            else:
                mul_shape = mul_scale.shape

            if not unity_scale:
                mul_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    mul_shape,
                )
                graph.value_info.append(mul_tensor)
                model.set_initializer(mul_tensor.name, mul_scale)

                output_shape = model.get_tensor_shape(n.output[0])
                act_mul_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    output_shape,
                )
                graph.value_info.append(act_mul_tensor)

                mul_node = helper.make_node(
                    "Mul",
                    [act_mul_tensor.name, mul_tensor.name],
                    [n.output[0]],
                )
                graph.node.insert(running_node_index, mul_node)
                running_node_index += 1
                mul_node = graph.node[running_node_index - 1]

                up_stream_node.output[0] = act_mul_tensor.name
                up_stream_node = mul_node

        self._remove_activation_node(mt_node)

        graph.node.remove(n)

        return self._model


class QuantReluHandler(QuantActBaseHandler):
    """Converts a quantized Relu or Selu."""

    @classmethod
    def valid_predecessor_op_types(self):
        return [
            "Relu",
            "Selu",
        ]

    def _check_compatibility(self):
        if self._q_node.op_type in INT_QUANT_OP_TYPES:
            q_inst = getCustomOp(self._q_node)
            narrow = q_inst.get_nodeattr("narrow")
            signed = q_inst.get_nodeattr("signed")
            if not self._model.get_initializer(self._q_node.input[2]) == 0:
                raise refused(
                    "quant-zero-point",
                    self._q_node,
                    "Only Quant nodes with zero-point == 0 "
                    "are currently supported for ReLu activations.",
                )
            act_node = self._model.find_direct_predecessors(self._q_node)
            act_node = act_node[0]
            if act_node.op_type == "Relu":
                if signed or narrow:
                    raise refused(
                        "relu-quant-signed",
                        self._q_node,
                        "FINN only supports unsigned and non-narrow Quant nodes "
                        "for Relu activations.",
                    )
        elif self._q_node.op_type == "BipolarQuant":
            return
        else:
            raise RuntimeError("Got an unexpected quantizer node type")

    def _calculate_act_bias(self):
        act_node = self._model.find_direct_predecessors(self._q_node)
        act_node = act_node[0]
        if act_node.op_type == "Relu":
            bias = np.array([0.0], dtype=np_default_dtype)
        elif act_node.op_type == "Selu":
            q_inst = getCustomOp(self._q_node)
            if self._q_node.op_type in INT_QUANT_OP_TYPES:
                bit_width = self._model.get_initializer(self._q_node.input[3])
                narrow = q_inst.get_nodeattr("narrow")
            elif self._q_node.op_type == "BipolarQuant":
                bit_width = 1.0
            else:
                raise RuntimeError("Got an unexpected quantizer node type")
            if bit_width == 1.0:
                bias = np.array([-0.5], dtype=np_default_dtype)
            else:
                if narrow:
                    min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
                else:
                    min_non_scaled_val = -(2 ** (bit_width - 1))
                bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
        return bias

    def _calculate_thresholds(self):
        if self._q_node.op_type in INT_QUANT_OP_TYPES:
            bit_width = self._model.get_initializer(self._q_node.input[3])
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")
        quant_scale = self._model.get_initializer(self._q_node.input[1]).astype(np.float32)
        act_node = self._model.find_direct_predecessors(self._q_node)
        act_node = act_node[0]
        if act_node.op_type == "Relu":
            num_distinct_values = 2**bit_width
            num_thresholds = int(num_distinct_values - 1)
            # In float64, narrowed once at the end (QuantIdentityHandler says why).
            flat_scale = quant_scale.flatten().astype(np.float64)
            num_scale_channels = flat_scale.shape[0]
            step = np.abs(flat_scale)
            min_threshold = step / 2
            thresholds = np.empty((num_scale_channels, num_thresholds), dtype=np.float64)
            for c in range(num_scale_channels):
                for t in range(num_thresholds):
                    thresholds[c][t] = min_threshold[c] + step[c] * t

        elif act_node.op_type == "Selu":
            q_inst = getCustomOp(self._q_node)
            narrow = q_inst.get_nodeattr("narrow")
            if narrow:
                num_distinct_values = 2**bit_width - 1
            else:
                num_distinct_values = 2**bit_width

            num_thresholds = int(num_distinct_values - 1)
            flat_scale = quant_scale.flatten().astype(np.float64)
            num_scale_channels = flat_scale.shape[0]
            scale = np.abs(flat_scale)
            half_scale = scale / 2
            # SELU's alpha and lambda (scale), as torch.nn.SELU defines them.
            alpha = 1.6732632423543772848170429916717
            selu_scale = 1.0507009873554804934193349852946
            thresholds = np.empty((num_scale_channels, num_thresholds), dtype=np.float64)
            for c in range(num_scale_channels):
                for t in range(num_thresholds):
                    step = -1.0 + half_scale[c] + scale[c] * t
                    if step <= 0:
                        thresholds[c][t] = np.log(step / (alpha * selu_scale) + 1)
                    else:
                        thresholds[c][t] = step / selu_scale

        thresholds = thresholds.astype(np_default_dtype)

        channel_axis = self._get_channel_axis(self._q_node.input[0])
        if channel_axis is None:
            assert thresholds.shape[0] == 1, (
                "Quant node cannot be converted to MultiThreshold: per-tensor "
                "scale must yield a single (global) threshold row."
            )
        else:
            num_output_channels = self._model.get_tensor_shape(self._q_node.output[0])[channel_axis]
            assert thresholds.shape[0] == 1 or thresholds.shape[0] == num_output_channels, (
                "Quant node cannot be converted to MultiThreshold because only "
                "per tensor or per channel quantization is supported."
            )

        return thresholds

    def _calculate_act_scale(self):
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        scale = quant_scale
        return scale

    def _remove_activation_node(self, multi_threshold_node):
        act_node = self._model.find_direct_predecessors(self._q_node)
        if act_node is None:
            raise RuntimeError(
                "For handling of Relu activations a predecessor to " "the Quant node must exist."
            )
        act_node = act_node[0]
        if act_node.op_type not in self.valid_predecessor_op_types():
            raise RuntimeError(
                "The predecessor of the Quant node must be Relu or Selu for handling "
                "of activations."
            )

        multi_threshold_node.input[0] = act_node.input[0]

        self._model.graph.node.remove(act_node)


class QuantIdentityHandler(QuantActBaseHandler):
    """Converts a quantized identity, which a quantized HardTanh is too."""

    @classmethod
    def valid_predecessor_op_types(self):
        return [
            "BatchNormalization",
            "Sub",
            "Add",
            "Mul",
            "Div",
            "DebugMarker",
            None,
        ]

    def _check_compatibility(self):
        if self._q_node.op_type in INT_QUANT_OP_TYPES:
            if not self._model.get_initializer(self._q_node.input[2]) == 0:
                raise refused(
                    "quant-zero-point",
                    self._q_node,
                    "Only Quant nodes with zero-point == 0 "
                    "are currently supported for identity activations.",
                )
        elif self._q_node.op_type == "BipolarQuant":
            quant_scale = self._model.get_initializer(self._q_node.input[1])
            if (quant_scale.flatten().shape[0] != 1) or quant_scale.flatten()[0] != 1.0:
                raise refused(
                    "bipolar-identity-scale",
                    self._q_node,
                    "FINN only supports Bipolar identity activations "
                    "with out per channel scaling and the scaling must be 1. ",
                )
        else:
            raise RuntimeError("Got an unexpected quantizer node type")

    def _calculate_act_bias(self):
        q_inst = getCustomOp(self._q_node)
        if self._q_node.op_type in INT_QUANT_OP_TYPES:
            bit_width = self._model.get_initializer(self._q_node.input[3])
            narrow = q_inst.get_nodeattr("narrow")
            signed = q_inst.get_nodeattr("signed")
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")
        if bit_width == 1.0:
            bias = np.array([-0.5], dtype=np_default_dtype)
        else:
            if not signed:
                min_non_scaled_val = 0
            else:
                if narrow:
                    min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
                else:
                    min_non_scaled_val = -(2 ** (bit_width - 1))
            bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
        return bias

    def _calculate_thresholds(self):
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        q_inst = getCustomOp(self._q_node)
        if self._q_node.op_type in INT_QUANT_OP_TYPES:
            bit_width = self._model.get_initializer(self._q_node.input[3])
            narrow = q_inst.get_nodeattr("narrow")
            signed = q_inst.get_nodeattr("signed")
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")

        if bit_width == 1.0:
            thresholds = np.empty([1, 1], dtype=np_default_dtype)
            thresholds[0] = 0
            return thresholds
        else:
            if narrow:
                num_distinct_values = 2**bit_width - 1
            else:
                num_distinct_values = 2**bit_width

            num_thresholds = int(num_distinct_values - 1)
            # In float64: a threshold is a large base plus a large multiple of the step,
            # and float32's rounding error at that magnitude can move a threshold below an
            # input Quant rounds down, a one-step mismatch.
            flat_scale = quant_scale.flatten().astype(np.float64)
            num_scale_channels = flat_scale.shape[0]
            step = np.abs(flat_scale)
            half_step = step / 2.0
            thresholds = np.empty((num_scale_channels, num_thresholds), dtype=np.float64)
            min_threshold = -half_step - step * ((num_thresholds // 2) - 1)
            if not narrow:
                min_threshold -= step
            if not signed:
                min_threshold = half_step
            for c in range(num_scale_channels):
                for t in range(num_thresholds):
                    thresholds[c][t] = min_threshold[c] + step[c] * t
            thresholds = thresholds.astype(np_default_dtype)

            channel_axis = self._get_channel_axis(self._q_node.input[0])
            if channel_axis is None:
                assert thresholds.shape[0] == 1, (
                    "Quant node cannot be converted to MultiThreshold: per-tensor "
                    "scale must yield a single (global) threshold row."
                )
            else:
                num_output_channels = self._model.get_tensor_shape(self._q_node.output[0])[
                    channel_axis
                ]
                assert thresholds.shape[0] == 1 or thresholds.shape[0] == num_output_channels, (
                    "Quant node cannot be converted to MultiThreshold because only "
                    "per tensor or per channel quantization is supported."
                )

            return thresholds

    def _calculate_act_scale(self):
        if self._q_node.op_type in INT_QUANT_OP_TYPES:
            bit_width = self._model.get_initializer(self._q_node.input[3])
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        if bit_width != 1:
            scale = quant_scale
        else:
            assert quant_scale.flatten().shape[0] == 1, "Unsupported BIPOLAR per channel scale"
            assert quant_scale.flatten()[0] == 1.0, "Unsupported BIPOLAR scale != 1"
            scale = quant_scale * 2
        return scale

    def _remove_activation_node(self, multi_threshold_node):
        # An identity activation has no node of its own.
        return
