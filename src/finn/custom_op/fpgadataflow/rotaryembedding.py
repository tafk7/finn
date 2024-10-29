# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class RotaryEmbedding(HWCustomOp):
    """Abstraction layer for HW impplementation of RotaryEmbedding.
    Pads input image by given amount."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # Sequence Length of the Input
            "SequenceLength": ("i", True, 0),
            # hidden dimension of the input
            "HiddenDimension": ("i", True, 0),
            # SIMD Input parallelism
            "SIMD": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            # shape describing input vecs per execution
            "numInputVectors": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        return 0

    def get_normal_input_shape(self, ind=0):
        seq_len = self.get_nodeattr("SequenceLength")
        hidden = self.get_nodeattr("HiddenDimension")
        ishape = (1, 1, seq_len, hidden)
        return ishape

    def get_normal_output_shape(self, ind=0):
        seq_len = self.get_nodeattr("SequenceLength")
        hidden = self.get_nodeattr("HiddenDimension")
        oshape = (1, 1, seq_len, hidden)
        return oshape

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        hidden = self.get_nodeattr("HiddenDimension")
        simd = self.get_nodeattr("SIMD")
        assert hidden % simd == 0, "SIMD must divide input channels"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        hidden = self.get_nodeattr("HiddenDimension")
        simd = self.get_nodeattr("SIMD")
        assert hidden % simd == 0, "SIMD must divide input channels"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for RotaryEmbedding."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        raise NotImplementedError("This Method is not implemented")

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        return ret

    def get_weight_datatype(self, ind=0):
        """Returns FINN DataType of weights."""
        ret = DataType[self.get_nodeattr("weightDataType")]
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def execute_node(self, context, graph):
        # Behavioral Model Code
        node = self.onnx_node
        # Rope Computation
        context[node.output[0]] = context[node.input[0]] * context[node.input[1]] + context[node.input[0]] * context[node.input[2]]


