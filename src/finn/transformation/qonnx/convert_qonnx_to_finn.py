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

from qonnx.transformation.base import Transformation
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps

from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import (
    AvgPoolAndTruncToQuantAvgPool,
)
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    ConvertQuantActToMultiThreshold,
    default_filter_function_generator,
)


class ConvertQONNXtoFINN(Transformation):
    """Converts the QONNX dialect to FINN's: the weights' Quant nodes folded into
    integer weights (FoldQuantWeights), the activations' lowered to MultiThresholds
    (ConvertQuantActToMultiThreshold). Raises ValueError or RuntimeError where the graph
    has no FINN form.

    :param filter_function: called with the model and each candidate Quant or
        BipolarQuant activation node; a node it returns False for stays a Quant, with a
        warning. By default, a Quant wider than 8 bits stays.
    """

    def __init__(
        self,
        filter_function=default_filter_function_generator(max_multithreshold_bit_width=8),
    ):
        super().__init__()
        self._filter_function = filter_function

    def apply(self, model):
        model = model.transform(ExtractBiasFromConv())
        model = model.transform(GemmToMatMul())
        model = model.transform(FoldTransposeIntoQuantInit())
        # FoldQuantWeights reads the datatypes.
        model = model.transform(InferDataTypes())
        model = model.transform(FoldQuantWeights())
        # The activation handlers set each MultiThreshold's data_layout from its input
        # tensor's.
        model = model.transform(InferDataLayouts())
        model = model.transform(
            ConvertQuantActToMultiThreshold(
                filter_function=self._filter_function,
            )
        )
        model = model.transform(InferDataTypes())
        # The MultiThresholds' layouts reach the tensors after them.
        model = model.transform(InferDataLayouts())
        model = model.transform(AvgPoolAndTruncToQuantAvgPool())
        model = model.transform(RemoveIdentityOps())

        return model, False
