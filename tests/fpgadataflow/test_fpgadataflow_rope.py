# Copyright (C) 2020-2022, Xilinx, Inc.
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

import pytest

import numpy as np
import os
import onnx
import numpy as np

import onnx.helper as helper
import onnx.numpy_helper as numpy_helper

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.custom_op.fpgadataflow.rotaryembedding import get_rope_onnx_filename
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.basic import pynq_part_map

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def make_single_rope_modelwrapper(seq_len, hidden, head_size, num_heads, idt, wdt, cos, sin, simd, impl_style):

    io_shape = [1, num_heads, seq_len, head_size]

    # Define the input tensor
    q = helper.make_tensor_value_info('q', onnx.TensorProto.FLOAT, io_shape)
    k = helper.make_tensor_value_info('k', onnx.TensorProto.FLOAT, io_shape)

    # Define the output tensor
    output_q = helper.make_tensor_value_info('output_q', onnx.TensorProto.FLOAT, io_shape)
    output_k = helper.make_tensor_value_info('output_k', onnx.TensorProto.FLOAT, io_shape)

    # Define the custom RoPE node
    rope_node = helper.make_node(
        'RotaryEmbedding',  # Custom node name
        ['q', 'k', 'cos', 'sin' ],
        ['output_q', 'output_k'],  # Outputs
        name='CustomRoPE',
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        HiddenDimension=hidden,
        SequenceLength=seq_len,
        HeadDimension=head_size,
        NumHeads=num_heads,
        RopeTheta=10000.0,
        inputDataType=str(idt.name),
        weightDataType=str(wdt.name),
        numInputVectors=1,
        SIMD=simd,
        preferred_impl_style=impl_style,
    )

    # Create the graph
    graph = helper.make_graph(
        [rope_node],  # Nodes
        'RopeGraph',  # Graph name
        [q, k],  # Inputs
        [output_q, output_k],  # Outputs
        initializer = [
            helper.make_tensor('cos', onnx.TensorProto.FLOAT, cos.shape, cos),
            helper.make_tensor('sin', onnx.TensorProto.FLOAT, sin.shape, sin),
        ]  # Initializer
    )

    # Create the model
    #model = helper.make_model(graph, producer_name='custom_rope_model')
    model = qonnx_make_model(graph, producer_name="rope-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("q", idt)
    model.set_tensor_datatype("k", idt)
    model.set_tensor_datatype("output_q", idt)
    model.set_tensor_datatype("output_k", idt)
    model.set_tensor_datatype("cos", wdt)
    model.set_tensor_datatype("sin", wdt)

    model.set_metadata_prop("rtlsim_trace", "trace.vcd")
    os.environ["RTLSIM_TRACE_DEPTH"] = "45"

    # Save the model to a file
    onnx.save(helper.make_model(graph, producer_name='custom_rope_model'), 'rope_node.onnx')

    return model


# input image dimension
#@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of channels
@pytest.mark.parametrize("seq_len", [2])
@pytest.mark.parametrize("hidden", [512])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("num_heads", [8])
# Input parallelism
@pytest.mark.parametrize("simd", [1])
# FINN input datatype
@pytest.mark.parametrize("idt", [DataType["FLOAT32"]])
@pytest.mark.parametrize("wdt", [DataType["FLOAT32"]])
# execution mode
#@pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
# implementation style
@pytest.mark.parametrize("impl_style", ["rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_rope(seq_len, hidden, head_size, num_heads, idt, wdt, simd, impl_style):

    q = gen_finn_dt_tensor(idt, [1, num_heads, seq_len, head_size]) % 5
    k = gen_finn_dt_tensor(idt, [1, num_heads, seq_len, head_size]) % 5

    midpoint = head_size // 2

    q1 = np.concatenate((-q[...,midpoint:], q[...,:midpoint]), axis=-1)
    k1 = np.concatenate((-k[...,midpoint:], k[...,:midpoint]), axis=-1)

    # open onnx file and retrieve weights
    onnx_filename = get_rope_onnx_filename(10000.0, 1, num_heads, seq_len, head_size)
    onnx_path     = "../../src/finn/custom_op/fpgadataflow/rotaryembedding/onnxgraphs/" + onnx_filename
    onnx_model    = onnx.load(onnx_path)

    qonnx_model = ModelWrapper(onnx_model)
    cos = np.expand_dims(qonnx_model.get_initializer("cos_param"), 0)
    sin = np.expand_dims(qonnx_model.get_initializer("sin_param"), 0)

    q_expected = q * cos + q1 * sin
    k_expected = k * cos + k1 * sin

    model = make_single_rope_modelwrapper(seq_len, hidden, head_size, num_heads, idt, wdt, cos, sin, simd, impl_style)

    input_dict = {"q": q, "k": k}
    onnx_output = oxe.execute_onnx(model, input_dict)

    assert (k_expected == onnx_output["output_k"]).all()
    assert (q_expected == onnx_output["output_q"]).all()

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(InferShapes())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))

    model.save("rope_model-before-infer-shapes.onnx")

    model.set_metadata_prop("exec_mode", "rtlsim")
    sim_output = oxe.execute_onnx(model, input_dict)


    assert (sim_output['output_k'] == k_expected).all()
    assert (sim_output['output_q'] == q_expected).all()

    # if mode == "rtlsim":
    op_type = "RotaryEmbedding_" + "rtl"
    model.save("rope_model-after-specialization.onnx")
    node = model.get_nodes_by_op_type(op_type)[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    #assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
    #assert exp_cycles != 0
