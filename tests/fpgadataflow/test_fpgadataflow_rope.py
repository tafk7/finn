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


def make_single_rope_modelwrapper(seq_len, hidden, idt, wdt, cos, sin, simd, impl_style):
    # Define the input tensor
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 1, seq_len, hidden])

    # Define the output tensor
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 1, seq_len, hidden])

    #cos_tensor = numpy_helper.from_array(cos_values, name='cos')
    #sin_tensor = numpy_helper.from_array(sin_values, name='sin')

    # Define the custom RoPE node
    rope_node = helper.make_node(
        'RotaryEmbedding',  # Custom node name
        #['input', 'cos', 'sin'],  # Inputs
        ['input', 'cos', 'sin'],
        ['output'],  # Outputs
        name='CustomRoPE',
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        HiddenDimension=hidden,
        SequenceLength=seq_len,
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
        [input_tensor],  # Inputs
        [output_tensor],  # Outputs

        initializer=[
            helper.make_tensor('cos', onnx.TensorProto.INT8, cos.shape, cos),
            helper.make_tensor('sin', onnx.TensorProto.INT8, sin.shape, sin),
        #    helper.make_tensor('sin', onnx.TensorProto.FLOAT, sin_values.shape, sin_values)
        ]  # Initializers
    )

    # Create the model
    #model = helper.make_model(graph, producer_name='custom_rope_model')
    model = qonnx_make_model(graph, producer_name="rope-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("input", idt)
    model.set_tensor_datatype("cos", wdt)
    model.set_tensor_datatype("sin", wdt)
    model.set_tensor_datatype("output", idt)

    model.set_metadata_prop("rtlsim_trace", "trace.vcd")
    os.environ["RTLSIM_TRACE_DEPTH"] = "45"

    # Save the model to a file
    onnx.save(helper.make_model(graph, producer_name='custom_rope_model'), 'rope_node.onnx')

    return model


# input image dimension
#@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of channels
@pytest.mark.parametrize("seq_len", [2])
@pytest.mark.parametrize("hidden", [128])
# Input parallelism
@pytest.mark.parametrize("simd", [1])
# FINN input datatype
@pytest.mark.parametrize("idt", [DataType["INT8"]])
@pytest.mark.parametrize("wdt", [DataType["INT8"]])
# execution mode
#@pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
# implementation style
@pytest.mark.parametrize("impl_style", ["rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_rope(seq_len, hidden, idt, wdt, simd, impl_style):
    #if num_ch % simd != 0:
    #    pytest.skip(" num_ch % simd != 0, skipping")

    #idim_h, idim_w = idim
    #pad_h = pad[0] + pad[2]
    #pad_w = pad[1] + pad[3]

    # generate input data
    #odim_h = idim_h + pad_h
    #odim_w = idim_w + pad_w

    #y_expected = np.pad(x, ((0, 0), (pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), "constant")
    #expected_oshape = (1, odim_h, odim_w, num_ch)

    # Define the cached tensors
    #cos_values = gen_finn_dt_tensor(idt, [1, 1, 1, num_ch])
    cos = np.random.randint(-7, 7, size=(1, 1, seq_len, hidden)).astype(np.int8)  # Random values
    sin = np.random.randint(-7, 7, size=(1, 1, seq_len, hidden)).astype(np.int8)  # Random values

    #sin_values = np.random.rand(32768, 64).astype(np.float32)  # Random values

    x = gen_finn_dt_tensor(idt, [1, 1, seq_len, hidden]) % 8
    #cos = gen_finn_dt_tensor(wdt, [1, 1, seq_len, hidden])
    print("x=",x)
    input_dict = {"input": x}

    midpoint = hidden // 2

    x1 = np.concatenate((-x[...,midpoint:], x[...,:midpoint]), axis=-1)

    y_expected = x * cos + x1 * sin
    #import pdb; pdb.set_trace()

    print("idt=",idt)
    print("wdt=",wdt)
    model = make_single_rope_modelwrapper(seq_len, hidden, idt, wdt, cos, sin, simd, impl_style)

    #inp = np.random.rand(1, num_ch).astype(np.float32)

    y_produced_cpu = oxe.execute_onnx(model, input_dict)["output"]
    print("output_cpu=",y_produced_cpu)
    # assert y_produced.shape == expected_oshape
    print("y_expected=", y_expected)
    assert (y_produced_cpu == y_expected).all(), "HW layer execution failed"

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(InferShapes())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.save("rope_model-before-infer-shapes.onnx")
    # if mode == "cppsim":
    #     model = model.transform(PrepareCppSim())
    #     model = model.transform(CompileCppSim())
    # elif mode == "rtlsim":
    #     model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    #     model = model.transform(HLSSynthIP())
    #     model = model.transform(PrepareRTLSim())
    #import pdb; pdb.set_trace()
    model.set_metadata_prop("exec_mode", "rtlsim")
    y_produced = oxe.execute_onnx(model, input_dict)["output"]
    print("x*cos=", x * cos)
    print("x*sin=", x * sin)
    print("y_expected=",y_expected)
    print("rtl output=",y_produced)
    # assert y_produced.shape == expected_oshape
    assert (y_produced == y_expected).all()

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
