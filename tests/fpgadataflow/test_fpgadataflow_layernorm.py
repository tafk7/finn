from typing import Tuple
import pytest
import torch
import onnx
import torch.nn as nn
import brevitas.nn as qnn
import finn.core.onnx_exec as oxe
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.transformation.infer_datatypes import InferDataTypes
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from qonnx.transformation.general import (
    ApplyConfig,
    GiveUniqueNodeNames,
)
import finn.transformation.streamline.absorb as absorb
import numpy as np
test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5
export_onnx_path_0 = "pytest_layernorm_dut_0.onnx"
export_onnx_path_1 = "pytest_layernorm_dut_1.onnx"


def build_layernorm_graph(
        input_datatype:str,
        weight_datatype:str,
        bias_datatype:str,
        output_datatype:str,
        epsilon:float,
        idm:tuple, # Input dimension
) -> ModelWrapper:

    # Datatypes restricted to "FLOAT16" or "FLOAT32" in current implementation
    bw = []
    for dt in [input_datatype, weight_datatype, bias_datatype, output_datatype]:
        if dt == "FLOAT16":
            bw += [16]
        elif dt == "FLOAT32":
            bw += [32]
        # else:
        #     raise ValueError(f"LayerNorm only supports FP16/FP32 inputs. Invalid input: {dt}")
    
    #(scale, zero_point, bitwidth)
    input_quant_params  = [1.0, 0.0, bw[0]]
    scale_quant_params  = [1.0, 0.0, bw[1]]
    bias_quant_params   = [1.0, 0.0, bw[2]]
    output_quant_params = [1.0, 0.0, bw[3]]

    last_dim = idm[-1]

    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, list(idm))
    outp = helper.make_tensor_value_info("gloabl_out", TensorProto.FLOAT, list(idm))

    graph = helper.make_graph(
        nodes=[], name="LayerNorm_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="LayerNorm_graph")
    model = ModelWrapper(model)

    Quant_0_out = helper.make_tensor_value_info("Quant_0_out", TensorProto.FLOAT, list(idm))
    # Create node
    Quant_0 = helper.make_node(
            'Quant',
            domain='qonnx.custom_op.general',
            inputs=[inp.name, 'quant0_scale', 'quant0_zeropt', 'quant0_bitwidth'],
            outputs=[Quant_0_out.name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name="Quant_0"
    )
    model.graph.node.append(Quant_0)
    model.graph.value_info.append(Quant_0_out)

    scale_bias_shape = [last_dim]

    Quant_LayerNorm_scale_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, scale_bias_shape)
    Quant_LayerNorm_scale = helper.make_node(
            'Quant',
            domain='qonnx.custom_op.general',
            inputs=["layernorm0_scale_param", 'layernorm_scale_quant_scale', 'layernorm_scale_quant_zeropt', 'layernorm_scale_quant_bitwidth'],
            outputs=[Quant_LayerNorm_scale_out.name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name="LayerNorm_Scale_Quant"
    )
    model.graph.node.append(Quant_LayerNorm_scale)
    model.graph.value_info.append(Quant_LayerNorm_scale_out)

    Quant_LayerNorm_bias_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, scale_bias_shape)
    Quant_LayerNorm_bias = helper.make_node(
            'Quant',
            domain='qonnx.custom_op.general',
            inputs=["layernorm0_b_param", 'layernorm_bias_quant_scale', 'layernorm_bias_quant_zeropt', 'layernorm_bias_quant_bitwidth'],
            outputs=[Quant_LayerNorm_bias_out.name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name="LayerNorm_bias_Quant"
    )
    model.graph.node.append(Quant_LayerNorm_bias)
    model.graph.value_info.append(Quant_LayerNorm_bias_out)

    LayerNorm_0_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, list(idm))
    LayerNorm_0 = helper.make_node(
        'LayerNormalization',
        inputs=[Quant_0_out.name, "layernorm0_scale_param", "layernorm0_b_param"],
        outputs=[LayerNorm_0_out.name],
        name='Layernorm_1',
    )
    model.graph.node.append(LayerNorm_0)
    model.graph.value_info.append(LayerNorm_0_out)
    epsilon0_attr = helper.make_attribute("epsilon", epsilon)
    LayerNorm_0.attribute.append(epsilon0_attr)

    ElementWiseMul_hls_0_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, list(idm))
    ElementWiseMul_hls_0 = helper.make_node(
        'ElementwiseMul_hls',
        inputs=[LayerNorm_0_out.name, Quant_LayerNorm_scale_out.name],
        outputs=[ElementWiseMul_hls_0_out.name],
        domain="finn.custom_op.fpgadataflow.hls",
        rhs_shape=scale_bias_shape,
        lhs_shape=idm,
        rhs_dtype=str(weight_datatype),
        lhs_dtype=str(input_datatype),
        out_shape=idm,
        name='ElementwiseMul_hls_0',
    )
    model.graph.node.append(ElementWiseMul_hls_0)
    model.graph.value_info.append(ElementWiseMul_hls_0_out)

    ElementWiseAdd_hls_0_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, list(idm))
    ElementWiseAdd_hls_0 = helper.make_node(
        'ElementwiseAdd_hls',
        inputs=[ElementWiseMul_hls_0_out.name, Quant_LayerNorm_bias_out.name],
        outputs=[ElementWiseAdd_hls_0_out.name],
        domain="finn.custom_op.fpgadataflow.hls",
        rhs_shape=scale_bias_shape,
        lhs_shape=idm,
        rhs_dtype=str(bias_datatype),
        lhs_dtype=str(input_datatype),
        out_shape=idm,
        name='ElementwiseAdd_hls_0',
    )
    model.graph.node.append(ElementWiseAdd_hls_0)
    model.graph.value_info.append(ElementWiseAdd_hls_0_out)

    # Create node
    Quant_1 = helper.make_node(
            'Quant',
            domain='qonnx.custom_op.general',
            inputs=[ElementWiseAdd_hls_0_out.name, 'quant1_scale', 'quant1_zeropt', 'quant1_bitwidth'],
            outputs=[outp.name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name="Quant_1"
    )
    model.graph.node.append(Quant_1)

    model.set_initializer("quant0_scale", np.asarray(input_quant_params[0], dtype=np.float32))
    model.set_initializer("quant0_zeropt", np.asarray(input_quant_params[1], dtype=np.float32))
    model.set_initializer("quant0_bitwidth", np.asarray(input_quant_params[2], dtype=np.float32))

    model.set_initializer("quant1_scale", np.asarray(output_quant_params[0], dtype=np.float32))
    model.set_initializer("quant1_zeropt", np.asarray(output_quant_params[1], dtype=np.float32))
    model.set_initializer("quant1_bitwidth", np.asarray(output_quant_params[2], dtype=np.float32))

    model.set_initializer("layernorm_bias_quant_scale", np.asarray(bias_quant_params[0], dtype=np.float32))
    model.set_initializer("layernorm_bias_quant_zeropt", np.asarray(bias_quant_params[1], dtype=np.float32))
    model.set_initializer("layernorm_bias_quant_bitwidth", np.asarray(bias_quant_params[2], dtype=np.float32))

    model.set_initializer("layernorm_scale_quant_scale", np.asarray(scale_quant_params[0], dtype=np.float32))
    model.set_initializer("layernorm_scale_quant_zeropt", np.asarray(scale_quant_params[1], dtype=np.float32))
    model.set_initializer("layernorm_scale_quant_bitwidth", np.asarray(scale_quant_params[2], dtype=np.float32))

    model.set_initializer("layernorm0_scale_param", np.zeros((last_dim), dtype=np.float32))
    model.set_initializer("layernorm0_b_param", np.zeros((last_dim), dtype=np.float32))
    model.set_initializer("layernorm0_epsilon_param", np.asarray(epsilon, dtype=np.float32))

    model.save(export_onnx_path_0)

    # Force the opset to 17 (TODO: Must be a better way to do this)
    _model = onnx.load(export_onnx_path_0)
    op = onnx.OperatorSetIdProto()
    op.version = 17
    _model_opset17 = helper.make_model(_model.graph, opset_imports=[op])    
    onnx.save(_model_opset17, export_onnx_path_0)

    return ModelWrapper(export_onnx_path_0) 


def make_single_layernorm_modelwrapper(impl_style="hls", simd=1, idt=DataType["FLOAT32"], odt=DataType["FLOAT32"], ifm_dim=(128, 384)):
    '''
    Create a single layernorm node with variable parameters.
    this is before SpecializeLayers() transformation.
    '''
    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, list(ifm_dim))
    outp = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, list(ifm_dim))

    # create and insert nodes
    new_node = helper.make_node(
        "LayerNorm",
        ["global_in"],
        ["global_out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=simd,
        ifm_dim=list(ifm_dim),
        epsilon=1e-5,
        inputDataType=idt.name,
        outputDataType=odt.name,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        [new_node],
        "layernorm_graph",
        inputs=[inp],
        outputs=[outp]
    )
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)

    model.set_tensor_datatype("global_in", idt)
    model.set_tensor_datatype("global_out", odt)

    return model

@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim", "stitched_ip"])
@pytest.mark.parametrize("simd", ["simd1", "simd2", "simd3", "simd4"])
@pytest.mark.fpgadataflow
def test_convert_to_hw_layernorm_layer(exec_mode, simd):
    '''
    This test checks that the FINN LayerNorm layer can be converted to a HW layer.
    '''
    if (exec_mode == "stitched_ip" or exec_mode == "rtlsim") and simd != "simd1":
        pytest.skip("Skipping this test to avoid long test times")
    # Create the qonnx model
    io_shape = (1, 1, 128, 384)
    # input = torch.randn(io_shape)
    
    input = gen_finn_dt_tensor(DataType["FLOAT32"], io_shape)
    input_t = {"global_in": input}

    model = make_single_layernorm_modelwrapper(impl_style="hls", simd=1, idt=DataType["FLOAT32"], odt=DataType["FLOAT32"], ifm_dim=(128, 384))

    simd = int(simd[-1])
    folding_config = {
        "Defaults": {},
        "LayerNorm_0": {
            "simd": simd,
            "preferred_impl_style": "hls"
        }
    }
    try:
        model = model.transform(ConvertQONNXtoFINN())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(to_hw.InferLayerNorm())
        model = model.transform(GiveUniqueNodeNames())
        # isolate fpga dataflow layers
        parent_model = model.transform(CreateDataflowPartition())
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node_path = getCustomOp(sdp_node).get_nodeattr("model")
        model = ModelWrapper(sdp_node_path)
        model = model.transform(ApplyConfig(folding_config))
        model = model.transform(SpecializeLayers(test_fpga_part))
        model = model.transform(GiveUniqueNodeNames())
        model.save(export_onnx_path_1)
        if exec_mode == "cppsim":
            model = model.transform(SetExecMode("cppsim"))
            model = model.transform(PrepareCppSim())
            model = model.transform(CompileCppSim())
        elif exec_mode == "rtlsim":
            model = model.transform(SetExecMode("rtlsim"))
            model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model = model.transform(HLSSynthIP())
            try:
                model = model.transform(PrepareRTLSim())
                pytest.fail("PrepareRTLSim should have failed")
            except Exception as e:
                print('expected to fail because this node do not support rtlsim')
                pass
        elif exec_mode == "stitched_ip":
            model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model = model.transform(HLSSynthIP())
            model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    except Exception as e:
        pytest.fail(f"Failed to transform the model: {str(e)}")
    print('We did it!')


@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.parametrize("simd", ["simd1", "simd2", "simd3", "simd4"])
@pytest.mark.parametrize("idt", ["FLOAT16", "FLOAT32"])
@pytest.mark.parametrize("wdt", ["FLOAT16", "FLOAT32"])
@pytest.mark.parametrize("bdt", ["FLOAT16", "FLOAT32"])
@pytest.mark.parametrize("odt", ["FLOAT16", "FLOAT32"])
@pytest.mark.parametrize("ifm_dim", [(1, 128, 384), (1, 12, 12, 128)])
@pytest.mark.fpgadataflow
def test_fpga_dataflow_layernorm(impl_style, simd, idt, wdt, bdt, odt, ifm_dim):
    '''
    This test checks that the ONNX LayerNormalization can lowered to FINN LayerNorm
    '''
    simd = int(simd[-1])
    io_shape = ifm_dim
    tolerance = 2

    epsilon = 1e-05
    
    model = build_layernorm_graph(idt, wdt, bdt, odt, epsilon, ifm_dim)
    model = model.transform(InferShapes())
    model.save(export_onnx_path_0)

    if(ifm_dim[-1] % simd != 0):
        pytest.skip(f"Skipping this test because the channel dimension is not a multiple of {simd}")

    # TODO: gen_finn_dt_tensor doesn't have FP16 support
    if idt == 'FLOAT16':
        input = np.random.randn(*io_shape).astype(np.float16)
        input = input.astype(np.float32)
    else:
        input = gen_finn_dt_tensor(idt, io_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Create reference values using the qonnx model
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    y_out = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_out, atol=tolerance), "Model output does not match expected output"

    try:
        model = model.transform(SpecializeLayers(test_fpga_part))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    except Exception as e:
        pytest.fail(f"Failed to transform the model: {str(e)}")

    # run the model
    y_hw = oxe.execute_onnx(model, input_t)[out_name]
    model.save(export_onnx_path_1)

    y_hw_flat = y_hw.flatten()
    y_ref_flat = y_ref.flatten()
    for i in range(len(y_hw_flat)):
        if np.allclose(y_hw_flat[i], y_ref_flat[i], atol=tolerance) == False:
            print(f"Index: {i}, Expected: {y_ref_flat[i]}, Got: {y_hw_flat[i]}")

    assert np.allclose(y_ref, y_hw, atol=tolerance), "Model output does not match expected output"
    print('')
