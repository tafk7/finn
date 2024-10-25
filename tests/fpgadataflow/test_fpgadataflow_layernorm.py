from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from typing import Tuple
import numpy as np


def build_Layernorm_graph(
        idm:tuple, # Input dimension
        input_quant_params:tuple[int,int,int], #(scale, zero_point, bitwidth)
        scale_quant_params:tuple[int,int,int], #(scale, zero_point, bitwidth)
        bias_quant_params:tuple[int,int,int],  #(scale, zero_point, bitwidth)
        output_quant_params:tuple[int,int,int] #(scale, zero_point, bitwidth)
) -> ModelWrapper:

    last_dim = idm[-1]

    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, list(idm))
    outp = helper.make_tensor_value_info("gloabl_out", TensorProto.FLOAT, list(idm))

    graph = helper.make_graph(
        nodes=[], name="LayerNorm_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="LayerNorm_graph")
    model = ModelWrapper(model)

    Quant_0_out = helper.make_tensor_value_info("Quant_0_out", TensorProto.FLOAT, list(idm))
    Quant_0 = helper.make_node(
            'Quant',
            domain='finn.custom_op.general',
            inputs=[inp.name, 'quant0_scale', 'quant0_zeropt', 'quant0_bitwidth'],
            outputs=[Quant_0_out.name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name="Quant_0"
    )
    model.graph.node.insert(0, Quant_0)
    model.graph.value_info.append(Quant_0_out)

    Quant_LayerNorm_scale_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, [last_dim])
    Quant_LayerNorm_scale = helper.make_node(
            'Quant',
            domain='finn.custom_op.general',
            inputs=["layernorm0_scale_param", 'layernorn_scale_quant_scale', 'layernorm_scale_quant_zeropt', 'layernorm_scale_quant_bitwidth'],
            outputs=[Quant_LayerNorm_scale_out.name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name="LayerNorm_Scale_Quant"
    )
    model.graph.node.insert(3, Quant_LayerNorm_scale)
    model.graph.value_info.append(Quant_LayerNorm_scale_out)

    Quant_LayerNorm_bias_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, [last_dim])
    Quant_LayerNorm_bias = helper.make_node(
            'Quant',
            domain='finn.custom_op.general',
            inputs=["layernorm0_b_param", 'layernorn_bias_quant_scale', 'layernorm_bias_quant_zeropt', 'layernorm_bias_quant_bitwidth'],
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
        domain="ai.onnx v18",
        name='Layernorm_1',
    )
    model.graph.node.insert(1, LayerNorm_0)
    model.graph.value_info.append(LayerNorm_0_out)
    epsilon0_attr = helper.make_attribute("epsilon", 9.999999960041972e-13)
    LayerNorm_0.attribute.append(epsilon0_attr)

    ElementWiseMul_hls_0_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, list(idm))
    ElementWiseMul_hls_0 = helper.make_node(
        'ElementwiseMul_hls',
        inputs=[LayerNorm_0_out.name, Quant_LayerNorm_scale_out.name],
        outputs=[ElementWiseMul_hls_0_out.name],
        domain="finn.custom_op.fpgadataflow.hls",
        name='ElementwiseMul_hls_0',
    )
    model.graph.node.insert(4, ElementWiseMul_hls_0)
    model.graph.value_info.append(ElementWiseMul_hls_0_out)

    ElementWiseAdd_hls_0_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, list(idm))
    ElementWiseAdd_hls_0 = helper.make_node(
        'ElementwiseAdd_hls',
        inputs=[ElementWiseMul_hls_0_out.name, Quant_LayerNorm_bias_out.name],
        outputs=[ElementWiseAdd_hls_0_out.name],
        domain="finn.custom_op.fpgadataflow.hls",
        name='ElementwiseAdd_hls_0',
    )
    model.graph.node.insert(4, ElementWiseAdd_hls_0)
    model.graph.value_info.append(ElementWiseAdd_hls_0_out)

    Quant_1 = helper.make_node(
            'Quant',
            domain='finn.custom_op.general',
            inputs=[ElementWiseAdd_hls_0_out.name, 'quant1_scale', 'quant1_zeropt', 'quant1_bitwidth'],
            outputs=[outp.name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name="Quant_1"
    )
    model.graph.node.insert(2, Quant_1)


    model.set_initializer("quant0_scale", np.asarray(1, dtype=np.float32))
    model.set_initializer("quant0_zeropt", np.asarray(0, dtype=np.float32))
    model.set_initializer("quant0_bitwidth", np.asarray(8, dtype=np.float32))

    model.set_initializer("quant1_scale", np.asarray(1, dtype=np.float32))
    model.set_initializer("quant1_zeropt", np.asarray(0, dtype=np.float32))
    model.set_initializer("quant1_bitwidth", np.asarray(8, dtype=np.float32))

    model.set_initializer("layernorm_bias_quant_scale", np.asarray(1, dtype=np.float32))
    model.set_initializer("layernorm_bias_quant_zeropt", np.asarray(0, dtype=np.float32))
    model.set_initializer("layernorm_bias_quant_bitwidth", np.asarray(8, dtype=np.float32))

    model.set_initializer("layernorm_scale_quant_scale", np.asarray(1, dtype=np.float32))
    model.set_initializer("layernorm_scale_quant_zeropt", np.asarray(0, dtype=np.float32))
    model.set_initializer("layernorm_scale_quant_bitwidth", np.asarray(8, dtype=np.float32))

    model.set_initializer("layernorm0_scale_param", np.zeros((384), dtype=np.float32))
    model.set_initializer("layernorm0_b_param", np.zeros((384), dtype=np.float32))
    model.set_initializer("layernorm0_epsilon_param", np.asarray(1e-5, dtype=np.float32))
    return model

