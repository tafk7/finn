#!/usr/bin/env python3

# Minimal test to verify CG thresholding implementations
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer

# Import our CG implementations
from finn.custom_op.fpgadataflow.hls.transition_thresholding_hls import CG_Thresholding_hls_Full

def make_simple_thresholding_model():
    """Create a simple test model."""
    num_channels = 4
    thresholds = np.array([
        [-2.0, 0.0, 2.0],  # Channel 0
        [-1.5, 0.5, 1.5],  # Channel 1  
        [-1.0, 1.0, 3.0],  # Channel 2
        [-0.5, 0.0, 0.5]   # Channel 3
    ], dtype=np.float32)
    
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, num_channels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, num_channels])

    # Create CG thresholding node directly
    cg_thresholding_node = helper.make_node(
        "CG_Thresholding_hls_Full",
        ["inp", "thresh"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        NumChannels=num_channels,
        numSteps=3,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="UINT4",
        ActVal=0,
        numInputVectors=[1],
        PE=2,
    )

    graph = helper.make_graph(
        nodes=[cg_thresholding_node],
        name="cg_thresholding_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[helper.make_tensor_value_info("thresh", TensorProto.FLOAT, thresholds.shape)],
    )

    model = helper.make_model(graph, producer_name="cg-thresholding-model")
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType["UINT4"])
    model.set_tensor_datatype("thresh", DataType["INT8"])
    model.set_initializer("thresh", thresholds)
    
    return model

def test_cg_instantiation():
    """Test CG implementation can be instantiated."""
    try:
        model = make_simple_thresholding_model()
        node = model.graph.node[0]
        inst = getCustomOp(node)
        print("✓ CG_Thresholding_hls_Full instantiated successfully:", type(inst).__name__)
        
        # Test template generation
        try:
            template_values = inst.get_template_values("hls_basic.cpp.j2")
            print("✓ Template values generated successfully:", len(template_values), "keys")
            return True
        except Exception as e:
            print("✗ Template generation failed:", e)
            return False
            
    except Exception as e:
        print("✗ CG instantiation test failed:", e)
        return False

if __name__ == "__main__":
    print("Testing CG Thresholding implementation...")
    success = test_cg_instantiation()
    print("Test completed:", "SUCCESS" if success else "FAILED")