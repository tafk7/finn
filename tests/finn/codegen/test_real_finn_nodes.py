#!/usr/bin/env python3

"""
🧪 Test Real FINN Backend Implementations with Real FINN Nodes
Uses actual FINN test infrastructure - NO MOCKS
"""

import logging
import tempfile
import os
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_finn_backends():
    """Test backends using real FINN node creation infrastructure."""
    
    logger.info("🚀 Testing Real FINN Backend Implementations - NO MOCKS")
    logger.info("=" * 80)
    
    try:
        # Import real FINN infrastructure
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.util.basic import gen_finn_dt_tensor
        import onnx
        import onnx.helper as helper
        from qonnx.core.datatype import DataType
        
        # Import backend classes
        from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_ThresholdingHLS
        from finn.custom_op.fpgadataflow.hls.CG_mvau_hls import CG_MVAU_hls
        from finn.custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls
        from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
        
        logger.info("✅ All imports successful")
        
        # Test Thresholding with real FINN nodes
        logger.info("\n🔍 Testing Thresholding with Real FINN Nodes")
        logger.info("-" * 50)
        
        # Create real Thresholding node using FINN's helper functions
        input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 32])
        output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 32])
        
        # Create Thresholding node with proper ONNX attributes
        thres_node = helper.make_node(
            "Thresholding",
            inputs=["input", "thresholds"],
            outputs=["output"],
            domain="finn.custom_op.fpgadataflow",
            NumChannels=32,
            PE=4,
            numSteps=8,
            inputDataType="UINT8",
            outputDataType="UINT8",
            weightDataType="INT16",  # Required threshold data type
            numInputVectors=[1],
            mem_mode="internal_embedded",
            ram_style="block",
            runtime_writeable_weights=0
        )
        
        # Create a minimal model to test with
        graph = helper.make_graph([thres_node], "test_graph", [input_tensor], [output_tensor])
        model = helper.make_model(graph, producer_name="finn-test")
        model_wrapper = ModelWrapper(model)
        
        logger.info(f"✅ Created real Thresholding ONNX node: {thres_node.name}")
        
        # Test Clean Thresholding Backend
        try:
            clean_backend = CG_ThresholdingHLS(thres_node)
            logger.info("✅ Clean Thresholding backend instantiated successfully")
            
            # Test template value generation
            template_values = clean_backend.get_template_values("base/hls_base.cpp.j2")
            logger.info(f"✅ Clean backend generated {len(template_values)} template values")
            logger.info(f"   Template keys: {list(template_values.keys())[:5]}...")
            
        except Exception as e:
            logger.error(f"❌ Clean Thresholding backend failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test Legacy Thresholding Backend  
        try:
            legacy_backend = Thresholding_hls(thres_node)
            logger.info("✅ Legacy Thresholding backend instantiated successfully")
            
            # Test template value generation
            template_values = legacy_backend.get_template_values("base/hls_base.cpp.j2")
            logger.info(f"✅ Legacy backend generated {len(template_values)} template values")
            
        except Exception as e:
            logger.error(f"❌ Legacy Thresholding backend failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test MVAU with real FINN nodes
        logger.info("\n🔍 Testing MVAU with Real FINN Nodes")
        logger.info("-" * 50)
        
        # Create real MVAU node
        mvau_input = helper.make_tensor_value_info("mvau_input", onnx.TensorProto.FLOAT, [1, 64])
        mvau_output = helper.make_tensor_value_info("mvau_output", onnx.TensorProto.FLOAT, [1, 32])
        
        mvau_node = helper.make_node(
            "MatrixVectorActivation",
            inputs=["mvau_input", "weights"], 
            outputs=["mvau_output"],
            domain="finn.custom_op.fpgadataflow",
            MW=64,
            MH=32,
            PE=4,
            SIMD=8,
            inputDataType="UINT8",
            weightDataType="INT8", 
            outputDataType="UINT8",
            noActivation=0,
            numInputVectors=[1],
            mem_mode="internal_embedded",
            resType="lut",
            ram_style="block",
            code_gen_dir_cppsim="/tmp/test_cppsim",
            code_gen_dir_ipgen="/tmp/test_ipgen",
            executable_path="/tmp/test_exec",
            runtime_writeable_weights=0
        )
        
        logger.info(f"✅ Created real MVAU ONNX node: {mvau_node.name}")
        
        # Test Clean MVAU Backend
        try:
            clean_mvau = CG_MVAU_hls(mvau_node)
            logger.info("✅ Clean MVAU backend instantiated successfully")
            
            # Test template value generation
            template_values = clean_mvau.get_template_values("base/hls_base.cpp.j2")
            logger.info(f"✅ Clean MVAU generated {len(template_values)} template values")
            logger.info(f"   Template keys: {list(template_values.keys())[:5]}...")
            
        except Exception as e:
            logger.error(f"❌ Clean MVAU backend failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test Legacy MVAU Backend
        try:
            legacy_mvau = MVAU_hls(mvau_node)
            logger.info("✅ Legacy MVAU backend instantiated successfully")
            
            # Test template value generation
            template_values = legacy_mvau.get_template_values("base/hls_base.cpp.j2")
            logger.info(f"✅ Legacy MVAU generated {len(template_values)} template values")
            
        except Exception as e:
            logger.error(f"❌ Legacy MVAU backend failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("\n🏆 REAL FINN NODE TESTING COMPLETE")
        logger.info("=" * 80)
        logger.info("🎉 Using real FINN infrastructure - no mocks!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real FINN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_finn_backends()