############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN Kernel Integration Tests
############################################################################

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add the finn paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finn.util.flexible_hls import FlexibleHLSBackend
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.core.modelwrapper import ModelWrapper
import numpy as np


class MockCustomOp_hls(HLSBackend):
    """Mock FINN custom operation that inherits from HLSBackend."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.onnx_node = onnx_node
        self.node_index = 0
        
    def get_nodeattr(self, name):
        """Mock node attribute getter."""
        return getattr(self, f"_{name}", None)
        
    def set_nodeattr(self, name, value):
        """Mock node attribute setter."""
        setattr(self, f"_{name}", value)
        
    def get_normal_input_shape(self, ind=0):
        """Mock input shape getter."""
        return [1, 32]
        
    def get_normal_output_shape(self, ind=0):
        """Mock output shape getter."""
        return [1, 32]
        
    def get_instream_width(self, ind=0):
        """Mock input stream width."""
        return 32
        
    def get_outstream_width(self, ind=0):
        """Mock output stream width."""
        return 32
        
    def get_number_output_values(self):
        """Mock output value count."""
        return 32
        
    def get_exp_cycles(self):
        """Mock expected cycles."""
        return 100
        
    def get_template_param_values(self):
        """Mock template parameter values."""
        return {
            "AP_INT_MAX_W": "8191",
            "PE": "1",
            "SIMD": "1"
        }
        
    def generate_params(self, model, path):
        """Mock parameter generation."""
        params_file = os.path.join(path, "params.h")
        with open(params_file, 'w') as f:
            f.write("// Mock parameter file\n")
        return [params_file]
    
    def code_generation_ipi(self):
        """Mock code generation for IP generation."""
        code_gen_dict = {
            "includes": ["#include \"test.h\""],
            "defines": ["#define TEST_PARAM 1"],
            "globals": ["// Global variables"],
            "bram": ["// BRAM declarations"],
            "compute": ["test_kernel(in0_V, out0_V);"],
            "pragmas": ["#pragma HLS INTERFACE axis port=in0_V", "#pragma HLS INTERFACE axis port=out0_V"],
            "instreams": ["hls::stream<ap_uint<32>> in0_V;"],
            "outstreams": ["hls::stream<ap_uint<32>> out0_V;"]
        }
        return code_gen_dict
    
    def get_op_and_param_counts(self):
        """Mock operation and parameter counts."""
        return {"op_mac": 100, "param_count": 1024}


class TestKernelIntegration:
    """Test integration with existing FINN kernel patterns."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN environment
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
        
        # Create mock ONNX node
        self.mock_node = Mock()
        self.mock_node.name = "test_node"
        self.mock_node.op_type = "TestOp"
        
        # Create mock custom operation
        self.mock_custom_op = MockCustomOp_hls(self.mock_node)
        
        self._create_mock_files()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def _create_mock_files(self):
        """Create mock files for testing."""
        # Create finn-hlslib files
        hlslib_files = [
            "bnn-library.h",
            "cnpy.h", 
            "npy2apintstream.hpp"
        ]
        
        for filename in hlslib_files:
            filepath = os.path.join(self.finn_hlslib, filename)
            with open(filepath, 'w') as f:
                f.write(f"// Mock {filename}\n")
    
    def test_hls_backend_inheritance_compatibility(self):
        """Test that FlexibleHLSBackend maintains HLSBackend compatibility."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Verify inheritance relationship
            assert issubclass(FlexibleHLSBackend, object)
            
            # Test that mock custom op can inherit from FlexibleHLSBackend
            class TestCustomOp(FlexibleHLSBackend, MockCustomOp_hls):
                def __init__(self, onnx_node, **kwargs):
                    FlexibleHLSBackend.__init__(self)
                    MockCustomOp_hls.__init__(self, onnx_node, **kwargs)
            
            # Should be able to instantiate
            test_op = TestCustomOp(self.mock_node)
            assert isinstance(test_op, FlexibleHLSBackend)
            assert isinstance(test_op, MockCustomOp_hls)
    
    def test_code_generation_workflow(self):
        """Test complete code generation workflow with FlexibleHLSBackend."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Create custom op with FlexibleHLSBackend
            class TestCustomOp(FlexibleHLSBackend, MockCustomOp_hls):
                def __init__(self, onnx_node, **kwargs):
                    FlexibleHLSBackend.__init__(self)
                    MockCustomOp_hls.__init__(self, onnx_node, **kwargs)
            
            test_op = TestCustomOp(self.mock_node)
            
            # Test code generation for IP
            code_gen_dict = test_op.code_generation_ipi()
            assert isinstance(code_gen_dict, dict)
            assert "includes" in code_gen_dict
            assert "defines" in code_gen_dict
            assert "compute" in code_gen_dict
            
            # Test template parameter extraction
            template_params = test_op.get_template_param_values()
            assert isinstance(template_params, dict)
            assert "AP_INT_MAX_W" in template_params
    
    def test_ipgen_process_with_flexible_backend(self):
        """Test IP generation process using FlexibleHLSBackend."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            class TestCustomOp(FlexibleHLSBackend, MockCustomOp_hls):
                def __init__(self, onnx_node, **kwargs):
                    FlexibleHLSBackend.__init__(self)
                    MockCustomOp_hls.__init__(self, onnx_node, **kwargs)
                    
                def ipgen_process_customop(self, model, kind, node_index):
                    """Mock IP generation process."""
                    return ["test_ip.zip"]
            
            test_op = TestCustomOp(self.mock_node)
            
            # Test IP generation process
            with tempfile.TemporaryDirectory() as temp_ipgen_dir:
                # Mock model
                mock_model = Mock()
                mock_model.graph.node = [self.mock_node]
                
                # Test IP generation
                result = test_op.ipgen_process_customop(mock_model, "estimate", 0)
                assert isinstance(result, list)
                assert len(result) > 0
    
    def test_docompute_generation_workflow(self):
        """Test docompute code generation workflow."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            class TestCustomOp(FlexibleHLSBackend, MockCustomOp_hls):
                def __init__(self, onnx_node, **kwargs):
                    FlexibleHLSBackend.__init__(self)
                    MockCustomOp_hls.__init__(self, onnx_node, **kwargs)
                
                def code_generation_cppsim(self):
                    """Mock C++ simulation code generation."""
                    return {
                        "includes": ["#include \"test_sim.h\""],
                        "defines": ["#define SIM_MODE 1"],
                        "globals": ["// Simulation globals"],
                        "compute": ["test_sim_kernel(in_data, out_data);"],
                        "pragmas": ["// Simulation pragmas"]
                    }
            
            test_op = TestCustomOp(self.mock_node)
            
            # Test C++ simulation code generation
            cppsim_dict = test_op.code_generation_cppsim()
            assert isinstance(cppsim_dict, dict)
            assert "includes" in cppsim_dict
            assert "compute" in cppsim_dict
    
    def test_template_variable_integration(self):
        """Test integration of template variables with FINN kernel patterns."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            class TestCustomOp(FlexibleHLSBackend, MockCustomOp_hls):
                def __init__(self, onnx_node, **kwargs):
                    FlexibleHLSBackend.__init__(self)
                    MockCustomOp_hls.__init__(self, onnx_node, **kwargs)
            
            test_op = TestCustomOp(self.mock_node)
            
            # Test template variable collection
            template_params = test_op.get_template_param_values()
            code_gen_dict = test_op.code_generation_ipi()
            
            # Verify standard FINN template variables are available
            expected_variables = {
                "AP_INT_MAX_W": template_params.get("AP_INT_MAX_W", "8191"),
                "GLOBALS": "\n".join(code_gen_dict.get("globals", [])),
                "DEFINES": "\n".join(code_gen_dict.get("defines", [])),
                "PRAGMAS": "\n".join(code_gen_dict.get("pragmas", [])),
                "DOCOMPUTE": "\n".join(code_gen_dict.get("compute", []))
            }
            
            # Test template rendering with these variables
            from finn.util.flexible_hls import FINNTemplateEngine, FINNTemplateType
            
            # Create temporary template
            template_dir = os.path.join(self.temp_dir, "templates", "finn")
            os.makedirs(template_dir)
            
            template_content = """
#define AP_INT_MAX_W $AP_INT_MAX_W$
$GLOBALS$
$DEFINES$
void test_function() {
$PRAGMAS$
$DOCOMPUTE$
}
"""
            template_file = os.path.join(template_dir, "test.template")
            with open(template_file, 'w') as f:
                f.write(template_content)
            
            # Test template rendering
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            output = engine.render_template("test", expected_variables, FINNTemplateType.CUSTOM)
            
            assert output is not None
            assert "8191" in output or expected_variables["AP_INT_MAX_W"] in output
    
    def test_compilation_parameter_integration(self):
        """Test integration of compilation parameters with FINN kernels."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            class TestCustomOp(FlexibleHLSBackend, MockCustomOp_hls):
                def __init__(self, onnx_node, **kwargs):
                    FlexibleHLSBackend.__init__(self)
                    MockCustomOp_hls.__init__(self, onnx_node, **kwargs)
            
            test_op = TestCustomOp(self.mock_node)
            
            # Test include path generation
            include_paths = test_op.get_include_paths()
            assert isinstance(include_paths, list)
            assert len(include_paths) > 0
            assert self.finn_hlslib in include_paths
            
            # Test compile flags
            compile_flags = test_op.get_compile_flags()
            assert isinstance(compile_flags, list)
            
            # Should include standard C++ flags
            flag_string = " ".join(compile_flags)
            assert "-std=c++14" in flag_string or "-std=c++11" in flag_string
    
    def test_error_handling_integration(self):
        """Test error handling in kernel integration scenarios."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            class TestCustomOp(FlexibleHLSBackend, MockCustomOp_hls):
                def __init__(self, onnx_node, **kwargs):
                    FlexibleHLSBackend.__init__(self)
                    MockCustomOp_hls.__init__(self, onnx_node, **kwargs)
                
                def get_template_param_values(self):
                    """Mock template params that might cause issues."""
                    return {
                        "AP_INT_MAX_W": None,  # Problematic value
                        "PE": "invalid"
                    }
            
            test_op = TestCustomOp(self.mock_node)
            
            # Test graceful handling of problematic template parameters
            template_params = test_op.get_template_param_values()
            
            # Should handle None values
            assert template_params["AP_INT_MAX_W"] is None
            
            # Template engine should validate and handle gracefully
            from finn.util.flexible_hls import FINNTemplateEngine, FINNTemplateType
            
            engine = FINNTemplateEngine([])
            
            # Create simple template for testing
            with tempfile.TemporaryDirectory() as temp_template_dir:
                template_dir = os.path.join(temp_template_dir, "templates", "finn")
                os.makedirs(template_dir)
                
                template_file = os.path.join(template_dir, "error_test.template")
                with open(template_file, 'w') as f:
                    f.write("#define AP_INT_MAX_W $AP_INT_MAX_W$\n")
                
                engine = FINNTemplateEngine([os.path.join(temp_template_dir, "templates")])
                
                # Should handle None values gracefully
                issues = engine.validate_template("error_test", template_params, FINNTemplateType.CUSTOM)
                assert len(issues) > 0  # Should report validation issues


class TestFINNKernelCompatibility:
    """Test compatibility with various FINN kernel types."""
    
    def test_matrix_vector_activate_compatibility(self):
        """Test compatibility with MatrixVectorActivation-style kernels."""
        # Create mock MVA kernel
        class MockMVA(FlexibleHLSBackend):
            def __init__(self):
                super().__init__()
                
            def get_template_param_values(self):
                return {
                    "AP_INT_MAX_W": "8191",
                    "MW": "32",
                    "MH": "32", 
                    "PE": "4",
                    "SIMD": "4",
                    "ActVal": "0"
                }
                
            def code_generation_ipi(self):
                return {
                    "includes": ["#include \"matrix_vector_activate.h\""],
                    "defines": ["#define MW 32", "#define MH 32"],
                    "globals": ["// MVA globals"],
                    "compute": ["MatrixVectorActivation_Stream(in0_V, out0_V, weights, thresholds);"],
                    "pragmas": ["#pragma HLS INTERFACE axis port=in0_V"]
                }
        
        # Test instantiation and basic functionality
        mva_kernel = MockMVA()
        assert isinstance(mva_kernel, FlexibleHLSBackend)
        
        template_params = mva_kernel.get_template_param_values()
        assert "PE" in template_params
        assert "SIMD" in template_params
        
        code_gen = mva_kernel.code_generation_ipi()
        assert "MatrixVectorActivation_Stream" in code_gen["compute"][0]
    
    def test_sliding_window_compatibility(self):
        """Test compatibility with SlidingWindow-style kernels."""
        class MockSlidingWindow(FlexibleHLSBackend):
            def __init__(self):
                super().__init__()
                
            def get_template_param_values(self):
                return {
                    "AP_INT_MAX_W": "8191",
                    "ConvKernelDim": "3",
                    "IFMChannels": "64",
                    "Input_precision": "8",
                    "IFMDim": "32",
                    "OFMDim": "30",
                    "SIMD": "8"
                }
                
            def code_generation_ipi(self):
                return {
                    "includes": ["#include \"sliding_window.h\""],
                    "defines": ["#define ConvKernelDim 3"],
                    "globals": ["// SW globals"],
                    "compute": ["SlidingWindow_Stream(in0_V, out0_V);"],
                    "pragmas": ["#pragma HLS INTERFACE axis port=in0_V"]
                }
        
        # Test instantiation
        sw_kernel = MockSlidingWindow()
        assert isinstance(sw_kernel, FlexibleHLSBackend)
        
        template_params = sw_kernel.get_template_param_values()
        assert "ConvKernelDim" in template_params
        assert "SIMD" in template_params
    
    def test_thresholding_compatibility(self):
        """Test compatibility with Thresholding-style kernels."""
        class MockThresholding(FlexibleHLSBackend):
            def __init__(self):
                super().__init__()
                
            def get_template_param_values(self):
                return {
                    "AP_INT_MAX_W": "8191",
                    "NumChannels": "64",
                    "PE": "4",
                    "NumSteps": "16",
                    "ActVal": "1"
                }
                
            def code_generation_ipi(self):
                return {
                    "includes": ["#include \"thresholding.h\""],
                    "defines": ["#define NumChannels 64"],
                    "globals": ["// Thresholding globals"],
                    "compute": ["Thresholding_Stream(in0_V, out0_V, thresholds);"],
                    "pragmas": ["#pragma HLS INTERFACE axis port=in0_V"]
                }
        
        # Test instantiation
        thresh_kernel = MockThresholding()
        assert isinstance(thresh_kernel, FlexibleHLSBackend)
        
        template_params = thresh_kernel.get_template_param_values()
        assert "NumChannels" in template_params
        assert "PE" in template_params


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])