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
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper

# Import the new architecture components
from finn.codegen.codegen import Codegen, UnsupportedTemplateError
from finn.custom_op.fpgadataflow.hls.thresholding_hls import ThresholdingHLS


class TestTemplateValueProvider:
    """Test the new Template Value Provider architecture."""
    
    def create_test_thresholding_node(self):
        """Create a test Thresholding ONNX node for testing."""
        # Create a simple thresholding node
        node = helper.make_node(
            "Thresholding_Batch",
            inputs=["input"],
            outputs=["output"],
            domain="finn.custom_op.fpgadataflow",
            name="test_thresholding"
        )
        
        # Create input/output value info
        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32])
        
        # Create graph
        graph = helper.make_graph([node], "test_graph", [input_vi], [output_vi])
        
        # Create model
        model = helper.make_model(graph)
        model_wrapper = ModelWrapper(model)
        
        # Set required attributes for Thresholding
        model_wrapper.set_node_attr(node.name, "PE", 4)
        model_wrapper.set_node_attr(node.name, "NumChannels", 32)
        model_wrapper.set_node_attr(node.name, "inputDataType", "INT8")
        model_wrapper.set_node_attr(node.name, "weightDataType", "INT8")
        model_wrapper.set_node_attr(node.name, "outputDataType", "INT8")
        model_wrapper.set_node_attr(node.name, "ActVal", 0.5)
        
        return node, model_wrapper
    
    def test_thresholding_hls_creation(self):
        """Test that ThresholdingHLS can be created successfully."""
        node, model = self.create_test_thresholding_node()
        
        # Create ThresholdingHLS instance
        thresholding_hls = ThresholdingHLS(node)
        
        # Test basic properties
        assert thresholding_hls.onnx_node.name == "test_thresholding"
        assert isinstance(thresholding_hls, ThresholdingHLS)
        assert hasattr(thresholding_hls, 'get_template_values')
        assert hasattr(thresholding_hls, 'get_supported_templates')
    
    def test_supported_templates(self):
        """Test that ThresholdingHLS reports correct supported templates."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        supported = thresholding_hls.get_supported_templates()
        
        # Check expected templates are supported
        assert "hls_thresholding_lut" in supported
        assert "hls_streaming_generic" in supported
        assert "hls_basic" in supported
        
        # Check template support detection
        assert thresholding_hls.supports_template("hls_thresholding_lut")
        assert thresholding_hls.supports_template("hls_basic")
        assert not thresholding_hls.supports_template("unsupported_template")
    
    def test_template_values_extraction(self):
        """Test the core fix - template value extraction provides appropriate values."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        # Test thresholding LUT template values - this is the core fix
        values = thresholding_hls.get_template_values("hls_thresholding_lut")
        
        # ✅ CORE FIX: These values should be provided without error
        assert 'mem_mode' in values
        assert values['mem_mode'] == 'const_embedded'  # Appropriate for Thresholding
        
        assert 'ram_style' in values
        assert values['ram_style'] == 'distributed'    # Appropriate for small LUTs
        
        assert 'simd_factor' in values
        assert values['simd_factor'] == 1              # Thresholding doesn't use SIMD
        
        assert 'pe_factor' in values
        assert values['pe_factor'] == 4                # From node attributes
        
        assert 'num_channels' in values
        assert values['num_channels'] == 32            # From node attributes
        
        assert 'parallelization_strategy' in values
        assert values['parallelization_strategy'] == 'pe_only'
    
    def test_streaming_template_values(self):
        """Test streaming template value extraction."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        values = thresholding_hls.get_template_values("hls_streaming_generic")
        
        # Check streaming-specific values
        assert 'pe_factor' in values
        assert values['pe_factor'] == 4
        
        assert 'simd_factor' in values
        assert values['simd_factor'] == 1  # Thresholding doesn't use SIMD
        
        assert 'channels_per_pe' in values
        assert values['channels_per_pe'] == 8  # 32 channels / 4 PE
    
    def test_basic_template_values(self):
        """Test basic template value extraction."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        values = thresholding_hls.get_template_values("hls_basic")
        
        # Check basic values
        assert 'mem_mode' in values
        assert values['mem_mode'] == 'const_embedded'
        
        assert 'ram_style' in values
        assert values['ram_style'] == 'auto'
        
        assert 'pe_factor' in values
        assert values['pe_factor'] == 4
    
    def test_unsupported_template_error(self):
        """Test that unsupported templates raise appropriate error."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        with pytest.raises(UnsupportedTemplateError) as exc_info:
            thresholding_hls.get_template_values("unsupported_template")
        
        assert "not supported by ThresholdingHLS" in str(exc_info.value)
    
    def test_template_priority_order(self):
        """Test HLS template priority order."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        priorities = thresholding_hls._get_template_priority_order()
        
        # Check HLS-specific priority order
        assert "hls_streaming_optimized" in priorities
        assert "hls_basic" in priorities
        assert priorities.index("hls_streaming_optimized") < priorities.index("hls_basic")
    
    def test_original_error_fixed(self):
        """Test that the original 'mem_mode' error is fixed."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        # This would previously fail with "Op has no such attribute: mem_mode"
        # Now it should work because ThresholdingHLS provides appropriate values
        try:
            values = thresholding_hls.get_template_values("hls_thresholding_lut")
            mem_mode = values.get('mem_mode')
            
            # ✅ SUCCESS: No AttributeError, and we get an appropriate value
            assert mem_mode is not None
            assert mem_mode == 'const_embedded'
            
            # Test passes - original error is fixed!
            print("✅ Original 'mem_mode' error successfully fixed!")
            
        except AttributeError as e:
            if "mem_mode" in str(e):
                pytest.fail("Original mem_mode error still present - fix failed")
            else:
                raise  # Different error, re-raise
    
    def test_inheritance_hierarchy(self):
        """Test that multiple inheritance works correctly."""
        node, model = self.create_test_thresholding_node()
        thresholding_hls = ThresholdingHLS(node)
        
        # Should have methods from both parent classes
        # From Thresholding
        assert hasattr(thresholding_hls, 'get_nodeattr')
        assert hasattr(thresholding_hls, 'calc_tmem')
        
        # From HLSBackend  
        assert hasattr(thresholding_hls, 'generate_hls_code')
        assert hasattr(thresholding_hls, '_extract_hls_parallelization_values')
        
        # From Codegen (via HLSBackend)
        assert hasattr(thresholding_hls, 'generate_code')
        assert hasattr(thresholding_hls, '_extract_common_values')


if __name__ == "__main__":
    # Run basic test to demonstrate the fix works
    test = TestTemplateValueProvider()
    
    print("🧪 Testing Template Value Provider Architecture...")
    
    test.test_thresholding_hls_creation()
    print("✅ ThresholdingHLS creation works")
    
    test.test_supported_templates()
    print("✅ Template support detection works")
    
    test.test_template_values_extraction()
    print("✅ Template value extraction works")
    
    test.test_original_error_fixed()
    print("✅ Original error fixed")
    
    print("\n🎉 All tests passed! Architecture implementation successful!")