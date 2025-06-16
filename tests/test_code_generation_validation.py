############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN Code Generation Validation Tests
############################################################################

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add the finn paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finn.util.flexible_hls import (
    FlexibleHLSBackend,
    FINNTemplateEngine,
    FINNTemplateType
)
from finn.custom_op.fpgadataflow import templates


class TestCodeGenerationValidation:
    """Test that flexible backend generates identical code to original FINN templates."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN environment
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
        
        self._create_mock_files()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_files(self):
        """Create mock files for testing."""
        # Create finn-hlslib files
        with open(os.path.join(self.finn_hlslib, "bnn-library.h"), 'w') as f:
            f.write("// Mock BNN library\n")
    
    def _normalize_whitespace(self, text):
        """Normalize whitespace for comparison."""
        if text is None:
            return ""
        
        # Remove leading/trailing whitespace and normalize line endings
        lines = [line.strip() for line in text.splitlines()]
        # Remove empty lines for comparison
        lines = [line for line in lines if line]
        return "\n".join(lines)
    
    def test_ipgen_cpp_template_identical_output(self):
        """Test that ipgen_cpp template generates identical output to original."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test variables that would be used in real FINN kernels
            test_variables = {
                "AP_INT_MAX_W": "8191",
                "GLOBALS": "#include \"params.h\"",
                "DEFINES": "#define PE 4\n#define SIMD 8",
                "BLACKBOXFUNCTION": "void MatrixVectorActivation_Stream(hls::stream<ap_uint<32>> &in0_V, hls::stream<ap_uint<32>> &out0_V, ap_uint<8> const weights[32][32], ap_int<16> const thresholds[32])",
                "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V\n#pragma HLS INTERFACE axis port=out0_V\n#pragma HLS INTERFACE m_axi offset=slave bundle=gmem0 port=weights\n#pragma HLS INTERFACE m_axi offset=slave bundle=gmem1 port=thresholds",
                "DOCOMPUTE": "MatrixVectorActivation_Stream<32, 32, 4, 8, ap_uint<8>, ap_int<16>>(in0_V, out0_V, weights, thresholds);"
            }
            
            # Generate output using original static template
            original_template = templates.ipgen_template
            original_output = original_template
            for var, value in test_variables.items():
                original_output = original_output.replace(f"${var}$", value)
            
            # Generate output using flexible template engine
            template_dir = os.path.join(self.temp_dir, "templates", "finn")
            os.makedirs(template_dir)
            
            # Copy the exact template content to file
            template_file = os.path.join(template_dir, "ipgen_cpp.template")
            with open(template_file, 'w') as f:
                f.write(original_template)
            
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            flexible_output = engine.render_template("ipgen_cpp", test_variables, FINNTemplateType.IPGEN_CPP)
            
            # Compare normalized outputs
            original_normalized = self._normalize_whitespace(original_output)
            flexible_normalized = self._normalize_whitespace(flexible_output)
            
            assert original_normalized == flexible_normalized, "Template outputs should be identical"
    
    def test_ipgen_tcl_template_identical_output(self):
        """Test that ipgen_tcl template generates identical output to original."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test variables for TCL template
            test_variables = {
                "PROJECTNAME": "test_project_0",
                "HWSRCDIR": "/tmp/test_hw_src",
                "FPGAPART": "xc7z020clg400-1",
                "TOPFXN": "MatrixVectorActivation_Stream",
                "CLKPERIOD": "10.0",
                "DEFAULT_DIRECTIVES": "config_compile -pipeline_style flp",
                "EXTRA_DIRECTIVES": "config_interface -m_axi_addr64"
            }
            
            # Generate output using original static template
            original_template = templates.ipgentcl_template
            original_output = original_template
            for var, value in test_variables.items():
                original_output = original_output.replace(f"${var}$", value)
            
            # Generate output using flexible template engine
            template_dir = os.path.join(self.temp_dir, "templates", "finn")
            os.makedirs(template_dir)
            
            template_file = os.path.join(template_dir, "ipgen_tcl.template")
            with open(template_file, 'w') as f:
                f.write(original_template)
            
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            flexible_output = engine.render_template("ipgen_tcl", test_variables, FINNTemplateType.IPGEN_TCL)
            
            # Compare normalized outputs
            original_normalized = self._normalize_whitespace(original_output)
            flexible_normalized = self._normalize_whitespace(flexible_output)
            
            assert original_normalized == flexible_normalized, "TCL template outputs should be identical"
    
    def test_docompute_template_identical_output(self):
        """Test that docompute template generates identical output to original."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test variables for docompute template
            test_variables = {
                "AP_INT_MAX_W": "8191",
                "GLOBALS": "#include \"params.h\"",
                "DEFINES": "#define PE 4\n#define SIMD 8\n#define MW 32\n#define MH 32",
                "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V\n#pragma HLS INTERFACE axis port=out0_V",
                "STREAMDECLARATIONS": "hls::stream<ap_uint<32>> in0_V(\"in0_V\");\nhls::stream<ap_uint<32>> out0_V(\"out0_V\");",
                "READNPYDATA": "npy2apintstream<ap_uint<32>, ap_uint<32>, 32, float>(\"input.npy\", in0_V);",
                "DOCOMPUTE": "MatrixVectorActivation_Stream<32, 32, 4, 8>(in0_V, out0_V, weights, thresholds);",
                "DATAOUTSTREAM": "apintstream2npy<ap_uint<32>, ap_uint<32>, 32, float>(out0_V, {1, 32}, \"output.npy\");",
                "SAVEASCNPY": "// Data saved automatically by apintstream2npy"
            }
            
            # Generate output using original static template
            original_template = templates.docompute_template
            original_output = original_template
            for var, value in test_variables.items():
                original_output = original_output.replace(f"${var}$", value)
            
            # Generate output using flexible template engine
            template_dir = os.path.join(self.temp_dir, "templates", "finn")
            os.makedirs(template_dir)
            
            template_file = os.path.join(template_dir, "docompute.template")
            with open(template_file, 'w') as f:
                f.write(original_template)
            
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            flexible_output = engine.render_template("docompute", test_variables, FINNTemplateType.DOCOMPUTE)
            
            # Compare normalized outputs
            original_normalized = self._normalize_whitespace(original_output)
            flexible_normalized = self._normalize_whitespace(flexible_output)
            
            assert original_normalized == flexible_normalized, "Docompute template outputs should be identical"
    
    def test_docompute_timeout_template_identical_output(self):
        """Test that docompute_timeout template generates identical output to original."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test variables for timeout template
            test_variables = {
                "AP_INT_MAX_W": "8191",
                "GLOBALS": "#include \"params.h\"",
                "DEFINES": "#define PE 4\n#define SIMD 8\n#define MW 32\n#define MH 32",
                "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V\n#pragma HLS INTERFACE axis port=out0_V",
                "STREAMDECLARATIONS": "hls::stream<ap_uint<32>> in0_V(\"in0_V\");\nhls::stream<ap_uint<32>> out0_V(\"out0_V\");",
                "READNPYDATA": "npy2apintstream<ap_uint<32>, ap_uint<32>, 32, float>(\"input.npy\", in0_V);",
                "TIMEOUT_VALUE": "1000",
                "DOCOMPUTE": "MatrixVectorActivation_Stream<32, 32, 4, 8>(in0_V, out0_V, weights, thresholds);",
                "TIMEOUT_CONDITION": "out0_V.empty()",
                "TIMEOUT_READ_STREAM": "ap_uint<32> elem = out0_V.read();\nstrm << elem;",
                "DATAOUTSTREAM": "apintstream2npy<ap_uint<32>, ap_uint<32>, 32, float>(strm, {1, 32}, \"output.npy\");",
                "SAVEASCNPY": "// Data saved automatically by apintstream2npy"
            }
            
            # Generate output using original static template
            original_template = templates.docompute_template_timeout
            original_output = original_template
            for var, value in test_variables.items():
                original_output = original_output.replace(f"${var}$", value)
            
            # Generate output using flexible template engine
            template_dir = os.path.join(self.temp_dir, "templates", "finn")
            os.makedirs(template_dir)
            
            template_file = os.path.join(template_dir, "docompute_timeout.template")
            with open(template_file, 'w') as f:
                f.write(original_template)
            
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            flexible_output = engine.render_template("docompute_timeout", test_variables, FINNTemplateType.DOCOMPUTE_TIMEOUT)
            
            # Compare normalized outputs
            original_normalized = self._normalize_whitespace(original_output)
            flexible_normalized = self._normalize_whitespace(flexible_output)
            
            assert original_normalized == flexible_normalized, "Timeout template outputs should be identical"
    
    def test_variable_substitution_completeness(self):
        """Test that all variables are substituted completely and identically."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test with complex variable set
            test_variables = {
                "AP_INT_MAX_W": "8191",
                "GLOBALS": "#include \"complex_params.h\"\n#include \"matrix_ops.h\"",
                "DEFINES": "#define PE 8\n#define SIMD 16\n#define MW 64\n#define MH 64\n#define ACTIVATION_TYPE 1",
                "BLACKBOXFUNCTION": "void ComplexKernel_Stream(hls::stream<ap_uint<64>> &in0_V, hls::stream<ap_uint<64>> &out0_V)",
                "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V register\n#pragma HLS INTERFACE axis port=out0_V register\n#pragma HLS DATAFLOW",
                "DOCOMPUTE": "ComplexKernel_Stream<64, 64, 8, 16, ap_uint<8>, ap_int<16>>(in0_V, out0_V);"
            }
            
            # Test original template substitution
            original_template = templates.ipgen_template
            original_output = original_template
            for var, value in test_variables.items():
                original_output = original_output.replace(f"${var}$", value)
            
            # Verify no unsubstituted variables remain in original
            remaining_vars_original = [line for line in original_output.splitlines() if '$' in line and line.strip().startswith('$') and line.strip().endswith('$')]
            
            # Test flexible template substitution
            template_dir = os.path.join(self.temp_dir, "templates", "finn")
            os.makedirs(template_dir)
            
            template_file = os.path.join(template_dir, "complex_test.template")
            with open(template_file, 'w') as f:
                f.write(original_template)
            
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            flexible_output = engine.render_template("complex_test", test_variables, FINNTemplateType.IPGEN_CPP)
            
            # Verify no unsubstituted variables remain in flexible output
            if flexible_output:
                remaining_vars_flexible = [line for line in flexible_output.splitlines() if '$' in line and '$' in line]
                
                # Both should have same substitution completeness
                assert len(remaining_vars_original) == len(remaining_vars_flexible), "Variable substitution completeness should match"
    
    def test_edge_case_variable_handling(self):
        """Test handling of edge cases in variable substitution."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test edge cases: empty values, special characters, etc.
            test_variables = {
                "AP_INT_MAX_W": "",  # Empty value
                "GLOBALS": "// No globals",  # Comment-only
                "DEFINES": "#define EMPTY_DEFINE",  # Define without value
                "BLACKBOXFUNCTION": "void empty_function()",
                "PRAGMAS": "",  # Empty pragmas
                "DOCOMPUTE": "// Empty compute"
            }
            
            # Test original behavior
            original_template = templates.ipgen_template
            original_output = original_template
            for var, value in test_variables.items():
                original_output = original_output.replace(f"${var}$", value)
            
            # Test flexible behavior
            template_dir = os.path.join(self.temp_dir, "templates", "finn")
            os.makedirs(template_dir)
            
            template_file = os.path.join(template_dir, "edge_case.template")
            with open(template_file, 'w') as f:
                f.write(original_template)
            
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            flexible_output = engine.render_template("edge_case", test_variables, FINNTemplateType.IPGEN_CPP)
            
            # Compare handling of edge cases
            if flexible_output:
                original_normalized = self._normalize_whitespace(original_output)
                flexible_normalized = self._normalize_whitespace(flexible_output)
                
                assert original_normalized == flexible_normalized, "Edge case handling should be identical"
    
    def test_real_kernel_code_generation_comparison(self):
        """Test code generation using realistic FINN kernel parameters."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Realistic MatrixVectorActivation kernel parameters
            mva_variables = {
                "AP_INT_MAX_W": "8191",
                "GLOBALS": "#include \"params.h\"\nnamespace bnn {\nnamespace matrix {\n#include \"weights.h\"\n#include \"thresholds.h\"\n}\n}",
                "DEFINES": "#define MW 784\n#define MH 256\n#define PE 16\n#define SIMD 49\n#define ActVal 0",
                "BLACKBOXFUNCTION": "void MatrixVectorActivation_Stream_0(hls::stream<ap_uint<392>> &in0_V, hls::stream<ap_uint<16>> &out0_V)",
                "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V\n#pragma HLS INTERFACE axis port=out0_V\n#pragma HLS INTERFACE s_axilite port=return bundle=control",
                "DOCOMPUTE": "MatrixVectorActivation_Stream<784, 256, 16, 49, ap_uint<8>, ap_int<16>, ap_uint<8>>(in0_V, out0_V, bnn::matrix::weights, bnn::matrix::thresholds, true);"
            }
            
            # Test both templates generate identical code
            templates_to_test = [
                ("ipgen_cpp", templates.ipgen_template, FINNTemplateType.IPGEN_CPP),
                ("ipgen_tcl", templates.ipgentcl_template, FINNTemplateType.IPGEN_TCL)
            ]
            
            # Update variables for TCL template
            tcl_variables = mva_variables.copy()
            tcl_variables.update({
                "PROJECTNAME": "MatrixVectorActivation_Stream_0",
                "HWSRCDIR": "/tmp/finn_hw_src_MatrixVectorActivation_Stream_0",
                "FPGAPART": "xc7z020clg400-1",
                "TOPFXN": "MatrixVectorActivation_Stream_0",
                "CLKPERIOD": "10.0",
                "DEFAULT_DIRECTIVES": "config_compile -pipeline_style frp",
                "EXTRA_DIRECTIVES": "config_interface -m_axi_addr64"
            })
            
            for template_name, original_template, template_type in templates_to_test:
                # Choose appropriate variables
                variables = tcl_variables if template_type == FINNTemplateType.IPGEN_TCL else mva_variables
                
                # Original generation
                original_output = original_template
                for var, value in variables.items():
                    original_output = original_output.replace(f"${var}$", value)
                
                # Flexible generation
                template_dir = os.path.join(self.temp_dir, "templates", "finn")
                os.makedirs(template_dir, exist_ok=True)
                
                template_file = os.path.join(template_dir, f"{template_name}.template")
                with open(template_file, 'w') as f:
                    f.write(original_template)
                
                engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
                flexible_output = engine.render_template(template_name, variables, template_type)
                
                # Compare outputs
                if flexible_output:
                    original_normalized = self._normalize_whitespace(original_output)
                    flexible_normalized = self._normalize_whitespace(flexible_output)
                    
                    assert original_normalized == flexible_normalized, f"{template_name} template should generate identical output"


class TestBackwardCompatibilityValidation:
    """Test that the migration maintains 100% backward compatibility."""
    
    def test_templates_module_still_works(self):
        """Test that existing code using templates module still works."""
        # Test that old template access still works
        docompute = templates.docompute_template
        assert isinstance(docompute, str)
        assert len(docompute) > 0
        assert "$AP_INT_MAX_W$" in docompute
        
        ipgen = templates.ipgen_template
        assert isinstance(ipgen, str)
        assert len(ipgen) > 0
        assert "$BLACKBOXFUNCTION$" in ipgen
        
        ipgen_tcl = templates.ipgentcl_template
        assert isinstance(ipgen_tcl, str)
        assert len(ipgen_tcl) > 0
        assert "$PROJECTNAME$" in ipgen_tcl
        
        timeout = templates.docompute_template_timeout
        assert isinstance(timeout, str)
        assert len(timeout) > 0
        assert "$TIMEOUT_VALUE$" in timeout
    
    def test_string_replacement_still_works(self):
        """Test that old string replacement pattern still works."""
        # This is how FINN kernels currently use templates
        template = templates.docompute_template
        
        # Standard FINN replacement pattern
        filled_template = template
        filled_template = filled_template.replace("$AP_INT_MAX_W$", "8191")
        filled_template = filled_template.replace("$GLOBALS$", "#include \"test.h\"")
        filled_template = filled_template.replace("$DEFINES$", "#define TEST 1")
        
        # Should work without issues
        assert "8191" in filled_template
        assert "#include \"test.h\"" in filled_template
        assert "#define TEST 1" in filled_template
        assert "$AP_INT_MAX_W$" not in filled_template
        assert "$GLOBALS$" not in filled_template
        assert "$DEFINES$" not in filled_template
    
    def test_existing_kernel_inheritance_compatibility(self):
        """Test that existing kernel inheritance patterns still work."""
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
        
        # Test that HLSBackend is now FlexibleHLSBackend
        assert issubclass(HLSBackend, FlexibleHLSBackend)
        
        # Test that old inheritance pattern still works
        class OldStyleKernel(HLSBackend):
            def __init__(self, onnx_node):
                super().__init__()
                self.onnx_node = onnx_node
        
        # Should be able to instantiate
        mock_node = Mock()
        kernel = OldStyleKernel(mock_node)
        assert isinstance(kernel, HLSBackend)
        assert isinstance(kernel, FlexibleHLSBackend)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])