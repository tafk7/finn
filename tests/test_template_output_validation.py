############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN Template Output Validation Tests
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
    FINNTemplateEngine,
    FINNTemplateType,
    FINNTemplateMigrator
)
from finn.custom_op.fpgadataflow import templates


class TestTemplateOutputValidation:
    """Test that template output matches expected FINN patterns."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create template directory structure
        self.template_dir = os.path.join(self.temp_dir, "templates", "finn")
        os.makedirs(self.template_dir)
        
        # Create template files
        self._create_test_templates()
        
        # Initialize template engine
        self.template_engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_templates(self):
        """Create test template files."""
        templates_to_create = {
            "ipgen_cpp.template": """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$

#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

$BLACKBOXFUNCTION$
{
$PRAGMAS$
$DOCOMPUTE$
}
""",
            "ipgen_tcl.template": """
set config_proj_name $PROJECTNAME$
puts "HLS project: $config_proj_name"
set config_hwsrcdir "$HWSRCDIR$"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "$FPGAPART$"
set config_bnnlibdir "$::env(FINN_DEPS_DIR)/finn-hlslib"
puts "finn-hlslib dir: $config_bnnlibdir"
set config_customhlsdir "$::env(FINN_ROOT)/custom_hls"
puts "custom HLS dir: $config_customhlsdir"
set config_toplevelfxn "$TOPFXN$"
set config_clkperiod $CLKPERIOD$

open_project $config_proj_name
add_files $config_hwsrcdir/top_$TOPFXN$.cpp -cflags "-std=c++14 -I$config_bnnlibdir -I$config_customhlsdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

$DEFAULT_DIRECTIVES$
$EXTRA_DIRECTIVES$

create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
""",
            "docompute.template": """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$
#define HLS_NO_XIL_FPO_LIB
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

$DOCOMPUTE$

$DATAOUTSTREAM$

$SAVEASCNPY$

}
""",
            "docompute_timeout.template": """
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

unsigned timeout = 0;
while(timeout < $TIMEOUT_VALUE$){

$DOCOMPUTE$

if($TIMEOUT_CONDITION$){
timeout++;
}

else{
$TIMEOUT_READ_STREAM$
timeout = 0;
}
}

$DATAOUTSTREAM$

$SAVEASCNPY$

}
"""
        }
        
        for filename, content in templates_to_create.items():
            with open(os.path.join(self.template_dir, filename), 'w') as f:
                f.write(content.strip())
    
    def test_ipgen_cpp_template_output(self):
        """Test that ipgen_cpp template produces expected output."""
        variables = {
            "AP_INT_MAX_W": "8191",
            "GLOBALS": "#include \"test_globals.h\"",
            "DEFINES": "#define TEST_DEFINE 1",
            "BLACKBOXFUNCTION": "void test_function(hls::stream<ap_uint<8>> &in, hls::stream<ap_uint<8>> &out)",
            "PRAGMAS": "#pragma HLS INTERFACE axis port=in\n#pragma HLS INTERFACE axis port=out",
            "DOCOMPUTE": "test_function(in0_V, out0_V);"
        }
        
        output = self.template_engine.render_template("ipgen_cpp", variables, FINNTemplateType.IPGEN_CPP)
        
        assert output is not None
        assert "AP_INT_MAX_W 8191" in output
        assert "#include \"test_globals.h\"" in output
        assert "#define TEST_DEFINE 1" in output
        assert "void test_function" in output
        assert "#pragma HLS INTERFACE axis port=in" in output
        assert "test_function(in0_V, out0_V);" in output
    
    def test_ipgen_tcl_template_output(self):
        """Test that ipgen_tcl template produces expected output."""
        variables = {
            "PROJECTNAME": "test_project",
            "HWSRCDIR": "/path/to/hw/src",
            "FPGAPART": "xc7z020clg400-1",
            "TOPFXN": "test_top",
            "CLKPERIOD": "10.0",
            "DEFAULT_DIRECTIVES": "config_compile -pipeline_style flp",
            "EXTRA_DIRECTIVES": "config_interface -m_axi_addr64"
        }
        
        output = self.template_engine.render_template("ipgen_tcl", variables, FINNTemplateType.IPGEN_TCL)
        
        assert output is not None
        assert "test_project" in output
        assert "/path/to/hw/src" in output
        assert "xc7z020clg400-1" in output
        assert "test_top" in output
        assert "10.0" in output
        assert "config_compile -pipeline_style flp" in output
        assert "config_interface -m_axi_addr64" in output
    
    def test_docompute_template_output(self):
        """Test that docompute template produces expected output."""
        variables = {
            "AP_INT_MAX_W": "8191",
            "GLOBALS": "#include \"test_params.h\"",
            "DEFINES": "#define BATCH_SIZE 1",
            "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V\n#pragma HLS INTERFACE axis port=out0_V",
            "STREAMDECLARATIONS": "hls::stream<ap_uint<8>> in0_V;\nhls::stream<ap_uint<8>> out0_V;",
            "READNPYDATA": "npy2apintstream<ap_uint<8>, ap_uint<8>, 8, float>(\"input.npy\", in0_V);",
            "DOCOMPUTE": "test_kernel(in0_V, out0_V);",
            "DATAOUTSTREAM": "apintstream2npy<ap_uint<8>, ap_uint<8>, 8, float>(out0_V, {1,8}, \"output.npy\");",
            "SAVEASCNPY": "// Data saved automatically"
        }
        
        output = self.template_engine.render_template("docompute", variables, FINNTemplateType.DOCOMPUTE)
        
        assert output is not None
        assert "AP_INT_MAX_W 8191" in output
        assert "#include \"test_params.h\"" in output
        assert "#define BATCH_SIZE 1" in output
        assert "hls::stream<ap_uint<8>> in0_V;" in output
        assert "test_kernel(in0_V, out0_V);" in output
        assert "apintstream2npy" in output
        assert "int main(){" in output
    
    def test_docompute_timeout_template_output(self):
        """Test that docompute_timeout template produces expected output."""
        variables = {
            "AP_INT_MAX_W": "8191",
            "GLOBALS": "#include \"test_params.h\"",
            "DEFINES": "#define BATCH_SIZE 1",
            "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V",
            "STREAMDECLARATIONS": "hls::stream<ap_uint<8>> in0_V;",
            "READNPYDATA": "npy2apintstream<ap_uint<8>, ap_uint<8>, 8, float>(\"input.npy\", in0_V);",
            "TIMEOUT_VALUE": "1000",
            "DOCOMPUTE": "test_kernel(in0_V, out0_V);",
            "TIMEOUT_CONDITION": "out0_V.empty()",
            "TIMEOUT_READ_STREAM": "strm << out0_V.read();",
            "DATAOUTSTREAM": "apintstream2npy<ap_uint<8>, ap_uint<8>, 8, float>(out0_V, {1,8}, \"output.npy\");",
            "SAVEASCNPY": "// Data saved automatically"
        }
        
        output = self.template_engine.render_template("docompute_timeout", variables, FINNTemplateType.DOCOMPUTE_TIMEOUT)
        
        assert output is not None
        assert "AP_INT_MAX_W 8191" in output
        assert "unsigned timeout = 0;" in output
        assert "while(timeout < 1000){" in output
        assert "if(out0_V.empty()){" in output
        assert "strm << out0_V.read();" in output
        assert "test_kernel(in0_V, out0_V);" in output
    
    def test_variable_substitution_completeness(self):
        """Test that all variables are properly substituted."""
        variables = {
            "AP_INT_MAX_W": "8191",
            "GLOBALS": "// Test globals",
            "DEFINES": "// Test defines"
        }
        
        output = self.template_engine.render_template("ipgen_cpp", variables, FINNTemplateType.IPGEN_CPP)
        
        # Check that no undefined variables remain
        assert "$AP_INT_MAX_W$" not in output
        assert "$GLOBALS$" not in output  
        assert "$DEFINES$" not in output
        
        # Check that values were substituted
        assert "8191" in output
        assert "// Test globals" in output
        assert "// Test defines" in output
    
    def test_template_variable_extraction(self):
        """Test that template variables are correctly extracted."""
        variables = self.template_engine.get_template_variables("ipgen_cpp", FINNTemplateType.IPGEN_CPP)
        
        expected_variables = {
            "AP_INT_MAX_W", "GLOBALS", "DEFINES", 
            "BLACKBOXFUNCTION", "PRAGMAS", "DOCOMPUTE"
        }
        
        assert variables is not None
        for var in expected_variables:
            assert var in variables, f"Missing expected variable: {var}"


class TestTemplateMigrationValidation:
    """Test migration from static templates to template engine."""
    
    def test_static_template_migration(self):
        """Test migration of FINN's static templates."""
        # Create template engine
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FINNTemplateEngine([temp_dir])
            migrator = FINNTemplateMigrator(engine)
            
            # Test static templates (simplified versions)
            static_templates = {
                "ipgen_template": """
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "bnn-library.h"
$GLOBALS$
$DEFINES$
$BLACKBOXFUNCTION$
{
$PRAGMAS$
$DOCOMPUTE$
}
""",
                "ipgentcl_template": """
set config_proj_name $PROJECTNAME$
set config_hwsrcdir "$HWSRCDIR$"
set config_proj_part "$FPGAPART$"
set config_toplevelfxn "$TOPFXN$"
set config_clkperiod $CLKPERIOD$

open_project $config_proj_name
add_files $config_hwsrcdir/top_$TOPFXN$.cpp

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

$DEFAULT_DIRECTIVES$
$EXTRA_DIRECTIVES$

create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
"""
            }
            
            # Test migration
            success = migrator.migrate_static_templates(static_templates)
            assert success, "Template migration should succeed"
            
            # Test that migrated templates work
            variables = {
                "AP_INT_MAX_W": "8191",
                "GLOBALS": "// Test",
                "DEFINES": "// Test", 
                "BLACKBOXFUNCTION": "void test()",
                "PRAGMAS": "// Test",
                "DOCOMPUTE": "// Test"
            }
            
            output = engine.render_template("ipgen_template", variables, FINNTemplateType.IPGEN_CPP)
            assert output is not None
            assert "8191" in output


class TestTemplateBackwardCompatibility:
    """Test backward compatibility with FINN's template usage patterns."""
    
    def test_templates_module_compatibility(self):
        """Test that templates module maintains backward compatibility."""
        # Test that template properties exist and work
        docompute = templates.docompute_template
        assert isinstance(docompute, str)
        assert "$AP_INT_MAX_W$" in docompute
        
        ipgen = templates.ipgen_template  
        assert isinstance(ipgen, str)
        assert "$AP_INT_MAX_W$" in ipgen
        
        ipgen_tcl = templates.ipgentcl_template
        assert isinstance(ipgen_tcl, str)
        assert "$PROJECTNAME$" in ipgen_tcl
        
        docompute_timeout = templates.docompute_template_timeout
        assert isinstance(docompute_timeout, str)
        assert "$TIMEOUT_VALUE$" in docompute_timeout
    
    def test_string_replacement_compatibility(self):
        """Test that templates work with string replacement (FINN pattern)."""
        template = templates.docompute_template
        
        # Test FINN's pattern of string replacement
        replaced = template.replace("$AP_INT_MAX_W$", "8191")
        replaced = replaced.replace("$GLOBALS$", "// Test globals")
        replaced = replaced.replace("$DEFINES$", "// Test defines")
        
        assert "8191" in replaced
        assert "// Test globals" in replaced
        assert "// Test defines" in replaced
        assert "$AP_INT_MAX_W$" not in replaced


class TestTemplateErrorHandling:
    """Test template error handling and fallback mechanisms."""
    
    def test_missing_template_fallback(self):
        """Test fallback when template file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create engine with empty directory
            engine = FINNTemplateEngine([temp_dir])
            
            variables = {"AP_INT_MAX_W": "8191"}
            
            # Should return None for missing template
            output = engine.render_template("nonexistent", variables, FINNTemplateType.CUSTOM)
            assert output is None
    
    def test_variable_validation(self):
        """Test template variable validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create simple template
            template_file = os.path.join(temp_dir, "test.template")
            with open(template_file, 'w') as f:
                f.write("Value: $TEST_VAR$")
            
            engine = FINNTemplateEngine([temp_dir])
            
            # Test with missing variable
            issues = engine.validate_template("test", {}, FINNTemplateType.CUSTOM)
            assert len(issues) > 0
            assert "TEST_VAR" in str(issues)
            
            # Test with all variables
            issues = engine.validate_template("test", {"TEST_VAR": "value"}, FINNTemplateType.CUSTOM)
            assert len(issues) == 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])