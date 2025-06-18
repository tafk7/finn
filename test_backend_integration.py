#!/usr/bin/env python3
"""
Test script to validate backend integration with new template system.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_hls_backend_template_integration():
    """Test HLS backend uses new template system."""
    print("Testing HLS backend template integration...")
    
    try:
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
        from finn.codegen import TemplateEngine
        
        # Create mock HLS backend
        class TestHLSBackend(HLSBackend):
            TEMPLATE_NAME = "hls/docompute.cpp.j2"
            
            def __init__(self):
                # Mock minimal initialization
                self.onnx_node = type('MockNode', (), {
                    'name': 'test_node',
                    'op_type': 'TestOp'
                })()
                self.template_engine = TemplateEngine()
                self.code_gen_dict = {}
                self._current_fpgapart = None
                self._current_clk = None
            
            def get_nodeattr(self, attr):
                """Mock node attribute getter."""
                defaults = {
                    "code_gen_dir_ipgen": "/tmp/test",
                    "code_gen_dir_cppsim": "/tmp/test",
                    "cpp_interface": "packed"
                }
                return defaults.get(attr, "")
            
            def get_ap_int_max_w(self):
                return 4096
                
            def ipgen_default_directives(self):
                return ["set_param hls.enable_hidden_option_error false"]
                
            def ipgen_extra_directives(self):
                return []
            
            # Implement abstract methods
            def global_includes(self):
                self.code_gen_dict["$GLOBALS$"] = ["// test globals"]
                
            def defines(self, var):
                self.code_gen_dict["$DEFINES$"] = ["// test defines"]
                
            def docompute(self):
                self.code_gen_dict["$DOCOMPUTE$"] = ["// test compute"]
                
            def blackboxfunction(self):
                self.code_gen_dict["$BLACKBOXFUNCTION$"] = ["void test_function()"]
        
        backend = TestHLSBackend()
        
        # Test template values extraction
        template_values = backend.get_template_values("hls/docompute.cpp.j2")
        
        # Should have basic required values
        assert 'AP_INT_MAX_W' in template_values
        assert template_values['AP_INT_MAX_W'] == 4096
        assert 'GLOBALS' in template_values
        assert 'DEFINES' in template_values
        
        print("✅ HLS backend template integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ HLS backend test failed: {e}")
        return False

def test_rtl_backend_template_integration():
    """Test RTL backend uses new template system."""
    print("Testing RTL backend template integration...")
    
    try:
        from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
        from finn.codegen import TemplateEngine
        
        # Create mock RTL backend
        class TestRTLBackend(RTLBackend):
            TEMPLATE_NAME = "rtl/swg_wrapper.v.j2"
            
            def __init__(self):
                # Mock minimal initialization
                self.onnx_node = type('MockNode', (), {
                    'name': 'test_rtl_node',
                    'op_type': 'TestRTLOp'
                })()
                self.template_engine = TemplateEngine()
                self._current_model = None
                self._current_fpgapart = None
                self._current_clk = None
                
                # Add mock logger
                import logging
                self.logger = logging.getLogger(__name__)
            
            def get_nodeattr(self, attr):
                """Mock node attribute getter."""
                defaults = {
                    "code_gen_dir_ipgen": "/tmp/test_rtl"
                }
                return defaults.get(attr, "")
            
            def _safe_extract_value(self, obj, attr, default):
                return default
                
            def _extract_data_width(self, obj):
                return 8
                
            def _generate_module_name(self, obj):
                return "test_rtl_module"
            
            # Implement abstract methods
            def get_rtl_file_list(self, abspath=False):
                return ["test_module.v"]
                
            def code_generation_ipi(self):
                return ["create_bd_cell test_module"]
        
        backend = TestRTLBackend()
        
        # Test template values extraction
        template_values = backend.get_template_values("rtl/swg_wrapper.v.j2")
        
        # Should have SWG-specific values
        assert 'TOP_MODULE_NAME' in template_values
        assert template_values['TOP_MODULE_NAME'] == 'test_rtl_node_wrapper'
        assert 'BIT_WIDTH' in template_values
        assert template_values['BIT_WIDTH'] == 8
        
        print("✅ RTL backend template integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ RTL backend test failed: {e}")
        return False

def test_template_files_exist():
    """Test that all template files were created."""
    print("Testing template file existence...")
    
    template_dir = Path(__file__).parent / "src" / "finn" / "codegen" / "templates"
    
    expected_templates = [
        "hls/docompute.cpp.j2",
        "hls/docompute_timeout.cpp.j2",
        "hls/ipgen.cpp.j2",
        "hls/ipgen.tcl.j2",
        "hls/ip_package.tcl.j2",
        "rtl/thresholding_wrapper.v.j2",
        "rtl/swg_wrapper.v.j2"
    ]
    
    missing_templates = []
    for template in expected_templates:
        template_path = template_dir / template
        if not template_path.exists():
            missing_templates.append(template)
    
    if missing_templates:
        print(f"❌ Missing templates: {missing_templates}")
        return False
    
    print("✅ All template files exist")
    return True

def test_simplified_components():
    """Test simplified library resolver and file manager."""
    print("Testing simplified components...")
    
    try:
        from finn.codegen import LibraryResolver, FileManager
        
        # Test library resolver
        resolver = LibraryResolver()
        templates = resolver.list_available_templates("hls")
        assert len(templates) > 0
        assert any("docompute.cpp.j2" in t for t in templates)
        
        # Test file manager
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FileManager(temp_dir)
            test_content = "test content"
            manager.write_file("test.txt", test_content)
            
            assert manager.file_exists("test.txt")
            read_content = manager.read_file("test.txt")
            assert read_content == test_content
        
        print("✅ Simplified components test passed")
        return True
        
    except Exception as e:
        print(f"❌ Simplified components test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running FINN Codegen Backend Integration Tests\n")
    
    tests = [
        test_template_files_exist,
        test_simplified_components,
        test_hls_backend_template_integration,
        test_rtl_backend_template_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Backend integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())