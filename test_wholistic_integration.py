#!/usr/bin/env python3

"""
Simple test script to validate the wholistic integration of template-native execution methods.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary imports work."""
    try:
        import qonnx
        print("✓ QONNX import successful")
    except ImportError as e:
        print(f"✗ QONNX import failed: {e}")
        return False
    
    try:
        from finn.custom_op.fpgadataflow.CG_hlsbackend import CG_HLSBackend
        print("✓ CG_HLSBackend import successful")
    except ImportError as e:
        print(f"✗ CG_HLSBackend import failed: {e}")
        return False
        
    try:
        from finn.custom_op.fpgadataflow.hls.transition_thresholding_hls import CG_Thresholding_hls_Full
        print("✓ CG_Thresholding_hls_Full import successful")
    except ImportError as e:
        print(f"✗ CG_Thresholding_hls_Full import failed: {e}")
        return False
        
    return True

def test_template_engine():
    """Test that template engine can be initialized."""
    try:
        from finn.codegen import TemplateEngine
        engine = TemplateEngine()
        print("✓ TemplateEngine initialization successful")
        return True
    except Exception as e:
        print(f"✗ TemplateEngine initialization failed: {e}")
        return False

def test_execution_methods():
    """Test that execution methods can be called."""
    try:
        from finn.custom_op.fpgadataflow.CG_hlsbackend import CG_HLSBackend
        
        # Create a mock instance to test methods
        class MockBackend(CG_HLSBackend):
            def __init__(self):
                # Minimal initialization without ONNX node
                self.logger = None
                self.template_engine = None
                
            def _generate_common_values(self, instance):
                return {}
                
            def _generate_operation_specific_values(self, template_name):
                return {}
                
            def get_template_name(self):
                return "hls_basic.cpp.j2"
        
        backend = MockBackend()
        
        # Test method existence
        assert hasattr(backend, 'code_generation_cppsim'), "code_generation_cppsim method missing"
        assert hasattr(backend, 'code_generation_ipgen'), "code_generation_ipgen method missing"
        assert hasattr(backend, 'compile_singlenode_code'), "compile_singlenode_code method missing"
        assert hasattr(backend, 'execute_node'), "execute_node method missing"
        
        print("✓ All execution methods present")
        return True
        
    except Exception as e:
        print(f"✗ Execution methods test failed: {e}")
        return False

def test_template_files():
    """Test that template files exist."""
    template_files = [
        "src/finn/codegen/templates/execution/cppsim.cpp.j2",
        "src/finn/codegen/templates/execution/ipgen.cpp.j2", 
        "src/finn/codegen/templates/execution/ipgen.tcl.j2",
        "src/finn/codegen/templates/execution/params.h.j2"
    ]
    
    all_exist = True
    for template_file in template_files:
        if os.path.exists(template_file):
            print(f"✓ Template file exists: {template_file}")
        else:
            print(f"✗ Template file missing: {template_file}")
            all_exist = False
    
    return all_exist

def main():
    """Run all validation tests."""
    print("=== Wholistic Integration Validation ===\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Template Engine Test", test_template_engine),
        ("Execution Methods Test", test_execution_methods),
        ("Template Files Test", test_template_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print("\n=== Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Wholistic integration is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())