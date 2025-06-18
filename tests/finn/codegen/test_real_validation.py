#!/usr/bin/env python3
"""
Test script for real backend validation functionality.
"""

import sys
import logging
sys.path.insert(0, '/home/tafk/dev/tafk-finn-1/src')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_real_backend_validation():
    """Test real backend validation with actual backend invocation."""
    
    try:
        from finn.codegen.codegen_validator import CodegenValidator
        
        print("🚀 Testing Real Backend Validation")
        print("=" * 50)
        
        # Create validator with real backend enabled
        validator = CodegenValidator(
            enable_clean_backends=True,
            enable_real_backends=True
        )
        print("✅ Validator created successfully")
        
        # Test simple backend validation
        print("\n🔍 Testing Thresholding HLS validation...")
        result = validator.validate_backend(
            operation_type='Thresholding',
            backend_type='hls',
            NumChannels=32,
            PE=4,
            NumSteps=8
        )
        
        print(f"\n📊 Validation Results:")
        print(f"   Status: {'✅ PASS' if result.validation_result.value == 'pass' else '❌ FAIL'}")
        print(f"   Functional Equivalent: {result.functional_equivalent}")
        
        if result.error_message:
            print(f"   Error: {result.error_message}")
            
        if result.clean_output:
            print(f"\n🔧 Clean Backend Output ({len(result.clean_output)} chars):")
            print(result.clean_output[:300] + "..." if len(result.clean_output) > 300 else result.clean_output)
            
        if result.legacy_output:
            print(f"\n🔧 Legacy Backend Output ({len(result.legacy_output)} chars):")
            print(result.legacy_output[:300] + "..." if len(result.legacy_output) > 300 else result.legacy_output)
            
        return result.validation_result.value == 'pass'
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fake_vs_real_comparison():
    """Compare fake vs real backend validation."""
    
    try:
        from finn.codegen.codegen_validator import CodegenValidator
        
        print("\n🔄 Comparing Fake vs Real Backend Validation")
        print("=" * 50)
        
        # Test with fake backends
        fake_validator = CodegenValidator(enable_real_backends=False)
        fake_result = fake_validator.validate_backend('Thresholding', 'hls')
        
        # Test with real backends  
        real_validator = CodegenValidator(enable_real_backends=True)
        real_result = real_validator.validate_backend('Thresholding', 'hls')
        
        print(f"\n📊 Comparison Results:")
        print(f"   Fake Backend Status: {fake_result.validation_result.value}")
        print(f"   Real Backend Status: {real_result.validation_result.value}")
        
        if fake_result.clean_output and real_result.clean_output:
            fake_has_fake_marker = "FAKE" in fake_result.clean_output
            real_has_fake_marker = "FAKE" in real_result.clean_output
            
            print(f"   Fake output contains 'FAKE': {fake_has_fake_marker}")
            print(f"   Real output contains 'FAKE': {real_has_fake_marker}")
            
            return fake_has_fake_marker and not real_has_fake_marker
            
        return False
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 FINN Real Backend Validation Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test 1: Real backend validation
    print("\n📋 Test 1: Real Backend Validation")
    success &= test_real_backend_validation()
    
    # Test 2: Fake vs Real comparison
    print("\n📋 Test 2: Fake vs Real Comparison")  
    success &= test_fake_vs_real_comparison()
    
    print(f"\n🎯 Overall Test Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    exit(0 if success else 1)