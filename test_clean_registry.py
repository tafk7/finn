#!/usr/bin/env python3
"""
Quick test script to verify CG_BackendRegistry functionality.
This validates that our clean backend registration and fallback logic works correctly.
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_clean_registry():
    """Test the clean backend registry functionality."""
    print("🧪 Testing Clean Backend Registry Functionality")
    print("=" * 50)
    
    try:
        # Import our clean registry
        from finn.codegen.CG_backend_registration import CG_BackendRegistry, get_clean_backend_registry
        
        print("✅ Successfully imported CG_BackendRegistry")
        
        # Test basic registry creation
        registry = CG_BackendRegistry()
        print("✅ Successfully created CG_BackendRegistry instance")
        
        # Test stats functionality
        stats = registry.get_registry_stats()
        print(f"✅ Registry stats: {stats}")
        
        # Test that registry starts with clean backends disabled
        assert not stats['clean_backends_enabled'], "Clean backends should be disabled by default"
        print("✅ Clean backends correctly disabled by default")
        
        # Test enabling clean backends
        registry.enable_clean_backends(['Thresholding'])
        assert 'Thresholding' in registry._clean_backend_whitelist
        print("✅ Successfully enabled clean backends for Thresholding")
        
        # Test disabling clean backends
        registry.disable_clean_backends(['Thresholding'])
        assert 'Thresholding' not in registry._clean_backend_whitelist
        print("✅ Successfully disabled clean backends for Thresholding")
        
        # Test global registry function
        global_registry = get_clean_backend_registry()
        assert isinstance(global_registry, CG_BackendRegistry)
        print("✅ Global registry function works correctly")
        
        # Test backend lookup (should fall back to legacy since no clean backends registered yet)
        hls_backend = registry.get_hls_backend('Thresholding')
        print(f"✅ HLS backend lookup works: {hls_backend}")
        
        rtl_backend = registry.get_rtl_backend('Thresholding')  
        print(f"✅ RTL backend lookup works: {rtl_backend}")
        
        # Test clean backend registration methods
        class MockCleanBackend:
            pass
        
        registry.register_clean_hls_backend('TestOp', MockCleanBackend)
        assert 'TestOp' in registry._clean_hls_backends
        print("✅ Clean HLS backend registration works")
        
        registry.register_clean_rtl_backend('TestOp', MockCleanBackend)
        assert 'TestOp' in registry._clean_rtl_backends  
        print("✅ Clean RTL backend registration works")
        
        # Test clean backend selection logic
        registry.enable_clean_backends(['TestOp'])
        clean_hls = registry.get_hls_backend('TestOp', prefer_clean=True)
        assert clean_hls == MockCleanBackend
        print("✅ Clean backend selection logic works correctly")
        
        # Test list clean backends
        clean_backends = registry.list_clean_backends()
        assert 'hls' in clean_backends and 'rtl' in clean_backends
        print("✅ List clean backends functionality works")
        
        print("\n🎉 ALL TESTS PASSED! Clean registry is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_dependencies():
    """Test that we can import the required dependencies."""
    print("\n🔍 Testing Import Dependencies")
    print("-" * 30)
    
    try:
        # Test existing backend registry import
        from finn.codegen.backend_registry import BackendRegistry
        print("✅ Successfully imported existing BackendRegistry")
        
        # Test existing backend registration
        from finn.codegen.backend_registration import get_backend_registry
        existing_registry = get_backend_registry()
        print(f"✅ Existing registry works: {type(existing_registry)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Clean Backend Registry Tests\n")
    
    # Test imports first
    import_success = test_import_dependencies()
    
    if import_success:
        # Test registry functionality
        test_success = test_clean_registry()
        
        if test_success:
            print("\n✅ All tests completed successfully!")
            print("Phase 1 Task 1.3 is COMPLETE ✅")
            sys.exit(0)
        else:
            print("\n❌ Registry tests failed!")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed - check dependencies!")
        sys.exit(1)