#!/usr/bin/env python3
"""
FINN Python Library Validation Test

This script validates that all FINN ecosystem Python libraries are properly
installed and accessible in the Docker environment.
"""

import sys
import importlib
import traceback
from pathlib import Path

def test_import(module_name, optional=False):
    """Test importing a module and return success status."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name}: Successfully imported")
        
        # Try to get version if available
        if hasattr(module, '__version__'):
            print(f"   Version: {module.__version__}")
        elif hasattr(module, 'version'):
            print(f"   Version: {module.version}")
        
        return True, module
    except ImportError as e:
        status = "⚠️" if optional else "❌"
        print(f"{status} {module_name}: Import failed - {e}")
        if not optional:
            print(f"   Error details: {traceback.format_exc()}")
        return False, None
    except Exception as e:
        print(f"❌ {module_name}: Unexpected error - {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False, None

def test_finn_functionality():
    """Test basic FINN functionality."""
    print("\n🔬 Testing FINN functionality...")
    
    try:
        import finn
        from finn.util.basic import get_finn_root
        
        finn_root = get_finn_root()
        print(f"✅ FINN root detected: {finn_root}")
        
        # Test if we can access FINN transformations
        try:
            from finn.transformation.general import CreateDataflowPartition
            print("✅ FINN transformations accessible")
        except ImportError as e:
            print(f"⚠️ FINN transformations not accessible: {e}")
            
        return True
    except Exception as e:
        print(f"❌ FINN functionality test failed: {e}")
        return False

def test_qonnx_functionality():
    """Test basic QONNX functionality."""
    print("\n🔬 Testing QONNX functionality...")
    
    try:
        import qonnx
        from qonnx.core.modelwrapper import ModelWrapper
        print("✅ QONNX ModelWrapper accessible")
        
        # Test QONNX transformations
        try:
            from qonnx.transformation.general import GiveUniqueNodeNames
            print("✅ QONNX transformations accessible")
        except ImportError as e:
            print(f"⚠️ QONNX transformations not accessible: {e}")
            
        return True
    except Exception as e:
        print(f"❌ QONNX functionality test failed: {e}")
        return False

def check_environment():
    """Check Python environment and paths."""
    print("🐍 Python Environment Information:")
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python path length: {len(sys.path)} entries")
    
    # Check for FINN-specific paths
    finn_paths = [p for p in sys.path if 'finn' in p.lower()]
    if finn_paths:
        print(f"   FINN-related paths: {len(finn_paths)}")
        for path in finn_paths:
            print(f"     - {path}")
    else:
        print("   No FINN-related paths found in sys.path")

def main():
    """Main test function."""
    print("🧪 FINN Docker Environment - Python Library Validation Test")
    print("=" * 60)
    
    check_environment()
    print()
    
    # Core libraries (required)
    core_libraries = [
        "numpy",
        "onnx", 
        "qonnx",
        "finn"
    ]
    
    # Optional/Extended libraries
    optional_libraries = [
        "brevitas",
        "finn_experimental", 
        "torch",
        "onnxruntime",
        "netron"
    ]
    
    print("📦 Testing Core Libraries (Required):")
    core_success = 0
    for lib in core_libraries:
        success, _ = test_import(lib, optional=False)
        if success:
            core_success += 1
    
    print(f"\n📦 Testing Optional Libraries:")
    optional_success = 0
    for lib in optional_libraries:
        success, _ = test_import(lib, optional=True)
        if success:
            optional_success += 1
    
    # Functionality tests
    test_finn_functionality()
    test_qonnx_functionality()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   Core libraries: {core_success}/{len(core_libraries)} ({'✅' if core_success == len(core_libraries) else '❌'})")
    print(f"   Optional libraries: {optional_success}/{len(optional_libraries)} ({'✅' if optional_success > 0 else '⚠️'})")
    
    if core_success == len(core_libraries):
        print("\n🎉 All core libraries are available! FINN environment is ready.")
        return 0
    else:
        print(f"\n⚠️ Missing {len(core_libraries) - core_success} core libraries. Environment may not be fully functional.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
