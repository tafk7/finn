#!/usr/bin/env python3
"""Simple FINN library test"""

def test_core_libraries():
    """Test core FINN libraries"""
    print("🧪 FINN Docker Environment - Library Test")
    print("=" * 40)
    
    # Test FINN
    try:
        import finn
        print("✅ FINN: Successfully imported")
        
        # Test FINN root
        from finn.util.basic import get_finn_root
        finn_root = get_finn_root()
        print(f"   FINN root: {finn_root}")
        
    except Exception as e:
        print(f"❌ FINN: {e}")
    
    # Test QONNX
    try:
        import qonnx
        print(f"✅ QONNX: Successfully imported (v{qonnx.__version__})")
    except Exception as e:
        print(f"❌ QONNX: {e}")
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✅ NumPy: Successfully imported (v{np.__version__})")
    except Exception as e:
        print(f"❌ NumPy: {e}")
    
    # Test ONNX
    try:
        import onnx
        print(f"✅ ONNX: Successfully imported (v{onnx.__version__})")
    except Exception as e:
        print(f"❌ ONNX: {e}")
    
    print("\n✨ Library test completed!")

if __name__ == "__main__":
    test_core_libraries()
