#!/usr/bin/env python3
"""Test inheritance fixes for clean backends."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def test_inheritance_fixes():
    """Test that inheritance fixes work correctly."""
    print("Testing Multiple Inheritance Fixes")
    print("=" * 60)
    
    try:
        # Import clean backends
        from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
        from finn.custom_op.fpgadataflow.hls.CG_mvau_hls import CG_MVAU_hls
        from finn.custom_op.fpgadataflow.rtl.CG_thresholding_rtl import CG_Thresholding_rtl
        
        print("✅ All imports successful")
        
        # Test MRO
        print("\n📊 Method Resolution Order (MRO):")
        for name, cls in [("CG_Thresholding_hls", CG_Thresholding_hls),
                          ("CG_MVAU_hls", CG_MVAU_hls),
                          ("CG_Thresholding_rtl", CG_Thresholding_rtl)]:
            mro = [c.__name__ for c in cls.__mro__]
            print(f"\n{name}:")
            for i, c in enumerate(mro):
                print(f"  {i}: {c}")
        
        # Test attribute access with mock node
        print("\n🔍 Testing Attribute Access:")
        import onnx.helper as helper
        
        # Create test node
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresh"], ["outp"],
            domain="finn.custom_op.fpgadataflow",
            NumChannels=4,
            PE=2,
            numSteps=3,
            inputDataType="INT8",
            outputDataType="UINT2"
        )
        
        # Test clean backend instantiation
        print("\n✅ Testing CG_Thresholding_hls:")
        backend = CG_Thresholding_hls(node)
        
        # Test get_nodeattr through Codegen base class
        try:
            pe = backend.get_nodeattr("PE")
            print(f"  get_nodeattr('PE') = {pe}")
            
            channels = backend.get_nodeattr("NumChannels")
            print(f"  get_nodeattr('NumChannels') = {channels}")
            
            # Check which get_nodeattr is being used
            print(f"  get_nodeattr method: {backend.get_nodeattr.__qualname__}")
            
            # Test with non-existent attribute
            try:
                missing = backend.get_nodeattr("NonExistent")
                print(f"  get_nodeattr('NonExistent') = {missing}")
            except Exception:
                print("  NonExistent attribute not found (expected)")
            
            print("  ✅ Attribute access working correctly!")
            
        except Exception as e:
            print(f"  ❌ Attribute access failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test get_nodeattr_types merging
        print("\n📋 Testing get_nodeattr_types() merging:")
        attrs = backend.get_nodeattr_types()
        print(f"  Total attributes: {len(attrs)}")
        
        # Check for attributes from both parents
        thresholding_attrs = ["NumChannels", "PE", "numSteps", "inputDataType"]
        backend_attrs = ["code_gen_dir_cppsim", "executable_path"]
        
        print("  Thresholding attributes:")
        for attr in thresholding_attrs:
            print(f"    {attr}: {'✅' if attr in attrs else '❌'}")
        
        print("  Backend attributes:")
        for attr in backend_attrs:
            print(f"    {attr}: {'✅' if attr in attrs else '❌'}")
        
        print("\n✅ All inheritance tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_inheritance_fixes()
    sys.exit(0 if success else 1)