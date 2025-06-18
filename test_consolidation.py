#!/usr/bin/env python3
"""Test script to validate the consolidation was successful."""

import sys
import os
sys.path.insert(0, 'src')

def test_consolidation():
    print("Testing FINN codegen consolidation...")
    
    try:
        # Test core imports
        from finn.codegen import TemplateEngine, BackendRegistry, CodegenConfig
        print("✅ Core imports successful")
        
        # Test template engine
        engine = TemplateEngine()
        print("✅ TemplateEngine initialized")
        
        # Test backend registry  
        from finn.codegen.backend_registration import get_backend_registry
        registry = get_backend_registry()
        print("✅ Backend registry initialized")
        
        # Test config
        from finn.codegen.config import get_global_config
        config = get_global_config()  
        print("✅ Config initialized")
        
        print("🎉 Consolidation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Consolidation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_consolidation()
    sys.exit(0 if success else 1)