#!/usr/bin/env python3

"""
🧪 Test Fixed Backend Implementations
Validates that all missing methods have been added and backends work correctly.
"""

import logging
import sys
import tempfile
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_backends():
    """Test all fixed backend implementations."""
    
    logger.info("🚀 Testing Fixed Backend Implementations")
    logger.info("=" * 80)
    
    results = {}
    
    try:
        # Import required modules
        from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
        from finn.custom_op.fpgadataflow.hls.CG_mvau_hls import CG_MVAU_hls
        from finn.custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls
        from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
        from finn.codegen.backend_instance_manager import BackendInstanceManager
        from finn.codegen.test_node_factory import TestNodeFactory
        
        logger.info("✅ All imports successful")
        
        # Setup manager and factory
        manager = BackendInstanceManager()
        factory = TestNodeFactory()
        
        # Test configurations
        backends_to_test = [
            ("Clean Thresholding", CG_Thresholding_hls, "thresholding"),
            ("Clean MVAU", CG_MVAU_hls, "mvau"),
            ("Legacy Thresholding", Thresholding_hls, "thresholding"),
            ("Legacy MVAU", MVAU_hls, "mvau")
        ]
        
        for name, backend_class, node_type in backends_to_test:
            logger.info(f"\n🔍 Testing {name}")
            logger.info("-" * 50)
            
            try:
                # Create appropriate test node
                if node_type == "thresholding":
                    test_node = factory.create_thresholding_node()
                else:
                    test_node = factory.create_mvau_node()
                
                logger.info(f"✅ Test node created: {test_node.name}")
                
                # Test backend instantiation
                backend_instance = manager.create_backend_instance(backend_class, test_node)
                logger.info(f"✅ Backend instantiated successfully")
                
                # Test method availability
                required_methods = ['get_template_values']
                if 'Clean' in name:
                    required_methods.extend(['_generate_common_values', '_generate_operation_specific_values'])
                    if 'MVAU' in name:
                        required_methods.append('code_generation_cppsim')
                else:
                    required_methods.append('code_generation_cppsim')
                
                missing_methods = []
                for method in required_methods:
                    if not hasattr(backend_instance, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    logger.error(f"❌ Missing methods: {missing_methods}")
                    results[name] = f"FAILED - Missing methods: {missing_methods}"
                    continue
                
                logger.info(f"✅ All required methods present: {required_methods}")
                
                # Test method invocation
                if hasattr(backend_instance, 'get_template_values'):
                    try:
                        template_values = backend_instance.get_template_values("base/hls_base.cpp.j2")
                        if template_values:
                            logger.info(f"✅ get_template_values() returned {len(template_values)} values")
                        else:
                            logger.warning("⚠️ get_template_values() returned empty values")
                    except Exception as e:
                        logger.error(f"❌ get_template_values() failed: {e}")
                        results[name] = f"FAILED - get_template_values error: {e}"
                        continue
                
                # Test code generation through manager
                try:
                    if 'Clean' in name:
                        generation_type = 'template'
                    else:
                        generation_type = 'legacy'
                    
                    generated_code = manager.call_backend_generation(backend_instance, generation_type)
                    
                    if generated_code and len(generated_code) > 100:
                        logger.info(f"✅ Code generation successful - {len(generated_code)} characters")
                        logger.info(f"   Preview: {generated_code[:80]}...")
                        results[name] = "SUCCESS"
                    else:
                        logger.error(f"❌ Code generation returned insufficient content")
                        results[name] = "FAILED - Insufficient generated code"
                        
                except Exception as e:
                    logger.error(f"❌ Code generation failed: {e}")
                    results[name] = f"FAILED - Code generation error: {e}"
                    continue
                    
            except Exception as e:
                logger.error(f"❌ {name} test failed: {e}")
                results[name] = f"FAILED - {e}"
        
        # Summary
        logger.info(f"\n🏆 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 80)
        
        success_count = 0
        for name, result in results.items():
            status = "✅" if "SUCCESS" in result else "❌"
            logger.info(f"{status} {name}: {result}")
            if "SUCCESS" in result:
                success_count += 1
        
        total_tests = len(results)
        logger.info(f"\n📊 Summary: {success_count}/{total_tests} backends working")
        
        if success_count == total_tests:
            logger.info("🎉 ALL BACKENDS FIXED AND WORKING!")
            return True
        else:
            logger.info(f"⚠️ {total_tests - success_count} backends still need work")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_methods():
    """Test specific method implementations that were added."""
    
    logger.info(f"\n🔬 Testing Specific Method Implementations") 
    logger.info("=" * 50)
    
    try:
        from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
        from finn.custom_op.fpgadataflow.hls.CG_mvau_hls import CG_MVAU_hls
        from finn.codegen.test_node_factory import TestNodeFactory
        
        factory = TestNodeFactory()
        
        # Test CG_Thresholding_hls methods
        logger.info("🔍 Testing CG_Thresholding_hls methods")
        thresh_node = factory.create_thresholding_node()
        thresh_backend = CG_Thresholding_hls(thresh_node)
        
        # Test _generate_common_values
        common_values = thresh_backend._generate_common_values()
        logger.info(f"✅ _generate_common_values returned {len(common_values)} values")
        logger.info(f"   Keys: {list(common_values.keys())[:5]}...")
        
        # Test _generate_operation_specific_values  
        op_values = thresh_backend._generate_operation_specific_values("base/hls_base.cpp.j2")
        logger.info(f"✅ _generate_operation_specific_values returned {len(op_values)} values")
        logger.info(f"   Keys: {list(op_values.keys())[:5]}...")
        
        # Test CG_MVAU_hls methods
        logger.info("🔍 Testing CG_MVAU_hls methods")
        mvau_node = factory.create_mvau_node()
        mvau_backend = CG_MVAU_hls(mvau_node)
        
        # Test _generate_common_values
        common_values = mvau_backend._generate_common_values()
        logger.info(f"✅ _generate_common_values returned {len(common_values)} values")
        logger.info(f"   Keys: {list(common_values.keys())[:5]}...")
        
        # Test _generate_operation_specific_values
        op_values = mvau_backend._generate_operation_specific_values("base/hls_base.cpp.j2")
        logger.info(f"✅ _generate_operation_specific_values returned {len(op_values)} values")
        logger.info(f"   Keys: {list(op_values.keys())[:5]}...")
        
        logger.info("🎉 All specific method tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Specific method test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🧪 RUNNING COMPREHENSIVE BACKEND FIX VALIDATION")
    logger.info("=" * 80)
    
    # Test fixed backends
    backend_success = test_fixed_backends()
    
    # Test specific methods
    method_success = test_specific_methods()
    
    # Final result
    if backend_success and method_success:
        logger.info(f"\n🏆 OVERALL RESULT: ✅ SUCCESS")
        logger.info("🎉 ALL BACKEND FIXES VALIDATED SUCCESSFULLY!")
        sys.exit(0)
    else:
        logger.info(f"\n🏆 OVERALL RESULT: ❌ SOME ISSUES REMAIN")
        sys.exit(1)