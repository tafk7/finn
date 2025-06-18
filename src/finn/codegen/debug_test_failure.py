#!/usr/bin/env python3
"""
Debug script for investigating critical test failures in FINN codegen validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_suite import CodegenTestSuite
import logging

def debug_specific_test(test_name):
    """Debug a specific failing test in detail"""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== DEBUGGING TEST FAILURE: {test_name} ===")
    
    # Initialize test suite
    suite = CodegenTestSuite()
    
    # Find the specific test
    target_test = None
    for test_case in suite.test_cases:
        if test_case.name == test_name:
            target_test = test_case
            break
    
    if not target_test:
        logger.error(f"Test '{test_name}' not found!")
        available_tests = [t.name for t in suite.test_cases]
        logger.info(f"Available tests: {available_tests}")
        return
    
    logger.info(f"Found test: {target_test.name}")
    logger.info(f"Operation: {target_test.operation_type}")
    logger.info(f"Backend: {target_test.backend_type}")
    logger.info(f"Node attributes: {target_test.node_attributes}")
    
    # Step 1: Try to run the test and capture detailed info
    logger.info("\n--- STEP 1: Running test with detailed logging ---")
    
    try:
        result = suite.run_single_test(target_test)
        logger.info(f"Test completed. Validation passed: {result.validation_passed}")
        logger.info(f"Comparison summary: {result.comparison_summary}")
        
        # Step 2: Examine what was actually generated
        logger.info("\n--- STEP 2: Examining generated outputs ---")
        
        if hasattr(result, 'clean_code') and result.clean_code:
            logger.info(f"Clean code generated: {len(result.clean_code)} characters")
            logger.info("Clean code preview:")
            logger.info("-" * 50)
            preview = result.clean_code[:500] + "..." if len(result.clean_code) > 500 else result.clean_code
            logger.info(preview)
            logger.info("-" * 50)
        else:
            logger.error("❌ NO CLEAN CODE GENERATED - This is the root issue!")
        
        if hasattr(result, 'legacy_code') and result.legacy_code:
            logger.info(f"Legacy code generated: {len(result.legacy_code)} characters")
            logger.info("Legacy code preview:")
            logger.info("-" * 50)
            preview = result.legacy_code[:500] + "..." if len(result.legacy_code) > 500 else result.legacy_code
            logger.info(preview)
            logger.info("-" * 50)
        else:
            logger.error("❌ NO LEGACY CODE GENERATED")
        
        # Step 3: Check pattern matching
        logger.info("\n--- STEP 3: Pattern analysis ---")
        expected_patterns = target_test.expected_patterns or []
        logger.info(f"Expected patterns: {expected_patterns}")
        
        if hasattr(result, 'clean_code') and result.clean_code:
            for pattern in expected_patterns:
                found = pattern in result.clean_code
                status = "✅" if found else "❌"
                logger.info(f"{status} Pattern '{pattern}': {'FOUND' if found else 'MISSING'}")
        
        # Step 4: Check the validation framework itself
        logger.info("\n--- STEP 4: Validation framework analysis ---")
        logger.info(f"Validator class: {type(suite.validator)}")
        
        # Try to access validator properties
        if hasattr(suite.validator, 'clean_registry') and hasattr(suite.validator, 'legacy_registry'):
            logger.info("✅ Validator has both clean and legacy registries")
            
            # Check registry stats
            try:
                clean_stats = suite.validator.clean_registry.get_registry_stats()
                logger.info(f"Clean registry stats: {clean_stats}")
            except Exception as e:
                logger.error(f"❌ Error getting clean registry stats: {e}")
                
            try:
                legacy_stats = suite.validator.legacy_registry.get_registry_stats()
                logger.info(f"Legacy registry stats: {legacy_stats}")
            except Exception as e:
                logger.error(f"❌ Error getting legacy registry stats: {e}")
        else:
            logger.error("❌ Validator missing expected registry structure")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR during test execution: {e}")
        import traceback
        traceback.print_exc()
        
    # Step 5: Manual backend testing
    logger.info("\n--- STEP 5: Manual backend instantiation test ---")
    
    try:
        # Try to manually instantiate the backend
        if target_test.operation_type.lower() == 'thresholding':
            logger.info("Testing Thresholding backend instantiation...")
            
            # Test clean implementation
            try:
                from finn.custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls
                clean_backend = Thresholding_hls()
                logger.info("✅ Clean Thresholding backend instantiated successfully")
                
                # Test if it has the expected methods
                if hasattr(clean_backend, 'get_template_values'):
                    logger.info("✅ Clean backend has get_template_values method")
                else:
                    logger.error("❌ Clean backend missing get_template_values method")
                    
            except Exception as e:
                logger.error(f"❌ Failed to instantiate clean Thresholding backend: {e}")
                
        logger.info("\n=== DEBUG SUMMARY ===")
        logger.info("Review the output above to identify the root cause of test failures.")
        logger.info("Look for:")
        logger.info("1. Missing code generation (no clean/legacy code)")
        logger.info("2. Registry configuration issues") 
        logger.info("3. Backend instantiation failures")
        logger.info("4. Template/pattern generation problems")
        
    except Exception as e:
        logger.error(f"❌ Error in manual backend testing: {e}")

def main():
    """Main debug function"""
    if len(sys.argv) < 2:
        print("Usage: python debug_test_failure.py <test_name>")
        print("Example: python debug_test_failure.py thresholding_hls_basic")
        sys.exit(1)
    
    test_name = sys.argv[1]
    debug_specific_test(test_name)

if __name__ == "__main__":
    main()