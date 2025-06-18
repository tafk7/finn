#!/usr/bin/env python3
"""
Simple validation runner for FINN codegen clean vs legacy comparison.
"""

import sys
import os
import logging

# Add the codegen directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codegen_validator import CodegenValidator
from test_suite import CodegenTestSuite

def main():
    """Run comprehensive validation tests."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== FINN Codegen Clean vs Legacy Validation ===")
    
    # Initialize test suite
    try:
        test_suite = CodegenTestSuite()
        logger.info(f"Initialized test suite with {len(test_suite.test_cases)} test cases")
    except Exception as e:
        logger.error(f"Failed to initialize test suite: {e}")
        return 1
    
    # Run validation tests
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_case in test_suite.test_cases:
        logger.info(f"\n--- Running test: {test_case.name} ---")
        total_tests += 1
        
        try:
            # Run the test case validation
            result = test_suite.run_single_test(test_case)
            
            if result and result.validation_passed:
                logger.info(f"✅ PASSED: {test_case.name}")
                passed_tests += 1
            else:
                logger.warning(f"❌ FAILED: {test_case.name}")
                if result:
                    logger.warning(f"   Reason: {result.comparison_summary}")
                failed_tests += 1
                
        except Exception as e:
            logger.error(f"💥 ERROR in {test_case.name}: {e}")
            failed_tests += 1
    
    # Print summary
    logger.info(f"\n=== VALIDATION SUMMARY ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    
    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    sys.exit(main())