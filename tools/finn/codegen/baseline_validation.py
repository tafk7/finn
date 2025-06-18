#!/usr/bin/env python3
"""
Baseline validation script for FINN codegen testing.
Verifies that implementations can generate code without errors.
"""

import sys
import os
import tempfile
import time
import logging

# Add the codegen directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_suite import CodegenTestSuite

def run_baseline_validation():
    """Run baseline validation to verify code generation works"""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== FINN Codegen Baseline Validation ===")
    
    # Initialize test suite
    try:
        test_suite = CodegenTestSuite()
        logger.info(f"Initialized test suite with {len(test_suite.test_cases)} test cases")
    except Exception as e:
        logger.error(f"Failed to initialize test suite: {e}")
        return False
    
    # Test results tracking
    total_tests = 0
    code_generation_success = 0
    template_processing_success = 0
    
    baseline_results = {}
    
    # Run each test case for baseline validation
    for test_case in test_suite.test_cases:
        total_tests += 1
        logger.info(f"\n--- Baseline test: {test_case.name} ---")
        
        try:
            # Measure generation time
            start_time = time.time()
            
            # Run test (this will attempt code generation)
            result = test_suite.run_single_test(test_case)
            
            end_time = time.time()
            generation_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Check if code was generated (even if comparison failed)
            code_generated = hasattr(result, 'clean_code') and len(result.clean_code) > 0
            
            if code_generated:
                code_generation_success += 1
                logger.info(f"✅ Code generation successful ({len(result.clean_code)} chars, {generation_time:.2f}ms)")
            else:
                logger.warning(f"❌ Code generation failed")
            
            # Check if template processing worked
            template_ok = not ('template' in result.comparison_summary.lower() and 'error' in result.comparison_summary.lower())
            
            if template_ok:
                template_processing_success += 1
                logger.info(f"✅ Template processing successful")
            else:
                logger.warning(f"❌ Template processing failed")
            
            # Store baseline results
            baseline_results[test_case.name] = {
                'code_generated': code_generated,
                'template_ok': template_ok,
                'generation_time_ms': generation_time,
                'code_size': len(result.clean_code) if code_generated else 0,
                'operation': test_case.operation_type,
                'backend': test_case.backend_type
            }
            
        except Exception as e:
            logger.error(f"❌ Baseline test {test_case.name} failed with error: {e}")
            baseline_results[test_case.name] = {
                'code_generated': False,
                'template_ok': False,
                'generation_time_ms': 0,
                'code_size': 0,
                'error': str(e),
                'operation': test_case.operation_type,
                'backend': test_case.backend_type
            }
    
    # Generate baseline summary
    logger.info(f"\n=== BASELINE VALIDATION SUMMARY ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Code generation success: {code_generation_success}/{total_tests} ({code_generation_success/total_tests*100:.1f}%)")
    logger.info(f"Template processing success: {template_processing_success}/{total_tests} ({template_processing_success/total_tests*100:.1f}%)")
    
    # Performance baseline metrics
    hls_times = [r['generation_time_ms'] for r in baseline_results.values() 
                 if r.get('backend') == 'hls' and r['code_generated']]
    rtl_times = [r['generation_time_ms'] for r in baseline_results.values() 
                 if r.get('backend') == 'rtl' and r['code_generated']]
    
    if hls_times:
        avg_hls_time = sum(hls_times) / len(hls_times)
        logger.info(f"Average HLS generation time: {avg_hls_time:.2f}ms")
    
    if rtl_times:
        avg_rtl_time = sum(rtl_times) / len(rtl_times)
        logger.info(f"Average RTL generation time: {avg_rtl_time:.2f}ms")
    
    # Operation-specific analysis
    operations = {}
    for test_name, result in baseline_results.items():
        op_key = f"{result['operation']}_{result['backend']}"
        if op_key not in operations:
            operations[op_key] = []
        operations[op_key].append(result)
    
    logger.info(f"\n=== OPERATION-SPECIFIC BASELINE ===")
    for op_key, results in operations.items():
        success_count = sum(1 for r in results if r['code_generated'])
        avg_time = sum(r['generation_time_ms'] for r in results if r['code_generated']) / max(1, success_count)
        avg_size = sum(r['code_size'] for r in results if r['code_generated']) / max(1, success_count)
        
        logger.info(f"{op_key}: {success_count}/{len(results)} success, {avg_time:.2f}ms avg, {avg_size:.0f} chars avg")
    
    # Save baseline results
    baseline_file = 'baseline_results.txt'
    with open(baseline_file, 'w') as f:
        f.write("# FINN Codegen Baseline Results\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Code generation success: {code_generation_success}/{total_tests}\n") 
        f.write(f"Template processing success: {template_processing_success}/{total_tests}\n")
        f.write("\n# Per-test results:\n")
        for test_name, result in baseline_results.items():
            f.write(f"{test_name}: {result}\n")
    
    logger.info(f"Baseline results saved to {baseline_file}")
    
    # Determine baseline success
    baseline_success = (code_generation_success >= total_tests * 0.8 and 
                       template_processing_success >= total_tests * 0.8)
    
    if baseline_success:
        logger.info("🎉 BASELINE VALIDATION PASSED - Ready for A/B testing")
    else:
        logger.warning("⚠️ BASELINE VALIDATION CONCERNS - Review before proceeding")
    
    return baseline_success, baseline_results

if __name__ == "__main__":
    success, results = run_baseline_validation()
    sys.exit(0 if success else 1)