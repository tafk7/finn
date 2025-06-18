#!/usr/bin/env python3
"""
Test Thresholding HLS operations individually
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_suite import CodegenTestSuite

def test_thresholding_hls():
    """Test all Thresholding HLS operations"""
    print("=== Testing Thresholding HLS Operations ===")
    
    suite = CodegenTestSuite()
    thresholding_tests = [t for t in suite.test_cases if 'thresholding_hls' in t.name]
    
    print(f"Found {len(thresholding_tests)} Thresholding HLS test cases")
    
    results = {}
    for test in thresholding_tests:
        print(f"\nTesting: {test.name}")
        try:
            result = suite.run_single_test(test)
            status = "PASS" if result.validation_passed else "FAIL"
            print(f"  Result: {status}")
            if not result.validation_passed:
                print(f"  Reason: {result.comparison_summary}")
            results[test.name] = {
                'status': status,
                'validation_passed': result.validation_passed,
                'reason': result.comparison_summary
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[test.name] = {
                'status': 'ERROR',
                'validation_passed': False,
                'reason': str(e)
            }
    
    # Summary
    print(f"\n=== THRESHOLDING HLS TEST SUMMARY ===")
    passed = sum(1 for r in results.values() if r['validation_passed'])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for test_name, result in results.items():
        status_icon = "✅" if result['status'] == 'PASS' else "❌"
        print(f"{status_icon} {test_name}: {result['status']}")
        if result['status'] != 'PASS':
            print(f"    {result['reason']}")
    
    return results

if __name__ == "__main__":
    test_thresholding_hls()