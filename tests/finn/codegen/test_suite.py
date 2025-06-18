"""
FINN Codegen Comprehensive Test Suite
Provides test cases and scenarios for validating clean implementations against legacy ones.
"""

import os
import tempfile
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from codegen_validator import CodegenValidator, ValidationResult


@dataclass
class TestCase:
    """Test case definition for codegen validation."""
    name: str
    operation_type: str
    backend_type: str
    node_attributes: Dict[str, Any]
    test_model: Optional[Any] = None
    expected_patterns: List[str] = None
    performance_targets: Dict[str, float] = None


class CodegenTestSuite:
    """Comprehensive test suite for FINN codegen validation."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.logger = logging.getLogger("finn.codegen.test_suite")
        self.validator = CodegenValidator()
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases for validation."""
        test_cases = []
        
        # Thresholding HLS test cases
        test_cases.extend(self._create_thresholding_hls_tests())
        
        # MVAU HLS test cases
        test_cases.extend(self._create_mvau_hls_tests())
        
        # Thresholding RTL test cases
        test_cases.extend(self._create_thresholding_rtl_tests())
        
        # MVAU RTL test cases
        test_cases.extend(self._create_mvau_rtl_tests())
        
        return test_cases
    
    def _create_thresholding_hls_tests(self) -> List[TestCase]:
        """Create test cases for Thresholding HLS operations."""
        return [
            TestCase(
                name="thresholding_hls_basic",
                operation_type="Thresholding",
                backend_type="hls",
                node_attributes={
                    "NumChannels": 32,
                    "PE": 4,
                    "NumSteps": 8,
                    "ram_style": "block",
                    "numInputVectors": [1, 32]
                },
                expected_patterns=[
                    "Thresholding_Batch",
                    "#pragma HLS INTERFACE axis",
                    "hls::stream",
                    "#include \"activations.hpp\""
                ],
                performance_targets={
                    "generation_time_improvement_pct": 20.0,  # Expect 20% improvement
                    "template_time_improvement_pct": 40.0,    # Expect 40% improvement
                }
            ),
            TestCase(
                name="thresholding_hls_high_pe",
                operation_type="Thresholding",
                backend_type="hls",
                node_attributes={
                    "NumChannels": 128,
                    "PE": 16,
                    "NumSteps": 16,
                    "ram_style": "distributed",
                    "numInputVectors": [1, 128]
                },
                expected_patterns=[
                    "PE1 16",
                    "NumChannels1 128",
                    "#pragma HLS ARRAY_PARTITION"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 25.0,
                    "template_time_improvement_pct": 45.0,
                }
            ),
            TestCase(
                name="thresholding_hls_timeout",
                operation_type="Thresholding",
                backend_type="hls",
                node_attributes={
                    "NumChannels": 64,
                    "PE": 8,
                    "NumSteps": 4,
                    "cpp_interface": "hls_vector",
                    "numInputVectors": [1, 64]
                },
                expected_patterns=[
                    "timeout",
                    "out0_V.empty()",
                    "strm << out0_V.read()"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 30.0,
                    "template_time_improvement_pct": 50.0,
                }
            )
        ]
    
    def _create_mvau_hls_tests(self) -> List[TestCase]:
        """Create test cases for MVAU HLS operations."""
        return [
            TestCase(
                name="mvau_hls_embedded",
                operation_type="MVAU",
                backend_type="hls",
                node_attributes={
                    "MW": 64,
                    "MH": 32,
                    "PE": 4,
                    "SIMD": 8,
                    "mem_mode": "internal_embedded",
                    "resType": "lut",
                    "numInputVectors": [1]
                },
                expected_patterns=[
                    "Matrix_Vector_Activate_Batch",
                    "MW1 64",
                    "MH1 32",
                    "PE1 4",
                    "SIMD1 8"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 35.0,
                    "template_time_improvement_pct": 60.0,
                }
            ),
            TestCase(
                name="mvau_hls_streaming",
                operation_type="MVAU",
                backend_type="hls",
                node_attributes={
                    "MW": 128,
                    "MH": 64,
                    "PE": 8,
                    "SIMD": 16,
                    "mem_mode": "internal_decoupled",
                    "resType": "dsp",
                    "numInputVectors": [1]
                },
                expected_patterns=[
                    "Matrix_Vector_Activate_Stream_Batch",
                    "in1_V",
                    "#pragma HLS INTERFACE axis port=in1_V"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 40.0,
                    "template_time_improvement_pct": 65.0,
                }
            ),
            TestCase(
                name="mvau_hls_external_weights",
                operation_type="MVAU",
                backend_type="hls",
                node_attributes={
                    "MW": 256,
                    "MH": 128,
                    "PE": 16,
                    "SIMD": 32,
                    "mem_mode": "external",
                    "resType": "auto",
                    "numInputVectors": [1]
                },
                expected_patterns=[
                    "external",
                    "WP1",
                    "ap_resource_dflt()"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 30.0,
                    "template_time_improvement_pct": 55.0,
                }
            )
        ]
    
    def _create_thresholding_rtl_tests(self) -> List[TestCase]:
        """Create test cases for Thresholding RTL operations."""
        return [
            TestCase(
                name="thresholding_rtl_basic",
                operation_type="Thresholding",
                backend_type="rtl",
                node_attributes={
                    "NumChannels": 32,
                    "PE": 4,
                    "NumSteps": 8,
                    "inputDataType": "INT8",
                    "outputDataType": "INT4"
                },
                expected_patterns=[
                    "module",
                    "input wire clk",
                    "input wire rst",
                    "axi_stream"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 15.0,
                    "template_time_improvement_pct": 25.0,
                }
            ),
            TestCase(
                name="thresholding_rtl_high_throughput",
                operation_type="Thresholding",
                backend_type="rtl",
                node_attributes={
                    "NumChannels": 128,
                    "PE": 32,
                    "NumSteps": 16,
                    "inputDataType": "INT4",
                    "outputDataType": "INT2"
                },
                expected_patterns=[
                    "PE_COUNT",
                    "DATA_WIDTH",
                    "always @(posedge clk)"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 20.0,
                    "template_time_improvement_pct": 30.0,
                }
            )
        ]
    
    def _create_mvau_rtl_tests(self) -> List[TestCase]:
        """Create test cases for MVAU RTL operations."""
        return [
            TestCase(
                name="mvau_rtl_basic",
                operation_type="MVAU",
                backend_type="rtl",
                node_attributes={
                    "MW": 64,
                    "MH": 32,
                    "PE": 4,
                    "SIMD": 8,
                    "inputDataType": "INT8",
                    "weightDataType": "INT4",
                    "outputDataType": "INT8"
                },
                expected_patterns=[
                    "module",
                    "MATRIX_WIDTH",
                    "MATRIX_HEIGHT",
                    "PE_COUNT"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 25.0,
                    "template_time_improvement_pct": 35.0,
                }
            ),
            TestCase(
                name="mvau_rtl_dsp_optimized",
                operation_type="MVAU",
                backend_type="rtl",
                node_attributes={
                    "MW": 128,
                    "MH": 64,
                    "PE": 16,
                    "SIMD": 16,
                    "resType": "dsp",
                    "inputDataType": "INT16",
                    "weightDataType": "INT8",
                    "outputDataType": "INT16"
                },
                expected_patterns=[
                    "DSP",
                    "dsp_block",
                    "multiply"
                ],
                performance_targets={
                    "generation_time_improvement_pct": 30.0,
                    "template_time_improvement_pct": 40.0,
                }
            )
        ]
    
    def run_single_test(self, test_case: TestCase) -> ValidationResult:
        """
        Run a single test case and return validation result.
        
        Args:
            test_case: Test case to run
            
        Returns:
            ValidationResult from the comparison
        """
        self.logger.info(f"Running test case: {test_case.name}")
        
        try:
            # Create a simple validation result structure
            result = self.validator.validate_backend(
                test_case.operation_type,
                test_case.backend_type,
                test_model=test_case.test_model
            )
            
            # Adapt to CodegenComparison object structure
            passed = True
            summary_parts = []
            
            # Check functional equivalence
            if hasattr(result, 'functional_equivalent'):
                if not result.functional_equivalent:
                    passed = False
                    summary_parts.append("Functional equivalence failed")
            
            # Check expected patterns if specified
            if test_case.expected_patterns:
                pattern_matches = 0
                clean_output = getattr(result, 'clean_output', '')
                for pattern in test_case.expected_patterns:
                    if pattern in str(clean_output):
                        pattern_matches += 1
                
                pattern_coverage = pattern_matches / len(test_case.expected_patterns) if test_case.expected_patterns else 1.0
                if pattern_coverage < 0.8:  # Require 80% pattern match
                    passed = False
                    summary_parts.append(f"Pattern coverage: {pattern_coverage:.1%}")
            
            # Check performance targets if specified
            if test_case.performance_targets and hasattr(result, 'performance_delta'):
                for metric, target in test_case.performance_targets.items():
                    if metric in result.performance_delta:
                        actual = result.performance_delta[metric]
                        if actual < target:  # Performance didn't meet target
                            passed = False
                            summary_parts.append(f"{metric} below target")
            
            # Create a wrapper result that matches expected interface
            class TestResult:
                def __init__(self, result_obj, passed, summary):
                    self.validation_passed = passed
                    self.comparison_summary = summary or "Test completed"
                    self.clean_code = getattr(result_obj, 'clean_output', '')
                    self.legacy_code = getattr(result_obj, 'legacy_output', '')
                    self.differences = getattr(result_obj, 'differences', [])
                    self.performance_comparison = getattr(result_obj, 'performance_delta', {})
                    self.semantic_equivalence = getattr(result_obj, 'functional_equivalent', False)
                    self.structural_similarity = getattr(result_obj, 'structural_similarity_score', 0.0)
            
            return TestResult(result, passed, " | ".join(summary_parts))
            
        except Exception as e:
            self.logger.error(f"Error running test {test_case.name}: {e}")
            # Return a simple failed result - create a mock result object
            class MockResult:
                def __init__(self):
                    self.validation_passed = False
                    self.comparison_summary = f"Test execution failed: {str(e)}"
                    self.clean_code = ""
                    self.legacy_code = ""
                    self.differences = []
                    self.performance_comparison = {}
                    self.semantic_equivalence = False
                    self.structural_similarity = 0.0
            
            return MockResult()
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all test cases and collect results."""
        self.logger.info(f"Running {len(self.test_cases)} comprehensive test cases")
        
        results = {}
        passed_tests = 0
        failed_tests = 0
        performance_met = 0
        
        for test_case in self.test_cases:
            self.logger.info(f"Running test case: {test_case.name}")
            
            try:
                # Run validation
                comparison = self.validator.validate_backend(
                    test_case.operation_type,
                    test_case.backend_type,
                    test_model=test_case.test_model
                )
                
                # Check expected patterns
                pattern_results = self._check_expected_patterns(comparison, test_case)
                
                # Check performance targets
                performance_results = self._check_performance_targets(comparison, test_case)
                
                # Determine overall test result
                test_passed = (
                    comparison.validation_result == ValidationResult.PASS and
                    pattern_results['all_patterns_found'] and
                    performance_results['targets_met']
                )
                
                if test_passed:
                    passed_tests += 1
                else:
                    failed_tests += 1
                
                if performance_results['targets_met']:
                    performance_met += 1
                
                results[test_case.name] = {
                    'comparison': comparison,
                    'patterns': pattern_results,
                    'performance': performance_results,
                    'test_passed': test_passed
                }
                
            except Exception as e:
                self.logger.error(f"Test case {test_case.name} failed with error: {e}")
                failed_tests += 1
                results[test_case.name] = {
                    'error': str(e),
                    'test_passed': False
                }
        
        # Generate summary
        summary = {
            'total_tests': len(self.test_cases),
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate_pct': (passed_tests / len(self.test_cases)) * 100,
            'performance_targets_met': performance_met,
            'performance_success_rate_pct': (performance_met / len(self.test_cases)) * 100,
            'detailed_results': results
        }
        
        return summary
    
    def _check_expected_patterns(self, comparison, test_case) -> Dict[str, Any]:
        """Check if expected patterns are present in generated code."""
        if not test_case.expected_patterns:
            return {'all_patterns_found': True, 'pattern_results': {}}
        
        pattern_results = {}
        for pattern in test_case.expected_patterns:
            found_in_clean = pattern in comparison.clean_output
            found_in_legacy = pattern in comparison.legacy_output
            pattern_results[pattern] = {
                'found_in_clean': found_in_clean,
                'found_in_legacy': found_in_legacy,
                'consistent': found_in_clean == found_in_legacy
            }
        
        all_patterns_found = all(
            result['found_in_clean'] for result in pattern_results.values()
        )
        
        return {
            'all_patterns_found': all_patterns_found,
            'pattern_results': pattern_results
        }
    
    def _check_performance_targets(self, comparison, test_case) -> Dict[str, Any]:
        """Check if performance targets are met."""
        if not test_case.performance_targets:
            return {'targets_met': True, 'target_results': {}}
        
        target_results = {}
        for metric, target_value in test_case.performance_targets.items():
            actual_value = comparison.performance_delta.get(metric, 0.0)
            target_met = actual_value >= target_value
            target_results[metric] = {
                'target': target_value,
                'actual': actual_value,
                'met': target_met,
                'delta': actual_value - target_value
            }
        
        targets_met = all(result['met'] for result in target_results.values())
        
        return {
            'targets_met': targets_met,
            'target_results': target_results
        }
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("# FINN Codegen Comprehensive Test Suite Report")
        report.append(f"**Test Run Date**: {self._get_timestamp()}")
        report.append("")
        
        # Summary
        report.append("## Test Summary")
        report.append(f"- **Total Tests**: {results['total_tests']}")
        report.append(f"- **Passed**: {results['passed_tests']}")
        report.append(f"- **Failed**: {results['failed_tests']}")
        report.append(f"- **Pass Rate**: {results['pass_rate_pct']:.1f}%")
        report.append(f"- **Performance Targets Met**: {results['performance_targets_met']}/{results['total_tests']}")
        report.append(f"- **Performance Success Rate**: {results['performance_success_rate_pct']:.1f}%")
        report.append("")
        
        # Detailed results by category
        categories = {
            'Thresholding HLS': [k for k in results['detailed_results'].keys() if 'thresholding_hls' in k],
            'MVAU HLS': [k for k in results['detailed_results'].keys() if 'mvau_hls' in k],
            'Thresholding RTL': [k for k in results['detailed_results'].keys() if 'thresholding_rtl' in k],
            'MVAU RTL': [k for k in results['detailed_results'].keys() if 'mvau_rtl' in k],
        }
        
        for category, test_names in categories.items():
            if test_names:
                report.append(f"## {category} Tests")
                
                for test_name in test_names:
                    test_result = results['detailed_results'][test_name]
                    status = "✅" if test_result.get('test_passed', False) else "❌"
                    report.append(f"### {status} {test_name}")
                    
                    if 'error' in test_result:
                        report.append(f"- **Error**: {test_result['error']}")
                    else:
                        comparison = test_result['comparison']
                        patterns = test_result['patterns']
                        performance = test_result['performance']
                        
                        # Functional equivalence
                        equiv_status = "✅" if comparison.functional_equivalent else "❌"
                        report.append(f"- **Functional Equivalence**: {equiv_status}")
                        
                        # Pattern verification
                        pattern_status = "✅" if patterns['all_patterns_found'] else "❌"
                        report.append(f"- **Expected Patterns**: {pattern_status}")
                        
                        # Performance targets
                        perf_status = "✅" if performance['targets_met'] else "❌"
                        report.append(f"- **Performance Targets**: {perf_status}")
                        
                        # Show key improvements
                        for metric, value in comparison.performance_delta.items():
                            if 'improvement' in metric and value > 0:
                                report.append(f"  - {metric}: {value:.1f}% improvement")
                    
                    report.append("")
        
        return "\n".join(report)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')


def run_comprehensive_test_suite():
    """Run the comprehensive test suite and generate report."""
    test_suite = CodegenTestSuite()
    
    # Run all tests
    results = test_suite.run_comprehensive_tests()
    
    # Generate report
    report = test_suite.generate_test_report(results)
    
    # Save report
    report_path = "FINN_Codegen_Comprehensive_Test_Report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Comprehensive test suite complete. Report saved to {report_path}")
    print(f"Pass rate: {results['pass_rate_pct']:.1f}%")
    print(f"Performance success rate: {results['performance_success_rate_pct']:.1f}%")
    
    return results


if __name__ == "__main__":
    run_comprehensive_test_suite()