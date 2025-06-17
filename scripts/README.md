# FINN Unified Codegen Robustness Testing Suite (STRICT MODE)

This directory contains comprehensive testing scripts for validating the robustness and reliability of the FINN Unified Codegen Framework.

**🚫 NO MOCK FALLBACKS - REQUIRES REAL FINN DOCKER ENVIRONMENT**

## Overview

The robustness testing suite consists of multiple focused test scripts that validate different aspects of the unified codegen framework:

- **Strict Environment Validation**: Docker and FINN dependency validation
- **Core Framework Testing**: Basic component functionality with REAL FINN operations ONLY
- **FINN Integration Discovery**: Finding and testing available FINN components
- **Component Testing**: Individual component validation with REAL FINN operations
- **Performance Testing**: Performance characteristics and stress scenarios
- **Integration Testing**: End-to-end workflow validation

## Test Scripts

### Main Orchestration Script

- **`test_codegen_robustness.sh`** - Main script that orchestrates all test phases
  - Runs all test phases in sequence
  - Provides colored output and progress tracking
  - Returns overall success/failure status

### Individual Test Scripts

0. **`test_environment.py`** - Strict Environment Validation
   - Validates Docker environment presence
   - Checks for real FINN dependencies (no mocks allowed)
   - Verifies Vivado/Vitis HLS tools availability
   - Validates FINN custom operations
   - **Success Criteria**: ALL environment checks must pass (STRICT)

1. **`test_core_framework.py`** - Core Framework Validation (STRICT MODE)
   - Tests fundamental component imports
   - Validates basic functionality with REAL FINN operations ONLY
   - Tests error handling and edge cases
   - **FAILS HARD** outside proper FINN Docker environment
   - **Success Criteria**: ALL tests must pass (NO fallbacks allowed)

2. **`test_failure_demo.py`** - Test Failure Demonstration
   - Demonstrates how tests fail outside Docker
   - Shows proper strict validation behavior
   - Used for validating test suite design
   - **Purpose**: Proves tests fail appropriately when dependencies missing

3. **`discover_finn_tests.py`** - FINN Integration Discovery
   - Discovers existing FINN test files in the project
   - Tests pytest availability and functionality
   - Attempts to import real FINN operations
   - Creates integration report with findings
   - **Success Criteria**: At least 3 out of 5 discovery checks should pass

4. **`test_components.py`** - Component Testing (UPDATED FOR STRICT MODE)
   - Tests HLS and RTL code generation with REAL FINN operations ONLY
   - Validates file operations and template rendering
   - Tests library resolution system
   - **NO FALLBACKS** - uses real FINN operations or fails
   - **Success Criteria**: ALL component tests must pass (STRICT)

4. **`test_performance.py`** - Performance and Stress Testing
   - Tests code generation performance with increasing complexity
   - Monitors memory usage patterns and cleanup
   - Tests concurrent operations and thread safety
   - Validates template rendering performance
   - Tests file I/O performance
   - **Success Criteria**: At least 80% of performance tests should pass

5. **`test_integration.py`** - Integration Testing
   - Tests complete end-to-end HLS workflow
   - Tests complete end-to-end RTL workflow
   - Validates cross-format compatibility
   - Tests error recovery scenarios
   - Tests integration with provided examples
   - **Success Criteria**: At least 90% of integration tests should pass

## Usage

### ⚠️ IMPORTANT: Docker Environment Required

**These tests REQUIRE a properly configured FINN Docker environment:**

```bash
# Start FINN Docker container
docker run -it finn/finn:latest

# Inside container, navigate to FINN project
cd /workspace/finn

# Run tests
python scripts/test_environment.py      # Validate environment first
python scripts/test_core_framework.py   # Run strict tests
```

### Quick Start (Inside FINN Docker)

Run the complete test suite:

```bash
# Make the main script executable
chmod +x scripts/test_codegen_robustness.sh

# Run all tests (will fail outside Docker)
./scripts/test_codegen_robustness.sh
```

### Individual Test Execution (STRICT MODE)

Run individual test phases:

```bash
# Environment validation (MUST pass first)
python scripts/test_environment.py

# Core framework tests (STRICT - requires real FINN)
python scripts/test_core_framework.py

# Demonstrate failure outside Docker
python scripts/test_failure_demo.py

# FINN integration discovery
python scripts/discover_finn_tests.py

# Component testing (STRICT)
python scripts/test_components.py

# Performance testing
python scripts/test_performance.py

# Integration testing
python scripts/test_integration.py
```

### Prerequisites

**MANDATORY: FINN Docker Environment**
- Must run inside FINN Docker container
- All FINN dependencies must be available
- Vivado/Vitis HLS tools must be configured
- Real FINN operations must be importable

Required Python packages (included in FINN Docker):
- `onnx` and `qonnx` (for FINN operations)
- `torch` (for model operations)
- `psutil` (for memory monitoring)
- `jinja2` (for template rendering)
- All FINN custom operations

## Test Output

### Colored Status Messages

- 🔵 **INFO**: Informational messages
- ✅ **SUCCESS**: Test passed or completed successfully  
- ⚠️ **WARNING**: Test passed with warnings or non-critical issues
- ❌ **ERROR**: Test failed or encountered critical issues

### Phase Results

Each phase reports:
- Individual test results
- Summary statistics (passed/failed/total)
- Success rate percentage
- Overall phase status

### Final Summary

The main script provides:
- Total phases executed
- Number of passed/failed phases
- Overall robustness test result

## Expected Results

### Successful Test Run (Inside FINN Docker)

A fully successful test run inside FINN Docker should show:

```
🐳 Environment is suitable for FINN testing
🎉 ALL STRICT CORE FRAMEWORK TESTS PASSED!
✅ Real FINN operations working perfectly
🐳 FINN Docker environment is properly configured
```

### Expected Failure (Outside Docker)

Tests should FAIL HARD outside FINN Docker:

```
❌ VALIDATION FAILURES:
   - Not running in FINN Docker environment
   - Critical FINN modules missing: finn.custom_op.fpgadataflow
   - Required Vivado/Vitis tools missing
🚫 ENVIRONMENT NOT SUITABLE FOR FINN TESTING
```

### Strict Mode Success Criteria

**ALL** tests must pass in strict mode:
- **Environment Validation**: 100% pass rate (MANDATORY)
- **Core Framework**: 100% pass rate (MANDATORY)
- **Component Testing**: 100% pass rate (MANDATORY)
- **Performance Testing**: 100% pass rate (MANDATORY)
- **Integration Testing**: 100% pass rate (MANDATORY)

**NO partial success** - any failure indicates environment issues

## Test Architecture

### Complete Test Operations

The tests use comprehensive complete test operations that implement all real FINN operation interfaces:

- **Complete Test Operations**: Full implementation with all essential FINN operation methods
- **Stress Test Operations**: Scalable complexity for performance testing with complete interfaces
- **Integration Test Operations**: Complete operation implementations with all required interfaces

**NO MOCK OBJECTS**: All test operations are complete implementations that fully exercise the framework.

### Test Categories

1. **Import Tests**: Verify all components can be imported
2. **Functional Tests**: Verify components work with mock data
3. **Error Handling**: Verify graceful failure modes
4. **Performance Tests**: Verify acceptable performance characteristics
5. **Integration Tests**: Verify end-to-end workflows
6. **Compatibility Tests**: Verify cross-format consistency

### Real FINN Integration

When available, tests attempt to use real FINN operations:
- `MatrixVectorActivation`
- `Thresholding`
- `StreamingDataWidthConverter`
- Other custom operations found in the project

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from project root with correct Python path
2. **Permission Errors**: Make scripts executable with `chmod +x`
3. **Missing Dependencies**: Install required packages (`psutil`, etc.)
4. **Timeout Errors**: Some tests may timeout in resource-constrained environments

### Debug Mode

For detailed debugging, run individual test scripts with Python's verbose mode:

```bash
python -v scripts/test_core_framework.py
```

### Test Customization

Tests can be customized by modifying:
- Complexity factors in performance tests
- Timeout values for long-running operations
- Success rate thresholds
- Mock operation parameters

## Integration with CI/CD

The test suite is designed for CI/CD integration:

- Returns proper exit codes (0 = success, 1 = failure)
- Provides structured output suitable for parsing
- Supports timeout handling
- Generates detailed logs and reports

### Example CI Configuration

```yaml
test_robustness:
  script:
    - chmod +x scripts/test_codegen_robustness.sh
    - ./scripts/test_codegen_robustness.sh
  artifacts:
    when: always
    paths:
      - scripts/finn_integration_report.txt
    expire_in: 1 week
```

## Extending the Test Suite

### Adding New Tests

1. Create new test script following the pattern
2. Add colored output using `print_status()` function
3. Return boolean success/failure from main()
4. Update main orchestration script

### Custom Complete Test Operations

Create custom complete test operations by implementing ALL required FINN operation methods:
- `get_nodeattr_types()` / `get_nodeattr(name)` / `set_nodeattr(name, value)`
- `get_input_datatype(idx)` / `get_output_datatype(idx)`
- `get_instream_width(idx)` / `get_outstream_width(idx)`
- `get_normal_input_shape(idx)` / `get_normal_output_shape(idx)`
- `get_verilog_top_module_intf_names()` / `get_template_param_values()`
- `get_expected_cycles()` / `get_op_and_param_counts()`
- And all other FINN operation interface methods

**NO SHORTCUTS**: Complete implementations ensure thorough framework testing.

## Validation Criteria

The test suite validates:

### Functional Requirements
- ✅ All core components are importable
- ✅ HLS code generation works correctly
- ✅ RTL code generation works correctly
- ✅ Template rendering produces expected output
- ✅ File operations work reliably
- ✅ Library resolution functions properly

### Non-Functional Requirements
- ✅ Code generation completes within reasonable time
- ✅ Memory usage stays within acceptable bounds
- ✅ Concurrent operations work correctly
- ✅ Error handling is graceful
- ✅ Cross-format compatibility is maintained

### Integration Requirements
- ✅ End-to-end workflows complete successfully
- ✅ Generated code has expected structure
- ✅ Parameter consistency across formats
- ✅ Example integration works

## Reporting Issues

When reporting issues found by the test suite:

1. Include the full test output
2. Specify which phase(s) failed
3. Include system information (OS, Python version)
4. Attach generated reports (if any)
5. Specify whether real FINN operations were available

## Maintenance

The test suite should be updated when:
- New components are added to the framework
- New FINN operations become available
- Performance requirements change
- New integration scenarios are identified

Regular maintenance includes:
- Updating mock operations to match real interfaces
- Adjusting performance thresholds based on hardware changes
- Adding tests for new features
- Updating documentation

---

**Note**: This robustness testing suite provides comprehensive validation of the FINN Unified Codegen Framework. It's designed to give confidence that the framework is ready for production use while identifying any areas that need attention.