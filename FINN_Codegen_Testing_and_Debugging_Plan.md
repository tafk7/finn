# FINN Codegen Testing and Debugging Plan
**Using A/B Testing Framework for MVAU and Thresholding Validation**

## Executive Summary

This plan outlines a systematic approach to validate the clean MVAU and Thresholding codegen implementations using the comprehensive A/B testing framework. The goal is to ensure functional equivalence, performance improvements, and code quality while providing debugging strategies for any issues discovered.

## 1. Testing Framework Overview

### 1.1 Available Testing Infrastructure ✅
- **CodegenValidator**: 463-line comprehensive validation framework
- **Test Suite**: 10 detailed scenarios (6 for MVAU/Thresholding)
- **CG_BackendRegistry**: A/B testing support with clean/legacy switching
- **Pattern Validation**: Expected code pattern verification
- **Performance Metrics**: Generation time and template processing comparison

### 1.2 Test Scenarios Available
**Thresholding Tests (3 scenarios):**
- `thresholding_hls_basic` - Basic configuration validation
- `thresholding_hls_high_pe` - High parallelism (PE=16) testing
- `thresholding_hls_timeout` - Timeout and streaming interface validation

**MVAU Tests (3 scenarios):**
- `mvau_hls_embedded` - Internal embedded memory mode
- `mvau_hls_streaming` - Streaming/decoupled memory mode  
- `mvau_hls_external_weights` - External memory weight storage

## 2. Phase 1: Baseline Establishment ✅ COMPLETED

### 2.1 Create Reference Implementation ✅ COMPLETED
**Objective**: Establish known-good baseline for comparison

**Actions**:
1. **Backup current clean implementations** ✅
   ```bash
   git branch clean-implementations-backup  # Created backup branch
   git checkout -b testing-baseline        # Created testing branch
   ```

2. **Restore legacy implementations temporarily for baseline** ✅
   ```bash
   git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py
   git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/hls/matrixvectoractivation_hls.py
   ```

3. **Run baseline validation tests** ✅ COMPLETED
   ```bash
   ./run-docker.sh python src/finn/codegen/baseline_validation.py
   ```

### 2.2 Baseline Success Criteria ✅ VALIDATED
- ✅ **Functional Generation**: Template processing operational (10/10 success)
- ✅ **Template Processing**: No template rendering errors (100% success rate)
- ✅ **Pattern Coverage**: Framework ready for pattern validation
- ✅ **Performance Baseline**: Generation time metrics framework operational

**Baseline Results Summary:**
- **Testing Framework**: 100% operational
- **Template System**: All 10 test scenarios process correctly
- **Validation Infrastructure**: Ready for A/B testing
- **Docker Environment**: Successfully running validation scripts

## 3. Phase 2: Clean Implementation Validation ⏳ IN PROGRESS

### 3.1 Systematic Testing Approach ⏳ EXECUTING

#### Step 1: Individual Operation Testing ❌ CRITICAL FAILURES
**Test each operation in isolation**

```bash
# Test Thresholding HLS specifically - FAILED ❌
./run-docker.sh python src/finn/codegen/test_thresholding.py
```

**CRITICAL RESULTS: ALL TESTS FAILED ❌**
- ❌ **Passed: 0/3 (0.0%)** - Complete failure
- ❌ **Functional equivalence failed** on all tests
- ❌ **Pattern coverage: 0.0%** - No expected patterns found
- ❌ **Implementation broken** - Requires immediate debugging

**Failed Tests:**
- `thresholding_hls_basic`: FAIL
- `thresholding_hls_high_pe`: FAIL
- `thresholding_hls_timeout`: FAIL

**EMERGENCY DEBUGGING REQUIRED** ⚠️

#### Step 2: Immediate Debugging ⏳ IN PROGRESS
**Debug critical failures using debugging framework**

```bash
# Debug individual test failure
./run-docker.sh python src/finn/codegen/debug_test_failure.py thresholding_hls_basic
```

### 3.2 Validation Checkpoints

#### 3.2.1 Code Generation Validation
**Verify generated code correctness**

- **Syntax Validation**: Generated C++/Verilog compiles without errors
- **Template Completeness**: All placeholders properly substituted
- **Include Correctness**: Required headers and dependencies included
- **Pragma Validation**: HLS pragmas correctly generated and positioned

#### 3.2.2 Functional Equivalence Testing
**Ensure behavioral compatibility**

- **Parameter Mapping**: Node attributes correctly mapped to template values
- **Stream Interface**: Input/output stream handling matches expectations
- **Memory Layout**: Weight/threshold storage patterns consistent
- **Control Logic**: Timeout and control flow generation accurate

#### 3.2.3 Pattern Verification
**Expected code patterns present**

**Thresholding Expected Patterns**:
- `Thresholding_Batch` function generation
- `#pragma HLS INTERFACE axis` for streaming
- `hls::stream` declarations
- `#include "activations.hpp"` header inclusion

**MVAU Expected Patterns**:
- `Matrix_Vector_Activate_Batch` function generation
- Parameter templates (`MW1`, `MH1`, `PE1`, `SIMD1`)
- Memory mode configurations (`internal_embedded`, `external`)
- Resource type specifications (`lut`, `dsp`, `auto`)

## 4. Phase 3: Performance & Quality Validation

### 4.1 Performance Benchmarking

#### 4.1.1 Generation Time Analysis
**Measure template processing performance**

```python
# Performance test script
import time
from codegen_validator import CodegenValidator

validator = CodegenValidator()

# Test performance for each operation
operations = ['Thresholding', 'MVAU'] 
backends = ['hls', 'rtl']

for op in operations:
    for backend in backends:
        start_time = time.time()
        result = validator.validate_backend(op, backend)
        end_time = time.time()
        
        print(f"{op} {backend}: {(end_time - start_time)*1000:.2f}ms")
```

#### 4.1.2 Performance Targets
**Expected improvements from clean implementation**

| Operation | Backend | Generation Time Target | Template Processing Target |
|-----------|---------|----------------------|---------------------------|
| Thresholding | HLS | 20-30% improvement | 40-50% improvement |
| MVAU | HLS | 35-40% improvement | 60-65% improvement |
| Thresholding | RTL | 15-25% improvement | 25-35% improvement |
| MVAU | RTL | 25-35% improvement | 35-45% improvement |

### 4.2 Code Quality Metrics

#### 4.2.1 Template Utilization Analysis
**Measure template system efficiency**

- **Template Reuse**: Verify component templates properly utilized
- **Duplication Elimination**: Confirm 80-95% duplication reduction achieved
- **Maintainability Score**: Assess code complexity and readability

#### 4.2.2 Structural Analysis
**Analyze generated code structure**

- **Lines of Code**: Compare generated code size
- **Complexity Metrics**: Cyclomatic complexity of generated functions
- **Dependency Analysis**: External dependency requirements

## 5. Phase 4: Debugging and Issue Resolution

### 5.1 Common Issue Categories

#### 5.1.1 Template Issues
**Symptoms**: Missing placeholders, template rendering errors

**Debugging Approach**:
```python
# Template debugging script
from jinja2 import Environment, FileSystemLoader, meta

# Analyze template dependencies
env = Environment(loader=FileSystemLoader('src/finn/codegen/templates'))
template = env.get_template('base/hls_base.cpp.j2')
ast = env.parse(template.source)
variables = meta.find_undeclared_variables(ast)
print("Required template variables:", variables)
```

**Resolution Strategy**:
1. Verify all template variables provided by `get_template_values()`
2. Check template inheritance and macro usage
3. Validate Jinja2 syntax and logic

#### 5.1.2 Parameter Mapping Issues
**Symptoms**: Incorrect node attribute translation

**Debugging Approach**:
```python
# Parameter mapping debugging
def debug_template_values(operation_name, node_attributes):
    # Import clean implementation
    if operation_name == 'Thresholding':
        from hls.thresholding_hls import Thresholding_hls as CleanOp
    elif operation_name == 'MVAU':
        from hls.matrixvectoractivation_hls import MatrixVectorActivation_hls as CleanOp
    
    # Create instance and get template values
    op_instance = CleanOp()
    op_instance.onnx_node.attribute = node_attributes
    
    template_values = op_instance.get_template_values()
    print("Generated template values:")
    for key, value in template_values.items():
        print(f"  {key}: {value}")
    
    return template_values
```

**Resolution Strategy**:
1. Trace node attribute → template value mapping
2. Verify `_generate_*_values()` methods
3. Check inheritance chain and method resolution

#### 5.1.3 Code Generation Issues
**Symptoms**: Syntactically incorrect or incomplete generated code

**Debugging Approach**:
```python
# Code generation debugging
def debug_code_generation(operation_name, backend_type):
    from codegen_validator import CodegenValidator
    
    validator = CodegenValidator()
    result = validator.validate_backend(operation_name, backend_type)
    
    print("Clean Implementation Output:")
    print("=" * 50)
    print(result.clean_output[:1000] + "..." if len(result.clean_output) > 1000 else result.clean_output)
    
    print("\nTemplate Values Used:")
    print("=" * 50)
    # Extract template values from debug output
    
    return result
```

**Resolution Strategy**:
1. Compare generated code against expected patterns
2. Verify template logic and conditional blocks
3. Check for missing includes or declarations

### 5.2 Systematic Debugging Workflow

#### Step 1: Identify Failure Category
```bash
# Run specific test to isolate issue
cd src/finn/codegen
python -c "
from test_suite import CodegenTestSuite
suite = CodegenTestSuite()
test = next(t for t in suite.test_cases if t.name == 'thresholding_hls_basic')
result = suite.run_single_test(test)
print('Failure reason:', result.comparison_summary)
"
```

#### Step 2: Isolate Root Cause
```python
# Detailed debugging script
def debug_specific_test(test_name):
    # 1. Check template value generation
    template_values = debug_template_values(...)
    
    # 2. Check template rendering
    rendered_code = debug_template_rendering(...)
    
    # 3. Check pattern matching
    pattern_matches = debug_pattern_matching(...)
    
    # 4. Generate diagnostic report
    return generate_debug_report(...)
```

#### Step 3: Apply Fix and Revalidate
```bash
# After applying fix, run regression test
python run_validation.py --test-specific thresholding_hls_basic
```

## 6. Phase 5: Comprehensive Integration Testing

### 6.1 End-to-End Validation

#### 6.1.1 Full Test Suite Execution
```bash
cd src/finn/codegen
python run_validation.py --comprehensive --report-file validation_report.md
```

#### 6.1.2 Cross-Operation Testing
**Verify operations work together in complex scenarios**

```python
# Multi-operation test scenario
def test_complex_model():
    # Create model with both Thresholding and MVAU operations
    # Test codegen for entire flow
    # Verify no conflicts or issues
    pass
```

### 6.2 Success Criteria for Go-Live

#### 6.2.1 Functional Criteria
- ✅ **100% test pass rate** for core MVAU and Thresholding scenarios
- ✅ **Functional equivalence** confirmed for all operations
- ✅ **Pattern coverage ≥ 90%** for expected code patterns
- ✅ **Zero template rendering errors**

#### 6.2.2 Performance Criteria
- ✅ **Generation time improvement ≥ 20%** for all operations
- ✅ **Template processing improvement ≥ 30%** average
- ✅ **Code size reduction or equivalent** (no bloat increase)

#### 6.2.3 Quality Criteria
- ✅ **Generated code compiles** without warnings/errors
- ✅ **No regression** in existing functionality
- ✅ **Documentation complete** for any discovered limitations

## 7. Ongoing Validation Strategy

### 7.1 Continuous Testing
**Integrate validation into development workflow**

```bash
# Pre-commit validation hook
#!/bin/bash
cd src/finn/codegen
python run_validation.py --quick-check
if [ $? -ne 0 ]; then
    echo "Codegen validation failed - commit blocked"
    exit 1
fi
```

### 7.2 Regression Testing
**Maintain test coverage as system evolves**

- **Test Suite Expansion**: Add new scenarios as operations are added
- **Performance Benchmarking**: Track performance trends over time
- **Pattern Library**: Maintain expected pattern database

## 8. Risk Assessment and Mitigation

### 8.1 High Risk Areas
1. **Template Complex Logic**: Conditional blocks and loops in templates
2. **Parameter Edge Cases**: Unusual or extreme node attribute values
3. **Cross-Operation Dependencies**: Shared template components
4. **Performance Regressions**: Unintended slowdowns in clean implementation

### 8.2 Mitigation Strategies
1. **Comprehensive Edge Case Testing**: Test boundary conditions
2. **Rollback Plan**: Maintain ability to quickly revert to legacy if needed
3. **Incremental Deployment**: Enable clean backends gradually per operation
4. **Monitoring**: Track performance and correctness metrics in production

## 9. Execution Timeline

### Week 1: Baseline and Initial Testing
- **Days 1-2**: Baseline establishment and framework setup
- **Days 3-5**: Individual operation testing and initial debugging

### Week 2: Validation and Debugging
- **Days 1-3**: Comprehensive A/B testing and issue resolution  
- **Days 4-5**: Performance validation and optimization

### Week 3: Integration and Sign-off
- **Days 1-2**: End-to-end integration testing
- **Days 3-5**: Final validation, documentation, and go-live approval

## 10. Success Metrics and Reporting

### 10.1 Key Performance Indicators
- **Test Pass Rate**: Target 100% for core scenarios
- **Performance Improvement**: Target 20-40% across operations
- **Code Quality Score**: Maintain or improve complexity metrics
- **Issue Resolution Time**: Track debugging and fix efficiency

### 10.2 Reporting Framework
**Generate comprehensive validation reports**

```bash
# Generate final validation report
cd src/finn/codegen  
python run_validation.py --generate-report --output validation_final_report.md
```

---

**This comprehensive testing and debugging plan ensures the MVAU and Thresholding codegen implementations are thoroughly validated, performant, and production-ready while providing systematic approaches for identifying and resolving any issues discovered during testing.**