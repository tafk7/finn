# FINN Codegen Testing Quick Reference
**Fast Access Guide for MVAU and Thresholding Validation**

## Quick Start Commands

### 1. Run Full Validation Suite
```bash
cd src/finn/codegen
python run_validation.py
```

### 2. Test Specific Operation
```bash
# Test only Thresholding HLS
python -c "
from test_suite import CodegenTestSuite
suite = CodegenTestSuite()
thresholding_tests = [t for t in suite.test_cases if 'thresholding_hls' in t.name]
for test in thresholding_tests:
    result = suite.run_single_test(test)
    print(f'{test.name}: {\"PASS\" if result.validation_passed else \"FAIL\"}')"
```

### 3. Debug Failed Test
```bash
# Replace TEST_NAME with actual failing test
python -c "
from test_suite import CodegenTestSuite
suite = CodegenTestSuite()
test = next(t for t in suite.test_cases if t.name == 'TEST_NAME')
result = suite.run_single_test(test)
print('Failure reason:', result.comparison_summary)
print('Generated code preview:', result.clean_code[:500] + '...' if len(result.clean_code) > 500 else result.clean_code)
"
```

## Test Scenarios Available

### Thresholding HLS Tests
| Test Name | Configuration | Focus Area |
|-----------|---------------|------------|
| `thresholding_hls_basic` | PE=4, NumChannels=32 | Basic functionality |
| `thresholding_hls_high_pe` | PE=16, NumChannels=128 | High parallelism |
| `thresholding_hls_timeout` | PE=8, cpp_interface=hls_vector | Timeout handling |

### MVAU HLS Tests  
| Test Name | Configuration | Focus Area |
|-----------|---------------|------------|
| `mvau_hls_embedded` | PE=4, SIMD=8, internal_embedded | Embedded memory |
| `mvau_hls_streaming` | PE=8, SIMD=16, internal_decoupled | Streaming mode |
| `mvau_hls_external_weights` | PE=16, SIMD=32, external | External memory |

## Expected Performance Improvements

| Operation | Backend | Generation Time | Template Processing |
|-----------|---------|----------------|-------------------|
| Thresholding | HLS | 20-30% faster | 40-50% faster |
| MVAU | HLS | 35-40% faster | 60-65% faster |
| Thresholding | RTL | 15-25% faster | 25-35% faster |
| MVAU | RTL | 25-35% faster | 35-45% faster |

## Critical Code Patterns to Verify

### Thresholding HLS Patterns
- ✅ `Thresholding_Batch` function name
- ✅ `#pragma HLS INTERFACE axis` for streaming
- ✅ `hls::stream` declarations
- ✅ `#include "activations.hpp"` header

### MVAU HLS Patterns  
- ✅ `Matrix_Vector_Activate_Batch` function name
- ✅ Parameter templates (`MW1`, `MH1`, `PE1`, `SIMD1`)
- ✅ Memory mode configurations (`internal_embedded`, `external`)
- ✅ Resource type specifications (`lut`, `dsp`, `auto`)

## Common Failure Diagnostics

### Template Value Errors
**Symptom**: `KeyError` or missing placeholders
**Quick Fix**: Check node attributes in test case match operation requirements

### Pattern Coverage Failures  
**Symptom**: "Pattern coverage: X.X%" in failure message
**Quick Fix**: Verify template logic generates expected code patterns

### Functional Equivalence Failures
**Symptom**: "Functional equivalence failed" 
**Quick Fix**: Compare clean vs legacy output for structural differences

## Debugging Scripts Location

All debugging utilities are documented in:
- `FINN_Codegen_Debugging_Workflows.md` - Complete debugging procedures
- Utility scripts can be found in the workflows document

## Emergency Rollback Plan

If clean implementations cause critical issues:

```bash
# Rollback to legacy implementations
git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/hlsbackend.py
git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/rtlbackend.py  
git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py
git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/hls/matrixvectoractivation_hls.py
git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py
git restore --source=HEAD~2 src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py
```

## Success Criteria Checklist

### Functional Requirements
- [ ] 100% test pass rate for core scenarios
- [ ] All expected code patterns present (≥90% coverage)
- [ ] Generated code compiles without errors
- [ ] No functional regressions vs legacy

### Performance Requirements  
- [ ] Generation time improvement ≥20% average
- [ ] Template processing improvement ≥30% average
- [ ] No significant code size increase

### Quality Requirements
- [ ] Zero template rendering errors
- [ ] Clean code follows architectural patterns
- [ ] Documentation complete for any limitations
- [ ] Debug procedures validated and working

## Support Resources

1. **Primary Documentation**:
   - `FINN_Codegen_Testing_and_Debugging_Plan.md` - Complete testing strategy
   - `FINN_Codegen_Debugging_Workflows.md` - Detailed debugging procedures

2. **Validation Framework**:
   - `src/finn/codegen/codegen_validator.py` - Core validation logic
   - `src/finn/codegen/test_suite.py` - Test scenarios and execution
   - `src/finn/codegen/run_validation.py` - Simple test runner

3. **Reference Results**:
   - `src/finn/codegen/VALIDATION_RESULTS.md` - Framework operational status
   - `FINN_Codegen_Implementation_Checklist.md` - Complete project status

---
**For detailed procedures, refer to the comprehensive testing and debugging documentation.**