# Plan: Remove Mocks and Enable Real FINN Integration

## Phase 1: Create Real FINN Operation Test Infrastructure

### 1.1 Create Real FINN Model Builder
- Import real FINN model creation utilities
- Build actual ONNX models with real MatrixVectorActivation operations
- Use real FINN DataType system throughout
- Create real tensor shapes and initializers

### 1.2 Replace MockHWCustomOp with Real MVAU
- Import `from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU`
- Create real ONNX nodes using FINN's model builder
- Use real FINN node attribute system
- Test with actual FINN datatype objects

### 1.3 Implement Real Operation Factory
```python
def create_real_mvau_operation(mw=64, mh=64, pe=4, simd=4):
    """Create real FINN MVAU operation with proper ONNX structure."""
    # Build real ONNX model with MVAU node
    # Set real attributes using FINN's attribute system
    # Return real MVAU instance
```

## Phase 2: Real FINN Model Integration Tests

### 2.1 End-to-End Model Creation
- Create complete FINN models with real operations
- Use real FINN transformation pipeline  
- Test with actual FINN model execution
- Validate against real FINN backends

### 2.2 Real Template Generation Testing
- Test template generation with real FINN operations
- Use real FINN code generation pipeline
- Validate generated code compiles with real tools
- Test in actual FINN Docker environment

### 2.3 Real Library Integration
- Test library resolution with real FINN dependencies
- Use actual finn-hlslib and finn-rtllib files
- Validate include path resolution works with real operations
- Test dependency validation with real libraries

## Phase 3: Docker Environment Integration

### 3.1 Docker Test Suite
- Run all tests inside FINN Docker container
- Use real Xilinx tool integration
- Test with actual compilation pipeline
- Validate generated code works with real hardware tools

### 3.2 Real FINN Pipeline Integration
- Test with actual FINN transformation sequences
- Use real FINN model optimization pipeline
- Validate framework works with real FINN workflows
- Test performance with real operations

## Phase 4: Production Validation

### 4.1 Real Hardware Compilation
- Test generated code compiles with Vivado HLS
- Validate RTL generation works with Vivado
- Test with real FPGA synthesis flow
- Validate timing and resource usage

### 4.2 Real FINN Integration
- Test framework integrates with existing FINN codebase
- Validate no regressions in existing functionality
- Test with real FINN example models
- Validate production readiness

## Implementation Checklist

### Immediate Actions (Next 2 hours):
- [ ] Create real FINN model builder utility
- [ ] Replace MockHWCustomOp with real MVAU operations
- [ ] Update all test cases to use real FINN objects
- [ ] Remove all `unittest.mock` imports and usages

### Integration Testing (Next 2 hours):
- [ ] Test in Docker environment with real FINN dependencies
- [ ] Validate code generation with real Xilinx tools
- [ ] Test with actual FINN transformation pipeline
- [ ] Validate library resolution with real files

### Validation (Next 1 hour):
- [ ] Run comprehensive test suite in Docker
- [ ] Validate all 29 tests still pass with real operations
- [ ] Test demo script with real FINN integration
- [ ] Confirm framework works with real FINN workflows

## Success Criteria

✅ **Zero mock objects** in test suite
✅ **Real FINN operations** used throughout
✅ **Real ONNX models** with proper structure  
✅ **Real FINN datatypes** and attributes
✅ **Docker environment** validation
✅ **Real Xilinx tools** integration
✅ **Actual code compilation** success
✅ **Production-ready** framework

## Risk Mitigation

- **Backup current working tests** before modifications
- **Incremental replacement** of mocks (one test class at a time)
- **Continuous validation** that tests still pass
- **Docker environment testing** at each step
- **Real compilation testing** before declaring success