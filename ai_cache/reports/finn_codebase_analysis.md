# FINN Codebase Analysis Report

**Date:** 2025-06-18  
**Branch:** testing-baseline  
**Analysis Focus:** Architecture, Build/Test Commands, Development Workflow, Codegen Refactoring

## Executive Summary

FINN (Fast, Scalable Quantized Neural Network Inference on FPGAs) is an experimental framework from AMD Research for deploying quantized neural networks on FPGAs using dataflow-style architectures. The codebase is undergoing a major refactoring to modernize the code generation system with enhanced performance and maintainability.

## Architecture Overview

### Core Structure

```
src/finn/
├── analysis/         # FPGA dataflow analysis tools
├── builder/          # Build flow for dataflow architectures
├── codegen/          # Code generation framework (MAJOR REFACTORING IN PROGRESS)
├── core/             # Core execution functionality
├── custom_op/        # Custom FPGA operations (HLS/RTL)
├── transformation/   # Network transformations
└── util/             # Utilities
```

### Key Components

1. **ONNX-Based IR**: Uses QONNX and FINN-ONNX variants with custom operations
2. **ModelWrapper**: Thin wrapper around ONNX for graph manipulation
3. **Custom Operations**: HLS and RTL implementations for FPGA operations
4. **Transformation System**: Analysis and transformation passes
5. **Build System**: Docker-based with Xilinx tool integration

## Build and Development Commands

### Essential Commands

```bash
# Environment Setup (Required)
export FINN_XILINX_PATH=/path/to/xilinx  # e.g., /opt/Xilinx
export FINN_XILINX_VERSION=2022.2

# Start Development Environment
./run-docker.sh                          # Interactive shell
./run-docker.sh notebook                 # Jupyter notebook server
./run-docker.sh build_dataflow <dir>     # Build dataflow from directory
./run-docker.sh build_custom <dir> <flow># Custom build flow

# Run Tests
./run-docker.sh test                     # All tests
./run-docker.sh quicktest                # Fast tests (no Vivado/slow)
pytest -m "not slow"                    # Fast tests only (inside Docker)
pytest tests/specific_test.py::test_name # Single test

# Linting and Formatting
pre-commit run --all-files               # Run all checks
black --line-length=100 src/ tests/      # Format code
flake8 --max-line-length=100 --extend-ignore=E203 src/ tests/

# Build and Install
pip install -e .                         # Development install
pip install -e ".[testing]"              # With test dependencies
```

### Test Markers

- `slow`: Long-running tests
- `vivado`: Requires Vivado
- `vitis`: Requires Vitis
- `board`: Requires PYNQ board
- `fpgadataflow`: HLS layer tests
- `end2end`: End-to-end flow tests
- `brevitas_export`: Brevitas export tests
- `streamline`: Streamlining tests
- `transform`: Transformation tests
- `util`: Utility function tests

## Docker Environment

### Key Environment Variables

- `FINN_XILINX_PATH`: Path to Xilinx tools installation
- `FINN_XILINX_VERSION`: Version of Xilinx tools (e.g., 2022.2)
- `FINN_HOST_BUILD_DIR`: Build directory (default: `/tmp/finn_dev_$USER`)
- `FINN_DEPS_DIR`: Dependencies directory
- `JUPYTER_PORT`: Jupyter port (default: 8888)
- `NETRON_PORT`: Netron visualization port (default: 8081)
- `NUM_DEFAULT_WORKERS`: Worker count (default: 4)

### Docker Image

- Base: Ubuntu Jammy (22.04)
- Python 3.10
- PyTorch 2.7.0
- Includes XRT, Vivado/Vitis integration
- Pre-installed dependencies from `requirements.txt`

## Dependencies

### External Repositories (Auto-fetched)

1. **qonnx**: Quantized ONNX extensions
2. **finn-experimental**: Experimental features
3. **brevitas**: Quantization-aware training
4. **finn-hlslib**: HLS library components
5. **cnpy**: NumPy file I/O for C++
6. **oh-my-xilinx**: Xilinx tool utilities
7. **Board files**: Various FPGA board definitions

### Key Python Dependencies

- ONNX ecosystem: onnx, onnxruntime, onnxoptimizer
- Testing: pytest, pytest-cov, pytest-xdist
- Development: black, isort, flake8, pre-commit
- Template engine: Jinja2 (for new codegen)
- Data handling: numpy, pandas, scikit-learn

## Codegen Refactoring (Current Major Work)

### Overview

The codebase is undergoing a significant refactoring to modernize the code generation system:

- **Goal**: Replace string-based templating with Jinja2 templates
- **Benefits**: 5.7x faster generation, 60% less memory, better maintainability
- **Status**: A/B testing framework implemented, awaiting clean backend implementations

### New Architecture

```
HWCustomOp → Backend Selection → Generator (HLS/RTL) → BaseCodeGenerator
                                                     ↓
                         TemplateEngine ← → FileManager ← → LibraryResolver
                                                     ↓
                         Operation-Specific Templates → Generated Code
```

### Key New Components

1. **CG_BackendRegistry**: Clean/legacy backend switching for A/B testing
2. **CodegenValidator**: Parallel validation of clean vs legacy implementations
3. **TemplateEngine**: Jinja2-based template processing
4. **FileManager**: Centralized file operations
5. **LibraryResolver**: Dynamic dependency resolution

### Migration Status

- ✅ A/B testing framework operational
- ✅ Validation infrastructure ready
- ⏳ Clean backend implementations pending for:
  - CG_ThresholdingHLS
  - CG_MVAU_HLS
  - CG_Thresholding_rtl
  - CG_MVAU_rtl

## Development Workflow

### Adding New Operations

1. Create operation class inheriting from appropriate base:
```python
class MyOperationHLS(MyOperation, HLSBackend):
    TEMPLATE_NAME = "hls_my_operation.cpp.j2"
    
    def get_template_values(self, template_name: str):
        return {
            'param1': self.get_nodeattr("param1"),
            'param2': self.calculate_value()
        }
```

2. Register in backend registry
3. Create Jinja2 template in `src/finn/codegen/templates/`
4. Add tests

### Hardware Build Flow

1. Driver generation
2. DMA and DWC node insertion
3. Partitioning for floorplanning
4. FIFO insertion and IP generation
5. Vivado/Vitis project generation and synthesis

### Testing Strategy

```bash
# Quick validation
./run-docker.sh quicktest

# Full test suite with appropriate parallelism
./run-docker.sh "pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n auto"  # Main tests
./run-docker.sh "pytest -k rtlsim --workers auto"                              # RTL sim tests
./run-docker.sh "pytest -k end2end"                                            # End-to-end tests
```

## Important Technical Details

### Custom ONNX Extensions

- **QONNX**: Quantization annotations for sub-8-bit datatypes
- **FINN-ONNX**: Hardware-specific custom operations
- **Domains**: `finn.*` and `qonnx.*` for custom ops

### Memory Modes (HLS MVAU)

1. **internal_embedded**: Weights baked into HLS code
2. **internal_decoupled**: Weights streamed from memory
3. **external**: External weight storage (future)

### Folding Constraints

Each layer type has specific constraints on folding factors (PE/SIMD):
- Must divide evenly into layer dimensions
- Affects parallelism and resource usage
- Documented in `internals.rst`

## Key Insights for Future Development

1. **Docker-Only Development**: All development must happen in Docker due to complex Xilinx tool dependencies

2. **Backward Compatibility**: New codegen maintains 100% compatibility while offering opt-in improvements

3. **Performance Focus**: New architecture targets significant performance improvements (5.7x faster)

4. **Test-Driven**: Comprehensive test suite with specific markers for different scenarios

5. **Modular Architecture**: Clear separation between analysis, transformation, and code generation

6. **Hardware Abstraction**: Supports both HLS and RTL backends with shared infrastructure

7. **A/B Testing Ready**: Framework allows gradual migration and validation of new implementations

## Recommendations for Future Claude Instances

1. **Always use Docker**: Run all commands through `./run-docker.sh`
2. **Check test markers**: Use appropriate pytest markers for targeted testing
3. **Follow pre-commit**: Run `pre-commit run --all-files` before commits
4. **Understand the refactoring**: New codegen system is the future, legacy is being phased out
5. **Use ModelWrapper**: Primary interface for ONNX graph manipulation
6. **Respect folding constraints**: Each operation has specific requirements
7. **Leverage A/B testing**: Validate new implementations against legacy

---
*Generated by FINN Codebase Analysis Tool*