# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FINN (Fast, Scalable Quantized Neural Network Inference on FPGAs) is an experimental framework from AMD Research that generates hardware code (HLS/RTL) for deploying quantized neural networks on FPGAs. The framework focuses on dataflow-style architectures for high throughput and low latency.

## Development Environment

All development must be done within the Docker container. The project requires Xilinx/AMD tools (Vivado/Vitis) for full functionality.

### Essential Commands

```bash
# Start development environment
export FINN_XILINX_PATH=/path/to/xilinx  # e.g., /opt/Xilinx
export FINN_XILINX_VERSION=2022.2
./run-docker.sh

# Run tests
pytest                                   # All tests
pytest -m "not slow"                    # Fast tests only
pytest tests/specific_test.py::test_name # Single test
pytest -k "test_pattern"                # Run tests matching pattern
./run-docker.sh quicktest               # Quick test suite in Docker
./run-docker.sh test                    # Full test suite in Docker

# Linting and formatting
pre-commit run --all-files              # Run all checks
black --line-length=100 src/ tests/     # Format code
flake8 --max-line-length=100 --extend-ignore=E203,W503 src/ tests/

# Build and install
pip install -e .                        # Development install
pip install -e ".[testing]"             # With test dependencies

# Other Docker commands
./run-docker.sh notebook                # Launch Jupyter server
./run-docker.sh build_dataflow <model>  # Build example model
```

### Test Markers

- `slow`: Long-running tests (>30s)
- `vivado`: Requires Vivado/Vitis tools
- `vitis`: Requires Vitis specifically
- `board`: Requires PYNQ board hardware
- `fpgadataflow`: HLS custom operation tests
- `end2end`: Full flow integration tests
- `util`: Utility function tests
- `analysis`: Analysis tool tests
- `transform`: Transformation tests

## Development Guidance

### Testing Guidelines

- ALWAYS run tests with `./run-docker.sh <command>`

## Architecture Overview

### Core Structure

```
src/finn/
├── analysis/         # FPGA resource and performance analysis
├── builder/          # Build flow orchestration for dataflow architectures
├── codegen/          # Code generation framework (NEW UNIFIED ARCHITECTURE)
├── core/             # Core ModelWrapper and execution functionality
├── custom_op/        # Custom FPGA operations (HLS/RTL implementations)
├── transformation/   # Graph transformation passes
└── util/             # Utilities (ONNX helpers, data packing, etc.)
```

### Key Concepts

1. **ModelWrapper**: Primary interface for ONNX graph manipulation
   - Wraps ONNX models with FINN/QONNX custom operations
   - Provides analysis, transformation, and execution methods
   - Located in `src/finn/core/modelwrapper.py`

2. **Custom Operations**: FPGA-specific layer implementations
   - Base classes in `src/finn/custom_op/fpgadataflow/`
   - HLS implementations in `hlsbackend.py` subclasses
   - RTL implementations for specialized operations

3. **Transformation Passes**: Graph modification pipeline
   - Located in `src/finn/transformation/`
   - Applied sequentially to optimize and prepare models
   - Categories: general, fpgadataflow, qonnx, streamline

### Codegen Architecture (Current Refactoring)

The codebase is undergoing a major refactoring to modernize the code generation system:

1. **Template Value Provider Pattern**: Operations explicitly provide values for templates
   ```python
   class MyOperationHLS(MyOperation, HLSBackend):
       TEMPLATE_NAME = "hls_my_operation.cpp.j2"
       
       def get_template_values(self, template_name: str):
           return {
               'param1': self.get_nodeattr("param1"),
               'param2': self.calculate_value()
           }
   ```

2. **Unified Framework Components**:
   - `TemplateEngine`: Jinja2 rendering with strategic caching
   - `BackendRegistry`: O(1) backend lookup via explicit registration
   - `Codegen` base class: Standard interface for all backends
   - Backend classes: HLS/RTL implementations with templates

3. **Performance Metrics**:
   - 5.7x faster code generation
   - 60% memory reduction
   - Template compilation caching for 10-50ms operations

### Hardware Build Flow

FINN uses a 5-stage build process (`src/finn/builder/build_dataflow_steps.py`):

1. **Step_Tidy**: Graph cleanup and preparation
2. **Step_StreamlineDataflow**: Optimize for FPGA dataflow
3. **Step_Convert**: Lower to HW custom operations
4. **Step_Specialize**: Apply folding and implementation choices
5. **Step_HLSSynth**: Generate and synthesize hardware

### Memory Modes for HLS Operations

- **const**: Weights embedded in HLS code
- **decoupled**: Weights in separate header files
- **external**: Runtime-loadable weights

### Adding New Operations

1. Create operation class inheriting from base and backend
2. Register in `src/finn/codegen/backend_registration.py`
3. Create Jinja2 template in `src/finn/codegen/templates/`
4. Implement required abstract methods (folding constraints, resource estimates)

### Template Development

Templates use Jinja2 with FINN-specific filters:
- `format_define`: HLS #define formatting
- `format_port`: Port declarations
- `format_array`: Array declarations
- `format_tensor`: Tensor shape formatting

Template organization: `hls/`, `rtl/`, `common/` subdirectories

### Key Design Principles

1. **Explicit Over Implicit**: Direct registration, no auto-discovery
2. **Strategic Minimalism**: Cache only expensive operations
3. **Performance Through Simplicity**: Fast lookups, minimal overhead
4. **Full Backward Compatibility**: Legacy code continues working
5. **Dataflow Architecture**: Streaming interfaces between layers

### ONNX Integration

- Uses QONNX as intermediate representation
- Custom operations extend ONNX with FPGA-specific nodes
- FINN-ONNX variant includes hardware implementation details
- Model files use `.onnx` extension with custom op definitions

### Working with Git

Current branch reflects active development on codegen refactoring.
Main branch for PRs: `main`

### Debugging Tips

- Use `FINN_ROOT` environment variable for paths
- Check `finn.log` for detailed execution traces
- Validate folding configurations match hardware constraints
- Use `get_folded_output_shape()` to verify tensor dimensions
- Enable verbose mode in transformations for debugging