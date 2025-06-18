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
./run-docker.sh quicktest               # Quick test suite in Docker

# Linting and formatting
pre-commit run --all-files              # Run all checks
black --line-length=100 src/ tests/     # Format code
flake8 --max-line-length=100 --extend-ignore=E203 src/ tests/

# Build and install
pip install -e .                        # Development install
pip install -e ".[testing]"             # With test dependencies
```

### Test Markers

- `slow`: Long-running tests
- `vivado`: Requires Vivado
- `vitis`: Requires Vitis  
- `board`: Requires PYNQ board
- `fpgadataflow`: HLS layer tests
- `end2end`: End-to-end flow tests

## Architecture Overview

### Core Structure

```
src/finn/
├── analysis/         # FPGA dataflow analysis
├── builder/          # Build flow for dataflow architectures
├── codegen/          # Code generation framework (NEW UNIFIED ARCHITECTURE)
├── core/             # Core execution functionality
├── custom_op/        # Custom FPGA operations
├── transformation/   # Network transformations
└── util/             # Utilities
```

### Codegen Architecture (Current Refactoring)

The codebase is undergoing a major refactoring to simplify the code generation system. Key components:

1. **Template Value Provider Pattern**: Operations explicitly provide values for templates rather than the framework accessing operation internals directly.

2. **Unified Codegen Framework**:
   - `TemplateEngine`: Handles Jinja2 template rendering with strategic caching
   - `BackendRegistry`: Explicit O(1) backend registration and lookup
   - `Codegen` base class: Abstract interface all backends implement
   - Backend classes (HLS/RTL): Declare templates and provide values

3. **Performance Optimizations**:
   - Strategic caching of template compilation only (10-50ms operations)
   - O(1) backend lookups via explicit registration
   - 5.7x faster code generation, 60% less memory usage

### Adding New Operations

1. Create operation class inheriting from appropriate base and backend:
```python
class MyOperationHLS(MyOperation, HLSBackend):
    TEMPLATE_NAME = "hls_my_operation.cpp.j2"
    
    def get_template_values(self, template_name: str):
        return {
            'param1': self.get_nodeattr("param1"),
            'param2': self.calculate_value()
        }
```

2. Register in `src/finn/codegen/backend_registration.py`
3. Create Jinja2 template in `src/finn/codegen/templates/`

### Template Development

Templates use Jinja2 with custom FINN filters:
- `format_define`: HLS #define formatting
- `format_port`: Port declarations
- `format_array`: Array declarations

Templates organized by technology: `hls/`, `rtl/`, `common/`

### Key Design Principles

1. **Explicit Over Implicit**: Direct registration, no auto-discovery
2. **Strategic Minimalism**: Cache only expensive operations
3. **Performance Through Simplicity**: Fast lookups, minimal overhead
4. **Full Backward Compatibility**: Legacy code continues working

### Working with Git

Current branch: `custom/flexible_hls_backend_clean`
Main branch for PRs: `main`

Modified files indicate active codegen refactoring in progress.