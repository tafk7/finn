# FINN Flexible HLS Backend Integration Guide

This guide provides comprehensive information for integrating and using the FINN Flexible HLS Backend system.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Template System](#template-system)
8. [Path Resolution](#path-resolution)
9. [Diagnostics](#diagnostics)
10. [Migration from Legacy Backend](#migration-from-legacy-backend)
11. [Advanced Usage](#advanced-usage)
12. [Troubleshooting](#troubleshooting)

## Overview

The FINN Flexible HLS Backend is a drop-in replacement for FINN's original HLSBackend that provides:

- **100% API Compatibility**: Seamless replacement with existing FINN kernels
- **Enhanced Flexibility**: Configurable path resolution and template inheritance
- **Improved Maintainability**: Structured configuration and diagnostics
- **Template System**: File-based templates with validation and inheritance
- **Environment Agnostic**: Works across different FINN installations and environments

### Key Benefits

- **Zero Migration Effort**: Existing kernels work without modification
- **Enhanced Debugging**: Comprehensive diagnostics and error reporting
- **Extensible Templates**: Easy customization and reuse of HLS code generation
- **Robust Path Handling**: Environment-aware path resolution with validation
- **Performance Optimized**: Minimal overhead over original backend

## Installation

### Prerequisites

- Python 3.7+
- FINN framework installed and configured
- Required environment variables (see [Configuration](#configuration))

### Standard Installation

The Flexible HLS Backend is integrated into FINN's structure:

```bash
# The backend is located at:
ex_finn/src/finn/util/flexible_hls/

# Templates are located at:
ex_finn/src/finn/util/flexible_hls/templates/finn/
```

### Verification

Run diagnostics to verify installation:

```bash
cd ex_finn/src/finn/util/flexible_hls
python diagnostics.py --verbose
```

## Quick Start

### Basic Usage

The Flexible HLS Backend is automatically used when you inherit from `HLSBackend`:

```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

class MyCustomOp(HLSBackend):
    def __init__(self, onnx_node):
        super().__init__()
        self.onnx_node = onnx_node
    
    def code_generation_ipi(self):
        # Your existing code generation logic
        return {
            "includes": ["#include \"my_kernel.h\""],
            "defines": ["#define MY_PARAM 1"],
            "compute": ["my_kernel(in0_V, out0_V);"],
            "pragmas": ["#pragma HLS INTERFACE axis port=in0_V"]
        }
```

### Template Rendering

```python
from finn.util.flexible_hls import FINNTemplateEngine, FINNTemplateType

# Initialize template engine
engine = FINNTemplateEngine()

# Render a template
variables = {
    "AP_INT_MAX_W": "8191",
    "GLOBALS": "#include \"params.h\"",
    "DEFINES": "#define PE 4",
    "DOCOMPUTE": "my_kernel(in0_V, out0_V);"
}

output = engine.render_template("ipgen_cpp", variables, FINNTemplateType.IPGEN_CPP)
```

### Path Resolution

```python
from finn.util.flexible_hls import FINNPathResolver, FINNPathType

# Initialize path resolver
resolver = FINNPathResolver()

# Resolve paths
finn_root = resolver.get_finn_root()
hlslib_path = resolver.resolve_path("bnn-library.h", FINNPathType.FINN_HLSLIB)
include_paths = resolver.get_include_paths()
```

## Architecture

### Component Overview

```
FINN Flexible HLS Backend
├── FlexibleHLSBackend (main backend class)
├── FINNPathResolver (path resolution)
├── FINNTemplateEngine (template processing)
├── FINNCompilerBuilder (compilation setup)
├── FINNConfig (configuration management)
└── FINNDiagnostics (error handling & diagnostics)
```

### Integration Points

1. **HLSBackend Replacement**: `finn.custom_op.fpgadataflow.hlsbackend.HLSBackend`
2. **Template Access**: `finn.custom_op.fpgadataflow.templates`
3. **Path Resolution**: Environment variable detection and validation
4. **Template Engine**: File-based template system with inheritance

### Backward Compatibility

The system maintains 100% backward compatibility through:

- **API Preservation**: All original HLSBackend methods preserved
- **Template Fallback**: Original string-based templates still accessible
- **Environment Detection**: Automatic fallback to defaults when variables missing
- **Error Handling**: Graceful degradation with informative messages

## API Reference

### FlexibleHLSBackend

The main backend class that replaces the original HLSBackend:

```python
class FlexibleHLSBackend:
    def __init__(self, config: Optional[FINNConfig] = None):
        """Initialize flexible HLS backend."""
    
    def get_include_paths(self) -> List[str]:
        """Get include paths for compilation."""
    
    def get_library_paths(self) -> List[str]:
        """Get library paths for linking."""
    
    def get_compile_flags(self) -> List[str]:
        """Get compilation flags."""
    
    def render_template(self, template_name: str, variables: Dict[str, str], 
                       template_type: FINNTemplateType) -> Optional[str]:
        """Render a template with variables."""
```

### FINNPathResolver

Handles path resolution and environment detection:

```python
class FINNPathResolver:
    def get_finn_root(self) -> Optional[str]:
        """Get FINN root directory."""
    
    def get_finn_deps_dir(self) -> Optional[str]:
        """Get FINN dependencies directory."""
    
    def get_finn_hlslib_dir(self) -> Optional[str]:
        """Get FINN HLS library directory."""
    
    def resolve_path(self, relative_path: str, path_type: FINNPathType) -> str:
        """Resolve a relative path within FINN structure."""
    
    def file_exists(self, filename: str, path_type: FINNPathType) -> bool:
        """Check if a file exists in the given path type."""
    
    def validate_finn_environment(self) -> bool:
        """Validate FINN environment setup."""
```

### FINNTemplateEngine

Manages template processing and rendering:

```python
class FINNTemplateEngine:
    def __init__(self, template_search_paths: List[str] = None):
        """Initialize template engine."""
    
    def render_template(self, template_name: str, variables: Dict[str, str],
                       template_type: FINNTemplateType) -> Optional[str]:
        """Render template with variables."""
    
    def get_template_variables(self, template_name: str, 
                              template_type: FINNTemplateType) -> Optional[Set[str]]:
        """Get required variables for a template."""
    
    def validate_template(self, template_name: str, variables: Dict[str, str],
                         template_type: FINNTemplateType) -> List[str]:
        """Validate template variables."""
```

### Template Types

```python
class FINNTemplateType(Enum):
    IPGEN_CPP = "ipgen_cpp"
    IPGEN_TCL = "ipgen_tcl"
    DOCOMPUTE = "docompute"
    DOCOMPUTE_TIMEOUT = "docompute_timeout"
    CUSTOM = "custom"
```

## Configuration

### Environment Variables

The system uses the following environment variables:

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FINN_ROOT` | Recommended | FINN framework root directory | Auto-detected |
| `FINN_DEPS_DIR` | Recommended | FINN dependencies directory | `${FINN_ROOT}/deps` |
| `FINN_HLSLIB_DIR` | Recommended | FINN HLS library directory | `${FINN_DEPS_DIR}/finn-hlslib` |
| `HLS_PATH` | Optional | Xilinx HLS installation path | None |
| `VITIS_PATH` | Optional | Xilinx Vitis installation path | None |

### Setting Up Environment

```bash
# Example FINN environment setup
export FINN_ROOT=/path/to/finn
export FINN_DEPS_DIR=/path/to/finn/deps
export FINN_HLSLIB_DIR=/path/to/finn/deps/finn-hlslib
export HLS_PATH=/opt/Xilinx/Vivado_HLS/2020.1
export VITIS_PATH=/opt/Xilinx/Vitis/2020.1
```

### Configuration Object

```python
from finn.util.flexible_hls import FINNConfig

# Create configuration
config = FINNConfig()

# Set include paths
config.set_include_paths(["/custom/include", "/another/include"])

# Set compilation flags
config.set_compile_flags(["-O2", "-std=c++14"])

# Set target device
config.set_target_device("xc7z020clg400-1")

# Use with backend
backend = FlexibleHLSBackend(config)
```

## Template System

### Template Structure

Templates use `$VARIABLE$` syntax for variable substitution:

```cpp
// ipgen_cpp.template
#define AP_INT_MAX_W $AP_INT_MAX_W$

#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

$BLACKBOXFUNCTION$
{
$PRAGMAS$
$DOCOMPUTE$
}
```

### Available Templates

1. **ipgen_cpp.template**: C++ code generation for IP creation
2. **ipgen_tcl.template**: TCL script for HLS synthesis
3. **docompute.template**: C++ simulation code
4. **docompute_timeout.template**: C++ simulation with timeout handling

### Template Variables

Common variables across templates:

- `AP_INT_MAX_W`: Maximum width for ap_int types
- `GLOBALS`: Global includes and declarations
- `DEFINES`: Preprocessor definitions
- `PRAGMAS`: HLS pragmas and directives
- `DOCOMPUTE`: Main computation code
- `BLACKBOXFUNCTION`: Function signature for IP generation
- `PROJECTNAME`: HLS project name
- `TOPFXN`: Top-level function name
- `FPGAPART`: Target FPGA part

### Custom Templates

Create custom templates in the template directory:

```bash
# Create custom template
mkdir -p /path/to/templates/custom
cat > /path/to/templates/custom/my_template.template << 'EOF'
// Custom template
#define CUSTOM_PARAM $CUSTOM_PARAM$
$CUSTOM_CODE$
EOF
```

Use custom templates:

```python
engine = FINNTemplateEngine(["/path/to/templates"])
output = engine.render_template("my_template", {
    "CUSTOM_PARAM": "42",
    "CUSTOM_CODE": "// My custom code"
}, FINNTemplateType.CUSTOM)
```

## Path Resolution

### Path Types

The system defines several path types for different FINN components:

```python
class FINNPathType(Enum):
    FINN_ROOT = "finn_root"
    FINN_DEPS = "finn_deps"
    FINN_HLSLIB = "finn_hlslib"
    HLS_INCLUDE = "hls_include"
    VITIS_INCLUDE = "vitis_include"
```

### Path Resolution Examples

```python
resolver = FINNPathResolver()

# Get base directories
finn_root = resolver.get_finn_root()
deps_dir = resolver.get_finn_deps_dir()
hlslib_dir = resolver.get_finn_hlslib_dir()

# Resolve specific files
bnn_lib = resolver.resolve_path("bnn-library.h", FINNPathType.FINN_HLSLIB)
custom_hls = resolver.resolve_path("custom_hls/matrix_ops.h", FINNPathType.FINN_ROOT)

# Check file existence
has_bnn_lib = resolver.file_exists("bnn-library.h", FINNPathType.FINN_HLSLIB)

# Get compilation paths
include_paths = resolver.get_include_paths()
library_paths = resolver.get_library_paths()
```

### Environment Validation

```python
# Validate environment
is_valid = resolver.validate_finn_environment()

# Get diagnostic information
diagnostics = resolver.get_environment_diagnostics()
print(f"FINN_ROOT exists: {diagnostics['finn_root']['exists']}")
print(f"FINN_HLSLIB_DIR exists: {diagnostics['finn_hlslib_dir']['exists']}")
```

## Diagnostics

### Running Diagnostics

```bash
# Run basic diagnostics
python -m finn.util.flexible_hls.diagnostics

# Verbose diagnostics
python -m finn.util.flexible_hls.diagnostics --verbose

# Save detailed report
python -m finn.util.flexible_hls.diagnostics --output diagnostic_report.txt
```

### Programmatic Diagnostics

```python
from finn.util.flexible_hls import FINNDiagnostics

# Create diagnostics instance
diagnostics = FINNDiagnostics()

# Run comprehensive diagnostics
results = diagnostics.diagnose_environment()

# Check for issues
if diagnostics.has_errors():
    print(f"Found {diagnostics.get_error_count()} errors")
    
if diagnostics.has_warnings():
    print(f"Found {diagnostics.get_warning_count()} warnings")

# Generate report
report = diagnostics.generate_report("diagnostic_report.txt")
print(report)
```

### Diagnostic Categories

1. **Environment Variables**: Checks FINN environment variable setup
2. **Path Validation**: Verifies directory existence and permissions
3. **Required Files**: Confirms presence of essential FINN files
4. **Template System**: Validates template availability and functionality
5. **Dependencies**: Checks Python module availability
6. **Permissions**: Verifies file system access rights

## Migration from Legacy Backend

### Automatic Migration

The flexible backend is designed for zero-effort migration:

1. **HLSBackend Inheritance**: Existing classes inheriting from `HLSBackend` automatically use the flexible backend
2. **Template Access**: Existing code accessing templates through `finn.custom_op.fpgadataflow.templates` continues to work
3. **Method Compatibility**: All original HLSBackend methods are preserved

### What Changes

- **Internal Implementation**: Backend logic is enhanced but API remains identical
- **Template Storage**: Templates moved from strings to files (transparent to users)
- **Path Resolution**: More robust environment handling
- **Error Reporting**: Enhanced diagnostics and error messages

### What Stays the Same

- **Public API**: All method signatures unchanged
- **Inheritance Patterns**: `CustomOp_hls(CustomOp, HLSBackend)` pattern preserved
- **Template Variables**: All existing template variables supported
- **Code Generation**: Identical output to original backend

### Validation

Verify migration success:

```python
# Test that your kernel still works
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.flexible_hls import FlexibleHLSBackend

# Verify inheritance
assert issubclass(HLSBackend, FlexibleHLSBackend)

# Test your kernel
class MyKernel(HLSBackend):
    # Your existing implementation
    pass

kernel = MyKernel(mock_node)
assert isinstance(kernel, FlexibleHLSBackend)
```

## Advanced Usage

### Custom Configuration

```python
from finn.util.flexible_hls import FINNConfig, FlexibleHLSBackend

# Create custom configuration
config = FINNConfig()
config.set_include_paths([
    "/custom/include/path",
    "/another/include/path"
])
config.set_compile_flags(["-O3", "-march=native"])
config.set_target_device("xczu7ev-ffvc1156-2-e")

# Use with backend
backend = FlexibleHLSBackend(config)
```

### Template Inheritance

Create template hierarchies:

```
templates/
├── base/
│   └── common.template
├── finn/
│   ├── ipgen_cpp.template
│   └── ipgen_tcl.template
└── custom/
    ├── my_ipgen.template
    └── specialized.template
```

```python
# Template engine searches paths in order
engine = FINNTemplateEngine([
    "/path/to/custom/templates",
    "/path/to/base/templates"
])
```

### Compiler Integration

```python
from finn.util.flexible_hls import FINNCompilerBuilder

# Build compilation command
builder = FINNCompilerBuilder(config)
builder.add_source_files(["kernel.cpp", "utils.cpp"])
builder.add_include_paths(["/custom/include"])
builder.set_optimization_level(2)

# Get compile command
compile_cmd = builder.get_compile_command()
print(f"Compile command: {' '.join(compile_cmd)}")
```

### Error Handling

```python
from finn.util.flexible_hls import FINNDiagnostics, DiagnosticLevel

# Custom error handling
diagnostics = FINNDiagnostics()

try:
    # Your code here
    result = some_operation()
except Exception as e:
    diagnostics.add_message(
        DiagnosticLevel.ERROR,
        "MyComponent",
        f"Operation failed: {str(e)}",
        details={"operation": "some_operation", "error": str(e)},
        suggestions=["Check input parameters", "Verify environment setup"],
        error_code="CUSTOM001"
    )
```

## Troubleshooting

### Common Issues

#### 1. Environment Variables Not Set

**Symptoms**: Warnings about missing environment variables

**Solution**:
```bash
# Set required environment variables
export FINN_ROOT=/path/to/finn
export FINN_DEPS_DIR=/path/to/finn/deps
export FINN_HLSLIB_DIR=/path/to/finn/deps/finn-hlslib
```

#### 2. Template Not Found

**Symptoms**: Template rendering returns `None`

**Diagnosis**:
```python
from finn.util.flexible_hls import FINNTemplateEngine

engine = FINNTemplateEngine()
variables = engine.get_template_variables("template_name", template_type)
if variables is None:
    print("Template not found")
```

**Solution**: Verify template file exists and is in template search path

#### 3. Path Resolution Failures

**Symptoms**: Files not found, compilation errors

**Diagnosis**:
```bash
python -m finn.util.flexible_hls.diagnostics --verbose
```

**Solution**: Check environment setup and file permissions

#### 4. Permission Errors

**Symptoms**: Access denied errors, unable to read/write files

**Solution**: Check directory permissions and user access rights

### Debug Mode

Enable verbose logging:

```python
import logging
logging.getLogger("finn.flexible_hls").setLevel(logging.DEBUG)
```

### Performance Issues

Monitor performance:

```python
import time
from finn.util.flexible_hls import FINNTemplateEngine

start_time = time.time()
engine = FINNTemplateEngine()
init_time = time.time() - start_time
print(f"Template engine initialization: {init_time:.4f}s")

# Template rendering performance
start_time = time.time()
output = engine.render_template("ipgen_cpp", variables, template_type)
render_time = time.time() - start_time
print(f"Template rendering: {render_time:.4f}s")
```

### Getting Help

1. **Run Diagnostics**: Always start with comprehensive diagnostics
2. **Check Environment**: Verify FINN environment variables
3. **Validate Templates**: Ensure template files are accessible
4. **Review Logs**: Enable debug logging for detailed information
5. **Test Isolation**: Create minimal test cases to isolate issues

For additional support, refer to the FINN documentation and community resources.