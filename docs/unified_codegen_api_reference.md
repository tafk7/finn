# FINN Unified Code Generation Framework - API Reference

## Table of Contents

1. [Core Classes](#core-classes)
2. [Generator Classes](#generator-classes) 
3. [Utility Classes](#utility-classes)
4. [Template System](#template-system)
5. [Error Handling](#error-handling)
6. [Configuration](#configuration)
7. [Examples](#examples)

## Core Classes

### BaseGenerator

Abstract base class for all code generators.

```python
class BaseGenerator(ABC):
    """Abstract base class for all FINN code generators."""
    
    def __init__(self, operation: 'HWCustomOp') -> None
    def generate_code(self, output_dir: Union[str, Path], 
                     model=None, fpgapart: str = "xc7z020clg400-1", 
                     clk: str = "100MHz") -> List[Path]
    def validate_operation(self) -> bool
    
    # Abstract methods to implement
    @abstractmethod
    def get_template_name(self) -> str
    @abstractmethod  
    def prepare_context(self, model, fpgapart: str, clk: str) -> Dict[str, Any]
```

**Parameters:**
- `operation`: FINN HWCustomOp instance to generate code for
- `output_dir`: Directory to write generated files
- `model`: FINN model (optional)
- `fpgapart`: Target FPGA part name
- `clk`: Clock frequency specification

**Returns:**
- `generate_code()`: List of generated file paths
- `validate_operation()`: Boolean indicating operation validity

**Example:**
```python
class MyGenerator(BaseGenerator):
    def get_template_name(self) -> str:
        return "hls/my_operation.cpp.j2"
        
    def prepare_context(self, model, fpgapart, clk):
        return {
            'node_name': self.operation.onnx_node.name,
            'op_type': self.operation.onnx_node.op_type,
            'defines': self._generate_defines()
        }
```

## Generator Classes

### ModernHLSGenerator

HLS code generator with comprehensive C++ code generation capabilities.

```python
class ModernHLSGenerator(BaseGenerator):
    def __init__(self, operation: 'HWCustomOp') -> None
    def get_template_name(self) -> str
    def prepare_context(self, model, fpgapart: str, clk: str) -> Dict[str, Any]
    def get_generated_files(self, output_dir: Union[str, Path]) -> List[str]
    def generate_defines(self) -> List[Tuple[str, Union[str, int]]]
    def generate_pragmas(self) -> List[str]
```

**Context Structure:**
```python
{
    'node_name': str,           # Operation node name
    'op_type': str,             # Operation type
    'defines': List[Tuple],     # C preprocessor defines
    'includes': List[str],      # Include file paths
    'pragmas': List[str],       # HLS pragma directives
    'operation_params': Dict,   # Operation-specific parameters
    'input_shapes': List,       # Input tensor shapes
    'output_shapes': List,      # Output tensor shapes
    'template_params': Dict,    # Template-specific parameters
    'fpgapart': str,           # Target FPGA part
    'clk_period': str          # Clock period
}
```

**Operation-Specific Template Selection:**
```python
def get_template_name(self) -> str:
    op_type = self.operation.onnx_node.op_type
    
    template_map = {
        'MatrixVectorActivation': 'hls/mvau_streaming.cpp.j2',
        'Thresholding': 'hls/thresholding.cpp.j2', 
        'Pool_Batch': 'hls/pool.cpp.j2',
        'ConvolutionInputGenerator': 'hls/sliding_window.cpp.j2'
    }
    
    return template_map.get(op_type, 'hls/generic.cpp.j2')
```

### ModernRTLGenerator  

RTL wrapper generator for SystemVerilog module generation.

```python
class ModernRTLGenerator(BaseGenerator):
    def __init__(self, operation: 'HWCustomOp') -> None
    def get_template_name(self) -> str
    def prepare_context(self, model, fpgapart: str, clk: str) -> Dict[str, Any]
    def get_input_ports(self) -> List[Dict[str, Any]]
    def get_output_ports(self) -> List[Dict[str, Any]]  
    def get_parameters(self) -> List[Dict[str, Any]]
    def get_axi_interfaces(self) -> List[Dict[str, Any]]
```

**Port Specification:**
```python
{
    'name': str,        # Port name
    'direction': str,   # 'input' or 'output'
    'width': int,       # Bit width
    'type': str,        # 'wire', 'reg', etc.
    'description': str  # Port description
}
```

**Parameter Specification:**
```python
{
    'name': str,        # Parameter name  
    'value': Union[str, int],  # Parameter value
    'type': str,        # Parameter type
    'description': str  # Parameter description
}
```

**Example Usage:**
```python
generator = ModernRTLGenerator(mvau_operation)
context = generator.prepare_context(model, "xc7z020", "100MHz")

# Access generated ports
input_ports = context['input_ports']
for port in input_ports:
    print(f"Port: {port['name']}, Width: {port['width']}")
```

## Utility Classes

### TemplateEngine

Jinja2-based template rendering engine with hardware-specific filters.

```python
class TemplateEngine:
    def __init__(self, template_dirs: Optional[List[str]] = None)
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str
    def render_string(self, template_string: str, context: Dict[str, Any]) -> str
    def list_templates(self) -> List[str]
    def add_filter(self, name: str, filter_func: Callable) -> None
```

**Built-in Filters:**

| Filter | Usage | Description |
|--------|-------|-------------|
| `to_hex` | `{{ value \| to_hex }}` | Convert integer to hexadecimal |
| `bit_width` | `{{ datatype \| bit_width }}` | Extract bit width from DataType |
| `generate_ports` | `{{ interfaces \| generate_ports }}` | Generate port declarations |
| `c_identifier` | `{{ name \| c_identifier }}` | Sanitize for C identifier |
| `verilog_identifier` | `{{ name \| verilog_identifier }}` | Sanitize for Verilog identifier |

**Custom Filter Example:**
```python
@jinja_filter
def finn_datatype_to_cpp(datatype) -> str:
    """Convert FINN DataType to C++ ap_uint type."""
    if hasattr(datatype, 'bitwidth'):
        width = datatype.bitwidth()
        return f"ap_uint<{width}>"
    return "int"

# Register filter
engine = TemplateEngine()
engine.add_filter('finn_datatype_to_cpp', finn_datatype_to_cpp)

# Use in template:
# {{ input_datatype | finn_datatype_to_cpp }} input_data;
```

### FileManager

Robust file operations with atomic writes and error handling.

```python
class FileManager:
    def __init__(self, base_dir: Optional[Union[str, Path]] = None)
    def write_file(self, path: Union[str, Path], content: str) -> Path
    def read_file(self, path: Union[str, Path]) -> str
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> Path
    def ensure_directory(self, path: Union[str, Path]) -> Path
    def list_files(self, directory: Union[str, Path], 
                  pattern: str = "*") -> List[Path]
    def get_code_gen_dir(self, operation_name: str, 
                        backend: str = "hls") -> Path
```

**Directory Structure:**
```
output_dir/
├── hls/
│   ├── operation_name/
│   │   ├── operation_name.cpp
│   │   ├── operation_name.h  
│   │   └── Makefile
└── rtl/
    ├── operation_name/
    │   ├── operation_name_wrapper.v
    │   └── operation_name.tcl
```

**Example:**
```python
fm = FileManager("./generated_code")

# Create operation directory
op_dir = fm.get_code_gen_dir("mvau_0", "hls")
# Returns: ./generated_code/hls/mvau_0/

# Write files atomically  
cpp_file = fm.write_file(op_dir / "mvau_0.cpp", cpp_content)
header_file = fm.write_file(op_dir / "mvau_0.h", header_content)
```

### LibraryResolver

Intelligent library dependency resolution and path management.

```python
class LibraryResolver:
    def __init__(self)
    def resolve_includes(self, operation) -> List[str]
    def resolve_libraries(self, operation) -> List[str]  
    def resolve_library_paths(self, operation) -> Dict[str, str]
    def register_library(self, library_spec: LibrarySpec) -> None
    def get_library_info(self, library_name: str) -> Optional[LibrarySpec]
    def list_libraries(self, library_type: Optional[LibraryType] = None) -> List[str]
    def validate_dependencies(self, operation) -> Dict[str, bool]
```

**LibrarySpec Structure:**
```python
@dataclass
class LibrarySpec:
    name: str                           # Library name
    path: str                          # Library path (supports env vars)
    include_files: List[str]           # Header files in library
    library_type: LibraryType         # HLS_LIBRARY, RTL_LIBRARY, etc.
    required_for: List[str] = None     # Operations requiring this library  
    dependencies: List[str] = None     # Other libraries this depends on
```

**Example:**
```python
resolver = LibraryResolver()

# Register custom library
resolver.register_library(LibrarySpec(
    name='my-custom-lib',
    path='/path/to/my/library',
    include_files=['my_ops.hpp', 'my_utils.hpp'],
    library_type=LibraryType.CUSTOM_LIBRARY,
    required_for=['MyCustomOperation']
))

# Resolve dependencies for operation
includes = resolver.resolve_includes(my_operation)
# Returns: ['/path/to/my/library/my_ops.hpp', ...]

# Check dependency availability
status = resolver.validate_dependencies(my_operation)
# Returns: {'my-custom-lib': True, 'finn-hlslib': True}
```

## Template System

### Template Context Variables

Templates have access to rich context information:

#### Common Variables (All Templates)
```jinja2
{{ node_name }}           <!-- Operation node name -->
{{ op_type }}             <!-- Operation type string -->
{{ fpgapart }}            <!-- Target FPGA part -->
{{ clk_period }}          <!-- Clock period -->
{{ timestamp }}           <!-- Generation timestamp -->
{{ generator_version }}   <!-- Framework version -->
```

#### HLS-Specific Variables
```jinja2
{% for define_name, define_value in defines %}
#define {{ define_name }} {{ define_value }}
{% endfor %}

{% for pragma in pragmas %}
#pragma {{ pragma }}
{% endfor %}

{% for include in includes %}
#include "{{ include }}"
{% endfor %}

{{ operation_params.MW }}    <!-- Matrix width -->
{{ operation_params.PE }}    <!-- Processing elements -->
```

#### RTL-Specific Variables
```jinja2
module {{ module_name }} (
    {%- for port in input_ports %}
    {{ port.direction }} {{ port.type }} {{ port.name }}{{ "," if not loop.last }}
    {%- endfor %}
);

{% for param in parameters %}
parameter {{ param.name }} = {{ param.value }};
{% endfor %}
```

### Template Inheritance

Use template inheritance for common patterns:

```jinja2
{# base_hls_template.j2 #}
// {{ node_name }}.cpp - Generated HLS Implementation
// Operation: {{ op_type }}
// Generated: {{ timestamp }}

{% block includes %}
#include <ap_int.h>
#include <hls_stream.h>
{% endblock %}

{% block defines %}{% endblock %}

{% block function_signature %}
void {{ node_name }}_hls(/* parameters */);
{% endblock %}

{% block implementation %}{% endblock %}

{# mvau_template.j2 #}
{% extends "base_hls_template.j2" %}

{% block defines %}
#define MW {{ operation_params.MW }}
#define PE {{ operation_params.PE }}
{% endblock %}

{% block implementation %}
// MVAU-specific implementation
{% endblock %}
```

## Error Handling

### Exception Hierarchy

```python
class CodegenError(Exception):
    """Base exception for code generation errors."""
    pass

class TemplateError(CodegenError):
    """Template-related errors."""
    def __init__(self, message: str, template_name: str = None, 
                 line_number: int = None):
        self.template_name = template_name
        self.line_number = line_number
        super().__init__(message)

class ValidationError(CodegenError):
    """Operation validation errors."""  
    def __init__(self, message: str, operation_name: str = None,
                 details: List[str] = None):
        self.operation_name = operation_name
        self.details = details or []
        super().__init__(message)

class LibraryError(CodegenError):
    """Library resolution errors."""
    def __init__(self, message: str, library_name: str = None,
                 suggested_paths: List[str] = None):
        self.library_name = library_name  
        self.suggested_paths = suggested_paths or []
        super().__init__(message)

class FileOperationError(CodegenError):
    """File operation errors."""
    pass
```

### Error Handling Patterns

```python
from finn.codegen.exceptions import *

try:
    generator = ModernHLSGenerator(operation)
    files = generator.generate_code("output/")
    
except ValidationError as e:
    print(f"Operation validation failed: {e}")
    for detail in e.details:
        print(f"  - {detail}")
        
except TemplateError as e:
    print(f"Template error in {e.template_name}:{e.line_number}")
    print(f"  {e}")
    
except LibraryError as e:
    print(f"Missing library: {e.library_name}")
    print("Suggested paths:")
    for path in e.suggested_paths:
        print(f"  - {path}")
        
except FileOperationError as e:
    print(f"File operation failed: {e}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FINN_ROOT` | FINN installation root | Auto-detected |
| `FINN_DEPS_DIR` | Dependencies directory | `${FINN_ROOT}/deps` |
| `FINN_CODEGEN_TEMPLATES` | Additional template directories | None |
| `FINN_CODEGEN_DEBUG` | Enable debug logging | `false` |

### Configuration File

Create `~/.finn/codegen.yaml`:

```yaml
# Template directories (in search order)
template_dirs:
  - "./custom_templates"
  - "${FINN_ROOT}/src/finn/codegen/templates"

# Library search paths  
library_paths:
  - "${FINN_DEPS_DIR}"
  - "./custom_libraries"

# Default code generation settings
defaults:
  fpgapart: "xc7z020clg400-1"
  clk: "100MHz"
  backend: "hls"

# Debug settings
debug:
  enabled: false
  log_level: "INFO"
  trace_templates: false
```

### Programmatic Configuration

```python
from finn.codegen.config import CodegenConfig

# Load configuration
config = CodegenConfig.load()

# Override settings
config.template_dirs.append("./my_templates")
config.defaults.fpgapart = "xczu3eg-sbva484-1-e"

# Apply configuration
generator = ModernHLSGenerator(operation)
generator.configure(config)
```

## Examples

### Complete HLS Generation Workflow

```python
from finn.codegen import ModernHLSGenerator
from pathlib import Path

# Create operation (example)
operation = MatrixVectorActivation()
operation.set_nodeattr("MW", 128)
operation.set_nodeattr("PE", 8)
operation.set_nodeattr("inputDataType", DataType["INT8"])

# Generate HLS code
generator = ModernHLSGenerator(operation)

try:
    # Validate operation first
    if not generator.validate_operation():
        raise ValueError("Operation validation failed")
    
    # Generate code
    output_dir = Path("./generated_hls")
    files = generator.generate_code(
        output_dir=output_dir,
        model=None,
        fpgapart="xc7z020clg400-1", 
        clk="100MHz"
    )
    
    print(f"Generated {len(files)} files:")
    for file_path in files:
        print(f"  - {file_path}")
        
except Exception as e:
    print(f"Code generation failed: {e}")
```

### Custom Template Development

```python
# 1. Create custom generator
class CustomStreamingGenerator(ModernHLSGenerator):
    def get_template_name(self) -> str:
        return "custom/streaming_operation.cpp.j2"
    
    def prepare_context(self, model, fpgapart, clk):
        # Get base context
        context = super().prepare_context(model, fpgapart, clk)
        
        # Add custom context
        context.update({
            'streaming_width': self.operation.get_nodeattr('StreamingWidth'),
            'buffer_depth': self.operation.get_nodeattr('BufferDepth'),
            'custom_defines': self._get_custom_defines()
        })
        
        return context
    
    def _get_custom_defines(self):
        return [
            ('STREAMING_WIDTH', self.operation.get_nodeattr('StreamingWidth')),
            ('BUFFER_DEPTH', self.operation.get_nodeattr('BufferDepth'))
        ]

# 2. Create custom template
template_content = """
// {{ node_name }}.cpp - Custom Streaming Operation
#include "streaming_ops.hpp"

{% for define_name, define_value in custom_defines %}
#define {{ define_name }} {{ define_value }}
{% endfor %}

void {{ node_name }}_hls(
    hls::stream<ap_uint<STREAMING_WIDTH>>& input_stream,
    hls::stream<ap_uint<STREAMING_WIDTH>>& output_stream
) {
    // Custom implementation
    static ap_uint<STREAMING_WIDTH> buffer[BUFFER_DEPTH];
    // ...
}
"""

# 3. Use custom generator
operation = CustomStreamingOperation()
generator = CustomStreamingGenerator(operation)
files = generator.generate_code("./output")
```

### Batch Code Generation

```python
from finn.codegen import ModernHLSGenerator, ModernRTLGenerator

def batch_generate_code(operations, output_base_dir):
    """Generate code for multiple operations."""
    results = {}
    
    for i, operation in enumerate(operations):
        try:
            # Determine generator type
            if hasattr(operation, 'get_rtl_template'):
                generator = ModernRTLGenerator(operation)
                backend = "rtl"
            else:
                generator = ModernHLSGenerator(operation)  
                backend = "hls"
            
            # Create operation-specific output directory
            op_name = operation.onnx_node.name
            output_dir = Path(output_base_dir) / backend / op_name
            
            # Generate code
            files = generator.generate_code(output_dir)
            results[op_name] = {
                'success': True,
                'files': files,
                'backend': backend
            }
            
            print(f"✓ Generated {backend.upper()} code for {op_name}")
            
        except Exception as e:
            results[op_name] = {
                'success': False,
                'error': str(e),
                'backend': 'unknown'
            }
            print(f"✗ Failed to generate code for {op_name}: {e}")
    
    return results

# Usage
operations = [mvau_op, threshold_op, pool_op]
results = batch_generate_code(operations, "./batch_output")

# Print summary
successful = sum(1 for r in results.values() if r['success'])
print(f"\nGenerated code for {successful}/{len(operations)} operations")
```

This API reference provides comprehensive documentation for developers to effectively use and extend the FINN Unified Code Generation Framework.