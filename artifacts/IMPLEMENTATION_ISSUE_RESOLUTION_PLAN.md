# FINN Codegen Implementation Issue Resolution Plan

## Overview

This plan addresses the core implementation gaps identified in the architecture evaluation, focusing on technical completeness while maintaining the "simple, explicit, fast" consolidation principles.

## Issue Prioritization

### **Critical Path Issues** (Block System Functionality)
1. Missing template infrastructure
2. Missing ModernHLSGenerator/ModernRTLGenerator classes

### **Architecture Integrity Issues** (Violate Design Principles)  
3. Library resolver complexity contradiction
4. File manager over-engineering

---

## Phase 1: Core Infrastructure Completion

### **Issue 1: Missing Template Infrastructure**

**Problem**: System expects `templates/hls/`, `templates/rtl/`, `templates/common/` directories but they don't exist.

**Impact**: Template engine fails when trying to load expected templates.

**Solution Plan**:

#### 1.1 Create Template Directory Structure
```bash
src/finn/codegen/templates/
├── hls/
│   ├── base_operation.cpp.j2
│   ├── thresholding.cpp.j2
│   ├── mvau.cpp.j2
│   └── common_hls.hpp.j2
├── rtl/
│   ├── base_wrapper.sv.j2
│   ├── thresholding_wrapper.sv.j2
│   ├── mvau_wrapper.sv.j2
│   └── common_rtl.sv.j2
└── common/
    ├── header_template.j2
    └── utils.j2
```

#### 1.2 Implement Minimal Working Templates

**HLS Base Template** (`templates/hls/base_operation.cpp.j2`):
```cpp
/*
 * Generated HLS code for {{ node_name }}
 * Operation: {{ op_type }}
 */

{% for include in includes %}
#include "{{ include }}"
{% endfor %}

{% for define_name, define_value in defines.items() %}
{{ define_value | format_define(define_name) }}
{% endfor %}

void {{ function_name }}(
    {% for param in function_params %}
    {{ param.type }} {{ param.name }}{% if not loop.last %},{% endif %}
    {% endfor %}
) {
    // Generated implementation
    {{ implementation_body }}
}
```

**RTL Base Template** (`templates/rtl/base_wrapper.sv.j2`):
```systemverilog
/*
 * Generated RTL wrapper for {{ node_name }}
 * Operation: {{ op_type }}
 */

module {{ module_name }} #(
    {% for param in parameters %}
    parameter {{ param.name }} = {{ param.value }}{% if not loop.last %},{% endif %}
    {% endfor %}
) (
    {% for port in ports %}
    {{ port.direction }} {{ port.type }} {{ port.name }}{% if not loop.last %},{% endif %}
    {% endfor %}
);

    // Generated implementation
    {{ implementation_body }}

endmodule
```

#### 1.3 Template Validation
- Create simple test templates that render successfully
- Verify template engine can find and compile them
- Ensure no missing template errors on system startup

---

## Phase 2: Missing API Implementation

### **Issue 2: Missing ModernHLSGenerator/ModernRTLGenerator Classes**

**Problem**: API classes referenced in system but not implemented.

**Impact**: Import errors when using documented API.

**Solution Plan**:

#### 2.1 Implement ModernHLSGenerator

**File**: `src/finn/codegen/modern_hls_generator.py`

```python
"""
Modern HLS Generator for FINN Operations

Provides simplified HLS code generation using the unified framework.
"""

from .codegen import Codegen
from .template_engine import TemplateEngine
from .file_manager import FileManager
from .library_resolver import LibraryResolver
from typing import Dict, Any, List, Optional
import logging


class ModernHLSGenerator(Codegen):
    """
    Modern HLS code generator using unified framework.
    
    Provides operation-specific HLS code generation with dynamic template
    selection and comprehensive context preparation.
    """
    
    def __init__(self, operation):
        """
        Initialize HLS generator for operation.
        
        Args:
            operation: HWCustomOp instance to generate code for
        """
        super().__init__()
        self.operation = operation
        self.file_manager = FileManager()
        self.library_resolver = LibraryResolver()
        
    def get_template_name(self) -> str:
        """Get HLS template name for this operation."""
        op_type = self._get_operation_type()
        
        # Try operation-specific template first
        specific_template = f"{op_type.lower()}.cpp.j2"
        if self.template_engine.template_exists(f"hls/{specific_template}"):
            return f"hls/{specific_template}"
            
        # Fall back to base template
        return "hls/base_operation.cpp.j2"
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for HLS template."""
        values = self._extract_common_values(self.operation)
        
        # Add HLS-specific values
        values.update({
            'includes': self._get_hls_includes(),
            'defines': self._get_hls_defines(),
            'function_name': self._get_function_name(),
            'function_params': self._get_function_params(),
            'implementation_body': self._get_implementation_body(),
        })
        
        return values
    
    def generate_code(self, model, fpgapart, clk, output_dir=None):
        """
        Generate HLS code for operation.
        
        Args:
            model: ONNX model containing the operation
            fpgapart: Target FPGA part
            clk: Clock specification
            output_dir: Optional output directory
            
        Returns:
            Generated HLS code as string
        """
        # Set context for generation
        self.model = model
        self.fpgapart = fpgapart
        self.clk = clk
        
        # Use parent class generation flow
        return super().generate_code()
    
    def write_generated_files(self, model, fpgapart, clk, code_gen_dir):
        """
        Write generated files to directory.
        
        Args:
            model: ONNX model
            fpgapart: Target FPGA part
            clk: Clock specification
            code_gen_dir: Output directory
            
        Returns:
            List of written file paths
        """
        code = self.generate_code(model, fpgapart, clk)
        
        # Write main implementation file
        impl_file = self.file_manager.write_file(
            f"{code_gen_dir}/{self._get_function_name()}.cpp",
            code
        )
        
        # Write header file if needed
        header_code = self._generate_header()
        header_file = self.file_manager.write_file(
            f"{code_gen_dir}/{self._get_function_name()}.hpp", 
            header_code
        )
        
        return [impl_file, header_file]
    
    def _get_operation_type(self) -> str:
        """Get operation type string."""
        if hasattr(self.operation, 'onnx_node'):
            return self.operation.onnx_node.op_type
        return self.operation.__class__.__name__
    
    def _get_hls_includes(self) -> List[str]:
        """Get required include files for HLS."""
        return self.library_resolver.resolve_includes(self.operation)
    
    def _get_hls_defines(self) -> Dict[str, Any]:
        """Get HLS preprocessor defines."""
        return {
            'PE_COUNT': self._safe_extract_value(self.operation, 'PE', 1),
            'SIMD_COUNT': self._safe_extract_value(self.operation, 'SIMD', 1),
        }
    
    def _get_function_name(self) -> str:
        """Get HLS function name."""
        op_type = self._get_operation_type()
        node_name = getattr(self.operation.onnx_node, 'name', 'op')
        return f"{op_type}_{node_name}_hls"
    
    def _get_function_params(self) -> List[Dict[str, str]]:
        """Get HLS function parameters."""
        params = []
        
        # Input stream parameter
        params.append({
            'type': f'hls::stream<ap_uint<{self.operation.get_instream_width()}>>',
            'name': 'in_stream'
        })
        
        # Output stream parameter
        params.append({
            'type': f'hls::stream<ap_uint<{self.operation.get_outstream_width()}>>',
            'name': 'out_stream'
        })
        
        return params
    
    def _get_implementation_body(self) -> str:
        """Get HLS implementation body."""
        return "// Implementation generated by ModernHLSGenerator"
    
    def _generate_header(self) -> str:
        """Generate header file content."""
        return f"""
#ifndef {self._get_function_name().upper()}_HPP
#define {self._get_function_name().upper()}_HPP

{self._get_function_name()}(/* parameters */);

#endif
"""
```

#### 2.2 Implement ModernRTLGenerator

**File**: `src/finn/codegen/modern_rtl_generator.py`

```python
"""
Modern RTL Generator for FINN Operations

Provides simplified RTL code generation using the unified framework.
"""

from .codegen import Codegen
from .template_engine import TemplateEngine
from .file_manager import FileManager
from typing import Dict, Any, List
import logging


class ModernRTLGenerator(Codegen):
    """
    Modern RTL code generator using unified framework.
    
    Builds on RTL's successful approach while adding Jinja2 template
    capabilities and maintaining RTL simplicity.
    """
    
    def __init__(self, operation):
        """
        Initialize RTL generator for operation.
        
        Args:
            operation: RTL HWCustomOp instance to generate code for
        """
        super().__init__()
        self.operation = operation
        self.file_manager = FileManager()
        
    def get_template_name(self) -> str:
        """Get RTL template name for this operation."""
        op_type = self._get_operation_type()
        
        # Try operation-specific template first
        specific_template = f"{op_type.lower()}_wrapper.sv.j2"
        if self.template_engine.template_exists(f"rtl/{specific_template}"):
            return f"rtl/{specific_template}"
            
        # Fall back to base template
        return "rtl/base_wrapper.sv.j2"
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for RTL template."""
        values = self._extract_common_values(self.operation)
        
        # Add RTL-specific values
        values.update({
            'module_name': self._get_module_name(),
            'parameters': self._get_module_parameters(),
            'ports': self._get_module_ports(),
            'implementation_body': self._get_implementation_body(),
        })
        
        return values
    
    def generate_code(self, model, fpgapart, clk, output_dir=None):
        """
        Generate RTL code for operation.
        
        Args:
            model: ONNX model containing the operation
            fpgapart: Target FPGA part
            clk: Clock specification
            output_dir: Optional output directory
            
        Returns:
            Generated RTL code as string
        """
        # Set context for generation
        self.model = model
        self.fpgapart = fpgapart
        self.clk = clk
        
        # Use parent class generation flow
        return super().generate_code()
    
    def write_generated_files(self, model, fpgapart, clk, code_gen_dir):
        """
        Write generated RTL files to directory.
        
        Args:
            model: ONNX model
            fpgapart: Target FPGA part  
            clk: Clock specification
            code_gen_dir: Output directory
            
        Returns:
            List of written file paths
        """
        code = self.generate_code(model, fpgapart, clk)
        
        # Write main module file
        module_file = self.file_manager.write_file(
            f"{code_gen_dir}/{self._get_module_name()}.sv",
            code
        )
        
        return [module_file]
    
    def _get_operation_type(self) -> str:
        """Get operation type string."""
        if hasattr(self.operation, 'onnx_node'):
            return self.operation.onnx_node.op_type
        return self.operation.__class__.__name__
    
    def _get_module_name(self) -> str:
        """Get RTL module name."""
        op_type = self._get_operation_type()
        node_name = getattr(self.operation.onnx_node, 'name', 'op')
        return f"{op_type}_{node_name}_wrapper"
    
    def _get_module_parameters(self) -> List[Dict[str, Any]]:
        """Get RTL module parameters."""
        return [
            {'name': 'DATA_WIDTH', 'value': self.operation.get_instream_width()},
            {'name': 'PE_COUNT', 'value': self._safe_extract_value(self.operation, 'PE', 1)},
        ]
    
    def _get_module_ports(self) -> List[Dict[str, str]]:
        """Get RTL module ports."""
        return [
            {'direction': 'input', 'type': 'logic', 'name': 'clk'},
            {'direction': 'input', 'type': 'logic', 'name': 'rst_n'},
            {'direction': 'input', 'type': f'logic [{self.operation.get_instream_width()-1}:0]', 'name': 'in_data'},
            {'direction': 'output', 'type': f'logic [{self.operation.get_outstream_width()-1}:0]', 'name': 'out_data'},
        ]
    
    def _get_implementation_body(self) -> str:
        """Get RTL implementation body."""
        return "// Implementation generated by ModernRTLGenerator"
```

#### 2.3 Update Module Exports

**File**: `src/finn/codegen/__init__.py` (add to exports)

```python
from .modern_hls_generator import ModernHLSGenerator
from .modern_rtl_generator import ModernRTLGenerator

__all__ = [
    # Core components
    'TemplateEngine',
    'BackendRegistry', 
    'CodegenConfig',
    
    # Modern generators
    'ModernHLSGenerator',
    'ModernRTLGenerator',
    
    # Utilities
    'FileManager',
    'LibraryResolver',
    
    # Registration
    'get_backend_registry',
    'register_all_backends',
    'get_global_config',
    
    # Base interface
    'Codegen',
    
    # Exceptions
    'UnsupportedTemplateError',
    'TemplateValidationError', 
    'CodeGenerationError'
]
```

---

## Phase 3: Architecture Simplification

### **Issue 3: Library Resolver Complexity Contradiction**

**Problem**: 513 lines of auto-discovery complexity violates "explicit over implicit" principle.

**Current State**: Complex path resolution, fallback logic, auto-detection
**Target State**: Simple, explicit library configuration

**Solution Plan**:

#### 3.1 Simplified Library Resolver

**File**: `src/finn/codegen/simple_library_resolver.py`

```python
"""
Simplified Library Resolver for FINN Code Generation

Provides explicit library dependency resolution without auto-discovery complexity.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging


@dataclass
class SimpleLibrarySpec:
    """Simplified library specification."""
    name: str
    includes: List[str]
    path: Optional[str] = None


class SimpleLibraryResolver:
    """
    Simplified library resolver - explicit configuration only.
    
    Eliminates auto-discovery complexity in favor of explicit configuration.
    """
    
    def __init__(self):
        """Initialize with default FINN libraries."""
        self.logger = logging.getLogger(__name__)
        self.libraries: Dict[str, SimpleLibrarySpec] = {}
        self._register_default_libraries()
    
    def _register_default_libraries(self):
        """Register essential FINN libraries only."""
        
        # Xilinx HLS - system includes
        self.register_library(SimpleLibrarySpec(
            name='xilinx-hls',
            includes=['ap_int.h', 'hls_stream.h', 'ap_fixed.h']
        ))
        
        # FINN HLS Library - environment variable based
        finn_hlslib_path = os.environ.get('FINN_HLSLIB_PATH')
        if finn_hlslib_path and os.path.exists(finn_hlslib_path):
            self.register_library(SimpleLibrarySpec(
                name='finn-hlslib',
                path=finn_hlslib_path,
                includes=['mvau.hpp', 'utils.hpp', 'activations.hpp']
            ))
        
        # Standard C++
        self.register_library(SimpleLibrarySpec(
            name='std-cpp',
            includes=['iostream', 'vector', 'string']
        ))
    
    def register_library(self, library_spec: SimpleLibrarySpec):
        """Register a library specification."""
        self.libraries[library_spec.name] = library_spec
        self.logger.debug(f"Registered library: {library_spec.name}")
    
    def resolve_includes(self, operation) -> List[str]:
        """
        Resolve include files for operation - simple explicit lookup.
        
        Args:
            operation: HWCustomOp instance
            
        Returns:
            List of include file paths
        """
        includes = []
        
        # Always include standard libraries
        for lib_name in ['xilinx-hls', 'std-cpp']:
            if lib_name in self.libraries:
                lib_spec = self.libraries[lib_name]
                includes.extend(lib_spec.includes)
        
        # Add FINN library if available
        if 'finn-hlslib' in self.libraries:
            lib_spec = self.libraries['finn-hlslib']
            if lib_spec.path:
                # Prepend path to includes
                for include in lib_spec.includes:
                    includes.append(os.path.join(lib_spec.path, include))
            else:
                includes.extend(lib_spec.includes)
        
        return includes
    
    def get_library_paths(self) -> Dict[str, str]:
        """Get library paths for compilation."""
        paths = {}
        for name, spec in self.libraries.items():
            if spec.path and os.path.exists(spec.path):
                paths[name] = spec.path
        return paths
```

#### 3.2 Replace Complex Library Resolver

**Implementation Steps**:
1. Replace `library_resolver.py` with `simple_library_resolver.py`
2. Update imports in `modern_hls_generator.py`
3. Update `__init__.py` to export `SimpleLibraryResolver` as `LibraryResolver`
4. **Result**: Reduce from 513 lines to ~80 lines of explicit configuration

### **Issue 4: File Manager Over-Engineering**

**Problem**: 332 lines duplicating `pathlib` functionality.

**Solution Plan**:

#### 4.1 Minimal File Manager

**File**: `src/finn/codegen/simple_file_manager.py`

```python
"""
Minimal File Manager for FINN Code Generation

Provides only FINN-specific file operations, using pathlib for everything else.
"""

from pathlib import Path
from typing import List, Dict, Union
import logging


class SimpleFileManager:
    """
    Minimal file manager - FINN-specific operations only.
    
    Uses pathlib for basic operations, adds only FINN-specific functionality.
    """
    
    def __init__(self, base_path: Union[str, Path] = None):
        """Initialize with optional base path."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.logger = logging.getLogger(__name__)
    
    def write_file(self, file_path: Union[str, Path], content: str) -> Path:
        """
        Write file with directory creation.
        
        Args:
            file_path: Path to write to
            content: File content
            
        Returns:
            Path object for written file
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        file_path.write_text(content)
        self.logger.info(f"Wrote file: {file_path}")
        return file_path
    
    def get_code_gen_dir(self, operation_name: str, backend: str = 'hls') -> Path:
        """
        Get code generation directory for operation.
        
        Args:
            operation_name: Name of operation
            backend: 'hls' or 'rtl'
            
        Returns:
            Path to code generation directory
        """
        code_dir = self.base_path / 'code_gen' / backend / operation_name
        code_dir.mkdir(parents=True, exist_ok=True)
        return code_dir
```

#### 4.2 Replace Complex File Manager

**Implementation Steps**:
1. Replace `file_manager.py` with `simple_file_manager.py`
2. Update imports in generator classes
3. Update `__init__.py` to export `SimpleFileManager` as `FileManager`
4. **Result**: Reduce from 332 lines to ~40 lines of FINN-specific operations

---

## Phase 4: Integration and Validation

### **4.1 System Integration Testing**

**Create Integration Test** (`test_implementation_completion.py`):

```python
"""Test that all implementation issues are resolved."""

def test_missing_classes_resolved():
    """Test that missing generator classes can be imported."""
    from finn.codegen import ModernHLSGenerator, ModernRTLGenerator
    assert ModernHLSGenerator is not None
    assert ModernRTLGenerator is not None

def test_template_infrastructure_exists():
    """Test that template directories exist."""
    from pathlib import Path
    template_base = Path('src/finn/codegen/templates')
    assert (template_base / 'hls').exists()
    assert (template_base / 'rtl').exists()
    assert (template_base / 'common').exists()

def test_simplified_components():
    """Test that simplified components work."""
    from finn.codegen import LibraryResolver, FileManager
    
    # Should be simplified versions
    resolver = LibraryResolver()
    manager = FileManager()
    
    # Basic functionality should work
    assert hasattr(resolver, 'resolve_includes')
    assert hasattr(manager, 'write_file')

def test_end_to_end_generation():
    """Test that code generation works end-to-end."""
    # Mock operation for testing
    class MockOperation:
        def get_instream_width(self): return 32
        def get_outstream_width(self): return 32
        def get_exp_cycles(self): return 1
        @property
        def onnx_node(self):
            class Node: 
                op_type = 'TestOp'
                name = 'test'
            return Node()
    
    from finn.codegen import ModernHLSGenerator
    generator = ModernHLSGenerator(MockOperation())
    
    # Should be able to generate code without errors
    code = generator.generate_code(None, 'test', '100MHz')
    assert len(code) > 0
```

### **4.2 Validation Checklist**

- [ ] Template directories created and populated
- [ ] ModernHLSGenerator and ModernRTLGenerator classes implemented
- [ ] Classes properly exported in `__init__.py`
- [ ] Library resolver simplified to <100 lines
- [ ] File manager simplified to <50 lines
- [ ] Integration tests pass
- [ ] No import errors when using documented API
- [ ] Template engine can find and render templates

---

## Implementation Timeline

### **Week 1: Core Infrastructure**
- Day 1-2: Create template directory structure and basic templates
- Day 3-4: Implement ModernHLSGenerator class
- Day 5: Implement ModernRTLGenerator class

### **Week 2: Simplification**
- Day 1-2: Implement simplified library resolver
- Day 3: Implement simplified file manager
- Day 4-5: Integration testing and validation

### **Week 3: Polish and Validation**
- Day 1-2: End-to-end testing
- Day 3-4: Performance validation of simplified components
- Day 5: Final integration verification

## Success Criteria

1. **✅ No Import Errors**: All documented classes can be imported successfully
2. **✅ Template Infrastructure**: Template engine finds and renders templates
3. **✅ Simplified Components**: Library resolver <100 lines, file manager <50 lines
4. **✅ Functional API**: Generator classes produce working code
5. **✅ Architecture Integrity**: No reintroduction of complexity patterns

## Risk Mitigation

1. **Template Compatibility**: Ensure new templates work with existing operations
2. **API Compatibility**: Generator classes match expected interface
3. **Performance Impact**: Verify simplifications don't degrade performance
4. **Integration Issues**: Test with real FINN operations, not just mocks

This plan addresses all critical implementation issues while maintaining the architectural principles of the consolidation effort.