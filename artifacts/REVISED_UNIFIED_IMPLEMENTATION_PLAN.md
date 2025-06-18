# FINN Codegen Revised Unified Implementation Plan

## Overview

**Corrected Approach**: Update existing `HLSBackend` and `RTLBackend` classes to use the new unified template system, rather than creating new `ModernHLSGenerator`/`ModernRTLGenerator` classes.

## Current State Analysis

### **Existing Backend Infrastructure**
- ✅ **HLSBackend** (line 49 in `hlsbackend.py`): Already inherits from `Codegen`
- ✅ **RTLBackend** (line 44 in `rtlbackend.py`): Already inherits from `Codegen`
- ✅ **Template Interface**: Both have `get_template_name()` and template selection logic
- ✅ **Template Declarations**: Both use `TEMPLATE_NAME` and `TEMPLATE_OPTIONS` class attributes

### **Current Template Usage**
- **HLS**: Uses `templates.ipgen_template`, `templates.docompute_template` etc. (string replacement)
- **RTL**: Uses abstract `generate_hdl()` method (needs template integration)

### **Missing Components**
- ❌ Jinja2 template files (need to convert from existing string templates)
- ❌ Integration of `TemplateEngine` with existing backend classes
- ❌ Update template rendering from `$VAR$` → `{{ var }}` syntax

---

## Phase 1: Template Infrastructure Creation

### **1.1 Convert Existing Templates to Jinja2**

**Create directory structure:**
```bash
src/finn/codegen/templates/
├── hls/
│   ├── docompute.cpp.j2
│   ├── docompute_timeout.cpp.j2
│   ├── ipgen.cpp.j2
│   ├── ipgen.tcl.j2
│   └── ip_package.tcl.j2
└── rtl/
    ├── thresholding_wrapper.v.j2
    └── swg_wrapper.v.j2
```

**Templates are direct conversions from existing `templates.py`** (as detailed in REVISED_TEMPLATE_IMPLEMENTATION_PLAN.md)

### **1.2 Template Files Implementation**

All templates exactly as specified in the previous plan - converting `$VAR$` → `{{ var }}` syntax while preserving all functionality.

---

## Phase 2: Update HLSBackend Integration

### **2.1 Modify HLSBackend to Use TemplateEngine**

**File**: `src/finn/custom_op/fpgadataflow/hlsbackend.py`

**Key Changes:**

#### **Update Template Integration**
```python
# Add to imports at top
from finn.codegen import TemplateEngine

class HLSBackend(Codegen):
    def __init__(self, **kwargs):
        """Initialize HLS backend with Codegen infrastructure."""
        # Existing initialization...
        super().__init__()
        
        # NEW: Initialize template engine
        self.template_engine = TemplateEngine()
        
        # Existing HLS-specific initialization...
        self.hls_template_path = "hls/"  # Updated path
        self.code_gen_dict = {}  # Keep for legacy compatibility
```

#### **Add Template Values Implementation**
```python
def get_template_values(self, template_name: str) -> Dict[str, Any]:
    """Extract values for HLS template - IMPLEMENTATION REQUIRED.
    
    Args:
        template_name: Name of template to extract values for
        
    Returns:
        Dictionary mapping template placeholders to values
    """
    # Convert existing code_gen_dict to template values
    template_values = {}
    
    # Common HLS values from existing logic
    template_values.update({
        'AP_INT_MAX_W': self.get_ap_int_max_w(),
        'GLOBALS': self._get_globals_from_code_gen_dict(),
        'DEFINES': self._get_defines_from_code_gen_dict(),
        'PRAGMAS': self._get_pragmas_from_code_gen_dict(),
        'STREAMDECLARATIONS': self._get_streams_from_code_gen_dict(),
        'DOCOMPUTE': self._get_docompute_from_code_gen_dict(),
    })
    
    # Template-specific values
    if 'docompute' in template_name:
        template_values.update({
            'READNPYDATA': self._get_readnpy_from_code_gen_dict(),
            'DATAOUTSTREAM': self._get_dataout_from_code_gen_dict(),
            'SAVEASCNPY': self._get_save_from_code_gen_dict(),
        })
        
        if 'timeout' in template_name:
            template_values.update({
                'TIMEOUT_VALUE': self._get_timeout_value(),
                'TIMEOUT_CONDITION': self._get_timeout_condition(),
                'TIMEOUT_READ_STREAM': self._get_timeout_read_stream(),
            })
    
    elif 'ipgen' in template_name:
        if template_name.endswith('.cpp.j2'):
            template_values.update({
                'BLACKBOXFUNCTION': self._get_blackbox_from_code_gen_dict(),
            })
        elif template_name.endswith('.tcl.j2'):
            template_values.update({
                'PROJECTNAME': f"project_{self.onnx_node.name}",
                'HWSRCDIR': self.get_nodeattr("code_gen_dir_ipgen"),
                'FPGAPART': self._current_fpgapart,  # Set during generation
                'TOPFXN': self.onnx_node.name,
                'CLKPERIOD': self._current_clk,  # Set during generation
                'DEFAULT_DIRECTIVES': '\n'.join(self.ipgen_default_directives()),
                'EXTRA_DIRECTIVES': '\n'.join(self.ipgen_extra_directives()),
            })
    
    return template_values

def _get_globals_from_code_gen_dict(self) -> str:
    """Extract globals from code_gen_dict."""
    if '$GLOBALS$' in self.code_gen_dict:
        return '\n'.join(self.code_gen_dict['$GLOBALS$'])
    return '// No globals'

def _get_defines_from_code_gen_dict(self) -> str:
    """Extract defines from code_gen_dict."""
    if '$DEFINES$' in self.code_gen_dict:
        return '\n'.join(self.code_gen_dict['$DEFINES$'])
    return '// No defines'

# Additional helper methods for extracting from code_gen_dict...
```

#### **Update Code Generation Methods**
```python
def code_generation_ipgen(self, model, fpgapart, clk):
    """Generate HLS code using new template system."""
    # Store context for template values
    self._current_fpgapart = fpgapart
    self._current_clk = clk
    
    node = self.onnx_node
    path = self.get_nodeattr("code_gen_dir_ipgen")
    
    # LEGACY: Still populate code_gen_dict for backward compatibility
    self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
    self.generate_params(model, path)
    self.global_includes()
    self.defines("ipgen")
    self.blackboxfunction()
    self.pragmas()
    self.docompute()
    
    # NEW: Use template engine
    cpp_template_name = "hls/ipgen.cpp.j2"
    cpp_code = self.generate_code()  # Uses get_template_name() and get_template_values()
    
    # Write CPP file
    cpp_path = os.path.join(path, f"top_{node.name}.cpp")
    with open(cpp_path, "w") as f:
        f.write(cpp_code)
    
    # Clear and prepare for TCL template
    self.code_gen_dict.clear()
    
    # Generate TCL script using template
    self.set_template_override("hls/ipgen.tcl.j2")
    tcl_code = self.generate_code()
    
    # Write TCL file
    tcl_path = os.path.join(path, f"hls_syn_{node.name}.tcl")
    with open(tcl_path, "w") as f:
        f.write(tcl_code)
    
    self.code_gen_dict.clear()

def code_generation_cppsim(self, model):
    """Generate C++ simulation code using new template system."""
    node = self.onnx_node
    path = self.get_nodeattr("code_gen_dir_cppsim")
    
    # LEGACY: Populate code_gen_dict for backward compatibility
    self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
    self.generate_params(model, path)
    self.global_includes()
    self.defines("cppsim")
    self.read_npy_data()
    self.strm_decl()
    self.pragmas()
    self.docompute()
    self.dataoutstrm()
    self.save_as_npy()
    
    # NEW: Determine template based on interface
    if self.get_nodeattr("cpp_interface") == "hls_vector":
        self.timeout_value()
        self.timeout_condition()
        self.timeout_read_stream()
        template_name = "hls/docompute_timeout.cpp.j2"
    else:
        template_name = "hls/docompute.cpp.j2"
    
    # Generate using template
    self.set_template_override(template_name)
    cpp_code = self.generate_code()
    
    # Write file
    cpp_path = os.path.join(path, f"execute_{node.op_type}.cpp")
    with open(cpp_path, "w") as f:
        f.write(cpp_code)
    
    self.code_gen_dict.clear()
```

---

## Phase 3: Update RTLBackend Integration

### **3.1 Modify RTLBackend to Use TemplateEngine**

**File**: `src/finn/custom_op/fpgadataflow/rtlbackend.py`

**Key Changes:**

#### **Update Template Integration**
```python
# Add to imports at top
from finn.codegen import TemplateEngine

class RTLBackend(Codegen):
    def __init__(self, **kwargs):
        """Initialize RTL backend with Codegen infrastructure."""
        # Existing initialization...
        super().__init__()
        
        # NEW: Initialize template engine
        self.template_engine = TemplateEngine()
        
        # Existing RTL-specific initialization...
        self.rtl_template_path = "rtl/"  # Updated path
```

#### **Add Template Values Implementation**
```python
def get_template_values(self, template_name: str) -> Dict[str, Any]:
    """Extract values for RTL template.
    
    Args:
        template_name: Name of template to extract values for
        
    Returns:
        Dictionary mapping template placeholders to values
    """
    # Base RTL values
    template_values = self._extract_common_values(self)
    
    # Template-specific values
    if 'thresholding' in template_name:
        template_values.update(self._get_thresholding_template_values())
    elif 'swg' in template_name:
        template_values.update(self._get_swg_template_values())
    else:
        # Generic RTL wrapper values
        template_values.update(self._get_generic_wrapper_values())
    
    return template_values

def _get_thresholding_template_values(self) -> Dict[str, Any]:
    """Get values for thresholding wrapper template."""
    return {
        'MODULE_NAME_AXI_WRAPPER': f"{self.onnx_node.name}_wrapper",
        'N': self._safe_extract_value(self, 'NumSteps', 8),
        'WI': self._safe_extract_value(self, 'inputDataType', 8),
        'WT': self._safe_extract_value(self, 'weightDataType', 8),
        'C': self._safe_extract_value(self, 'NumChannels', 32),
        'PE': self._safe_extract_value(self, 'PE', 4),
        'SIGNED': 0,
        'FPARG': 0,
        'BIAS': 0,
        'THRESHOLDS_PATH': '""',
        'USE_AXILITE': 1,
        'DEPTH_TRIGGER_URAM': 0,
        'DEPTH_TRIGGER_BRAM': 0,
        'DEEP_PIPELINE': 0,
        'O_BITS': self._safe_extract_value(self, 'NumSteps', 8),
    }

def _get_swg_template_values(self) -> Dict[str, Any]:
    """Get values for SWG wrapper template."""
    return {
        'TOP_MODULE_NAME': f"{self.onnx_node.name}_wrapper",
        'BIT_WIDTH': 8,
        'SIMD': self._safe_extract_value(self, 'SIMD', 4),
        'MMV_IN': 32,
        'MMV_OUT': 32,
        'IN_WIDTH_PADDED': 256,
        'OUT_WIDTH_PADDED': 256,
    }

def _get_generic_wrapper_values(self) -> Dict[str, Any]:
    """Get values for generic RTL wrapper."""
    return {
        'MODULE_NAME': f"{self.onnx_node.name}_wrapper",
        'DATA_WIDTH': self._extract_data_width(self),
        'PE_COUNT': self._safe_extract_value(self, 'PE', 1),
    }
```

#### **Update RTL Generation Method**
```python
def generate_hdl(self, model, fpgapart, clk):
    """Generate HDL code using template system."""
    # Store context
    self._current_model = model
    self._current_fpgapart = fpgapart
    self._current_clk = clk
    
    # Generate using template system
    rtl_code = self.generate_code()  # Uses get_template_name() and get_template_values()
    
    # Write to file
    code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
    module_name = self._generate_module_name(self)
    rtl_path = os.path.join(code_gen_dir, f"{module_name}.sv")
    
    with open(rtl_path, "w") as f:
        f.write(rtl_code)
    
    # Set node attributes
    self.set_nodeattr("gen_top_module", module_name)
```

---

## Phase 4: Simplification Updates

### **4.1 Simplified Library Resolver Integration**

**File**: `src/finn/codegen/simple_library_resolver.py` (as in previous plan)

### **4.2 Simplified File Manager Integration**

**File**: `src/finn/codegen/simple_file_manager.py` (as in previous plan)

### **4.3 Update Module Exports**

**File**: `src/finn/codegen/__init__.py`
```python
# Replace complex components with simplified versions
from .simple_library_resolver import SimpleLibraryResolver as LibraryResolver
from .simple_file_manager import SimpleFileManager as FileManager

# Export existing backend classes (no new generator classes needed)
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

__all__ = [
    # Core components
    'TemplateEngine',
    'BackendRegistry', 
    'CodegenConfig',
    
    # Backend classes (existing, updated)
    'HLSBackend',
    'RTLBackend',
    
    # Utilities (simplified versions)
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

## Phase 5: Integration and Testing

### **5.1 Update Template Engine for Backward Compatibility**

**File**: `src/finn/codegen/template_engine.py` (add method)
```python
def render_legacy(self, template_content: str, replacements: Dict[str, Any]) -> str:
    """
    Render using legacy string replacement method.
    
    This method provides backward compatibility with existing code
    that uses simple string replacement (e.g., $KEY$ -> value).
    """
    result = template_content
    for key, value in replacements.items():
        # Handle both $KEY$ and KEY formats
        placeholder = f"${key}$" if not key.startswith('$') else key
        if not placeholder.endswith('$'):
            placeholder += '$'
        result = result.replace(placeholder, str(value))
    return result
```

### **5.2 Integration Test**

**File**: `test_backend_integration.py`
```python
"""Test that backend integration works with new template system."""

def test_hls_backend_template_integration():
    """Test HLS backend uses new template system."""
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
    
    # Create mock HLS backend
    class TestHLSBackend(HLSBackend):
        TEMPLATE_NAME = "hls/docompute.cpp.j2"
        
        def get_template_values(self, template_name):
            return {
                'AP_INT_MAX_W': 4096,
                'GLOBALS': '// test globals',
                'DEFINES': '// test defines',
                'PRAGMAS': '// test pragmas',
                'STREAMDECLARATIONS': '// test streams',
                'READNPYDATA': '// test read',
                'DOCOMPUTE': '// test compute',
                'DATAOUTSTREAM': '// test output',
                'SAVEASCNPY': '// test save',
            }
    
    backend = TestHLSBackend()
    code = backend.generate_code()
    
    # Should generate valid C++ code
    assert '#define AP_INT_MAX_W 4096' in code
    assert '// test globals' in code
    assert len(code) > 100

def test_rtl_backend_template_integration():
    """Test RTL backend uses new template system."""
    from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
    
    # Create mock RTL backend
    class TestRTLBackend(RTLBackend):
        TEMPLATE_NAME = "rtl/swg_wrapper.v.j2"
        
        def get_template_values(self, template_name):
            return {
                'TOP_MODULE_NAME': 'test_module',
                'BIT_WIDTH': 8,
                'SIMD': 4,
                'MMV_IN': 32,
                'MMV_OUT': 32,
                'IN_WIDTH_PADDED': 256,
                'OUT_WIDTH_PADDED': 256,
            }
    
    backend = TestRTLBackend()
    code = backend.generate_code()
    
    # Should generate valid SystemVerilog code
    assert 'module test_module' in code
    assert 'parameter BIT_WIDTH = 8' in code
    assert len(code) > 100
```

---

## Success Criteria

1. **✅ Template Infrastructure**: All 7 templates converted and working
2. **✅ Backend Integration**: HLSBackend and RTLBackend use new template system
3. **✅ Backward Compatibility**: All existing code continues working
4. **✅ Simplified Components**: Library resolver and file manager simplified
5. **✅ End-to-End Testing**: Full code generation workflow validated

## Timeline

- **Week 1**: Template creation and validation
- **Week 2**: HLSBackend integration and testing
- **Week 3**: RTLBackend integration and testing  
- **Week 4**: Simplification, integration testing, and validation

## Key Benefits

1. **Maintains Existing Architecture**: Updates existing classes rather than replacing them
2. **Backward Compatible**: All existing HLS/RTL operations continue working
3. **Template-Driven**: Leverages new Jinja2 template system for flexibility
4. **Simplified Dependencies**: Reduces complexity in utility components
5. **Incremental Migration**: Existing operations can be updated gradually

This revised approach correctly targets the existing backend infrastructure rather than creating new parallel systems.