# Focused Thresholding Code Generation Refactor Plan
**Minimal Changes - Template System Only**

## Overview

This plan focuses **exclusively** on refactoring the code generation system for Thresholding operations, eliminating legacy `code_gen_dict` and string replacement approaches while changing as little as possible in the actual operation logic.

---

## Current State - Exactly What We Have

### **Templates:**
1. **RTL Template**: `finn-rtllib/thresholding/hdl/thresholding_template_wrapper.v` (122 lines, `$VAR$` syntax)
2. **HLS Templates**: From `src/finn/custom_op/fpgadataflow/templates.py` (`$VAR$` syntax)
   - `docompute_template` - for cppsim
   - `docompute_template_timeout` - for cppsim with timeout
   - `ipgen_template` - for IP generation (cpp)
   - `ipgentcl_template` - for IP generation (tcl)

### **Backends:**
1. **HLS**: `src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py` (261 lines)
2. **RTL**: `src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py` (516 lines)

### **Base Operation:**
- `src/finn/custom_op/fpgadataflow/thresholding.py` (274 lines) - **KEEP UNCHANGED**

---

## Target State - What We Want

### **Templates (Jinja2):**
```
src/finn/codegen/templates/thresholding/
├── hls/
│   ├── docompute.cpp.j2              # From docompute_template
│   ├── docompute_timeout.cpp.j2      # From docompute_template_timeout  
│   ├── ipgen.cpp.j2                  # From ipgen_template
│   └── ipgen.tcl.j2                  # From ipgentcl_template
└── rtl/
    └── wrapper.v.j2                  # From thresholding_template_wrapper.v
```

### **Backends (Updated):**
- **ThresholdingHLS**: Use Jinja2 templates, eliminate `code_gen_dict`
- **Thresholding_rtl**: Use Jinja2 templates, eliminate string replacement

---

## Implementation Plan

### **Phase 1: Template Conversion (2 days)**

#### **1.1 Convert RTL Template**
```jinja2
{# src/finn/codegen/templates/thresholding/rtl/wrapper.v.j2 #}
module {{ MODULE_NAME_AXI_WRAPPER }} #(
	parameter  N = {{ N }},		// output precision
	parameter  WI = {{ WI }},	// input precision
	parameter  WT = {{ WT }},	// threshold precision
	parameter  C = {{ C }},	// Channels
	parameter  PE = {{ PE }},	// Processing Parallelism, requires C = k*PE

	parameter  SIGNED = {{ SIGNED }},	// signed inputs
	parameter  FPARG  = {{ FPARG }},	// floating-point inputs
	parameter  BIAS   = {{ BIAS }},		// offsetting the output

	parameter  THRESHOLDS_PATH = {{ THRESHOLDS_PATH }},	// Directory with initial threshold data
	parameter  USE_AXILITE = {{ USE_AXILITE }},	// Implement AXI-Lite for threshold read/write

	// Force Use of On-Chip Memory Blocks
	parameter  DEPTH_TRIGGER_URAM = {{ DEPTH_TRIGGER_URAM }},
	parameter  DEPTH_TRIGGER_BRAM = {{ DEPTH_TRIGGER_BRAM }},
	parameter  DEEP_PIPELINE = {{ DEEP_PIPELINE }},

	parameter  O_BITS = {{ O_BITS }}
)(
	// ... rest of template exactly the same, just {{ var }} instead of $VAR$
```

#### **1.2 Convert HLS Templates**
```jinja2
{# src/finn/codegen/templates/thresholding/hls/docompute.cpp.j2 #}
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W {{ AP_INT_MAX_W }}
#define HLS_NO_XIL_FPO_LIB
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
{{ GLOBALS }}

// defines for network parameters  
{{ DEFINES }}

int main(){
{{ PRAGMAS }}

{{ STREAMDECLARATIONS }}

{{ READNPYDATA }}

{{ DOCOMPUTE }}

{{ DATAOUTSTREAM }}

{{ SAVEASCNPY }}

}
```

### **Phase 2: Update HLS Backend (3 days)**

#### **2.1 Eliminate code_gen_dict Dependency**
```python
# src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py
class ThresholdingHLS(Thresholding, HLSBackend):
    """HLS backend for Thresholding operations - NO code_gen_dict dependency."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        # Remove any code_gen_dict initialization
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for Jinja2 templates - NO code_gen_dict."""
        if 'docompute' in template_name:
            return self._get_docompute_values(template_name)
        elif 'ipgen.cpp' in template_name:
            return self._get_ipgen_cpp_values()
        elif 'ipgen.tcl' in template_name:
            return self._get_ipgen_tcl_values()
        else:
            raise ValueError(f"Unsupported template: {template_name}")
    
    def _get_docompute_values(self, template_name: str) -> Dict[str, Any]:
        """Get values for docompute templates."""
        values = {
            'AP_INT_MAX_W': self.get_ap_int_max_w(),
            'GLOBALS': self._generate_globals(),
            'DEFINES': self._generate_defines("cppsim"),
            'PRAGMAS': self._generate_pragmas(),
            'STREAMDECLARATIONS': self._generate_stream_declarations(),
            'READNPYDATA': self._generate_read_npy_data(),
            'DOCOMPUTE': self._generate_docompute(),
            'DATAOUTSTREAM': self._generate_data_out_stream(),
            'SAVEASCNPY': self._generate_save_as_npy(),
        }
        
        if 'timeout' in template_name:
            values.update({
                'TIMEOUT_VALUE': self._generate_timeout_value(),
                'TIMEOUT_CONDITION': self._generate_timeout_condition(),
                'TIMEOUT_READ_STREAM': self._generate_timeout_read_stream(),
            })
        
        return values
    
    def _get_ipgen_cpp_values(self) -> Dict[str, Any]:
        """Get values for ipgen C++ template."""
        return {
            'AP_INT_MAX_W': self.get_ap_int_max_w(),
            'GLOBALS': self._generate_globals(),
            'DEFINES': self._generate_defines("ipgen"),
            'BLACKBOXFUNCTION': self._generate_blackbox_function(),
            'PRAGMAS': self._generate_pragmas(),
            'DOCOMPUTE': self._generate_docompute(),
        }
    
    def _get_ipgen_tcl_values(self) -> Dict[str, Any]:
        """Get values for ipgen TCL template."""
        return {
            'PROJECTNAME': f"project_{self.onnx_node.name}",
            'HWSRCDIR': self.get_nodeattr("code_gen_dir_ipgen"),
            'FPGAPART': self._current_fpgapart,  # Set during code generation
            'TOPFXN': self.onnx_node.name,
            'CLKPERIOD': self._current_clk,  # Set during code generation
            'DEFAULT_DIRECTIVES': '\n'.join(self.ipgen_default_directives()),
            'EXTRA_DIRECTIVES': '\n'.join(self.ipgen_extra_directives()),
        }
    
    # Direct template value generation methods (replace code_gen_dict usage)
    def _generate_globals(self) -> str:
        """Generate global includes directly."""
        # Call existing global_includes() but capture output as string
        # instead of populating code_gen_dict
        pass
    
    def _generate_defines(self, mode: str) -> str:
        """Generate defines directly."""
        # Call existing defines() but capture output as string
        pass
    
    # ... other _generate_* methods
    
    def code_generation_cppsim(self, model):
        """Generate C++ simulation code using Jinja2 templates."""
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        
        # Determine template
        if self.get_nodeattr("cpp_interface") == "hls_vector":
            template_name = "thresholding/hls/docompute_timeout.cpp.j2"
        else:
            template_name = "thresholding/hls/docompute.cpp.j2"
        
        # Generate code using template engine
        template_values = self.get_template_values(template_name)
        cpp_code = self.template_engine.render_template(template_name, template_values)
        
        # Write file
        cpp_path = os.path.join(path, f"execute_{node.op_type}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
    
    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate IP generation files using Jinja2 templates."""
        # Store context
        self._current_fpgapart = fpgapart
        self._current_clk = clk
        
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Generate C++ file
        cpp_values = self.get_template_values("thresholding/hls/ipgen.cpp.j2")
        cpp_code = self.template_engine.render_template("thresholding/hls/ipgen.cpp.j2", cpp_values)
        
        cpp_path = os.path.join(path, f"top_{node.name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        # Generate TCL file
        tcl_values = self.get_template_values("thresholding/hls/ipgen.tcl.j2")
        tcl_code = self.template_engine.render_template("thresholding/hls/ipgen.tcl.j2", tcl_values)
        
        tcl_path = os.path.join(path, f"hls_syn_{node.name}.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl_code)
```

### **Phase 3: Update RTL Backend (3 days)**

#### **3.1 Replace String Replacement with Jinja2**
```python
# src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py
class Thresholding_rtl(Thresholding, RTLBackend):
    """RTL backend for Thresholding - using Jinja2 templates."""
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for RTL template."""
        if template_name == "thresholding/rtl/wrapper.v.j2":
            return self._get_wrapper_values()
        else:
            raise ValueError(f"Unsupported template: {template_name}")
    
    def _get_wrapper_values(self) -> Dict[str, Any]:
        """Get values for RTL wrapper template."""
        # Extract values that were previously built in prepare_codegen_rtl_values()
        # but return them as a dictionary instead of populating code_gen_dict
        
        bias = self.get_nodeattr("ActVal")
        output_data_type = self.get_nodeattr("outputDataType")
        input_data_type = self.get_nodeattr("inputDataType")
        o_bitwidth = DataType[output_data_type].bitwidth()
        i_bitwidth = DataType[input_data_type].bitwidth()
        wdt = self.get_input_datatype(1)
        
        # Calculate O_BITS (existing logic)
        if bias >= 0:
            o_bits = math.ceil(math.log2(2**o_bitwidth + bias))
        else:
            o_bits = 1 + math.ceil(
                math.log2(-bias if -bias >= 2 ** (o_bitwidth - 1) else 2**o_bitwidth + bias)
            )
        
        return {
            'MODULE_NAME_AXI_WRAPPER': self.get_verilog_top_module_name(),
            'N': o_bitwidth,
            'WI': i_bitwidth,
            'WT': wdt.bitwidth(),
            'C': self.get_nodeattr("NumChannels"),
            'PE': self.get_nodeattr("PE"),
            'SIGNED': 1 if self.get_input_datatype(0).signed() else 0,
            'FPARG': 0 if self.get_input_datatype().is_integer() else 1,
            'BIAS': bias,
            'THRESHOLDS_PATH': f'"./{self.onnx_node.name}_"',
            'USE_AXILITE': self.get_nodeattr("runtime_writeable_weights"),
            'DEPTH_TRIGGER_URAM': self.get_nodeattr("depth_trigger_uram"),
            'DEPTH_TRIGGER_BRAM': self.get_nodeattr("depth_trigger_bram"),
            'DEEP_PIPELINE': self.get_nodeattr("deep_pipeline"),
            'O_BITS': int(o_bits),
        }
    
    def generate_hdl(self, model, fpgapart, clk):
        """Generate HDL using Jinja2 template instead of string replacement."""
        # Generate threshold data files (keep existing logic)
        self._generate_threshold_files(model)
        
        # Generate RTL wrapper using template
        template_values = self.get_template_values("thresholding/rtl/wrapper.v.j2")
        rtl_code = self.template_engine.render_template("thresholding/rtl/wrapper.v.j2", template_values)
        
        # Write RTL file
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        module_name = template_values['MODULE_NAME_AXI_WRAPPER']
        rtl_path = os.path.join(code_gen_dir, f"{module_name}.v")
        with open(rtl_path, "w") as f:
            f.write(rtl_code)
        
        # Copy library files (keep existing logic)
        self._copy_rtl_library_files(code_gen_dir)
        
        # Set node attributes (keep existing logic)
        self.set_nodeattr("gen_top_module", module_name)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
    
    def _generate_threshold_files(self, model):
        """Generate threshold data files (extract from prepare_codegen_rtl_values)."""
        # Extract the threshold file generation logic from 
        # prepare_codegen_rtl_values() into a separate method
        pass
    
    def _copy_rtl_library_files(self, code_gen_dir):
        """Copy RTL library files (extract from existing generate_hdl)."""
        # Extract the file copying logic from existing generate_hdl()
        sv_files = ["axilite_if.v", "thresholding.sv", "thresholding_axi.sv"]
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/thresholding/hdl"
        for sv_file in sv_files:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)
```

### **Phase 4: Testing & Validation (2 days)**

#### **4.1 Template Syntax Validation**
```python
def test_template_syntax():
    """Test that all templates have valid Jinja2 syntax."""
    from jinja2 import Environment, FileSystemLoader
    
    env = Environment(loader=FileSystemLoader('src/finn/codegen/templates'))
    
    # Test all templates can be loaded without syntax errors
    templates = [
        'thresholding/hls/docompute.cpp.j2',
        'thresholding/hls/docompute_timeout.cpp.j2', 
        'thresholding/hls/ipgen.cpp.j2',
        'thresholding/hls/ipgen.tcl.j2',
        'thresholding/rtl/wrapper.v.j2'
    ]
    
    for template_name in templates:
        template = env.get_template(template_name)
        assert template is not None
```

#### **4.2 Output Equivalence Testing**
```python
def test_output_equivalence():
    """Test that new Jinja2 system produces identical output to legacy."""
    # Create test thresholding node
    node = create_test_thresholding_node()
    
    # Generate with legacy system
    legacy_hls = create_legacy_thresholding_hls(node)
    legacy_output = legacy_hls.code_generation_cppsim(test_model)
    
    # Generate with new system  
    new_hls = ThresholdingHLS(node)
    new_output = new_hls.code_generation_cppsim(test_model)
    
    # Compare outputs
    assert normalize_code(legacy_output) == normalize_code(new_output)
```

---

## Key Benefits

### **1. Minimal Disruption**
- **Thresholding base class unchanged** - no risk to existing functionality
- **Existing API preserved** - drop-in replacement for backends
- **Template conversion only** - same logic, different rendering

### **2. Clean Architecture**
- **Eliminates code_gen_dict** - no more global state pollution
- **Eliminates string replacement** - proper template engine
- **Clear value extraction** - explicit template value methods

### **3. Maintainability** 
- **Template syntax validation** - catch errors early
- **Cleaner separation** - template logic separate from domain logic
- **Easier debugging** - template rendering traceable

---

## Success Criteria

1. **Zero code_gen_dict usage** in Thresholding backends
2. **Zero string replacement** in RTL backend  
3. **Identical output** to legacy system (character-by-character)
4. **All tests pass** with existing Thresholding workflows
5. **Templates validate** with Jinja2 syntax checker

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Template Conversion | 5 Jinja2 templates |
| 3-5 | HLS Backend Update | Updated ThresholdingHLS |
| 6-8 | RTL Backend Update | Updated Thresholding_rtl |
| 9-10 | Testing & Validation | Equivalence verification |

**Total: 10 days**

---

This focused plan addresses exactly what was requested: refactoring the code generation system for Thresholding with minimal changes to the operation itself, using the existing templates and backends as the foundation.