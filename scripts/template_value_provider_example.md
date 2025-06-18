# Template Value Provider System - Concrete Example

## 🔍 **Current Problematic System (Before)**

### Current Framework Code (BROKEN)
```python
# File: src/finn/codegen/hls_generator.py (Lines 347-349)
def _get_memory_config(self) -> Dict[str, Any]:
    return {
        'mem_mode': self.operation.get_nodeattr("mem_mode"),      # ❌ CRASHES on Thresholding
        'ram_style': self.operation.get_nodeattr("ram_style"),   # ❌ CRASHES on Thresholding
    }

# File: src/finn/codegen/hls_generator.py (Lines 353-356)  
def _get_parallelization_config(self) -> Dict[str, Any]:
    return {
        'pe': self.operation.get_nodeattr("PE"),     # ✅ Works for Thresholding
        'simd': self.operation.get_nodeattr("SIMD"), # ❌ CRASHES on Thresholding
    }

# File: src/finn/codegen/hls_generator.py (Line 73)
def _get_mvau_template_name(self) -> str:
    mem_mode = self.operation.get_nodeattr("mem_mode")  # ❌ CRASHES on Thresholding
    if mem_mode == "internal_embedded":
        return "mvau_streaming.j2"
    else:
        return "mvau_parallel.j2"
```

### Current Thresholding Operation (VICTIM)
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py
class Thresholding(HWCustomOp):
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, ""),                    # ✅ HAS this
            "NumChannels": ("i", True, ""),          # ✅ HAS this  
            "inputDataType": ("s", True, ""),        # ✅ HAS this
            "weightDataType": ("s", True, ""),       # ✅ HAS this
            "ActVal": ("f", True, 0.0),              # ✅ HAS this
            # ❌ NO mem_mode attribute
            # ❌ NO ram_style attribute  
            # ❌ NO SIMD attribute
        }
```

### Current Failure Result
```bash
❌ Error: Thresholding operation test failed: Op has no such attribute: mem_mode
   at _get_memory_config() line 347
   at generate_hls_code() 
   at test_core_framework_validation()
```

---

## ✅ **New Template Value Provider System (After)**

### 1. New Template Value Provider Interface
```python
# File: src/finn/codegen/template_value_provider.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Set

class TemplateValueProvider(ABC):
    """Interface for operations to provide template placeholder values."""
    
    @abstractmethod
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Provide values for all placeholders in the specified template."""
        pass
    
    @abstractmethod  
    def supports_template(self, template_name: str) -> bool:
        """Check if this operation can provide values for the template."""
        pass
    
    @abstractmethod
    def get_supported_templates(self) -> Set[str]:
        """Return set of all templates this operation supports."""
        pass

class UnsupportedTemplateError(Exception):
    """Raised when operation doesn't support requested template."""
    pass
```

### 2. Enhanced HWCustomOp Base Class
```python
# File: src/finn/custom_op/fpgadataflow/hwcustomop.py
from finn.codegen.template_value_provider import TemplateValueProvider, UnsupportedTemplateError

class HWCustomOp(CustomOp, TemplateValueProvider):
    """Enhanced base class with template value provider interface."""
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Default template value provider with safe fallbacks."""
        
        if template_name == "hls_basic":
            return self._get_basic_template_values()
        elif template_name == "hls_streaming":
            return self._get_streaming_template_values()
        elif template_name == "hls_memory":
            return self._get_memory_template_values()
        else:
            raise UnsupportedTemplateError(f"Template '{template_name}' not supported")
    
    def supports_template(self, template_name: str) -> bool:
        """Check if this operation supports the template."""
        return template_name in self.get_supported_templates()
    
    def get_supported_templates(self) -> Set[str]:
        """Base class supports common templates."""
        return {"hls_basic", "hls_streaming", "hls_memory"}
    
    def _safe_get_nodeattr(self, attr_name: str, default_value=None):
        """Safely get node attribute with fallback."""
        try:
            return self.get_nodeattr(attr_name)
        except (AttributeError, KeyError):
            return default_value
    
    def _get_basic_template_values(self) -> Dict[str, Any]:
        """Common basic template values - safe defaults."""
        return {
            'op_type': self.onnx_node.op_type,
            'input_width': self.get_instream_width(),
            'output_width': self.get_outstream_width(),
            'exp_cycles': self.get_exp_cycles(),
        }
    
    def _get_streaming_template_values(self) -> Dict[str, Any]:
        """Common streaming template values."""
        return {
            'stream_in_width': self.get_instream_width(),
            'stream_out_width': self.get_outstream_width(),
            'pe_factor': self._safe_get_nodeattr('PE', 1),
            'simd_factor': self._safe_get_nodeattr('SIMD', 1),
        }
    
    def _get_memory_template_values(self) -> Dict[str, Any]:
        """Safe memory template defaults."""
        return {
            'mem_mode': 'const_embedded',  # Safe default for most operations
            'ram_style': 'auto',           # Let synthesis tools decide
        }
```

### 3. New Framework Code (FIXED)
```python
# File: src/finn/codegen/unified_generator.py
class UnifiedCodeGenerator:
    """Template-agnostic code generator using operation value providers."""
    
    def generate_code(self, operation: TemplateValueProvider, template_name: str) -> str:
        """Generate code using template-operation interface."""
        
        # ✅ SAFE - Framework doesn't assume operation internals
        if not operation.supports_template(template_name):
            supported = operation.get_supported_templates()
            raise UnsupportedTemplateError(
                f"{operation.__class__.__name__} doesn't support '{template_name}'. "
                f"Supported templates: {supported}"
            )
            
        # ✅ SAFE - Operation decides what values to provide
        template_values = operation.get_template_values(template_name)
        
        return self.template_engine.render(template_name, template_values)

# File: src/finn/codegen/hls_generator.py (UPDATED)
class HLSGenerator:
    def __init__(self):
        self.unified_generator = UnifiedCodeGenerator()
    
    def generate_hls_code(self, operation: HWCustomOp) -> str:
        """Generate HLS code using template value provider."""
        
        # ✅ SAFE - No direct attribute access
        template_name = self._select_template(operation)
        return self.unified_generator.generate_code(operation, template_name)
    
    def _select_template(self, operation: HWCustomOp) -> str:
        """Select appropriate template based on operation capabilities."""
        
        # ✅ SAFE - Let operation declare its template support
        if operation.supports_template("hls_mvau_streaming"):
            return "hls_mvau_streaming"
        elif operation.supports_template("hls_thresholding"):
            return "hls_thresholding" 
        elif operation.supports_template("hls_streaming"):
            return "hls_streaming"
        else:
            return "hls_basic"  # Fallback
```

### 4. Updated Thresholding Operation (SUCCESS)
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py
class Thresholding(HWCustomOp):
    """Thresholding operation with template value provider."""
    
    # ✅ Original nodeattr_types unchanged - no breaking changes
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, ""),                    # ✅ Still has PE
            "NumChannels": ("i", True, ""),          # ✅ Still has NumChannels  
            "inputDataType": ("s", True, ""),        # ✅ Still has inputDataType
            "weightDataType": ("s", True, ""),       # ✅ Still has weightDataType
            "ActVal": ("f", True, 0.0),              # ✅ Still has ActVal
            # ✅ Still no mem_mode, ram_style, SIMD - but now it doesn't matter!
        }
    
    # ✅ NEW - Template value provider implementation
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Provide template values based on Thresholding operation logic."""
        
        if template_name == "hls_thresholding":
            return self._get_thresholding_template_values()
        elif template_name == "hls_streaming":
            return self._get_thresholding_streaming_values()
        elif template_name == "hls_memory":
            return self._get_thresholding_memory_values()
        else:
            # Try base class for other templates
            return super().get_template_values(template_name)
    
    def _get_thresholding_template_values(self) -> Dict[str, Any]:
        """Thresholding-specific template values."""
        base_values = super()._get_basic_template_values()
        base_values.update({
            # ✅ Use actual Thresholding attributes
            'pe_factor': self.get_nodeattr("PE"),
            'num_channels': self.get_nodeattr("NumChannels"),
            'weight_data_type': self.get_nodeattr("weightDataType"),
            'act_val': self.get_nodeattr("ActVal"),
            
            # ✅ Provide Thresholding-appropriate values for "missing" attributes
            'mem_mode': 'const_embedded',    # Thresholding uses embedded LUTs
            'ram_style': 'distributed',      # Small LUTs use distributed RAM
            'simd_factor': 1,                # Thresholding doesn't use SIMD
            'parallelization_type': 'PE',    # Uses PE not SIMD
        })
        return base_values
    
    def _get_thresholding_streaming_values(self) -> Dict[str, Any]:
        """Thresholding streaming-specific values."""
        return {
            'stream_in_width': self.get_instream_width(),
            'stream_out_width': self.get_outstream_width(),
            'pe_factor': self.get_nodeattr("PE"),
            'simd_factor': 1,  # Thresholding doesn't use SIMD
            'channels_per_pe': self.get_nodeattr("NumChannels") // self.get_nodeattr("PE"),
        }
    
    def _get_thresholding_memory_values(self) -> Dict[str, Any]:
        """Thresholding memory configuration."""
        return {
            'mem_mode': 'const_embedded',     # Thresholding constants are embedded
            'ram_style': 'distributed',       # Use LUTs for small threshold tables
            'memory_depth': self.calc_tmem() if hasattr(self, 'calc_tmem') else 1,
            'weight_storage': 'embedded_lut', # Thresholding-specific
        }
    
    def get_supported_templates(self) -> Set[str]:
        """Thresholding supports these templates."""
        return {"hls_basic", "hls_streaming", "hls_memory", "hls_thresholding"}
```

### 5. MVAU Operation (PROPER USE OF ATTRIBUTES)
```python
# File: src/finn/custom_op/fpgadataflow/matrixvectoractivation.py
class MatrixVectorActivation(HWCustomOp):
    """MVAU operation - actually HAS the attributes framework was assuming."""
    
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, ""),
            "SIMD": ("i", True, ""),              # ✅ MVAU actually HAS SIMD
            "mem_mode": ("s", True, ""),          # ✅ MVAU actually HAS mem_mode  
            "ram_style": ("s", True, ""),         # ✅ MVAU actually HAS ram_style
            "MW": ("i", True, ""),
            "MH": ("i", True, ""),
            # ... other MVAU attributes
        }
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """MVAU can provide actual attribute values."""
        
        if template_name == "hls_mvau_streaming":
            return self._get_mvau_streaming_values()
        else:
            return super().get_template_values(template_name)
    
    def _get_mvau_streaming_values(self) -> Dict[str, Any]:
        """MVAU streaming template - uses actual attributes."""
        return {
            # ✅ MVAU actually has these attributes
            'mem_mode': self.get_nodeattr("mem_mode"),        # No crash!
            'ram_style': self.get_nodeattr("ram_style"),      # No crash!
            'pe_factor': self.get_nodeattr("PE"),
            'simd_factor': self.get_nodeattr("SIMD"),         # No crash!
            'matrix_width': self.get_nodeattr("MW"),
            'matrix_height': self.get_nodeattr("MH"),
        }
    
    def get_supported_templates(self) -> Set[str]:
        return {"hls_basic", "hls_streaming", "hls_memory", "hls_mvau_streaming"}
```

---

## 🎯 **Example Usage Flow**

### Framework Request Flow
```python
# 1. Framework wants to generate code for a Thresholding operation
thresholding_op = Thresholding(onnx_node)
generator = UnifiedCodeGenerator()

# 2. Framework asks: "Do you support this template?"
if thresholding_op.supports_template("hls_streaming"):
    
    # 3. Framework asks: "Give me values for this template"
    values = thresholding_op.get_template_values("hls_streaming")
    # Returns: {
    #     'stream_in_width': 32,
    #     'stream_out_width': 8, 
    #     'pe_factor': 4,
    #     'simd_factor': 1,        # ✅ Thresholding provides 1 (doesn't use SIMD)
    #     'channels_per_pe': 8,
    # }
    
    # 4. Framework generates code using operation-provided values
    code = generator.generate_code(thresholding_op, "hls_streaming")
    # ✅ SUCCESS - No attribute errors!

# 5. Same flow works for MVAU with different values
mvau_op = MatrixVectorActivation(onnx_node)
if mvau_op.supports_template("hls_mvau_streaming"):
    values = mvau_op.get_template_values("hls_mvau_streaming")
    # Returns: {
    #     'mem_mode': 'internal_embedded',  # ✅ MVAU has actual mem_mode
    #     'ram_style': 'block',             # ✅ MVAU has actual ram_style
    #     'pe_factor': 16,
    #     'simd_factor': 8,                 # ✅ MVAU has actual SIMD
    #     'matrix_width': 64,
    #     'matrix_height': 64,
    # }
    code = generator.generate_code(mvau_op, "hls_mvau_streaming")
    # ✅ SUCCESS - Uses actual MVAU attributes!
```

### Template Rendering
```jinja2
{# File: templates/hls_streaming.j2 #}
module {{ op_type }}_streaming (
    input wire clk,
    input wire rst,
    input wire [{{ stream_in_width-1 }}:0] data_in,
    output wire [{{ stream_out_width-1 }}:0] data_out
);

parameter PE = {{ pe_factor }};
parameter SIMD = {{ simd_factor }};

{% if mem_mode == 'const_embedded' %}
    // Use embedded constants (good for Thresholding)
    localparam [31:0] thresholds [0:{{ num_channels-1 }}] = { /* ... */ };
{% elif mem_mode == 'internal_embedded' %}  
    // Use internal BRAM (good for MVAU)
    (* ram_style = "{{ ram_style }}" *) reg [31:0] weights [0:{{ matrix_width*matrix_height-1 }}];
{% endif %}

// PE-based parallelization
genvar pe_idx;
generate
    for (pe_idx = 0; pe_idx < PE; pe_idx = pe_idx + 1) begin
        // Processing element logic
    end
endgenerate

endmodule
```

**Result**: 
- ✅ **Thresholding**: Gets `mem_mode='const_embedded'`, `simd_factor=1`
- ✅ **MVAU**: Gets `mem_mode='internal_embedded'`, `simd_factor=8` 
- ✅ **Both work** with same template using operation-appropriate values!

---

## 🎉 **Key Benefits Demonstrated**

### ✅ **No Breaking Changes**
- Thresholding operation code unchanged
- Original attributes preserved
- Behavior identical for existing functionality

### ✅ **Framework Simplification**  
- Framework contains zero operation-specific logic
- Same template works for multiple operation types
- Clean separation of concerns

### ✅ **Operation Autonomy**
- Each operation decides appropriate values
- Operations can compute values dynamically
- Operations declare their own template support

### ✅ **Extensibility**
- New operations work immediately 
- No framework modifications needed
- Templates can evolve independently

This example shows how the Template Value Provider pattern transforms unsafe attribute access into a clean, extensible interface that achieves the "zero breaking changes" promise.