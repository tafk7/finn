# Elegant Breaking Changes Plan for Template Value Provider System

## 🎯 **Philosophy: Elegant Breaking Changes**

The user explicitly approves breaking changes to improve design elegantly. This opens up opportunities to make **principled architectural improvements** rather than just backward-compatible patches.

---

## 📋 **Required Changes by Component**

### **1. HWCustomOp Base Class Changes**

#### **Current State**
```python
# File: src/finn/custom_op/fpgadataflow/hwcustomop.py
class HWCustomOp(CustomOp):
    """Current base class - no template interface."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
    
    def get_nodeattr_types(self):
        """Returns operation-specific attributes."""
        pass
    
    # Various operation-specific methods...
```

#### **New Elegant Design**
```python
# File: src/finn/custom_op/fpgadataflow/hwcustomop.py
from abc import abstractmethod
from finn.codegen.template_value_provider import TemplateValueProvider

class HWCustomOp(CustomOp, TemplateValueProvider):
    """Enhanced base class with mandatory template interface."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
    
    # ✅ ELEGANT BREAKING CHANGE: Make template support mandatory
    @abstractmethod
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """REQUIRED: Every operation must implement template value provision."""
        pass
    
    @abstractmethod
    def get_supported_templates(self) -> Set[str]:
        """REQUIRED: Every operation must declare template support."""
        pass
    
    def supports_template(self, template_name: str) -> bool:
        """Default implementation using supported_templates."""
        return template_name in self.get_supported_templates()
    
    # ✅ ELEGANT ADDITION: Template value helpers
    def _safe_get_nodeattr(self, attr_name: str, default_value=None):
        """Safe attribute access with fallback."""
        try:
            return self.get_nodeattr(attr_name)
        except (AttributeError, KeyError):
            return default_value
    
    def _get_common_template_values(self) -> Dict[str, Any]:
        """Common values all operations can provide."""
        return {
            'op_type': self.onnx_node.op_type,
            'input_width': self.get_instream_width(),
            'output_width': self.get_outstream_width(),
            'exp_cycles': self.get_exp_cycles(),
        }
```

**Breaking Change Impact**: ✅ **Elegant** - Forces all operations to be explicit about template support, improving clarity.

---

### **2. Individual Custom Operation Changes**

#### **2.1 Thresholding Operation Refactor**

**Current Implementation**:
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py
class Thresholding(HWCustomOp):
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, ""),
            "NumChannels": ("i", True, ""),
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "ActVal": ("f", True, 0.0),
        }
    
    # No template interface - relies on framework assumptions
```

**New Elegant Implementation**:
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py
class Thresholding(HWCustomOp):
    """Thresholding operation with explicit template interface."""
    
    def get_nodeattr_types(self):
        # ✅ Attributes unchanged - no breaking change here
        return {
            "PE": ("i", True, ""),
            "NumChannels": ("i", True, ""),
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "ActVal": ("f", True, 0.0),
        }
    
    # ✅ REQUIRED: Template interface implementation
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Provide template values based on thresholding characteristics."""
        
        base_values = self._get_common_template_values()
        
        if template_name == "hls_basic":
            return {**base_values, **self._get_basic_values()}
        elif template_name == "hls_streaming":
            return {**base_values, **self._get_streaming_values()}
        elif template_name == "hls_parallel":
            return {**base_values, **self._get_parallel_values()}
        else:
            raise UnsupportedTemplateError(f"Thresholding doesn't support '{template_name}'")
    
    def get_supported_templates(self) -> Set[str]:
        """Thresholding supports these templates."""
        return {"hls_basic", "hls_streaming", "hls_parallel"}
    
    def _get_basic_values(self) -> Dict[str, Any]:
        """Basic thresholding template values."""
        return {
            'pe_factor': self.get_nodeattr("PE"),
            'num_channels': self.get_nodeattr("NumChannels"),
            'weight_data_type': self.get_nodeattr("weightDataType"),
            'activation_value': self.get_nodeattr("ActVal"),
            # Thresholding-specific defaults
            'mem_mode': 'const_embedded',
            'ram_style': 'distributed',
            'simd_factor': 1,
        }
    
    def _get_streaming_values(self) -> Dict[str, Any]:
        """Streaming-specific values for thresholding."""
        return {
            'pe_factor': self.get_nodeattr("PE"),
            'simd_factor': 1,  # Thresholding doesn't use SIMD
            'parallelization_strategy': 'pe_only',
            'channels_per_pe': self.get_nodeattr("NumChannels") // self.get_nodeattr("PE"),
        }
    
    def _get_parallel_values(self) -> Dict[str, Any]:
        """Parallel implementation values."""
        return {
            'pe_factor': self.get_nodeattr("PE"),
            'parallel_channels': self.get_nodeattr("NumChannels"),
            'lookup_table_style': 'distributed_lut',
        }
```

#### **2.2 MVAU Operation Refactor**

**Current Implementation**:
```python
# File: src/finn/custom_op/fpgadataflow/matrixvectoractivation.py
class MatrixVectorActivation(HWCustomOp):
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, ""),
            "SIMD": ("i", True, ""),
            "mem_mode": ("s", True, ""),
            "ram_style": ("s", True, ""),
            # ... other MVAU attributes
        }
```

**New Elegant Implementation**:
```python
# File: src/finn/custom_op/fpgadataflow/matrixvectoractivation.py
class MatrixVectorActivation(HWCustomOp):
    """MVAU operation with comprehensive template interface."""
    
    def get_nodeattr_types(self):
        # ✅ Attributes unchanged
        return {
            "PE": ("i", True, ""),
            "SIMD": ("i", True, ""),
            "mem_mode": ("s", True, ""),
            "ram_style": ("s", True, ""),
            "MW": ("i", True, ""),
            "MH": ("i", True, ""),
            # ... other attributes
        }
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """MVAU template values using actual attributes."""
        
        base_values = self._get_common_template_values()
        
        if template_name == "hls_mvau_streaming":
            return {**base_values, **self._get_mvau_streaming_values()}
        elif template_name == "hls_mvau_parallel":
            return {**base_values, **self._get_mvau_parallel_values()}
        elif template_name == "hls_basic":
            return {**base_values, **self._get_basic_mvau_values()}
        else:
            raise UnsupportedTemplateError(f"MVAU doesn't support '{template_name}'")
    
    def get_supported_templates(self) -> Set[str]:
        """MVAU supports specialized templates."""
        return {"hls_basic", "hls_mvau_streaming", "hls_mvau_parallel"}
    
    def _get_mvau_streaming_values(self) -> Dict[str, Any]:
        """MVAU streaming implementation."""
        return {
            'mem_mode': self.get_nodeattr("mem_mode"),        # MVAU has this
            'ram_style': self.get_nodeattr("ram_style"),      # MVAU has this
            'pe_factor': self.get_nodeattr("PE"),
            'simd_factor': self.get_nodeattr("SIMD"),         # MVAU has this
            'matrix_width': self.get_nodeattr("MW"),
            'matrix_height': self.get_nodeattr("MH"),
            'parallelization_strategy': 'pe_simd',
        }
```

---

### **3. HLS Backend Changes**

#### **Current HLS Backend Implementation**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
class HLSBackend:
    def generate_hls_code(self, operation):
        # Current approach - direct attribute access
        pe = operation.get_nodeattr("PE")              # May fail
        simd = operation.get_nodeattr("SIMD")          # May fail
        mem_mode = operation.get_nodeattr("mem_mode")  # May fail
```

#### **New Elegant HLS Backend**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
from finn.codegen.unified_generator import UnifiedCodeGenerator

class HLSBackend:
    """Template-driven HLS backend."""
    
    def __init__(self):
        self.code_generator = UnifiedCodeGenerator()
    
    def generate_hls_code(self, operation: HWCustomOp) -> str:
        """Generate HLS using operation's template interface."""
        
        # ✅ ELEGANT: Template selection based on operation capabilities
        template_name = self._select_optimal_template(operation)
        
        # ✅ ELEGANT: Let operation provide values
        return self.code_generator.generate_code(operation, template_name)
    
    def _select_optimal_template(self, operation: HWCustomOp) -> str:
        """Select best template based on operation's declared support."""
        
        # Priority order for template selection
        template_priorities = [
            "hls_mvau_streaming",      # MVAU-specific optimized
            "hls_thresholding",        # Thresholding-specific
            "hls_channelwise",         # Channelwise-specific
            "hls_streaming",           # Generic streaming
            "hls_parallel",            # Generic parallel
            "hls_basic",               # Basic fallback
        ]
        
        for template in template_priorities:
            if operation.supports_template(template):
                return template
        
        raise ValueError(f"Operation {operation.__class__.__name__} supports no known templates")
```

**Breaking Change**: ✅ **Elegant** - HLS backend becomes operation-agnostic, much cleaner.

---

### **4. RTL Backend Changes**

#### **Current RTL Backend**
```python
# File: src/finn/custom_op/fpgadataflow/rtlbackend.py
class RTLBackend:
    def generate_rtl_code(self, operation):
        # Operation-specific RTL generation logic
        if isinstance(operation, MatrixVectorActivation):
            # MVAU-specific RTL
        elif isinstance(operation, Thresholding):
            # Thresholding-specific RTL
```

#### **New Elegant RTL Backend**
```python
# File: src/finn/custom_op/fpgadataflow/rtlbackend.py
class RTLBackend:
    """Template-driven RTL backend."""
    
    def __init__(self):
        self.code_generator = UnifiedCodeGenerator()
    
    def generate_rtl_code(self, operation: HWCustomOp) -> str:
        """Generate RTL using operation's template interface."""
        
        # ✅ ELEGANT: RTL template selection
        template_name = self._select_rtl_template(operation)
        
        return self.code_generator.generate_code(operation, template_name)
    
    def _select_rtl_template(self, operation: HWCustomOp) -> str:
        """Select appropriate RTL template."""
        
        rtl_templates = [
            "rtl_mvau_streaming",
            "rtl_thresholding", 
            "rtl_streaming",
            "rtl_basic",
        ]
        
        for template in rtl_templates:
            if operation.supports_template(template):
                return template
                
        raise ValueError(f"No RTL template found for {operation.__class__.__name__}")
```

---

### **5. Template Library Changes**

#### **Current Template Organization**
```
src/finn/codegen/templates/
├── hls/
│   └── mvau_streaming.cpp.j2     # MVAU-specific only
└── rtl/
    └── mvau_wrapper.v.j2         # MVAU-specific only
```

#### **New Elegant Template Organization**
```
src/finn/codegen/templates/
├── hls/
│   ├── common/
│   │   ├── hls_basic.cpp.j2           # Universal basic template
│   │   ├── hls_streaming.cpp.j2       # Generic streaming
│   │   └── hls_parallel.cpp.j2        # Generic parallel
│   ├── operation_specific/
│   │   ├── hls_mvau_streaming.cpp.j2  # MVAU-optimized
│   │   ├── hls_thresholding.cpp.j2    # Thresholding-optimized
│   │   └── hls_channelwise.cpp.j2     # Channelwise-optimized
└── rtl/
    ├── common/
    │   ├── rtl_basic.v.j2             # Universal RTL
    │   └── rtl_streaming.v.j2         # Generic streaming RTL
    └── operation_specific/
        ├── rtl_mvau_streaming.v.j2    # MVAU-optimized RTL
        └── rtl_thresholding.v.j2      # Thresholding-optimized RTL
```

**New Template Content Example**:
```jinja2
{# File: templates/hls/common/hls_streaming.cpp.j2 #}
// Universal streaming template - works for all operations
#include "{{ op_type }}_streaming.hpp"

void {{ op_type }}_streaming(
    hls::stream<ap_uint<{{ input_width }}>>& in_stream,
    hls::stream<ap_uint<{{ output_width }}>>& out_stream
) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    
    // Memory configuration based on operation type
    {% if mem_mode == 'const_embedded' %}
        // Embedded constants (good for Thresholding)
        const ap_uint<{{ weight_data_type }}> thresholds[{{ num_channels }}] = { /* ... */ };
    {% elif mem_mode == 'internal_embedded' %}
        // Internal BRAM (good for MVAU)
        #pragma HLS resource variable=weights core=RAM_2P_BRAM
        static ap_uint<{{ weight_data_type }}> weights[{{ matrix_width * matrix_height }}];
    {% endif %}
    
    // Parallelization based on operation capabilities  
    {% if parallelization_strategy == 'pe_only' %}
        // PE-only parallelization (Thresholding)
        for (int pe = 0; pe < {{ pe_factor }}; pe++) {
            #pragma HLS UNROLL
            // Process {{ channels_per_pe }} channels per PE
        }
    {% elif parallelization_strategy == 'pe_simd' %}
        // PE+SIMD parallelization (MVAU)
        for (int pe = 0; pe < {{ pe_factor }}; pe++) {
            for (int simd = 0; simd < {{ simd_factor }}; simd++) {
                #pragma HLS UNROLL
                // Matrix-vector multiply with SIMD
            }
        }
    {% endif %}
}
```

---

## 🔄 **Migration Strategy for Breaking Changes**

### **Phase 1: Foundation (Week 1)**
- [ ] **Create new base interfaces**
  - Create `TemplateValueProvider` interface
  - Create `UnifiedCodeGenerator` service
  - Create abstract method stubs in `HWCustomOp`

### **Phase 2: Core Operations (Week 2)**
- [ ] **Implement template interfaces for critical operations**
  - Update `Thresholding` operation
  - Update `MatrixVectorActivation` operation
  - Update `AddStreams`, `ChannelwiseOp` operations
  - Create comprehensive unit tests

### **Phase 3: Template Library (Week 3)**
- [ ] **Create new template library**
  - Design universal templates
  - Create operation-specific optimized templates
  - Migrate existing template content
  - Add template validation tools

### **Phase 4: Backend Refactoring (Week 4)**
- [ ] **Update backends to use new system**
  - Refactor `HLSBackend` to use templates
  - Refactor `RTLBackend` to use templates
  - Remove operation-specific logic from backends
  - Add comprehensive integration tests

### **Phase 5: Remaining Operations (Week 5)**
- [ ] **Complete operation migration**
  - Update all remaining custom operations
  - Add template interfaces to specialized operations
  - Create migration validation tools
  - Performance testing and optimization

---

## ✅ **Benefits of These Breaking Changes**

### **Architectural Benefits**
1. **Clean Separation**: Templates, operations, and backends have clear responsibilities
2. **Extensibility**: New operations require zero backend changes  
3. **Maintainability**: Template evolution doesn't break operations
4. **Testability**: Each component can be tested independently

### **Developer Experience Benefits**
1. **Clear Contracts**: Explicit template support declarations
2. **Type Safety**: Template value types enforced
3. **Documentation**: Self-documenting template requirements
4. **Error Prevention**: Compile-time validation of template support

### **System Benefits**
1. **Performance**: Operation-optimized templates
2. **Consistency**: All operations use same interface pattern
3. **Reliability**: No more attribute access errors
4. **Flexibility**: Operations can compute values dynamically

---

## 🎯 **Breaking Change Migration Checklist**

### **For Each Custom Operation**
- [ ] Add `get_template_values()` implementation
- [ ] Add `get_supported_templates()` implementation
- [ ] Create operation-specific template value methods
- [ ] Update unit tests to verify template interface
- [ ] Add integration tests with actual templates

### **For Each Backend**
- [ ] Remove direct `get_nodeattr()` calls
- [ ] Replace with `get_template_values()` calls
- [ ] Update template selection logic
- [ ] Add error handling for unsupported templates
- [ ] Update backend unit tests

### **For Each Template**
- [ ] Identify all placeholders used
- [ ] Document required vs optional placeholders
- [ ] Create template validation tests
- [ ] Add operation compatibility matrix
- [ ] Performance benchmark template rendering

---

## 🏁 **Expected Final State**

After implementing these elegant breaking changes:

### **What Gets Better**
✅ **Framework Simplicity**: Zero operation-specific logic in backends  
✅ **Operation Clarity**: Explicit template support declarations  
✅ **Template Reuse**: Same template works for multiple operations  
✅ **Extensibility**: New operations work immediately  
✅ **Maintainability**: Clean separation of concerns  

### **What Gets Broken (Intentionally)**
❌ **Direct Attribute Access**: Operations must implement template interface  
❌ **Implicit Template Support**: Operations must declare supported templates  
❌ **Backend Operation Coupling**: Backends become operation-agnostic  

### **Net Result**
🎉 **Elegant Architecture**: Clean, extensible, maintainable system that truly delivers "zero breaking changes" for future operations while requiring one-time migration of existing operations to a much better design pattern.

The breaking changes force better architecture while providing clear migration paths and immediate benefits.