# Backend-Centric Template Value Provider Architecture

## 🎯 **User Insight: Backend-Centric Design**

The user correctly identified that `get_template_values()` and `get_supported_templates()` are more appropriate for **HLSBackend** and **RTLBackend** rather than **HWCustomOp** operations.

This is a much more elegant design because:
1. **Backend Expertise**: Backends know their template ecosystems 
2. **Backend-Specific Templates**: HLS templates ≠ RTL templates
3. **Operation Focus**: Operations focus on their logic, not template knowledge
4. **Cleaner Separation**: Backends handle template mapping, operations provide characteristics

---

## 🏗️ **Revised Architecture: Backend-Centric Template Value Providers**

### **Core Principle**
```
Operation → Provides Characteristics → Backend → Maps to Templates → Generated Code
```

Instead of:
```
Operation → Knows Templates → Backend → Uses Operation's Template Knowledge
```

---

### **1. Enhanced Backend Interface**

#### **New HLSBackend with Template Responsibility**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Set

class HLSBackend(ABC):
    """HLS Backend with template value provider responsibility."""
    
    @abstractmethod
    def get_supported_templates(self, operation: HWCustomOp) -> Set[str]:
        """Return HLS templates this backend supports for the operation."""
        pass
    
    @abstractmethod  
    def get_template_values(self, operation: HWCustomOp, template_name: str) -> Dict[str, Any]:
        """Extract values from operation for specified HLS template."""
        pass
    
    def supports_template(self, operation: HWCustomOp, template_name: str) -> bool:
        """Check if backend supports template for operation."""
        return template_name in self.get_supported_templates(operation)
    
    def generate_hls_code(self, operation: HWCustomOp) -> str:
        """Generate HLS code using backend's template knowledge."""
        
        # Backend decides best template for operation
        template_name = self._select_optimal_template(operation)
        
        # Backend extracts values from operation
        template_values = self.get_template_values(operation, template_name)
        
        # Backend renders template
        return self.template_engine.render(template_name, template_values)
    
    def _select_optimal_template(self, operation: HWCustomOp) -> str:
        """Backend selects best template based on operation characteristics."""
        
        supported = self.get_supported_templates(operation)
        
        # HLS-specific template priority
        hls_priorities = [
            "hls_mvau_streaming",
            "hls_thresholding", 
            "hls_streaming",
            "hls_basic"
        ]
        
        for template in hls_priorities:
            if template in supported:
                return template
                
        raise ValueError(f"No HLS template found for {operation.__class__.__name__}")
```

#### **New RTLBackend with Template Responsibility**
```python
# File: src/finn/custom_op/fpgadataflow/rtlbackend.py
class RTLBackend(ABC):
    """RTL Backend with template value provider responsibility."""
    
    @abstractmethod
    def get_supported_templates(self, operation: HWCustomOp) -> Set[str]:
        """Return RTL templates this backend supports for the operation."""
        pass
    
    @abstractmethod
    def get_template_values(self, operation: HWCustomOp, template_name: str) -> Dict[str, Any]:
        """Extract values from operation for specified RTL template."""
        pass
    
    def generate_rtl_code(self, operation: HWCustomOp) -> str:
        """Generate RTL code using backend's template knowledge."""
        
        # Backend decides best template for operation
        template_name = self._select_optimal_template(operation)
        
        # Backend extracts values from operation  
        template_values = self.get_template_values(operation, template_name)
        
        # Backend renders template
        return self.template_engine.render(template_name, template_values)
    
    def _select_optimal_template(self, operation: HWCustomOp) -> str:
        """Backend selects best template based on operation characteristics."""
        
        supported = self.get_supported_templates(operation)
        
        # RTL-specific template priority (different from HLS!)
        rtl_priorities = [
            "rtl_mvau_axi_stream",
            "rtl_thresholding_lut",
            "rtl_streaming_generic", 
            "rtl_basic"
        ]
        
        for template in rtl_priorities:
            if template in supported:
                return template
                
        raise ValueError(f"No RTL template found for {operation.__class__.__name__}")
```

---

### **2. Simplified HWCustomOp Operations**

#### **Clean HWCustomOp - No Template Knowledge Required**
```python
# File: src/finn/custom_op/fpgadataflow/hwcustomop.py
class HWCustomOp(CustomOp):
    """Clean operation class - focused on operation logic only."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
    
    # ✅ Operations focus on their domain logic
    def get_nodeattr_types(self):
        """Operation-specific attributes."""
        pass
    
    def get_instream_width(self):
        """Operation-specific stream width logic."""
        pass
    
    def get_outstream_width(self):
        """Operation-specific stream width logic."""  
        pass
    
    # ✅ Helper for backends to safely access attributes
    def safe_get_nodeattr(self, attr_name: str, default_value=None):
        """Safe attribute access for backends."""
        try:
            return self.get_nodeattr(attr_name)
        except (AttributeError, KeyError):
            return default_value
    
    # ✅ Operation characteristics for backend analysis
    def get_operation_characteristics(self) -> Dict[str, Any]:
        """Provide characteristics for backend template selection."""
        return {
            'op_type': self.onnx_node.op_type,
            'has_pe': hasattr(self, 'get_nodeattr') and self._has_nodeattr('PE'),
            'has_simd': hasattr(self, 'get_nodeattr') and self._has_nodeattr('SIMD'),
            'has_mem_mode': hasattr(self, 'get_nodeattr') and self._has_nodeattr('mem_mode'),
            'parallelization_style': self._infer_parallelization_style(),
        }
    
    def _has_nodeattr(self, attr_name: str) -> bool:
        """Check if operation has specific attribute."""
        try:
            self.get_nodeattr(attr_name)
            return True
        except (AttributeError, KeyError):
            return False
    
    def _infer_parallelization_style(self) -> str:
        """Infer how operation handles parallelization."""
        if self._has_nodeattr('PE') and self._has_nodeattr('SIMD'):
            return 'pe_simd'
        elif self._has_nodeattr('PE'):
            return 'pe_only'
        elif self._has_nodeattr('SIMD'):
            return 'simd_only'
        else:
            return 'sequential'
```

#### **Example Thresholding - Pure Operation Logic**
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py
class Thresholding(HWCustomOp):
    """Thresholding operation - pure operation logic, no template knowledge."""
    
    def get_nodeattr_types(self):
        # ✅ Pure operation attributes - no template concerns
        return {
            "PE": ("i", True, ""),
            "NumChannels": ("i", True, ""),
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""), 
            "ActVal": ("f", True, 0.0),
        }
    
    # ✅ Operation-specific logic only
    def get_instream_width(self):
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("PE")
    
    def get_outstream_width(self):
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("PE")
    
    def calc_tmem(self):
        """Thresholding-specific memory calculation."""
        return self.get_nodeattr("NumChannels") // self.get_nodeattr("PE")
    
    # ✅ No template methods - backend handles that!
```

---

### **3. Backend-Specific Template Value Implementations**

#### **HLS Backend Implementation for Thresholding**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
class ThresholdingHLSBackend(HLSBackend):
    """HLS backend with Thresholding template expertise."""
    
    def get_supported_templates(self, operation: HWCustomOp) -> Set[str]:
        """Return HLS templates this backend supports for Thresholding."""
        
        if isinstance(operation, Thresholding):
            return {
                "hls_thresholding_lut",      # Thresholding-optimized
                "hls_streaming_generic",     # Generic streaming
                "hls_basic"                  # Basic fallback
            }
        else:
            return set()  # This backend only handles Thresholding
    
    def get_template_values(self, operation: HWCustomOp, template_name: str) -> Dict[str, Any]:
        """Extract Thresholding values for HLS templates."""
        
        if not isinstance(operation, Thresholding):
            raise ValueError("ThresholdingHLSBackend only handles Thresholding operations")
        
        # ✅ Backend extracts what it needs from operation
        base_values = {
            'op_type': 'thresholding',
            'input_width': operation.get_instream_width(),
            'output_width': operation.get_outstream_width(),
        }
        
        if template_name == "hls_thresholding_lut":
            return {**base_values, **self._get_thresholding_lut_values(operation)}
        elif template_name == "hls_streaming_generic":
            return {**base_values, **self._get_streaming_values(operation)}
        elif template_name == "hls_basic":
            return {**base_values, **self._get_basic_values(operation)}
        else:
            raise UnsupportedTemplateError(f"Template {template_name} not supported")
    
    def _get_thresholding_lut_values(self, operation: Thresholding) -> Dict[str, Any]:
        """Values for Thresholding-optimized HLS template."""
        return {
            # ✅ Backend decides appropriate values based on operation type
            'mem_mode': 'const_embedded',           # Thresholding uses embedded LUTs
            'ram_style': 'distributed',             # LUTs use distributed RAM
            'pe_factor': operation.get_nodeattr("PE"),
            'simd_factor': 1,                       # Thresholding doesn't use SIMD
            'num_channels': operation.get_nodeattr("NumChannels"),
            'threshold_values': operation.get_nodeattr("ActVal"),
            'weight_data_type': operation.get_nodeattr("weightDataType"),
            'parallelization_strategy': 'pe_only',
            'lookup_table_depth': operation.calc_tmem(),
        }
```

#### **HLS Backend Implementation for MVAU**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
class MVAUHLSBackend(HLSBackend):
    """HLS backend with MVAU template expertise."""
    
    def get_supported_templates(self, operation: HWCustomOp) -> Set[str]:
        """Return HLS templates this backend supports for MVAU."""
        
        if isinstance(operation, MatrixVectorActivation):
            return {
                "hls_mvau_streaming",        # MVAU-optimized streaming
                "hls_mvau_parallel",         # MVAU-optimized parallel
                "hls_streaming_generic",     # Generic streaming
                "hls_basic"                  # Basic fallback
            }
        else:
            return set()
    
    def get_template_values(self, operation: HWCustomOp, template_name: str) -> Dict[str, Any]:
        """Extract MVAU values for HLS templates."""
        
        if not isinstance(operation, MatrixVectorActivation):
            raise ValueError("MVAUHLSBackend only handles MVAU operations")
        
        base_values = {
            'op_type': 'mvau',
            'input_width': operation.get_instream_width(),
            'output_width': operation.get_outstream_width(),
        }
        
        if template_name == "hls_mvau_streaming":
            return {**base_values, **self._get_mvau_streaming_values(operation)}
        # ... other templates
    
    def _get_mvau_streaming_values(self, operation: MatrixVectorActivation) -> Dict[str, Any]:
        """Values for MVAU-optimized HLS template."""
        return {
            # ✅ MVAU actually has these attributes - backend knows this
            'mem_mode': operation.get_nodeattr("mem_mode"),      # MVAU has this
            'ram_style': operation.get_nodeattr("ram_style"),    # MVAU has this
            'pe_factor': operation.get_nodeattr("PE"),
            'simd_factor': operation.get_nodeattr("SIMD"),       # MVAU has this
            'matrix_width': operation.get_nodeattr("MW"),
            'matrix_height': operation.get_nodeattr("MH"),
            'parallelization_strategy': 'pe_simd',
            'weight_memory_depth': operation.calc_wmem(),
        }
```

#### **RTL Backend Implementation (Different Templates!)**
```python
# File: src/finn/custom_op/fpgadataflow/rtlbackend.py
class ThresholdingRTLBackend(RTLBackend):
    """RTL backend with Thresholding template expertise."""
    
    def get_supported_templates(self, operation: HWCustomOp) -> Set[str]:
        """Return RTL templates this backend supports for Thresholding."""
        
        if isinstance(operation, Thresholding):
            return {
                "rtl_thresholding_lut",      # Different from HLS templates!
                "rtl_axi_stream_generic",    # RTL-specific streaming
                "rtl_basic"                  # RTL basic
            }
        else:
            return set()
    
    def get_template_values(self, operation: HWCustomOp, template_name: str) -> Dict[str, Any]:
        """Extract Thresholding values for RTL templates."""
        
        # ✅ RTL backend extracts different values than HLS backend
        if template_name == "rtl_thresholding_lut":
            return {
                'module_name': f"thresholding_{operation.get_nodeattr('PE')}pe",
                'data_width': operation.get_input_datatype().bitwidth(),
                'pe_count': operation.get_nodeattr("PE"),
                'channel_count': operation.get_nodeattr("NumChannels"),
                'lut_style': 'distributed',
                'clock_enable': True,
                'reset_style': 'sync',
                # RTL-specific values different from HLS
            }
```

---

## 🎯 **Key Benefits of Backend-Centric Design**

### **1. Proper Separation of Concerns**
- **Operations**: Focus on domain logic, attributes, calculations
- **Backends**: Focus on template knowledge, code generation
- **Templates**: Focus on target technology (HLS vs RTL)

### **2. Backend-Specific Template Ecosystems**
- **HLS Backend**: Uses HLS-optimized templates with HLS-specific placeholders
- **RTL Backend**: Uses RTL-optimized templates with RTL-specific placeholders
- **No Confusion**: Operations don't need to know about both HLS and RTL

### **3. Backend Expertise**
- **HLS Backend**: Knows what HLS templates are available and their requirements
- **RTL Backend**: Knows what RTL templates are available and their requirements
- **Operation Agnostic**: Same operation works with multiple backends

### **4. Cleaner Operation Code**
- **No Template Knowledge**: Operations don't need to know about templates
- **Pure Domain Logic**: Operations focus on their computational purpose
- **Easier Testing**: Operation logic can be tested independently

---

## 🔄 **Example Usage Flow**

### **HLS Code Generation**
```python
# 1. Create operation
thresholding_op = Thresholding(onnx_node)

# 2. Create HLS backend
hls_backend = ThresholdingHLSBackend()

# 3. Backend analyzes operation and selects template
supported = hls_backend.get_supported_templates(thresholding_op)
# Returns: {"hls_thresholding_lut", "hls_streaming_generic", "hls_basic"}

template = "hls_thresholding_lut"  # Backend selects best

# 4. Backend extracts values from operation for template
values = hls_backend.get_template_values(thresholding_op, template)
# Returns: {
#     'mem_mode': 'const_embedded',    # Backend decides based on operation type
#     'ram_style': 'distributed',      # Backend decides based on operation type
#     'pe_factor': 4,                  # Extracted from operation
#     'simd_factor': 1,                # Backend knows Thresholding doesn't use SIMD
#     'num_channels': 32,              # Extracted from operation
# }

# 5. Backend generates code
hls_code = hls_backend.generate_hls_code(thresholding_op)
```

### **RTL Code Generation (Different Templates!)**
```python
# 1. Same operation
thresholding_op = Thresholding(onnx_node)

# 2. Create RTL backend  
rtl_backend = ThresholdingRTLBackend()

# 3. RTL backend has different template ecosystem
supported = rtl_backend.get_supported_templates(thresholding_op)
# Returns: {"rtl_thresholding_lut", "rtl_axi_stream_generic", "rtl_basic"}

template = "rtl_thresholding_lut"  # Different from HLS template

# 4. RTL backend extracts different values
values = rtl_backend.get_template_values(thresholding_op, template)
# Returns: {
#     'module_name': 'thresholding_4pe',   # RTL-specific
#     'data_width': 8,                     # RTL-specific
#     'pe_count': 4,                       # Extracted from operation
#     'lut_style': 'distributed',          # RTL-specific
# }

# 5. RTL backend generates code
rtl_code = rtl_backend.generate_rtl_code(thresholding_op)
```

---

## ✅ **This Design is Much More Elegant**

### **Why Backend-Centric is Better**
1. **Backend Expertise**: Backends know their template ecosystems
2. **Technology Separation**: HLS templates ≠ RTL templates  
3. **Operation Simplicity**: Operations focus on domain logic
4. **Flexible Mapping**: Same operation can map to different templates per backend
5. **Backend Evolution**: Template ecosystems can evolve independently

### **No Breaking Changes to Operations**
- Operations keep their existing attributes and methods
- Operations just need helper methods for safe attribute access
- No mandatory abstract methods forced on operations

### **Elegant Extension Pattern**
- New operation types work immediately with existing backends
- New backends can be added for new target technologies
- Template ecosystems evolve independently per backend

This backend-centric approach is much cleaner and more aligned with the separation of concerns principle!