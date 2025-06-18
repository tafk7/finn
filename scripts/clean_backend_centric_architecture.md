# Clean Backend-Centric Architecture - Final Design

## 🎯 **User Refinements: Even Cleaner Design**

The user correctly identified two over-engineering issues:

1. **`safe_get_nodeattr` is undesirable** - Required attributes should fail clearly, not silently return defaults
2. **`get_operation_characteristics` is unnecessary** - Operations should just define attributes and let backends decide how to use them

This leads to an even cleaner design: **Operations define their domain, Backends decide how to use it.**

---

## 🏗️ **Final Clean Architecture**

### **Core Principle**
```
Operation → Defines Attributes & Helpers → Backend → Extracts What It Needs → Templates
```

**Clean Separation**:
- **Operations**: Pure domain logic with clear attribute definitions
- **Backends**: Template expertise and operation-to-template mapping logic
- **Templates**: Target technology specifics

---

### **1. Minimal, Clean HWCustomOp**

```python
# File: src/finn/custom_op/fpgadataflow/hwcustomop.py
class HWCustomOp(CustomOp):
    """Clean operation base class - pure domain focus."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
    
    # ✅ Operations define their domain through attributes
    @abstractmethod
    def get_nodeattr_types(self):
        """Define operation-specific attributes."""
        pass
    
    # ✅ Operations provide domain-specific calculations
    def get_instream_width(self):
        """Calculate input stream width."""
        pass
    
    def get_outstream_width(self):
        """Calculate output stream width."""
        pass
    
    def get_exp_cycles(self):
        """Calculate expected cycles."""
        pass
    
    # ✅ That's it - no template knowledge, no "safe" accessors, no characteristics
```

### **2. Clean Operation Implementations**

#### **Thresholding - Pure Domain Logic**
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py
class Thresholding(HWCustomOp):
    """Thresholding operation - pure domain definition."""
    
    def get_nodeattr_types(self):
        # ✅ Clear attribute definition - if missing, should fail
        return {
            "PE": ("i", True, ""),
            "NumChannels": ("i", True, ""),
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "ActVal": ("f", True, 0.0),
        }
    
    # ✅ Domain-specific helper methods
    def get_instream_width(self):
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("PE")
    
    def get_outstream_width(self):
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("PE")
    
    def calc_tmem(self):
        """Thresholding-specific memory depth calculation."""
        return self.get_nodeattr("NumChannels") // self.get_nodeattr("PE")
    
    def get_threshold_count(self):
        """Number of threshold values needed."""
        return self.get_nodeattr("NumChannels")
    
    # ✅ No template methods, no safe accessors, no characteristics
    # ✅ If PE is missing, get_nodeattr("PE") will fail clearly
```

#### **MVAU - Pure Domain Logic**
```python
# File: src/finn/custom_op/fpgadataflow/matrixvectoractivation.py
class MatrixVectorActivation(HWCustomOp):
    """MVAU operation - pure domain definition."""
    
    def get_nodeattr_types(self):
        # ✅ MVAU actually has these attributes
        return {
            "PE": ("i", True, ""),
            "SIMD": ("i", True, ""),
            "mem_mode": ("s", True, ""),
            "ram_style": ("s", True, ""),
            "MW": ("i", True, ""),
            "MH": ("i", True, ""),
        }
    
    # ✅ MVAU-specific helper methods
    def calc_wmem(self):
        """Weight memory depth calculation."""
        return self.get_nodeattr("MW") * self.get_nodeattr("MH") // self.get_nodeattr("SIMD")
    
    def get_matrix_dimensions(self):
        """Return matrix width and height."""
        return self.get_nodeattr("MW"), self.get_nodeattr("MH")
    
    def get_weight_storage_requirement(self):
        """Calculate weight storage needs."""
        return self.get_nodeattr("MW") * self.get_nodeattr("MH")
    
    # ✅ No template methods - backend decides how to use these
```

---

### **3. Backend Template Value Extraction**

#### **HLS Backend for Thresholding**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
class ThresholdingHLSBackend(HLSBackend):
    """HLS backend that knows how to use Thresholding operations."""
    
    def get_supported_templates(self, operation: HWCustomOp) -> Set[str]:
        if isinstance(operation, Thresholding):
            return {"hls_thresholding_lut", "hls_streaming_generic", "hls_basic"}
        return set()
    
    def get_template_values(self, operation: HWCustomOp, template_name: str) -> Dict[str, Any]:
        """Extract values from Thresholding operation for HLS templates."""
        
        if not isinstance(operation, Thresholding):
            raise ValueError("ThresholdingHLSBackend only handles Thresholding")
        
        if template_name == "hls_thresholding_lut":
            return self._extract_thresholding_lut_values(operation)
        elif template_name == "hls_streaming_generic":
            return self._extract_streaming_values(operation)
        elif template_name == "hls_basic":
            return self._extract_basic_values(operation)
        else:
            raise UnsupportedTemplateError(f"Template {template_name} not supported")
    
    def _extract_thresholding_lut_values(self, operation: Thresholding) -> Dict[str, Any]:
        """Extract values for Thresholding LUT template."""
        
        # ✅ Backend decides how to interpret operation attributes
        return {
            # Backend knowledge: Thresholding uses embedded constants
            'mem_mode': 'const_embedded',
            'ram_style': 'distributed',
            
            # Extract operation attributes directly
            'pe_factor': operation.get_nodeattr("PE"),              # Will fail if missing
            'num_channels': operation.get_nodeattr("NumChannels"), # Will fail if missing
            'weight_data_type': operation.get_nodeattr("weightDataType"),
            'activation_value': operation.get_nodeattr("ActVal"),
            
            # Backend knowledge: Thresholding doesn't use SIMD
            'simd_factor': 1,
            'parallelization_strategy': 'pe_only',
            
            # Use operation helper methods
            'lookup_table_depth': operation.calc_tmem(),
            'threshold_count': operation.get_threshold_count(),
            'input_width': operation.get_instream_width(),
            'output_width': operation.get_outstream_width(),
        }
```

#### **HLS Backend for MVAU**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
class MVAUHLSBackend(HLSBackend):
    """HLS backend that knows how to use MVAU operations."""
    
    def get_supported_templates(self, operation: HWCustomOp) -> Set[str]:
        if isinstance(operation, MatrixVectorActivation):
            return {"hls_mvau_streaming", "hls_mvau_parallel", "hls_basic"}
        return set()
    
    def get_template_values(self, operation: HWCustomOp, template_name: str) -> Dict[str, Any]:
        """Extract values from MVAU operation for HLS templates."""
        
        if not isinstance(operation, MatrixVectorActivation):
            raise ValueError("MVAUHLSBackend only handles MVAU")
        
        if template_name == "hls_mvau_streaming":
            return self._extract_mvau_streaming_values(operation)
        # ... other templates
    
    def _extract_mvau_streaming_values(self, operation: MatrixVectorActivation) -> Dict[str, Any]:
        """Extract values for MVAU streaming template."""
        
        # ✅ Backend extracts what it needs - will fail clearly if missing
        return {
            # MVAU actually has these attributes - extract directly
            'mem_mode': operation.get_nodeattr("mem_mode"),
            'ram_style': operation.get_nodeattr("ram_style"),
            'pe_factor': operation.get_nodeattr("PE"),
            'simd_factor': operation.get_nodeattr("SIMD"),
            
            # Extract matrix dimensions
            matrix_width, matrix_height = operation.get_matrix_dimensions()
            'matrix_width': matrix_width,
            'matrix_height': matrix_height,
            
            # Backend knowledge: MVAU uses PE+SIMD parallelization
            'parallelization_strategy': 'pe_simd',
            
            # Use operation helper methods
            'weight_memory_depth': operation.calc_wmem(),
            'weight_storage_requirement': operation.get_weight_storage_requirement(),
            'input_width': operation.get_instream_width(),
            'output_width': operation.get_outstream_width(),
        }
```

---

### **4. Error Handling - Fail Fast and Clear**

#### **Clear Failure When Attributes Missing**
```python
# If Thresholding operation is missing PE attribute:
pe_factor = operation.get_nodeattr("PE")  # ❌ Raises AttributeError: "PE"

# This is GOOD - clear, immediate failure with specific attribute name
# Much better than silent default that could cause subtle bugs
```

#### **Backend Handles Operation Type Mismatch**
```python
# If wrong operation type is passed to backend:
class ThresholdingHLSBackend:
    def get_template_values(self, operation, template_name):
        if not isinstance(operation, Thresholding):
            raise ValueError(f"ThresholdingHLSBackend only handles Thresholding, got {type(operation)}")
        
        # Clear, immediate failure with specific backend-operation mismatch
```

---

### **5. Backend Registration and Selection**

#### **Backend Registry**
```python
# File: src/finn/codegen/backend_registry.py
class BackendRegistry:
    """Registry of available backends for different operations."""
    
    def __init__(self):
        self.hls_backends = [
            ThresholdingHLSBackend(),
            MVAUHLSBackend(),
            GenericHLSBackend(),  # Fallback
        ]
        
        self.rtl_backends = [
            ThresholdingRTLBackend(),
            MVAURTLBackend(),
            GenericRTLBackend(),  # Fallback
        ]
    
    def find_hls_backend(self, operation: HWCustomOp) -> HLSBackend:
        """Find appropriate HLS backend for operation."""
        for backend in self.hls_backends:
            if backend.get_supported_templates(operation):
                return backend
        raise ValueError(f"No HLS backend found for {type(operation)}")
    
    def find_rtl_backend(self, operation: HWCustomOp) -> RTLBackend:
        """Find appropriate RTL backend for operation."""
        for backend in self.rtl_backends:
            if backend.get_supported_templates(operation):
                return backend
        raise ValueError(f"No RTL backend found for {type(operation)}")
```

#### **Code Generation Flow**
```python
# File: src/finn/codegen/unified_generator.py
class UnifiedCodeGenerator:
    """Main code generator using backend registry."""
    
    def __init__(self):
        self.backend_registry = BackendRegistry()
    
    def generate_hls_code(self, operation: HWCustomOp) -> str:
        """Generate HLS code for operation."""
        
        # 1. Find appropriate backend
        backend = self.backend_registry.find_hls_backend(operation)
        
        # 2. Backend generates code
        return backend.generate_hls_code(operation)
    
    def generate_rtl_code(self, operation: HWCustomOp) -> str:
        """Generate RTL code for operation."""
        
        # 1. Find appropriate backend
        backend = self.backend_registry.find_rtl_backend(operation)
        
        # 2. Backend generates code
        return backend.generate_rtl_code(operation)
```

---

## ✅ **Benefits of This Clean Design**

### **1. Fail Fast and Clear**
- Missing required attributes fail immediately with clear error messages
- No silent defaults that could cause subtle bugs
- Easy debugging when something is wrong

### **2. Pure Separation of Concerns**
- **Operations**: Define domain attributes and helper methods
- **Backends**: Know how to map operations to templates
- **No Confusion**: Each component has a single, clear responsibility

### **3. Backend Freedom**
- Backends can use operation attributes however they want
- Backends can provide different values for different templates
- Backends can use operation helper methods as needed

### **4. Simple Operation Development**
- New operations just define their attributes and helpers
- No template knowledge required
- No forced abstract methods for template support

### **5. Clear Error Messages**
```python
# Clear errors when things go wrong:

# Missing attribute:
AttributeError: "Thresholding operation missing required attribute 'PE'"

# Wrong operation type:  
ValueError: "ThresholdingHLSBackend only handles Thresholding, got MatrixVectorActivation"

# No backend found:
ValueError: "No HLS backend found for <class 'NewOperation'>"

# Unsupported template:
UnsupportedTemplateError: "Template 'hls_unsupported' not supported by ThresholdingHLSBackend"
```

---

## 🎯 **Example: Fixing the Original Problem**

### **Before (Broken)**
```python
# Framework assumes all operations have mem_mode
mem_mode = operation.get_nodeattr("mem_mode")  # ❌ Crashes on Thresholding
```

### **After (Clean)**
```python
# Backend knows how to handle Thresholding appropriately
class ThresholdingHLSBackend:
    def get_template_values(self, operation: Thresholding, template_name: str):
        return {
            # ✅ Backend provides appropriate value for Thresholding
            'mem_mode': 'const_embedded',
            'pe_factor': operation.get_nodeattr("PE"),  # ✅ Will fail clearly if missing
        }
```

**Result**: Thresholding operations work because the HLS backend provides appropriate template values, while MVAU operations work because they actually have the `mem_mode` attribute.

---

## 🏁 **Summary**

This clean design eliminates over-engineering while maintaining all benefits:

✅ **Operations**: Simple, focused, clear attribute definitions  
✅ **Backends**: Template expertise and operation-to-template mapping  
✅ **Error Handling**: Fail fast with clear messages  
✅ **Extensibility**: New operations and backends easy to add  
✅ **No Breaking Changes**: Existing operations work as-is  

The architecture fixes the original test failure elegantly while creating a maintainable, extensible system.