# Correct Inheritance-Based Architecture

## 🎯 **Critical Inheritance Structure Clarification**

The user clarified the actual inheritance structure in FINN:

- **`HWCustomOp`**, **`RTLBackend`**, and **`HLSBackend`** are all **abstract base classes**
- For each operation, there are **three concrete classes** with multiple inheritance:
  - `Thresholding(HWCustomOp)` - Core operation logic
  - `ThresholdingRTL(Thresholding, RTLBackend)` - RTL-specific template logic 
  - `ThresholdingHLS(Thresholding, HLSBackend)` - HLS-specific template logic

This is a **mixin pattern** where backend classes add template generation capability to core operations.

---

## 🏗️ **Correct Architecture with Multiple Inheritance**

### **File Structure**
```
src/finn/custom_op/fpgadataflow/
├── hwcustomop.py              # Abstract HWCustomOp base
├── hlsbackend.py              # Abstract HLSBackend base  
├── rtlbackend.py              # Abstract RTLBackend base
├── thresholding.py            # Thresholding(HWCustomOp)
├── matrixvectoractivation.py  # MatrixVectorActivation(HWCustomOp)
├── hls/
│   ├── thresholding_hls.py    # ThresholdingHLS(Thresholding, HLSBackend)
│   └── mvau_hls.py            # MVAU_HLS(MatrixVectorActivation, HLSBackend)
└── rtl/
    ├── thresholding_rtl.py    # ThresholdingRTL(Thresholding, RTLBackend)
    └── mvau_rtl.py            # MVAU_RTL(MatrixVectorActivation, RTLBackend)
```

---

### **1. Abstract Base Classes**

#### **HWCustomOp Abstract Base**
```python
# File: src/finn/custom_op/fpgadataflow/hwcustomop.py
from abc import ABC, abstractmethod

class HWCustomOp(CustomOp, ABC):
    """Abstract base class for hardware custom operations."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
    
    @abstractmethod
    def get_nodeattr_types(self):
        """Define operation-specific attributes."""
        pass
    
    @abstractmethod
    def get_instream_width(self):
        """Calculate input stream width."""
        pass
    
    @abstractmethod
    def get_outstream_width(self):
        """Calculate output stream width."""
        pass
    
    @abstractmethod
    def get_exp_cycles(self):
        """Calculate expected cycles."""
        pass
```

#### **HLSBackend Abstract Base**
```python
# File: src/finn/custom_op/fpgadataflow/hlsbackend.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Set

class HLSBackend(ABC):
    """Abstract base class for HLS code generation."""
    
    @abstractmethod
    def get_supported_templates(self) -> Set[str]:
        """Return HLS templates this backend supports."""
        pass
    
    @abstractmethod
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for specified HLS template."""
        pass
    
    def supports_template(self, template_name: str) -> bool:
        """Check if backend supports template."""
        return template_name in self.get_supported_templates()
    
    def generate_hls_code(self) -> str:
        """Generate HLS code using template system."""
        
        # Select best template
        template_name = self._select_optimal_template()
        
        # Extract values for template
        template_values = self.get_template_values(template_name)
        
        # Render template
        return self.template_engine.render(template_name, template_values)
    
    def _select_optimal_template(self) -> str:
        """Select best HLS template."""
        supported = self.get_supported_templates()
        
        # HLS template priority order
        hls_priorities = [
            "hls_streaming_optimized",
            "hls_parallel_optimized", 
            "hls_streaming_generic",
            "hls_basic"
        ]
        
        for template in hls_priorities:
            if template in supported:
                return template
                
        raise ValueError(f"No HLS template found for {self.__class__.__name__}")
```

#### **RTLBackend Abstract Base**
```python
# File: src/finn/custom_op/fpgadataflow/rtlbackend.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Set

class RTLBackend(ABC):
    """Abstract base class for RTL code generation."""
    
    @abstractmethod
    def get_supported_templates(self) -> Set[str]:
        """Return RTL templates this backend supports."""
        pass
    
    @abstractmethod
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for specified RTL template."""
        pass
    
    def supports_template(self, template_name: str) -> bool:
        """Check if backend supports template."""
        return template_name in self.get_supported_templates()
    
    def generate_rtl_code(self) -> str:
        """Generate RTL code using template system."""
        
        # Select best template
        template_name = self._select_optimal_template()
        
        # Extract values for template
        template_values = self.get_template_values(template_name)
        
        # Render template
        return self.template_engine.render(template_name, template_values)
    
    def _select_optimal_template(self) -> str:
        """Select best RTL template."""
        supported = self.get_supported_templates()
        
        # RTL template priority order (different from HLS!)
        rtl_priorities = [
            "rtl_axi_stream_optimized",
            "rtl_parallel_optimized",
            "rtl_streaming_generic", 
            "rtl_basic"
        ]
        
        for template in rtl_priorities:
            if template in supported:
                return template
                
        raise ValueError(f"No RTL template found for {self.__class__.__name__}")
```

---

### **2. Core Operation Implementations**

#### **Thresholding Core Operation**
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py
class Thresholding(HWCustomOp):
    """Thresholding operation - core domain logic only."""
    
    def get_nodeattr_types(self):
        # ✅ Pure operation attributes
        return {
            "PE": ("i", True, ""),
            "NumChannels": ("i", True, ""),
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "ActVal": ("f", True, 0.0),
        }
    
    def get_instream_width(self):
        """Thresholding input stream width calculation."""
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("PE")
    
    def get_outstream_width(self):
        """Thresholding output stream width calculation."""
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("PE")
    
    def get_exp_cycles(self):
        """Expected cycles for thresholding."""
        return self.get_nodeattr("NumChannels") // self.get_nodeattr("PE")
    
    # ✅ Thresholding-specific helper methods
    def calc_tmem(self):
        """Threshold memory depth calculation."""
        return self.get_nodeattr("NumChannels") // self.get_nodeattr("PE")
    
    def get_threshold_count(self):
        """Number of threshold values needed."""
        return self.get_nodeattr("NumChannels")
    
    def get_activation_values(self):
        """Get activation threshold values."""
        return self.get_nodeattr("ActVal")
```

#### **MVAU Core Operation**
```python
# File: src/finn/custom_op/fpgadataflow/matrixvectoractivation.py
class MatrixVectorActivation(HWCustomOp):
    """MVAU operation - core domain logic only."""
    
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
    
    def get_instream_width(self):
        """MVAU input stream width calculation."""
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("SIMD")
    
    def get_outstream_width(self):
        """MVAU output stream width calculation."""
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("PE")
    
    def get_exp_cycles(self):
        """Expected cycles for MVAU."""
        return (self.get_nodeattr("MW") // self.get_nodeattr("SIMD")) * \
               (self.get_nodeattr("MH") // self.get_nodeattr("PE"))
    
    # ✅ MVAU-specific helper methods
    def calc_wmem(self):
        """Weight memory depth calculation."""
        return self.get_nodeattr("MW") * self.get_nodeattr("MH") // self.get_nodeattr("SIMD")
    
    def get_matrix_dimensions(self):
        """Return matrix width and height."""
        return self.get_nodeattr("MW"), self.get_nodeattr("MH")
    
    def get_parallelization_factors(self):
        """Return PE and SIMD factors."""
        return self.get_nodeattr("PE"), self.get_nodeattr("SIMD")
```

---

### **3. HLS Backend Implementations**

#### **Thresholding HLS Backend**
```python
# File: src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

class ThresholdingHLS(Thresholding, HLSBackend):
    """HLS backend for Thresholding operations."""
    
    def get_supported_templates(self) -> Set[str]:
        """HLS templates for Thresholding."""
        return {
            "hls_thresholding_lut",      # Thresholding-optimized HLS
            "hls_streaming_generic",     # Generic streaming HLS
            "hls_basic"                  # Basic HLS fallback
        }
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract Thresholding values for HLS templates."""
        
        if template_name == "hls_thresholding_lut":
            return self._get_thresholding_lut_values()
        elif template_name == "hls_streaming_generic":
            return self._get_streaming_values()
        elif template_name == "hls_basic":
            return self._get_basic_values()
        else:
            raise UnsupportedTemplateError(f"Template {template_name} not supported")
    
    def _get_thresholding_lut_values(self) -> Dict[str, Any]:
        """Values for Thresholding LUT HLS template."""
        return {
            # ✅ HLS backend provides appropriate values for Thresholding
            'mem_mode': 'const_embedded',           # Thresholding uses embedded LUTs
            'ram_style': 'distributed',             # LUTs use distributed RAM
            'simd_factor': 1,                       # Thresholding doesn't use SIMD
            'parallelization_strategy': 'pe_only',
            
            # ✅ Extract from core operation (will fail clearly if missing)
            'pe_factor': self.get_nodeattr("PE"),
            'num_channels': self.get_nodeattr("NumChannels"),
            'weight_data_type': self.get_nodeattr("weightDataType"),
            'activation_value': self.get_nodeattr("ActVal"),
            
            # ✅ Use core operation helper methods
            'lookup_table_depth': self.calc_tmem(),
            'threshold_count': self.get_threshold_count(),
            'input_width': self.get_instream_width(),
            'output_width': self.get_outstream_width(),
            'exp_cycles': self.get_exp_cycles(),
        }
    
    def _get_streaming_values(self) -> Dict[str, Any]:
        """Values for generic streaming HLS template."""
        return {
            # Generic streaming values
            'stream_input_width': self.get_instream_width(),
            'stream_output_width': self.get_outstream_width(),
            'pe_factor': self.get_nodeattr("PE"),
            'simd_factor': 1,  # Thresholding doesn't use SIMD
        }
```

#### **MVAU HLS Backend**
```python
# File: src/finn/custom_op/fpgadataflow/hls/mvau_hls.py
from finn.custom_op.fpgadataflow.matrixvectoractivation import MatrixVectorActivation
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

class MVAU_HLS(MatrixVectorActivation, HLSBackend):
    """HLS backend for MVAU operations."""
    
    def get_supported_templates(self) -> Set[str]:
        """HLS templates for MVAU."""
        return {
            "hls_mvau_streaming",        # MVAU-optimized streaming HLS
            "hls_mvau_parallel",         # MVAU-optimized parallel HLS
            "hls_streaming_generic",     # Generic streaming HLS
            "hls_basic"                  # Basic HLS fallback
        }
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract MVAU values for HLS templates."""
        
        if template_name == "hls_mvau_streaming":
            return self._get_mvau_streaming_values()
        elif template_name == "hls_mvau_parallel":
            return self._get_mvau_parallel_values()
        # ... other templates
    
    def _get_mvau_streaming_values(self) -> Dict[str, Any]:
        """Values for MVAU streaming HLS template."""
        
        pe_factor, simd_factor = self.get_parallelization_factors()
        matrix_width, matrix_height = self.get_matrix_dimensions()
        
        return {
            # ✅ MVAU actually has these attributes - extract directly
            'mem_mode': self.get_nodeattr("mem_mode"),
            'ram_style': self.get_nodeattr("ram_style"),
            'pe_factor': pe_factor,
            'simd_factor': simd_factor,
            'matrix_width': matrix_width,
            'matrix_height': matrix_height,
            
            # ✅ HLS backend knowledge
            'parallelization_strategy': 'pe_simd',
            
            # ✅ Use core operation helper methods
            'weight_memory_depth': self.calc_wmem(),
            'input_width': self.get_instream_width(),
            'output_width': self.get_outstream_width(),
            'exp_cycles': self.get_exp_cycles(),
        }
```

---

### **4. RTL Backend Implementations**

#### **Thresholding RTL Backend**
```python
# File: src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

class ThresholdingRTL(Thresholding, RTLBackend):
    """RTL backend for Thresholding operations."""
    
    def get_supported_templates(self) -> Set[str]:
        """RTL templates for Thresholding."""
        return {
            "rtl_thresholding_lut",      # Thresholding-optimized RTL
            "rtl_axi_stream_generic",    # Generic AXI Stream RTL
            "rtl_basic"                  # Basic RTL fallback
        }
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract Thresholding values for RTL templates."""
        
        if template_name == "rtl_thresholding_lut":
            return self._get_thresholding_lut_values()
        # ... other templates
    
    def _get_thresholding_lut_values(self) -> Dict[str, Any]:
        """Values for Thresholding LUT RTL template."""
        return {
            # ✅ RTL-specific values (different from HLS!)
            'module_name': f"thresholding_{self.get_nodeattr('PE')}pe",
            'data_width': self.get_input_datatype().bitwidth(),
            'pe_count': self.get_nodeattr("PE"),
            'channel_count': self.get_nodeattr("NumChannels"),
            'lut_style': 'distributed',
            'clock_enable': True,
            'reset_style': 'sync',
            
            # ✅ Use core operation methods
            'input_width': self.get_instream_width(),
            'output_width': self.get_outstream_width(),
            'lookup_depth': self.calc_tmem(),
        }
```

---

## 🎯 **How This Fixes the Original Problem**

### **Before (Broken Framework)**
```python
# Framework assumes all operations have mem_mode
mem_mode = operation.get_nodeattr("mem_mode")  # ❌ Crashes on Thresholding
```

### **After (Correct Inheritance Structure)**
```python
# Usage with correct inheritance:
thresholding_hls = ThresholdingHLS(onnx_node)  # Multiple inheritance
hls_code = thresholding_hls.generate_hls_code()

# Inside ThresholdingHLS.get_template_values():
return {
    'mem_mode': 'const_embedded',  # ✅ Appropriate for Thresholding HLS
    'pe_factor': self.get_nodeattr("PE"),  # ✅ From Thresholding base class
}

# Usage with MVAU:
mvau_hls = MVAU_HLS(onnx_node)  # Multiple inheritance
hls_code = mvau_hls.generate_hls_code()

# Inside MVAU_HLS.get_template_values():
return {
    'mem_mode': self.get_nodeattr("mem_mode"),  # ✅ MVAU actually has this attribute
    'pe_factor': self.get_nodeattr("PE"),
}
```

---

## ✅ **Benefits of Correct Inheritance Architecture**

### **1. Clean Separation with Multiple Inheritance**
- **Core Operation**: Pure domain logic and attributes
- **HLS Backend Mixin**: HLS-specific template knowledge for that operation
- **RTL Backend Mixin**: RTL-specific template knowledge for that operation

### **2. Operation-Specific Template Logic**
- Each operation can have completely different HLS and RTL template strategies
- Template logic is co-located with the operation it serves
- No generic backends trying to handle all operation types

### **3. Technology-Specific Backends**
- HLS backends provide HLS-appropriate values
- RTL backends provide RTL-appropriate values
- Same operation, different template values per technology

### **4. Clear Inheritance Chain**
- `ThresholdingHLS` inherits domain logic from `Thresholding`
- `ThresholdingHLS` inherits template interface from `HLSBackend`
- Both capabilities combined in one class via multiple inheritance

This inheritance structure elegantly solves the original problem while maintaining clean separation of concerns and enabling operation-specific template logic for different target technologies.