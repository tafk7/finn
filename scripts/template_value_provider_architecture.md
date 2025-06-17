# Template Value Provider Architecture

## Context Snapshot
The unified codegen framework violates separation of concerns by making assumptions about custom operation internals instead of using templates as ground truth. Operations should provide an interface to populate template placeholders—whether from nodeattrs, computed values, or other logic.

## Critical Survey

### 🎯 **Architectural Strengths Already Present**
1. **Template-Based Design**: HLS templates already exist as the contract definition
2. **Polymorphic Custom Operations**: Each operation already inherits from `HWCustomOp` 
3. **Existing Abstraction Layer**: Operations already provide methods like `get_instream_width()`, `get_exp_cycles()`

### 🚨 **Fundamental Design Violations** 

#### **Issue 1: Framework Attribute Coupling (High Impact)**
```python
# ❌ VIOLATES - Lines 347-349 in hls_generator.py
def _get_memory_config(self) -> Dict[str, Any]:
    return {
        'mem_mode': self.operation.get_nodeattr("mem_mode"),    # ASSUMES internal structure
        'ram_style': self.operation.get_nodeattr("ram_style"), # BREAKS encapsulation
    }
```
**Root Cause**: Framework reaching into operation internals instead of using abstraction interface.

#### **Issue 2: Hard-coded Template Population (High Impact)**
```python
# ❌ VIOLATES - Line 73 in hls_generator.py
def _get_mvau_template_name(self) -> str:
    mem_mode = self.operation.get_nodeattr("mem_mode")  # ASSUMES operation type
```
**Root Cause**: Framework making operation-specific decisions instead of letting operations self-describe.

#### **Issue 3: Mixed Responsibilities (Medium Impact)**
- Framework tries to be both **template engine** AND **operation attribute validator**
- Should be pure **template population service** using operation-provided values

### 🔍 **Deep Dive: Template-Operation Contract Violation**

The current architecture violates the **Interface Segregation Principle**:

```python
# Current BAD architecture:
Framework -> knows about -> Operation internals -> knows about -> Templates

# Should be:
Framework -> requests from -> Operation interface -> knows about -> Templates
```

**Downstream Effects**:
1. Every new operation requires framework changes
2. Framework becomes tightly coupled to operation implementation details  
3. "Zero breaking changes" promise impossible to fulfill
4. Template evolution requires coordinated changes across multiple layers

## Actionable Remedies

### **Solution Architecture: Template Value Provider Pattern**

#### **1. Template Contract Interface**
```python
# File: src/finn/codegen/template_value_provider.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Set

class TemplateValueProvider(ABC):
    """Interface for custom operations to provide template placeholder values."""
    
    @abstractmethod
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """
        Provide values for all placeholders in the specified template.
        
        Args:
            template_name: Name of template requesting values
            
        Returns:
            Dict mapping placeholder names to values
            
        Raises:
            UnsupportedTemplateError: If template not supported by this operation
        """
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

#### **2. Framework Template Population Service**
```python
# File: src/finn/codegen/unified_generator.py

class UnifiedCodeGenerator:
    """Template-agnostic code generator using operation value providers."""
    
    def __init__(self, template_engine):
        self.template_engine = template_engine
    
    def generate_code(self, operation: TemplateValueProvider, template_name: str) -> str:
        """Generate code using template-operation interface."""
        
        # ✅ GOOD - Framework doesn't know about operation internals
        if not operation.supports_template(template_name):
            supported = operation.get_supported_templates()
            raise UnsupportedTemplateError(
                f"{operation.__class__.__name__} doesn't support '{template_name}'. "
                f"Supported templates: {supported}"
            )
            
        template_values = operation.get_template_values(template_name)
        
        # Validate all required placeholders are provided
        self._validate_template_values(template_name, template_values)
        
        return self.template_engine.render(template_name, template_values)
    
    def _validate_template_values(self, template_name: str, values: Dict[str, Any]):
        """Ensure all required template placeholders have values."""
        required_placeholders = self.template_engine.get_required_placeholders(template_name)
        missing = required_placeholders - set(values.keys())
        
        if missing:
            raise ValueError(f"Missing required template values for '{template_name}': {missing}")
```

#### **3. Enhanced HWCustomOp Base Class**
```python
# File: src/finn/custom_op/fpgadataflow/hwcustomop.py

from finn.codegen.template_value_provider import TemplateValueProvider

class HWCustomOp(CustomOp, TemplateValueProvider):
    """Enhanced base class with template value provider interface."""
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Default template value provider - can be overridden by subclasses."""
        
        if template_name == "hls_basic":
            return self._get_basic_template_values()
        elif template_name == "hls_streaming":
            return self._get_streaming_template_values()
        elif template_name == "hls_memory":
            return self._get_memory_template_values()
        else:
            raise UnsupportedTemplateError(f"Template '{template_name}' not supported by base class")
    
    def supports_template(self, template_name: str) -> bool:
        """Check if this operation supports the template."""
        return template_name in self.get_supported_templates()
    
    def get_supported_templates(self) -> Set[str]:
        """Base class supports common templates."""
        return {"hls_basic", "hls_streaming", "hls_memory"}
    
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
            'folded_shape': str(self.get_folded_input_shape()),
        }
    
    def _get_memory_template_values(self) -> Dict[str, Any]:
        """Safe memory template defaults."""
        return {
            'mem_mode': 'const_embedded',  # Safe default
            'ram_style': 'auto',           # Let tools decide
        }
```

#### **4. Operation-Specific Template Value Implementation**
```python
# File: src/finn/custom_op/fpgadataflow/thresholding.py

class Thresholding(HWCustomOp):
    """Thresholding operation with template value provider."""
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Provide template values based on Thresholding operation logic."""
        
        if template_name == "hls_basic":
            return self._get_thresholding_basic_values()
        elif template_name == "hls_streaming":
            return self._get_thresholding_streaming_values()
        elif template_name == "hls_memory":
            return self._get_thresholding_memory_values()
        else:
            # Try base class for other templates
            return super().get_template_values(template_name)
    
    def _get_thresholding_basic_values(self) -> Dict[str, Any]:
        """Thresholding-specific basic template values."""
        base_values = super()._get_basic_template_values()
        base_values.update({
            'pe_factor': self.get_nodeattr("PE"),
            'num_channels': self.get_nodeattr("NumChannels"),
            'weight_data_type': self.get_nodeattr("weightDataType"),
            'act_val': self.get_nodeattr("ActVal"),
        })
        return base_values
    
    def _get_thresholding_streaming_values(self) -> Dict[str, Any]:
        """Thresholding-specific streaming values."""
        base_values = super()._get_streaming_template_values()
        base_values.update({
            'pe_factor': self.get_nodeattr("PE"),
            # No SIMD - Thresholding doesn't use it
            'parallelization_type': 'PE',
        })
        return base_values
    
    def _get_thresholding_memory_values(self) -> Dict[str, Any]:
        """Thresholding-specific memory logic."""
        return {
            'mem_mode': 'const_embedded',  # Thresholding uses embedded constants
            'ram_style': 'distributed',    # Small lookup tables use LUTRAM
            'memory_depth': self.calc_tmem() if hasattr(self, 'calc_tmem') else 1,
        }
    
    def get_supported_templates(self) -> Set[str]:
        """Thresholding supports basic templates."""
        return {"hls_basic", "hls_streaming", "hls_memory"}
```

#### **5. MVAU Operation Implementation**
```python
# File: src/finn/custom_op/fpgadataflow/matrixvectoractivation.py

class MatrixVectorActivation(HWCustomOp):
    """MVAU operation with full template support."""
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """MVAU-specific template value logic."""
        
        if template_name == "hls_mvau_streaming":
            return self._get_mvau_streaming_values()
        elif template_name == "hls_mvau_parallel":
            return self._get_mvau_parallel_values()
        elif template_name == "hls_memory":
            return self._get_mvau_memory_values()
        else:
            return super().get_template_values(template_name)
    
    def _get_mvau_streaming_values(self) -> Dict[str, Any]:
        """MVAU streaming template - uses both PE and SIMD."""
        return {
            'pe_factor': self.get_nodeattr("PE"),
            'simd_factor': self.get_nodeattr("SIMD"),
            'mw': self.get_nodeattr("MW"),
            'mh': self.get_nodeattr("MH"),
            'parallelization_type': 'PE_SIMD',
        }
    
    def _get_mvau_memory_values(self) -> Dict[str, Any]:
        """MVAU memory configuration - has mem_mode."""
        return {
            'mem_mode': self.get_nodeattr("mem_mode"),  # MVAU actually has this
            'ram_style': self.get_nodeattr("ram_style"),
            'memory_depth': self.calc_wmem(),
        }
    
    def get_supported_templates(self) -> Set[str]:
        """MVAU supports specialized templates."""
        return {"hls_basic", "hls_streaming", "hls_memory", "hls_mvau_streaming", "hls_mvau_parallel"}
```

## Validation Path

### **Implementation Phases**

#### **Phase 1: Interface Definition** 
1. ✅ Define `TemplateValueProvider` abstract interface
2. ✅ Create `UnifiedCodeGenerator` template-agnostic service
3. ✅ Identify all template placeholders across existing templates

#### **Phase 2: Framework Decoupling**
1. Replace direct `get_nodeattr()` calls with `get_template_values()` calls in `hls_generator.py`
2. Remove operation-type-specific logic from framework  
3. Make framework purely template-agnostic

#### **Phase 3: Operation Implementation**
1. Make `HWCustomOp` implement `TemplateValueProvider` with safe defaults
2. Override in specific operations (`Thresholding`, `MatrixVectorActivation`, etc.)
3. Move operation-specific template logic into operations themselves

#### **Phase 4: Validation**
1. Verify all existing operations work with new interface
2. Test that new operations can be added without framework changes
3. Confirm "zero breaking changes" promise fulfilled

### **Expected Validation Results**
- **Template Coverage**: All placeholders can be populated by appropriate operations
- **Framework Agnosticism**: Framework code contains zero operation-specific logic
- **Extensibility**: New operations added without framework modifications
- **Backward Compatibility**: Existing operations continue working unchanged

### **Migration Strategy**

#### **Backward Compatibility Bridge**
```python
# File: src/finn/codegen/legacy_bridge.py

class LegacyOperationBridge:
    """Temporary bridge for operations not yet implementing TemplateValueProvider."""
    
    @staticmethod
    def wrap_legacy_operation(operation: HWCustomOp) -> TemplateValueProvider:
        """Wrap legacy operation with template value provider interface."""
        
        if isinstance(operation, TemplateValueProvider):
            return operation  # Already implemented
        
        # Create adapter for legacy operations
        return LegacyTemplateValueAdapter(operation)

class LegacyTemplateValueAdapter(TemplateValueProvider):
    """Adapter to make legacy operations work with new interface."""
    
    def __init__(self, legacy_operation: HWCustomOp):
        self.operation = legacy_operation
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Provide safe defaults for legacy operations."""
        
        # Try to extract values safely with fallbacks
        values = {}
        
        # Safe attribute access with defaults
        for attr, default in [
            ('mem_mode', 'const_embedded'),
            ('ram_style', 'auto'),
            ('PE', 1),
            ('SIMD', 1),
        ]:
            try:
                values[attr.lower()] = self.operation.get_nodeattr(attr)
            except (AttributeError, KeyError):
                values[attr.lower()] = default
        
        return values
    
    def supports_template(self, template_name: str) -> bool:
        return template_name in ["hls_basic"]
    
    def get_supported_templates(self) -> Set[str]:
        return {"hls_basic"}
```

## Respectful Closure

This architecture elegantly solves the fundamental design violation while **honoring existing strengths**:

✅ **Preserves**: Template-based design, existing operation abstractions  
✅ **Improves**: Separation of concerns, extensibility, maintainability  
✅ **Enables**: True "zero breaking changes" for new operations

**Key Benefits**:
1. **Framework Simplicity**: No operation-specific logic in framework
2. **Operation Autonomy**: Each operation controls its template values
3. **Template Evolution**: Templates can change without breaking operations
4. **Extensibility**: New operations work immediately with existing templates
5. **Testability**: Each component can be tested independently

The architecture transforms the framework from a **tightly-coupled attribute accessor** into a **clean template population service**, achieving the original design goals.