# Simplified Explicit Codegen Implementation Plan

## 🎯 **Implementation Overview**

Based on the architectural analysis and caching mechanism findings, this plan implements a **dramatically simplified codegen architecture** that eliminates complexity while maintaining performance through strategic caching.

## 📋 **Phase 1: Simplified Template Engine (Week 1)**

### **1.1 Replace Complex TemplateEngine with Simplified Version**

**Goal:** Remove 80% of caching complexity while keeping essential performance benefits.

**Current State:** 467 lines with 5 different caching mechanisms
**Target State:** ~150 lines with single bounded cache

#### **New Simplified Template Engine**

```python
# src/finn/codegen/simple_template_engine.py
from functools import lru_cache
from typing import Dict, Any, List, Optional
import jinja2
import logging

class SimpleTemplateEngine:
    """Simplified template engine with single performance-critical cache."""
    
    def __init__(self, template_dirs: Optional[List[str]] = None):
        if template_dirs is None:
            template_dirs = self._get_default_template_dirs()
            
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dirs, followlinks=True),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols']
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @lru_cache(maxsize=50)  # ONLY cache we need - bounded LRU
    def _get_compiled_template(self, template_name: str):
        """Cache compiled templates - the only expensive operation."""
        return self.jinja_env.get_template(template_name)
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Simple rendering without validation complexity."""
        try:
            template = self._get_compiled_template(template_name)
            return template.render(**context)
        except jinja2.TemplateNotFound as e:
            raise jinja2.TemplateNotFound(f"Template '{template_name}' not found.")
    
    def template_exists(self, template_name: str) -> bool:
        """Check if template exists.""" 
        try:
            self._get_compiled_template(template_name)
            return True
        except jinja2.TemplateNotFound:
            return False
    
    def clear_cache(self):
        """Clear template compilation cache."""
        self._get_compiled_template.cache_clear()
        
    # Remove: _placeholder_cache, _dependency_cache, validate_template, 
    #         get_required_placeholders, render_with_validation
```

#### **Migration Steps:**
1. Create `simple_template_engine.py` (new file)
2. Update imports in `codegen.py` to use SimpleTemplateEngine
3. Remove complex validation calls from `_validate_template_values()`
4. Test with existing templates to ensure compatibility

### **1.2 Simplify Codegen Base Class**

**Remove complexity from codegen.py:**

```python
# Simplified codegen.py changes
class Codegen(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
        
        # Simple template engine - no registry needed
        self.template_engine = self._initialize_simple_template_engine()
        
        # Remove: template_registry, complex config, performance tracking
    
    def _initialize_simple_template_engine(self):
        """Initialize simplified template engine."""
        from .simple_template_engine import SimpleTemplateEngine
        return SimpleTemplateEngine()
    
    def generate_code(self) -> str:
        """Simplified code generation flow."""
        # 1. Get template name (explicit from backend)
        template_name = self.get_template_name()  # New required method
        
        # 2. Extract values
        template_values = self.get_template_values(template_name)
        
        # 3. Render (no validation complexity)
        code = self.template_engine.render(template_name, template_values)
        
        # 4. Simple post-processing
        return self._post_process_code(code)
    
    @abstractmethod  
    def get_template_name(self) -> str:
        """Get explicit template name for this backend."""
        pass
        
    # Remove: _validate_template_values, _select_optimal_template,
    #         registry methods, debug saving, performance tracking
```

## 📋 **Phase 2: Explicit Template Architecture (Week 2)**

### **2.1 Update Base Backend Classes**

#### **HLSBackend with Explicit Templates**

```python
# src/finn/custom_op/fpgadataflow/hlsbackend.py additions
class HLSBackend(Codegen):
    # Class-level template declaration (explicit)
    TEMPLATE_NAME: Optional[str] = None
    TEMPLATE_OPTIONS: Optional[Dict[str, str]] = None
    
    def get_template_name(self) -> str:
        """Get template name - explicit declaration required."""
        # Check for instance override first
        if hasattr(self, '_template_override'):
            return self._template_override
            
        # Use class-level declaration
        if self.TEMPLATE_NAME:
            return self.TEMPLATE_NAME
            
        if self.TEMPLATE_OPTIONS:
            return self._select_template_from_options()
            
        raise NotImplementedError(
            f"{self.__class__.__name__} must declare TEMPLATE_NAME or TEMPLATE_OPTIONS"
        )
    
    def _select_template_from_options(self) -> str:
        """Override in subclass for template selection logic."""
        return next(iter(self.TEMPLATE_OPTIONS.values()))
    
    def set_template_override(self, template_name: str):
        """Allow runtime template override."""
        self._template_override = template_name
    
    # Remove: get_supported_templates() - replaced by explicit declaration
```

#### **RTLBackend Similar Changes**

```python
# src/finn/custom_op/fpgadataflow/rtlbackend.py
class RTLBackend(Codegen):
    # Same pattern as HLSBackend
    TEMPLATE_NAME: Optional[str] = None
    TEMPLATE_OPTIONS: Optional[Dict[str, str]] = None
    
    def get_template_name(self) -> str:
        # Identical implementation to HLSBackend
```

### **2.2 Convert Existing Backends**

#### **ThresholdingHLS Example**

```python
# src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py
class ThresholdingHLS(Thresholding, HLSBackend):
    # Explicit template declaration
    TEMPLATE_OPTIONS = {
        'basic': 'hls_thresholding_basic.cpp.j2',
        'streaming': 'hls_thresholding_streaming.cpp.j2', 
        'parallel': 'hls_thresholding_parallel.cpp.j2'
    }
    
    def _select_template_from_options(self) -> str:
        """Select template based on operation characteristics."""
        if self._safe_extract_value(self, 'parallel_mode', False):
            return self.TEMPLATE_OPTIONS['parallel']
        elif self._safe_extract_value(self, 'streaming_mode', True):
            return self.TEMPLATE_OPTIONS['streaming']
        return self.TEMPLATE_OPTIONS['basic']
    
    # Remove: get_supported_templates() method entirely
```

#### **MVAU_HLS Example**

```python
# src/finn/custom_op/fpgadataflow/hls/mvau_hls.py  
class MVAU_HLS(MatrixVectorActivation, HLSBackend):
    TEMPLATE_OPTIONS = {
        'basic': 'hls_mvau_basic.cpp.j2',
        'streaming': 'hls_mvau_streaming.cpp.j2',
        'optimized': 'hls_mvau_optimized.cpp.j2'
    }
    
    def _select_template_from_options(self) -> str:
        pe_factor = self._safe_extract_value(self, 'PE', 1)
        if pe_factor > 8:
            return self.TEMPLATE_OPTIONS['optimized']
        elif pe_factor > 1:
            return self.TEMPLATE_OPTIONS['streaming']
        return self.TEMPLATE_OPTIONS['basic']
```

## 📋 **Phase 3: Remove Backend Registry Complexity (Week 3)**

### **3.1 Replace BackendRegistry with Simple Registration**

**Create simplified registry:**

```python
# src/finn/codegen/explicit_backend_registry.py
from typing import Dict, Type, Optional

class ExplicitBackendRegistry:
    """Simple explicit backend registration - no auto-discovery."""
    
    def __init__(self):
        self._hls_backends: Dict[str, Type] = {}
        self._rtl_backends: Dict[str, Type] = {}
    
    def register_hls_backend(self, operation_name: str, backend_class: Type):
        """Explicit HLS backend registration."""
        self._hls_backends[operation_name] = backend_class
    
    def register_rtl_backend(self, operation_name: str, backend_class: Type):
        """Explicit RTL backend registration."""
        self._rtl_backends[operation_name] = backend_class
    
    def get_hls_backend(self, operation_name: str) -> Optional[Type]:
        """Simple lookup - no fuzzy matching."""
        return self._hls_backends.get(operation_name)
    
    def get_rtl_backend(self, operation_name: str) -> Optional[Type]:
        """Simple lookup - no fuzzy matching."""
        return self._rtl_backends.get(operation_name)

# Registration file
# src/finn/codegen/backend_registration.py
def register_all_backends() -> ExplicitBackendRegistry:
    """Explicitly register all backends."""
    registry = ExplicitBackendRegistry()
    
    # HLS backends
    from ..custom_op.fpgadataflow.hls.thresholding_hls import ThresholdingHLS
    from ..custom_op.fpgadataflow.hls.mvau_hls import MVAU_HLS
    # ... other imports
    
    registry.register_hls_backend('Thresholding', ThresholdingHLS)
    registry.register_hls_backend('MatrixVectorActivation', MVAU_HLS)
    # ... other registrations
    
    return registry

_global_registry = register_all_backends()

def get_backend_registry() -> ExplicitBackendRegistry:
    return _global_registry
```

### **3.2 Remove Complex Registry**

**Delete/deprecate:**
- `backend_registry.py` (486 lines) - replace with above
- Remove auto-discovery code
- Remove fuzzy matching logic
- Remove backend scanning/introspection

## 📋 **Phase 4: Handle Multiple Inheritance (Week 4)**

### **4.1 Standardize Constructor Pattern**

**Constructor template for all backends:**

```python
class MVAU_HLS(MatrixVectorActivation, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        # 1. Separate kwargs by prefix to avoid conflicts
        hls_kwargs = {k: v for k, v in kwargs.items() if k.startswith('hls_')}
        op_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('hls_')}
        
        # 2. Initialize in defined order
        MatrixVectorActivation.__init__(self, onnx_node, **op_kwargs)
        HLSBackend.__init__(self, **hls_kwargs) 
        
        # 3. Backend-specific initialization
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
```

### **4.2 Explicit Attribute Handling**

**Handle get_nodeattr_types conflicts explicitly:**

```python
class MVAU_HLS(MatrixVectorActivation, HLSBackend):
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """Explicit attribute conflict resolution."""
        # Get attributes from both parents
        mvau_attrs = MatrixVectorActivation.get_nodeattr_types(self)
        hls_attrs = HLSBackend.get_nodeattr_types(self) 
        
        # Explicit merge policy: HLS attributes override operation attributes
        merged = {**mvau_attrs, **hls_attrs}
        
        # Log conflicts for debugging
        conflicts = set(mvau_attrs.keys()) & set(hls_attrs.keys())
        if conflicts:
            self.logger.debug(f"HLS attributes override MVAU: {conflicts}")
            
        return merged
```

### **4.3 Method Scoping Enforcement**

**Add method prefixes to prevent future conflicts:**

```python
class HLSBackend:
    # HLS-specific methods with hls_ prefix
    def hls_generate_pragmas(self) -> List[str]: pass
    def hls_estimate_resources(self) -> Dict[str, int]: pass
    def hls_get_timing_info(self) -> Dict[str, float]: pass
    
class RTLBackend:
    # RTL-specific methods with rtl_ prefix  
    def rtl_generate_constraints(self) -> List[str]: pass
    def rtl_estimate_resources(self) -> Dict[str, int]: pass
    def rtl_generate_testbench(self) -> str: pass
```

## 📋 **Phase 5: Remove Configuration Complexity (Week 5)**

### **5.1 Simplified Configuration**

**Replace complex config system:**

```python
# src/finn/codegen/simple_config.py
from dataclasses import dataclass
from typing import List
import os

@dataclass 
class SimpleCodegenConfig:
    """Simple configuration without layered complexity."""
    template_dirs: List[str] = None
    debug_mode: bool = False
    log_level: str = 'INFO'
    
    def __post_init__(self):
        if self.template_dirs is None:
            self.template_dirs = self._get_default_template_dirs()
            
        # Simple environment variable override (no complex layering)
        if os.getenv('FINN_DEBUG', '').lower() == 'true':
            self.debug_mode = True
        if os.getenv('FINN_LOG_LEVEL'):
            self.log_level = os.getenv('FINN_LOG_LEVEL')
    
    def _get_default_template_dirs(self) -> List[str]:
        finn_root = os.environ.get('FINN_ROOT', '.')
        return [
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'hls'),
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'rtl'),
        ]

# Simple global config
_global_config = SimpleCodegenConfig()

def get_global_config() -> SimpleCodegenConfig:
    return _global_config
```

### **5.2 Remove Complex Config**

**Delete/deprecate:**
- `codegen_config.py` complex layered system
- Environment variable processing complexity
- File-based configuration loading
- Config validation and merging logic

## 📋 **Phase 6: Testing and Migration (Week 6)**

### **6.1 Backward Compatibility Layer**

```python
# src/finn/codegen/legacy_compat.py  
import warnings

def find_backend_for_operation(operation_name: str, backend_type: str):
    """Legacy compatibility - deprecated."""
    warnings.warn(
        "find_backend_for_operation is deprecated. Use explicit backend registry.",
        DeprecationWarning, stacklevel=2
    )
    
    from .backend_registration import get_backend_registry
    registry = get_backend_registry()
    
    if backend_type.lower() == 'hls':
        return registry.get_hls_backend(operation_name)
    elif backend_type.lower() == 'rtl': 
        return registry.get_rtl_backend(operation_name)
    
    return None
```

### **6.2 Comprehensive Testing**

**Test categories:**
1. **Unit Tests**: Simple template engine, explicit registry
2. **Integration Tests**: End-to-end code generation 
3. **Performance Tests**: Startup time, memory usage, generation speed
4. **Compatibility Tests**: Legacy API still works with deprecation warnings
5. **Template Tests**: All existing templates work with new engine

### **6.3 Performance Validation**

**Expected metrics:**
- **Startup time**: 80% faster (no auto-discovery)
- **Memory usage**: 70% lower (single bounded cache)
- **Code generation**: <5% performance impact
- **Codebase size**: ~800 lines removed

## 🎯 **Implementation Summary**

### **Files to Create:**
- `src/finn/codegen/simple_template_engine.py` (~150 lines)
- `src/finn/codegen/explicit_backend_registry.py` (~50 lines)
- `src/finn/codegen/backend_registration.py` (~100 lines)  
- `src/finn/codegen/simple_config.py` (~80 lines)
- `src/finn/codegen/legacy_compat.py` (~50 lines)

### **Files to Modify:**
- `src/finn/codegen/codegen.py` (simplify from 545 to ~300 lines)
- `src/finn/custom_op/fpgadataflow/hlsbackend.py` (add explicit template interface)
- `src/finn/custom_op/fpgadataflow/rtlbackend.py` (add explicit template interface)
- All HLS backends: add explicit template declarations
- All RTL backends: add explicit template declarations

### **Files to Remove/Deprecate:**
- `src/finn/codegen/template_engine.py` (467 lines - replace with 150-line version)
- `src/finn/codegen/backend_registry.py` (551 lines - replace with 50-line version)
- `src/finn/codegen/codegen_config.py` (complex config - replace with simple version)

### **Net Code Reduction:**
- **Remove:** ~1500 lines of complexity
- **Add:** ~430 lines of simple, explicit code
- **Net reduction:** ~1070 lines (42% smaller)

## 📊 **Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Template compatibility | Low | Medium | Comprehensive testing of existing templates |
| Performance regression | Low | Medium | Performance benchmarks for each phase |  
| Integration issues | Medium | High | Backward compatibility layer with gradual migration |
| Developer adoption | Low | Low | Clear migration guide and examples |

## 🗓️ **Timeline: 6 Weeks Total**

**Week 1:** Simple template engine + codegen base
**Week 2:** Explicit template architecture for backends  
**Week 3:** Replace backend registry complexity
**Week 4:** Multiple inheritance standardization
**Week 5:** Simplified configuration system
**Week 6:** Testing, compatibility, performance validation

**Outcome:** A dramatically simplified, explicit codegen architecture that eliminates complexity while maintaining performance through strategic caching of template compilation - the only expensive operation that truly matters.