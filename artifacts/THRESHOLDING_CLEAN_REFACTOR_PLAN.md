# FINN Thresholding Clean Refactor Plan
**No Legacy Systems - Pure Template-Driven Architecture**

## Overview

This plan outlines a complete refactor of FINN's Thresholding operation (both HLS and RTL) to eliminate all legacy dependencies and create a modern, maintainable template-driven architecture.

---

## Current State Analysis

### **What We Have:**
- `ThresholdingHLS`: Mixed legacy/modern approach with `code_gen_dict` dependency
- `Thresholding_rtl`: Pure legacy string replacement using `$VAR$` syntax
- Complex `prepare_codegen_rtl_values()` method (282 lines)
- Template infrastructure already exists but not properly utilized

### **Critical Issues:**
- ❌ **Legacy Dependency**: Both implementations still use `code_gen_dict`
- ❌ **String Replacement**: RTL uses primitive `template.replace()` approach
- ❌ **Monolithic Methods**: 100+ line methods doing multiple responsibilities
- ❌ **Poor Separation**: Template logic mixed with domain logic
- ❌ **No Validation**: No template syntax or semantic validation

---

## Target Architecture

### **Core Principles:**
1. **Zero Legacy Dependency**: No `code_gen_dict` usage
2. **Pure Template-Driven**: Jinja2 templates for all code generation
3. **Separation of Concerns**: Clear boundaries between domain logic and template rendering
4. **Explicit Configuration**: No auto-discovery or implicit behavior
5. **Comprehensive Validation**: Template syntax and semantic correctness

### **Architecture Components:**

```
ThresholdingOperation (Base Domain Logic)
├── ThresholdingHLS (HLS Implementation)
│   ├── ThresholdingHLSTemplateProvider
│   └── ThresholdingHLSValidator
└── ThresholdingRTL (RTL Implementation)
    ├── ThresholdingRTLTemplateProvider
    └── ThresholdingRTLValidator

Templates/
├── hls/
│   ├── thresholding_basic.cpp.j2
│   ├── thresholding_streaming.cpp.j2
│   ├── thresholding_parallel.cpp.j2
│   └── thresholding_lut.cpp.j2
└── rtl/
    ├── thresholding_wrapper.sv.j2
    ├── thresholding_core.sv.j2
    └── thresholding_mem_init.dat.j2
```

---

## Implementation Plan

### **Phase 1: Clean Domain Model (Week 1)**

#### **1.1 Create Pure Domain Logic**
```python
# src/finn/custom_op/fpgadataflow/thresholding/core.py
class ThresholdingOperation:
    """Pure domain logic for Thresholding operations - no codegen concerns."""
    
    def __init__(self, onnx_node):
        self.onnx_node = onnx_node
        self._validate_parameters()
    
    def get_threshold_parameters(self) -> ThresholdingParameters:
        """Extract all thresholding-specific parameters."""
        return ThresholdingParameters(
            num_channels=self.get_nodeattr("NumChannels"),
            pe_factor=self.get_nodeattr("PE"),
            input_datatype=self.get_input_datatype(0),
            output_datatype=self.get_output_datatype(),
            activation_value=self.get_nodeattr("ActVal"),
            num_steps=self.get_nodeattr("numSteps")
        )
    
    def get_memory_requirements(self) -> MemoryRequirements:
        """Calculate memory requirements for threshold storage."""
        pass
    
    def validate_configuration(self) -> ValidationResult:
        """Validate thresholding configuration for correctness."""
        pass
```

#### **1.2 Create Configuration Data Classes**
```python
# src/finn/custom_op/fpgadataflow/thresholding/config.py
@dataclass(frozen=True)
class ThresholdingParameters:
    """Immutable thresholding parameters."""
    num_channels: int
    pe_factor: int
    input_datatype: DataType
    output_datatype: DataType
    activation_value: int
    num_steps: int
    
    def __post_init__(self):
        self._validate()

@dataclass(frozen=True)
class MemoryRequirements:
    """Memory requirements for threshold storage."""
    threshold_depth: int
    threshold_width: int
    memory_type: str  # 'BRAM', 'URAM', 'LUTRAM'
    total_memory_blocks: int

@dataclass(frozen=True)
class ThresholdingHLSConfig:
    """HLS-specific configuration."""
    template_name: str
    parallelization_strategy: str
    memory_interface: str
    optimization_target: str  # 'area', 'performance', 'power'

@dataclass(frozen=True)
class ThresholdingRTLConfig:
    """RTL-specific configuration."""
    template_name: str
    memory_primitive: str
    pipeline_depth: int
    use_axilite: bool
```

### **Phase 2: Template Infrastructure (Week 2)**

#### **2.1 Create Modern Jinja2 Templates**

**HLS Templates:**
```jinja2
{# src/finn/codegen/templates/thresholding/hls/thresholding_streaming.cpp.j2 #}
// Generated Thresholding HLS Implementation - Streaming Architecture
#include "ap_int.h"
#include "hls_stream.h"
#include "finn_thresholding.hpp"

// Configuration Constants
static const unsigned CHANNELS = {{ config.num_channels }};
static const unsigned PE = {{ config.pe_factor }};
static const unsigned CHANNELS_PER_PE = CHANNELS / PE;
static const unsigned INPUT_BITS = {{ config.input_datatype.bitwidth() }};
static const unsigned OUTPUT_BITS = {{ config.output_datatype.bitwidth() }};

// Threshold Memory Type
{% if config.memory_type == 'LUTRAM' %}
static const ap_uint<{{ config.threshold_width }}> thresholds[PE][CHANNELS_PER_PE][{{ config.threshold_depth }}] = {
    // Threshold data will be inserted here
    {% for pe in range(config.pe_factor) %}
    { // PE {{ pe }}
        {% for ch in range(config.channels_per_pe) %}
        { {{ config.threshold_data[pe][ch] | join(', ') }} }{{ ',' if not loop.last }}
        {% endfor %}
    }{{ ',' if not loop.last }}
    {% endfor %}
};
{% endif %}

void {{ config.function_name }}(
    hls::stream<ap_uint<{{ config.input_stream_width }}>>& input_stream,
    hls::stream<ap_uint<{{ config.output_stream_width }}>>& output_stream
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
    
    // Streaming processing loop
    // Implementation details...
}
```

**RTL Templates:**
```systemverilog
{# src/finn/codegen/templates/thresholding/rtl/thresholding_wrapper.sv.j2 #}
// Generated Thresholding RTL Wrapper
module {{ config.module_name }} #(
    parameter N = {{ config.output_bits }},
    parameter WI = {{ config.input_bits }},
    parameter WT = {{ config.threshold_bits }},
    parameter C = {{ config.num_channels }},
    parameter PE = {{ config.pe_factor }}
)(
    input logic clk,
    input logic rst_n,
    
    // AXI Stream Input
    input logic [{{ config.input_stream_width-1 }}:0] s_axis_tdata,
    input logic s_axis_tvalid,
    output logic s_axis_tready,
    
    // AXI Stream Output
    output logic [{{ config.output_stream_width-1 }}:0] m_axis_tdata,
    output logic m_axis_tvalid,
    input logic m_axis_tready
    
    {% if config.use_axilite %}
    // AXI-Lite Interface for runtime reconfiguration
    , axi_lite_if.slave s_axilite
    {% endif %}
);

// Threshold memory initialization
{% for pe in range(config.pe_factor) %}
initial begin
    $readmemh("{{ config.threshold_files[pe] }}", thresholds_pe{{ pe }});
end
{% endfor %}

// Core thresholding instance
thresholding_core #(
    .N(N), .WI(WI), .WT(WT), .C(C), .PE(PE)
) core_inst (
    .clk(clk),
    .rst_n(rst_n),
    // Connect streams...
);

endmodule
```

#### **2.2 Template Provider Classes**
```python
# src/finn/custom_op/fpgadataflow/thresholding/hls/template_provider.py
class ThresholdingHLSTemplateProvider:
    """Provides HLS template values for Thresholding operations."""
    
    def __init__(self, operation: ThresholdingOperation):
        self.operation = operation
        self.template_engine = TemplateEngine()
    
    def get_template_config(self, template_name: str) -> ThresholdingHLSConfig:
        """Get configuration for specific template."""
        if template_name == "thresholding_streaming.cpp.j2":
            return self._get_streaming_config()
        elif template_name == "thresholding_parallel.cpp.j2":
            return self._get_parallel_config()
        else:
            raise ValueError(f"Unsupported template: {template_name}")
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Get all values needed for template rendering."""
        config = self.get_template_config(template_name)
        params = self.operation.get_threshold_parameters()
        
        return {
            'config': {
                'num_channels': params.num_channels,
                'pe_factor': params.pe_factor,
                'input_datatype': params.input_datatype,
                'output_datatype': params.output_datatype,
                'input_stream_width': self._calculate_input_stream_width(),
                'output_stream_width': self._calculate_output_stream_width(),
                'function_name': f"thresholding_{self.operation.onnx_node.name}",
                'threshold_data': self._extract_threshold_data(),
                'memory_type': config.memory_interface,
                'threshold_width': self._calculate_threshold_width(),
                'threshold_depth': self._calculate_threshold_depth(),
                'channels_per_pe': params.num_channels // params.pe_factor
            }
        }
    
    def render_template(self, template_name: str) -> str:
        """Render template with calculated values."""
        template_values = self.get_template_values(template_name)
        return self.template_engine.render_template(template_name, template_values)
```

### **Phase 3: Clean Backend Implementation (Week 3)**

#### **3.1 Modern HLS Backend**
```python
# src/finn/custom_op/fpgadataflow/thresholding/hls/backend.py
class ThresholdingHLS(ThresholdingOperation):
    """Clean HLS backend for Thresholding - NO legacy dependencies."""
    
    # Explicit template declaration
    SUPPORTED_TEMPLATES = {
        'streaming': 'thresholding/hls/thresholding_streaming.cpp.j2',
        'parallel': 'thresholding/hls/thresholding_parallel.cpp.j2', 
        'basic': 'thresholding/hls/thresholding_basic.cpp.j2',
        'lut_optimized': 'thresholding/hls/thresholding_lut.cpp.j2'
    }
    
    def __init__(self, onnx_node, template_strategy: str = 'auto'):
        super().__init__(onnx_node)
        self.template_provider = ThresholdingHLSTemplateProvider(self)
        self.validator = ThresholdingHLSValidator(self)
        self.template_strategy = template_strategy
    
    def select_template(self) -> str:
        """Select optimal template based on operation characteristics."""
        if self.template_strategy != 'auto':
            return self.SUPPORTED_TEMPLATES[self.template_strategy]
        
        params = self.get_threshold_parameters()
        
        # Smart template selection logic
        if params.pe_factor > 1:
            return self.SUPPORTED_TEMPLATES['parallel']
        elif self._should_use_lut_optimization():
            return self.SUPPORTED_TEMPLATES['lut_optimized']
        else:
            return self.SUPPORTED_TEMPLATES['streaming']
    
    def generate_hls_code(self) -> str:
        """Generate HLS code using pure template approach."""
        template_name = self.select_template()
        
        # Validate before generation
        validation_result = self.validator.validate_for_template(template_name)
        if not validation_result.is_valid:
            raise ThresholdingValidationError(validation_result.errors)
        
        # Generate code
        return self.template_provider.render_template(template_name)
    
    def generate_synthesis_files(self, output_dir: Path) -> List[Path]:
        """Generate all files needed for HLS synthesis."""
        generated_files = []
        
        # Main HLS file
        hls_code = self.generate_hls_code()
        hls_file = output_dir / f"{self.onnx_node.name}.cpp"
        hls_file.write_text(hls_code)
        generated_files.append(hls_file)
        
        # TCL synthesis script
        tcl_code = self._generate_synthesis_script()
        tcl_file = output_dir / f"synthesize_{self.onnx_node.name}.tcl"
        tcl_file.write_text(tcl_code)
        generated_files.append(tcl_file)
        
        # Threshold data files if needed
        if self._needs_threshold_files():
            threshold_files = self._generate_threshold_files(output_dir)
            generated_files.extend(threshold_files)
        
        return generated_files
```

#### **3.2 Modern RTL Backend**
```python
# src/finn/custom_op/fpgadataflow/thresholding/rtl/backend.py
class ThresholdingRTL(ThresholdingOperation):
    """Clean RTL backend for Thresholding - NO legacy dependencies."""
    
    SUPPORTED_TEMPLATES = {
        'basic': 'thresholding/rtl/thresholding_wrapper.sv.j2',
        'optimized': 'thresholding/rtl/thresholding_optimized.sv.j2'
    }
    
    def __init__(self, onnx_node, template_strategy: str = 'basic'):
        super().__init__(onnx_node)
        self.template_provider = ThresholdingRTLTemplateProvider(self)
        self.validator = ThresholdingRTLValidator(self)
        self.template_strategy = template_strategy
    
    def generate_rtl_code(self) -> str:
        """Generate RTL code using pure template approach."""
        template_name = self.SUPPORTED_TEMPLATES[self.template_strategy]
        
        # Validate before generation
        validation_result = self.validator.validate_for_template(template_name)
        if not validation_result.is_valid:
            raise ThresholdingValidationError(validation_result.errors)
        
        # Generate code
        return self.template_provider.render_template(template_name)
    
    def generate_synthesis_files(self, output_dir: Path) -> List[Path]:
        """Generate all files needed for RTL synthesis."""
        generated_files = []
        
        # Main RTL wrapper
        rtl_code = self.generate_rtl_code()
        rtl_file = output_dir / f"{self.onnx_node.name}_wrapper.sv"
        rtl_file.write_text(rtl_code)
        generated_files.append(rtl_file)
        
        # Memory initialization files
        memory_files = self._generate_memory_init_files(output_dir)
        generated_files.extend(memory_files)
        
        # Copy required RTL library files
        library_files = self._copy_rtl_library_files(output_dir)
        generated_files.extend(library_files)
        
        return generated_files
```

### **Phase 4: Validation Framework (Week 4)**

#### **4.1 Template Validation**
```python
# src/finn/custom_op/fpgadataflow/thresholding/validation.py
class ThresholdingValidator:
    """Validates Thresholding configurations and generated code."""
    
    def validate_parameters(self, params: ThresholdingParameters) -> ValidationResult:
        """Validate thresholding parameters for correctness."""
        errors = []
        
        # Channel/PE alignment
        if params.num_channels % params.pe_factor != 0:
            errors.append("NumChannels must be divisible by PE factor")
        
        # Data type compatibility
        if not self._validate_datatype_compatibility(params):
            errors.append("Input/output datatype mismatch")
        
        return ValidationResult(len(errors) == 0, errors)
    
    def validate_generated_code(self, code: str, language: str) -> ValidationResult:
        """Validate generated code syntax."""
        if language == 'cpp':
            return self._validate_cpp_syntax(code)
        elif language == 'systemverilog':
            return self._validate_sv_syntax(code)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def validate_template_output_equivalence(self, 
                                           old_output: str, 
                                           new_output: str) -> ValidationResult:
        """Validate that new template produces equivalent output."""
        # Character-by-character comparison after normalization
        normalized_old = self._normalize_code(old_output)
        normalized_new = self._normalize_code(new_output)
        
        if normalized_old == normalized_new:
            return ValidationResult(True, [])
        else:
            diff = self._generate_diff(normalized_old, normalized_new)
            return ValidationResult(False, [f"Output mismatch: {diff}"])
```

#### **4.2 Integration Testing**
```python
# tests/thresholding/test_clean_refactor.py
class TestThresholdingCleanRefactor:
    """Comprehensive testing for clean Thresholding refactor."""
    
    def test_hls_code_generation(self):
        """Test HLS code generation with real parameters."""
        # Create real ThresholdingHLS instance
        node = create_thresholding_onnx_node()
        hls_backend = ThresholdingHLS(node)
        
        # Generate code
        code = hls_backend.generate_hls_code()
        
        # Validate syntax
        assert self._is_valid_cpp(code)
        
        # Validate contains expected elements
        assert "thresholding_" in code
        assert "#pragma HLS" in code
        assert "ap_uint" in code
    
    def test_rtl_code_generation(self):
        """Test RTL code generation with real parameters."""
        node = create_thresholding_onnx_node()
        rtl_backend = ThresholdingRTL(node)
        
        # Generate code
        code = rtl_backend.generate_rtl_code()
        
        # Validate syntax
        assert self._is_valid_systemverilog(code)
        
        # Validate contains expected elements
        assert "module" in code
        assert "thresholding_core" in code
    
    def test_equivalence_with_legacy(self):
        """Test that new implementation produces equivalent results."""
        # Generate with legacy system
        legacy_output = generate_with_legacy_system()
        
        # Generate with new system
        node = create_thresholding_onnx_node()
        new_backend = ThresholdingHLS(node)
        new_output = new_backend.generate_hls_code()
        
        # Validate equivalence
        validator = ThresholdingValidator()
        result = validator.validate_template_output_equivalence(legacy_output, new_output)
        assert result.is_valid, f"Outputs not equivalent: {result.errors}"
    
    def test_end_to_end_synthesis(self):
        """Test end-to-end synthesis flow."""
        node = create_thresholding_onnx_node()
        hls_backend = ThresholdingHLS(node)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            files = hls_backend.generate_synthesis_files(Path(temp_dir))
            
            # Validate all files exist
            assert len(files) > 0
            for file in files:
                assert file.exists()
            
            # Validate synthesis succeeds (if HLS tools available)
            if hls_tools_available():
                result = run_hls_synthesis(temp_dir)
                assert result.success
```

---

## Success Criteria

### **Technical Requirements:**
1. **Zero Legacy Dependency**: No `code_gen_dict` usage anywhere
2. **Pure Template System**: All code generation via Jinja2 templates
3. **Syntactic Correctness**: Generated code compiles without errors
4. **Semantic Equivalence**: New system produces identical output to legacy
5. **Comprehensive Validation**: Template syntax and semantic validation
6. **Performance Parity**: No regression in synthesis time or quality

### **Quality Requirements:**
1. **Test Coverage**: >90% line coverage with real-world tests
2. **Documentation**: Complete API documentation with examples
3. **Maintainability**: Clean, modular code with clear separation of concerns
4. **Extensibility**: Easy to add new template variants

### **Integration Requirements:**
1. **Backward Compatibility**: Existing FINN workflows continue working
2. **Drop-in Replacement**: New backends replace old without API changes
3. **Tool Integration**: Works with existing HLS/RTL synthesis flows

---

## Timeline & Milestones

| Week | Phase | Deliverables | Success Criteria |
|------|-------|-------------|------------------|
| 1 | Domain Model | Pure domain classes, config data classes | Unit tests pass, clear separation |
| 2 | Templates | Jinja2 templates, template providers | Templates render without errors |
| 3 | Backends | Clean HLS/RTL backend implementations | Generated code compiles |
| 4 | Validation | Validation framework, integration tests | Equivalence with legacy verified |

---

## Risk Mitigation

### **High Risk: Output Equivalence**
- **Mitigation**: Character-by-character diff validation between old/new
- **Fallback**: Maintain legacy system until equivalence proven

### **Medium Risk: Template Complexity**
- **Mitigation**: Start with simple templates, incrementally add complexity
- **Fallback**: Use simpler templates with post-processing if needed

### **Low Risk: Performance Regression**
- **Mitigation**: Benchmark template rendering performance
- **Fallback**: Template caching and optimization

---

## Expected Benefits

1. **Maintainability**: Clean, modular architecture easy to understand and modify
2. **Extensibility**: Adding new Thresholding variants becomes trivial
3. **Reliability**: Comprehensive validation prevents code generation errors
4. **Performance**: Template-driven approach enables optimizations
5. **Developer Experience**: Clear APIs and comprehensive testing
6. **Foundation**: Establishes pattern for refactoring other FINN operations

This plan provides a roadmap for creating a modern, maintainable Thresholding implementation that eliminates all legacy dependencies while maintaining full compatibility with existing FINN workflows.