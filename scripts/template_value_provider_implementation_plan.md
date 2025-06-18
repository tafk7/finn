# Template Value Provider Architecture - Detailed Implementation Plan

## 🎯 **Executive Summary**
Transform the unified codegen framework from tightly-coupled attribute accessor to clean template population service using the Template Value Provider pattern. This plan ensures "zero breaking changes" while fixing the core attribute access violations.

## 📋 **Phase 1: Interface & Base Implementation**
*Duration: 2-3 days | Priority: Critical*

### 1.1 Core Interface Definition
- [ ] **Create `TemplateValueProvider` abstract interface**
  - [ ] Define `get_template_values(template_name: str) -> Dict[str, Any]` method
  - [ ] Define `supports_template(template_name: str) -> bool` method
  - [ ] Define `get_supported_templates() -> Set[str]` method
  - [ ] Create `UnsupportedTemplateError` exception class
  - [ ] Add comprehensive docstrings with examples
  - [ ] **File**: `src/finn/codegen/template_value_provider.py`

### 1.2 Enhanced HWCustomOp Base Class
- [ ] **Modify `HWCustomOp` to implement `TemplateValueProvider`**
  - [ ] Add `TemplateValueProvider` to inheritance chain
  - [ ] Implement default `get_template_values()` with safe fallbacks
  - [ ] Implement default `supports_template()` method
  - [ ] Implement default `get_supported_templates()` method
  - [ ] **File**: `src/finn/custom_op/fpgadataflow/hwcustomop.py`

- [ ] **Create safe template value methods**
  - [ ] `_get_basic_template_values()` - common values all operations have
  - [ ] `_get_streaming_template_values()` - streaming-specific values
  - [ ] `_get_memory_template_values()` - memory configuration with safe defaults
  - [ ] Add error handling for missing attributes

### 1.3 Template Registry System
- [ ] **Create template placeholder registry**
  - [ ] Scan existing templates to identify all placeholders
  - [ ] Create `TemplateRegistry` class to track required placeholders
  - [ ] Add validation methods to ensure completeness
  - [ ] **File**: `src/finn/codegen/template_registry.py`

### 1.4 Testing Infrastructure
- [ ] **Create unit tests for interface**
  - [ ] Test `TemplateValueProvider` abstract methods
  - [ ] Test `HWCustomOp` default implementations
  - [ ] Test error handling for unsupported templates
  - [ ] **File**: `tests/test_template_value_provider.py`

### 1.5 Phase 1 Validation
- [ ] **Verify base implementation works**
  - [ ] All existing operations inherit `TemplateValueProvider` interface
  - [ ] Default template values can be retrieved without errors
  - [ ] Template support detection works correctly
  - [ ] Unit tests pass

---

## 📋 **Phase 2: Framework Decoupling**
*Duration: 3-4 days | Priority: Critical*

### 2.1 Unified Code Generator Service
- [ ] **Create template-agnostic code generator**
  - [ ] Implement `UnifiedCodeGenerator` class
  - [ ] Add `generate_code(operation, template_name)` method
  - [ ] Add template value validation logic
  - [ ] Add comprehensive error handling
  - [ ] **File**: `src/finn/codegen/unified_generator.py`

### 2.2 HLS Generator Refactoring
- [ ] **Remove unsafe attribute access from `hls_generator.py`**
  - [ ] ❌ Remove `_get_memory_config()` direct attribute access (lines 347-349)
  - [ ] ❌ Remove `_get_parallelization_config()` direct attribute access (lines 353-356)
  - [ ] ❌ Remove `_get_mvau_template_name()` direct attribute access (line 73)
  - [ ] **File**: `src/finn/codegen/hls_generator.py`

- [ ] **Replace with template value provider calls**
  - [ ] ✅ Replace with `operation.get_template_values(template_name)` calls
  - [ ] ✅ Add template support validation before generation
  - [ ] ✅ Add graceful error handling for unsupported templates
  - [ ] ✅ Maintain existing public API for backward compatibility

### 2.3 Legacy Compatibility Bridge
- [ ] **Create backward compatibility system**
  - [ ] Implement `LegacyOperationBridge` for operations not yet migrated
  - [ ] Implement `LegacyTemplateValueAdapter` with safe attribute access
  - [ ] Add migration detection and warnings
  - [ ] **File**: `src/finn/codegen/legacy_bridge.py`

### 2.4 Template Engine Integration
- [ ] **Enhance template engine capabilities**
  - [ ] Add `get_required_placeholders(template_name)` method
  - [ ] Add placeholder validation before rendering
  - [ ] Add template dependency resolution
  - [ ] **File**: `src/finn/codegen/template_engine.py`

### 2.5 Phase 2 Validation
- [ ] **Verify framework decoupling**
  - [ ] Framework no longer directly accesses operation attributes
  - [ ] All template generation goes through `TemplateValueProvider` interface
  - [ ] Legacy operations still work through compatibility bridge
  - [ ] No breaking changes to existing API

---

## 📋 **Phase 3: Operation-Specific Implementation**
*Duration: 4-5 days | Priority: High*

### 3.1 Thresholding Operation Implementation
- [ ] **Implement `Thresholding` template value provider**
  - [ ] Override `get_template_values()` with Thresholding-specific logic
  - [ ] Implement `_get_thresholding_basic_values()`
  - [ ] Implement `_get_thresholding_streaming_values()`
  - [ ] Implement `_get_thresholding_memory_values()` with proper defaults
  - [ ] **File**: `src/finn/custom_op/fpgadataflow/thresholding.py`

- [ ] **Add Thresholding-specific template logic**
  - [ ] Handle PE parallelization (no SIMD)
  - [ ] Handle `NumChannels`, `weightDataType`, `ActVal` attributes
  - [ ] Set appropriate memory defaults: `mem_mode='const_embedded'`, `ram_style='distributed'`
  - [ ] Add template support for `hls_basic`, `hls_streaming`, `hls_memory`

### 3.2 MVAU Operation Implementation
- [ ] **Implement `MatrixVectorActivation` template value provider**
  - [ ] Override `get_template_values()` with MVAU-specific logic
  - [ ] Implement `_get_mvau_streaming_values()` (PE + SIMD)
  - [ ] Implement `_get_mvau_parallel_values()`
  - [ ] Implement `_get_mvau_memory_values()` with actual `mem_mode` attribute
  - [ ] **File**: `src/finn/custom_op/fpgadataflow/matrixvectoractivation.py`

### 3.3 Additional High-Priority Operations
- [ ] **Implement `AddStreams` template value provider**
  - [ ] Handle PE parallelization, no SIMD
  - [ ] Handle `inFIFODepths` specific configuration
  - [ ] **File**: `src/finn/custom_op/fpgadataflow/addstreams.py`

- [ ] **Implement `ChannelwiseOp` template value provider**
  - [ ] Handle `ram_style` attribute (has it)
  - [ ] Handle `Func`, `paramDataType` attributes
  - [ ] **File**: `src/finn/custom_op/fpgadataflow/channelwise_op.py`

- [ ] **Implement `ConvolutionInputGenerator` template value provider**
  - [ ] Handle SIMD parallelization (no PE)
  - [ ] Handle `IFMChannels` instead of `NumChannels`
  - [ ] Handle complex convolution-specific attributes
  - [ ] **File**: `src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py`

### 3.4 Operation Family Templates
- [ ] **Create operation-family-specific templates**
  - [ ] `hls_thresholding.j2` - optimized for thresholding operations
  - [ ] `hls_mvau_streaming.j2` - optimized for MVAU with streaming
  - [ ] `hls_mvau_parallel.j2` - optimized for MVAU with parallelization
  - [ ] `hls_channelwise.j2` - optimized for channelwise operations

### 3.5 Phase 3 Validation
- [ ] **Verify operation-specific implementations**
  - [ ] Thresholding operations no longer fail with "mem_mode" errors
  - [ ] MVAU operations work with actual `mem_mode` attributes
  - [ ] All high-priority operations provide appropriate template values
  - [ ] Operations correctly declare supported templates

---

## 📋 **Phase 4: Template Analysis & Optimization**
*Duration: 2-3 days | Priority: Medium*

### 4.1 Comprehensive Template Analysis
- [ ] **Audit all existing templates**
  - [ ] Identify all placeholders in existing `.j2` template files
  - [ ] Create comprehensive placeholder inventory
  - [ ] Map placeholders to operation families
  - [ ] **Document**: `scripts/template_placeholder_analysis.md`

- [ ] **Template dependency analysis**
  - [ ] Identify required vs optional placeholders
  - [ ] Create template compatibility matrix
  - [ ] Document placeholder defaults and fallbacks

### 4.2 Template Optimization
- [ ] **Create optimized operation-family templates**
  - [ ] Streamline templates for specific operation patterns
  - [ ] Remove unnecessary placeholders
  - [ ] Add operation-specific optimizations

- [ ] **Template inheritance system**
  - [ ] Create base templates with common patterns
  - [ ] Create specialized templates that extend base templates
  - [ ] Reduce template duplication

### 4.3 Documentation & Examples
- [ ] **Create comprehensive documentation**
  - [ ] Template Value Provider developer guide
  - [ ] Examples for implementing new operations
  - [ ] Template creation best practices
  - [ ] **File**: `docs/template_value_provider_guide.md`

- [ ] **Create example implementations**
  - [ ] Simple operation example
  - [ ] Complex operation example
  - [ ] Custom template example

### 4.4 Phase 4 Validation
- [ ] **Verify template optimization**
  - [ ] All templates have documented placeholders
  - [ ] Template inheritance reduces duplication
  - [ ] Examples demonstrate best practices
  - [ ] Documentation is comprehensive

---

## 🧪 **Testing & Validation Strategy**

### Integration Testing
- [ ] **End-to-end template generation tests**
  - [ ] Test Thresholding operation with various templates
  - [ ] Test MVAU operation with specialized templates
  - [ ] Test error handling for unsupported templates
  - [ ] **File**: `tests/test_template_integration.py`

### Regression Testing
- [ ] **Verify existing functionality preserved**
  - [ ] All existing operations continue to work
  - [ ] Generated code is identical to previous versions
  - [ ] Performance characteristics unchanged
  - [ ] **File**: `tests/test_regression.py`

### Performance Testing
- [ ] **Measure template generation performance**
  - [ ] Compare new vs old generation times
  - [ ] Identify any performance regressions
  - [ ] Optimize bottlenecks if necessary
  - [ ] **File**: `tests/test_performance.py`

---

## 🚀 **Deployment & Migration Strategy**

### Rollout Plan
- [ ] **Phase-by-phase deployment**
  - [ ] Deploy Phase 1 & 2 together (interface + framework)
  - [ ] Deploy Phase 3 operations incrementally
  - [ ] Deploy Phase 4 optimizations as enhancement

### Migration Support
- [ ] **Developer migration tools**
  - [ ] Create migration checklist for new operations
  - [ ] Provide template value provider skeleton generator
  - [ ] Add validation tools for template compliance

### Monitoring & Validation
- [ ] **Post-deployment validation**
  - [ ] Monitor for template generation errors
  - [ ] Track operation coverage and compliance
  - [ ] Gather developer feedback on new interface

---

## ✅ **Success Criteria**

### Technical Success Metrics
- [ ] **Zero Breaking Changes**: All existing operations continue to work
- [ ] **Error Resolution**: Thresholding operations no longer fail with attribute errors
- [ ] **Framework Decoupling**: No operation-specific logic in framework code
- [ ] **Template Coverage**: All templates can be populated by appropriate operations

### Architectural Success Metrics
- [ ] **Extensibility**: New operations can be added without framework changes
- [ ] **Maintainability**: Template evolution doesn't require coordinated changes
- [ ] **Testability**: Each component can be tested independently
- [ ] **Documentation**: Comprehensive developer guides and examples

### Performance Success Metrics
- [ ] **Generation Speed**: Template generation performance equivalent or better
- [ ] **Code Quality**: Generated code quality maintained or improved
- [ ] **Resource Usage**: Memory and CPU usage within acceptable bounds

---

## 🔄 **Dependencies & Prerequisites**

### Technical Dependencies
- [ ] Python 3.7+ with ABC support
- [ ] Existing FINN custom operation infrastructure
- [ ] Jinja2 template engine (or equivalent)
- [ ] Comprehensive test suite infrastructure

### Knowledge Dependencies
- [ ] Understanding of existing FINN operation patterns
- [ ] Template engine expertise
- [ ] Python interface design patterns
- [ ] FINN code generation workflow

### Process Dependencies
- [ ] Code review process for interface changes
- [ ] Testing strategy for backward compatibility
- [ ] Documentation update procedures
- [ ] Deployment and rollback procedures

---

This implementation plan transforms the unified codegen framework into a truly extensible, maintainable system that honors its "zero breaking changes" architectural promise while fixing the core design violations that cause test failures.