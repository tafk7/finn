# Detailed Implementation Plan for Final Inheritance Architecture

## 🎯 **Executive Summary**

Transform the FINN unified codegen framework using the final inheritance architecture to fix test failures and create a robust, extensible code generation system.

**Estimated Total Timeline**: 6-8 weeks  
**Team Size**: 2-3 developers  
**Risk Level**: Medium (well-defined architecture, clear migration path)

---

## 📋 **Phase 1: Foundation - Base Class Creation**
*Duration: 1-2 weeks | Priority: Critical | Dependencies: None*

### **Objectives**
- Create the foundational abstract base classes
- Establish the inheritance hierarchy
- Set up shared infrastructure interfaces

### **Phase 1 Tasks**

#### **1.1 Create Codegen Abstract Base Class**
- [ ] **Create `src/finn/codegen/codegen.py`**
  - [ ] Define `Codegen(ABC)` class with shared infrastructure
  - [ ] Implement `generate_code()` method with template selection pipeline
  - [ ] Add `_select_optimal_template()` shared logic
  - [ ] Add `_validate_template_values()` shared validation
  - [ ] Add `_render_template()` shared rendering
  - [ ] Add `_post_process_code()` shared post-processing
  - [ ] Add `_extract_common_values()` helper method
  - [ ] Add `_safe_extract_value()` utility method
  - [ ] Add comprehensive logging and error handling

- [ ] **Abstract Methods for Subclasses**
  - [ ] `get_supported_templates() -> Set[str]` 
  - [ ] `get_template_values(template_name: str) -> Dict[str, Any]`
  - [ ] `_get_template_priority_order() -> List[str]`

- [ ] **Add Exception Classes**
  - [ ] `UnsupportedTemplateError` exception
  - [ ] `TemplateValidationError` exception
  - [ ] `CodeGenerationError` exception

#### **1.2 Update HLSBackend Abstract Class**
- [ ] **Modify `src/finn/custom_op/fpgadataflow/hlsbackend.py`**
  - [ ] Change inheritance: `class HLSBackend(Codegen):`
  - [ ] Remove duplicate methods now in `Codegen`
  - [ ] Override `_get_template_priority_order()` with HLS priorities
  - [ ] Add `generate_hls_code()` wrapper method
  - [ ] Add HLS-specific helper methods:
    - [ ] `_extract_hls_parallelization_values()`
    - [ ] `_infer_hls_parallelization_strategy()`
    - [ ] `_extract_hls_memory_config()`

#### **1.3 Update RTLBackend Abstract Class**
- [ ] **Modify `src/finn/custom_op/fpgadataflow/rtlbackend.py`**
  - [ ] Change inheritance: `class RTLBackend(Codegen):`
  - [ ] Remove duplicate methods now in `Codegen`
  - [ ] Override `_get_template_priority_order()` with RTL priorities
  - [ ] Add `generate_rtl_code()` wrapper method
  - [ ] Add RTL-specific helper methods:
    - [ ] `_extract_rtl_interface_values()`
    - [ ] `_generate_module_name()`
    - [ ] `_extract_data_width()`

#### **1.4 Clean HWCustomOp Base Class**
- [ ] **Verify `src/finn/custom_op/fpgadataflow/hwcustomop.py`**
  - [ ] Ensure clean, minimal interface focused on domain logic
  - [ ] Verify abstract methods: `get_nodeattr_types()`, `get_instream_width()`, etc.
  - [ ] No template-related methods (backends handle that)
  - [ ] No breaking changes to existing interface

#### **1.5 Phase 1 Testing**
- [ ] **Create Unit Tests**
  - [ ] `tests/test_codegen_base.py` - Test `Codegen` abstract functionality
  - [ ] `tests/test_hlsbackend_base.py` - Test `HLSBackend` abstract functionality  
  - [ ] `tests/test_rtlbackend_base.py` - Test `RTLBackend` abstract functionality

- [ ] **Integration Tests**
  - [ ] Test inheritance hierarchy works correctly
  - [ ] Test template priority selection logic
  - [ ] Test error handling and exception paths

### **Phase 1 Success Criteria**
- [ ] All abstract base classes created and tested
- [ ] Inheritance hierarchy established
- [ ] Shared infrastructure methods implemented
- [ ] Unit tests pass for all base classes
- [ ] No breaking changes to existing `HWCustomOp` interface

---

## 📋 **Phase 2: Core Infrastructure - Template Engine Integration**
*Duration: 1-2 weeks | Priority: Critical | Dependencies: Phase 1*

### **Objectives**
- Integrate template engine with new architecture
- Create template validation and rendering infrastructure
- Set up logging and error handling systems

### **Phase 2 Tasks**

#### **2.1 Template Engine Enhancement**
- [ ] **Update `src/finn/codegen/template_engine.py`**
  - [ ] Add `get_required_placeholders(template_name: str)` method
  - [ ] Add template dependency resolution
  - [ ] Add template validation before rendering
  - [ ] Add template caching for performance
  - [ ] Add template error reporting with line numbers

- [ ] **Template Registry System**
  - [ ] Create `src/finn/codegen/template_registry.py`
  - [ ] Implement `TemplateRegistry` class to track available templates
  - [ ] Add template metadata (required placeholders, target technology)
  - [ ] Add template compatibility validation

#### **2.2 Configuration System**
- [ ] **Create `src/finn/codegen/codegen_config.py`**
  - [ ] Template search paths configuration
  - [ ] Backend priority configuration
  - [ ] Logging level configuration
  - [ ] Debug mode configuration

#### **2.3 Utility Functions**
- [ ] **Create `src/finn/codegen/codegen_utils.py`**
  - [ ] Template value validation utilities
  - [ ] Code formatting utilities
  - [ ] Backend discovery utilities
  - [ ] Error reporting utilities

#### **2.4 Logging Infrastructure**
- [ ] **Enhance logging in `Codegen` base class**
  - [ ] Add structured logging for template selection
  - [ ] Add debug logging for template value extraction
  - [ ] Add performance logging for rendering times
  - [ ] Add error logging with context

#### **2.5 Phase 2 Testing**
- [ ] **Template Engine Tests**
  - [ ] `tests/test_template_engine.py` - Enhanced engine functionality
  - [ ] `tests/test_template_registry.py` - Registry functionality
  - [ ] `tests/test_template_validation.py` - Validation logic

- [ ] **Integration Tests**
  - [ ] End-to-end template rendering with new architecture
  - [ ] Error handling and recovery scenarios
  - [ ] Performance benchmarks for template rendering

### **Phase 2 Success Criteria**
- [ ] Template engine fully integrated with inheritance architecture
- [ ] Template validation working correctly
- [ ] Logging and error handling comprehensive
- [ ] Performance meets or exceeds existing system
- [ ] All integration tests pass

---

## 📋 **Phase 3: Technology Backend Implementation**
*Duration: 2-3 weeks | Priority: High | Dependencies: Phase 2*

### **Objectives**
- Implement concrete HLS and RTL backend classes for critical operations
- Create operation-specific template logic
- Ensure template value extraction works correctly

### **Phase 3 Tasks**

#### **3.1 Critical Operation Selection**
- [ ] **Priority Operations for Implementation**
  - [ ] Thresholding (causing current test failures)
  - [ ] MatrixVectorActivation (MVAU) 
  - [ ] AddStreams
  - [ ] ChannelwiseOp
  - [ ] ConvolutionInputGenerator

#### **3.2 Thresholding Backend Implementation**
- [ ] **Create `src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py`**
  - [ ] Implement `ThresholdingHLS(Thresholding, HLSBackend)`
  - [ ] Override `get_supported_templates()` with Thresholding HLS templates
  - [ ] Implement `get_template_values()` with Thresholding-specific logic
  - [ ] Add helper methods: `_get_thresholding_lut_values()`, `_get_streaming_values()`
  - [ ] Handle Thresholding characteristics: PE-only parallelization, embedded LUTs

- [ ] **Create `src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py`**
  - [ ] Implement `ThresholdingRTL(Thresholding, RTLBackend)`
  - [ ] Override `get_supported_templates()` with Thresholding RTL templates
  - [ ] Implement `get_template_values()` with RTL-specific values
  - [ ] Add RTL-specific helper methods for interface generation

#### **3.3 MVAU Backend Implementation**
- [ ] **Create `src/finn/custom_op/fpgadataflow/hls/mvau_hls.py`**
  - [ ] Implement `MVAU_HLS(MatrixVectorActivation, HLSBackend)`
  - [ ] Override `get_supported_templates()` with MVAU HLS templates
  - [ ] Implement `get_template_values()` using actual MVAU attributes
  - [ ] Handle MVAU characteristics: PE+SIMD parallelization, weight memory

- [ ] **Create `src/finn/custom_op/fpgadataflow/rtl/mvau_rtl.py`**
  - [ ] Implement `MVAU_RTL(MatrixVectorActivation, RTLBackend)`
  - [ ] Override `get_supported_templates()` with MVAU RTL templates
  - [ ] Implement `get_template_values()` with MVAU RTL-specific values

#### **3.4 Additional Operation Backends**
- [ ] **AddStreams Backend Implementation**
  - [ ] `src/finn/custom_op/fpgadataflow/hls/addstreams_hls.py`
  - [ ] `src/finn/custom_op/fpgadataflow/rtl/addstreams_rtl.py`

- [ ] **ChannelwiseOp Backend Implementation**
  - [ ] `src/finn/custom_op/fpgadataflow/hls/channelwise_hls.py`
  - [ ] `src/finn/custom_op/fpgadataflow/rtl/channelwise_rtl.py`

- [ ] **ConvolutionInputGenerator Backend Implementation**
  - [ ] `src/finn/custom_op/fpgadataflow/hls/convgen_hls.py`
  - [ ] `src/finn/custom_op/fpgadataflow/rtl/convgen_rtl.py`

#### **3.5 Backend Registry Implementation**
- [ ] **Create `src/finn/codegen/backend_registry.py`**
  - [ ] Implement `BackendRegistry` class
  - [ ] Add `find_hls_backend(operation)` method
  - [ ] Add `find_rtl_backend(operation)` method
  - [ ] Add backend auto-discovery
  - [ ] Add fallback backend handling

#### **3.6 Phase 3 Testing**
- [ ] **Backend Unit Tests**
  - [ ] `tests/test_thresholding_hls.py` - Thresholding HLS backend
  - [ ] `tests/test_thresholding_rtl.py` - Thresholding RTL backend
  - [ ] `tests/test_mvau_hls.py` - MVAU HLS backend
  - [ ] `tests/test_mvau_rtl.py` - MVAU RTL backend

- [ ] **Template Value Tests**
  - [ ] Test template value extraction for each operation
  - [ ] Test error handling for missing attributes
  - [ ] Test template compatibility validation

- [ ] **Integration Tests**
  - [ ] End-to-end code generation for each operation
  - [ ] Backend registry functionality
  - [ ] Cross-operation template compatibility

### **Phase 3 Success Criteria**
- [ ] All critical operation backends implemented
- [ ] Template value extraction working correctly
- [ ] Backend registry operational
- [ ] No more "Op has no such attribute" errors
- [ ] All backend tests passing

---

## 📋 **Phase 4: Operation Migration & Framework Integration**
*Duration: 1-2 weeks | Priority: High | Dependencies: Phase 3*

### **Objectives**
- Integrate new backend system with existing framework
- Update framework code generation calls
- Ensure backward compatibility where possible

### **Phase 4 Tasks**

#### **4.1 Framework Integration Points**
- [ ] **Update `src/finn/codegen/hls_generator.py`**
  - [ ] Remove unsafe direct attribute access (lines 347-349, 353-356, 73)
  - [ ] Replace with backend registry calls
  - [ ] Update `generate_hls_code()` to use new architecture
  - [ ] Add backward compatibility layer if needed

- [ ] **Update `src/finn/codegen/rtl_generator.py`**
  - [ ] Remove operation-specific RTL generation logic
  - [ ] Replace with backend registry calls
  - [ ] Update `generate_rtl_code()` to use new architecture

#### **4.2 Unified Code Generator Service**
- [ ] **Create `src/finn/codegen/unified_generator.py`**
  - [ ] Implement `UnifiedCodeGenerator` class
  - [ ] Add `generate_hls_code(operation)` method using backend registry
  - [ ] Add `generate_rtl_code(operation)` method using backend registry
  - [ ] Add operation type detection and routing

#### **4.3 Migration of Remaining Operations**
- [ ] **Audit all existing custom operations**
  - [ ] Identify operations still using old framework
  - [ ] Create migration priority list
  - [ ] Implement backend classes for remaining operations

- [ ] **Create Generic Fallback Backends**
  - [ ] `GenericHLSBackend` for operations without specific backends
  - [ ] `GenericRTLBackend` for operations without specific backends
  - [ ] Safe default template value provision

#### **4.4 Backward Compatibility Layer**
- [ ] **Create `src/finn/codegen/compatibility_layer.py`**
  - [ ] Wrapper functions for old framework API
  - [ ] Deprecation warnings for old usage patterns
  - [ ] Migration guidance for developers

#### **4.5 Configuration Updates**
- [ ] **Update configuration files**
  - [ ] Template search path configuration
  - [ ] Backend registration configuration
  - [ ] Feature flags for gradual rollout

#### **4.6 Phase 4 Testing**
- [ ] **Framework Integration Tests**
  - [ ] `tests/test_framework_integration.py` - Full framework testing
  - [ ] `tests/test_unified_generator.py` - Unified generator functionality
  - [ ] `tests/test_backward_compatibility.py` - Compatibility layer testing

- [ ] **Regression Tests**
  - [ ] Run existing test suite against new framework
  - [ ] Verify all existing operations still work
  - [ ] Performance regression testing

### **Phase 4 Success Criteria**
- [ ] Framework fully integrated with new architecture
- [ ] All existing operations continue working
- [ ] No performance regressions
- [ ] Backward compatibility maintained where feasible
- [ ] Original test failures resolved

---

## 📋 **Phase 5: Template Library & Optimization**
*Duration: 1-2 weeks | Priority: Medium | Dependencies: Phase 4*

### **Objectives**
- Create optimized templates for new architecture
- Organize template library efficiently
- Optimize template rendering performance

### **Phase 5 Tasks**

#### **5.1 Template Library Reorganization**
- [ ] **Create new template directory structure**
  ```
  src/finn/codegen/templates/
  ├── hls/
  │   ├── common/
  │   │   ├── hls_basic.cpp.j2
  │   │   ├── hls_streaming_generic.cpp.j2
  │   │   └── hls_parallel_generic.cpp.j2
  │   └── operation_specific/
  │       ├── hls_thresholding_lut.cpp.j2
  │       ├── hls_mvau_streaming.cpp.j2
  │       └── hls_channelwise.cpp.j2
  └── rtl/
      ├── common/
      │   ├── rtl_basic.v.j2
      │   ├── rtl_axi_stream_generic.v.j2
      │   └── rtl_parallel_generic.v.j2
      └── operation_specific/
          ├── rtl_thresholding_lut.v.j2
          ├── rtl_mvau_streaming.v.j2
          └── rtl_channelwise.v.j2
  ```

#### **5.2 Template Creation and Migration**
- [ ] **Create Common Templates**
  - [ ] `hls_basic.cpp.j2` - Universal basic HLS template
  - [ ] `hls_streaming_generic.cpp.j2` - Generic streaming HLS
  - [ ] `rtl_basic.v.j2` - Universal basic RTL template
  - [ ] `rtl_axi_stream_generic.v.j2` - Generic AXI Stream RTL

- [ ] **Create Operation-Specific Templates**
  - [ ] `hls_thresholding_lut.cpp.j2` - Thresholding-optimized HLS
  - [ ] `hls_mvau_streaming.cpp.j2` - MVAU-optimized HLS
  - [ ] `rtl_thresholding_lut.v.j2` - Thresholding-optimized RTL
  - [ ] `rtl_mvau_streaming.v.j2` - MVAU-optimized RTL

- [ ] **Migrate Existing Templates**
  - [ ] Update existing templates to work with new architecture
  - [ ] Add placeholder documentation
  - [ ] Add template metadata

#### **5.3 Template Optimization**
- [ ] **Template Performance Optimization**
  - [ ] Template inheritance to reduce duplication
  - [ ] Template macros for common patterns
  - [ ] Template caching for frequently used templates

- [ ] **Template Quality Improvements**
  - [ ] Code generation quality validation
  - [ ] Template syntax validation
  - [ ] Generated code formatting improvements

#### **5.4 Template Documentation**
- [ ] **Create Template Documentation**
  - [ ] Document all template placeholders
  - [ ] Create template usage examples
  - [ ] Document template inheritance hierarchy
  - [ ] Create template development guide

#### **5.5 Phase 5 Testing**
- [ ] **Template Quality Tests**
  - [ ] `tests/test_template_quality.py` - Generated code quality
  - [ ] `tests/test_template_performance.py` - Rendering performance
  - [ ] `tests/test_template_syntax.py` - Template syntax validation

- [ ] **Template Compatibility Tests**
  - [ ] Test all operation+template combinations
  - [ ] Validate generated code compiles correctly
  - [ ] Performance comparison with old templates

### **Phase 5 Success Criteria**
- [ ] Template library well-organized and documented
- [ ] Template rendering performance optimized
- [ ] Generated code quality maintained or improved
- [ ] All template tests passing

---

## 📋 **Phase 6: Testing, Documentation & Deployment**
*Duration: 1-2 weeks | Priority: High | Dependencies: Phase 5*

### **Objectives**
- Comprehensive testing of entire system
- Create complete documentation
- Plan and execute deployment strategy

### **Phase 6 Tasks**

#### **6.1 Comprehensive Testing**
- [ ] **System Integration Tests**
  - [ ] End-to-end workflow testing
  - [ ] Multi-operation model testing
  - [ ] Performance benchmarking
  - [ ] Memory usage analysis

- [ ] **Stress Testing**
  - [ ] Large model processing
  - [ ] Complex operation combinations
  - [ ] Error recovery scenarios
  - [ ] Resource limit testing

- [ ] **Original Test Failure Validation**
  - [ ] Run original failing tests
  - [ ] Verify 5/5 test phases now pass
  - [ ] Verify "Op has no such attribute: mem_mode" errors eliminated
  - [ ] Performance comparison with original system

#### **6.2 Documentation Creation**
- [ ] **Developer Documentation**
  - [ ] `docs/architecture_overview.md` - System architecture
  - [ ] `docs/backend_development_guide.md` - Creating new backends
  - [ ] `docs/operation_migration_guide.md` - Migrating operations
  - [ ] `docs/template_development_guide.md` - Creating templates

- [ ] **API Documentation**
  - [ ] Complete API reference for all classes
  - [ ] Usage examples for common scenarios
  - [ ] Migration guide from old system
  - [ ] Troubleshooting guide

- [ ] **Code Documentation**
  - [ ] Inline documentation for all methods
  - [ ] Class and module docstrings
  - [ ] Example usage in docstrings

#### **6.3 Deployment Planning**
- [ ] **Deployment Strategy**
  - [ ] Gradual rollout plan
  - [ ] Feature flags for controlled deployment
  - [ ] Rollback procedures
  - [ ] Monitoring and alerting setup

- [ ] **Migration Tools**
  - [ ] Operation compatibility checker
  - [ ] Backend migration wizard
  - [ ] Template validation tools
  - [ ] Performance comparison tools

#### **6.4 Training and Knowledge Transfer**
- [ ] **Team Training**
  - [ ] Architecture overview sessions
  - [ ] Hands-on development workshops
  - [ ] Best practices training
  - [ ] Troubleshooting training

#### **6.5 Phase 6 Testing**
- [ ] **Final Validation**
  - [ ] Complete test suite execution
  - [ ] Performance acceptance testing
  - [ ] Documentation review and validation
  - [ ] User acceptance testing

### **Phase 6 Success Criteria**
- [ ] All tests passing consistently
- [ ] Documentation complete and accurate
- [ ] Deployment plan approved and ready
- [ ] Team trained on new architecture
- [ ] Performance meets or exceeds requirements

---

## 🚨 **Risk Management & Mitigation**

### **High Risk Areas**
1. **Complex Inheritance Hierarchy**
   - *Risk*: Multiple inheritance conflicts
   - *Mitigation*: Comprehensive unit testing, careful method resolution order
   
2. **Template Compatibility**
   - *Risk*: Existing templates break with new architecture
   - *Mitigation*: Gradual migration, compatibility layer, thorough testing

3. **Performance Regression**
   - *Risk*: New architecture slower than existing
   - *Mitigation*: Performance benchmarking, optimization, caching

### **Medium Risk Areas**
1. **Operation Migration Complexity**
   - *Risk*: Some operations difficult to migrate
   - *Mitigation*: Generic fallback backends, phased migration

2. **Framework Integration**
   - *Risk*: Breaking existing integrations
   - *Mitigation*: Backward compatibility layer, gradual rollout

### **Mitigation Strategies**
- [ ] **Comprehensive Testing**: Unit, integration, and system tests at each phase
- [ ] **Gradual Rollout**: Feature flags and phased deployment
- [ ] **Rollback Plan**: Ability to revert to old system if needed
- [ ] **Performance Monitoring**: Continuous performance tracking
- [ ] **Documentation**: Clear documentation for troubleshooting

---

## ✅ **Success Metrics**

### **Primary Success Criteria**
- [ ] **Test Failures Eliminated**: Original "Op has no such attribute: mem_mode" errors resolved
- [ ] **Test Suite Passes**: 5/5 test phases pass consistently
- [ ] **Zero Breaking Changes**: Existing operations continue working
- [ ] **Performance Maintained**: No significant performance regression

### **Secondary Success Criteria**
- [ ] **Architecture Clean**: Clear separation of concerns achieved
- [ ] **Extensibility Improved**: New operations easy to add
- [ ] **Maintainability Enhanced**: Code easier to understand and modify
- [ ] **Documentation Complete**: Comprehensive guides and API docs

### **Quality Metrics**
- [ ] **Code Coverage**: >90% test coverage for new code
- [ ] **Performance**: Template generation within 10% of original performance
- [ ] **Reliability**: No regressions in existing functionality
- [ ] **Usability**: Developer productivity maintained or improved

---

## 🎯 **Final Deliverables**

### **Code Deliverables**
- [ ] Complete inheritance architecture implementation
- [ ] All critical operation backend classes
- [ ] Template library reorganization
- [ ] Framework integration updates
- [ ] Comprehensive test suite

### **Documentation Deliverables**
- [ ] Architecture overview and design documents
- [ ] Developer guides and API documentation
- [ ] Migration guides and best practices
- [ ] Troubleshooting and maintenance guides

### **Process Deliverables**
- [ ] Deployment procedures and rollback plans
- [ ] Testing procedures and quality gates
- [ ] Performance monitoring and alerting
- [ ] Training materials and knowledge transfer

This detailed implementation plan provides a clear roadmap for transforming the FINN unified codegen framework to fix the test failures while creating a robust, extensible architecture for future development.