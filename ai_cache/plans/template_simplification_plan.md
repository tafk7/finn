# FINN Template Simplification Plan

**Date**: December 19, 2024  
**Objective**: Simplify FINN template system to improve accessibility for open source contributors  
**Timeline**: 4 phases over 8 weeks

## Phase 1: Assessment and Examples (Week 1-2)

### Goals
- Create working examples of simplified templates
- Test simplification approach with real operations
- Gather feedback from contributors

### Tasks
- [ ] Create simplified template for Thresholding HLS
- [ ] Create simplified template for MVAU HLS  
- [ ] Create simplified template for basic RTL operation
- [ ] Document variable naming conventions
- [ ] Build side-by-side comparison with complex templates
- [ ] Test code generation with simplified templates

### Deliverables
1. `examples/simple_templates/` directory with working examples
2. Template simplification guide
3. Comparison metrics (lines of code, files needed, etc.)

## Phase 2: Infrastructure Updates (Week 3-4)

### Goals
- Update template engine to support simple templates
- Maintain backward compatibility with complex templates
- Add configuration for template style preference

### Tasks
- [ ] Add simple template discovery in template engine
- [ ] Create `SimplifiedCodegen` base class option
- [ ] Add template style configuration option
- [ ] Update backend registration to support both styles
- [ ] Create template validation tools
- [ ] Add debugging helpers for simple templates

### Deliverables
1. Updated `template_engine.py` with simple template support
2. New `simple_codegen.py` base class
3. Configuration system for template preferences

## Phase 3: Migration of Core Operations (Week 5-6)

### Goals
- Migrate high-usage operations to simple templates
- Create migration guide for other operations
- Ensure performance parity

### Priority Operations to Migrate
1. **Thresholding** (both HLS and RTL)
2. **MVAU** (MatrixVectorActivation)
3. **Convolution/Im2Col** 
4. **Pooling**
5. **StreamingDataWidthConverter**

### Tasks
- [ ] Create simple templates for each priority operation
- [ ] Update operation classes to use simple templates
- [ ] Add configuration flag to choose template style
- [ ] Performance testing (generation speed)
- [ ] Correctness testing (generated code matches)
- [ ] Document migration process

### Deliverables
1. Simple templates for 5+ core operations
2. Migration guide document
3. Performance comparison report

## Phase 4: Documentation and Rollout (Week 7-8)

### Goals
- Complete documentation for simple template system
- Enable community contributions
- Plan for full migration

### Tasks
- [ ] Write comprehensive template authoring guide
- [ ] Create template cookbook with patterns
- [ ] Update FINN documentation
- [ ] Create video tutorial for template creation
- [ ] Set up template contribution process
- [ ] Plan deprecation timeline for complex templates

### Deliverables
1. Complete template authoring documentation
2. Video tutorials
3. Contribution guidelines
4. Deprecation roadmap

## Implementation Details

### Simple Template Structure
```
finn/codegen/simple_templates/
├── [operation_name]/
│   ├── hls_main.cpp.j2        # Main HLS template
│   ├── hls_testbench.cpp.j2   # Testbench template
│   ├── rtl_wrapper.v.j2       # RTL wrapper
│   ├── variables.yaml         # Variable documentation
│   └── examples.md            # Usage examples
```

### Variable Naming Convention
- Use full words: `num_channels` not `nc`
- Include units: `clock_period_ns` not `clk`
- Be consistent with FINN attributes
- Document expected types and ranges

### Template Selection Logic
```python
def get_template_path(operation, backend_type):
    # 1. Check for simple template
    simple_path = f"simple_templates/{operation}/{backend_type}_main.j2"
    if exists(simple_path) and config.use_simple_templates:
        return simple_path
    
    # 2. Fall back to complex template
    return legacy_template_path(operation, backend_type)
```

## Success Metrics

### Quantitative
- Reduce average template complexity from 6+ files to 1 file
- Reduce average template size by 70%
- Improve template understanding time from hours to minutes
- Maintain 100% backward compatibility

### Qualitative  
- New contributors can modify templates without help
- Reduced questions about template system
- Increased community contributions
- Positive feedback from users

## Risk Mitigation

### Risk: Breaking Existing Workflows
**Mitigation**: 
- All changes are opt-in via configuration
- Extensive testing before enabling by default
- Keep complex templates indefinitely if needed

### Risk: Performance Regression
**Mitigation**:
- Benchmark template rendering performance
- Simple templates should be faster (less parsing)
- Cache compiled templates as before

### Risk: Feature Parity
**Mitigation**:
- Start with most common use cases
- Keep complex templates for advanced features
- Gradually add features to simple templates as needed

## Long-term Vision

### Year 1: Adoption
- 50% of operations using simple templates
- Active community contributions
- Template cookbook with 20+ patterns

### Year 2: Standard
- Simple templates become the default
- Complex templates marked as legacy
- Template marketplace for custom operations

### Year 3: Innovation
- Community-driven template improvements
- Domain-specific template sets
- Integration with other FPGA frameworks

## Conclusion

Simplifying FINN's template system is crucial for growing the open source community. By reducing complexity while maintaining power, we can enable more hardware engineers to contribute custom operations and accelerate FINN's adoption in the FPGA community.