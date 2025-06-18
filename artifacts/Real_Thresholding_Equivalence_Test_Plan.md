# Real Thresholding Equivalence Test Plan
## Preventing Mock Objects and Ensuring Functional Validation

**Document Purpose**: Define explicit, mock-proof testing requirements for FINN Thresholding Jinja2 refactor validation  
**Target**: Implementers cannot use shallow validations or fake success indicators  
**Success Criteria**: Byte-for-byte output equivalence + functional correctness + performance parity  

---

## Phase 1: Legacy Baseline Capture
**Objective**: Preserve exact output of legacy string replacement system before modification

### Deliverable 1.1: Legacy Output Archive
**Requirements**:
- **5 distinct test configurations** (different PE, NumChannels, data types)
- **All generation methods** captured for each configuration:
  - HLS `docompute()` method output
  - HLS `defines()` method output  
  - HLS `global_includes()` method output
  - HLS `blackboxfunction()` method output
  - RTL `generate_hdl()` method output
- **Exact file outputs** saved with checksums
- **Complete parameter sets** documented for reproducibility

**Implementation Steps**:
```python
def capture_legacy_baseline():
    """MANDATORY: Run this BEFORE any refactor changes."""
    test_configs = [
        {'PE': 1, 'NumChannels': 4, 'inputDataType': 'INT8', 'outputDataType': 'INT4'},
        {'PE': 2, 'NumChannels': 8, 'inputDataType': 'INT8', 'outputDataType': 'INT4'}, 
        {'PE': 4, 'NumChannels': 16, 'inputDataType': 'INT16', 'outputDataType': 'INT8'},
        {'PE': 1, 'NumChannels': 1, 'inputDataType': 'INT4', 'outputDataType': 'INT2'},
        {'PE': 8, 'NumChannels': 32, 'inputDataType': 'INT8', 'outputDataType': 'INT4'}
    ]
    
    baseline_archive = {}
    for i, config in enumerate(test_configs):
        # Create legacy node (BEFORE refactor)
        legacy_node = create_legacy_thresholding_node(config)
        
        # Capture ALL method outputs
        baseline_archive[f'config_{i}'] = {
            'parameters': config,
            'hls_docompute': legacy_node.docompute(),
            'hls_defines': legacy_node.defines(),
            'hls_global_includes': legacy_node.global_includes(),
            'hls_blackboxfunction': legacy_node.blackboxfunction(),
            'rtl_generate_hdl': legacy_node.generate_hdl(model, "xc7z020clg400-1", 10),
            'checksums': calculate_all_checksums(...)
        }
    
    # Save with timestamp and version info
    save_baseline_archive('baseline_legacy_outputs.json', baseline_archive)
    return baseline_archive
```

**Acceptance Criteria**:
- [ ] 5 complete baseline configurations captured
- [ ] All method outputs saved with SHA256 checksums
- [ ] Baseline archive includes FINN version, timestamp, dependencies
- [ ] Archive can be loaded and verified independently

### Deliverable 1.2: Legacy Compilation Validation
**Requirements**:
- **All generated HLS C++ code compiles** with Vivado HLS
- **All generated RTL code compiles** with Vivado
- **Compilation logs saved** for comparison

**Implementation Steps**:
```bash
# For each baseline configuration
compile_legacy_outputs() {
    for config in config_0 config_1 config_2 config_3 config_4; do
        # HLS Compilation
        vivado_hls compile_${config}_hls.tcl 2>&1 | tee ${config}_hls_compile.log
        
        # RTL Compilation  
        vivado -mode batch -source compile_${config}_rtl.tcl 2>&1 | tee ${config}_rtl_compile.log
        
        # Verify no errors
        grep -i "error" ${config}_*_compile.log && exit 1
    done
}
```

**Acceptance Criteria**:
- [ ] All 5 configurations compile without errors
- [ ] Compilation logs show successful synthesis
- [ ] Generated IP cores pass basic validation

---

## Phase 2: Template System Implementation Validation
**Objective**: Prove template system produces identical output to legacy system

### Deliverable 2.1: Byte-for-Byte Output Comparison
**Requirements**:
- **Exact character-by-character matching** between legacy and template outputs
- **No normalization allowed** - outputs must be identical
- **All 5 test configurations** must pass comparison

**Implementation Steps**:
```python
def test_exact_output_equivalence():
    """NO MOCKING ALLOWED - Direct byte comparison."""
    baseline_archive = load_baseline_archive('baseline_legacy_outputs.json')
    
    failures = []
    for config_name, baseline in baseline_archive.items():
        # Create template-based node with identical parameters
        template_node = create_template_thresholding_node(baseline['parameters'])
        
        # Generate outputs with new template system
        template_outputs = {
            'hls_docompute': template_node.docompute(),
            'hls_defines': template_node.defines(),
            'hls_global_includes': template_node.global_includes(),
            'hls_blackboxfunction': template_node.blackboxfunction(),
            'rtl_generate_hdl': template_node.generate_hdl(model, "xc7z020clg400-1", 10)
        }
        
        # CRITICAL: Byte-for-byte comparison (no normalization)
        for method_name, baseline_output in baseline.items():
            if method_name in template_outputs:
                template_output = template_outputs[method_name]
                
                if baseline_output != template_output:
                    failures.append({
                        'config': config_name,
                        'method': method_name,
                        'baseline_checksum': hashlib.sha256(baseline_output.encode()).hexdigest(),
                        'template_checksum': hashlib.sha256(template_output.encode()).hexdigest(),
                        'diff': unified_diff(baseline_output, template_output)
                    })
    
    # ZERO TOLERANCE for differences
    assert len(failures) == 0, f"Output differences detected: {failures}"
```

**Anti-Mock Protections**:
- **No string normalization functions allowed**
- **No "close enough" comparisons**
- **Complete diff output required for any failures**
- **Checksums must match exactly**

**Acceptance Criteria**:
- [ ] All 5 configurations produce identical output
- [ ] Zero character differences between legacy and template systems
- [ ] All checksums match exactly
- [ ] No template rendering errors or missing variables

### Deliverable 2.2: Template Rendering Validation
**Requirements**:
- **All templates render without Jinja2 errors**
- **No unreplaced placeholders** in final output
- **Template variables validated** for correctness

**Implementation Steps**:
```python
def test_template_rendering_correctness():
    """Verify templates actually work, not just exist."""
    
    # Test ALL template files with real data
    template_files = [
        'thresholding/hls/docompute.cpp.j2',
        'thresholding/hls/docompute_timeout.cpp.j2',
        'thresholding/hls/ipgen.cpp.j2',
        'thresholding/hls/ipgen.tcl.j2',
        'thresholding/rtl/wrapper.v.j2'
    ]
    
    for template_path in template_files:
        for config in test_configurations:
            node = create_template_node(config)
            
            # Get template values
            values = node.get_template_values(template_path)
            
            # Attempt rendering
            try:
                rendered = render_template(template_path, **values)
            except Exception as e:
                raise AssertionError(f"Template {template_path} failed to render: {e}")
            
            # Validate no unreplaced placeholders
            jinja_patterns = [r'\{\{.*?\}\}', r'\{%.*?%\}', r'\{#.*?#\}']
            for pattern in jinja_patterns:
                matches = re.findall(pattern, rendered)
                assert len(matches) == 0, f"Unreplaced Jinja2 in {template_path}: {matches}"
```

**Acceptance Criteria**:
- [ ] All templates render successfully for all 5 configurations
- [ ] Zero unreplaced Jinja2 placeholders in rendered output
- [ ] All template variables have valid, non-None values
- [ ] Template rendering produces syntactically valid output

---

## Phase 3: Compilation Equivalence Validation
**Objective**: Prove generated code compiles identically to legacy system

### Deliverable 3.1: HLS Compilation Validation
**Requirements**:
- **Template-generated HLS code compiles** with Vivado HLS
- **Identical compilation results** to legacy baseline
- **Resource utilization comparison**

**Implementation Steps**:
```bash
compile_template_hls_validation() {
    for config in config_0 config_1 config_2 config_3 config_4; do
        # Generate code with template system
        python generate_template_hls.py --config ${config}
        
        # Compile with Vivado HLS
        vivado_hls compile_template_${config}.tcl 2>&1 | tee template_${config}_hls.log
        
        # Compare compilation results with baseline
        diff baseline_${config}_hls.log template_${config}_hls.log > ${config}_hls_diff.log
        
        # Verify identical results (allowing timestamp differences only)
        python validate_compilation_equivalence.py \
            --baseline baseline_${config}_hls.log \
            --template template_${config}_hls.log \
            --config ${config}
    done
}
```

**Acceptance Criteria**:
- [ ] All template-generated HLS code compiles successfully
- [ ] Resource utilization matches legacy within 5%
- [ ] No compilation warnings that weren't in legacy version
- [ ] Synthesis results are functionally equivalent

### Deliverable 3.2: RTL Compilation Validation
**Requirements**:
- **Template-generated RTL compiles** with Vivado
- **Timing closure equivalent** to legacy
- **Resource usage equivalent** to legacy

**Implementation Steps**:
```bash
compile_template_rtl_validation() {
    for config in config_0 config_1 config_2 config_3 config_4; do
        # Generate RTL with template system
        python generate_template_rtl.py --config ${config}
        
        # Compile with Vivado
        vivado -mode batch -source compile_template_${config}.tcl 2>&1 | tee template_${config}_rtl.log
        
        # Extract resource usage and timing
        python extract_vivado_metrics.py \
            --log template_${config}_rtl.log \
            --output template_${config}_metrics.json
            
        # Compare with baseline metrics
        python compare_vivado_metrics.py \
            --baseline baseline_${config}_metrics.json \
            --template template_${config}_metrics.json \
            --tolerance 0.05  # 5% tolerance
    done
}
```

**Acceptance Criteria**:
- [ ] All template-generated RTL compiles successfully  
- [ ] LUT/FF/BRAM usage within 5% of legacy
- [ ] Timing closure equivalent to legacy
- [ ] No new critical warnings

---

## Phase 4: Runtime Functional Validation
**Objective**: Prove generated hardware produces correct computational results

### Deliverable 4.1: RTL Simulation Validation
**Requirements**:
- **Identical simulation results** between legacy and template systems
- **Multiple test vectors** for thorough validation
- **Bit-accurate comparison** of outputs

**Implementation Steps**:
```python
def test_rtl_functional_equivalence():
    """Test actual computation correctness, not just compilation."""
    
    test_vectors = generate_comprehensive_test_vectors()  # 100+ vectors
    
    for config in test_configurations:
        # Generate RTL with both systems
        legacy_rtl = generate_legacy_rtl(config)
        template_rtl = generate_template_rtl(config)
        
        # Run simulation with test vectors
        legacy_results = run_rtl_simulation(legacy_rtl, test_vectors)
        template_results = run_rtl_simulation(template_rtl, test_vectors)
        
        # Bit-accurate comparison
        for i, (legacy_out, template_out) in enumerate(zip(legacy_results, template_results)):
            assert legacy_out == template_out, \
                f"Config {config}, Vector {i}: {legacy_out} != {template_out}"
```

**Anti-Mock Protections**:
- **Actual RTL simulation required** (not software models)
- **100+ test vectors** to prevent cherry-picking
- **Bit-accurate comparison** (no tolerance)
- **All configurations must pass** (no exceptions)

**Acceptance Criteria**:
- [ ] 100+ test vectors pass for all 5 configurations
- [ ] Bit-accurate output matching between legacy and template RTL
- [ ] Simulation completes without errors or warnings
- [ ] All edge cases (min/max values) handled correctly

### Deliverable 4.2: HLS C-Simulation Validation
**Requirements**:
- **HLS C-simulation produces identical results**
- **Co-simulation validation** with RTL
- **Performance metrics comparison**

**Implementation Steps**:
```bash
validate_hls_csim() {
    for config in config_0 config_1 config_2 config_3 config_4; do
        # Run C-simulation for both legacy and template
        run_hls_csim baseline_${config} test_vectors.dat > baseline_${config}_csim.out
        run_hls_csim template_${config} test_vectors.dat > template_${config}_csim.out
        
        # Compare outputs bit-for-bit
        diff baseline_${config}_csim.out template_${config}_csim.out > ${config}_csim_diff.log
        
        # Verify no differences
        test ! -s ${config}_csim_diff.log || (echo "C-sim differences in ${config}" && exit 1)
    done
}
```

**Acceptance Criteria**:
- [ ] HLS C-simulation outputs match exactly
- [ ] Co-simulation passes for all configurations
- [ ] Performance (cycles/latency) equivalent to legacy
- [ ] All numerical edge cases validated

---

## Phase 5: Performance & Integration Validation
**Objective**: Prove system performs equivalently in real FINN workflows

### Deliverable 5.1: Performance Benchmarking
**Requirements**:
- **Template rendering performance** measured
- **Memory usage comparison**
- **End-to-end workflow timing**

**Implementation Steps**:
```python
def benchmark_performance():
    """Measure actual performance impact, not theoretical."""
    
    # Template rendering performance
    for config in test_configurations:
        legacy_time = time_legacy_generation(config, iterations=100)
        template_time = time_template_generation(config, iterations=100)
        
        performance_ratio = template_time / legacy_time
        assert performance_ratio < 2.0, f"Template system >2x slower: {performance_ratio}"
        
    # Memory usage comparison
    legacy_memory = measure_memory_usage(legacy_generation_workflow)
    template_memory = measure_memory_usage(template_generation_workflow)
    
    memory_ratio = template_memory / legacy_memory
    assert memory_ratio < 1.5, f"Template system uses >50% more memory: {memory_ratio}"
```

**Acceptance Criteria**:
- [ ] Template system <2x slower than legacy
- [ ] Memory usage <50% higher than legacy
- [ ] No memory leaks in template rendering
- [ ] Performance scales linearly with problem size

### Deliverable 5.2: Integration Testing
**Requirements**:
- **Real FINN model compilation** end-to-end
- **Integration with FINN transformations**
- **Multi-operation model validation**

**Implementation Steps**:
```python
def test_finn_integration():
    """Test in real FINN workflows, not isolated units."""
    
    # Create multi-layer model with Thresholding operations
    model = create_realistic_finn_model_with_thresholding()
    
    # Run complete FINN build flow
    build_steps = [
        StreamliningTransformation(),
        ConvertToHW(),
        FoldConstants(),
        SpecializeLayers(),
        # ... other transforms
        MakeZynqProject()
    ]
    
    for step in build_steps:
        model = step.apply(model)
        validate_model_integrity(model)
    
    # Verify final bitstream generation
    assert model.get_generated_bitstream() is not None
    assert model.passes_hardware_validation()
```

**Acceptance Criteria**:
- [ ] Complete FINN build flow succeeds
- [ ] Generated bitstream runs on actual hardware
- [ ] Integration with all FINN transformations works
- [ ] Multi-operation models compile successfully

---

## Phase 6: Regression & Edge Case Validation
**Objective**: Ensure robustness across all parameter combinations

### Deliverable 6.1: Parameter Space Coverage
**Requirements**:
- **All valid parameter combinations** tested
- **Boundary condition validation**
- **Error handling verification**

**Implementation Steps**:
```python
def test_parameter_space_coverage():
    """Test ALL valid combinations, not just happy path."""
    
    # Generate comprehensive parameter matrix
    pe_values = [1, 2, 4, 8, 16]
    channel_values = [1, 4, 8, 16, 32, 64]
    input_types = ['INT4', 'INT8', 'INT16']
    output_types = ['INT2', 'INT4', 'INT8']
    
    total_configs = len(pe_values) * len(channel_values) * len(input_types) * len(output_types)
    
    for pe in pe_values:
        for channels in channel_values:
            if channels % pe != 0:  # Skip invalid combinations
                continue
            for input_type in input_types:
                for output_type in output_types:
                    config = {
                        'PE': pe,
                        'NumChannels': channels,
                        'inputDataType': input_type,
                        'outputDataType': output_type
                    }
                    
                    # Test template generation
                    node = create_template_node(config)
                    verify_template_generation_success(node, config)
```

**Acceptance Criteria**:
- [ ] All valid parameter combinations work
- [ ] Boundary conditions (PE=1, channels=1) handled
- [ ] Invalid combinations properly rejected
- [ ] Error messages are clear and actionable

### Deliverable 6.2: Error Condition Testing
**Requirements**:
- **Graceful handling of missing templates**
- **Clear error messages for invalid parameters**
- **Fallback behavior documentation**

**Implementation Steps**:
```python
def test_error_conditions():
    """Test system behavior under error conditions."""
    
    # Test missing template files
    with temporarily_remove_template('docompute.cpp.j2'):
        with pytest.raises(TemplateNotFoundError, match="docompute.cpp.j2"):
            node.get_template_values('thresholding/hls/docompute.cpp.j2')
    
    # Test missing template variables
    with mock_incomplete_node_attributes():
        with pytest.raises(TemplateVariableError, match="Missing required"):
            node.get_template_values('thresholding/hls/docompute.cpp.j2')
    
    # Test malformed templates
    with temporarily_corrupt_template('docompute.cpp.j2'):
        with pytest.raises(TemplateSyntaxError):
            render_template('docompute.cpp.j2', **valid_values)
```

**Acceptance Criteria**:
- [ ] Clear error messages for all failure modes
- [ ] No silent failures or corrupted output
- [ ] Graceful degradation where possible
- [ ] Error handling documented for users

---

## Final Acceptance Criteria

### Must-Pass Requirements (No Exceptions)
- [ ] **100% output equivalence** - All 5 configurations produce identical output
- [ ] **100% compilation success** - All generated code compiles without errors  
- [ ] **100% functional correctness** - All test vectors produce correct results
- [ ] **Performance within bounds** - <2x slower, <50% more memory
- [ ] **Zero regressions** - No functionality lost from legacy system

### Documentation Requirements
- [ ] **Migration guide** for users upgrading from legacy system
- [ ] **Template customization guide** for advanced users
- [ ] **Performance characteristics** documented
- [ ] **Error handling guide** with common solutions

### Maintainability Requirements
- [ ] **Template versioning strategy** implemented
- [ ] **Backwards compatibility plan** for future changes
- [ ] **Testing framework** reusable for future template changes
- [ ] **Code coverage >95%** for all template-related code

---

## Mock Prevention Measures

### Explicit Anti-Mock Requirements
1. **No string normalization** in comparisons
2. **No "close enough" tolerances** in output validation
3. **No test isolation** - must work in full FINN workflows
4. **No cherry-picked test cases** - comprehensive parameter coverage required
5. **No simulation shortcuts** - actual RTL simulation required

### Validation Scripts Must Include
1. **Checksum verification** of all outputs
2. **Diff generation** for any failures
3. **Compilation verification** with real tools
4. **Hardware simulation** with real test vectors
5. **Performance measurement** with actual timing

This test plan ensures that implementers cannot fake success through mocks or shallow validations. Every requirement has explicit acceptance criteria and anti-mock protections to guarantee functional equivalence.