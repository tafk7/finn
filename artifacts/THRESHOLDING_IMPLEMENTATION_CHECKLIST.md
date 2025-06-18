# Thresholding Code Generation Refactor - Implementation Checklist

## 📋 **Phase 1: Template Conversion (Days 1-2)**

### **1.1 Setup Template Directory Structure**
- [x] Create `src/finn/codegen/templates/thresholding/` directory
- [x] Create `src/finn/codegen/templates/thresholding/hls/` subdirectory
- [x] Create `src/finn/codegen/templates/thresholding/rtl/` subdirectory

### **1.2 Convert RTL Template**
- [x] Read `finn-rtllib/thresholding/hdl/thresholding_template_wrapper.v`
- [x] Convert all `$VAR$` placeholders to `{{ VAR }}` syntax
- [x] Create `src/finn/codegen/templates/thresholding/rtl/wrapper.v.j2`
- [x] Validate Jinja2 syntax

### **1.3 Convert HLS Templates**
- [x] Read templates from `src/finn/custom_op/fpgadataflow/templates.py`
- [x] Convert `docompute_template` → `src/finn/codegen/templates/thresholding/hls/docompute.cpp.j2`
- [x] Convert `docompute_template_timeout` → `src/finn/codegen/templates/thresholding/hls/docompute_timeout.cpp.j2`
- [x] Convert `ipgen_template` → `src/finn/codegen/templates/thresholding/hls/ipgen.cpp.j2`
- [x] Convert `ipgentcl_template` → `src/finn/codegen/templates/thresholding/hls/ipgen.tcl.j2`
- [x] Validate all Jinja2 syntax

### **1.4 Template Validation**
- [x] Create template syntax validation test
- [x] Test all 5 templates load without errors
- [x] Fix any syntax issues

---

## 📋 **Phase 2: HLS Backend Update (Days 3-5)**

### **2.1 Update ThresholdingHLS Initialization**
- [x] Remove `code_gen_dict` initialization from `__init__`
- [x] Add template engine initialization
- [x] Add context storage for fpgapart/clk

### **2.2 Implement Template Value Extraction**
- [x] Implement `get_template_values(template_name)` main dispatcher
- [x] Implement `_get_docompute_values(template_name)` for cppsim templates
- [x] Implement `_get_ipgen_cpp_values()` for IP generation C++
- [x] Implement `_get_ipgen_tcl_values()` for IP generation TCL

### **2.3 Implement Direct Value Generation Methods**
- [x] Implement `_generate_globals()` - extract from existing global_includes()
- [x] Implement `_generate_defines(mode)` - extract from existing defines()
- [x] Implement `_generate_pragmas()` - extract from existing pragmas()
- [x] Implement `_generate_stream_declarations()` - extract from existing strm_decl()
- [x] Implement `_generate_read_npy_data()` - extract from existing read_npy_data()
- [x] Implement `_generate_docompute()` - extract from existing docompute()
- [x] Implement `_generate_data_out_stream()` - extract from existing dataoutstrm()
- [x] Implement `_generate_save_as_npy()` - extract from existing save_as_npy()
- [x] Implement `_generate_blackbox_function()` - extract from existing blackboxfunction()

### **2.4 Implement Timeout-Specific Methods**
- [x] Implement `_generate_timeout_value()` - extract from existing timeout_value()
- [x] Implement `_generate_timeout_condition()` - extract from existing timeout_condition()
- [x] Implement `_generate_timeout_read_stream()` - extract from existing timeout_read_stream()

### **2.5 Update Code Generation Methods**
- [x] Rewrite `code_generation_cppsim()` to use Jinja2 templates
- [x] Rewrite `code_generation_ipgen()` to use Jinja2 templates for both C++ and TCL
- [x] Remove all `code_gen_dict` usage
- [x] Test HLS code generation produces output

---

## 📋 **Phase 3: RTL Backend Update (Days 6-8)**

### **3.1 Update Thresholding_rtl Template Integration**
- [x] Add template engine initialization to `__init__`
- [x] Implement `get_template_values(template_name)` dispatcher
- [x] Implement `_get_wrapper_values()` for RTL wrapper template

### **3.2 Extract RTL Value Calculation Logic**
- [x] Extract parameter calculations from `prepare_codegen_rtl_values()`
- [x] Implement bias calculation logic
- [x] Implement O_BITS calculation logic
- [x] Implement signed/FPARG logic
- [x] Implement all template parameter extraction

### **3.3 Extract Threshold File Generation**
- [x] Implement `_generate_threshold_files(model)` method
- [x] Extract threshold file generation logic from `prepare_codegen_rtl_values()`
- [x] Preserve all existing threshold processing logic

### **3.4 Update RTL Generation Method**
- [x] Rewrite `generate_hdl()` to use Jinja2 template instead of string replacement
- [x] Implement `_copy_rtl_library_files()` method
- [x] Remove all string replacement (`template.replace()`) usage
- [x] Test RTL code generation produces output

---

## 📋 **Phase 4: Testing & Validation (Days 9-10)**

### **4.1 Template Syntax Testing**
- [ ] Create `test_thresholding_template_syntax.py`
- [ ] Test all 5 templates load without Jinja2 syntax errors
- [ ] Validate template variables are correctly defined

### **4.2 Output Equivalence Testing**
- [ ] Create test that compares legacy vs new HLS output
- [ ] Create test that compares legacy vs new RTL output  
- [ ] Implement character-by-character diff comparison
- [ ] Create test with real Thresholding node parameters

### **4.3 Integration Testing**
- [ ] Test HLS cppsim code generation end-to-end
- [ ] Test HLS ipgen code generation end-to-end
- [ ] Test RTL wrapper generation end-to-end
- [ ] Test with multiple Thresholding configurations

### **4.4 Regression Testing**
- [ ] Run existing Thresholding tests to ensure no breakage
- [ ] Test with real FINN models that use Thresholding
- [ ] Validate performance (no significant slowdown)

---

## 🎯 **Success Criteria Validation**

- [ ] **Zero code_gen_dict usage** - grep for code_gen_dict in thresholding files
- [ ] **Zero string replacement** - grep for template.replace in RTL backend
- [ ] **Identical output** - diff tests pass for legacy vs new
- [ ] **All tests pass** - existing Thresholding tests still work
- [ ] **Templates validate** - Jinja2 syntax checker passes

---

## 📊 **Progress Tracking**

| Phase | Tasks | Completed | Status |
|-------|-------|-----------|--------|
| 1. Template Conversion | 12 | 0 | ⏳ Not Started |
| 2. HLS Backend Update | 16 | 0 | ⏳ Not Started |  
| 3. RTL Backend Update | 12 | 0 | ⏳ Not Started |
| 4. Testing & Validation | 12 | 0 | ⏳ Not Started |
| **TOTAL** | **52** | **0** | **0%** |

---

## 🚀 **Next Action**
Begin with Phase 1.1: Setup Template Directory Structure