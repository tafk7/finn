# FINN Codegen Folder Cleanup and Organization Plan

**Date**: June 18, 2025  
**Current State**: Development artifacts mixed with core implementation  
**Goal**: Clean separation of production code vs development/testing artifacts

---

## 📁 Current Folder Analysis

### Core Implementation Files (KEEP in /src/finn/codegen)
These are essential parts of the codegen infrastructure:

1. **Core Framework Files**
   - `__init__.py` - Package initialization
   - `codegen.py` - Base codegen infrastructure
   - `template_engine.py` - Jinja2 template engine
   - `file_manager.py` - File operations management
   - `library_resolver.py` - Library dependency resolution
   - `backend_registry.py` - Backend registration system
   - `config.py` - Configuration management

2. **Clean Backend Infrastructure**
   - `CG_backend_registration.py` - Clean backend registration

3. **Templates Directory**
   - `templates/` - All template files (critical for code generation)

### Development Artifacts (RELOCATE or REMOVE)

1. **Test Files (Move to /tests/finn/codegen/)**
   - `test_architecture_validation.py`
   - `test_node_factory.py` 
   - `test_real_finn_nodes.py`
   - `test_suite.py`
   - `test_thresholding.py`
   - `test_fixed_backends.py`
   - `test_real_backends_docker.py`
   - `test_real_finn_backends.py`
   - `test_real_validation.py`

2. **Validation/Debugging Tools (Move to /tools/finn/codegen/)**
   - `codegen_validator.py` - A/B testing framework
   - `debug_test_failure.py` - Debugging utility
   - `diagnostic_backend_checker.py` - Backend diagnostics
   - `inspect_generated_code.py` - Code inspection tool
   - `inspect_template_values.py` - Template debugging
   - `generate_actual_code.py` - Code generation testing
   - `run_validation.py` - Validation runner
   - `baseline_validation.py` - Baseline comparison tool

3. **Documentation (Move to /docs/finn/codegen/)**
   - `README.md` - Main documentation
   - `IMPLEMENTATION_SUMMARY.md` - Implementation details
   - `TESTING_GUIDE.md` - Testing documentation
   - `VALIDATION_RESULTS.md` - Validation results
   - `FINN_Template_Validation_Report.md` - Template validation report

4. **Scripts (Move to /scripts/finn/codegen/)**
   - `run_docker_tests.sh` - Docker test runner

5. **Generated Artifacts (REMOVE)**
   - `generated_clean_backend.cpp` - Generated test file
   - `generated_legacy_backend.cpp` - Generated test file
   - `test_output/` - Test output directory

6. **Development Support Files (EVALUATE)**
   - `backend_instance_manager.py` - Could be core or test infrastructure
   - `backend_registration.py` - Appears to be duplicate of backend_registry.py
   - `simple_file_manager.py` - Simplified version, possibly test code
   - `simple_library_resolver.py` - Simplified version, possibly test code
   - `template_validator.py` - Could be core validation or test tool

---

## 🎯 Recommended Actions

### 1. Create Directory Structure
```bash
# Create new directories for organization
mkdir -p tests/finn/codegen
mkdir -p tools/finn/codegen  
mkdir -p docs/finn/codegen
mkdir -p scripts/finn/codegen
```

### 2. Move Test Files
```bash
# Move all test files to proper test directory
mv src/finn/codegen/test_*.py tests/finn/codegen/
```

### 3. Move Development Tools
```bash
# Move validation and debugging tools
mv src/finn/codegen/codegen_validator.py tools/finn/codegen/
mv src/finn/codegen/debug_test_failure.py tools/finn/codegen/
mv src/finn/codegen/diagnostic_backend_checker.py tools/finn/codegen/
mv src/finn/codegen/inspect_*.py tools/finn/codegen/
mv src/finn/codegen/generate_actual_code.py tools/finn/codegen/
mv src/finn/codegen/run_validation.py tools/finn/codegen/
mv src/finn/codegen/baseline_validation.py tools/finn/codegen/
```

### 4. Move Documentation
```bash
# Move all documentation files
mv src/finn/codegen/*.md docs/finn/codegen/
# Keep README.md as a symlink for discoverability
ln -s ../../../docs/finn/codegen/README.md src/finn/codegen/README.md
```

### 5. Move Scripts
```bash
# Move shell scripts
mv src/finn/codegen/*.sh scripts/finn/codegen/
```

### 6. Clean Generated Artifacts
```bash
# Remove generated files and test outputs
rm src/finn/codegen/generated_*.cpp
rm -rf src/finn/codegen/test_output/
```

### 7. Evaluate Ambiguous Files
Need to determine if these are core or development:
- `backend_instance_manager.py` - If used only for testing → tools/
- `backend_registration.py` - If duplicate → remove
- `simple_file_manager.py` - If test version → tests/
- `simple_library_resolver.py` - If test version → tests/
- `template_validator.py` - If production validation → keep, else → tools/

---

## 📋 Final Structure

### Production Code (/src/finn/codegen/)
```
src/finn/codegen/
├── __init__.py
├── codegen.py
├── template_engine.py
├── file_manager.py
├── library_resolver.py
├── backend_registry.py
├── CG_backend_registration.py
├── config.py
├── README.md (symlink to docs)
└── templates/
    ├── base/
    ├── hls/
    └── rtl/
```

### Tests (/tests/finn/codegen/)
```
tests/finn/codegen/
├── test_architecture_validation.py
├── test_node_factory.py
├── test_real_finn_nodes.py
├── test_suite.py
├── test_thresholding.py
├── test_fixed_backends.py
├── test_real_backends_docker.py
├── test_real_finn_backends.py
└── test_real_validation.py
```

### Development Tools (/tools/finn/codegen/)
```
tools/finn/codegen/
├── codegen_validator.py
├── debug_test_failure.py
├── diagnostic_backend_checker.py
├── inspect_generated_code.py
├── inspect_template_values.py
├── generate_actual_code.py
├── run_validation.py
└── baseline_validation.py
```

### Documentation (/docs/finn/codegen/)
```
docs/finn/codegen/
├── README.md
├── IMPLEMENTATION_SUMMARY.md
├── TESTING_GUIDE.md
├── VALIDATION_RESULTS.md
└── FINN_Template_Validation_Report.md
```

---

## 🚀 Benefits of Cleanup

1. **Clear Separation**: Production code vs development artifacts
2. **Easier Navigation**: Developers can find what they need quickly
3. **Reduced Confusion**: No mixing of test code with production code
4. **Better Testing**: Tests in standard location for test runners
5. **Documentation Access**: All docs in one place
6. **Clean Deployment**: Only production code in src/

---

## ⚠️ Important Considerations

1. **Update Imports**: After moving files, update all import statements
2. **CI/CD Updates**: Update any CI/CD scripts that reference old paths
3. **Documentation Links**: Update any documentation with file paths
4. **Tool Scripts**: Update any scripts that use these tools
5. **Git History**: Consider using `git mv` to preserve file history

---

## 🎯 Priority Order

1. **High Priority**: Remove generated artifacts (`.cpp` files, `test_output/`)
2. **Medium Priority**: Move test files to `/tests/`
3. **Medium Priority**: Move documentation to `/docs/`
4. **Low Priority**: Move development tools to `/tools/`
5. **Low Priority**: Evaluate and categorize ambiguous files

This cleanup will make the codebase more maintainable and professional while preserving all valuable work.