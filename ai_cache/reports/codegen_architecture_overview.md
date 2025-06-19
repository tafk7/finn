# FINN Codegen System: Comprehensive Architecture Overview

## Executive Summary

The FINN codegen system is undergoing a major refactoring to modernize code generation while maintaining 100% backward compatibility. The new architecture replaces complex string-based template systems with a clean, Jinja2-based approach that achieves 5.7x faster code generation with 60% less memory usage.

## Current Implementation State

### Core Architecture Components

1. **Base Codegen Framework** (`codegen.py`)
   - Abstract base class providing simplified infrastructure
   - Template-based code generation with explicit template selection
   - Minimal complexity while maintaining essential performance through strategic caching
   - Key methods:
     - `get_template_name()`: Explicit template declaration
     - `get_template_values()`: Direct value generation
     - `generate_code()`: Simplified generation flow

2. **Template Engine** (`template_engine.py`)
   - Jinja2-based template processing
   - Single performance-critical cache (LRU cache for compiled templates)
   - Custom FINN-specific filters
   - Support for legacy string replacement (backward compatibility)
   - Template compilation caching (10-50ms operations)

3. **Backend Registry** (`backend_registry.py`)
   - Simple explicit backend registration without auto-discovery
   - O(1) backend lookup via dictionary
   - Separate HLS and RTL backend registries
   - Deterministic backend selection

4. **Backend Registration** (`backend_registration.py`)
   - Explicit registration of all available backends
   - Currently registers ~30 HLS backends and ~8 RTL backends
   - Fixed class names matching actual implementation files
   - Global registry singleton pattern

5. **Configuration System** (`config.py`)
   - Simple dataclass-based configuration
   - Environment variable overrides
   - Debug and logging settings
   - Template directory management

6. **File Management** (`simple_file_manager.py`)
   - Basic file operations without complex dependency tracking
   - Directory creation, file I/O, path resolution
   - Focused on reliability and simplicity

7. **Library Resolution** (`simple_library_resolver.py`)
   - Dynamic dependency resolution (if implemented)
   - Include file detection
   - Library path resolution

### Backend Implementation Pattern

The system follows a clean inheritance pattern:

```
Operation Base Class (e.g., Thresholding)
    ↓
Backend Implementation (e.g., CG_Thresholding_hls)
    ↓
Inherits from both Operation and CG_HLSBackend
```

Example clean backend structure (`CG_Thresholding_hls.py`):
- Explicit template declaration: `TEMPLATE_NAME = "thresholding/hls/docompute.cpp.j2"`
- Direct value generation methods
- No `code_gen_dict` or legacy string replacement
- Clean separation of concerns

### Template System

Templates use Jinja2 with custom FINN filters:

```
templates/
├── hls/          # HLS-specific templates
├── rtl/          # RTL-specific templates  
├── common/       # Shared templates
├── base/         # Base templates
├── components/   # Reusable components
└── thresholding/ # Operation-specific templates
    ├── hls/
    │   ├── docompute.cpp.j2
    │   ├── docompute_timeout.cpp.j2
    │   ├── ipgen.cpp.j2
    │   └── ipgen.tcl.j2
    └── rtl/
        └── wrapper.v.j2
```

Custom filters include:
- `format_define`: HLS #define formatting
- `format_port`: Port declarations
- `format_array`: Array declarations
- `cpp_type_name`: FINN datatype to C++ type conversion

## Key Design Principles

1. **Explicit Over Implicit**: Direct registration, no auto-discovery
2. **Strategic Minimalism**: Cache only expensive operations (template compilation)
3. **Performance Through Simplicity**: Fast lookups, minimal overhead
4. **Full Backward Compatibility**: Legacy code continues working
5. **Template Value Provider Pattern**: Operations explicitly provide values

## Performance Metrics

- **5.7x faster** code generation
- **60% reduction** in memory usage
- Template compilation caching for 10-50ms operations
- O(1) backend lookup performance

## Migration Path

The system supports both:
1. **Legacy backends**: Using existing string-based code generation
2. **Clean backends**: Using new Jinja2 template system

Clean backends are identified by:
- Inheriting from `CG_HLSBackend` or `CG_RTLBackend`
- Having explicit `TEMPLATE_NAME` declaration
- Implementing template value generation methods
- No `code_gen_dict` usage

## Current Status

### Implemented
- ✅ Core framework components
- ✅ Template engine with caching
- ✅ Backend registry system
- ✅ Configuration management
- ✅ File management utilities
- ✅ Example clean backend (CG_Thresholding_hls)
- ✅ Template structure and organization

### In Progress
- 🔄 Migration of all operations to clean backends
- 🔄 Full template library creation
- 🔄 Documentation completion

### Testing
- 37 comprehensive tests (100% passing according to README)
- Docker integration validated
- Real FINN operations tested
- Performance benchmarks completed

## Advantages of New Architecture

1. **Maintainability**: Clear separation of concerns, operation-specific templates
2. **Performance**: Strategic caching, faster code generation
3. **Flexibility**: Jinja2 templates with conditionals, loops, inheritance
4. **Debugging**: Rich error messages, template validation
5. **Extensibility**: Easy to add new operations and backends

## Recommendations

1. Continue migrating operations to clean backends incrementally
2. Maintain backward compatibility throughout migration
3. Expand template library with reusable components
4. Add comprehensive documentation for template developers
5. Consider template inheritance for common patterns

The new codegen architecture represents a significant improvement in FINN's code generation capabilities while maintaining full compatibility with existing operations.