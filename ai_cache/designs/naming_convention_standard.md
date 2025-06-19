# FINN Codegen Naming Convention Standard

## Overview

This document defines the standardized naming conventions for FINN codegen backends to ensure consistency across the codebase.

## Naming Standards

### 1. Backend Class Names

#### Legacy Backends
- **Pattern**: `{OperationName}_{backend}`
- **Examples**: 
  - `Thresholding_hls`
  - `MVAU_rtl`
  - `AddStreams_hls`

#### Clean Backends
- **Pattern**: `CG_{OperationName}_{backend}`
- **Examples**:
  - `CG_Thresholding_hls` (not CG_ThresholdingHLS)
  - `CG_MVAU_rtl`
  - `CG_AddStreams_hls`

### 2. File Names

- **Pattern**: `{operation_name}_{backend}.py`
- **Examples**:
  - `thresholding_hls.py`
  - `mvau_rtl.py`
  - `CG_thresholding_hls.py`

### 3. Backend Type Identifiers

- **Always lowercase**: `hls`, `rtl`
- **Never**: `HLS`, `RTL`, `Hls`, `Rtl`

### 4. Operation Names in Classes

- **Use PascalCase for operation part**: `Thresholding`, `MVAU`, `AddStreams`
- **Keep acronyms uppercase**: `MVAU` (Matrix-Vector Activation Unit)

## Examples of Correct Naming

### Legacy Backend
```python
# File: thresholding_hls.py
class Thresholding_hls(Thresholding, HLSBackend):
    pass
```

### Clean Backend
```python
# File: CG_thresholding_hls.py
class CG_Thresholding_hls(Thresholding, CG_HLSBackend):
    pass
```

## Common Mistakes to Avoid

1. ❌ `CG_ThresholdingHLS` → ✅ `CG_Thresholding_hls`
2. ❌ `MVAU_HLS` → ✅ `MVAU_hls`
3. ❌ `AddStreamsRTL` → ✅ `AddStreams_rtl`

## Migration Guidelines

When creating new backends or migrating existing ones:

1. Check if the class name follows the standard pattern
2. Ensure file names match the class name in snake_case
3. Update all imports and registrations
4. Run tests to verify no breaking changes

## Enforcement

- Pre-commit hooks will validate naming conventions
- CI/CD will fail on non-compliant names
- Code review checklist includes naming validation