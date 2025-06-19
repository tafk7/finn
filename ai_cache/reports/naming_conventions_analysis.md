# FINN Codegen Naming Conventions Analysis

## Executive Summary

The FINN codebase exhibits several inconsistent naming patterns between legacy and clean backends. This report identifies the patterns found and highlights inconsistencies that need standardization.

## Naming Patterns Identified

### 1. Legacy Backend Naming Pattern

The legacy backends follow a consistent pattern:
- **Format**: `{OperationName}_{backend}`
- **Examples**:
  - `Thresholding_hls`
  - `Thresholding_rtl`
  - `MVAU_hls`
  - `MVAU_rtl`
  - `StreamingDataWidthConverter_hls`
  - `StreamingDataWidthConverter_rtl`

### 2. Clean Backend Naming Pattern

The clean backends show **inconsistency**:
- **HLS Clean Backends**:
  - `CG_ThresholdingHLS` (inconsistent - uses "HLS" suffix instead of "_hls")
  - `CG_MVAU_hls` (consistent with legacy suffix pattern)
  
- **RTL Clean Backends**:
  - `CG_Thresholding_rtl` (consistent with legacy suffix pattern)
  - `CG_MVAU_rtl` (consistent with legacy suffix pattern)

### 3. File Naming Conventions

File naming is consistent across both legacy and clean implementations:
- **Pattern**: `{operation_name}_{backend}.py` (lowercase with underscores)
- **Examples**:
  - Legacy: `thresholding_hls.py`, `thresholding_rtl.py`
  - Clean: `CG_thresholding_hls.py`, `CG_thresholding_rtl.py`

### 4. Special Cases and Variations

#### a. Alternative Class Names
Some operations have alternative naming:
- `MVAU_hls` (MatrixVectorActivation)
- `VVAU_hls` (VectorVectorActivation)
- `DynMVU_rtl` (Dynamic MVU implementation)

#### b. Multiple Classes in Single File
`elementwise_binary_hls.py` contains multiple specific operation classes:
- `ElementwiseAdd_hls`
- `ElementwiseSub_hls`
- `ElementwiseMul_hls`
- etc.

#### c. Duplicate Implementation
There appears to be a duplicate MVAU implementation:
- `matrixvectoractivation_hls.py` contains `MVAU_hls`
- `mvau_hls.py` contains `MVAU_HLS` (note the uppercase HLS)

## Key Inconsistencies Found

### 1. Clean Backend Suffix Inconsistency
- **Problem**: `CG_ThresholdingHLS` uses "HLS" while all others use "_hls"
- **Expected**: `CG_Thresholding_hls` for consistency

### 2. Case Inconsistency
- **Problem**: `MVAU_HLS` in `mvau_hls.py` uses uppercase "HLS"
- **Standard**: All other classes use lowercase "_hls"

### 3. Naming Convention Documentation
- No clear documentation found on the "CG_" prefix meaning or when to use it
- Inconsistent application of the clean backend pattern

## Backend Registration Patterns

The `backend_registration.py` file shows:
- Legacy backends are registered by their base operation name
- Multiple aliases are supported (e.g., "MatrixVectorActivation" and "MVAU")
- Clean backends are not explicitly registered in the current registration file

## Recommendations

### 1. Standardize Clean Backend Naming
- Use consistent suffix pattern: `CG_{OperationName}_{backend}`
- Fix: `CG_ThresholdingHLS` → `CG_Thresholding_hls`

### 2. Resolve Duplicate Implementations
- Investigate `MVAU_hls` vs `MVAU_HLS` duplication
- Consolidate to single implementation

### 3. Document Naming Conventions
- Define what "CG_" prefix represents
- Document when to use clean vs legacy backends
- Create naming convention guide for new implementations

### 4. Update Registration System
- Ensure clean backends are properly registered
- Consider separate registration for clean backends if they follow different patterns

### 5. File Organization
- Consider separating clean and legacy implementations into different directories
- e.g., `hls/legacy/` and `hls/clean/` or similar structure

## Summary

The main inconsistency is in the clean backend naming, specifically `CG_ThresholdingHLS` which deviates from the otherwise consistent `_{backend}` suffix pattern. All other naming follows predictable patterns, though the codebase would benefit from clearer documentation and organization of the dual backend system.