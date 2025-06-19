# Fix Clean Backend Inheritance Issue

## Problem Summary

The clean backend (CG_ThresholdingHLS) is generating placeholder values because:

1. **Wrong Backend Retrieved**: `getCustomOp(node)` returns `Thresholding_hls` (legacy) not `CG_ThresholdingHLS` (clean)
2. **Legacy Methods Called**: The legacy HLSBackend.get_template_values() overrides clean values with placeholder strings from _get_*_from_code_gen_dict() methods
3. **Inheritance Conflict**: Both backends implement get_template_values() but with different behavior

## Root Cause

The node type is "Thresholding_hls" which is registered to the legacy backend class. The clean backend CG_ThresholdingHLS is not being used at all because it's not registered.

## Solution Options

### Option 1: Force Clean Backend in Test (Quick Fix)
Modify the test to explicitly create and use CG_ThresholdingHLS instance:
```python
# Don't use getCustomOp, create clean backend directly
from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_ThresholdingHLS
inst = CG_ThresholdingHLS(node)
```

### Option 2: Register Clean Backend (Proper Fix)
Register CG_ThresholdingHLS as an alternative backend for Thresholding nodes.

### Option 3: Fix Template Value Generation
Ensure CG_ThresholdingHLS doesn't inherit legacy template value methods.

## Recommended Approach

Start with Option 1 to verify the clean backend works correctly, then implement Option 2 for proper integration.