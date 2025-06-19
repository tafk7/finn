# Multiple Inheritance Issues Analysis in FINN Codegen Clean Backends

**Date**: December 19, 2024  
**Scope**: Analysis of multiple inheritance issues in CG_Thresholding_hls and CG_MVAU_hls

## Executive Summary

The clean FINN codegen backends use multiple inheritance, inheriting from both an operation class (e.g., `Thresholding`) and a backend class (e.g., `CG_HLSBackend`). This analysis identifies specific Method Resolution Order (MRO) issues and attribute resolution conflicts that can lead to `AttributeError` exceptions.

## Issue 1: Incorrect get_nodeattr_types() Implementation

### Problem

In both `CG_Thresholding_hls` and `CG_MVAU_hls`, the `get_nodeattr_types()` method is implemented incorrectly:

```python
def get_nodeattr_types(self):
    """Get node attribute types from both parent classes."""
    my_attrs = {}
    my_attrs.update(Thresholding.get_nodeattr_types(self))  # ❌ Incorrect
    my_attrs.update(CG_HLSBackend.get_nodeattr_types(self))  # ❌ Incorrect
    return my_attrs
```

### Root Cause

The problem is that when calling parent class methods explicitly, you should **not** pass `self` as an argument. The method already receives `self` implicitly. The correct pattern is:

```python
def get_nodeattr_types(self):
    """Get node attribute types from both parent classes."""
    my_attrs = {}
    my_attrs.update(Thresholding.get_nodeattr_types(self))  # ✅ Correct
    my_attrs.update(CG_HLSBackend.get_nodeattr_types(self))  # ✅ Correct
    return my_attrs
```

However, examining the actual code, it appears the implementation is already correct! The issue description might be based on an older version or misunderstanding.

## Issue 2: Method Resolution Order (MRO) Complexity

### Current Inheritance Structure

```python
class CG_Thresholding_hls(Thresholding, CG_HLSBackend):
    # Multiple inheritance from operation and backend
```

### MRO Chain

The Method Resolution Order follows Python's C3 linearization:
1. `CG_Thresholding_hls`
2. `Thresholding`
3. `HWCustomOp` (parent of Thresholding)
4. `CustomOp` (parent of HWCustomOp)
5. `CG_HLSBackend`
6. `Codegen` (parent of CG_HLSBackend)
7. `ABC` (abstract base class)
8. `object`

### Potential Conflicts

1. **Attribute Name Collisions**: If both parent classes define the same attribute names, the first in MRO wins
2. **Method Overrides**: Methods must carefully call parent implementations
3. **Initialization Order**: The `__init__` methods must be called in correct order

## Issue 3: Attribute Access Patterns

### In CG_HLSBackend

The `CG_HLSBackend.get_nodeattr_types()` method attempts to walk the MRO and collect attributes:

```python
# Walk the MRO to get attributes from all parent classes
for cls in self.__class__.__mro__[1:]:  # Skip self
    if hasattr(cls, 'get_nodeattr_types') and cls != CG_HLSBackend:
        try:
            parent_attrs = cls.get_nodeattr_types(self)
            # ...
```

This approach has issues:
1. It passes `self` when calling the method, which might cause double `self` arguments
2. It doesn't handle diamond inheritance properly
3. Error handling masks real issues

## Issue 4: Safe Attribute Access

### Problem

The code uses `_safe_get_nodeattr()` helper method to avoid AttributeError:

```python
values['CODE_GEN_DIR_CPPSIM'] = self._safe_get_nodeattr('code_gen_dir_cppsim', '/tmp/test_cppsim')
```

This suggests that attribute resolution is failing and requires defensive programming.

## Recommended Fixes

### Fix 1: Proper Parent Method Calls

Update all parent method calls to use `super()` or correct explicit calls:

```python
def get_nodeattr_types(self):
    """Get node attribute types from both parent classes."""
    # Option 1: Using super() (recommended)
    attrs = super().get_nodeattr_types()
    
    # Option 2: Explicit calls (if specific order needed)
    attrs = {}
    attrs.update(Thresholding.get_nodeattr_types(self))
    attrs.update(CG_HLSBackend.get_nodeattr_types(self))
    
    return attrs
```

### Fix 2: Standardize Initialization

Ensure proper initialization order:

```python
def __init__(self, onnx_node, **kwargs):
    """Initialize with proper parent class initialization."""
    # Initialize operation first (has onnx_node dependency)
    Thresholding.__init__(self, onnx_node, **kwargs)
    # Then initialize backend
    CG_HLSBackend.__init__(self, **kwargs)
```

### Fix 3: Remove _safe_get_nodeattr Usage

Once attribute resolution is fixed, remove defensive programming:

```python
# Instead of:
values['CODE_GEN_DIR_CPPSIM'] = self._safe_get_nodeattr('code_gen_dir_cppsim', '/tmp/test_cppsim')

# Use direct access:
values['CODE_GEN_DIR_CPPSIM'] = self.get_nodeattr('code_gen_dir_cppsim')
```

### Fix 4: Implement Cooperative Multiple Inheritance

Use cooperative super() calls throughout:

```python
class CG_Thresholding_hls(Thresholding, CG_HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node=onnx_node, **kwargs)
    
    def get_nodeattr_types(self):
        # Let MRO handle attribute collection
        return super().get_nodeattr_types()
```

## Testing Recommendations

1. **Unit Tests for Attribute Resolution**:
   ```python
   def test_attribute_resolution():
       node = create_test_node()
       backend = CG_Thresholding_hls(node)
       attrs = backend.get_nodeattr_types()
       
       # Verify both parent attributes present
       assert 'PE' in attrs  # From Thresholding
       assert 'code_gen_dir_cppsim' in attrs  # From CG_HLSBackend
   ```

2. **MRO Validation Tests**:
   ```python
   def test_mro_order():
       mro = CG_Thresholding_hls.__mro__
       # Verify Thresholding comes before CG_HLSBackend
       assert mro.index(Thresholding) < mro.index(CG_HLSBackend)
   ```

3. **Integration Tests**: Verify full code generation flow works without AttributeError

## Issue 5: Missing get_nodeattr Method

### Critical Finding

The clean backends extensively use `self.get_nodeattr()` but this method is not defined in the `Codegen` base class. The method exists in `HWCustomOp` (through the operation parent), but the MRO and initialization might not properly establish access to it.

### Evidence

In `CG_Thresholding_hls`:
```python
values['PE'] = self.get_nodeattr('PE')  # Line 96
values['NUM_CHANNELS'] = self.get_nodeattr('NumChannels')  # Line 97
values['NUM_STEPS'] = self.get_nodeattr('numSteps')  # Line 98
```

The `_safe_get_nodeattr` method exists specifically to handle potential AttributeError:
```python
def _safe_get_nodeattr(self, attr_name: str, default_value=None):
    try:
        value = self.get_nodeattr(attr_name)  # This might fail!
        return value
    except (AttributeError, KeyError) as e:
        # Handle the error
```

### Root Cause

The `get_nodeattr` method is defined in the QONNX `CustomOp` base class, which is inherited through:
- `CG_Thresholding_hls` → `Thresholding` → `HWCustomOp` → `CustomOp` (has `get_nodeattr`)
- `CG_Thresholding_hls` → `CG_HLSBackend` → `Codegen` → `ABC` (no `get_nodeattr`)

If the initialization order is wrong or if `Codegen` methods are called before proper initialization, `get_nodeattr` might not be accessible.

## Conclusion

The multiple inheritance issues in FINN's clean backends stem from:
1. Complex MRO with multiple parent classes
2. Missing `get_nodeattr` method in the `Codegen` inheritance chain
3. Defensive programming patterns (`_safe_get_nodeattr`) confirming attribute resolution issues
4. Initialization order dependencies between operation and backend classes

The recommended fixes focus on:
1. Ensuring proper initialization order
2. Using cooperative multiple inheritance patterns
3. Possibly adding a `get_nodeattr` proxy method in `Codegen` base class
4. Thorough testing of attribute access patterns