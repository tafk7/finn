# Development Notes

## 2024-12-19: Cleaned up unnecessary get_nodeattr proxy

After review, determined that the get_nodeattr proxy method added to Codegen base class was unnecessary because:

1. QONNX's CustomOp already provides get_nodeattr
2. Due to Method Resolution Order (MRO), CustomOp.get_nodeattr is always used (CustomOp comes before Codegen)
3. The proxy was never actually being called

### Changes made:
- Removed get_nodeattr proxy method from Codegen base class (50 lines)
- Removed _get_nodeattr_impl references from all clean backends
- Updated test to not expect 'default' parameter support
- Verified all tests still pass

### Result:
- Cleaner, simpler code
- Same functionality through normal inheritance
- All tests passing