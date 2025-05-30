#!/usr/bin/env python3
import sys
print("Python sys.path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("\nChecking for FINN package locations:")
try:
    import finn
    print(f"finn.__file__: {getattr(finn, '__file__', 'Not available')}")
    print(f"finn.__path__: {getattr(finn, '__path__', 'Not available')}")
except Exception as e:
    print(f"Error importing finn: {e}")

print("\nChecking for transformation module:")
try:
    from finn.transformation.general import CreateDataflowPartition
    print("✅ finn.transformation.general imported successfully")
except Exception as e:
    print(f"❌ finn.transformation.general import failed: {e}")
