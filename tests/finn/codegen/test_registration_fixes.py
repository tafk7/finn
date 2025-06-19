#!/usr/bin/env python3
"""
Test script to verify backend registration fixes are working correctly.

This script:
1. Tests the fixed backend registration
2. Validates registries are populated
3. Checks A/B testing capability
4. Reports on registration health
"""

import sys
import logging
from pathlib import Path

# Configure logging to see registration details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add FINN to path if needed
finn_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(finn_root))

def test_backend_registration():
    """Test that backend registration is working correctly."""
    print("=" * 80)
    print("Testing Backend Registration Fixes")
    print("=" * 80)
    
    # Test 1: Import and initialize registry
    print("\n1. Testing registry initialization...")
    try:
        from finn.codegen.backend_registration import get_backend_registry, reset_backend_registry
        
        # Reset to ensure clean state
        reset_backend_registry()
        
        # Get registry (triggers registration)
        registry = get_backend_registry()
        print("✓ Registry initialized successfully")
        
        # Get stats
        stats = registry.get_registry_stats()
        print(f"\nRegistry Statistics:")
        print(f"  HLS Backends: {stats['hls_backends']}")
        print(f"  RTL Backends: {stats['rtl_backends']}")
        
        # List all registered backends
        print(f"\nRegistered HLS Operations ({len(registry._hls_backends)}):")
        for op_name in sorted(registry._hls_backends.keys()):
            print(f"  - {op_name}: {registry._hls_backends[op_name].__name__}")
        
        print(f"\nRegistered RTL Operations ({len(registry._rtl_backends)}):")
        for op_name in sorted(registry._rtl_backends.keys()):
            print(f"  - {op_name}: {registry._rtl_backends[op_name].__name__}")
        
    except Exception as e:
        print(f"✗ Failed to initialize registry: {e}")
        return False
    
    # Test 2: Validate critical backends are available
    print("\n2. Testing critical backend availability...")
    critical_tests = [
        ('Thresholding', 'hls'),
        ('MatrixVectorActivation', 'hls'),
        ('MVAU', 'hls'),
        ('Thresholding', 'rtl'),
        ('MatrixVectorActivation', 'rtl'),
        ('MVAU', 'rtl'),
    ]
    
    all_passed = True
    for op_name, backend_type in critical_tests:
        if backend_type == 'hls':
            backend_class = registry.get_hls_backend(op_name)
        else:
            backend_class = registry.get_rtl_backend(op_name)
        
        if backend_class:
            print(f"  ✓ {op_name} ({backend_type}): {backend_class.__name__}")
        else:
            print(f"  ✗ {op_name} ({backend_type}): NOT FOUND")
            all_passed = False
    
    # Test 3: Run validation
    print("\n3. Running comprehensive validation...")
    try:
        from finn.codegen.registration_validator import RegistrationValidator
        
        validator = RegistrationValidator()
        report = validator.validate_global_registries(registry)
        
        print(f"\nValidation Report:")
        print(f"  Status: {'PASS' if report.is_valid else 'FAIL'}")
        print(f"  Errors: {len(report.errors)}")
        print(f"  Warnings: {len(report.warnings)}")
        
        if report.errors:
            print("\n  Errors found:")
            for error in report.errors:
                print(f"    - {error.message}")
        
        if report.warnings:
            print("\n  Warnings found:")
            for warning in report.warnings[:5]:  # Show first 5
                print(f"    - {warning.message}")
            if len(report.warnings) > 5:
                print(f"    ... and {len(report.warnings) - 5} more warnings")
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False
    
    # Test 4: Check A/B testing capability
    print("\n4. Testing A/B testing infrastructure...")
    try:
        from finn.codegen.CG_backend_registration import get_clean_backend_registry
        
        clean_registry = get_clean_backend_registry()
        clean_stats = clean_registry.get_registry_stats()
        
        print(f"  Clean HLS Backends: {clean_stats['clean_hls_backends']}")
        print(f"  Clean RTL Backends: {clean_stats['clean_rtl_backends']}")
        print(f"  Legacy HLS Backends: {clean_stats['hls_backends']}")
        print(f"  Legacy RTL Backends: {clean_stats['rtl_backends']}")
        
        # Test getting backends with clean preference
        test_backend = clean_registry.get_hls_backend('Thresholding', prefer_clean=True)
        if test_backend:
            print(f"  ✓ Can retrieve backends for A/B testing: {test_backend.__name__}")
        else:
            print(f"  ✗ Failed to retrieve backend for A/B testing")
        
    except Exception as e:
        print(f"✗ A/B testing infrastructure error: {e}")
        return False
    
    # Final summary
    print("\n" + "=" * 80)
    if all_passed and report.is_valid:
        print("✅ ALL TESTS PASSED - Registration fixes are working!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Registration still has issues")
        return False


def main():
    """Main test execution."""
    success = test_backend_registration()
    
    # Diagnostics if failed
    if not success:
        print("\nRunning diagnostics...")
        try:
            from finn.codegen.registration_validator import RegistrationValidator
            validator = RegistrationValidator()
            diagnostics = validator.diagnose_registration_failures()
            for msg in diagnostics:
                print(f"  {msg}")
        except Exception as e:
            print(f"  Failed to run diagnostics: {e}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()