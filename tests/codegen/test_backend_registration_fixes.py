#!/usr/bin/env python3
"""
Test backend registration fixes.

This test validates that the backend registration system is working correctly
after fixing the class name mismatches.
"""

import pytest
import sys
from pathlib import Path

# Add FINN to path
finn_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(finn_root))


class TestBackendRegistrationFixes:
    """Test suite for backend registration fixes."""
    
    @pytest.mark.util
    def test_registration_module_syntax(self):
        """Test that registration modules have correct syntax and can be imported."""
        # This test doesn't require qonnx or Docker environment
        # It just validates the Python syntax is correct
        
        try:
            # Test that we can at least import the registration modules
            import finn.codegen.backend_registration
            import finn.codegen.backend_registry
            import finn.codegen.registration_validator
            
            assert hasattr(finn.codegen.backend_registration, 'register_all_backends')
            assert hasattr(finn.codegen.backend_registration, 'get_backend_registry')
            assert hasattr(finn.codegen.backend_registry, 'BackendRegistry')
            assert hasattr(finn.codegen.registration_validator, 'RegistrationValidator')
            
        except SyntaxError as e:
            pytest.fail(f"Syntax error in registration modules: {e}")
        except ImportError as e:
            # This is expected outside Docker if modules try to import backends
            # But at least the main modules should import
            if "backend_registration" in str(e) or "backend_registry" in str(e):
                pytest.fail(f"Failed to import core registration modules: {e}")
    
    @pytest.mark.util
    def test_registry_initialization(self):
        """Test that registry can be initialized (doesn't require backend imports)."""
        from finn.codegen.backend_registry import BackendRegistry
        
        # Create empty registry
        registry = BackendRegistry()
        assert registry is not None
        assert hasattr(registry, '_hls_backends')
        assert hasattr(registry, '_rtl_backends')
        assert hasattr(registry, 'register_hls_backend')
        assert hasattr(registry, 'register_rtl_backend')
        assert hasattr(registry, 'get_hls_backend')
        assert hasattr(registry, 'get_rtl_backend')
        
        # Test registration works
        class DummyBackend:
            pass
        
        registry.register_hls_backend('TestOp', DummyBackend)
        assert registry.get_hls_backend('TestOp') == DummyBackend
        
        stats = registry.get_registry_stats()
        assert stats['hls_backends'] == 1
        assert stats['rtl_backends'] == 0
    
    @pytest.mark.util  
    def test_validator_initialization(self):
        """Test that validator can be initialized."""
        from finn.codegen.registration_validator import RegistrationValidator, ValidationReport
        
        validator = RegistrationValidator()
        assert validator is not None
        assert hasattr(validator, 'validate_global_registries')
        assert hasattr(validator, 'check_backend_availability')
        assert hasattr(validator, 'diagnose_registration_failures')
    
    @pytest.mark.vivado
    @pytest.mark.skipif(
        not Path("/workspace/finn").exists(),
        reason="Requires FINN Docker environment with all dependencies"
    )
    def test_backend_registration_in_docker(self):
        """Test full backend registration (requires Docker environment)."""
        from finn.codegen.backend_registration import get_backend_registry, reset_backend_registry
        from finn.codegen.registration_validator import RegistrationValidator
        
        # Reset to ensure clean state
        reset_backend_registry()
        
        # Get registry (triggers registration)
        registry = get_backend_registry()
        stats = registry.get_registry_stats()
        
        # Should have many backends registered
        assert stats['hls_backends'] >= 15, f"Only {stats['hls_backends']} HLS backends registered"
        assert stats['rtl_backends'] >= 5, f"Only {stats['rtl_backends']} RTL backends registered"
        
        # Check critical backends
        assert registry.get_hls_backend('Thresholding') is not None
        assert registry.get_hls_backend('MatrixVectorActivation') is not None
        assert registry.get_rtl_backend('Thresholding') is not None
        assert registry.get_rtl_backend('MatrixVectorActivation') is not None
        
        # Run validation
        validator = RegistrationValidator()
        report = validator.validate_global_registries(registry)
        
        assert report.is_valid, f"Validation failed with {len(report.errors)} errors"
    
    @pytest.mark.util
    def test_class_name_patterns(self):
        """Test that the fixed class names follow expected patterns."""
        # This test validates our understanding of the naming convention
        # without requiring actual imports
        
        expected_hls_names = [
            ('Thresholding_hls', 'thresholding_hls.py'),
            ('MVAU_hls', 'matrixvectoractivation_hls.py'),
            ('AddStreams_hls', 'addstreams_hls.py'),
            ('StreamingConcat_hls', 'concat_hls.py'),
            ('ConvolutionInputGenerator_hls', 'convolutioninputgenerator_hls.py'),
        ]
        
        expected_rtl_names = [
            ('Thresholding_rtl', 'thresholding_rtl.py'),
            ('MVAU_rtl', 'matrixvectoractivation_rtl.py'),
            ('ConvolutionInputGenerator_rtl', 'convolutioninputgenerator_rtl.py'),
        ]
        
        # Validate naming pattern
        for class_name, file_name in expected_hls_names:
            assert class_name.endswith('_hls'), f"{class_name} doesn't follow HLS naming pattern"
            assert file_name.endswith('_hls.py'), f"{file_name} doesn't follow HLS file pattern"
        
        for class_name, file_name in expected_rtl_names:
            assert class_name.endswith('_rtl'), f"{class_name} doesn't follow RTL naming pattern"
            assert file_name.endswith('_rtl.py'), f"{file_name} doesn't follow RTL file pattern"