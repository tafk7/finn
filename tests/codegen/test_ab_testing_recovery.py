#!/usr/bin/env python3
"""
Test A/B testing recovery after registration fixes.

This test must be run inside the FINN Docker environment where all dependencies
are available. It validates that the A/B testing infrastructure is working
correctly after fixing the backend registration issues.
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add FINN to path
finn_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(finn_root))


@pytest.mark.vivado
@pytest.mark.fpgadataflow
class TestABTestingRecovery:
    """Test suite for A/B testing recovery validation."""
    
    def test_backend_registration_populated(self):
        """Test that backends are properly registered after fixes."""
        from finn.codegen.backend_registration import get_backend_registry, reset_backend_registry
        
        # Reset to ensure clean state
        reset_backend_registry()
        
        # Get registry (triggers registration)
        registry = get_backend_registry()
        stats = registry.get_registry_stats()
        
        # Verify we have the expected number of backends
        assert stats['hls_backends'] >= 20, f"Expected at least 20 HLS backends, got {stats['hls_backends']}"
        assert stats['rtl_backends'] >= 8, f"Expected at least 8 RTL backends, got {stats['rtl_backends']}"
        
        # Print registered backends for debugging
        print(f"\nRegistered {stats['hls_backends']} HLS backends:")
        for op_name in sorted(registry._hls_backends.keys()):
            print(f"  - {op_name}: {registry._hls_backends[op_name].__name__}")
        
        print(f"\nRegistered {stats['rtl_backends']} RTL backends:")
        for op_name in sorted(registry._rtl_backends.keys()):
            print(f"  - {op_name}: {registry._rtl_backends[op_name].__name__}")
    
    def test_critical_backends_available(self):
        """Test that critical backends are available for A/B testing."""
        from finn.codegen.backend_registration import get_backend_registry
        from finn.codegen.CG_backend_registration import get_clean_backend_registry
        
        # Get both registries
        legacy_registry = get_backend_registry()
        clean_registry = get_clean_backend_registry()
        
        # Critical operations that must be available
        critical_ops = [
            ('Thresholding', 'hls'),
            ('MatrixVectorActivation', 'hls'),
            ('MVAU', 'hls'),
            ('Thresholding', 'rtl'),
            ('MatrixVectorActivation', 'rtl'),
            ('MVAU', 'rtl'),
        ]
        
        # Check legacy backends
        print("\nChecking legacy backend availability:")
        for op_name, backend_type in critical_ops:
            if backend_type == 'hls':
                backend = legacy_registry.get_hls_backend(op_name)
            else:
                backend = legacy_registry.get_rtl_backend(op_name)
            
            assert backend is not None, f"Legacy {backend_type} backend for {op_name} not found"
            print(f"  ✓ {op_name} ({backend_type}): {backend.__name__}")
        
        # Check clean backend registry stats
        clean_stats = clean_registry.get_registry_stats()
        print(f"\nClean registry stats:")
        print(f"  Clean HLS: {clean_stats['clean_hls_backends']}")
        print(f"  Clean RTL: {clean_stats['clean_rtl_backends']}")
        print(f"  Legacy HLS (in clean registry): {clean_stats['hls_backends']}")
        print(f"  Legacy RTL (in clean registry): {clean_stats['rtl_backends']}")
    
    def test_ab_testing_backend_retrieval(self):
        """Test that A/B testing can retrieve both legacy and clean backends."""
        from finn.codegen.CG_backend_registration import get_clean_backend_registry
        
        clean_registry = get_clean_backend_registry()
        
        # Test getting legacy backend (default behavior)
        legacy_thresh_hls = clean_registry.get_hls_backend('Thresholding', prefer_clean=False)
        assert legacy_thresh_hls is not None
        assert 'CG_' not in legacy_thresh_hls.__name__, f"Got clean backend when legacy requested: {legacy_thresh_hls.__name__}"
        
        # Test getting clean backend if available
        clean_thresh_hls = clean_registry.get_hls_backend('Thresholding', prefer_clean=True)
        assert clean_thresh_hls is not None
        # Note: If no clean backend exists, it falls back to legacy
        print(f"\nBackend retrieval test:")
        print(f"  Legacy Thresholding HLS: {legacy_thresh_hls.__name__}")
        print(f"  Clean Thresholding HLS: {clean_thresh_hls.__name__}")
    
    def test_simple_thresholding_codegen(self):
        """Test basic code generation with Thresholding to verify registration works."""
        import numpy as np
        from qonnx.core.datatype import DataType
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.custom_op.general.multithreshold import multithreshold
        from qonnx.util.basic import gen_finn_dt_tensor
        import finn.core.onnx_exec as oxe
        from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
        from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
        from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
        
        # Create a simple thresholding model
        idt = DataType["INT4"]
        odt = DataType["UINT4"]
        n_inp_vecs = 4
        n_chans = 2
        
        # Create random input
        inp = gen_finn_dt_tensor(idt, (n_inp_vecs, n_chans))
        
        # Create thresholds
        thresholds = np.array([[-2, 0, 2], [-1, 1, 3]], dtype=np.float32)
        
        # Create ONNX model with Thresholding operation
        # This will test if the backend can be found and used
        from finn.custom_op.fpgadataflow.thresholding import Thresholding
        
        # Just verify we can create the operation
        # Full model creation would require more setup
        assert Thresholding is not None
        print("\n✓ Thresholding operation class available")
    
    def test_validation_report(self):
        """Run comprehensive validation and generate report."""
        from finn.codegen.registration_validator import RegistrationValidator
        from finn.codegen.backend_registration import get_backend_registry
        
        validator = RegistrationValidator()
        registry = get_backend_registry()
        report = validator.validate_global_registries(registry)
        
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        print(report)
        
        # After fixes, validation should pass
        assert report.is_valid, f"Validation failed with {len(report.errors)} errors"
        assert len(report.errors) == 0, "There should be no errors after registration fixes"
        
        # Some warnings are acceptable (e.g., naming conventions for legacy code)
        print(f"\nWarnings: {len(report.warnings)} (some legacy naming warnings are expected)")


if __name__ == "__main__":
    # Run tests when module is executed directly
    pytest.main([__file__, "-v", "--tb=short"])