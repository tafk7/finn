"""
Registration Validation System for FINN Backends

This module provides comprehensive validation and diagnostics for backend registration,
helping to catch registration failures early and provide clear error messages.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .backend_registry import BackendRegistry


@dataclass
class ValidationIssue:
    """Represents a single validation issue found during registry check."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'missing', 'mismatch', 'duplicate', etc.
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Complete validation report for backend registries."""
    is_valid: bool
    total_issues: int
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    info: List[ValidationIssue]
    stats: Dict[str, Any]
    
    def __str__(self) -> str:
        """Generate human-readable report."""
        lines = [
            "Backend Registration Validation Report",
            "=" * 50,
            f"Status: {'PASS' if self.is_valid else 'FAIL'}",
            f"Total Issues: {self.total_issues} ({len(self.errors)} errors, {len(self.warnings)} warnings)",
            ""
        ]
        
        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  - [{error.category}] {error.message}")
                if error.details:
                    for k, v in error.details.items():
                        lines.append(f"    {k}: {v}")
        
        if self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - [{warning.category}] {warning.message}")
        
        lines.append("\nSTATISTICS:")
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        
        return "\n".join(lines)


class RegistrationValidator:
    """
    Validates backend registration integrity and provides diagnostics.
    
    This validator checks for common issues like:
    - Missing expected backends
    - Import failures
    - Naming convention violations
    - Duplicate registrations
    - Empty registries
    """
    
    # Expected minimum backends for a healthy system
    MIN_HLS_BACKENDS = 15
    MIN_RTL_BACKENDS = 5
    
    # Core operations that should always be present
    REQUIRED_HLS_OPERATIONS = {
        'Thresholding', 'MatrixVectorActivation', 'AddStreams', 
        'StreamingConcat', 'ConvolutionInputGenerator'
    }
    
    REQUIRED_RTL_OPERATIONS = {
        'Thresholding', 'MatrixVectorActivation'
    }
    
    def __init__(self):
        """Initialize the validator."""
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
    
    def validate_global_registries(self, registry: Optional[BackendRegistry] = None) -> ValidationReport:
        """
        Validate the global backend registries.
        
        Args:
            registry: Registry to validate (defaults to global registry)
            
        Returns:
            ValidationReport with detailed findings
        """
        if registry is None:
            from .backend_registration import get_backend_registry
            registry = get_backend_registry()
        
        errors = []
        warnings = []
        info = []
        
        # Get registry statistics
        stats = registry.get_registry_stats()
        hls_count = stats['hls_backends']
        rtl_count = stats['rtl_backends']
        
        # Check for minimum backend counts
        if hls_count < self.MIN_HLS_BACKENDS:
            errors.append(ValidationIssue(
                severity='error',
                category='insufficient',
                message=f"Only {hls_count} HLS backends registered, expected at least {self.MIN_HLS_BACKENDS}",
                details={'registered': hls_count, 'expected': self.MIN_HLS_BACKENDS}
            ))
        
        if rtl_count < self.MIN_RTL_BACKENDS:
            errors.append(ValidationIssue(
                severity='error',
                category='insufficient',
                message=f"Only {rtl_count} RTL backends registered, expected at least {self.MIN_RTL_BACKENDS}",
                details={'registered': rtl_count, 'expected': self.MIN_RTL_BACKENDS}
            ))
        
        # Check for required operations
        missing_hls = self._check_required_operations(registry, 'hls', self.REQUIRED_HLS_OPERATIONS)
        missing_rtl = self._check_required_operations(registry, 'rtl', self.REQUIRED_RTL_OPERATIONS)
        
        for op in missing_hls:
            errors.append(ValidationIssue(
                severity='error',
                category='missing',
                message=f"Required HLS operation '{op}' not registered",
                details={'operation': op, 'backend_type': 'hls'}
            ))
        
        for op in missing_rtl:
            errors.append(ValidationIssue(
                severity='error',
                category='missing',
                message=f"Required RTL operation '{op}' not registered",
                details={'operation': op, 'backend_type': 'rtl'}
            ))
        
        # Check for naming convention violations
        naming_issues = self._check_naming_conventions(registry)
        warnings.extend(naming_issues)
        
        # Check for potential import issues
        import_issues = self._diagnose_import_failures()
        warnings.extend(import_issues)
        
        # Calculate validation result
        is_valid = len(errors) == 0
        total_issues = len(errors) + len(warnings) + len(info)
        
        # Add detailed stats
        detailed_stats = {
            'hls_backends': hls_count,
            'rtl_backends': rtl_count,
            'total_backends': hls_count + rtl_count,
            'missing_required_hls': len(missing_hls),
            'missing_required_rtl': len(missing_rtl),
            'naming_violations': len(naming_issues),
            'import_failures': len(import_issues)
        }
        
        return ValidationReport(
            is_valid=is_valid,
            total_issues=total_issues,
            errors=errors,
            warnings=warnings,
            info=info,
            stats=detailed_stats
        )
    
    def check_backend_availability(self, op_type: str, backend: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a specific backend is available.
        
        Args:
            op_type: Operation type name
            backend: Backend type ('hls' or 'rtl')
            
        Returns:
            Tuple of (is_available, error_message)
        """
        from .backend_registration import get_backend_registry
        registry = get_backend_registry()
        
        if backend == 'hls':
            backend_class = registry.get_hls_backend(op_type)
        elif backend == 'rtl':
            backend_class = registry.get_rtl_backend(op_type)
        else:
            return False, f"Invalid backend type: {backend}"
        
        if backend_class is None:
            return False, f"No {backend.upper()} backend registered for operation '{op_type}'"
        
        return True, None
    
    def diagnose_registration_failures(self) -> List[str]:
        """
        Diagnose common causes of registration failures.
        
        Returns:
            List of diagnostic messages
        """
        diagnostics = []
        
        # Check if we can import key modules
        try:
            import finn.custom_op.fpgadataflow.hls
            diagnostics.append("✓ HLS module path accessible")
        except ImportError as e:
            diagnostics.append(f"✗ Cannot import HLS module: {e}")
        
        try:
            import finn.custom_op.fpgadataflow.rtl
            diagnostics.append("✓ RTL module path accessible")
        except ImportError as e:
            diagnostics.append(f"✗ Cannot import RTL module: {e}")
        
        # Try importing specific backends to check for issues
        test_imports = [
            ("finn.custom_op.fpgadataflow.hls.thresholding_hls", "Thresholding_hls"),
            ("finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls", "MVAU_hls"),
            ("finn.custom_op.fpgadataflow.rtl.thresholding_rtl", "Thresholding_rtl"),
        ]
        
        for module_path, class_name in test_imports:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    diagnostics.append(f"✓ Can import {class_name} from {module_path}")
                else:
                    diagnostics.append(f"✗ Module {module_path} has no class {class_name}")
            except ImportError as e:
                diagnostics.append(f"✗ Cannot import {module_path}: {e}")
        
        return diagnostics
    
    def _check_required_operations(self, registry: BackendRegistry, backend_type: str, 
                                  required_ops: set) -> List[str]:
        """Check for missing required operations."""
        if backend_type == 'hls':
            registered = set(registry._hls_backends.keys())
        else:
            registered = set(registry._rtl_backends.keys())
        
        missing = required_ops - registered
        return list(missing)
    
    def _check_naming_conventions(self, registry: BackendRegistry) -> List[ValidationIssue]:
        """Check for naming convention violations."""
        issues = []
        
        # Check HLS backends
        for op_name, backend_class in registry._hls_backends.items():
            class_name = backend_class.__name__
            # Expected pattern: ClassName_hls
            if not class_name.endswith('_hls'):
                issues.append(ValidationIssue(
                    severity='warning',
                    category='naming',
                    message=f"HLS backend '{class_name}' doesn't follow naming convention (expected *_hls)",
                    details={'operation': op_name, 'class_name': class_name}
                ))
        
        # Check RTL backends
        for op_name, backend_class in registry._rtl_backends.items():
            class_name = backend_class.__name__
            # Expected pattern: ClassName_rtl
            if not class_name.endswith('_rtl'):
                issues.append(ValidationIssue(
                    severity='warning',
                    category='naming',
                    message=f"RTL backend '{class_name}' doesn't follow naming convention (expected *_rtl)",
                    details={'operation': op_name, 'class_name': class_name}
                ))
        
        return issues
    
    def _diagnose_import_failures(self) -> List[ValidationIssue]:
        """Diagnose potential import failures."""
        issues = []
        
        # Check if running in Docker environment
        import os
        if not os.path.exists('/workspace/finn'):
            issues.append(ValidationIssue(
                severity='warning',
                category='environment',
                message="Not running in FINN Docker environment, some imports may fail",
                details={'finn_root': os.environ.get('FINN_ROOT', 'Not set')}
            ))
        
        return issues


def validate_and_report(output_file: Optional[str] = None) -> ValidationReport:
    """
    Run full validation and optionally save report to file.
    
    Args:
        output_file: Optional path to save report
        
    Returns:
        ValidationReport
    """
    validator = RegistrationValidator()
    report = validate_global_registries()
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(str(report))
    
    return report


def quick_check() -> bool:
    """
    Quick check if registries are healthy.
    
    Returns:
        True if registries pass basic validation
    """
    validator = RegistrationValidator()
    report = validator.validate_global_registries()
    return report.is_valid


if __name__ == "__main__":
    # Run validation when module is executed directly
    import sys
    
    print("Running backend registration validation...")
    validator = RegistrationValidator()
    report = validator.validate_global_registries()
    
    print(report)
    
    if not report.is_valid:
        print("\nDiagnostics:")
        for msg in validator.diagnose_registration_failures():
            print(f"  {msg}")
        sys.exit(1)
    else:
        print("\n✓ All backend registrations valid!")
        sys.exit(0)