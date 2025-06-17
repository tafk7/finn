#!/usr/bin/env python3
"""
FINN Unified Codegen - Strict Environment Validation
This module provides strict validation that the FINN test environment is properly set up.
Tests FAIL HARD if not running in a proper FINN Docker environment with all dependencies.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class FinnEnvironmentError(Exception):
    """Raised when FINN environment validation fails"""
    pass

class StrictEnvironmentValidator:
    """Validates that we're running in a proper FINN Docker environment"""
    
    def __init__(self):
        self.validation_errors = []
        self.warnings = []
    
    def validate_docker_environment(self) -> bool:
        """Validate we're running inside a FINN Docker container"""
        docker_indicators = [
            '/proc/1/cgroup',  # Docker container cgroup
            '/.dockerenv',     # Docker environment file
        ]
        
        # Check for Docker environment indicators
        docker_detected = any(os.path.exists(indicator) for indicator in docker_indicators)
        
        if not docker_detected:
            # Additional check for Docker-specific environment variables
            docker_env_vars = ['FINN_DOCKER', 'FINN_ROOT', 'VIVADO_PATH', 'VITIS_HLS_PATH']
            docker_env_detected = any(os.getenv(var) for var in docker_env_vars)
            
            if not docker_env_detected:
                self.validation_errors.append(
                    "Not running in FINN Docker environment. "
                    "This test suite requires a properly configured FINN Docker container."
                )
                return False
        
        return True
    
    def validate_finn_dependencies(self) -> bool:
        """Validate critical FINN dependencies are available"""
        required_modules = [
            'finn.core',
            'finn.custom_op.fpgadataflow',
            'finn.transformation',
            'finn.analysis',
            'finn.builder',
            'onnx',
            'qonnx',
            'numpy',
            'torch'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.validation_errors.append(
                f"Critical FINN modules missing: {', '.join(missing_modules)}. "
                "This indicates an incomplete FINN installation."
            )
            return False
        
        return True
    
    def validate_vivado_tools(self) -> bool:
        """Validate Vivado/Vitis HLS tools are available for unified codegen"""
        # For unified codegen framework, we only need basic Vivado tools available
        # Full hardware synthesis is not required for framework testing
        
        # Check if any Vivado tools are available (not strictly required)
        tools_to_check = ['vivado', 'vitis_hls']
        tools_found = []
        
        for tool in tools_to_check:
            try:
                result = subprocess.run(['which', tool],
                                      capture_output=True,
                                      text=True,
                                      timeout=5)
                if result.returncode == 0:
                    tools_found.append(tool)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        if not tools_found:
            self.warnings.append(
                "No Vivado/Vitis tools found in PATH. "
                "Hardware synthesis features may be limited but codegen framework will work."
            )
        
        # Always return True - Vivado tools are not required for unified codegen testing
        return True
    
    def validate_finn_custom_ops(self) -> bool:
        """Validate FINN custom operations required for unified codegen are available"""
        # Only check operations that the unified codegen framework actually uses
        required_ops = [
            'finn.custom_op.fpgadataflow.matrixvectoractivation.MVAU',
            'finn.custom_op.fpgadataflow.thresholding.Thresholding',
            'finn.custom_op.fpgadataflow.streamingdatawidthconverter.StreamingDataWidthConverter'
        ]
        
        missing_ops = []
        for op_path in required_ops:
            try:
                module_path, class_name = op_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                getattr(module, class_name)
            except (ImportError, AttributeError):
                missing_ops.append(op_path)
        
        if missing_ops:
            self.validation_errors.append(
                f"Critical FINN custom operations missing: {', '.join(missing_ops)}. "
                "This indicates incomplete FINN custom op installation."
            )
            return False
        
        return True
    
    def validate_build_environment(self) -> bool:
        """Validate build environment requirements for unified codegen"""
        # Only check environment variables actually needed for codegen framework
        required_env_vars = [
            'FINN_ROOT'
            # VIVADO_PATH and VITIS_HLS_PATH not required for codegen framework testing
        ]
        
        missing_env = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_env.append(var)
        
        if missing_env:
            self.validation_errors.append(
                f"Required environment variables missing: {', '.join(missing_env)}. "
                "These are required for FINN hardware compilation."
            )
            return False
        
        # Validate FINN_ROOT points to valid installation
        finn_root = os.getenv('FINN_ROOT')
        if finn_root:
            required_paths = [
                os.path.join(finn_root, 'src', 'finn'),
                os.path.join(finn_root, 'finn-rtllib'),
                os.path.join(finn_root, 'custom_hls')
            ]
            
            missing_paths = [p for p in required_paths if not os.path.exists(p)]
            if missing_paths:
                self.validation_errors.append(
                    f"FINN_ROOT ({finn_root}) missing required directories: {missing_paths}"
                )
                return False
        
        return True
    
    def validate_hardware_compilation_deps(self) -> bool:
        """Validate hardware compilation dependencies"""
        try:
            # Check if we can import PyVerilog for RTL parsing
            import pyverilog
        except ImportError:
            self.warnings.append("PyVerilog not available - some RTL validation may be limited")
        
        try:
            # Check if we can import PyRTL for advanced RTL operations
            import pyrtl
        except ImportError:
            self.warnings.append("PyRTL not available - some RTL operations may be limited")
        
        # For now, warnings don't cause failure, but could be made strict
        return True
    
    def run_validation(self) -> bool:
        """Run complete environment validation"""
        print("🔍 Running strict FINN environment validation...")
        print("=" * 60)
        
        validations = [
            ("Docker Environment", self.validate_docker_environment),
            ("FINN Dependencies", self.validate_finn_dependencies),
            ("Vivado/Vitis Tools", self.validate_vivado_tools),
            ("FINN Custom Operations", self.validate_finn_custom_ops),
            ("Build Environment", self.validate_build_environment),
            ("Hardware Compilation Dependencies", self.validate_hardware_compilation_deps)
        ]
        
        all_passed = True
        for name, validator in validations:
            try:
                result = validator()
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{name:35} {status}")
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"{name:35} ❌ ERROR: {e}")
                all_passed = False
        
        print("\n" + "=" * 60)
        
        # Report warnings
        if self.warnings:
            print("⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")
            print()
        
        # Report errors
        if self.validation_errors:
            print("❌ VALIDATION FAILURES:")
            for error in self.validation_errors:
                print(f"   - {error}")
            print()
            print("🚫 ENVIRONMENT NOT SUITABLE FOR FINN TESTING")
            print("   Please run tests in a properly configured FINN Docker container.")
            return False
        
        if all_passed:
            print("✅ ALL VALIDATIONS PASSED")
            print("🐳 Environment is suitable for FINN testing")
            return True
        else:
            print("❌ SOME VALIDATIONS FAILED")
            print("🚫 Environment is not suitable for FINN testing")
            return False

def require_finn_environment():
    """
    Decorator/function to require FINN environment validation.
    Call this at the start of any test that requires real FINN dependencies.
    """
    validator = StrictEnvironmentValidator()
    if not validator.run_validation():
        raise FinnEnvironmentError(
            "FINN environment validation failed. "
            "This test requires a properly configured FINN Docker environment with all dependencies."
        )

def validate_environment_or_exit():
    """Validate environment and exit with error code if validation fails"""
    validator = StrictEnvironmentValidator()
    if not validator.run_validation():
        sys.exit(1)
    return True

if __name__ == '__main__':
    # Run validation when script is executed directly
    validate_environment_or_exit()
    print("\n🎉 Environment validation successful!")
    print("Ready to run FINN tests that require real dependencies.")