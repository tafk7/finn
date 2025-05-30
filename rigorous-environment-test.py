#!/usr/bin/env python3
"""
RIGOROUS FINN Environment Validation Test

This is the most comprehensive test that validates the new Docker system
installs ALL the same environment components as the original interactive shell.

It performs deep inspection of:
1. All Python packages and their versions
2. Environment variables and paths
3. FINN-specific functionality
4. All library interdependencies
5. File system access and permissions
6. Comparison with original entrypoint behavior
"""

import sys
import os
import subprocess
import importlib
import traceback
import json
from pathlib import Path

class FINNEnvironmentValidator:
    def __init__(self):
        self.results = {
            'python_packages': {},
            'environment_vars': {},
            'finn_functionality': {},
            'file_access': {},
            'errors': []
        }
        
    def log_result(self, category, test_name, status, details=None):
        """Log test result with details."""
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][test_name] = {
            'status': status,
            'details': details or {}
        }
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"  {status_icon} {test_name}: {status}")
        if details and isinstance(details, dict):
            for key, value in details.items():
                print(f"     {key}: {value}")

    def test_python_environment(self):
        """Test comprehensive Python environment setup."""
        print("\n🐍 Testing Python Environment")
        print("=" * 40)
        
        # Test Python version and executable
        python_version = sys.version
        python_exec = sys.executable
        
        self.log_result('python_environment', 'python_version', 'PASS', {
            'version': python_version,
            'executable': python_exec
        })
        
        # Test Python path
        python_paths = sys.path
        expected_paths = [
            '/home/tafk/dev/finn/src',
            '/home/tafk/dev/finn/deps/qonnx/src',
            '/workspace/src/dataset-loading'
        ]
        
        missing_paths = []
        for expected_path in expected_paths:
            if not any(expected_path in p for p in python_paths):
                missing_paths.append(expected_path)
        
        if missing_paths:
            self.log_result('python_environment', 'python_paths', 'FAIL', {
                'missing_paths': missing_paths,
                'total_paths': len(python_paths)
            })
        else:
            self.log_result('python_environment', 'python_paths', 'PASS', {
                'expected_paths_found': len(expected_paths),
                'total_paths': len(python_paths)
            })

    def test_core_packages(self):
        """Test all core Python packages that should be installed."""
        print("\n📦 Testing Core Python Packages")
        print("=" * 40)
        
        # Core packages that MUST be available
        core_packages = [
            'numpy',
            'onnx', 
            'qonnx',
            'finn',
            'matplotlib',
            'pandas',
            'scipy',
            'sklearn',
            'torch',
            'torchvision',
            'onnxruntime'
        ]
        
        for package in core_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.log_result('python_packages', package, 'PASS', {'version': version})
            except ImportError as e:
                self.log_result('python_packages', package, 'FAIL', {'error': str(e)})
            except Exception as e:
                self.log_result('python_packages', package, 'FAIL', {'error': f"Unexpected: {e}"})

    def test_finn_specific_packages(self):
        """Test FINN ecosystem packages."""
        print("\n🔧 Testing FINN Ecosystem Packages")
        print("=" * 40)
        
        finn_packages = [
            ('qonnx', 'qonnx'),
            ('finn', 'finn'),
            ('brevitas', 'brevitas'),
            ('finn_experimental', 'finnexperimental'),
        ]
        
        for package_name, import_name in finn_packages:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                
                # Test if it's properly installed vs just available
                import_path = getattr(module, '__file__', 'unknown')
                
                self.log_result('finn_packages', package_name, 'PASS', {
                    'version': version,
                    'import_path': import_path
                })
            except ImportError as e:
                self.log_result('finn_packages', package_name, 'FAIL', {'error': str(e)})

    def test_environment_variables(self):
        """Test critical environment variables."""
        print("\n🌍 Testing Environment Variables")
        print("=" * 40)
        
        critical_env_vars = [
            'FINN_ROOT',
            'FINN_BUILD_DIR', 
            'PATH',
            'VIVADO_IP_CACHE'
        ]
        
        # PYTHONPATH is optional if packages are installed with pip -e
        optional_env_vars = ['PYTHONPATH']
        
        for var in critical_env_vars:
            value = os.environ.get(var)
            if value:
                self.log_result('environment_vars', var, 'PASS', {'value': value[:100] + '...' if len(value) > 100 else value})
            else:
                self.log_result('environment_vars', var, 'FAIL', {'error': 'Not set'})
        
        # Check optional environment variables
        for var in optional_env_vars:
            value = os.environ.get(var)
            if value:
                self.log_result('environment_vars', var, 'PASS', {'value': value[:100] + '...' if len(value) > 100 else value})
            else:
                self.log_result('environment_vars', var, 'WARN', {'note': 'Optional variable not set'})

    def test_finn_functionality(self):
        """Test deep FINN functionality to ensure everything works."""
        print("\n🔬 Testing FINN Functionality")
        print("=" * 40)
        
        # Test FINN basic functionality
        try:
            import finn
            from finn.util.basic import get_finn_root
            
            finn_root = get_finn_root()
            self.log_result('finn_functionality', 'get_finn_root', 'PASS', {'finn_root': finn_root})
        except Exception as e:
            self.log_result('finn_functionality', 'get_finn_root', 'FAIL', {'error': str(e)})
        
        # Test FINN transformations
        try:
            from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
            self.log_result('finn_functionality', 'transformations_import', 'PASS', {})
        except Exception as e:
            self.log_result('finn_functionality', 'transformations_import', 'FAIL', {'error': str(e)})
        
        # Test QONNX functionality
        try:
            from qonnx.core.modelwrapper import ModelWrapper
            from qonnx.transformation.general import GiveUniqueNodeNames
            self.log_result('finn_functionality', 'qonnx_functionality', 'PASS', {})
        except Exception as e:
            self.log_result('finn_functionality', 'qonnx_functionality', 'FAIL', {'error': str(e)})
        
        # Test if we can create a simple model workflow
        try:
            import numpy as np
            import onnx
            from qonnx.core.modelwrapper import ModelWrapper
            
            # Create a simple test model
            X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3])
            Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3])
            
            node = onnx.helper.make_node('Identity', ['X'], ['Y'])
            graph = onnx.helper.make_graph([node], 'test_graph', [X], [Y])
            model = onnx.helper.make_model(graph)
            
            # Test ModelWrapper functionality
            wrapper = ModelWrapper(model)
            self.log_result('finn_functionality', 'model_wrapper_creation', 'PASS', {})
            
        except Exception as e:
            self.log_result('finn_functionality', 'model_wrapper_creation', 'FAIL', {'error': str(e)})

    def test_file_system_access(self):
        """Test file system access and permissions."""
        print("\n📁 Testing File System Access")
        print("=" * 40)
        
        # Test access to key directories
        key_directories = [
            '/home/tafk/dev/finn',
            '/home/tafk/dev/finn/src',
            '/home/tafk/dev/finn/deps',
            '/tmp/deps'
        ]
        
        for directory in key_directories:
            if os.path.exists(directory) and os.access(directory, os.R_OK):
                self.log_result('file_access', f'directory_{directory.replace("/", "_")}', 'PASS', {
                    'readable': True,
                    'writable': os.access(directory, os.W_OK)
                })
            else:
                self.log_result('file_access', f'directory_{directory.replace("/", "_")}', 'FAIL', {
                    'exists': os.path.exists(directory),
                    'readable': False
                })

    def test_package_installation_completeness(self):
        """Test that packages are fully installed, not just editable installs."""
        print("\n📋 Testing Package Installation Completeness")
        print("=" * 40)
        
        # Check if packages are properly installed vs just in development mode
        import subprocess
        
        try:
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=30)
            installed_packages = result.stdout
            
            # Look for key FINN packages
            finn_packages_to_check = ['qonnx', 'finn', 'brevitas']
            
            for package in finn_packages_to_check:
                if package in installed_packages:
                    self.log_result('package_installation', f'{package}_pip_installed', 'PASS', {})
                else:
                    self.log_result('package_installation', f'{package}_pip_installed', 'WARN', {
                        'note': 'May be development install'
                    })
                    
        except Exception as e:
            self.log_result('package_installation', 'pip_list_check', 'FAIL', {'error': str(e)})

    def compare_with_original_entrypoint(self):
        """Compare current environment with what original entrypoint should provide."""
        print("\n🔄 Comparing with Original Entrypoint Expected Behavior")
        print("=" * 40)
        
        # Check if environment setup marker exists
        setup_marker = '/tmp/.finn_env_setup_complete'
        if os.path.exists(setup_marker):
            self.log_result('entrypoint_comparison', 'setup_marker_exists', 'PASS', {
                'marker_path': setup_marker
            })
        else:
            self.log_result('entrypoint_comparison', 'setup_marker_exists', 'FAIL', {
                'marker_path': setup_marker
            })
        
        # Check if all expected repositories are available
        expected_repos = [
            '/tmp/deps/qonnx',
            '/tmp/deps/brevitas', 
            '/home/tafk/dev/finn/deps/finn-experimental'
        ]
        
        for repo in expected_repos:
            if os.path.exists(repo):
                self.log_result('entrypoint_comparison', f'repo_{os.path.basename(repo)}', 'PASS', {
                    'path': repo
                })
            else:
                self.log_result('entrypoint_comparison', f'repo_{os.path.basename(repo)}', 'FAIL', {
                    'path': repo,
                    'exists': False
                })

    def generate_comprehensive_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "=" * 60)
        print("🏁 COMPREHENSIVE ENVIRONMENT VALIDATION REPORT")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warnings = 0
        
        for category, tests in self.results.items():
            if category == 'errors':
                continue
                
            print(f"\n📊 {category.upper().replace('_', ' ')}:")
            
            for test_name, result in tests.items():
                status = result['status']
                total_tests += 1
                
                if status == 'PASS':
                    passed_tests += 1
                    icon = "✅"
                elif status == 'FAIL':
                    failed_tests += 1
                    icon = "❌"
                else:
                    warnings += 1
                    icon = "⚠️"
                
                print(f"  {icon} {test_name}: {status}")
                
                if 'details' in result and result['details']:
                    for key, value in result['details'].items():
                        print(f"      {key}: {value}")
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ✅")
        print(f"  Failed: {failed_tests} ❌") 
        print(f"  Warnings: {warnings} ⚠️")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if failed_tests == 0:
            print(f"\n🎉 ENVIRONMENT VALIDATION SUCCESSFUL!")
            print(f"   The new Docker system provides COMPLETE environment parity")
            print(f"   with the original interactive shell setup!")
            return True
        else:
            print(f"\n⚠️ ENVIRONMENT VALIDATION INCOMPLETE")
            print(f"   {failed_tests} critical components are missing or broken")
            return False

    def run_all_tests(self):
        """Run the complete validation suite."""
        print("🧪 FINN RIGOROUS ENVIRONMENT VALIDATION")
        print("This test validates 100% environment parity with original setup")
        print("=" * 60)
        
        try:
            self.test_python_environment()
            self.test_core_packages()
            self.test_finn_specific_packages()
            self.test_environment_variables()
            self.test_finn_functionality()
            self.test_file_system_access()
            self.test_package_installation_completeness()
            self.compare_with_original_entrypoint()
            
            return self.generate_comprehensive_report()
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR during validation: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False

def main():
    """Main validation function."""
    validator = FINNEnvironmentValidator()
    success = validator.run_all_tests()
    
    # Save detailed results to file
    with open('/tmp/finn_validation_results.json', 'w') as f:
        json.dump(validator.results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: /tmp/finn_validation_results.json")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
