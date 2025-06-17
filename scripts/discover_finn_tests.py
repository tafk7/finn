#!/usr/bin/env python3
"""
FINN Unified Codegen - FINN Integration Discovery
Discovers existing FINN test files and components for integration testing.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def print_status(status, message):
    """Print colored status messages"""
    colors = {
        'INFO': '\033[0;34m',
        'SUCCESS': '\033[0;32m',
        'WARNING': '\033[1;33m',
        'ERROR': '\033[0;31m',
        'NC': '\033[0m'
    }
    
    icons = {
        'INFO': 'ℹ️ ',
        'SUCCESS': '✅',
        'WARNING': '⚠️ ',
        'ERROR': '❌'
    }
    
    print(f"{colors.get(status, '')}{icons.get(status, '')}{message}{colors['NC']}")

def discover_test_files():
    """Discover all test files in the FINN project"""
    print_status('INFO', 'Discovering FINN test files...')
    
    test_files = {
        'all_tests': [],
        'transform_tests': [],
        'hls_tests': [],
        'custom_op_tests': [],
        'fpgadataflow_tests': []
    }
    
    # Find all test files
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                full_path = os.path.join(root, file)
                test_files['all_tests'].append(full_path)
                
                # Categorize tests
                if 'transform' in file.lower() or 'convert' in file.lower():
                    test_files['transform_tests'].append(full_path)
                    
                if 'hls' in file.lower():
                    test_files['hls_tests'].append(full_path)
                    
                if 'custom_op' in root or 'customop' in file.lower():
                    test_files['custom_op_tests'].append(full_path)
                    
                if 'fpgadataflow' in root:
                    test_files['fpgadataflow_tests'].append(full_path)
    
    # Report findings
    print(f"📄 Found {len(test_files['all_tests'])} total test files")
    print(f"🔄 Found {len(test_files['transform_tests'])} transformation test files")
    print(f"🏗️  Found {len(test_files['hls_tests'])} HLS test files")
    print(f"⚙️  Found {len(test_files['custom_op_tests'])} custom operation test files")
    print(f"💾 Found {len(test_files['fpgadataflow_tests'])} FPGA dataflow test files")
    
    return test_files

def discover_finn_operations():
    """Discover available FINN operations"""
    print_status('INFO', 'Discovering FINN operations...')
    
    operations = {
        'found_operations': [],
        'importable_operations': []
    }
    
    # Look for custom operations in source code
    custom_op_dir = Path('src/finn/custom_op')
    if custom_op_dir.exists():
        for py_file in custom_op_dir.rglob('*.py'):
            if py_file.name != '__init__.py':
                rel_path = py_file.relative_to(Path('src'))
                module_path = str(rel_path.with_suffix('')).replace('/', '.')
                operations['found_operations'].append(module_path)
    
    # Try to import common FINN operations
    common_ops = [
        'finn.custom_op.fpgadataflow.matrixvectoractivation.MatrixVectorActivation',
        'finn.custom_op.fpgadataflow.thresholding.Thresholding',
        'finn.custom_op.fpgadataflow.streamingdatawidthconverter.StreamingDataWidthConverter',
        'finn.custom_op.fpgadataflow.convolution.Conv_Batch',
        'finn.custom_op.fpgadataflow.addstreams.AddStreams'
    ]
    
    for op_path in common_ops:
        try:
            module_path, class_name = op_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            op_class = getattr(module, class_name)
            operations['importable_operations'].append((op_path, op_class))
            print_status('SUCCESS', f'Successfully imported {class_name}')
        except ImportError as e:
            print_status('WARNING', f'Could not import {op_path}: {e}')
        except Exception as e:
            print_status('WARNING', f'Error with {op_path}: {e}')
    
    print(f"🔍 Found {len(operations['found_operations'])} operation modules")
    print(f"✅ Successfully imported {len(operations['importable_operations'])} operations")
    
    return operations

def test_pytest_availability():
    """Test if pytest is available and working"""
    print_status('INFO', 'Testing pytest availability...')
    
    try:
        result = subprocess.run(['python', '-m', 'pytest', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_status('SUCCESS', f'pytest available: {result.stdout.strip()}')
            return True
        else:
            print_status('WARNING', f'pytest available but returned error: {result.stderr}')
            return False
    except subprocess.TimeoutExpired:
        print_status('WARNING', 'pytest command timed out')
        return False
    except Exception as e:
        print_status('WARNING', f'pytest not available: {e}')
        return False

def test_sample_test_file(test_files):
    """Test if we can collect tests from a sample file"""
    print_status('INFO', 'Testing sample test file collection...')
    
    if not test_files['all_tests']:
        print_status('WARNING', 'No test files found to test')
        return False
    
    # Try the first test file
    sample_test = test_files['all_tests'][0]
    print(f"Testing collection from: {sample_test}")
    
    try:
        result = subprocess.run(['python', '-m', 'pytest', sample_test, '--collect-only'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and 'collected' in result.stdout:
            print_status('SUCCESS', f'Successfully collected tests from {sample_test}')
            return True
        else:
            print_status('WARNING', f'Could not collect from {sample_test}: {result.stderr[:200]}')
            return False
    except subprocess.TimeoutExpired:
        print_status('WARNING', f'Test collection from {sample_test} timed out')
        return False
    except Exception as e:
        print_status('WARNING', f'Error testing {sample_test}: {e}')
        return False

def discover_finn_utilities():
    """Discover FINN utility modules"""
    print_status('INFO', 'Discovering FINN utilities...')
    
    utilities = {
        'found_utils': [],
        'importable_utils': []
    }
    
    # Common FINN utility modules
    common_utils = [
        'finn.util.basic',
        'finn.util.data_packing',
        'finn.util.pytorch',
        'finn.analysis.fpgadataflow',
        'finn.transformation.fpgadataflow',
        'finn.builder'
    ]
    
    for util_path in common_utils:
        try:
            module = __import__(util_path, fromlist=[''])
            utilities['importable_utils'].append(util_path)
            print_status('SUCCESS', f'Successfully imported {util_path}')
        except ImportError as e:
            print_status('WARNING', f'Could not import {util_path}: {e}')
        except Exception as e:
            print_status('WARNING', f'Error with {util_path}: {e}')
    
    # Look for utility modules in source
    util_dirs = [
        Path('src/finn/util'),
        Path('src/finn/analysis'),
        Path('src/finn/transformation')
    ]
    
    for util_dir in util_dirs:
        if util_dir.exists():
            for py_file in util_dir.rglob('*.py'):
                if py_file.name != '__init__.py':
                    rel_path = py_file.relative_to(Path('src'))
                    module_path = str(rel_path.with_suffix('')).replace('/', '.')
                    utilities['found_utils'].append(module_path)
    
    print(f"🛠️  Found {len(utilities['found_utils'])} utility modules")
    print(f"✅ Successfully imported {len(utilities['importable_utils'])} utilities")
    
    return utilities

def create_integration_report(test_files, operations, utilities):
    """Create a detailed integration report"""
    print_status('INFO', 'Creating integration report...')
    
    report_path = 'scripts/finn_integration_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("FINN Integration Discovery Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Test files section
        f.write("TEST FILES DISCOVERED\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total test files: {len(test_files['all_tests'])}\n")
        f.write(f"Transformation tests: {len(test_files['transform_tests'])}\n")
        f.write(f"HLS tests: {len(test_files['hls_tests'])}\n")
        f.write(f"Custom operation tests: {len(test_files['custom_op_tests'])}\n")
        f.write(f"FPGA dataflow tests: {len(test_files['fpgadataflow_tests'])}\n\n")
        
        # Sample test files
        f.write("SAMPLE TEST FILES (first 10):\n")
        for test_file in test_files['all_tests'][:10]:
            f.write(f"  - {test_file}\n")
        f.write("\n")
        
        # Operations section
        f.write("FINN OPERATIONS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Found operation modules: {len(operations['found_operations'])}\n")
        f.write(f"Successfully imported: {len(operations['importable_operations'])}\n\n")
        
        f.write("IMPORTABLE OPERATIONS:\n")
        for op_path, op_class in operations['importable_operations']:
            f.write(f"  ✅ {op_path}\n")
        f.write("\n")
        
        # Utilities section
        f.write("FINN UTILITIES\n")
        f.write("-" * 30 + "\n")
        f.write(f"Found utility modules: {len(utilities['found_utils'])}\n")
        f.write(f"Successfully imported: {len(utilities['importable_utils'])}\n\n")
        
        f.write("IMPORTABLE UTILITIES:\n")
        for util_path in utilities['importable_utils']:
            f.write(f"  ✅ {util_path}\n")
        f.write("\n")
        
        # Recommendations
        f.write("INTEGRATION RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        if operations['importable_operations']:
            f.write("✅ FINN operations are available for codegen integration testing\n")
        else:
            f.write("⚠️  No FINN operations could be imported - may need environment setup\n")
            
        if test_files['all_tests']:
            f.write("✅ Test files are available for robustness validation\n")
        else:
            f.write("⚠️  No test files found - may not be in correct directory\n")
            
        if utilities['importable_utils']:
            f.write("✅ FINN utilities are available for extended testing\n")
        else:
            f.write("⚠️  FINN utilities not available - basic testing only\n")
    
    print_status('SUCCESS', f'Integration report saved to {report_path}')
    return report_path

def main():
    """Run FINN integration discovery"""
    print("🔍 FINN Unified Codegen - Integration Discovery")
    print("==============================================")
    
    success_count = 0
    total_checks = 5
    
    # Discover test files
    print("\n📄 Discovering Test Files")
    print("-" * 30)
    test_files = discover_test_files()
    if test_files['all_tests']:
        success_count += 1
    
    # Test pytest
    print("\n🧪 Testing Pytest")
    print("-" * 30)
    if test_pytest_availability():
        success_count += 1
    
    # Test sample file
    print("\n📋 Testing Sample Collection")
    print("-" * 30)
    if test_sample_test_file(test_files):
        success_count += 1
    
    # Discover operations
    print("\n⚙️  Discovering FINN Operations")
    print("-" * 30)
    operations = discover_finn_operations()
    if operations['importable_operations']:
        success_count += 1
    
    # Discover utilities
    print("\n🛠️  Discovering FINN Utilities")
    print("-" * 30)
    utilities = discover_finn_utilities()
    if utilities['importable_utils']:
        success_count += 1
    
    # Create report
    print("\n📊 Creating Integration Report")
    print("-" * 30)
    create_integration_report(test_files, operations, utilities)
    
    # Summary
    print("\n📈 Integration Discovery Summary")
    print("=" * 40)
    print(f"Successful checks: {success_count}/{total_checks}")
    
    if success_count >= 3:  # At least 3 out of 5 checks should pass
        print_status('SUCCESS', 'FINN integration environment is ready for testing')
        return True
    else:
        print_status('WARNING', 'FINN integration environment has limitations')
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_status('ERROR', f'Integration discovery failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)