#!/usr/bin/env python3
"""
FINN Unified Codegen - Test Runner
Convenient wrapper for running individual test phases or the complete suite.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

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

def check_environment():
    """Enhanced environment checking with dependency validation"""
    # Check if we're in FINN project root
    if not Path('src/finn/codegen').exists():
        print_status('ERROR', 'Not in FINN project root directory')
        print_status('INFO', 'Please run from the directory containing src/finn/codegen/')
        return False
    
    # Check Python version
    if sys.version_info < (3, 7):  # Updated requirement
        print_status('ERROR', f'Python 3.7+ required, found {sys.version_info.major}.{sys.version_info.minor}')
        return False
    
    # Check critical dependencies
    critical_deps = ['psutil', 'jinja2', 'pathlib']
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print_status('ERROR', f'Missing required dependencies: {", ".join(missing_deps)}')
        print_status('INFO', f'Install with: pip install {" ".join(missing_deps)}')
        return False
    
    return True

def run_test_script(script_name, description):
    """Enhanced script runner with permission validation and output capture"""
    script_path = Path('scripts') / script_name
    
    if not script_path.exists():
        print_status('ERROR', f'Test script not found: {script_path}')
        return False
    
    # Check if script is readable
    if not os.access(script_path, os.R_OK):
        print_status('ERROR', f'Test script not readable: {script_path}')
        return False
    
    # For Python scripts, check basic syntax
    try:
        with open(script_path, 'r') as f:
            first_line = f.readline().strip()
            if not (first_line.startswith('#!') and 'python' in first_line):
                print_status('WARNING', f'Script may not be a Python script: {script_path}')
    except Exception:
        pass  # Not critical, continue anyway
    
    print_status('INFO', f'Running {description}...')
    print(f"{'=' * 60}")
    
    try:
        result = subprocess.run([sys.executable, str(script_path)],
                              cwd=Path.cwd(),
                              timeout=600,  # 10 minute timeout per test
                              capture_output=True,
                              text=True)
        
        # Always show output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print_status('SUCCESS', f'{description} completed successfully')
            return True
        else:
            print_status('ERROR', f'{description} failed with exit code {result.returncode}')
            return False
            
    except subprocess.TimeoutExpired:
        print_status('ERROR', f'{description} timed out after 10 minutes')
        return False
    except Exception as e:
        print_status('ERROR', f'Failed to run {description}: {e}')
        return False

def run_complete_suite():
    """Run the complete robustness test suite using Python"""
    # Run each test phase individually in sequence
    test_phases = [
        ('test_core_framework.py', 'Core Framework Validation'),
        ('discover_finn_tests.py', 'FINN Integration Discovery'),
        ('test_components.py', 'Component Testing'),
        ('test_performance.py', 'Performance & Stress Testing'),
        ('test_integration.py', 'Integration Testing')
    ]
    
    print_status('INFO', 'Running complete robustness test suite...')
    print(f"{'=' * 60}")
    
    total_phases = len(test_phases)
    passed_phases = 0
    
    for script_name, description in test_phases:
        print(f"\n🎯 === {description} ===")
        if run_test_script(script_name, description):
            passed_phases += 1
        else:
            print_status('WARNING', f'{description} failed, continuing with remaining tests')
    
    print(f"\n📊 Complete Suite Results: {passed_phases}/{total_phases} phases passed")
    return passed_phases == total_phases

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description='FINN Unified Codegen Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Run complete test suite
  %(prog)s --core                   # Run core framework tests only
  %(prog)s --discover               # Run FINN integration discovery
  %(prog)s --components             # Run component tests
  %(prog)s --performance            # Run performance tests
  %(prog)s --integration            # Run integration tests
  %(prog)s --list                   # List available test phases
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--all', action='store_true', 
                           help='Run complete robustness test suite')
    test_group.add_argument('--core', action='store_true',
                           help='Run core framework validation tests')
    test_group.add_argument('--discover', action='store_true',
                           help='Run FINN integration discovery')
    test_group.add_argument('--components', action='store_true',
                           help='Run component testing')
    test_group.add_argument('--performance', action='store_true',
                           help='Run performance and stress testing')
    test_group.add_argument('--integration', action='store_true',
                           help='Run integration testing')
    test_group.add_argument('--list', action='store_true',
                           help='List available test phases')
    
    # Additional options
    parser.add_argument('--check-env', action='store_true',
                       help='Check environment setup only')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    print("🧪 FINN Unified Codegen Test Runner")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return 1
    
    if args.check_env:
        print_status('SUCCESS', 'Environment check passed')
        return 0
    
    # Handle list option
    if args.list:
        print("\n📋 Available Test Phases:")
        print("-" * 30)
        phases = [
            ("--core", "Core Framework Validation", "test_core_framework.py"),
            ("--discover", "FINN Integration Discovery", "discover_finn_tests.py"), 
            ("--components", "Component Testing", "test_components.py"),
            ("--performance", "Performance & Stress Testing", "test_performance.py"),
            ("--integration", "Integration Testing", "test_integration.py"),
            ("--all", "Complete Test Suite", "test_codegen_robustness.sh")
        ]
        
        for flag, description, script in phases:
            print(f"  {flag:<15} {description:<30} ({script})")
        
        return 0
    
    # Run selected tests
    success = True
    
    if args.all:
        success = run_complete_suite()
    elif args.core:
        success = run_test_script('test_core_framework.py', 'Core Framework Validation')
    elif args.discover:
        success = run_test_script('discover_finn_tests.py', 'FINN Integration Discovery')
    elif args.components:
        success = run_test_script('test_components.py', 'Component Testing')
    elif args.performance:
        success = run_test_script('test_performance.py', 'Performance & Stress Testing')
    elif args.integration:
        success = run_test_script('test_integration.py', 'Integration Testing')
    
    # Final result
    print("\n" + "=" * 60)
    if success:
        print_status('SUCCESS', 'Test execution completed successfully')
        return 0
    else:
        print_status('ERROR', 'Test execution failed')
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_status('WARNING', 'Test execution interrupted by user')
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print_status('ERROR', f'Unexpected error: {e}')
        sys.exit(1)