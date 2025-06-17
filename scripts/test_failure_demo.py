#!/usr/bin/env python3
"""
FINN Test Failure Demonstration
Demonstrates how the redesigned test suite fails hard outside Docker environment.
This script should be run outside FINN Docker to show proper failure behavior.
"""

import sys
import os
import subprocess
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

def check_docker_environment():
    """Check if we're running in Docker"""
    docker_indicators = [
        '/proc/1/cgroup',
        '/.dockerenv',
    ]
    
    docker_detected = any(os.path.exists(indicator) for indicator in docker_indicators)
    
    if not docker_detected:
        docker_env_vars = ['FINN_DOCKER', 'FINN_ROOT', 'VIVADO_PATH', 'VITIS_HLS_PATH']
        docker_env_detected = any(os.getenv(var) for var in docker_env_vars)
        docker_detected = docker_env_detected
    
    return docker_detected

def run_test_and_capture_failure(test_script):
    """Run a test script and capture its failure"""
    print_status('INFO', f'Running {test_script}...')
    
    try:
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent
        )
        
        print_status('INFO', f'Exit code: {result.returncode}')
        
        if result.returncode == 0:
            print_status('WARNING', f'{test_script} unexpectedly PASSED')
            print("STDOUT:")
            print(result.stdout)
        else:
            print_status('SUCCESS', f'{test_script} correctly FAILED as expected')
            print("STDERR (last 10 lines):")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-10:]:
                print(f"  {line}")
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print_status('ERROR', f'{test_script} timed out')
        return -1
    except Exception as e:
        print_status('ERROR', f'Failed to run {test_script}: {e}')
        return -1

def main():
    """Demonstrate test failures outside Docker"""
    print("🧪 FINN Test Failure Demonstration")
    print("==================================")
    print("This script demonstrates that the redesigned test suite")
    print("fails hard when not running in a proper FINN Docker environment.")
    print()
    
    # Check current environment
    in_docker = check_docker_environment()
    
    if in_docker:
        print_status('WARNING', 'Running inside Docker environment')
        print_status('WARNING', 'Tests may pass unexpectedly if FINN dependencies are available')
        print()
    else:
        print_status('SUCCESS', 'Running outside Docker environment')
        print_status('SUCCESS', 'Tests should fail as expected')
        print()
    
    # List of strict tests to run
    strict_tests = [
        'test_environment.py',
        'test_core_framework.py'
    ]
    
    print("Running strict tests that should fail outside Docker:")
    print("=" * 60)
    
    failure_count = 0
    for test in strict_tests:
        test_path = Path(__file__).parent / test
        if test_path.exists():
            exit_code = run_test_and_capture_failure(str(test_path))
            if exit_code != 0:
                failure_count += 1
            print("-" * 60)
        else:
            print_status('ERROR', f'Test file {test} not found')
    
    print()
    print("Summary:")
    print("=" * 30)
    
    if not in_docker:
        if failure_count == len(strict_tests):
            print_status('SUCCESS', '🎉 All tests correctly FAILED outside Docker')
            print_status('SUCCESS', '✅ Strict validation is working properly')
            print_status('SUCCESS', '🚫 No mock fallbacks allowed tests to pass')
        else:
            print_status('WARNING', f'Only {failure_count}/{len(strict_tests)} tests failed')
            print_status('WARNING', 'Some tests may have unexpected fallbacks')
    else:
        print_status('INFO', 'Cannot properly demonstrate failures inside Docker')
        print_status('INFO', 'Tests may pass if FINN dependencies are available')
    
    print()
    print("Expected behavior:")
    print("- Outside Docker: All tests should FAIL with clear error messages")
    print("- Inside FINN Docker: All tests should PASS with real FINN operations")
    print("- No mock objects or fallbacks should be used anywhere")

if __name__ == '__main__':
    main()