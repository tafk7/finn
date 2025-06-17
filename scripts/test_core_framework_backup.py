#!/usr/bin/env python3
"""
FINN Unified Codegen - Core Framework Validation (STRICT MODE)
Tests the fundamental components and imports of the unified codegen framework.
REQUIRES REAL FINN DEPENDENCIES - NO MOCK FALLBACKS
This test suite FAILS HARD outside proper FINN Docker environment.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import strict environment validation
from test_environment import require_finn_environment, FinnEnvironmentError

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

def test_core_imports():
    """Test core framework component imports"""
    print_status('INFO', 'Testing core framework imports...')
    
    tests = []
    
    # Test 1: Core components
    try:
        from finn.codegen import ModernHLSGenerator, ModernRTLGenerator
        print_status('SUCCESS', 'Core generators import successful')
        tests.append(True)
    except ImportError as e:
        print_status('ERROR', f'Core generators import failed: {e}')
        tests.append(False)
    
    # Test 2: Template engine
    try:
        from finn.codegen import TemplateEngine
        print_status('SUCCESS', 'Template engine import successful')
        tests.append(True)
    except ImportError as e:
        print_status('ERROR', f'Template engine import failed: {e}')
        tests.append(False)
    
    # Test 3: File manager
    try:
        from finn.codegen import FileManager
        print_status('SUCCESS', 'File manager import successful')
        tests.append(True)
    except ImportError as e:
        print_status('ERROR', f'File manager import failed: {e}')
        tests.append(False)
    
    # Test 4: Library resolver
    try:
        from finn.codegen import LibraryResolver
        print_status('SUCCESS', 'Library resolver import successful')
        tests.append(True)
    except ImportError as e:
        print_status('ERROR', f'Library resolver import failed: {e}')
        tests.append(False)
    
    # Test 5: Base classes
    try:
        from finn.codegen.base import BaseCodeGenerator
        print_status('SUCCESS', 'Base classes import successful')
        tests.append(True)
    except ImportError as e:
        print_status('ERROR', f'Base classes import failed: {e}')
        tests.append(False)
    
    return tests

def create_real_finn_mvau_operation():
    """Create a real FINN MVAU operation - NO MOCK OBJECTS ALLOWED"""
    try:
        from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
        import onnx
        from onnx import helper
        
        # Create proper ONNX node with all required attributes
        onnx_node = helper.make_node(
            'MatrixVectorActivation',
            inputs=['input0', 'weights'],
            outputs=['output0'],
            name='real_test_mvau',
            domain='finn.custom_op.fpgadataflow',
            MW=32,
            MH=32,
            PE=4,
            SIMD=2,
            mem_mode='internal_decoupled',
            runtime_writeable_weights=0,
            inputDataType='INT8',
            outputDataType='INT8',
            weightDataType='INT8',
            ActVal=0,
            binaryXnorMode=0,
            noActivation=0,
            backend='hls',
            resType='auto',
            numInputVectors=1
        )
        
        # Create the actual FINN operation
        finn_op = MVAU(onnx_node)
        return finn_op
        
    except ImportError as e:
        raise FinnEnvironmentError(
            f"Cannot create real FINN MVAU operation: {e}. "
            "This test requires real FINN dependencies in a proper Docker environment."
        )
    except Exception as e:
        raise FinnEnvironmentError(
            f"Failed to create real FINN MVAU operation: {e}. "
            "This indicates a problem with the FINN installation."
        )

def create_real_finn_thresholding_operation():
    """Create a real FINN Thresholding operation for additional testing"""
    try:
        from finn.custom_op.fpgadataflow.thresholding import Thresholding
        import onnx
        from onnx import helper
        
        # Create proper ONNX node for thresholding
        onnx_node = helper.make_node(
            'Thresholding',
            inputs=['input0'],
            outputs=['output0'],
            name='real_test_thresholding',
            domain='finn.custom_op.fpgadataflow',
            PE=4,
            inputDataType='INT8',
            outputDataType='INT8',
            numSteps=16,
            backend='hls'
        )
        
        # Create the actual FINN operation
        finn_op = Thresholding(onnx_node)
        return finn_op
        
    except ImportError as e:
        raise FinnEnvironmentError(
            f"Cannot create real FINN Thresholding operation: {e}. "
            "This test requires real FINN dependencies in a proper Docker environment."
        )
    except Exception as e:
        raise FinnEnvironmentError(
            f"Failed to create real FINN Thresholding operation: {e}. "
            "This indicates a problem with the FINN installation."
        )

def test_basic_functionality():
    """Test basic functionality with REAL FINN operations ONLY"""
    print_status('INFO', 'Testing basic functionality with REAL FINN operations...')
    
    tests = []
    
    # Test 1: HLS Generator with REAL FINN MVAU operation - NO FALLBACKS
    try:
        from finn.codegen import ModernHLSGenerator
        print_status('INFO', 'Creating real FINN MVAU operation for HLS testing...')
        
        # This WILL FAIL if not in proper FINN environment
        real_op = create_real_finn_mvau_operation()
        real_op.set_nodeattr('backend', 'hls')
        
        hls_gen = ModernHLSGenerator(real_op)
        context = hls_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
        print_status('SUCCESS', f'HLS generator with REAL FINN operation: {len(context)} context keys')
        tests.append(True)
        
    except FinnEnvironmentError as e:
        print_status('ERROR', f'FINN Environment Error in HLS test: {e}')
        raise  # Re-raise to fail hard
    except Exception as e:
        print_status('ERROR', f'HLS generator with real FINN operation failed: {e}')
        raise FinnEnvironmentError(f"HLS generator test failed with real FINN operation: {e}")
    
    # Test 2: RTL Generator with REAL FINN MVAU operation - NO FALLBACKS
    try:
        from finn.codegen import ModernRTLGenerator
        print_status('INFO', 'Creating real FINN MVAU operation for RTL testing...')
        
        # This WILL FAIL if not in proper FINN environment
        real_op = create_real_finn_mvau_operation()
        real_op.set_nodeattr('backend', 'rtl')
        
        rtl_gen = ModernRTLGenerator(real_op)
        context = rtl_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
        print_status('SUCCESS', f'RTL generator with REAL FINN operation: {len(context)} context keys')
        tests.append(True)
        
    except FinnEnvironmentError as e:
        print_status('ERROR', f'FINN Environment Error in RTL test: {e}')
        raise  # Re-raise to fail hard
    except Exception as e:
        print_status('ERROR', f'RTL generator with real FINN operation failed: {e}')
        raise FinnEnvironmentError(f"RTL generator test failed with real FINN operation: {e}")
    
    # Test 3: Additional real FINN operation - Thresholding
    try:
        from finn.codegen import ModernHLSGenerator
        print_status('INFO', 'Testing with real FINN Thresholding operation...')
        
        # This WILL FAIL if not in proper FINN environment
        thresh_op = create_real_finn_thresholding_operation()
        
        hls_gen = ModernHLSGenerator(thresh_op)
        context = hls_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
        print_status('SUCCESS', f'Thresholding operation HLS generation: {len(context)} context keys')
        tests.append(True)
        
    except FinnEnvironmentError as e:
        print_status('ERROR', f'FINN Environment Error in Thresholding test: {e}')
        raise  # Re-raise to fail hard
    except Exception as e:
        print_status('ERROR', f'Thresholding operation test failed: {e}')
        raise FinnEnvironmentError(f"Thresholding operation test failed: {e}")
    
    # Test 4: Template Engine (STRICT)
    try:
        from finn.codegen import TemplateEngine
        engine = TemplateEngine()
        result = engine.render_string('Test {{ value }}', {'value': 'SUCCESS'})
        assert 'Test SUCCESS' in result
        print_status('SUCCESS', 'Template engine rendering working')
        tests.append(True)
    except Exception as e:
        print_status('ERROR', f'Template engine test failed: {e}')
        raise FinnEnvironmentError(f"Template engine test failed: {e}")
    
    # Test 5: File Manager (STRICT)
    try:
        import tempfile
        from finn.codegen import FileManager
        
        with tempfile.TemporaryDirectory() as tmp:
            fm = FileManager(tmp)
            path = fm.write_file('test.txt', 'Hello World')
            content = fm.read_file(path)
            assert content == 'Hello World'
        print_status('SUCCESS', 'File manager operations working')
        tests.append(True)
    except Exception as e:
        print_status('ERROR', f'File manager test failed: {e}')
        raise FinnEnvironmentError(f"File manager test failed: {e}")
    
    # Test 6: Library Resolver (STRICT)
    try:
        from finn.codegen import LibraryResolver
        resolver = LibraryResolver()
        libs = resolver.list_libraries()
        print_status('SUCCESS', f'Library resolver found {len(libs)} libraries')
        tests.append(True)
    except Exception as e:
        print_status('ERROR', f'Library resolver test failed: {e}')
        raise FinnEnvironmentError(f"Library resolver test failed: {e}")
    
    return tests

def test_error_handling():
    """Test error handling and edge cases (STRICT MODE)"""
    print_status('INFO', 'Testing error handling (STRICT MODE)...')
    
    tests = []
    
    # Test 1: Invalid template (STRICT)
    try:
        from finn.codegen import TemplateEngine
        engine = TemplateEngine()
        try:
            engine.render_string('{{ invalid_syntax', {})
            print_status('ERROR', 'Template engine should have failed on invalid syntax')
            raise FinnEnvironmentError('Template engine failed to detect invalid syntax')
        except FinnEnvironmentError:
            raise  # Re-raise FINN environment errors
        except Exception:
            print_status('SUCCESS', 'Template engine correctly handles invalid syntax')
            tests.append(True)
    except FinnEnvironmentError:
        raise  # Re-raise FINN environment errors
    except Exception as e:
        print_status('ERROR', f'Template engine error handling test failed: {e}')
        raise FinnEnvironmentError(f"Template engine error handling test failed: {e}")
    
    # Test 2: File operations with invalid paths (STRICT)
    try:
        from finn.codegen import FileManager
        fm = FileManager('/nonexistent/path')
        try:
            fm.read_file('nonexistent.txt')
            print_status('ERROR', 'File manager should have failed on nonexistent file')
            raise FinnEnvironmentError('File manager failed to detect nonexistent file')
        except FinnEnvironmentError:
            raise  # Re-raise FINN environment errors
        except Exception:
            print_status('SUCCESS', 'File manager correctly handles nonexistent files')
            tests.append(True)
    except FinnEnvironmentError:
        raise  # Re-raise FINN environment errors
    except Exception as e:
        print_status('ERROR', f'File manager error handling test failed: {e}')
        raise FinnEnvironmentError(f"File manager error handling test failed: {e}")
    
    # Test 3: Generator with None operation (STRICT)
    try:
        from finn.codegen import ModernHLSGenerator
        
        try:
            gen = ModernHLSGenerator(None)
            gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
            print_status('ERROR', 'HLS generator should have failed on None operation')
            raise FinnEnvironmentError('HLS generator failed to detect None operation')
        except FinnEnvironmentError:
            raise  # Re-raise FINN environment errors
        except Exception:
            print_status('SUCCESS', 'HLS generator correctly handles None operations')
            tests.append(True)
    except FinnEnvironmentError:
        raise  # Re-raise FINN environment errors
    except Exception as e:
        print_status('ERROR', f'HLS generator error handling test failed: {e}')
        raise FinnEnvironmentError(f"HLS generator error handling test failed: {e}")
    
    return tests

def main():
    """Run all core framework tests with STRICT validation"""
    print("🧪 FINN Unified Codegen - Core Framework Validation (STRICT MODE)")
    print("================================================================")
    print("🚫 NO MOCK FALLBACKS - REQUIRES REAL FINN DOCKER ENVIRONMENT")
    print("================================================================")
    
    try:
        # STRICT: Validate environment first - FAIL HARD if not proper FINN environment
        print_status('INFO', 'Validating FINN environment (STRICT MODE)...')
        require_finn_environment()
        print_status('SUCCESS', 'FINN environment validation passed')
        
    except FinnEnvironmentError as e:
        print_status('ERROR', f'FINN environment validation failed: {e}')
        print_status('ERROR', 'This test suite requires a properly configured FINN Docker environment')
        print_status('ERROR', 'Please run inside FINN Docker container with all dependencies')
        return False
    
    all_tests = []
    
    try:
        # Run test suites - ALL MUST PASS IN STRICT MODE
        print("\n📦 Testing Core Imports (STRICT)")
        print("-" * 40)
        import_tests = test_core_imports()
        all_tests.extend(import_tests)
        
        # STRICT: All imports must pass
        if not all(import_tests):
            print_status('ERROR', 'CRITICAL: Core imports failed in strict mode')
            raise FinnEnvironmentError('Core imports failed - environment not suitable')
        
        print("\n⚙️  Testing Basic Functionality (REAL FINN OPS ONLY)")
        print("-" * 50)
        functionality_tests = test_basic_functionality()
        all_tests.extend(functionality_tests)
        
        print("\n🛡️  Testing Error Handling (STRICT)")
        print("-" * 40)
        error_tests = test_error_handling()
        all_tests.extend(error_tests)
        
    except FinnEnvironmentError as e:
        print_status('ERROR', f'STRICT MODE FAILURE: {e}')
        return False
    except Exception as e:
        print_status('ERROR', f'UNEXPECTED FAILURE in strict mode: {e}')
        print_status('ERROR', 'This indicates a serious problem with the FINN environment')
        return False
    
    # Summary
    total_tests = len(all_tests)
    passed_tests = sum(all_tests)
    failed_tests = total_tests - passed_tests
    
    print("\n📊 STRICT Core Framework Test Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    # STRICT SUCCESS CRITERIA: ALL tests must pass
    if failed_tests == 0:
        print_status('SUCCESS', '🎉 ALL STRICT CORE FRAMEWORK TESTS PASSED!')
        print_status('SUCCESS', '✅ Real FINN operations working perfectly')
        print_status('SUCCESS', '🐳 FINN Docker environment is properly configured')
        return True
    else:
        print_status('ERROR', f'❌ STRICT MODE FAILURE: {failed_tests} tests failed')
        print_status('ERROR', '🚫 In strict mode, ALL tests must pass')
        print_status('ERROR', '🔧 This indicates issues with FINN dependencies or environment')
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_status('ERROR', f'Core framework test suite failed: {e}')
        traceback.print_exc()
        sys.exit(1)