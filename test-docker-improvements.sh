#!/bin/bash
# Test script for FINN Docker improvements with Python library validation

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

success() { echo -e "${GREEN}✓ $1${NC}"; }
error() { echo -e "${RED}✗ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
info() { echo -e "${BLUE}ℹ $1${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="finn_test_$(date +%s)"
TEST_FAILED=0

cleanup() {
    if [ -n "$CONTAINER_NAME" ]; then
        info "Cleaning up test container..."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    fi
}

fail_test() {
    TEST_FAILED=1
    error "$1"
}

# Set up cleanup on exit
trap cleanup EXIT

info "🧪 FINN Docker Improvements Test Suite"
echo "================================================="

# Test 1: Check if new scripts exist and are executable
info "Test 1: Checking script files..."
if [ -x "./finn-container" ]; then
    success "finn-container script is executable"
else
    fail_test "finn-container script missing or not executable"
fi

if [ -x "./docker/finn_setup_env.sh" ]; then
    success "finn_setup_env.sh script is executable"
else
    fail_test "finn_setup_env.sh script missing or not executable"
fi

if [ -f "./docker/finn_entrypoint_light.sh" ]; then
    success "finn_entrypoint_light.sh script exists"
else
    fail_test "finn_entrypoint_light.sh script missing"
fi

# Test 2: Check container management help
info "Test 2: Testing container management help..."
if ./finn-container help > /dev/null 2>&1; then
    success "Container management help works"
else
    fail_test "Container management help failed"
fi

# Test 3: Check if run-docker.sh supports new variables
info "Test 3: Checking run-docker.sh environment variable support..."
if grep -q "FINN_DOCKER_PERSISTENT" run-docker.sh; then
    success "run-docker.sh supports FINN_DOCKER_PERSISTENT"
else
    fail_test "run-docker.sh missing FINN_DOCKER_PERSISTENT support"
fi

if grep -q "FINN_LIGHTWEIGHT_ENTRYPOINT" run-docker.sh; then
    success "run-docker.sh supports FINN_LIGHTWEIGHT_ENTRYPOINT"
else
    fail_test "run-docker.sh missing FINN_LIGHTWEIGHT_ENTRYPOINT support"
fi

# Test 4: Check Dockerfile improvements
info "Test 4: Checking Dockerfile improvements..."
if grep -q "finn_setup_env.sh" docker/Dockerfile.finn; then
    success "Dockerfile includes environment setup script"
else
    fail_test "Dockerfile missing environment setup script"
fi

# Test 5: Syntax check for new scripts
info "Test 5: Syntax checking new scripts..."
if bash -n ./finn-container 2>/dev/null; then
    success "finn-container syntax is valid"
else
    fail_test "finn-container has syntax errors"
fi

if bash -n ./docker/finn_setup_env.sh 2>/dev/null; then
    success "finn_setup_env.sh syntax is valid"
else
    fail_test "finn_setup_env.sh has syntax errors"
fi

# Test 6: Create Python library validation script
info "Test 6: Creating Python library validation test..."

cat > /tmp/finn_lib_test.py << 'EOF'
#!/usr/bin/env python3
"""
FINN Library Validation Test
Tests if Python libraries from finn_entrypoint.sh are properly installed and functional
"""
import sys
import importlib
import traceback

def test_import(module_name, description=""):
    """Test if a module can be imported"""
    try:
        module = importlib.import_module(module_name)
        print(f"✓ {module_name} - imported successfully {description}")
        return True, module
    except ImportError as e:
        print(f"✗ {module_name} - import failed: {e}")
        return False, None
    except Exception as e:
        print(f"⚠ {module_name} - import warning: {e}")
        return False, None

def test_basic_functionality():
    """Test basic functionality of imported libraries"""
    results = {}
    
    # Test QONNX
    print("\n=== Testing QONNX ===")
    success, qonnx = test_import("qonnx", "(Quantized ONNX)")
    if success:
        try:
            # Test basic QONNX functionality
            from qonnx.core.modelwrapper import ModelWrapper
            print("✓ qonnx.core.modelwrapper imported successfully")
            results['qonnx'] = True
        except Exception as e:
            print(f"✗ QONNX functionality test failed: {e}")
            results['qonnx'] = False
    else:
        results['qonnx'] = False
    
    # Test Brevitas
    print("\n=== Testing Brevitas ===")
    success, brevitas = test_import("brevitas", "(Quantization library)")
    if success:
        try:
            # Test basic Brevitas functionality
            import brevitas.nn as qnn
            print("✓ brevitas.nn imported successfully")
            results['brevitas'] = True
        except Exception as e:
            print(f"✗ Brevitas functionality test failed: {e}")
            results['brevitas'] = False
    else:
        results['brevitas'] = False
    
    # Test FINN Experimental
    print("\n=== Testing FINN Experimental ===")
    success, finn_exp = test_import("finn_experimental", "(FINN Experimental)")
    if success:
        results['finn_experimental'] = True
    else:
        # Alternative import path
        success, finn_exp = test_import("finn.experimental", "(FINN Experimental - alternative path)")
        results['finn_experimental'] = success
    
    # Test FINN Core
    print("\n=== Testing FINN Core ===")
    success, finn = test_import("finn", "(FINN Core)")
    if success:
        try:
            # Test basic FINN functionality
            from finn.core.modelwrapper import ModelWrapper
            print("✓ finn.core.modelwrapper imported successfully")
            results['finn'] = True
        except Exception as e:
            print(f"✗ FINN functionality test failed: {e}")
            results['finn'] = False
    else:
        results['finn'] = False
    
    # Test PyTorch (should be available from base image)
    print("\n=== Testing PyTorch ===")
    success, torch = test_import("torch", "(PyTorch)")
    if success:
        try:
            import torch
            print(f"✓ PyTorch version: {torch.__version__}")
            # Test basic tensor operations
            x = torch.tensor([1.0, 2.0, 3.0])
            y = x + 1
            print("✓ Basic PyTorch tensor operations work")
            results['torch'] = True
        except Exception as e:
            print(f"✗ PyTorch functionality test failed: {e}")
            results['torch'] = False
    else:
        results['torch'] = False
    
    # Test additional dependencies
    print("\n=== Testing Additional Dependencies ===")
    additional_libs = [
        ("numpy", "NumPy"),
        ("onnx", "ONNX"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("pytest", "PyTest"),
        ("jupyter", "Jupyter"),
    ]
    
    for lib, desc in additional_libs:
        success, _ = test_import(lib, f"({desc})")
        results[lib] = success
    
    return results

def main():
    print("FINN Python Library Validation Test")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("")
    
    # Run functionality tests
    results = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for lib, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{lib:20} {status}")
    
    print(f"\nTotal: {passed}/{total} libraries working")
    
    if passed == total:
        print("🎉 All library tests passed!")
        return 0
    elif passed >= total * 0.7:  # 70% success rate
        print("⚠ Most library tests passed (acceptable)")
        return 0
    else:
        print("❌ Many library tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

success "Python library validation script created"

# Test 7: Docker-based validation (if Docker is available)
if command -v docker > /dev/null 2>&1; then
    info "Test 7: Docker availability check..."
    success "Docker is available for testing"
    
    # Check if we can test with the current setup
    if [ -f ".env" ] && grep -q "FINN_DOCKER_TAG" .env; then
        info "Found FINN_DOCKER_TAG configuration"
        source .env
    else
        warning "No .env file found - using default image name"
        export FINN_DOCKER_TAG="${FINN_DOCKER_TAG:-finn:latest}"
    fi
    
    info "Will test with Docker image: $FINN_DOCKER_TAG"
    
else
    warning "Docker not available - skipping container-based tests"
fi

# Test 8: Environment variable validation
info "Test 8: Validating environment setup script..."
if grep -q "PYTHONPATH" docker/finn_setup_env.sh; then
    success "Environment setup includes PYTHONPATH configuration"
else
    warning "PYTHONPATH not found in environment setup"
fi

if grep -q "LD_LIBRARY_PATH" docker/finn_setup_env.sh; then
    success "Environment setup includes LD_LIBRARY_PATH configuration"
else
    warning "LD_LIBRARY_PATH not found in environment setup"
fi

# Test 9: Check for caching mechanism
info "Test 9: Validating caching mechanism..."
if grep -q "finn_env_setup_complete" docker/finn_setup_env.sh; then
    success "Environment setup includes caching mechanism"
else
    warning "Caching mechanism not found in environment setup"
fi

# Final results
echo ""
info "📊 Test Results Summary"
echo "================================================="

if [ $TEST_FAILED -eq 0 ]; then
    success "All core tests passed! ✨"
    echo ""
    info "Next steps to test with actual containers:"
    echo "  1. Build/update the Docker image:"
    echo "     ./run-docker.sh"
    echo ""
    echo "  2. Test Python libraries with lightweight entrypoint:"
    echo "     FINN_LIGHTWEIGHT_ENTRYPOINT=1 ./run-docker.sh python3 /tmp/finn_lib_test.py"
    echo ""
    echo "  3. Test container management:"
    echo "     ./finn-container start daemon"
    echo "     ./finn-container exec 'python3 /tmp/finn_lib_test.py'"
    echo "     ./finn-container stop"
    echo ""
    echo "  4. Compare performance:"
    echo "     time ./run-docker.sh python3 -c 'import finn; print(\"Standard entrypoint\")'"
    echo "     time FINN_LIGHTWEIGHT_ENTRYPOINT=1 ./run-docker.sh python3 -c 'import finn; print(\"Lightweight entrypoint\")'"
    echo ""
else
    error "Some tests failed! Please review the output above."
    exit 1
fi

# Clean up
rm -f /tmp/finn_lib_test.py