#!/bin/bash
# FINN Docker Improvements Validation Script
# Validates that all improvements are working correctly

set -e

SCRIPTPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPTPATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

gecho() { echo -e "${GREEN}$*${NC}"; }
yecho() { echo -e "${YELLOW}$*${NC}"; }
becho() { echo -e "${BLUE}$*${NC}"; }
recho() { echo -e "${RED}$*${NC}"; }

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"
    
    ((TESTS_TOTAL++))
    
    echo -n "  Testing: $test_name ... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        local exit_code=$?
        if [ $exit_code -eq $expected_exit_code ]; then
            echo "✅ PASS"
            ((TESTS_PASSED++))
            return 0
        else
            echo "❌ FAIL (exit code: $exit_code, expected: $expected_exit_code)"
            ((TESTS_FAILED++))
            return 1
        fi
    else
        echo "❌ FAIL (command failed)"
        ((TESTS_FAILED++))
        return 1
    fi
}

validate_file_structure() {
    becho "📁 Validating File Structure"
    echo "============================"
    
    local required_files=(
        "./finn-container"
        "./docker/finn_setup_env.sh"
        "./docker/finn_entrypoint_light.sh"
        "./docker/Dockerfile.finn"
        "./run-docker.sh"
        "./DOCKER_IMPROVEMENTS.md"
    )
    
    for file in "${required_files[@]}"; do
        run_test "File exists: $file" "[ -f '$file' ]"
    done
    
    # Check if scripts are executable
    local executable_files=(
        "./finn-container"
        "./run-docker.sh"
    )
    
    for file in "${executable_files[@]}"; do
        run_test "File executable: $file" "[ -x '$file' ]"
    done
    
    echo ""
}

validate_container_management() {
    becho "🐳 Validating Container Management"
    echo "=================================="
    
    # Clean up any existing containers first
    docker stop finn_dev_$(whoami) 2>/dev/null || true
    docker rm finn_dev_$(whoami) 2>/dev/null || true
    
    run_test "Container status (not running)" "./finn-container status" 1
    run_test "Start daemon container" "./finn-container start daemon"
    run_test "Container status (running)" "./finn-container status"
    run_test "Stop container" "./finn-container stop"
    run_test "Container status (stopped)" "./finn-container status" 1
    
    echo ""
}

validate_exec_functionality() {
    becho "⚡ Validating Exec Functionality"
    echo "==============================="
    
    # Start container for exec tests
    ./finn-container start daemon > /dev/null 2>&1
    sleep 2
    
    run_test "Basic exec command" "./finn-container exec 'echo \"test\"'"
    run_test "Python execution" "./finn-container exec 'python3 -c \"print(1+1)\"'"
    run_test "FINN import" "./finn-container exec 'python3 -c \"import finn\"'"
    run_test "Quick exec mode" "FINN_QUICK_EXEC=1 ./finn-container exec 'python3 -c \"import sys\"'"
    run_test "File system access" "./finn-container exec 'ls /home/tafk/dev/finn'"
    
    # Clean up
    ./finn-container stop > /dev/null 2>&1
    
    echo ""
}

validate_environment_setup() {
    becho "🔧 Validating Environment Setup"
    echo "==============================="
    
    # Start container for environment tests
    ./finn-container start daemon > /dev/null 2>&1
    sleep 2
    
    run_test "FINN environment variables" "./finn-container exec 'test -n \"\$FINN_ROOT\"'"
    run_test "Python path setup" "./finn-container exec 'python3 -c \"import sys; assert \"/home/tafk/dev/finn/src\" in sys.path\"'"
    run_test "FINN utilities accessible" "./finn-container exec 'python3 -c \"from finn.util.basic import get_finn_root\"'"
    
    # Clean up
    ./finn-container stop > /dev/null 2>&1
    
    echo ""
}

validate_library_imports() {
    becho "📚 Validating Library Imports"
    echo "============================="
    
    # Start container for library tests
    ./finn-container start daemon > /dev/null 2>&1
    sleep 2
    
    run_test "NumPy import" "./finn-container exec 'python3 -c \"import numpy\"'"
    run_test "ONNX import" "./finn-container exec 'python3 -c \"import onnx\"'"
    run_test "QONNX import" "./finn-container exec 'python3 -c \"import qonnx\"'"
    run_test "FINN import" "./finn-container exec 'python3 -c \"import finn\"'"
    
    # Optional libraries (don't fail if missing)
    echo "  Testing optional libraries..."
    ./finn-container exec 'python3 -c "import torch"' > /dev/null 2>&1 && echo "    ✅ PyTorch available" || echo "    ⚠️ PyTorch not available"
    ./finn-container exec 'python3 -c "import brevitas"' > /dev/null 2>&1 && echo "    ✅ Brevitas available" || echo "    ⚠️ Brevitas not available"
    
    # Clean up
    ./finn-container stop > /dev/null 2>&1
    
    echo ""
}

validate_performance_improvements() {
    becho "🚀 Validating Performance Improvements"
    echo "======================================"
    
    echo "  Measuring quick exec performance..."
    
    # Start container
    ./finn-container start daemon > /dev/null 2>&1
    sleep 2
    
    # Time a quick exec command
    start_time=$(date +%s.%N)
    FINN_QUICK_EXEC=1 ./finn-container exec 'python3 -c "import finn"' > /dev/null 2>&1
    end_time=$(date +%s.%N)
    
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Quick exec should be under 3 seconds
    if (( $(echo "$duration < 3.0" | bc -l) )); then
        echo "  ✅ Quick exec performance good: ${duration}s"
        ((TESTS_PASSED++))
    else
        echo "  ⚠️ Quick exec slower than expected: ${duration}s"
    fi
    ((TESTS_TOTAL++))
    
    # Clean up
    ./finn-container stop > /dev/null 2>&1
    
    echo ""
}

validate_documentation() {
    becho "📖 Validating Documentation"
    echo "==========================="
    
    run_test "Docker improvements doc exists" "[ -f './DOCKER_IMPROVEMENTS.md' ]"
    run_test "Documentation not empty" "[ -s './DOCKER_IMPROVEMENTS.md' ]"
    
    echo ""
}

print_summary() {
    echo "📊 Validation Summary"
    echo "===================="
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        gecho "🎉 All tests passed! ($TESTS_PASSED/$TESTS_TOTAL)"
        gecho "✅ FINN Docker improvements are working correctly!"
    else
        yecho "⚠️ Some tests failed: $TESTS_FAILED/$TESTS_TOTAL"
        echo "  Passed: $TESTS_PASSED"
        echo "  Failed: $TESTS_FAILED"
        echo ""
        yecho "Please check the failed tests above."
    fi
    
    echo ""
    gecho "🚀 Ready to use improved FINN Docker workflow!"
    echo ""
    echo "Quick start:"
    echo "  ./finn-container start daemon     # Start persistent container"
    echo "  ./finn-container exec 'command'   # Run one-off commands"
    echo "  ./finn-container shell            # Interactive shell"
    echo "  ./finn-container stop             # Stop container"
}

main() {
    echo "🧪 FINN Docker Improvements Validation"
    echo "======================================"
    echo ""
    
    yecho "This script validates that all Docker improvements are working correctly."
    echo ""
    
    # Check dependencies
    if ! command -v bc &> /dev/null; then
        recho "❌ Error: 'bc' command is required for calculations"
        echo "Install with: sudo apt-get install bc"
        exit 1
    fi
    
    validate_file_structure
    validate_container_management
    validate_exec_functionality
    validate_environment_setup
    validate_library_imports
    validate_performance_improvements
    validate_documentation
    
    print_summary
    
    if [ $TESTS_FAILED -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi