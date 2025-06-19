#!/bin/bash
# Integration tests for FINN Docker system
# Tests the complete workflow from initialization to cleanup

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
FINN_ROOT=$(dirname "$SCRIPT_DIR")
FINN_DOCKER="$FINN_ROOT/finn-docker"
TEST_LOG="/tmp/finn_docker_integration_test.log"
TEST_CONTAINER="finn_test_$$"

# Logging functions
log() {
    echo -e "${BLUE}[TEST]${NC} $1" | tee -a $TEST_LOG
}

pass() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a $TEST_LOG
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a $TEST_LOG
    cleanup
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a $TEST_LOG
}

# Cleanup function
cleanup() {
    log "Cleaning up test environment..."
    $FINN_DOCKER clean 2>/dev/null || true
    rm -f /tmp/test_*.py
}

# Setup
setup() {
    log "Setting up integration test environment..."
    echo "Integration test started at $(date)" > $TEST_LOG
    
    # Check Docker availability
    if ! docker version >/dev/null 2>&1; then
        fail "Docker is not available. Please install Docker and try again."
    fi
    
    # Set test environment variables
    export FINN_SHOW_INIT_LOGS=true
    export FINN_DEBUG=1
    
    cleanup  # Clean any previous test containers
}

# Test 1: Basic container lifecycle
test_basic_lifecycle() {
    log "Test 1: Basic container lifecycle"
    
    # Check initial status
    log "Checking initial status..."
    OUTPUT=$($FINN_DOCKER status 2>&1)
    if [[ "$OUTPUT" == *"does not exist"* ]]; then
        pass "Container correctly shows as not existing"
    else
        fail "Unexpected initial status: $OUTPUT"
    fi
    
    # Initialize container
    log "Initializing container..."
    if $FINN_DOCKER init; then
        pass "Container initialized successfully"
    else
        fail "Container initialization failed"
    fi
    
    # Check running status
    log "Checking running status..."
    OUTPUT=$($FINN_DOCKER status 2>&1)
    if [[ "$OUTPUT" == *"is running"* ]]; then
        pass "Container shows as running"
    else
        fail "Container not running after init: $OUTPUT"
    fi
    
    # Execute simple command
    log "Executing test command..."
    OUTPUT=$($FINN_DOCKER exec "echo 'Hello from FINN container'" 2>&1)
    if [[ "$OUTPUT" == *"Hello from FINN container"* ]]; then
        pass "Command execution successful"
    else
        fail "Command execution failed: $OUTPUT"
    fi
    
    # Stop container
    log "Stopping container..."
    if $FINN_DOCKER stop; then
        pass "Container stopped successfully"
    else
        fail "Failed to stop container"
    fi
    
    # Clean up
    log "Cleaning container..."
    if $FINN_DOCKER clean; then
        pass "Container cleaned successfully"
    else
        fail "Failed to clean container"
    fi
}

# Test 2: Fast execution path
test_fast_execution() {
    log "Test 2: Fast execution path"
    
    # Initialize container first
    log "Initializing container for fast exec test..."
    $FINN_DOCKER init || fail "Failed to initialize container"
    
    # Time a simple command
    log "Testing execution speed..."
    START=$(date +%s.%N)
    $FINN_DOCKER exec "echo fast" >/dev/null 2>&1
    END=$(date +%s.%N)
    DURATION=$(echo "$END - $START" | bc)
    
    log "Execution took ${DURATION}s"
    if (( $(echo "$DURATION < 2.0" | bc -l) )); then
        pass "Fast execution confirmed (${DURATION}s < 2s)"
    else
        warn "Execution slower than expected (${DURATION}s)"
    fi
}

# Test 3: Python environment
test_python_environment() {
    log "Test 3: Python environment"
    
    # Create test Python script
    cat > /tmp/test_env.py << 'EOF'
import os
import sys

print(f"Python version: {sys.version}")
print(f"FINN_ROOT: {os.environ.get('FINN_ROOT', 'NOT SET')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")

# Try importing key packages
try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except ImportError:
    print("NumPy not available")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not available")
EOF

    log "Testing Python environment..."
    OUTPUT=$($FINN_DOCKER exec "python3 /tmp/test_env.py" 2>&1)
    
    if [[ "$OUTPUT" == *"Python version:"* ]]; then
        pass "Python environment accessible"
    else
        fail "Python environment test failed: $OUTPUT"
    fi
    
    if [[ "$OUTPUT" == *"FINN_ROOT:"* ]] && [[ "$OUTPUT" != *"NOT SET"* ]]; then
        pass "FINN environment variables set correctly"
    else
        warn "FINN environment variables may not be set correctly"
    fi
}

# Test 4: Error handling
test_error_handling() {
    log "Test 4: Error handling"
    
    # Test invalid command
    log "Testing error handling for invalid command..."
    OUTPUT=$($FINN_DOCKER invalid_command 2>&1 || true)
    if [[ "$OUTPUT" == *"Unknown command"* ]] || [[ "$OUTPUT" == *"help"* ]]; then
        pass "Invalid command handled correctly"
    else
        warn "Unexpected output for invalid command: $OUTPUT"
    fi
    
    # Test exec without container
    $FINN_DOCKER clean 2>/dev/null || true
    log "Testing exec without running container..."
    OUTPUT=$($FINN_DOCKER exec "echo test" 2>&1 || true)
    if [[ "$OUTPUT" == *"not running"* ]]; then
        pass "Exec without container handled correctly"
    else
        fail "Unexpected output for exec without container: $OUTPUT"
    fi
}

# Test 5: Advanced features
test_advanced_features() {
    log "Test 5: Advanced features (shortcuts)"
    
    # Initialize container
    $FINN_DOCKER init || fail "Failed to initialize container"
    
    # Test Python shortcut
    log "Testing Python shortcut..."
    OUTPUT=$($FINN_DOCKER python -c "print('Python shortcut works')" 2>&1)
    if [[ "$OUTPUT" == *"Python shortcut works"* ]]; then
        pass "Python shortcut working"
    else
        fail "Python shortcut failed: $OUTPUT"
    fi
    
    # Test pytest shortcut
    log "Testing pytest shortcut..."
    OUTPUT=$($FINN_DOCKER pytest --version 2>&1 || true)
    if [[ "$OUTPUT" == *"pytest"* ]]; then
        pass "Pytest shortcut working"
    else
        warn "Pytest shortcut may not be working: $OUTPUT"
    fi
    
    # Test working directory support
    log "Testing working directory support..."
    mkdir -p $FINN_ROOT/test_workspace
    cd $FINN_ROOT/test_workspace
    OUTPUT=$($FINN_DOCKER exec "pwd" 2>&1)
    cd $FINN_ROOT
    rm -rf test_workspace
    
    if [[ "$OUTPUT" == *"test_workspace"* ]]; then
        pass "Working directory support confirmed"
    else
        warn "Working directory support may not be working as expected"
    fi
}

# Test 6: Health check
test_health_check() {
    log "Test 6: Health check system"
    
    # Run health check
    log "Running health check..."
    OUTPUT=$($FINN_DOCKER health 2>&1)
    
    if [[ "$OUTPUT" == *"Overall Status:"* ]]; then
        pass "Health check executed successfully"
        
        if [[ "$OUTPUT" == *"HEALTHY"* ]]; then
            pass "Container reported as healthy"
        elif [[ "$OUTPUT" == *"DEGRADED"* ]]; then
            warn "Container reported as degraded"
        else
            warn "Container health status unclear"
        fi
    else
        fail "Health check failed to execute"
    fi
}

# Test 7: Backward compatibility
test_backward_compatibility() {
    log "Test 7: Backward compatibility"
    
    # Test run-docker.sh wrapper
    if [ -x "$FINN_ROOT/run-docker.sh" ]; then
        log "Testing run-docker.sh wrapper..."
        OUTPUT=$($FINN_ROOT/run-docker.sh help 2>&1 || true)
        
        if [[ "$OUTPUT" == *"deprecated"* ]]; then
            pass "Deprecation notice shown correctly"
        else
            warn "Deprecation notice not shown"
        fi
        
        # Test quicktest command
        OUTPUT=$($FINN_ROOT/run-docker.sh quicktest 2>&1 || true)
        if [[ "$?" -eq 0 ]] || [[ "$OUTPUT" == *"quicktest"* ]]; then
            pass "Legacy quicktest command handled"
        else
            warn "Legacy quicktest command may not work correctly"
        fi
    else
        warn "run-docker.sh wrapper not found"
    fi
}

# Test 8: Performance monitoring
test_performance_monitoring() {
    log "Test 8: Performance monitoring"
    
    # Clean and test with profiling enabled
    $FINN_DOCKER clean 2>/dev/null || true
    
    log "Testing startup profiling..."
    export FINN_PROFILE_STARTUP=1
    OUTPUT=$($FINN_DOCKER init 2>&1)
    unset FINN_PROFILE_STARTUP
    
    if [[ "$OUTPUT" == *"[PROFILE]"* ]]; then
        pass "Startup profiling working"
        
        # Extract timing if possible
        if [[ "$OUTPUT" =~ "Total initialization time: "([0-9.]+)"s" ]]; then
            INIT_TIME="${BASH_REMATCH[1]}"
            log "Initialization time: ${INIT_TIME}s"
            
            if (( $(echo "$INIT_TIME < 300" | bc -l) )); then
                pass "Initialization completed within 5 minutes"
            else
                warn "Initialization took longer than expected"
            fi
        fi
    else
        warn "Startup profiling output not found"
    fi
}

# Test 9: Dependency verification
test_dependency_verification() {
    log "Test 9: Dependency verification"
    
    # Run verification
    log "Running dependency verification..."
    OUTPUT=$($FINN_DOCKER verify 2>&1 || true)
    
    if [[ "$OUTPUT" == *"Starting FINN installation verification"* ]]; then
        pass "Verification system accessible"
        
        if [[ "$OUTPUT" == *"verification completed successfully"* ]]; then
            pass "All verifications passed"
        else
            warn "Some verifications may have failed"
        fi
    else
        warn "Verification system may not be working correctly"
    fi
}

# Test 10: Stress test
test_stress() {
    log "Test 10: Stress test (multiple rapid commands)"
    
    # Execute multiple commands rapidly
    log "Executing rapid commands..."
    FAILURES=0
    for i in {1..10}; do
        if ! $FINN_DOCKER exec "echo Test $i" >/dev/null 2>&1; then
            ((FAILURES++))
        fi
    done
    
    if [ $FAILURES -eq 0 ]; then
        pass "All rapid commands executed successfully"
    else
        warn "$FAILURES out of 10 rapid commands failed"
    fi
}

# Main test execution
main() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "FINN Docker Integration Test Suite"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    echo "Test log: $TEST_LOG"
    echo ""
    
    # Setup
    setup
    
    # Run tests
    TOTAL_TESTS=10
    CURRENT_TEST=0
    
    tests=(
        test_basic_lifecycle
        test_fast_execution
        test_python_environment
        test_error_handling
        test_advanced_features
        test_health_check
        test_backward_compatibility
        test_performance_monitoring
        test_dependency_verification
        test_stress
    )
    
    for test in "${tests[@]}"; do
        ((CURRENT_TEST++))
        echo ""
        echo "[$CURRENT_TEST/$TOTAL_TESTS] Running $test..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Run test in subshell to isolate failures
        if ( $test ); then
            echo -e "${GREEN}✓ $test completed${NC}"
        else
            echo -e "${RED}✗ $test failed${NC}"
        fi
    done
    
    # Final cleanup
    cleanup
    
    # Summary
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Integration Test Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "End time: $(date)"
    echo "Test log: $TEST_LOG"
    echo ""
    
    # Count results
    PASSES=$(grep -c "\[PASS\]" $TEST_LOG || true)
    FAILS=$(grep -c "\[FAIL\]" $TEST_LOG || true)
    WARNS=$(grep -c "\[WARN\]" $TEST_LOG || true)
    
    echo -e "${GREEN}Passed: $PASSES${NC}"
    echo -e "${YELLOW}Warnings: $WARNS${NC}"
    echo -e "${RED}Failed: $FAILS${NC}"
    
    if [ $FAILS -eq 0 ]; then
        echo ""
        echo -e "${GREEN}All integration tests completed successfully!${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}Some tests failed. Check $TEST_LOG for details.${NC}"
        exit 1
    fi
}

# Run main if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi