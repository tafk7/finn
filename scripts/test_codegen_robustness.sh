#!/bin/bash
# FINN Unified Codegen Robustness Testing Suite
# Main orchestration script that runs all robustness tests

set -e  # Exit on any error

echo "🧪 FINN Unified Codegen Robustness Testing Suite"
echo "================================================="
echo "Started at: $(date)"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "PHASE")
            echo -e "${BLUE}🎯 === $message ===${NC}"
            ;;
    esac
}

# Initialize test results
TOTAL_PHASES=0
PASSED_PHASES=0

# Function to run a test phase
run_phase() {
    local phase_name=$1
    local script_path=$2
    
    print_status "PHASE" "$phase_name"
    TOTAL_PHASES=$((TOTAL_PHASES + 1))
    
    if [ -f "$script_path" ]; then
        if bash "$script_path"; then
            print_status "SUCCESS" "$phase_name completed successfully"
            PASSED_PHASES=$((PASSED_PHASES + 1))
        else
            print_status "ERROR" "$phase_name failed"
        fi
    else
        print_status "ERROR" "Script not found: $script_path"
    fi
    echo
}

# Check prerequisites
print_status "INFO" "Checking prerequisites..."

# Check if we're in the right directory
if [ ! -d "src/finn/codegen" ]; then
    print_status "ERROR" "Not in FINN project root directory. Please run from project root."
    exit 1
fi

# Check Python environment
if ! python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null; then
    print_status "ERROR" "Python not available"
    exit 1
fi

print_status "SUCCESS" "Prerequisites check passed"
echo

# Phase 1: Core Framework Validation
run_phase "Phase 1: Core Framework Validation" "scripts/test_core_framework.py"

# Phase 2: FINN Integration Discovery
run_phase "Phase 2: FINN Integration Discovery" "scripts/discover_finn_tests.py"

# Phase 3: Component Testing
run_phase "Phase 3: Component Testing" "scripts/test_components.py"

# Phase 4: Performance and Stress Testing
run_phase "Phase 4: Performance and Stress Testing" "scripts/test_performance.py"

# Phase 5: Integration Testing
run_phase "Phase 5: Integration Testing" "scripts/test_integration.py"

# Final Summary
echo
print_status "PHASE" "Final Results Summary"
echo "Total phases: $TOTAL_PHASES"
echo "Passed phases: $PASSED_PHASES"
echo "Failed phases: $((TOTAL_PHASES - PASSED_PHASES))"

if [ $PASSED_PHASES -eq $TOTAL_PHASES ]; then
    print_status "SUCCESS" "ALL ROBUSTNESS TESTS PASSED!"
    print_status "SUCCESS" "Unified codegen framework is robust and ready for use"
    exit 0
else
    print_status "WARNING" "$((TOTAL_PHASES - PASSED_PHASES)) phases failed - investigation needed"
    exit 1
fi