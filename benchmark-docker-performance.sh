#!/bin/bash
# FINN Docker Performance Benchmark Script
# Compares old vs new Docker workflow execution times

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

# Test commands to benchmark
TEST_COMMANDS=(
    "python3 -c \"import finn; print('FINN imported')\""
    "python3 -c \"import qonnx; print('QONNX imported')\""
    "python3 -c \"import numpy as np; print('NumPy version:', np.__version__)\""
    "ls -la /home/tafk/dev/finn/src"
    "python3 -c \"from finn.util.basic import get_finn_root; print('FINN root:', get_finn_root())\""
)

benchmark_old_workflow() {
    becho "🕐 Benchmarking OLD workflow (traditional docker run)..."
    
    # Clean up any existing containers
    docker stop finn_dev_$(whoami) 2>/dev/null || true
    docker rm finn_dev_$(whoami) 2>/dev/null || true
    
    local total_time=0
    
    for cmd in "${TEST_COMMANDS[@]}"; do
        echo "  Testing: $cmd"
        
        start_time=$(date +%s.%N)
        
        # Old workflow: full container startup each time
        timeout 60s bash -c "
            FINN_DOCKER_PERSISTENT=0 FINN_LIGHTWEIGHT_ENTRYPOINT=0 \\
            ./run-docker.sh bash -c '$cmd'
        " > /dev/null 2>&1 || echo "    ⚠️ Command timed out or failed"
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $duration" | bc)
        
        printf "    Time: %.2fs\\n" "$duration"
    done
    
    printf "  ${YELLOW}Total OLD workflow time: %.2fs${NC}\\n" "$total_time"
    echo "$total_time" > /tmp/old_workflow_time.txt
}

benchmark_new_workflow() {
    becho "🚀 Benchmarking NEW workflow (persistent container + quick exec)..."
    
    # Start persistent container
    ./finn-container start daemon
    sleep 2
    
    local total_time=0
    
    for cmd in "${TEST_COMMANDS[@]}"; do
        echo "  Testing: $cmd"
        
        start_time=$(date +%s.%N)
        
        # New workflow: quick exec on running container
        FINN_QUICK_EXEC=1 ./finn-container exec "$cmd" > /dev/null 2>&1 || echo "    ⚠️ Command failed"
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $duration" | bc)
        
        printf "    Time: %.2fs\\n" "$duration"
    done
    
    printf "  ${GREEN}Total NEW workflow time: %.2fs${NC}\\n" "$total_time"
    echo "$total_time" > /tmp/new_workflow_time.txt
    
    # Clean up
    ./finn-container stop
}

calculate_improvement() {
    if [ -f /tmp/old_workflow_time.txt ] && [ -f /tmp/new_workflow_time.txt ]; then
        old_time=$(cat /tmp/old_workflow_time.txt)
        new_time=$(cat /tmp/new_workflow_time.txt)
        
        if [ "$(echo "$old_time > 0" | bc)" -eq 1 ] && [ "$(echo "$new_time > 0" | bc)" -eq 1 ]; then
            speedup=$(echo "scale=2; $old_time / $new_time" | bc)
            time_saved=$(echo "scale=2; $old_time - $new_time" | bc)
            
            echo ""
            gecho "📊 Performance Summary:"
            echo "  Old workflow total time: ${old_time}s"
            echo "  New workflow total time: ${new_time}s"
            echo "  Time saved: ${time_saved}s"
            echo "  Speedup: ${speedup}x faster"
            
            # Clean up temp files
            rm -f /tmp/old_workflow_time.txt /tmp/new_workflow_time.txt
        fi
    fi
}

main() {
    echo "🏃‍♂️ FINN Docker Performance Benchmark"
    echo "======================================"
    echo ""
    
    # Check dependencies
    if ! command -v bc &> /dev/null; then
        recho "❌ Error: 'bc' command is required for calculations"
        echo "Install with: sudo apt-get install bc"
        exit 1
    fi
    
    yecho "⚠️  This benchmark will:"
    echo "  1. Test the old Docker workflow (slower)"
    echo "  2. Test the new persistent container workflow (faster)"
    echo "  3. Compare execution times"
    echo ""
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Benchmark cancelled."
        exit 0
    fi
    
    benchmark_old_workflow
    echo ""
    benchmark_new_workflow
    echo ""
    calculate_improvement
    
    gecho "✅ Benchmark completed!"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi