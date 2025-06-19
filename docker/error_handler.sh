#!/bin/bash
# FINN Error Handler
# Provides structured error handling with recovery suggestions

# Error codes
readonly ERR_GENERAL=1
readonly ERR_DOCKER=2
readonly ERR_DEPS=3
readonly ERR_BUILD=4
readonly ERR_XILINX=5
readonly ERR_PYTHON=6
readonly ERR_DISK=7
readonly ERR_MEMORY=8
readonly ERR_NETWORK=9
readonly ERR_PERMISSIONS=10

# Color codes
readonly RED='\033[0;31m'
readonly YELLOW='\033[0;33m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Error log file
ERROR_LOG="/tmp/.finn_cache/error.log"
mkdir -p $(dirname $ERROR_LOG)

# Function to handle errors with suggestions
handle_error() {
    local error_code=$1
    local error_msg=$2
    local context=$3
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log to file
    echo "[$timestamp] ERROR $error_code: $error_msg (Context: $context)" >> $ERROR_LOG
    
    # Display error
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ ERROR${NC} (Code: $error_code)"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}Message:${NC} $error_msg"
    echo -e "${RED}Context:${NC} $context"
    echo ""
    
    # Provide specific recovery suggestions
    echo -e "${YELLOW}Recovery Suggestions:${NC}"
    
    case $error_code in
        $ERR_DOCKER)
            echo "  • Check if Docker daemon is running: sudo systemctl status docker"
            echo "  • Verify Docker permissions: groups | grep docker"
            echo "  • Try: sudo usermod -aG docker $USER && newgrp docker"
            echo "  • Check Docker version: docker --version"
            ;;
            
        $ERR_DEPS)
            echo "  • Clean dependency cache: rm -rf $FINN_DEPS_DIR"
            echo "  • Re-run dependency fetch: ./finn-docker exec './docker/fetch-repos.sh'"
            echo "  • Check network connectivity: ping -c 3 github.com"
            echo "  • Verify git configuration: git config --list"
            echo "  • Try manual clone: git clone <repo_url>"
            ;;
            
        $ERR_BUILD)
            echo "  • Clean build directory: rm -rf $FINN_BUILD_DIR/*"
            echo "  • Check disk space: df -h $FINN_BUILD_DIR"
            echo "  • Verify Xilinx tools: echo \$FINN_XILINX_PATH"
            echo "  • Check build logs: tail -50 $FINN_BUILD_DIR/build.log"
            echo "  • Try with verbose output: export FINN_DEBUG=1"
            ;;
            
        $ERR_XILINX)
            echo "  • Set Xilinx path: export FINN_XILINX_PATH=/opt/Xilinx"
            echo "  • Set version: export FINN_XILINX_VERSION=2022.2"
            echo "  • Verify installation: ls -la \$FINN_XILINX_PATH/Vivado/"
            echo "  • Check license: echo \$XILINXD_LICENSE_FILE"
            echo "  • Run without Xilinx: export FINN_SKIP_XILINX=1"
            ;;
            
        $ERR_PYTHON)
            echo "  • Check Python version: python3 --version"
            echo "  • Reinstall packages: ./finn-docker exec 'pip install -r requirements.txt'"
            echo "  • Clear pip cache: pip cache purge"
            echo "  • Check virtual environment: which python3"
            echo "  • Try manual import: python3 -c 'import $context'"
            ;;
            
        $ERR_DISK)
            echo "  • Check disk usage: df -h"
            echo "  • Clean Docker: docker system prune -a"
            echo "  • Remove old builds: rm -rf /tmp/finn_dev_*"
            echo "  • Clear cache: rm -rf /tmp/.finn_cache"
            echo "  • Check inode usage: df -i"
            ;;
            
        $ERR_MEMORY)
            echo "  • Check memory: free -h"
            echo "  • Stop other containers: docker ps && docker stop <containers>"
            echo "  • Increase Docker memory limit in Docker Desktop settings"
            echo "  • Close memory-intensive applications"
            echo "  • Add swap space: sudo fallocate -l 4G /swapfile"
            ;;
            
        $ERR_NETWORK)
            echo "  • Check connectivity: ping -c 3 8.8.8.8"
            echo "  • Check DNS: nslookup github.com"
            echo "  • Check proxy settings: echo \$http_proxy"
            echo "  • Try different network: switch WiFi/Ethernet"
            echo "  • Use offline mode: export FINN_OFFLINE=1"
            ;;
            
        $ERR_PERMISSIONS)
            echo "  • Check file ownership: ls -la $context"
            echo "  • Fix permissions: sudo chown -R \$USER:\$USER $context"
            echo "  • Check Docker socket: ls -la /var/run/docker.sock"
            echo "  • Run with sudo (not recommended): sudo ./finn-docker ..."
            echo "  • Check SELinux: getenforce"
            ;;
            
        *)
            echo "  • Check recent changes: git status"
            echo "  • View full error log: cat $ERROR_LOG"
            echo "  • Run health check: ./finn-docker health"
            echo "  • Clean and retry: ./finn-docker clean && ./finn-docker init"
            echo "  • Report issue: https://github.com/Xilinx/finn/issues"
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}Additional Information:${NC}"
    echo "  • Error log: $ERROR_LOG"
    echo "  • Container: $DOCKER_INST_NAME"
    echo "  • Timestamp: $timestamp"
    
    # Check if we should run diagnostics
    if [ "${FINN_ERROR_DIAGNOSTICS:-1}" = "1" ]; then
        echo ""
        echo -e "${BLUE}Running diagnostics...${NC}"
        run_error_diagnostics "$error_code" "$context"
    fi
    
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Run diagnostics based on error type
run_error_diagnostics() {
    local error_code=$1
    local context=$2
    
    case $error_code in
        $ERR_DOCKER)
            echo "Docker status:"
            docker version 2>&1 | head -5 || echo "  Docker not accessible"
            echo ""
            echo "Container status:"
            docker ps -a | grep finn || echo "  No FINN containers found"
            ;;
            
        $ERR_DISK)
            echo "Disk usage:"
            df -h | grep -E "/$|$FINN_BUILD_DIR|/tmp" | head -5
            echo ""
            echo "Large directories:"
            du -sh /tmp/finn_* 2>/dev/null | sort -h | tail -5
            ;;
            
        $ERR_MEMORY)
            echo "Memory status:"
            free -h | head -3
            echo ""
            echo "Top memory users:"
            ps aux --sort=-%mem | head -5
            ;;
            
        $ERR_PYTHON)
            echo "Python environment:"
            which python3
            python3 --version
            echo "PYTHONPATH: $PYTHONPATH"
            ;;
    esac
}

# Export functions for use in other scripts
export -f handle_error
export -f run_error_diagnostics

# If sourced with arguments, handle the error
if [ $# -ge 3 ]; then
    handle_error "$@"
fi