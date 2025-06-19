#!/bin/bash
# FINN Docker Container Fast Execution Entrypoint
# Provides fast command execution without re-initialization

# Fast environment setup - minimal overhead
export FINN_ROOT=${FINN_ROOT:-/workspace/finn}
export PYTHONPATH=$FINN_ROOT/src:$PYTHONPATH
export PATH=$FINN_ROOT/src/finn/qnn-data/bin:$PATH

# Check if we need full environment setup
needs_xilinx() {
    # Check if command needs Xilinx tools
    case "$1" in
        vivado|vitis|xsct|hls|*synthesis*|*build*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Lazy Xilinx setup only if needed
if needs_xilinx "$@" && [ -f "/tmp/.xilinx_setup_done" ]; then
    # Source cached environment
    if [ -f "/tmp/.finn_env_cache" ]; then
        source /tmp/.finn_env_cache
    else
        # Full setup needed
        SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
        source $SCRIPT_DIR/entrypoint_common.sh
        setup_finn_environment
        
        # Cache environment for next time
        export -p > /tmp/.finn_env_cache
    fi
elif [ ! -f "/tmp/.finn_basic_setup" ]; then
    # One-time basic setup
    export FINN_BUILD_DIR=${FINN_BUILD_DIR:-/tmp/finn_build}
    export VIVADO_IP_CACHE=${VIVADO_IP_CACHE:-$FINN_BUILD_DIR/vivado_ip_cache}
    export NUM_DEFAULT_WORKERS=${NUM_DEFAULT_WORKERS:-4}
    
    # Mark basic setup done
    touch /tmp/.finn_basic_setup
fi

# Execute command with minimal overhead
exec "$@"