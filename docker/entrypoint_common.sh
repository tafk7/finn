#!/bin/bash
# FINN Docker Container Common Environment Setup
# Shared environment configuration for all entrypoints

# Setup FINN environment variables and paths
setup_finn_environment() {
    # Ensure FINN_ROOT is set
    if [ -z "$FINN_ROOT" ]; then
        export FINN_ROOT=/workspace/finn
    fi
    
    # Add FINN to Python path
    export PYTHONPATH=$FINN_ROOT/src:$PYTHONPATH
    
    # Set up build directory
    if [ -z "$FINN_BUILD_DIR" ]; then
        export FINN_BUILD_DIR=/tmp/finn_build
    fi
    mkdir -p $FINN_BUILD_DIR
    
    # Set up Vivado IP cache if not set
    if [ -z "$VIVADO_IP_CACHE" ]; then
        export VIVADO_IP_CACHE=$FINN_BUILD_DIR/vivado_ip_cache
    fi
    mkdir -p $VIVADO_IP_CACHE
    
    # Source Xilinx tools if available
    if [ ! -z "$XILINX_VIVADO" ] && [ -d "$XILINX_VIVADO" ]; then
        if [ -f "$XILINX_VIVADO/settings64.sh" ]; then
            source $XILINX_VIVADO/settings64.sh
        fi
    fi
    
    # Source Vitis if available
    if [ ! -z "$VITIS_PATH" ] && [ -d "$VITIS_PATH" ]; then
        if [ -f "$VITIS_PATH/settings64.sh" ]; then
            source $VITIS_PATH/settings64.sh
        fi
    fi
    
    # Source HLS if available
    if [ ! -z "$HLS_PATH" ] && [ -d "$HLS_PATH" ]; then
        if [ -f "$HLS_PATH/settings64.sh" ]; then
            source $HLS_PATH/settings64.sh
        fi
    fi
    
    # Set up Oh My Xilinx if available
    if [ ! -z "$OHMYXILINX" ] && [ -d "$OHMYXILINX" ]; then
        export PATH=$OHMYXILINX:$PATH
    fi
    
    # Add FINN scripts to PATH
    export PATH=$FINN_ROOT/src/finn/qnn-data/bin:$PATH
    
    # Configure Jupyter if running notebook
    if [ ! -z "$JUPYTER_PORT" ]; then
        export JUPYTER_CONFIG_DIR=$HOME/.jupyter
        mkdir -p $JUPYTER_CONFIG_DIR
    fi
    
    # Set default parallelism
    if [ -z "$NUM_DEFAULT_WORKERS" ]; then
        export NUM_DEFAULT_WORKERS=4
    fi
    
    # Ensure XRT environment is set if available
    if [ -f "/opt/xilinx/xrt/setup.sh" ]; then
        source /opt/xilinx/xrt/setup.sh
    fi
    
    # Create cache directory for package installations
    mkdir -p /tmp/.finn_cache
    
    # Export any additional environment variables needed
    export FINN_DOCKER_CONTAINER=1
}