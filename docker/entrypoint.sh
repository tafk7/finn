#!/bin/bash
# FINN Docker Container Entrypoint
# Handles initialization and setup for FINN containers

# Source common environment setup
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source $SCRIPT_DIR/entrypoint_common.sh

# Emit status for monitoring
emit_status() {
    echo "FINN_STATUS:$1"
}

# Initialize container
initialize_container() {
    emit_status "INITIALIZING"
    
    # Set up base environment
    setup_finn_environment
    
    # Check if running in daemon mode
    if [ "$FINN_CONTAINER_MODE" = "daemon" ]; then
        echo "Starting FINN container in daemon mode..."
        # Still need to do dependency installation in daemon mode
        # but we'll return to the tail command at the end
    fi
    
    # Fetch dependencies if not skipped
    if [ "$FINN_SKIP_DEP_REPOS" = "0" ] && [ -d "$FINN_ROOT" ]; then
        emit_status "FETCHING_DEPENDENCIES"
        cd $FINN_ROOT
        
        # Use Python-based fetcher if available
        if [ -f "/usr/local/bin/fetch_deps.py" ] && python3 -c "import yaml" 2>/dev/null; then
            echo "Using enhanced dependency fetcher..."
            python3 /usr/local/bin/fetch_deps.py
            if [ $? -ne 0 ]; then
                echo "Dependency fetching failed, falling back to basic installation"
            fi
        fi
        
        # Install git repo dependencies as editable packages
        if [ -d "$FINN_DEPS_DIR" ]; then
            # Use Python-based installer if available
            if [ -f "/usr/local/bin/install_deps.py" ]; then
                echo "Installing Python dependencies..."
                python3 /usr/local/bin/install_deps.py
                if [ $? -ne 0 ]; then
                    echo "Python package installation had errors"
                    emit_status "ERROR:Failed to install some packages"
                    return 1
                fi
            else
                # Fallback to basic installation
                install_with_cache "qonnx" "$FINN_DEPS_DIR/qonnx"
                install_with_cache "finn_experimental" "$FINN_DEPS_DIR/finn-experimental" 
                install_with_cache "brevitas" "$FINN_DEPS_DIR/brevitas"
                install_with_cache "pyverilator" "$FINN_DEPS_DIR/pyverilator"
            fi
        fi
    fi
    
    emit_status "READY"
    echo "FINN container ready"
    
    # If daemon mode, keep container running
    if [ "$FINN_CONTAINER_MODE" = "daemon" ]; then
        exec tail -f /dev/null
    fi
}

# Install package with caching to avoid redundant installations
install_with_cache() {
    local package_name=$1
    local package_path=$2
    
    if [ ! -d "$package_path" ]; then
        echo "Warning: Package directory not found: $package_path"
        return 1
    fi
    
    # Check if package is already installed and up to date
    local cache_file="/tmp/.finn_cache/${package_name}.installed"
    local current_commit=$(cd "$package_path" && git rev-parse HEAD 2>/dev/null || echo "unknown")
    
    if [ -f "$cache_file" ]; then
        local cached_commit=$(cat "$cache_file")
        if [ "$cached_commit" = "$current_commit" ]; then
            echo "Package $package_name already installed (cached)"
            return 0
        fi
    fi
    
    emit_status "INSTALLING_PACKAGES:$package_name"
    echo "Installing $package_name from $package_path"
    
    # Install as editable package
    if pip install -e "$package_path" --no-deps; then
        # Update cache
        mkdir -p /tmp/.finn_cache
        echo "$current_commit" > "$cache_file"
        echo "Successfully installed $package_name"
        return 0
    else
        echo "Failed to install $package_name"
        emit_status "ERROR:Failed to install $package_name"
        return 1
    fi
}

# Main execution
if [ $# -eq 0 ]; then
    # No command provided, initialize and run bash
    initialize_container
    exec bash
else
    # Command provided, initialize and run it
    initialize_container
    exec "$@"
fi