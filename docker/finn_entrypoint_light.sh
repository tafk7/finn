#!/bin/bash
# Lightweight FINN entrypoint script
# This script provides a fast environment setup without reinstalling packages

set -e

# Source the lightweight environment setup
if [ -f "/usr/local/bin/finn_setup_env.sh" ]; then
    source /usr/local/bin/finn_setup_env.sh
else
    echo "Warning: finn_setup_env.sh not found, falling back to basic setup"
    
    # Basic environment setup if the comprehensive script isn't available
    export FINN_ROOT=${FINN_ROOT:-/workspace/finn}
    export FINN_DEPS_DIR=${FINN_DEPS_DIR:-/workspace/finn/deps}
    
    # Add FINN to Python path
    if [ -d "$FINN_ROOT" ]; then
        export PYTHONPATH="$FINN_ROOT:$PYTHONPATH"
    fi
    
    # Add dependencies to Python path
    if [ -d "$FINN_DEPS_DIR" ]; then
        for dep_dir in "$FINN_DEPS_DIR"/*; do
            if [ -d "$dep_dir" ] && [ -f "$dep_dir/setup.py" ]; then
                export PYTHONPATH="$dep_dir:$PYTHONPATH"
            fi
        done
    fi
fi

# Execute the provided command
exec "$@"