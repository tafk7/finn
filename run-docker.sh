#!/bin/bash
# FINN run-docker.sh - Backward Compatibility Wrapper
# This script provides backward compatibility for existing workflows
# while redirecting to the new finn-docker command

# Get script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Color codes for output
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Show deprecation notice
echo -e "${YELLOW}Note: run-docker.sh is deprecated. Please use 'finn-docker' for better performance.${NC}"
echo -e "${YELLOW}See 'finn-docker help' for more information.${NC}"
echo ""

# Check if finn-docker exists
if [ ! -f "$SCRIPT_DIR/finn-docker" ]; then
    echo "Error: finn-docker not found. Using legacy script."
    exec "$SCRIPT_DIR/run-docker.sh.legacy" "$@"
fi

# Initialize container if needed for commands that require it
init_if_needed() {
    # Check if container is already running
    if ! $SCRIPT_DIR/finn-docker status 2>&1 | grep -q "is running"; then
        echo "Initializing FINN container..."
        $SCRIPT_DIR/finn-docker init
        # Wait a moment for initialization
        sleep 2
    fi
}

# Map old commands to new interface
case "$1" in
    "test")
        init_if_needed
        exec $SCRIPT_DIR/finn-docker test
        ;;
    "quicktest")
        init_if_needed
        exec $SCRIPT_DIR/finn-docker quicktest
        ;;
    "notebook")
        init_if_needed
        exec $SCRIPT_DIR/finn-docker notebook
        ;;
    "build_dataflow")
        shift
        init_if_needed
        exec $SCRIPT_DIR/finn-docker build_dataflow "$@"
        ;;
    "build_custom")
        shift
        init_if_needed
        exec $SCRIPT_DIR/finn-docker build_custom "$@"
        ;;
    "")
        # No arguments - interactive shell
        # For interactive shell, we don't use persistent container
        exec $SCRIPT_DIR/finn-docker shell
        ;;
    *)
        # Pass through any other commands
        init_if_needed
        exec $SCRIPT_DIR/finn-docker exec "$@"
        ;;
esac