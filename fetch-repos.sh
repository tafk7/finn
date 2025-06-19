#!/bin/bash
# Wrapper script for backward compatibility
# The actual fetch-repos.sh has been moved to the docker/ directory

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
exec "$SCRIPT_DIR/docker/fetch-repos.sh" "$@"