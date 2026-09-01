#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# activate.sh - Environment activation for a bare-host FINN installation.
#
# Source this script to set up the FINN environment:
#   source scripts/activate.sh
#
# RENAMED from scripts/finn-env.sh, which was a near-homograph of
# docker/finn-env doing an unrelated job -- and did not even wrap it. The two
# programs it DOES use are named below.
#
# This script:
#   - Activates the Python virtual environment
#   - Sets FINN-specific environment variables
#   - Sources Xilinx tools if configured
#   - Sets up library paths for finn_xsi

# Color definitions
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

_finn_gecho() {
    echo -e "${GREEN}$1${NC}"
}

_finn_yecho() {
    echo -e "${YELLOW}$1${NC}"
}

# Determine FINN_ROOT from script location
if [ -n "${BASH_SOURCE[0]}" ]; then
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export FINN_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
else
    # Fallback if BASH_SOURCE is not available
    if [ -z "$FINN_ROOT" ]; then
        echo "ERROR: Could not determine FINN_ROOT. Please set it manually."
        return 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "$FINN_ROOT/.venv" ]; then
    echo "ERROR: Virtual environment not found at $FINN_ROOT/.venv"
    echo "Please run ./setup-local.sh first."
    return 1
fi

# Activate virtual environment
source "$FINN_ROOT/.venv/bin/activate"
_finn_gecho "Activated FINN environment at $FINN_ROOT"

# Set FINN environment variables
# The build directory, from finn-env -- which resolves it and creates it. This
# used to default to /tmp/finn_local_$(whoami), a fifth answer to a question
# that should have one.
eval "$("$FINN_ROOT/docker/finn-env" inspect --tier dev --format sh 2>/dev/null \
        | grep '^FINN_HOST_BUILD_DIR=' | sed 's/^/export /')"
export FINN_BUILD_DIR="$FINN_HOST_BUILD_DIR"

# Board files path
export FINN_BOARD_FILES_PATH="$FINN_ROOT/deps/board_files"

# Xilinx tools setup
# The Xilinx toolchain, resolved by docker/finn-env -- the same program the
# container uses, and the same one setup-local.sh calls.
#
# This block used to be a fifth copy of the toolchain logic, and it carried the
# same defect the others did:
#
#     export VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"
#
# which is the PRE-2024.2 directory layout only. AMD reorganised the tree after
# 2024.2, so on any recent installation this reported "Vivado not found" at a
# path the user could see was wrong. finn-env probes both layouts.
#
# It also duplicated the LD_LIBRARY_PATH additions (lib/lnx64.o, fpo_v7_1) and
# the XRT sourcing, both of which finn-env now owns.
if [ -n "$FINN_XILINX_PATH" ] && [ -n "$FINN_XILINX_VERSION" ]; then
    # Two steps, and the split is the point. finn-env probes the host and says
    # WHERE the tools are; finn-toolchain.sh takes that and applies it. The
    # image sources the same second file, so the bare host and the container
    # apply the toolchain through identical code.
    eval "$("$FINN_ROOT/docker/finn-env" inspect --tier build --format sh 2>/dev/null | sed 's/^/export /')"
    . "$FINN_ROOT/docker/finn-toolchain.sh"

    if [ -n "${XILINX_VIVADO:-}" ]; then
        _finn_gecho "Xilinx toolchain configured (finn-env): $XILINX_VIVADO"
    else
        _finn_yecho "No Vivado found under $FINN_XILINX_PATH for $FINN_XILINX_VERSION"
    fi
else
    _finn_yecho "FINN_XILINX_PATH and/or FINN_XILINX_VERSION not set"
    _finn_yecho "Vivado, Vitis, HLS and rtlsim are unavailable."
fi

export XILINX_LOCAL_USER_DATA=no

# Handle Xilinx init scripts if present
if [ -d "$FINN_ROOT/.Xilinx" ]; then
    if [ ! -d "$HOME/.Xilinx" ]; then
        mkdir -p "$HOME/.Xilinx"
    fi
    if [ -f "$FINN_ROOT/.Xilinx/HLS_init.tcl" ]; then
        cp "$FINN_ROOT/.Xilinx/HLS_init.tcl" "$HOME/.Xilinx/"
    fi
    if [ -f "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" ]; then
        mkdir -p "$HOME/.Xilinx/Vivado"
        cp "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" "$HOME/.Xilinx/Vivado/"
    fi
fi

# Vivado IP cache directory
export VIVADO_IP_CACHE="${VIVADO_IP_CACHE:-$FINN_BUILD_DIR/vivado_ip_cache}"
mkdir -p "$VIVADO_IP_CACHE"

echo ""
_finn_gecho "FINN environment ready!"
echo "  FINN_ROOT=$FINN_ROOT"
echo "  FINN_BUILD_DIR=$FINN_BUILD_DIR"
if [ -n "$XILINX_VIVADO" ]; then
    echo "  XILINX_VIVADO=$XILINX_VIVADO"
fi
echo ""
