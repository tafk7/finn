#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# setup-local.sh - Set up a FINN development environment on the host (no Docker)
#
# The Python environment is plain uv: `git submodule update --init && uv sync`
# creates .venv with FINN (editable), its workspace members and the locked
# dependencies. This script adds prerequisite checks, Xilinx detection and the
# optional XSI build around that.
#
# Usage:
#   ./setup-local.sh [OPTIONS]
#
# Options:
#   --help          Show this help message
#   --check         Validate prerequisites without changing files
#   --skip-xsi      Skip building finn_xsi (Vivado Python interface)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

gecho() {
    echo -e "${GREEN}$1${NC}"
}
yecho() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}
recho() {
    echo -e "${RED}ERROR: $1${NC}"
}

SCRIPT=$(readlink -f "$0")
FINN_ROOT=$(dirname "$SCRIPT")
export FINN_ROOT

SKIP_XSI=0
CHECK_ONLY=0
# FINN_VENV selects a venv other than .venv.
if [ -n "${FINN_VENV:-}" ]; then
    export UV_PROJECT_ENVIRONMENT="$FINN_VENV"
fi

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Set up a FINN development environment on the host (no Docker)."
    echo ""
    echo "Options:"
    echo "  --help          Show this help message"
    echo "  --check         Validate prerequisites without installing"
    echo "  --skip-xsi      Skip building finn_xsi (Vivado Python interface)"
    echo ""
    echo "Environment variables:"
    echo "  FINN_VENV            Virtual environment path (default: .venv)"
    echo "  FINN_XILINX_PATH     Path to Xilinx tools (e.g., /opt/Xilinx)"
    echo "  FINN_XILINX_VERSION  Xilinx tools version (e.g., 2024.2)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            print_usage
            exit 0
            ;;
        --skip-xsi)
            SKIP_XSI=1
            shift
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        *)
            recho "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "FINN Local Setup"
echo "=============================================="
echo ""

gecho "Step 1: Checking prerequisites..."
HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" != "x86_64" ]; then
    recho "Native FINN setup supports x86-64; found $HOST_ARCH"
    recho "Use ./docker/run for the Docker-built environment."
    exit 1
fi
if [ ! -r /etc/os-release ]; then
    recho "Cannot identify the host distribution from /etc/os-release"
    recho "Use ./docker/run for the Docker-built environment."
    exit 1
fi
HOST_OS=$(. /etc/os-release; printf '%s' "$ID")
HOST_OS_VERSION=$(. /etc/os-release; printf '%s' "$VERSION_ID")
if [ "$HOST_OS" != ubuntu ] || [ "$HOST_OS_VERSION" != 24.04 ]; then
    yecho "FINN's hardware flows are tested on Ubuntu 24.04; found $HOST_OS $HOST_OS_VERSION"
    yecho "Python-only use works elsewhere; for Vivado flows consider ./docker/run."
fi
gecho "  $HOST_OS $HOST_OS_VERSION / $HOST_ARCH - OK"

check_command() {
    if ! command -v "$1" &> /dev/null; then
        recho "$1 not found. $2"
        exit 1
    fi
    gecho "  $1 - OK"
}
check_command "g++" "Install it with: sudo apt-get install build-essential"
check_command "git" "Install it with: sudo apt-get install git"
check_command "uv" "Install it from https://docs.astral.sh/uv/getting-started/installation/"

if [ ! -f "${FINN_ROOT}/pyproject.toml" ]; then
    recho "pyproject.toml not found in ${FINN_ROOT}"
    exit 1
fi
gecho "  FINN source - OK"

if [ "$CHECK_ONLY" -eq 1 ]; then
    gecho "Native FINN prerequisites are satisfied."
    exit 0
fi
echo ""

gecho "Step 2: Creating the Python environment..."
# Workspace members (packages/*) are submodules at their pinned commits.
git -C "${FINN_ROOT}" submodule update --init
# The exact locked environment; Python 3.12 is provided by uv if the host lacks it.
uv sync --frozen --project "${FINN_ROOT}"
gecho "  ${UV_PROJECT_ENVIRONMENT:-${FINN_ROOT}/.venv}: FINN (editable) and locked dependencies"
echo ""

gecho "Step 3: Checking Xilinx tools..."
XILINX_AVAILABLE=0
if [ -n "$FINN_XILINX_PATH" ] && [ -n "$FINN_XILINX_VERSION" ]; then
    # docker/config.py locates the tools (both AMD install layouts);
    # docker/finn-toolchain.sh applies them, as it does in the image.
    eval "$("${FINN_ROOT}/docker/config.py" inspect --tier build --format sh 2>/dev/null | sed 's/^/export /')"
    if [ -n "${XILINX_VIVADO:-}" ]; then
        gecho "  Found Vivado at $XILINX_VIVADO"
        XILINX_AVAILABLE=1
    else
        yecho "Vivado not found under $FINN_XILINX_PATH for version $FINN_XILINX_VERSION"
    fi
    if [ -n "${XILINX_VITIS:-}" ]; then
        gecho "  Found Vitis at $XILINX_VITIS"
    else
        yecho "Vitis not found (optional, for Alveo)"
    fi
    if [ -n "${XILINX_HLS:-}" ]; then
        gecho "  Found Vitis HLS at $XILINX_HLS"
    else
        yecho "Vitis HLS not found"
    fi
    if [ "$XILINX_AVAILABLE" -eq 1 ]; then
        . "${FINN_ROOT}/docker/finn-toolchain.sh"
    fi
else
    yecho "FINN_XILINX_PATH and/or FINN_XILINX_VERSION not set"
    yecho "Vivado, Vitis, HLS and rtlsim are unavailable until they are."
fi
echo ""

PYTHON="uv run --frozen --project ${FINN_ROOT} python"
if [ "$SKIP_XSI" -eq 0 ] && [ "$XILINX_AVAILABLE" -eq 1 ]; then
    gecho "Step 4: Preparing hardware flows..."
    $PYTHON -m finn.xsi.setup
    gecho "  finn_xsi built and verified"
    # Fetched on first use anyway; fetching now lets later builds run offline.
    $PYTHON -m finn.util.external fetch-boards >/dev/null
    gecho "  Board files fetched"
elif [ "$SKIP_XSI" -eq 1 ]; then
    yecho "Step 4: Skipping hardware preparation (--skip-xsi)"
else
    yecho "Step 4: Skipping hardware preparation (Vivado not available)"
fi
echo ""

gecho "Step 5: Verifying installation..."
$PYTHON -c "import finn.util.basic, qonnx, brevitas; print('  All imports successful')"
echo ""

echo "=============================================="
gecho "FINN local setup complete!"
echo "=============================================="
echo ""
echo "Activate the environment (and the Xilinx toolchain, if configured):"
echo "  source scripts/activate.sh"
echo ""
echo "After pulling changes to uv.lock or the submodules, update it with:"
echo "  git submodule update --init && uv sync"
echo ""
if [ "$XILINX_AVAILABLE" -eq 0 ]; then
    echo "Note: Xilinx tools not configured. For hardware flows, set:"
    echo "  export FINN_XILINX_PATH=/opt/Xilinx"
    echo "  export FINN_XILINX_VERSION=2024.2"
    echo ""
fi
