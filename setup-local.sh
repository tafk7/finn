#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# setup-local.sh - Set up FINN for local (non-Docker) execution
#
# This script provides an alternative to the Docker-based setup for environments
# where Docker is not available.
#
# Usage:
#   ./setup-local.sh [OPTIONS]
#
# Options:
#   --help          Show this help message
#   --ci            CI mode (non-interactive, fail fast on errors)
#   --skip-xsi      Skip building finn_xsi (Vivado Python interface)
#   --skip-deps     Skip fetching git dependencies (assumes fetch-repos.sh already run)

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Colored output helpers
gecho() {
    echo -e "${GREEN}$1${NC}"
}

yecho() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

recho() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Script configuration
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
export FINN_ROOT="$SCRIPTPATH"

# Default values
CI_MODE=0
SKIP_XSI=0
SKIP_DEPS=0
VENV_DIR="$FINN_ROOT/.venv"

# Parse arguments
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Set up FINN for local (non-Docker) execution."
    echo ""
    echo "Options:"
    echo "  --help          Show this help message"
    echo "  --ci            CI mode (non-interactive, fail fast on errors)"
    echo "  --skip-xsi      Skip building finn_xsi (Vivado Python interface)"
    echo "  --skip-deps     Skip fetching git dependencies"
    echo ""
    echo "Environment variables:"
    echo "  FINN_XILINX_PATH     Path to Xilinx tools (e.g., /opt/Xilinx)"
    echo "  FINN_XILINX_VERSION  Xilinx tools version (e.g., 2022.2)"
    echo "  FINN_BUILD_DIR       Build output directory (default: /tmp/finn_local_\$USER)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            print_usage
            exit 0
            ;;
        --ci)
            CI_MODE=1
            shift
            ;;
        --skip-xsi)
            SKIP_XSI=1
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=1
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

# Step 1: Check prerequisites
gecho "Step 1: Checking prerequisites..."

# Check Python version (require 3.10+)
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
    recho "Python 3.10 or higher required, found $PYTHON_VERSION"
    recho "Set FINN_PYTHON to point to a Python 3.10+ interpreter"
    exit 1
fi
gecho "  Python $PYTHON_VERSION - OK"

# Check for required system tools
check_command() {
    if ! command -v "$1" &> /dev/null; then
        recho "$1 not found. Please install it first."
        echo "  Run: sudo apt-get install $2"
        exit 1
    fi
    gecho "  $1 - OK"
}

check_command "g++" "build-essential g++"
check_command "git" "git"
check_command "pip3" "python3-pip"

# Check for setup.py (verify we're in FINN root)
if [ ! -f "${FINN_ROOT}/setup.py" ]; then
    recho "setup.py not found in ${FINN_ROOT}"
    recho "Please run this script from the FINN repository root"
    exit 1
fi
gecho "  FINN source - OK"

echo ""

# Step 2: Fetch git dependencies
if [ "$SKIP_DEPS" -eq 0 ]; then
    gecho "Step 2: Fetching git dependencies..."
    if [ -f "${FINN_ROOT}/fetch-repos.sh" ]; then
        bash "${FINN_ROOT}/fetch-repos.sh"
    else
        recho "fetch-repos.sh not found"
        exit 1
    fi
else
    yecho "Step 2: Skipping git dependencies (--skip-deps)"
fi

echo ""

# Step 3: Create virtual environment
gecho "Step 3: Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    yecho "Virtual environment already exists at $VENV_DIR"
    yecho "Reusing existing environment. Delete .venv to start fresh."
else
    # Use FINN_PYTHON if set, otherwise look for Python 3.10
    if [ -z "$FINN_PYTHON" ]; then
        if [ -x "/usr/bin/python3.10" ]; then
            FINN_PYTHON="/usr/bin/python3.10"
        else
            FINN_PYTHON="python3"
        fi
    fi
    gecho "  Using Python: $FINN_PYTHON ($($FINN_PYTHON --version 2>&1))"
    "$FINN_PYTHON" -m venv "$VENV_DIR"
    gecho "  Created virtual environment at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
gecho "  Activated virtual environment"

# Upgrade pip and install essential build tools
pip install --upgrade pip setuptools wheel > /dev/null
gecho "  Upgraded pip, setuptools, wheel"

echo ""

# Step 4: Install Python dependencies
gecho "Step 4: Installing Python dependencies..."

# The SAME pin files the image uses. This lane previously carried 27 hardcoded
# `pip install` lines duplicating the image's tool pins -- another copy of the
# pin set, in the branch whose purpose is removing extra copies. It also
# installed the CUDA build of torch while the image deliberately installs CPU,
# so "all lanes use the same dependency versions" was false.
PIN_DIR="${FINN_ROOT}/docker"

pip install -r "${FINN_ROOT}/requirements.txt"
gecho "  Installed requirements.txt"

# CPU torch, matching docker/Dockerfile.finn. The CUDA wheels pull about 5 GB
# of nvidia/* and triton that FINN never calls: it compiles networks, it does
# not train them. If you want CUDA locally, install it yourself afterwards.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu
gecho "  Installed PyTorch (CPU)"

PIP_CONSTRAINT="$PIN_DIR/pip-constraints.txt" pip install -r "$PIN_DIR/pip-tools.txt"
if [ -f "$PIN_DIR/pip-tools-force-post.txt" ]; then
    pip install -r "$PIN_DIR/pip-tools-force-post.txt"
fi
gecho "  Installed the tool pins"

# Same check the image build runs, for the same reason: a resolvable install is
# not necessarily a consistent one.
pip check || yecho "pip check reported conflicts; see above"

# Install qonnx (with pyproject.toml workaround)
# See: https://github.com/pypa/pip/issues/7953
if [ -f "${FINN_ROOT}/deps/qonnx/pyproject.toml" ]; then
    mv "${FINN_ROOT}/deps/qonnx/pyproject.toml" "${FINN_ROOT}/deps/qonnx/pyproject.tmp"
    pip install -e "${FINN_ROOT}/deps/qonnx"
    mv "${FINN_ROOT}/deps/qonnx/pyproject.tmp" "${FINN_ROOT}/deps/qonnx/pyproject.toml"
else
    pip install -e "${FINN_ROOT}/deps/qonnx"
fi
gecho "  Installed qonnx"

# Install finn-experimental (use --no-build-isolation to avoid pkg_resources issues)
pip install --no-build-isolation -e "${FINN_ROOT}/deps/finn-experimental"
gecho "  Installed finn-experimental"

# Install brevitas
pip install -e "${FINN_ROOT}/deps/brevitas"
gecho "  Installed brevitas"

# Install FINN itself
pip install -e "${FINN_ROOT}"
gecho "  Installed finn"

echo ""

# Step 5: Check Xilinx tools (optional)
gecho "Step 5: Checking Xilinx tools..."

XILINX_AVAILABLE=0

if [ -n "$FINN_XILINX_PATH" ] && [ -n "$FINN_XILINX_VERSION" ]; then
    # Resolution is delegated to docker/finn-env, which is NOT container-specific
    # -- `print` and `exec` just resolve and source a toolchain wherever they
    # run. All three supported lanes (docker, sbx, bare host) therefore share
    # one implementation.
    #
    # This block used to hardcode
    #
    #     VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"
    #
    # which is the PRE-2024.2 layout only. AMD reorganised the tree after
    # 2024.2 to $ROOT/$VERSION/Vivado, so on any recent install this found
    # nothing and reported "Vivado not found" at a path the user could see was
    # wrong. That is the same defect that had the sbx backend silently mounting
    # no toolchain -- the fourth instance of one fact being derived in a fourth
    # place. finn-env probes both layouts rather than assuming either.
    eval "$("${FINN_ROOT}/docker/finn-env" inspect --tier build --format sh 2>/dev/null | sed 's/^/export /')"

    if [ -n "${XILINX_VIVADO:-}" ]; then
        gecho "  Found Vivado at $XILINX_VIVADO"
        XILINX_AVAILABLE=1
    else
        yecho "Vivado not found under $FINN_XILINX_PATH for version $FINN_XILINX_VERSION"
    fi
    [ -n "${XILINX_VITIS:-}" ] && gecho "  Found Vitis at $XILINX_VITIS" \
        || yecho "Vitis not found (optional, for Alveo)"
    [ -n "${XILINX_HLS:-}" ] && gecho "  Found Vitis HLS at $XILINX_HLS" \
        || yecho "Vitis HLS not found"

    if [ "$XILINX_AVAILABLE" -eq 1 ]; then
        # Source the toolchain into THIS shell, the same way the container does.
        eval "$("${FINN_ROOT}/docker/finn-env" print --format sh 2>/dev/null)"
        export FINN_ENV_APPLIED=1
        # The FLEXlm/libudev workaround. Baked as ENV in the container image;
        # on a bare host it has to be applied here. Without it a licence
        # checkout dies with "realloc(): invalid pointer" inside
        # udev_enumerate_scan_devices.
        libudev=$(ls /lib/*-linux-gnu/libudev.so.1 2>/dev/null | head -1)
        if [ -n "$libudev" ]; then
            export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$libudev"
        else
            yecho "libudev.so.1 not found; licence checkout may abort"
        fi
    fi
else
    yecho "FINN_XILINX_PATH and/or FINN_XILINX_VERSION not set"
    yecho "Xilinx tools will not be available. Set these variables for full functionality."
fi

echo ""

# Step 6: Build finn_xsi (if Vivado available)
if [ "$SKIP_XSI" -eq 0 ] && [ "$XILINX_AVAILABLE" -eq 1 ]; then
    gecho "Step 6: Building finn_xsi..."

    if [ -f "${FINN_ROOT}/finn_xsi/xsi.so" ]; then
        gecho "  Found existing finn_xsi at ${FINN_ROOT}/finn_xsi/xsi.so"
    else
        python -m finn.xsi.setup --quiet
        if [ $? -eq 0 ]; then
            gecho "  finn_xsi built successfully"
        else
            yecho "Failed to build finn_xsi - RTL simulation may not work"
        fi
    fi
elif [ "$SKIP_XSI" -eq 1 ]; then
    yecho "Step 6: Skipping finn_xsi build (--skip-xsi)"
else
    yecho "Step 6: Skipping finn_xsi build (Vivado not available)"
fi

echo ""

# Step 7: Verify installation
gecho "Step 7: Verifying installation..."

python3 -c "import finn; import qonnx; import brevitas; print('  All imports successful')"
if [ $? -ne 0 ]; then
    recho "Import verification failed"
    exit 1
fi

echo ""
echo "=============================================="
gecho "FINN local setup complete!"
echo "=============================================="
echo ""
echo "To use FINN, activate the environment:"
echo "  source scripts/finn-env.sh"
echo ""
echo "To validate the installation:"
echo "  ./scripts/quicktest-local.sh"
echo ""
if [ "$XILINX_AVAILABLE" -eq 0 ]; then
    echo "Note: Xilinx tools not configured. For full functionality, set:"
    echo "  export FINN_XILINX_PATH=/opt/Xilinx"
    echo "  export FINN_XILINX_VERSION=2022.2"
    echo ""
fi
