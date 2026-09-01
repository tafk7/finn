#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# setup-local.sh - Set up FINN for local (non-Docker) execution
#
# This is the primary setup path for the supported Ubuntu/Python host contract.
#
# Usage:
#   ./setup-local.sh [OPTIONS]
#
# Options:
#   --help          Show this help message
#   --check         Validate native-install prerequisites without changing files
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
SKIP_XSI=0
SKIP_DEPS=0
CHECK_ONLY=0
VENV_DIR="$FINN_ROOT/.venv"

# Parse arguments
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Set up FINN for local (non-Docker) execution."
    echo ""
    echo "Options:"
    echo "  --help          Show this help message"
    echo "  --check         Validate prerequisites without installing"
    echo "  --skip-xsi      Skip building finn_xsi (Vivado Python interface)"
    echo "  --skip-deps     Skip fetching git dependencies"
    echo ""
    echo "Environment variables:"
    echo "  FINN_XILINX_PATH     Path to Xilinx tools (e.g., /opt/Xilinx)"
    echo "  FINN_XILINX_VERSION  Xilinx tools version (e.g., 2022.2)"
    echo "  FINN_HOST_BUILD_DIR  Build output directory (default: /tmp/finn_build_<uid>)"
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
if [ "$HOST_OS" != ubuntu ] || [ "$HOST_OS_VERSION" != 22.04 ]; then
    recho "Native FINN setup supports Ubuntu 22.04; found $HOST_OS $HOST_OS_VERSION"
    recho "Use ./docker/run for the Docker-built environment."
    exit 1
fi
gecho "  Ubuntu $HOST_OS_VERSION / $HOST_ARCH - OK"

# Select the interpreter before validating it, so FINN_PYTHON can satisfy the
# requirement on a host whose default python3 is older.
if [ -z "${FINN_PYTHON:-}" ]; then
    if [ -x "/usr/bin/python3.10" ]; then
        FINN_PYTHON="/usr/bin/python3.10"
    else
        FINN_PYTHON="python3"
    fi
fi
if ! command -v "$FINN_PYTHON" >/dev/null 2>&1; then
    recho "Python interpreter not found: $FINN_PYTHON"
    exit 1
fi
PYTHON_VERSION=$("$FINN_PYTHON" --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

if [[ $PYTHON_MAJOR -ne 3 ]] || [[ $PYTHON_MINOR -ne 10 ]]; then
    recho "Native FINN setup supports Python 3.10; found $PYTHON_VERSION"
    recho "Set FINN_PYTHON to a Python 3.10 interpreter, or use ./docker/run."
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

# Check for setup.py (verify we're in FINN root)
if [ ! -f "${FINN_ROOT}/setup.py" ]; then
    recho "setup.py not found in ${FINN_ROOT}"
    recho "Please run this script from the FINN repository root"
    exit 1
fi
gecho "  FINN source - OK"

if [ "$CHECK_ONLY" -eq 1 ]; then
    gecho "Native FINN prerequisites are satisfied."
    exit 0
fi

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
    gecho "  Using Python: $FINN_PYTHON ($("$FINN_PYTHON" --version 2>&1))"
    "$FINN_PYTHON" -m venv "$VENV_DIR"
    gecho "  Created virtual environment at $VENV_DIR"
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
gecho "  Activated virtual environment"

# Upgrade pip and install essential build tools
python -m pip install --upgrade pip setuptools wheel > /dev/null
gecho "  Upgraded pip, setuptools, wheel"

echo ""

# Step 4: Install Python dependencies
gecho "Step 4: Installing Python dependencies..."

# Use the same ordered pin files and global constraints as the image build.
PIN_DIR="${FINN_ROOT}/docker"
export PIP_CONSTRAINT="$PIN_DIR/pip-constraints.txt"

python -m pip install -r "$PIN_DIR/pip-torch.txt"
gecho "  Installed PyTorch (CPU)"

python -m pip install -r "${FINN_ROOT}/requirements.txt"
gecho "  Installed requirements.txt"

python -m pip install -r "$PIN_DIR/pip-tools.txt"
python -m pip install --ignore-installed -r "$PIN_DIR/pip-post.txt"
python -m pip install -r "$PIN_DIR/pip-extra.txt"
gecho "  Installed the tool pins"

# Install qonnx (with pyproject.toml workaround)
# See: https://github.com/pypa/pip/issues/7953
QONNX_PYPROJECT="${FINN_ROOT}/deps/qonnx/pyproject.toml"
QONNX_PYPROJECT_TMP="${FINN_ROOT}/deps/qonnx/pyproject.tmp"
if [ ! -f "$QONNX_PYPROJECT" ] && [ -f "$QONNX_PYPROJECT_TMP" ]; then
    mv "$QONNX_PYPROJECT_TMP" "$QONNX_PYPROJECT"
fi
if [ -f "$QONNX_PYPROJECT" ]; then
    mv "$QONNX_PYPROJECT" "$QONNX_PYPROJECT_TMP"
    trap 'mv "$QONNX_PYPROJECT_TMP" "$QONNX_PYPROJECT"' EXIT
    python -m pip install -e "${FINN_ROOT}/deps/qonnx"
    mv "$QONNX_PYPROJECT_TMP" "$QONNX_PYPROJECT"
    trap - EXIT
else
    python -m pip install -e "${FINN_ROOT}/deps/qonnx"
fi
gecho "  Installed qonnx"

# Install finn-experimental (use --no-build-isolation to avoid pkg_resources issues)
python -m pip install --no-build-isolation -e "${FINN_ROOT}/deps/finn-experimental"
gecho "  Installed finn-experimental"

# Install brevitas
python -m pip install -e "${FINN_ROOT}/deps/brevitas"
gecho "  Installed brevitas"

# Install FINN itself
python -m pip install -e "${FINN_ROOT}"
gecho "  Installed finn"

python -m pip check
gecho "  Dependency metadata is consistent"

echo ""

# Step 5: Check Xilinx tools (optional)
gecho "Step 5: Checking Xilinx tools..."

XILINX_AVAILABLE=0

if [ -n "$FINN_XILINX_PATH" ] && [ -n "$FINN_XILINX_VERSION" ]; then
    # Two steps, and the split is the point: docker/config probes the host
    # and says WHERE the tools are, docker/finn-toolchain.sh applies them. The
    # image sources that same second file, so every lane applies the toolchain
    # through identical code.
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
    # place. docker/config probes both layouts rather than assuming either.
    eval "$("${FINN_ROOT}/docker/config" inspect --tier build --format sh 2>/dev/null | sed 's/^/export /')"

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
        # Source the toolchain into THIS shell, the same way the container does.
        # That includes the FLEXlm/libudev workaround, which used to be a
        # separate copy here and a byte-identical one in scripts/activate.sh.
        . "${FINN_ROOT}/docker/finn-toolchain.sh"
        case ":${LD_PRELOAD:-}:" in
            *libudev.so.1*) ;;
            *) yecho "libudev.so.1 not found; licence checkout may abort" ;;
        esac
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
        gecho "  finn_xsi built successfully"
    fi
elif [ "$SKIP_XSI" -eq 1 ]; then
    yecho "Step 6: Skipping finn_xsi build (--skip-xsi)"
else
    yecho "Step 6: Skipping finn_xsi build (Vivado not available)"
fi

echo ""

# Step 7: Verify installation
gecho "Step 7: Verifying installation..."

python -c "import finn; import qonnx; import brevitas; print('  All imports successful')"

echo ""
echo "=============================================="
gecho "FINN local setup complete!"
echo "=============================================="
echo ""
echo "To use FINN, activate the environment:"
echo "  source scripts/activate.sh"
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
