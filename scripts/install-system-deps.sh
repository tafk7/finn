#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# install-system-deps.sh - Install system dependencies for FINN
#
# This script installs the system packages required to run FINN locally
# (without Docker). It requires sudo privileges.
#
# Usage:
#   sudo ./scripts/install-system-deps.sh
#
# Supported distributions:
#   - Ubuntu 22.04 (primary, tested)
#   - Debian 11+ (should work)
#   - Other apt-based distributions (may work)

set -e

# Color definitions
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

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    recho "This script requires root privileges."
    echo "Please run: sudo $0"
    exit 1
fi

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION=$VERSION_ID
else
    DISTRO="unknown"
    VERSION="unknown"
fi

gecho "Detected distribution: $DISTRO $VERSION"

# Install based on package manager
if command -v apt-get &> /dev/null; then
    gecho "Installing packages with apt-get..."

    # Allow update to continue even if some repos fail (e.g., stale third-party repos)
    apt-get update || yecho "apt-get update had some failures, continuing anyway..."

    # Essential: build tools and Python
    apt-get install -y \
        build-essential \
        g++ \
        git \
        wget \
        unzip \
        python3 \
        python3-pip \
        python3-venv

    # Required for finn_xsi (Vivado Python bindings) and finn-hlslib
    apt-get install -y \
        pybind11-dev \
        libboost-dev

    # ---------------------------------------------------------------------
    # The two things the container image ASSERTS at build time, and which this
    # script used to omit.
    #
    # docker/Dockerfile.finn fails the build if either is missing. The bare-host
    # lane had no equivalent: the ncurses 5 ABI was demoted to a printed
    # suggestion ("may be required by some Vivado versions") and the LSB loader
    # was absent entirely. Both are hard requirements for running Vivado, so the
    # bare-host lane was silently narrower than the lane it claims to mirror --
    # the same defect shape as the mount bugs, applied to OS packages.
    # ---------------------------------------------------------------------

    # ncurses 5 ABI. Vivado links against it. focal and jammy ship it; noble
    # dropped it for ncurses 6, so ask apt and backport from the jammy pool when
    # the distro has nothing. Checksums match Dockerfile.finn.
    NCURSES5_VERSION=6.3-2ubuntu0.2
    LIBTINFO5_SHA256=b9bb64e716a7d9de05b1b33992763142ca81bcae3a7f8ce7e29fa3c6fd32f1e8
    LIBNCURSES5_SHA256=91d18fcc4165a40d27e8181eb282bcaf89c2a5e6c6dc182b37df33827407361c
    if apt-cache show libtinfo5 2>/dev/null | grep -q '^Package:'; then
        apt-get install -y libtinfo5 libncurses5
    else
        . /etc/os-release
        yecho "ncurses 5 ABI absent on $VERSION_ID, backporting from the jammy pool"
        pool=http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses
        arch=$(dpkg --print-architecture)
        wget -q "$pool/libtinfo5_${NCURSES5_VERSION}_${arch}.deb"   -O /tmp/libtinfo5.deb
        wget -q "$pool/libncurses5_${NCURSES5_VERSION}_${arch}.deb" -O /tmp/libncurses5.deb
        echo "$LIBTINFO5_SHA256  /tmp/libtinfo5.deb"     | sha256sum -c -
        echo "$LIBNCURSES5_SHA256  /tmp/libncurses5.deb" | sha256sum -c -
        apt-get install -y /tmp/libtinfo5.deb /tmp/libncurses5.deb
        rm -f /tmp/libtinfo5.deb /tmp/libncurses5.deb
    fi
    ldconfig
    if [ ! -e "/usr/lib/$(uname -m)-linux-gnu/libtinfo.so.5" ]; then
        recho "libtinfo.so.5 missing after install; Vivado will not start"
        exit 1
    fi

    # LSB loader. lmutil and several tool wrappers request the interpreter
    # /lib64/ld-lsb-x86-64.so.3. Without it they fail to exec with "No such file
    # or directory" naming a file that plainly exists -- the classic missing-ELF-
    # interpreter error, and a confusing one to chase. `lsb-core` no longer
    # exists on noble, so create the alias directly.
    if [ ! -e /lib64/ld-lsb-x86-64.so.3 ]; then
        loader=$(ls /lib/*-linux-gnu/ld-linux-x86-64.so.2 2>/dev/null | head -1)
        if [ -n "$loader" ]; then
            mkdir -p /lib64
            ln -sf "$loader" /lib64/ld-lsb-x86-64.so.3
            gecho "  Created the LSB loader alias for lmutil"
        else
            yecho "no ld-linux-x86-64.so.2 found; lmutil will fail to exec"
        fi
    fi

    gecho "System dependencies installed successfully!"
    echo ""
    echo "This is a SUBSET of what docker/Dockerfile.finn installs -- enough to"
    echo "build and run FINN on a bare host. The image additionally carries"
    echo "tcsh (LSF esub), locales, cmake, g++-10 and several -dev packages that"
    echo "matter only inside the container."
    echo ""
    echo "Optional packages (install if needed):"
    echo "  - libglib2.0-0 libsm6 libxext6 libxrender-dev: for matplotlib/visualization"

elif command -v dnf &> /dev/null; then
    yecho "DNF (Fedora/RHEL) detected but not officially supported."
    yecho "Attempting to install equivalent packages..."

    dnf install -y \
        gcc-c++ \
        make \
        git \
        wget \
        unzip \
        zip \
        rsync \
        python3 \
        python3-pip \
        python3-devel \
        pybind11-devel \
        boost-devel \
        fmt-devel \
        jansson-devel \
        glib2 \
        libSM \
        libXext \
        libXrender

    yecho "Packages installed, but this configuration is not tested."

elif command -v yum &> /dev/null; then
    yecho "YUM (CentOS/RHEL) detected but not officially supported."
    recho "Please install the following packages manually:"
    echo "  - gcc-c++ make git wget unzip"
    echo "  - python3 python3-pip python3-devel"
    echo "  - pybind11-devel boost-devel"
    exit 1

else
    recho "Unsupported package manager."
    recho "Please install the following packages manually:"
    echo ""
    echo "Build tools:"
    echo "  - g++ (GCC C++ compiler)"
    echo "  - make"
    echo "  - git"
    echo "  - wget, unzip, zip, rsync"
    echo ""
    echo "Python:"
    echo "  - python3 (3.9 or higher)"
    echo "  - python3-pip"
    echo "  - python3-venv"
    echo "  - python3-dev (development headers)"
    echo ""
    echo "Libraries:"
    echo "  - pybind11 development files"
    echo "  - boost development files"
    echo "  - fmt development files"
    echo "  - jansson development files"
    echo ""
    exit 1
fi

echo ""
gecho "Done! You can now run ./setup-local.sh to set up FINN."
