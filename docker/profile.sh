# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# FINN's container environment, sourced by the entrypoint, by login shells and, as
# BASH_ENV, by every bash (so `docker exec` sessions get it too); once per process
# tree. A home and identity for the caller's uid, and AMD's environment: the launcher
# passes XILINX_VIVADO, and this sources that release's settings64.sh, as a user's
# shell would natively (and the image's XRT setup, when it has XRT).
if [ -z "${FINN_PROFILE:-}" ]; then
    export FINN_PROFILE=1
    # A caller uid has no passwd entry or writable image home.
    if [ -z "${HOME:-}" ] || [ "$HOME" = / ] || ! [ -w "$HOME" ]; then
        export HOME="/tmp/finn-home-$(id -u)"
    fi
    export FINN_HOME="${FINN_HOME:-$HOME/.finn}"
    if [ -f "$FINN_HOME/identity/$(id -u)/passwd" ]; then
        export NSS_WRAPPER_PASSWD="$FINN_HOME/identity/$(id -u)/passwd"
        export NSS_WRAPPER_GROUP="$FINN_HOME/identity/$(id -u)/group"
    fi
    # env scripts may return non-zero, so do not let that abort the container
    if [ -n "${XILINX_VIVADO:-}" ] && [ -f "$XILINX_VIVADO/settings64.sh" ]; then
        . "$XILINX_VIVADO/settings64.sh" >/dev/null 2>&1 || true
    fi
    if [ -f /opt/xilinx/xrt/setup.sh ]; then
        . /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true
    fi
fi
