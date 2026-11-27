#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Container start: the container environment (docker/profile.sh), an identity for the
# caller's uid, and in the development image FINN itself from the mounted checkout
# (FINN_ROOT), installed editable over the image's locked dependencies.
set -eo pipefail
. /etc/profile.d/finn.sh
mkdir -p "$HOME" "$FINN_HOME"
export USER=${USER:-finn} LOGNAME=${LOGNAME:-${USER:-finn}}
# Assigned first, so that a failure stops the start (set -e does not see into eval).
identity=$(finn-identity --directory "$FINN_HOME/identity/$(id -u)" --username "$USER" --home "$HOME")
eval "$identity"
if [ -d /run/finn/ssh ] && [ ! -e "$HOME/.ssh" ]; then
    ln -s /run/finn/ssh "$HOME/.ssh"
fi
# The release image has FINN installed from its wheel.
if [ -n "${FINN_ROOT:-}" ] && [ -z "${FINN_RELEASE:-}" ]; then
    cmp -s "$FINN_ROOT/uv.lock" /opt/finn/uv.lock || {
        echo "finn: $FINN_ROOT/uv.lock differs from the image's; rebuild it: docker/run --build" >&2
        exit 1
    }
    uv pip install --quiet --no-deps --no-build-isolation -e "$FINN_ROOT"
    # The checkout's opt-in Vivado initialization scripts (.Xilinx), in the caller's HOME.
    if [ -d "$FINN_ROOT/.Xilinx" ]; then
        mkdir -p "$HOME/.Xilinx" && cp -a "$FINN_ROOT/.Xilinx/." "$HOME/.Xilinx/"
    fi
fi
# Tools routed outside the container (FINN_TOOL_DIR_OVERRIDE: an LSF farm) cannot see
# the image's resource cache: copy finn-hlslib, verified, into the host's FINN_HOME,
# which lookups search first.
if [ -n "${FINN_TOOL_DIR_OVERRIDE:-}" ]; then
    python -m finn.resources fetch --kind hls-include >/dev/null
fi
# execute the provided command(s)
exec "$@"
