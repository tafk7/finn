#!/bin/bash
# Copyright (c) 2021, Xilinx
# Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Container start: a usable HOME, then the mounted FINN checkout linked into the
# image's environment, then the command. Nothing here is fatal: a container must
# start even without a checkout (sbx starts PID 1 before any workspace is used).
set -e
if [ -z "${HOME:-}" ] || [ "$HOME" = / ] || ! mkdir -p "$HOME" 2>/dev/null || [ ! -w "$HOME" ]; then
    HOME="/tmp/finn-home-$(id -u)"
    export HOME
fi
mkdir -p "$HOME"
# `id -un` prints the numeric uid (and fails) for a uid with no passwd entry.
if [ -z "${USER:-}" ]; then
    USER=$(id -un 2>/dev/null) || true
    USER="${USER:-finn}"
fi
export USER
export LOGNAME="${LOGNAME:-$USER}"
export PATH="$PATH:$HOME/.local/bin"
# Keep uv's cache in the build directory, which outlives the container, so the
# editable builds below are reused instead of redone on every start.
if [ -z "${UV_CACHE_DIR:-}" ] && [ -n "${FINN_BUILD_DIR:-}" ] \
   && mkdir -p "$FINN_BUILD_DIR/.uv-cache" 2>/dev/null; then
    export UV_CACHE_DIR="$FINN_BUILD_DIR/.uv-cache"
fi

# In sbx the checkout is the sandbox's workspace, which sbx names WORKSPACE_DIR.
if [ -z "${FINN_ROOT:-}" ] && [ -n "${WORKSPACE_DIR:-}" ]; then
    export FINN_ROOT="$WORKSPACE_DIR"
fi

# Install the checkout at FINN_ROOT editable into /opt/venv, with any difference
# between its uv.lock and the image. This follows the
# committed lock exactly; set FINN_SYNC=0 to skip it.
finn_sync() {
    [ "${FINN_SYNC:-1}" != 0 ] || return 0
    root="${FINN_ROOT:-}"
    if [ -z "$root" ] || [ ! -f "$root/uv.lock" ]; then
        echo "finn: no FINN checkout at FINN_ROOT; using the image environment" >&2
        return 0
    fi
    set -- uv sync --frozen --inexact --quiet --project "$root" \
        --no-build-isolation-package finn
    # Offline first: normally nothing needs downloading. Online only when the
    # checkout's uv.lock asks for packages the image does not have.
    "$@" --offline 2>/dev/null || "$@" || {
        echo "finn: could not install $root into the image environment (see above)." >&2
        echo "finn: rebuild the image (docker/build), or run uv sync with network access." >&2
    }
}
# Readiness marker: docker exec and sbx exec do not wait for the entrypoint, so
# scripts that exec into a just-started container can wait for this file.
finn_sync_and_mark() {
    finn_sync || true
    touch /tmp/finn-ready 2>/dev/null || true
}
# sbx clone mode clones into FINN_ROOT after the container starts, so here it
# is a fresh volume (empty but for lost+found) or a clone in progress. Sync once
# git has finished the checkout, in the background so the sandbox's own command
# still starts at once.
fresh_volume() {
    for entry in "$1"/* "$1"/.[!.]* "$1"/..?*; do
        [ -e "$entry" ] || continue
        [ "$entry" = "$1/lost+found" ] || return 1
    done
}
root="${FINN_ROOT:-}"
if [ -n "$root" ] && [ ! -f "$root/uv.lock" ] \
   && { [ -e "$root/.git" ] || fresh_volume "$root"; }; then
    (
        for _ in $(seq 600); do
            if [ -f "$root/uv.lock" ] && [ -f "$root/.git/index" ] \
               && [ ! -e "$root/.git/index.lock" ]; then
                break
            fi
            sleep 1
        done
        finn_sync_and_mark
    ) &
else
    finn_sync_and_mark
fi

exec "$@"
