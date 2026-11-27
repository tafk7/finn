# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# FINN's container environment, sourced by the entrypoint, by login shells and, as
# BASH_ENV, by every bash (so `docker exec` sessions get it too); once per process
# tree. A home and identity for the caller's uid, and AMD's environment: the launcher
# passes XILINX_VIVADO, and this sources that release's settings64.sh, as a user's
# shell would natively (and the image's XRT setup, when it has XRT). A job that names
# the release with run-docker.sh's FINN_XILINX_PATH and FINN_XILINX_VERSION instead
# gets the same, in either of AMD's layouts, and the names run-docker.sh gave the
# release (VIVADO_PATH, VITIS_PATH, HLS_PATH), which a site's LSF wrapper forwards:
# scripts/xilinx-release.sh, which CI's native stage sources as well.
#
# A caller's BASH_ENV (a site's, such as an LSF kit's enable.sh) replaces the image's
# one: this keeps its own and sources the caller's after it, on every bash.
if [ -n "${BASH_ENV:-}" ] && [ "$BASH_ENV" != /etc/profile.d/finn.sh ]; then
    export FINN_CALLER_BASH_ENV="$BASH_ENV" BASH_ENV=/etc/profile.d/finn.sh
fi
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
    # The release (scripts/xilinx-release.sh, which a native shell sources too): in
    # the image at /opt/finn, in a checkout beside docker/.
    finn_release=/opt/finn/xilinx-release.sh
    case ${BASH_SOURCE:-} in
        docker/profile.sh | */docker/profile.sh)
            finn_release=${BASH_SOURCE%docker/profile.sh}scripts/xilinx-release.sh
            ;;
    esac
    if ! . "$finn_release"; then
        unset finn_release
        # Unmarked, so that the entrypoint's own sourcing fails too and stops the start.
        unset FINN_PROFILE
        return 1
    fi
    unset finn_release
    if [ -f /opt/xilinx/xrt/setup.sh ]; then
        . /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true
    fi
    # FINN_TOOL_DIR_OVERRIDE's tools replace PATH's (finn.util.launcher), so put them
    # first for every other caller too, ahead of the release's own.
    if [ -n "${FINN_TOOL_DIR_OVERRIDE:-}" ]; then
        finn_path=":$PATH:"
        finn_dir=":$FINN_TOOL_DIR_OVERRIDE:"
        while case $finn_path in *"$finn_dir"*) true ;; *) false ;; esac do
            finn_path="${finn_path%%"$finn_dir"*}:${finn_path#*"$finn_dir"}"
        done
        finn_path=${finn_path#:}
        finn_path=${finn_path%:}
        export PATH="$FINN_TOOL_DIR_OVERRIDE${finn_path:+:$finn_path}"
        unset finn_path finn_dir
    fi
fi
if [ -n "${FINN_CALLER_BASH_ENV:-}" ]; then
    . "$FINN_CALLER_BASH_ENV" \
        || echo "finn: the caller's BASH_ENV failed: $FINN_CALLER_BASH_ENV" >&2
fi
