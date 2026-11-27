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
# release (VIVADO_PATH, VITIS_PATH, HLS_PATH), which a site's LSF wrapper forwards.
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
    if [ -z "${XILINX_VIVADO:-}" ] && [ -n "${FINN_XILINX_PATH:-}" ] \
        && [ -n "${FINN_XILINX_VERSION:-}" ]; then
        # <path>/<release>/Vivado from 2025.1, <path>/Vivado/<release> before.
        for finn_vivado in "$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vivado" \
            "$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"; do
            if [ -f "$finn_vivado/settings64.sh" ]; then
                export XILINX_VIVADO="$finn_vivado"
                # Vitis and (before 2024.2) Vitis HLS are its siblings, in either layout.
                finn_vitis=$(dirname "$finn_vivado")/Vitis
                finn_hls=$(dirname "$finn_vivado")/Vitis_HLS
                case $finn_vivado in
                    */Vivado/"$FINN_XILINX_VERSION")
                        finn_vitis=$FINN_XILINX_PATH/Vitis/$FINN_XILINX_VERSION
                        finn_hls=$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION
                        ;;
                esac
                break
            fi
        done
        unset finn_vivado
        if [ -z "${XILINX_VIVADO:-}" ]; then
            echo "finn: no settings64.sh for FINN_XILINX_VERSION=$FINN_XILINX_VERSION under" \
                "FINN_XILINX_PATH=$FINN_XILINX_PATH (in <path>/<version>/Vivado or" \
                "<path>/Vivado/<version>)" >&2
            # Unmarked, so that the entrypoint's own sourcing fails too and stops the start.
            unset FINN_PROFILE
            return 1
        fi
    fi
    # env scripts may return non-zero, so do not let that abort the container
    if [ -n "${XILINX_VIVADO:-}" ] && [ -f "$XILINX_VIVADO/settings64.sh" ]; then
        . "$XILINX_VIVADO/settings64.sh" >/dev/null 2>&1 || true
    fi
    # A job's separate trees (Vivado/<release>, Vitis/<release>): what Vivado's
    # settings64.sh did not set up, as run-docker.sh's entrypoint sourced them.
    if [ -z "${XILINX_VITIS:-}" ] && [ -n "${finn_vitis:-}" ] \
        && [ -f "$finn_vitis/settings64.sh" ]; then
        export XILINX_VITIS="$finn_vitis"
        . "$finn_vitis/settings64.sh" >/dev/null 2>&1 || true
    fi
    if [ -z "${XILINX_HLS:-}" ] && [ -n "${finn_hls:-}" ] \
        && [ -f "$finn_hls/settings64.sh" ]; then
        export XILINX_HLS="$finn_hls"
        . "$finn_hls/settings64.sh" >/dev/null 2>&1 || true
    fi
    unset finn_vitis finn_hls
    # run-docker.sh's names for the release, which an LSF wrapper forwards to the farm.
    if [ -n "${XILINX_VIVADO:-}" ]; then
        export VIVADO_PATH="${VIVADO_PATH:-$XILINX_VIVADO}"
    fi
    if [ -n "${XILINX_VITIS:-}" ]; then
        export VITIS_PATH="${VITIS_PATH:-$XILINX_VITIS}"
    fi
    if [ -n "${XILINX_HLS:-${XILINX_VITIS:-}}" ]; then
        export HLS_PATH="${HLS_PATH:-${XILINX_HLS:-$XILINX_VITIS}}"
    fi
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
