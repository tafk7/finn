# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# AMD's environment for one release, sourced (POSIX sh) by the container's profile
# (docker/profile.sh) and by a native shell, such as CI's local setup stage:
#
#   FINN_XILINX_PATH=/tools/Xilinx FINN_XILINX_VERSION=2025.2 . scripts/xilinx-release.sh
#
# With XILINX_VIVADO set (a sourced settings64.sh, a site module, a launcher), that
# release. Otherwise a job written for run-docker.sh names it with FINN_XILINX_PATH and
# FINN_XILINX_VERSION, in either of AMD's layouts: <path>/<version>/Vivado (from
# 2025.1) or <path>/Vivado/<version> (separate trees, as on xbuilds); with neither, a
# return status of 1 and a message. This sources the release's settings64.sh, and its
# Vitis and (before 2024.2) Vitis HLS beside it for what Vivado's did not set up, as
# run-docker.sh's entrypoint did, and sets run-docker.sh's names for the release
# (VIVADO_PATH, VITIS_PATH, HLS_PATH), which a site's LSF wrapper forwards. With neither
# XILINX_VIVADO nor the pair, it does nothing.
finn_vitis=
finn_hls=
if [ -z "${XILINX_VIVADO:-}" ] && [ -n "${FINN_XILINX_PATH:-}" ] \
    && [ -n "${FINN_XILINX_VERSION:-}" ]; then
    for finn_vivado in "$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vivado" \
        "$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"; do
        if [ -f "$finn_vivado/settings64.sh" ]; then
            export XILINX_VIVADO="$finn_vivado"
            # Vitis and Vitis HLS are its siblings, in either layout.
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
        unset finn_vitis finn_hls
        return 1
    fi
fi
# AMD's scripts may return non-zero; that does not fail the caller.
if [ -n "${XILINX_VIVADO:-}" ] && [ -f "$XILINX_VIVADO/settings64.sh" ]; then
    . "$XILINX_VIVADO/settings64.sh" >/dev/null 2>&1 || true
fi
if [ -z "${XILINX_VITIS:-}" ] && [ -n "$finn_vitis" ] && [ -f "$finn_vitis/settings64.sh" ]; then
    export XILINX_VITIS="$finn_vitis"
    . "$finn_vitis/settings64.sh" >/dev/null 2>&1 || true
fi
if [ -z "${XILINX_HLS:-}" ] && [ -n "$finn_hls" ] && [ -f "$finn_hls/settings64.sh" ]; then
    export XILINX_HLS="$finn_hls"
    . "$finn_hls/settings64.sh" >/dev/null 2>&1 || true
fi
unset finn_vitis finn_hls
if [ -n "${XILINX_VIVADO:-}" ]; then
    export VIVADO_PATH="${VIVADO_PATH:-$XILINX_VIVADO}"
fi
if [ -n "${XILINX_VITIS:-}" ]; then
    export VITIS_PATH="${VITIS_PATH:-$XILINX_VITIS}"
fi
if [ -n "${XILINX_HLS:-${XILINX_VITIS:-}}" ]; then
    export HLS_PATH="${HLS_PATH:-${XILINX_HLS:-$XILINX_VITIS}}"
fi
true
