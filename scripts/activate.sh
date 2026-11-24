#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Source this file after setup-local.sh. It does not install packages.
#
# It activates the virtual environment and sets FINN_ROOT, which the notebooks and
# tutorials read. FINN uses the AMD tools this shell has: source a release's
# settings64.sh (or load the site's Vivado module) before or after, as for any AMD tool.
_finn_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_finn_venv=${FINN_VENV:-$_finn_root/.venv}
if [ ! -f "$_finn_venv/bin/activate" ]; then
    echo "FINN environment missing: run $_finn_root/setup-local.sh" >&2
    unset _finn_root _finn_venv
    return 1
fi
export FINN_ROOT=$_finn_root
. "$_finn_venv/bin/activate"
if [ -z "${XILINX_VIVADO:-}${XILINX_HLS:-}${XILINX_VITIS:-}" ]; then
    python -c 'from finn.util.launcher import setup_hint; print("FINN: no AMD tools in this shell. " + setup_hint())' >&2
fi
unset _finn_root _finn_venv
