#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
prefetch=0
for arg in "$@"; do
    case "$arg" in
        # Resources and the XSI binding are prepared on first use; these are no-ops.
        --ci|--skip-xsi|--skip-resources|--skip-deps) ;;
        --prefetch) prefetch=1 ;;
        --help|-h)
            echo 'Usage: setup-local.sh [--prefetch]'
            echo 'Requires Python 3.12, pip, git and a C++ compiler; bootstraps pinned uv locally if needed.'
            echo 'FINN_VENV selects the environment; FINN_PYTHON selects the interpreter.'
            echo '--prefetch fetches every declared resource now, for later use offline.'
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done
for command in git g++; do
    command -v "$command" >/dev/null || { echo "Missing prerequisite: $command" >&2; exit 1; }
done
. "$root/scripts/ensure-uv.sh"
cd "$root"
export UV_PYTHON_DOWNLOADS=${UV_PYTHON_DOWNLOADS:-never}
export UV_PROJECT_ENVIRONMENT=${FINN_VENV:-$root/.venv}
uv sync --locked --python "${FINN_PYTHON:-python3.12}"
if [ "$prefetch" = 1 ]; then
    "$UV_PROJECT_ENVIRONMENT/bin/python" -m finn.resources fetch --all
fi
. "$root/scripts/activate.sh"
python -c 'import finn, qonnx, brevitas, finnexperimental, torch, onnxruntime'
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
if [ "${FINN_VCK190_GOLDEN:-0}" = 1 ]; then
    python -m finn.resources.prepare golden
fi
printf 'FINN ready. Activate with: source %q/scripts/activate.sh\n' "$root"
