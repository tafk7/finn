#!/usr/bin/env bash
# Compiler tests use the full runtime profile (FINN_RUNTIMES, default xrt)
# and require the AMD tools. A prepared image in FINN_IMAGE (from `finn_ci image
# build` or `load`) is run as it is. Host-side finn_ci and board-side PYNQ/XRT
# environments remain separate.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
args=(--fpga --runtime "${FINN_RUNTIMES:-xrt}")
[ -z "${FINN_IMAGE:-}" ] || args+=(--no-build)
exec "$root/docker/run" "${args[@]}" -- "$@"
