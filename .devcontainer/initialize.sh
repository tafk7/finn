#!/usr/bin/env bash
# Resolve (and build if missing) the FINN image through the same Bake entry point
# as docker/run, and point the Dev Container at it. Host inputs come from the same
# resolver as docker/run: the Xilinx toolchain and licence when this machine has
# one (~/.config/finn/xilinx.env or FINN_XILINX_PATH), and resource overrides.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./docker/lib.sh
export FINN_RUNTIMES=""
finn_set_provenance
finn_prepare_image "$(finn_bake_target "")"
FINN_DEVCONTAINER_INPUTS=$(./docker/config.py compose --tier auto --inputs-only --service dev)
export FINN_DEVCONTAINER_INPUTS
python3 - <<'PY'
import json
import os
from pathlib import Path

override = json.loads(os.environ["FINN_DEVCONTAINER_INPUTS"])
override["services"]["dev"]["image"] = os.environ["FINN_IMAGE"]
Path(".devcontainer/image.compose.json").write_text(json.dumps(override, indent=2) + "\n")
PY
