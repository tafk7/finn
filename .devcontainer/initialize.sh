#!/usr/bin/env bash
# Resolve (and build if missing) the FINN image through the same Bake entry point
# as docker/run, and point the Dev Container at it.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./docker/lib.sh
export FINN_RUNTIMES=""
finn_set_provenance
finn_prepare_image "$(finn_bake_target "")"
python3 - <<'PY'
import json
import os
from pathlib import Path

Path(".devcontainer/image.compose.json").write_text(
    json.dumps({"services": {"dev": {"image": os.environ["FINN_IMAGE"]}}}) + "\n"
)
PY
