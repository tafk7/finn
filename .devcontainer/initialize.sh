#!/usr/bin/env bash
# Resolve/reuse the dependency artifact through the same Bake entry point.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./docker/lib.sh
export FINN_ARTIFACT=dependencies FINN_RUNTIMES=""
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
