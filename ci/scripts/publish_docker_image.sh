#!/usr/bin/env bash
# Publish the prepared image and archive its provenance in the Jenkins workspace.
set -euo pipefail
[ "$#" -eq 2 ] || { echo 'Usage: publish_docker_image.sh IMAGE_DIR IMAGE' >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export FINN_RUNTIMES=${FINN_RUNTIMES:-xrt}
PYTHONPATH="$REPO/ci" exec python3 -m finn_ci image publish "$1" "$2" --workspace "$REPO/image-metadata"
