#!/usr/bin/env bash
# Build FINN images with bake and emit a provenance record.
#
#   ci/scripts/build-images.sh <target> [output-dir]
#
# e.g.  ci/scripts/build-images.sh finn-xrt "$IMAGE_DIR"
#
# Application identity includes the installed FINN source/resources, dependency
# pins and runtime selection. The digest identifies exact image contents;
# source revision/dirty metadata additionally describes the selected checkout.
# Explicit editable jobs must record their preparation separately.

set -euo pipefail

TARGET="${1:?usage: $0 <bake-target> [output-dir]}"
OUTDIR="${2:-}"

cd "$(dirname "$0")/../.."

# shellcheck source=docker/lib.sh
. ./docker/lib.sh

# Explicit artifact selection; Bake remains authoritative for tags/targets.
export FINN_ARTIFACT=application
case "$TARGET" in finn-dependencies*) export FINN_ARTIFACT=dependencies ;; esac
finn_set_provenance
FINN_COMMIT="$FINN_SOURCE_REVISION"

# FINN and its dependency metadata are installed in the application image.
if [ -n "${FINN_DEPS:-}" ]; then
    recho "FINN_DEPS was removed; prepare explicit installations instead."
    exit 1
fi

if [ "$FINN_SOURCE_DIRTY" = 1 ]; then
    echo "WARNING: building from a dirty tree; provenance records the commit," >&2
    echo "         which does not describe the uncommitted changes." >&2
fi

echo "Building bake target $TARGET"
finn_prepare_image "$TARGET" build

# Ask bake for the tag rather than recomputing it. This is the whole point of
# moving the rule into docker-bake.hcl: there is exactly one implementation.
#
# Through finn_bake_tag, which carries the `sed -n '/^{/,$p'` that strips bake's
# progress lines before the JSON. This script's own copy omitted it and would
# have died on any bake that printed one.
TAG="$FINN_IMAGE"
[ -n "$TAG" ] || { recho "could not resolve a tag for $TARGET"; exit 1; }

# Image ID, not RepoDigests. RepoDigests is populated only after a push, and is
# empty for a locally built image -- which is the case CI is in when it uses the
# OCI-archive transport rather than a registry.
DIGEST=$(docker image inspect --format '{{.Id}}' "$TAG")

# Read exact resolved dependency wheels/source commits from the built artifact.
# Do not substitute requested refs or the caller's edited checkout for image facts.
PROVENANCE_TMP=$(mktemp -d -t finn-provenance-XXXXXX)
trap 'rm -rf "$PROVENANCE_TMP"' EXIT
docker run --rm --network none --entrypoint cat "$TAG" /opt/finn/wheelhouse.json > "$PROVENANCE_TMP/dependencies.json"
: > "$PROVENANCE_TMP/application.sha256"
if [ "$FINN_ARTIFACT" = application ]; then
    docker run --rm --network none --entrypoint cat "$TAG" /opt/finn/application-wheel.sha256 > "$PROVENANCE_TMP/application.sha256"
fi
export TARGET TAG DIGEST FINN_COMMIT
PROVENANCE=$(python3 - "$PROVENANCE_TMP" <<'PY'
import json
import os
import sys
from pathlib import Path
records = Path(sys.argv[1])
values = os.environ
print(json.dumps({
    "target": values["TARGET"],
    "tag": values["TAG"],
    "image_digest": values["DIGEST"],
    "image_revision": values["FINN_IMAGE_REVISION"],
    "dependency_revision": values["FINN_DEPENDENCY_REVISION"],
    "artifact": values["FINN_ARTIFACT"],
    "application_wheel": (records / "application.sha256").read_text().strip() or None,
    "finn_commit": values["FINN_COMMIT"],
    "git_describe": values["FINN_SOURCE_DESCRIBE"],
    "source_dirty": values["FINN_SOURCE_DIRTY"] == "1",
    "resolved_dependencies": json.loads((records / "dependencies.json").read_text()),
}, indent=2, sort_keys=True))
PY
)

echo "$PROVENANCE"

if [ -n "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
    printf '%s\n' "$PROVENANCE" > "$OUTDIR/finn-image-provenance.json"
    # The digest on its own line, for shards that just need to pin the image.
    printf '%s\n' "$DIGEST" > "$OUTDIR/finn-image-digest.txt"
    echo "Wrote provenance to $OUTDIR/finn-image-provenance.json" >&2
fi
