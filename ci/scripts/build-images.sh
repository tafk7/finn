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

# Record the image's installed package set, read from the built image itself
# rather than inferred from the caller's checkout.
PACKAGES=$(docker run --rm --network none --entrypoint uv "$TAG" pip freeze)
export TARGET TAG DIGEST FINN_COMMIT PACKAGES
PROVENANCE=$(python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
values = os.environ
print(json.dumps({
    "target": values["TARGET"],
    "tag": values["TAG"],
    "image_digest": values["DIGEST"],
    "image_revision": values["FINN_IMAGE_REVISION"],
    "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
    "finn_commit": values["FINN_COMMIT"],
    "git_describe": values["FINN_SOURCE_DESCRIBE"],
    "source_dirty": values["FINN_SOURCE_DIRTY"] == "1",
    "installed_packages": values["PACKAGES"].splitlines(),
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
