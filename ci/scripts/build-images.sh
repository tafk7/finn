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

# Image identity and mounted-source provenance are deliberately independent.
finn_set_provenance
FINN_COMMIT="$FINN_SOURCE_REVISION"

# FINN and its dependency metadata are installed in the application image.
FINN_DEPS_MODE=installed
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

# Resolve dependency refs to commits. deps.env may name a branch or tag, and a
# branch is not a provenance record -- it resolves differently tomorrow.
DEPS_JSON=$(
  # `set -a` matters: sourcing alone leaves the variables shell-local, so the
  # python below sees an empty environment and silently records no dependencies
  # at all -- a provenance file that looks complete and says nothing.
  set -a
  # shellcheck disable=SC1091
  . ./deps.env
  set +a
  # No "$@": the program below reads only the environment `set -a` exported.
  python3 <<'PY'
import os, subprocess, sys, json
out = {}
for key, value in sorted(os.environ.items()):
    if not key.endswith("_COMMIT"):
        continue
    entry = {"ref": value}
    if not all(c in "0123456789abcdef" for c in value.lower()) or len(value) < 7:
        # A branch or tag. Record that it is unresolved rather than pretending
        # a moving ref is provenance.
        entry["resolved"] = False
    else:
        entry["resolved"] = True
    out[key] = entry
json.dump(out, sys.stdout)
PY
)

PROVENANCE=$(python3 - <<PY
import json
print(json.dumps({
    "target": "$TARGET",
    "tag": "$TAG",
    "image_digest": "$DIGEST",
    "image_revision": "$FINN_IMAGE_REVISION",
    "finn_commit": "$FINN_COMMIT",
    "git_describe": "$FINN_SOURCE_DESCRIBE",
    "source_dirty": $([ "$FINN_SOURCE_DIRTY" = 1 ] && echo True || echo False),
    "finn_deps_mode": "$FINN_DEPS_MODE",
    "deps": json.loads('''$DEPS_JSON'''),
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
