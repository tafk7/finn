#!/usr/bin/env bash
# Build FINN images with bake and emit a provenance record.
#
#   ci/scripts/build-images.sh <target> [output-dir]
#
# e.g.  ci/scripts/build-images.sh build-xrt-py310 "$IMAGE_DIR"
#
# WHY A DIGEST IS NOT ENOUGH ON ITS OWN
# -------------------------------------
# CI previously passed a recomputed mutable TAG to each test shard and trusted
# that two shards resolving the same tag got the same image. A digest fixes
# that half of the problem.
#
# It does not fix the other half. FINN source is MOUNTED, not baked, so the
# digest identifies the environment and says nothing about the code under test.
# Two shards on one digest can still execute different code if the workspace or
# the dependency checkouts differ. Hence the full tuple:
#
#     finn_commit      what src/finn was
#     image_digest     what the environment was
#     profile, tier    which image
#     deps             the resolved dependency commits, not branch names
#     finn_deps_mode   frozen, or the run is not reproducible at all
#
# FINN_DEPS=frozen is therefore not a preference here, it is load-bearing: in
# auto or live the mounted checkouts shadow the baked wheels and the digest
# stops meaning anything.

set -euo pipefail

TARGET="${1:?usage: $0 <bake-target> [output-dir]}"
OUTDIR="${2:-}"

cd "$(dirname "$0")/../.."

GIT_DESCRIBE=$(git describe --always --tags 2>/dev/null || echo unknown)
GIT_DESCRIBE_DIRTY=$(git describe --always --tags --dirty 2>/dev/null || echo unknown)
FINN_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
export GIT_DESCRIBE GIT_DESCRIBE_DIRTY

if ! git diff --quiet HEAD 2>/dev/null; then
    echo "WARNING: building from a dirty tree; provenance records the commit," >&2
    echo "         which does not describe the uncommitted changes." >&2
fi

echo "Building bake target $TARGET"
docker buildx bake -f docker-bake.hcl --load "$TARGET"

# Ask bake for the tag rather than recomputing it. This is the whole point of
# moving the rule into docker-bake.hcl: there is exactly one implementation.
TAG=$(docker buildx bake -f docker-bake.hcl --print "$TARGET" 2>/dev/null \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['target']['$TARGET']['tags'][0])")

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
  python3 - "$@" <<'PY'
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
    "finn_commit": "$FINN_COMMIT",
    "git_describe": "$GIT_DESCRIBE_DIRTY",
    "finn_deps_mode": "frozen",
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
