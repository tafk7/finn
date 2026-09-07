#!/usr/bin/env bash
# Load a prebuilt FINN image from the CI shared-image directory.
#
#   ci/scripts/load-shared-image.sh <tag>
#
# CI transport, not a developer concern. It lived in the old public launcher,
# where every local quick test carried ~50 lines of NFS handling it would
# never execute.
#
# This is the OCI-archive path: where a registry is unavailable and NFS is the
# only transport, the image is saved to a shared directory and other agents load
# it rather than rebuilding. Prefer publishing by digest to a registry when one
# exists; see ci/scripts/build-images.sh. A tag alone is not an immutable image
# identity.
#
# Reads FINN_DOCKER_SHARED_IMAGE_DIR and FINN_DOCKER_PREBUILT, exactly as
# before. Exits 0 having loaded nothing when neither is set, so callers can
# invoke it unconditionally.

set -uo pipefail

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; NC=$'\033[0m'
gecho () { echo "${GREEN}$*${NC}"; }
recho () { echo "${RED}$*${NC}" >&2; }

FINN_DOCKER_TAG="${1:?usage: $0 <tag>}"
: "${FINN_DOCKER_PREBUILT:=0}"
: "${FINN_DOCKER_SHARED_IMAGE_DIR:=}"

# Echo the single image ID recorded by build-images.sh, or fail with a reason.
# Called BEFORE the load: a malformed sidecar is a publisher bug, and there is
# no point pulling several gigabytes off NFS to discover it.
read_expected_image_id () {
  local file="$1" recorded lines
  # `grep .` drops the trailing newline printf leaves behind. Any other line is
  # not a single recorded ID, whatever it looks like.
  recorded=$(grep . "$file")
  if [ -z "$recorded" ]; then
    recho "$file is empty; expected one image ID in sha256:<hex> form"
    return 1
  fi
  lines=$(printf '%s\n' "$recorded" | wc -l)
  if [ "$lines" -ne 1 ]; then
    recho "$file records $lines image IDs; expected exactly one"
    return 1
  fi
  if ! [[ "$recorded" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    recho "$file does not record a Docker image ID in sha256:<hex> form: $recorded"
    return 1
  fi
  printf '%s\n' "$recorded"
}

# fail fast on PREBUILT=1 with no usable image source: with no shared dir
# configured and no local image, docker run further down would fail with
# a generic "Unable to find image" much later in the pipeline.
if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ -z "$FINN_DOCKER_SHARED_IMAGE_DIR" ] \
   && ! docker image inspect "$FINN_DOCKER_TAG" > /dev/null 2>&1; then
  recho "FINN_DOCKER_PREBUILT=1 but FINN_DOCKER_SHARED_IMAGE_DIR is unset and tag $FINN_DOCKER_TAG is not loaded locally"
  recho "Set FINN_DOCKER_SHARED_IMAGE_DIR to a directory containing finn-docker-image.tar.gz, or unset FINN_DOCKER_PREBUILT to build locally."
  exit 1
fi

# If a shared-image dir is configured, load from there. In prebuilt mode
# the shared image is authoritative and any same-tag local image is ignored.
if [ -n "$FINN_DOCKER_SHARED_IMAGE_DIR" ] && \
   { [ "$FINN_DOCKER_PREBUILT" = "1" ] || ! docker image inspect "$FINN_DOCKER_TAG" > /dev/null 2>&1; }; then
  SHARED_DIR="$FINN_DOCKER_SHARED_IMAGE_DIR"
  SHARED_LOADED="0"
  SHARED_IMG="$SHARED_DIR/finn-docker-image.tar.gz"
  SHARED_TAG_FILE="$SHARED_DIR/finn-docker-tag.txt"
  SHARED_DIGEST_FILE="$SHARED_DIR/finn-image-digest.txt"
  EXPECTED_ID=""
  if [ -f "$SHARED_IMG" ] && [ -f "$SHARED_TAG_FILE" ]; then
    gecho "Loading Docker image from shared storage ($SHARED_DIR)..."
    SHARED_TAG=$(cat "$SHARED_TAG_FILE")
    if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ "$SHARED_TAG" != "$FINN_DOCKER_TAG" ]; then
      recho "Shared Docker tag $SHARED_TAG does not match requested tag $FINN_DOCKER_TAG"
      exit 1
    fi
    # The digest sidecar is what makes the tag more than a promise. It is
    # optional only for backwards compatibility: archives published before
    # build-images.sh recorded an image ID have the archive and the tag alone,
    # and Jenkins must keep loading those.
    if [ -f "$SHARED_DIGEST_FILE" ]; then
      EXPECTED_ID=$(read_expected_image_id "$SHARED_DIGEST_FILE") || exit 1
    else
      gecho "WARNING: no finn-image-digest.txt in $SHARED_DIR. This is a legacy, tag-only archive; its image identity cannot be verified"
    fi
    # local /tmp lock to serialise concurrent loads on the same host
    # $1 is intentionally expanded by the inner bash.
    # shellcheck disable=SC2016
    if flock /tmp/finn-docker-load.lock \
         bash -c 'set -o pipefail; gunzip -c "$1" | docker load' _ "$SHARED_IMG"; then
      SHARED_LOADED="1"
      if [ "$SHARED_TAG" != "$FINN_DOCKER_TAG" ]; then
        gecho "Tagging $SHARED_TAG as $FINN_DOCKER_TAG"
        docker tag "$SHARED_TAG" "$FINN_DOCKER_TAG"
      fi
    else
      gecho "WARNING: Failed to load Docker image from shared storage ($SHARED_DIR)"
    fi
    # Verify what the requested tag now references, not what the archive
    # claimed: the re-tag above is the last thing that moves it, and the
    # requested tag is what Compose will run.
    #
    # This proves the identity immediately after the load. It does NOT stop
    # another concurrent build from reassigning the same mutable tag before the
    # container starts. Closing that interval needs run-by-image-ID,
    # build-specific tags or a registry, and is deferred.
    if [ "$SHARED_LOADED" = "1" ] && [ -n "$EXPECTED_ID" ]; then
      if ! ACTUAL_ID=$(docker image inspect --format '{{.Id}}' "$FINN_DOCKER_TAG") \
         || [ -z "$ACTUAL_ID" ]; then
        recho "loaded $SHARED_IMG but could not read the image ID of $FINN_DOCKER_TAG"
        exit 1
      fi
      if [ "$ACTUAL_ID" != "$EXPECTED_ID" ]; then
        recho "image identity mismatch after loading $SHARED_IMG"
        recho "  $SHARED_DIGEST_FILE records $EXPECTED_ID"
        recho "  $FINN_DOCKER_TAG resolves to  $ACTUAL_ID"
        recho "Refusing to run tests against an image that is not the published one."
        exit 1
      fi
      gecho "Verified $FINN_DOCKER_TAG against recorded image ID $EXPECTED_ID"
    fi
  fi
  if [ "$SHARED_LOADED" != "1" ] && [ "$FINN_DOCKER_PREBUILT" != "1" ]; then
    gecho "WARNING: No usable shared Docker image found at FINN_DOCKER_SHARED_IMAGE_DIR=$SHARED_DIR. Falling back to local build"
  fi
  if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ "$SHARED_LOADED" != "1" ]; then
    recho "FINN_DOCKER_PREBUILT=1 but no usable shared Docker image at FINN_DOCKER_SHARED_IMAGE_DIR=$SHARED_DIR (expected finn-docker-image.tar.gz and finn-docker-tag.txt)"
    exit 1
  fi
fi


exit 0
