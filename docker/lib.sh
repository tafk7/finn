# shellcheck shell=bash
# Shared shell helpers for FINN's launchers. SOURCE this file.
#
#     . "$(dirname "$0")/lib.sh"          # from docker/
#
# WHY THIS EXISTS
# ---------------
# Every fact must be derived in exactly one place. Four defects came from two
# code paths deriving the same host fact and drifting. That discipline was not
# originally applied to build-matrix facts, which were duplicated across four
# launchers and two CI scripts. They had already drifted in three silent ways:
#
#   - the old source-derived tag fallback was `local` in five places and
#     `unknown` in ci/scripts/build-images.sh, so provenance could name an image
#     that did not exist;
#   - the `sed -n '/^{/,$p'` that survives bake's pre-JSON progress output was
#     present in the four launchers and MISSING from build-images.sh and
#     ci/Jenkinsfile, so the two CI paths carried the fragile variant;
#   - `-f docker-bake.hcl` was missing from ci/Jenkinsfile:421. Both
#     docker/run and docker-bake.hcl document that this is required
#     rather than tidy: without it bake auto-loads compose.yaml and dies on
#     ${FINN_XILINX_PATH:?} interpolation, on any machine with no Xilinx
#     configured.
#
# Conformance test 12 exists because the tag rule had two implementations. It
# checked two of seven sites. This file is the fix that test was pointing at.
#
# NOT A HOST-FACT RESOLVER. Nothing here reads the machine. Host facts stay in
# docker/config.py; the toolchain is applied by docker/finn-toolchain.sh. This
# file knows only about the build matrix, which is docker-bake.hcl's subject.

# --------------------------------------------------------------------------
# Diagnostics
#
# Defined in eight places before this file, in two dialects -- some `echo -e`
# with '...' quoting, some `echo` with $'...', some routing to stderr and some
# not. docker/finn_entrypoint.sh deliberately keeps its own copy: it runs INSIDE
# the image, where the host repo is not guaranteed to be mounted.
# --------------------------------------------------------------------------

FINN_RED=$'\033[0;31m'; FINN_GREEN=$'\033[0;32m'
FINN_YELLOW=$'\033[0;33m'; FINN_NC=$'\033[0m'

gecho () { echo "${FINN_GREEN}$*${FINN_NC}"; }
recho () { echo "${FINN_RED}$*${FINN_NC}" >&2; }
yecho () { echo "${FINN_YELLOW}$*${FINN_NC}" >&2; }

# --------------------------------------------------------------------------
# The build matrix
# --------------------------------------------------------------------------

finn_normalize_runtimes () {
    printf '%s' "${1:-}" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -
}

# The image identity: a hash of the files listed in docker/image-inputs.txt plus
# the build arguments that change image content. FINN's sources are not inputs;
# containers install the mounted checkout. Image IDs still identify the exact
# build.
finn_image_revision () (
    set -o pipefail
    if [ -n "${FINN_IMAGE_REVISION:-}" ]; then
        case "$FINN_IMAGE_REVISION" in
            *[!A-Za-z0-9_.-]*|"")
                recho "FINN_IMAGE_REVISION contains characters invalid in a Docker tag"
                return 2
                ;;
        esac
        printf '%s' "$FINN_IMAGE_REVISION"
        return 0
    fi

    local repo manifest pattern optional path found
    repo="${FINN_IMAGE_INPUT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    manifest="$repo/docker/image-inputs.txt"
    [ -f "$manifest" ] || { recho "image input manifest not found: $manifest"; return 2; }

    (
        cd "$repo" || exit 1
        shopt -s globstar
        printf 'finn-image-inputs-v3\n'
        while IFS= read -r pattern || [ -n "$pattern" ]; do
            case "$pattern" in ""|\#*) continue ;; esac
            optional=0
            case "$pattern" in \?*) optional=1; pattern=${pattern#?} ;; esac
            found=0
            while IFS= read -r path; do
                [ -f "$path" ] || continue
                case "$path" in */__pycache__/*|*.pyc|*.so|*.egg-info/*) continue ;; esac
                found=1
                printf 'path=%s mode=%s sha256=' "$path" "$(stat -c '%a' "$path")"
                sha256sum "$path" | awk '{print $1}'
            done < <(compgen -G "$pattern" | LC_ALL=C sort || true)
            if [ "$found" = 0 ] && [ "$optional" = 0 ]; then
                recho "image input pattern matched no files: $pattern"
                exit 2
            fi
        done < "$manifest"
        # Resource pins, only of the resources the image bakes in: the
        # redistributable ones (python stage) and the board files (dev stage) of
        # docker/Dockerfile.finn. Moving another pin, such as FinnLib's, which is
        # never baked, leaves the image as it is.
        [ ! -f src/finn/resources.toml ] \
            || python3 -B - src/finn/resources.toml <<'PY' || exit 2
import sys, tomllib
with open(sys.argv[1], "rb") as file:
    declared = tomllib.load(file)["resources"]
for name, fields in declared.items():
    if "package" in fields or "path" in fields:
        continue
    if fields.get("redistributable") or "vivado-boards" in fields.get("kind", []):
        pin = [fields.get(key, "") for key in ("git", "commit", "url", "sha256", "subdir", "into")]
        print("resource=" + name, *pin, fields["digest"])
PY
        # Build arguments that change image contents without changing a file.
        # The runtime set is part of the tag suffix instead.
        printf 'arg=UBUNTU_TAG=%s\n' "${UBUNTU_TAG:-noble-20240605}"
    ) | sha256sum | awk '{print "img-" substr($1, 1, 16)}'
)

finn_source_revision () {
    local repo="${FINN_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    git -C "$repo" rev-parse HEAD 2>/dev/null || printf '%s' unknown
}

finn_source_describe () {
    local repo="${FINN_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    git -C "$repo" describe --always --tags --abbrev=12 --dirty 2>/dev/null || printf '%s' unknown
}

finn_source_dirty () {
    local repo="${FINN_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    [ -z "$(git -C "$repo" status --porcelain --untracked-files=normal 2>/dev/null)" ] \
        && printf '0' || printf '1'
}

finn_set_provenance () {
    FINN_IMAGE_REVISION=$(finn_image_revision) || return
    FINN_SOURCE_REVISION=$(finn_source_revision)
    FINN_SOURCE_DESCRIBE=$(finn_source_describe)
    FINN_SOURCE_DIRTY=$(finn_source_dirty)
    export FINN_IMAGE_REVISION FINN_SOURCE_REVISION FINN_SOURCE_DESCRIBE FINN_SOURCE_DIRTY
}

# The bake target for a runtime set and a variant ("", sbx or release).
#
#   finn_bake_target ""              -> finn
#   finn_bake_target "xrt"           -> finn-xrt
#   finn_bake_target "xrt,slash"     -> finn-runtime        (parameterized)
#   finn_bake_target "xrt" sbx       -> finn-sbx-xrt
#   finn_bake_target "xrt" release   -> finn-release        (parameterized)
finn_bake_target () {
    local runtimes variant="${2:-}"
    runtimes=$(finn_normalize_runtimes "${1:-}")
    case "$variant:$runtimes" in
        :)          printf '%s' finn ;;
        :xrt)       printf '%s' finn-xrt ;;
        sbx:)       printf '%s' finn-sbx ;;
        sbx:xrt)    printf '%s' finn-sbx-xrt ;;
        sbx:*)      printf '%s' finn-sbx-runtime ;;
        release:*)  printf '%s' finn-release ;;
        *)          printf '%s' finn-runtime ;;
    esac
}

# Ask bake for a target's tag. The ONE place that parses bake's output.
#
# Two things here are load-bearing and were each missing somewhere:
#
#   -f docker-bake.hcl   without it bake also auto-loads compose.yaml and fails
#                        to interpolate ${FINN_XILINX_PATH:?} on a machine with
#                        no Xilinx, refusing every target including ones that
#                        have nothing to do with the toolchain.
#   sed -n '/^{/,$p'     bake writes progress lines to stdout before the JSON.
#                        Without this, json.load sees "#1 reading ..." and dies.
finn_bake_tag () {
    local target="$1"
    docker buildx bake -f docker-bake.hcl --print "$target" 2>/dev/null \
        | sed -n '/^{/,$p' \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['target']['$target']['tags'][0])" 2>/dev/null
}

# Build one Bake target. Image references are resolved separately.
finn_bake_build () {
    local target="$1"; shift
    gecho "Building $target"
    # shellcheck disable=SC2086
    docker buildx bake -f docker-bake.hcl --load "$@" "$target" \
        || { recho "docker buildx bake $target failed"; return 1; }
}

# Prepare a Docker-daemon image for any consumer. Sets FINN_IMAGE. Explicit
# builds refresh cached layers; runs reuse the selected environment-input tag.
finn_prepare_image () {
    local target="$1" mode="${2:-ensure}"
    local build_args=()
    FINN_IMAGE=$(finn_bake_tag "$target") || return 1
    [ -n "$FINN_IMAGE" ] || { recho "No image tag for $target"; return 1; }
    export FINN_IMAGE
    if [ "$mode" != build ] && { [ "${FINN_CONTAINER_NO_BUILD:-0}" = 1 ] || [ "${FINN_DOCKER_PREBUILT:-0}" = 1 ]; }; then
        docker image inspect "$FINN_IMAGE" >/dev/null 2>&1 || {
            recho "No prepared Docker image for $FINN_IMAGE; run ./docker/build first."
            return 1
        }
        return 0
    fi
    if [ "$mode" = ensure ] && [ "${FINN_CONTAINER_REBUILD:-0}" != 1 ] \
       && [ -z "${FINN_DOCKER_BUILD_EXTRA:-}" ] \
       && docker image inspect "$FINN_IMAGE" >/dev/null 2>&1; then
        return 0
    fi
    [ "${FINN_CONTAINER_REBUILD:-0}" != 1 ] || build_args+=(--no-cache)
    # Legacy build flags are a shell word list, never evaluated as shell code.
    # shellcheck disable=SC2206
    build_args+=(${FINN_DOCKER_BUILD_EXTRA:-})
    finn_bake_build "$target" "${build_args[@]}"
}
