# shellcheck shell=bash
# Shared shell helpers for FINN's launchers and CI scripts. SOURCE this file.
#
#     . "$(dirname "$0")/lib.sh"          # from docker/
#     . "$(dirname "$0")/../../docker/lib.sh"   # from ci/scripts/
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
#     docker/run-docker and docker-bake.hcl document that this is required
#     rather than tidy: without it bake auto-loads compose.yaml and dies on
#     ${FINN_XILINX_PATH:?} interpolation, on any machine with no Xilinx
#     configured.
#
# Conformance test 12 exists because the tag rule had two implementations. It
# checked two of seven sites. This file is the fix that test was pointing at.
#
# NOT A HOST-FACT RESOLVER. Nothing here reads the machine. Host facts stay in
# docker/config; the toolchain is applied by docker/finn-toolchain.sh. This
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

# A source-neutral revision for the Docker-built environment. FINN source is
# mounted at run time, so its commit must not change this value. The hash covers
# every declared image input plus build-argument overrides; the final image ID
# remains the immutable identity of one concrete build.
finn_image_revision () {
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

    local repo manifest pattern optional path found value name
    repo="${FINN_IMAGE_INPUT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    manifest="$repo/docker/image-inputs.txt"
    [ -f "$manifest" ] || { recho "image input manifest not found: $manifest"; return 2; }

    (
        cd "$repo" || exit 1
        printf 'finn-image-inputs-v1\n'
        while IFS= read -r pattern || [ -n "$pattern" ]; do
            case "$pattern" in ""|\#*) continue ;; esac
            optional=0
            case "$pattern" in \?*) optional=1; pattern=${pattern#?} ;; esac
            found=0
            while IFS= read -r path; do
                [ -f "$path" ] || continue
                found=1
                printf 'path=%s mode=%s sha256=' "$path" "$(stat -c '%a' "$path")"
                sha256sum "$path" | awk '{print $1}'
            done < <(compgen -G "$pattern" | LC_ALL=C sort || true)
            if [ "$found" = 0 ] && [ "$optional" = 0 ]; then
                recho "image input pattern matched no files: $pattern"
                exit 2
            fi
        done < "$manifest"

        # These are the Bake variables that can change image contents without
        # changing a file. Runtime selection is already represented in the tag
        # suffix, so it is intentionally not duplicated here.
        for name in UBUNTU_TAG QONNX_COMMIT FINN_EXP_COMMIT BREVITAS_COMMIT \
                    HLSLIB_COMMIT AVNET_BDF_COMMIT XIL_BDF_COMMIT \
                    RFSOC4x2_BDF_COMMIT KV260_BDF_COMMIT AUPZU3_BDF_COMMIT; do
            value="${!name-}"
            [ "$name" != UBUNTU_TAG ] || value="${value:-jammy-20230126}"
            printf 'arg=%s=%s\n' "$name" "$value"
        done
    ) | sha256sum | awk '{print "env-" substr($1, 1, 16)}'
}

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

# The bake target for a runtime set and the sbx flag.
#
#   finn_bake_target ""          -> finn
#   finn_bake_target "xrt"       -> finn-xrt
#   finn_bake_target "xrt,slash" -> finn-runtime        (parameterized)
#   finn_bake_target "xrt" sbx   -> finn-sbx-xrt
#
# The two supported runtime sets keep their descriptive fixed targets. Any other
# set uses a parameterized target; Bake computes its args, labels and tag.
finn_bake_target () {
    local runtimes variant="${2:-}"
    runtimes=$(finn_normalize_runtimes "${1:-}")
    case "$variant:$runtimes" in
        :)       printf '%s' finn ;;
        :xrt)    printf '%s' finn-xrt ;;
        sbx:)    printf '%s' finn-sbx ;;
        sbx:xrt) printf '%s' finn-sbx-xrt ;;
        sbx:*)   printf '%s' finn-sbx-runtime ;;
        *)       printf '%s' finn-runtime ;;
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

# Build a target and echo its tag. Fails loudly; callers need not check twice.
finn_bake_build () {
    local target="$1"; shift
    gecho "Building $target"
    # shellcheck disable=SC2086
    docker buildx bake -f docker-bake.hcl --load "$@" "$target" \
        || { recho "docker buildx bake $target failed"; return 1; }
}
