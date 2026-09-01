# Shared shell helpers for FINN's launchers and CI scripts. SOURCE this file.
#
#     . "$(dirname "$0")/lib.sh"          # from docker/
#     . "$(dirname "$0")/../../docker/lib.sh"   # from ci/scripts/
#
# WHY THIS EXISTS
# ---------------
# docs/containerization.md's central claim is that every fact is derived in
# exactly one place, and four defects came from two code paths deriving the same
# fact and drifting. That discipline was applied rigorously to HOST facts -- and
# not at all to BUILD-MATRIX facts, which were duplicated across four launchers
# and two CI scripts. They had already drifted, in three ways, all silent:
#
#   - the `git describe` fallback was `local` in five places and `unknown` in
#     ci/scripts/build-images.sh -- a tag bake will never emit, so a provenance
#     record could name an image that does not exist;
#   - the `sed -n '/^{/,$p'` that survives bake's pre-JSON progress output was
#     present in the four launchers and MISSING from build-images.sh and
#     ci/Jenkinsfile, so the two CI paths carried the fragile variant;
#   - `-f docker-bake.hcl` was missing from ci/Jenkinsfile:421. Both
#     run-docker.sh and docker-bake.hcl document at length that this is required
#     rather than tidy: without it bake auto-loads compose.yaml and dies on
#     ${FINN_XILINX_PATH:?} interpolation, on any machine with no Xilinx
#     configured.
#
# Conformance test 12 exists because the tag rule had two implementations. It
# checked two of seven sites. This file is the fix that test was pointing at.
#
# NOT A HOST-FACT RESOLVER. Nothing here reads the machine. Host facts stay in
# docker/finn-env; the toolchain is applied by docker/finn-toolchain.sh. This
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

# The provenance string that goes in every tag.
#
# `local`, never `unknown`. docker-bake.hcl's GIT_DESCRIBE default is `local`,
# so a different fallback here produces a tag bake cannot reproduce.
finn_git_describe () {
    git describe --always --tags 2>/dev/null || echo local
}

# The bake target for a runtime set and the sbx flag.
#
#   finn_bake_target ""          -> finn
#   finn_bake_target "xrt"       -> finn-xrt
#   finn_bake_target "xrt,slash" -> finn-slash-xrt      (sorted)
#   finn_bake_target "xrt" sbx   -> finn-sbx-xrt
#
# Sorted, because the target -- like the tag suffix it produces -- is a function
# of the SET. Must agree with tag() in docker-bake.hcl and runtime_tag() in
# docker/finn-env; conformance test 12 asserts that it does.
#
# Only the combinations enumerated in docker-bake.hcl are reachable. Build any
# other with bake directly and --set.
finn_bake_target () {
    local runtimes="${1:-}" variant="${2:-}" t="finn" r
    [ "$variant" = "sbx" ] && t="finn-sbx"
    for r in $(printf '%s' "$runtimes" | tr ',' '\n' | sort); do
        [ -n "$r" ] && t="$t-$r"
    done
    printf '%s' "$t"
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
