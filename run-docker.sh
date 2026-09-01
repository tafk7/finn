#!/bin/bash
# Copyright (c) 2020-2022, Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ============================================================================
# COMPATIBILITY WRAPPER. This script owns no facts.
#
# It has been FINN's entry point for close to a decade and the muscle memory is
# real, so every verb and environment variable still works. But each one is now
# a translation onto an artifact that owns the thing:
#
#   build matrix, args, tags   docker-bake.hcl
#   host discovery             docker/finn-env
#   docker runtime             compose.yaml
#   sbx runtime                docker/sbxenv/*.sbxenv.yaml + docker/finn-sbx
#   CI image transport         ci/scripts/
#
# If you are writing something new, target those directly:
#
#   docker compose run --rm dev quicktest.sh
#   docker compose --profile fpga run --rm build
#   docker buildx bake -f docker-bake.hcl finn-xrt
#   docker/finn-sbx build
#
# 407 lines on `dev`, ~350 now. The point is not the count -- it is that this
# script derives no host facts. The mount defects that started this work were
# all one failure, two code paths deriving the same fact and drifting, and a
# script that derives nothing cannot reproduce it. See docs/containerization.md.
# ============================================================================

set -uo pipefail

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; NC=$'\033[0m'
gecho () { echo "${GREEN}$1${NC}"; }
recho () { echo "${RED}$1${NC}" >&2; }
yecho () { echo "${YELLOW}$1${NC}" >&2; }

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH" || exit 1

# Which accelerator stacks the image should carry. Empty is the base image.
# See docker/runtimes/README.md. FINN_DOCKER_TARGET=build-xrt still works and
# is translated to FINN_RUNTIMES=xrt below.
: "${FINN_RUNTIMES:=}"
# Was the tier chosen by the caller, or is it about to be defaulted? The
# fallback below only applies to a DEFAULT, never to an explicit request.
FINN_DOCKER_TARGET_EXPLICIT="${FINN_DOCKER_TARGET+yes}"

# `build` as the nominal default, but see the `auto` block below: an
# unspecified tier is resolved against the host, so this only decides what an
# EXPLICIT request without a value would mean.
#
# NOT build-xrt, which was the historic default. That was Jenkins history: the
# widest-exposure tier -- XRT, the platform repository, the largest image -- for
# work that mostly needs Vivado and nothing more. RTL simulation needs xsim, not
# XRT. CI now names its tier explicitly through FINN_CI_IMAGE_TIER.
: "${FINN_DOCKER_TARGET:=build}"
: "${FINN_DEPS:=frozen}"
: "${FINN_DOCKER_PREBUILT:=0}"
: "${FINN_DOCKER_EXTRA:=}"
: "${FINN_DOCKER_BUILD_EXTRA:=}"
: "${FINN_DOCKER_RUN_AS_ROOT:=0}"
: "${FINN_SKIP_DEP_REPOS:=0}"
: "${FINN_SINGULARITY:=}"
: "${JUPYTER_PORT:=8888}"
: "${NETRON_PORT:=8081}"
: "${IMAGENET_VAL_PATH:=}"
: "${FINN_DOCKER_CACHE_DIR:=}"
: "${FINN_SSH_KEY_DIR:=}"
# FINN_HOST_BUILD_DIR is deliberately NOT defaulted here. finn-env resolves it
# and creates it; this script had a fifth, different default
# (/tmp/finn_dev_$USER) for the same question.

# Translate a legacy target name onto the current three axes.
#
# There used to be six images: {dev,build,build-xrt} x {generic,sbx}. There is
# now ONE image, plus a set of runtime targets in its tag and an sbx variant.
# `dev` and `build` name the same image and differ only in what the launcher
# mounts and allows, so both map to the same bake target; `build-xrt` was never
# a set of grants, it was XRT, so it maps to a runtime target.
#
#   dev, build      -> finn        tier dev / build
#   build-xrt       -> finn-xrt    tier build, FINN_RUNTIMES=xrt
#   sbx-*           -> finn-sbx[-xrt]
#
# Sets FINN_TIER, FINN_SBX_VARIANT and FINN_RUNTIMES.
finn_resolve_axes () {
    local target="$1"
    FINN_TIER="${target#sbx-}"
    case "$target" in sbx-*) FINN_SBX_VARIANT=1 ;; *) FINN_SBX_VARIANT=0 ;; esac
    case "$FINN_TIER" in
        build-xrt)
            FINN_TIER="build"
            case ",$FINN_RUNTIMES," in *,xrt,*) ;; *)
                FINN_RUNTIMES="${FINN_RUNTIMES:+$FINN_RUNTIMES,}xrt" ;;
            esac
            ;;
        dev|build) ;;
        *) recho "Unknown target '$target'; expected dev, build, build-xrt or sbx-*"; exit 2 ;;
    esac
    export FINN_RUNTIMES
}

finn_resolve_axes "$FINN_DOCKER_TARGET"

# An UNSPECIFIED tier becomes `auto`, and finn-env decides.
#
# This script used to carry the degrade itself: if the default tier is `build`
# and FINN_XILINX_PATH is unset, fall back to `dev`. That made a launcher the
# place where a host fact was interpreted, which is the shape of the four
# defects this redesign exists to prevent. `--tier auto` moves the same rule
# into the resolver, where every lane gets it.
#
# An EXPLICIT FINN_DOCKER_TARGET=build still hard-errors without a toolchain, as
# it should: that is a request, not a default, and silently narrowing it is how
# a CI shard passes without testing anything.
if [ -z "${FINN_DOCKER_TARGET_EXPLICIT:-}" ]; then
    FINN_TIER="auto"
fi

# Resolve `auto` to a concrete tier ONCE, and pass the concrete value to every
# later call. Not for speed -- so that finn-env's "no FINN_XILINX_PATH" warning
# is printed once per invocation rather than once per call.
if [ "$FINN_TIER" = "auto" ]; then
    # stderr is NOT suppressed: finn-env's "no FINN_XILINX_PATH, resolving to
    # dev" warning is the whole user-visible signal that a degrade happened.
    FINN_TIER=$(./docker/finn-env inspect --tier auto --format json \
                | python3 -c 'import json,sys;print(json.load(sys.stdin)["tier"])' 2>/dev/null) \
        || FINN_TIER=""
    if [ -z "$FINN_TIER" ]; then
        recho "finn-env could not resolve a tier"; exit 3
    fi
fi

# ----------------------------------------------------------------------------
# Tags come from bake. This script no longer has a tag rule of its own -- it had
# one, kept in sync with docker-bake.hcl by an assertion, and one authority is
# better than two that agree.
# ----------------------------------------------------------------------------
# No GIT_DESCRIBE_DIRTY any more. FINN's source is mounted, not baked, so an
# edited tree does not change the image; bake dropped the dirty variant with the
# tier split. See the note on the variable in docker-bake.hcl.
GIT_DESCRIBE="$(git describe --always --tags 2>/dev/null || echo local)"
export GIT_DESCRIBE

# -f docker-bake.hcl is REQUIRED, not tidiness.
#
# With no -f, bake auto-loads every definition it finds in the directory --
# including compose.yaml. compose's fpga services guard the toolchain mount with
# ${FINN_XILINX_PATH:?...}, so on a machine with no Xilinx configured bake fails
# to interpolate and refuses to run ANY target, including dev-py310 which has
# nothing to do with the toolchain. Bake reads the bake file; compose reads the
# compose file.

# The bake target for the resolved axes. `finn`, `finn-xrt`, `finn-sbx`, ... --
# the names in docker-bake.hcl. Only the combinations bake enumerates are
# reachable from here; build anything else with bake directly.
finn_bake_target () {
    local t="finn"
    [ "$FINN_SBX_VARIANT" = "1" ] && t="finn-sbx"
    for r in $(echo "$FINN_RUNTIMES" | tr ',' ' ' | tr ' ' '\n' | sort); do
        t="$t-$r"
    done
    echo "$t"
}

finn_bake_tag () {
    local target="$1"
    docker buildx bake -f docker-bake.hcl --print "$target" 2>/dev/null \
        | sed -n '/^{/,$p' \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['target']['$target']['tags'][0])" 2>/dev/null
}

if [ "${1:-}" = "print-tag" ]; then
    [ "$#" -le 2 ] || { recho "Usage: $0 print-tag [dev|build|build-xrt|sbx-*]"; exit 2; }
    [ -n "${2:-}" ] && finn_resolve_axes "$2"
    tag=$(finn_bake_tag "$(finn_bake_target)")
    [ -n "$tag" ] || { recho "no bake target for '$FINN_DOCKER_TARGET' with FINN_RUNTIMES='$FINN_RUNTIMES'"; exit 2; }
    echo "$tag"
    exit 0
fi

# ----------------------------------------------------------------------------
# Dependency sources. Fetched only when asked for -- the image already carries
# the pinned wheels, and this used to run a network round trip and mutate the
# checkout before every invocation.
# ----------------------------------------------------------------------------
if [ "$FINN_SKIP_DEP_REPOS" = "0" ] \
   && { [ "$FINN_DEPS" = "live" ] || [ "${FINN_FETCH_DEPS:-0}" = "1" ]; }; then
    gecho "Fetching dependency sources (FINN_DEPS=live)"
    if [ "$FINN_TIER" = "dev" ]; then ./fetch-repos.sh python || exit 1
    else ./fetch-repos.sh || exit 1; fi
fi

# ----------------------------------------------------------------------------
# Verbs that do not run a container
# ----------------------------------------------------------------------------

case "${1:-}" in
  build)
    [ -n "${2:-}" ] && finn_resolve_axes "$2"
    target=$(finn_bake_target)
    gecho "Building bake target $target"
    # shellcheck disable=SC2086
    docker buildx bake -f docker-bake.hcl --load $FINN_DOCKER_BUILD_EXTRA "$target" \
        || { recho "docker buildx bake $target failed"; exit 1; }
    gecho "Built $(finn_bake_tag "$target")"
    exit 0
    ;;
  sbx)
    tier="${2:-$FINN_TIER}"
    if [ -n "${2:-}" ]; then shift 2 2>/dev/null || shift $#; else shift; fi
    [ "${1:-}" = "--" ] && shift
    exec "$SCRIPTPATH/docker/finn-sbx" "$tier" "$@"
    ;;
esac

# FINN_SINGULARITY keeps working, and now means more than it did. It used to
# require a .sif you had built yourself, because nothing here could build one;
# docker/finn-apptainer can, so the variable is an override rather than the only
# way in.
if [ -n "$FINN_SINGULARITY" ]; then
    exec "$SCRIPTPATH/docker/finn-apptainer" -- "$@"
fi

# ----------------------------------------------------------------------------
# Everything else runs a command in a container, via compose.
# ----------------------------------------------------------------------------

# A SEPARATE env file from the human `.env`, passed with --env-file.
#
# The legacy contract mirrors the host workspace path; the compose default uses
# the fixed /workspace/finn. If this script wrote `.env`, a later bare
# `docker compose run dev` would silently inherit the legacy mirroring from
# whenever run-docker.sh last ran. Two contracts, one file, and the second
# caller gets whichever the first left behind.
RUN_ENV=".finn-run.env"
./docker/finn-env inspect --tier "$FINN_TIER" --format sh > "$RUN_ENV" \
    || { recho "finn-env inspect failed"; exit 1; }
# The legacy contract mirrors the host path for EVERY tier, including dev, so
# that `./run-docker.sh` behaves as it always has. New work should use
# `docker compose run --rm dev`, which uses the fixed /workspace/finn path.
{
    echo "FINN_WORKSPACE_SOURCE=$SCRIPTPATH"
    echo "FINN_WORKSPACE_TARGET=$SCRIPTPATH"
    echo "FINN_ROOT=$SCRIPTPATH"
    echo "FINN_DEPS=$FINN_DEPS"
    echo "JUPYTER_PORT=$JUPYTER_PORT"
    echo "NETRON_PORT=$NETRON_PORT"
    echo "GIT_DESCRIBE=$GIT_DESCRIBE"
} >> "$RUN_ENV"

SERVICE="$FINN_TIER"
COMPOSE_PROFILE=()
[ "$FINN_TIER" != "dev" ] && COMPOSE_PROFILE=(--profile fpga)

# Optional per-site mounts stay FLAGS rather than compose services, because
# compose cannot conditionally omit a bind mount and a `${VAR:-/dev/null}`
# placeholder is worse than an explicit flag.
RUN_OPTS=()
[ "$FINN_DOCKER_RUN_AS_ROOT" = "1" ] && RUN_OPTS+=(--user 0:0)
if [ -n "$FINN_SSH_KEY_DIR" ] && [ -d "$FINN_SSH_KEY_DIR" ]; then
    # Opt-in now. It used to be mounted unconditionally from a directory this
    # script created, which put a secret path into the default profile for a
    # deployment feature most runs never use.
    RUN_OPTS+=(-v "$FINN_SSH_KEY_DIR:$HOME/.ssh")
fi
if [ -n "$IMAGENET_VAL_PATH" ]; then
    RUN_OPTS+=(-v "$IMAGENET_VAL_PATH:$IMAGENET_VAL_PATH"
               -e "IMAGENET_VAL_PATH=$IMAGENET_VAL_PATH")
fi
if [ -n "$FINN_DOCKER_CACHE_DIR" ]; then
    mkdir -p "$FINN_DOCKER_CACHE_DIR/torch" "$FINN_DOCKER_CACHE_DIR/huggingface"
    RUN_OPTS+=(-v "$FINN_DOCKER_CACHE_DIR:/finn_cache"
               -e TORCH_HOME=/finn_cache/torch -e HF_HOME=/finn_cache/huggingface)
fi
if [ "${VERIFICATION_EN:-0}" = "1" ]; then
    for v in FINN_EXAMPLES_ROOT VERIFICATION_IO; do
        eval "val=\${$v:-}"
        [ -n "$val" ] || { recho "VERIFICATION_EN=1 but $v is unset"; exit 1; }
        [ -d "$val" ] || { recho "$v=$val is not a directory"; exit 1; }
        RUN_OPTS+=(-e "$v=$val" -v "$val:$val")
    done
    RUN_OPTS+=(-e VERIFICATION_EN=1)
fi
# A site escape hatch, kept because sites depend on it. `docker compose run`
# accepts -v/-e/-p/--user/-w, which covers what it is actually used for.
# Word-split deliberately: it is a flag string, not a path.
# shellcheck disable=SC2206
[ -n "$FINN_DOCKER_EXTRA" ] && RUN_OPTS+=($FINN_DOCKER_EXTRA)

case "${1:-}" in
  test)       gecho "Running test suite (all tests)"; CMD=(pytest) ;;
  quicktest)  gecho "Running test suite (non-Vivado, non-slow tests)"; CMD=(quicktest.sh) ;;
  notebook)
    gecho "Running Jupyter notebook server on $JUPYTER_PORT"
    exec docker compose --env-file "$RUN_ENV" --profile notebook up
    ;;
  build_dataflow)
    dir=$(readlink -f "${2:?build_dataflow needs a directory}")
    RUN_OPTS+=(-v "$dir:$dir")
    gecho "Running build_dataflow for folder $dir"
    CMD=(build_dataflow "$dir")
    ;;
  build_custom)
    dir=$(readlink -f "${2:?build_custom needs a directory}")
    flow="${3:-build}"
    RUN_OPTS+=(-v "$dir:$dir" -w "$dir")
    gecho "Running build_custom: $dir/$flow.py"
    CMD=(python -mpdb -cc -cq "$flow.py" "${@:4}")
    ;;
  "")         gecho "Running container only"; CMD=(bash); RUN_OPTS+=(-it) ;;
  *)          gecho "Running container with passed arguments"; CMD=("$@") ;;
esac

# Prebuilt images come from the CI shared directory; developers build locally.
if [ "$FINN_DOCKER_PREBUILT" = "1" ] || [ -n "${FINN_DOCKER_SHARED_IMAGE_DIR:-}" ]; then
    ./ci/scripts/load-shared-image.sh "$(finn_bake_tag "$(finn_bake_target)")" || exit 1
else
    # shellcheck disable=SC2086
    docker buildx bake -f docker-bake.hcl --load $FINN_DOCKER_BUILD_EXTRA \
        "$(finn_bake_target)" \
        || { recho "image build failed"; exit 1; }
fi

exec docker compose --env-file "$RUN_ENV" "${COMPOSE_PROFILE[@]}" \
     run --rm "${RUN_OPTS[@]}" "$SERVICE" "${CMD[@]}"
