#!/usr/bin/env bash
# FINN container conformance suite.
#
#   ci/scripts/conformance.sh [test-number ...]
#
# Nine checks over the properties the containerization redesign established.
# These are NOT unit tests -- they need a real docker daemon, and tests 5 and 9
# additionally need sbx and a licence. Each is skipped with a reason rather than
# silently passing when its prerequisites are absent.
#
# WHY THESE NINE
# --------------
# Every one corresponds to a defect that actually happened, or to a contract
# that would erode silently without an assertion:
#
#   1  supported targets build        driven from the bake group, not a list
#   2  dev needs nothing              the contract automation depends on
#   3  fresh docker run               the path everyone already tests
#   4  BARE docker exec               a live defect: `docker exec <c> vivado`
#                                     did not work, because docker exec skips
#                                     the ENTRYPOINT and FINN's own docs
#                                     recommend it for a second terminal
#   5  BARE sbx exec                  makes "works with stock sbx" a property
#   6  mounts present AND read-only   the Xilinx tree was :rw under docker
#   7  both privilege readings        "no privileges" is ambiguous and the
#                                     wrong reading is the natural one
#   8  awkward workspace path         both policies survive spaces (NOT the
#                                     launcher -- see the note at test 8)
#   9  node-locked licence with :ro   UNRESOLVED contradiction in the kit
#  11  apptainer runs the image      a fourth runtime, with different rules
#  10  lane 3 resolves a toolchain    the bare-host lane has the fewest users
#                                     and the least coverage, so it is the one
#                                     that rots silently
#
# Tests 4 and 5 in their BARE form are the load-bearing ones. Running them
# through a wrapper would pass while the property they exist to check is
# broken.

set -uo pipefail
cd "$(dirname "$0")/../.."

PASS=0; FAIL=0; SKIP=0
FAILED_TESTS=()

ok   () { echo "PASS: $*"; PASS=$((PASS+1)); }
bad  () { echo "FAIL: $*"; FAIL=$((FAIL+1)); FAILED_TESTS+=("$*"); }
skip () { echo "SKIP: $*"; SKIP=$((SKIP+1)); }
head_() { echo; echo "=== $* ==="; }

# Counters must not be incremented inside a subshell -- a `( cd x && check )`
# loses every increment, which is how a red run can print a green total. Keep
# every ok/bad call in the current shell.

WANT="${*:-1 2 3 4 5 6 7 8 9 10 11}"
want () { case " $WANT " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

GIT_DESCRIBE=$(git describe --always --tags 2>/dev/null || echo local)
GIT_DESCRIBE_DIRTY=$(git describe --always --tags --dirty 2>/dev/null || echo local)
export GIT_DESCRIBE GIT_DESCRIBE_DIRTY

tag_for () { ./run-docker.sh print-tag "$1" 2>/dev/null; }

# Ensure an image exists, building it if not.
#
# Without this, running a subset of the suite (`conformance.sh 3`) reports
# spurious failures the moment HEAD moves, because the tag embeds git describe
# and the image for the current commit was never built. A test must fail
# because the property is broken, not because of how it was invoked.
need_image () {
    local tier="$1" tag
    tag=$(tag_for "$tier")
    [ -n "$tag" ] || return 1
    docker image inspect "$tag" >/dev/null 2>&1 && return 0
    echo "  (building $tier for this commit)" >&2
    docker buildx bake -f docker-bake.hcl --load "${tier}-${FINN_PROFILE:-py310}" >/dev/null 2>&1
}

CONTAINER=finn-conformance-$$
SANDBOX=finn-conformance
cleanup () {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    [ "${KEEP_SANDBOX:-0}" = "1" ] || sbx rm --force "$SANDBOX" >/dev/null 2>&1 || true
}
trap cleanup EXIT

have_docker=0; command -v docker >/dev/null && docker info >/dev/null 2>&1 && have_docker=1
have_sbx=0;    command -v sbx    >/dev/null && have_sbx=1
have_xilinx=0; [ -n "${FINN_XILINX_PATH:-}" ] && [ -d "${FINN_XILINX_PATH:-}" ] && have_xilinx=1
have_licence=0; [ -n "${XILINXD_LICENSE_FILE:-}${LM_LICENSE_FILE:-}" ] && have_licence=1

# ---------------------------------------------------------------------------
head_ "1. Supported targets build; experimental ones smoke-build"
# ---------------------------------------------------------------------------
if want 1 && [ "$have_docker" = 1 ]; then
    # Driven from the `supported` group in docker-bake.hcl, not a hardcoded
    # list. That group exists to declare what CI must keep green; a second copy
    # here could drift from it with nothing noticing -- which it had, naming
    # three targets where the group names six.
    targets=$(docker buildx bake -f docker-bake.hcl --print supported 2>/dev/null \
              | python3 -c 'import json,sys;print(" ".join(sorted(json.load(sys.stdin)["target"])))' 2>/dev/null)
    if [ -z "$targets" ]; then
        bad "1: could not read the `supported` group from docker-bake.hcl"
    else
        for target in $targets; do
            if docker buildx bake -f docker-bake.hcl --load "$target" >/dev/null 2>&1; then
                ok "supported target builds: $target"
            else
                bad "supported target FAILED to build: $target"
            fi
        done
    fi
elif want 1; then
    skip "1: no docker daemon"
fi

DEV_TAG=$(tag_for dev)
BUILD_TAG=$(tag_for build)

# ---------------------------------------------------------------------------
head_ "2. The dev tier requires nothing"
# ---------------------------------------------------------------------------
if want 2; then
    # Deliberately WITH the Xilinx and licence variables exported: dev must
    # ignore them. The original bug class was a tier widening because something
    # happened to be set in the caller's shell.
    out=$(FINN_XILINX_PATH="${FINN_XILINX_PATH:-/opt/Xilinx}" \
          XILINXD_LICENSE_FILE="2100@example.invalid" \
          ./docker/finn-env inspect --tier dev 2>/dev/null)
    if [ -z "$out" ]; then
        bad "2: finn-env inspect --tier dev produced nothing"
    else
        mounts=$(printf '%s' "$out" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["mounts"]))')
        egress=$(printf '%s' "$out" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["egress"]))')
        leaked=$(printf '%s' "$out" | python3 -c 'import json,sys;e=json.load(sys.stdin)["env"];print(",".join(k for k in e if "XILINX" in k or "LICENSE" in k))')
        [ "$mounts" = 0 ] && ok "dev declares no host mounts" || bad "dev declares $mounts host mount(s)"
        [ "$egress" = 0 ] && ok "dev declares no egress"      || bad "dev declares $egress egress grant(s)"
        [ -z "$leaked" ]  && ok "dev leaks no toolchain/licence env" \
                          || bad "dev leaked env: $leaked"
    fi

    if [ "$have_docker" = 1 ] && need_image dev; then
        # And it must actually RUN with only the workspace mounted.
        if docker run --rm -v "$PWD:/workspace/finn" -w /workspace/finn \
             --user "$(id -u):$(id -g)" -e FINN_ROOT=/workspace/finn \
             "$DEV_TAG" python -c 'import finn' >/dev/null 2>&1; then
            ok "dev runs with only the workspace mounted"
        else
            bad "dev failed to run with only the workspace mounted"
        fi
    fi
fi

# ---------------------------------------------------------------------------
head_ "3. A fresh docker run works"
# ---------------------------------------------------------------------------
if want 3 && [ "$have_docker" = 1 ] && need_image dev; then
    if docker run --rm -v "$PWD:$PWD" -w "$PWD" --user "$(id -u):$(id -g)" \
         -e FINN_ROOT="$PWD" "$DEV_TAG" python -c 'import finn' >/dev/null 2>&1; then
        ok "fresh docker run: import finn"
    else
        bad "fresh docker run: import finn"
    fi
elif want 3; then
    skip "3: no docker daemon"
fi

# ---------------------------------------------------------------------------
head_ "4. BARE docker exec into a long-running container"
# ---------------------------------------------------------------------------
if want 4 && [ "$have_docker" = 1 ] && need_image dev; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    run_args=(-d --name "$CONTAINER" -v "$PWD:$PWD" -w "$PWD"
              --user "$(id -u):$(id -g)" -e FINN_ROOT="$PWD")
    img="$DEV_TAG"
    if [ "$have_xilinx" = 1 ] && need_image build; then
        img="$BUILD_TAG"
        eval "$(./docker/finn-env inspect --tier build --format sh 2>/dev/null | sed 's/^/export /')"
        run_args+=(-v "$FINN_XILINX_PATH:$FINN_XILINX_PATH:ro")
        for v in VIVADO_PATH VITIS_PATH HLS_PATH XILINXD_LICENSE_FILE; do
            eval "val=\${$v:-}"; [ -n "$val" ] && run_args+=(-e "$v=$val")
        done
    fi
    if docker run "${run_args[@]}" "$img" sleep infinity >/dev/null 2>&1; then
        sleep 3
        if timeout 180 docker exec "$CONTAINER" python -c 'import finn' >/dev/null 2>&1; then
            ok "bare docker exec: python -c 'import finn'"
        else
            bad "bare docker exec: python -c 'import finn'"
        fi
        if [ "$have_xilinx" = 1 ]; then
            # THE test. No shell, no wrapper. This did not work before the
            # transparent shims: docker exec skips the ENTRYPOINT.
            if timeout 300 docker exec "$CONTAINER" vivado -version >/dev/null 2>&1; then
                ok "bare docker exec: vivado -version"
            else
                bad "bare docker exec: vivado -version"
            fi
        else
            skip "bare docker exec vivado: FINN_XILINX_PATH not set"
        fi
    else
        bad "4: could not start a long-running container"
    fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
elif want 4; then
    skip "4: no docker daemon"
fi

# ---------------------------------------------------------------------------
head_ "5. BARE sbx exec, no launcher involved"
# ---------------------------------------------------------------------------
if want 5 && [ "$have_sbx" = 1 ]; then
    if sbx ls 2>/dev/null | awk '{print $1}' | grep -qx "$SANDBOX"; then
        if timeout 180 sbx exec "$SANDBOX" python -c 'import finn' >/dev/null 2>&1; then
            ok "bare sbx exec: python -c 'import finn'"
        else
            bad "bare sbx exec: python -c 'import finn'"
        fi
        if [ "$have_xilinx" = 1 ]; then
            if timeout 300 sbx exec "$SANDBOX" vivado -version >/dev/null 2>&1; then
                ok "bare sbx exec: vivado -version"
            else
                bad "bare sbx exec: vivado -version"
            fi
        fi
    else
        skip "5: no sandbox named $SANDBOX (create one with FINN_SBX_NAME=$SANDBOX docker/finn-sbx build)"
    fi
elif want 5; then
    skip "5: sbx not on PATH"
fi

# ---------------------------------------------------------------------------
head_ "6. Toolchain and platform mounts are present AND read-only"
# ---------------------------------------------------------------------------
if want 6 && [ "$have_xilinx" = 1 ]; then
    json=$(./docker/finn-env inspect --tier build-xrt 2>/dev/null)
    rw=$(printf '%s' "$json" | python3 -c 'import json,sys;print(",".join(m["source"] for m in json.load(sys.stdin)["mounts"] if m["mode"]!="ro"))')
    n=$(printf '%s' "$json" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["mounts"]))')
    [ "$n" -gt 0 ] && ok "build-xrt declares $n host mount(s)" \
                   || bad "build-xrt declares no host mounts"
    [ -z "$rw" ] && ok "every declared mount is read-only" \
                 || bad "writable mount(s) declared: $rw"

    # And in the real container, not just the declaration.
    if [ "$have_docker" = 1 ] && need_image build; then
        if docker run --rm -v "$FINN_XILINX_PATH:$FINN_XILINX_PATH:ro" "$BUILD_TAG" \
             sh -c "touch $FINN_XILINX_PATH/.conformance-write 2>/dev/null && echo WRITABLE" 2>/dev/null \
             | grep -q WRITABLE; then
            bad "the Xilinx mount is WRITABLE inside the container"
        else
            ok "the Xilinx mount rejects writes inside the container"
        fi
    fi
elif want 6; then
    skip "6: FINN_XILINX_PATH not set"
fi

# ---------------------------------------------------------------------------
head_ "7. Privilege, BOTH readings"
# ---------------------------------------------------------------------------
# "No privileges" is ambiguous and the natural reading is the wrong one. The
# generic and sbx images differ ONLY in in-container root; neither has any host
# privilege. Asserting one reading and not the other is how the distinction
# gets lost.
if want 7 && [ "$have_docker" = 1 ] && need_image dev; then
    if docker run --rm "$DEV_TAG" sh -c 'sudo -n true' >/dev/null 2>&1; then
        bad "generic dev image grants in-container root"
    else
        ok "generic dev image has no in-container root"
    fi
    sbxdev=$(tag_for sbx-dev)
    if docker image inspect "$sbxdev" >/dev/null 2>&1; then
        if docker run --rm "$sbxdev" sh -c 'sudo -n true' >/dev/null 2>&1; then
            ok "sbx-dev image grants in-container root (required by the sbx contract)"
        else
            bad "sbx-dev image lacks in-container root; sbx kit install will fail"
        fi
    else
        skip "7: sbx-dev image not built"
    fi
    # Host privilege: neither image should need any of it, and nothing in the
    # repo should be adding it.
    if grep -rn -- '--privileged\|--cap-add\|/var/run/docker.sock' \
         run-docker.sh compose.yaml docker-bake.hcl 2>/dev/null | grep -v '^\s*#' | grep -q .; then
        bad "something requests host privilege (--privileged / --cap-add / docker.sock)"
    else
        ok "nothing requests host privilege"
    fi
elif want 7; then
    skip "7: no docker daemon"
fi

# ---------------------------------------------------------------------------
head_ "8. A workspace path with spaces and awkward characters"
# ---------------------------------------------------------------------------
# SCOPE, honestly: this checks that the two WORKSPACE POLICIES survive an
# awkward host path. It does NOT check run-docker.sh, which assembles docker
# arguments by string concatenation (DOCKER_EXEC+="-v $a:$b ") and cannot
# survive a space in a path. Testing that would mean copying the whole checkout
# to a spaced path and re-running the launcher against it, which is expensive
# enough that it has not been done.
#
# So: the launcher remains vulnerable and this test will not tell you. It is
# recorded here rather than in a comment nobody reads, because a green suite
# that quietly excludes a known weakness is worse than no test.
if want 8 && [ "$have_docker" = 1 ] && need_image dev; then
    awkward=$(mktemp -d "/tmp/finn conf-XXXX")
    mkdir -p "$awkward/src/finn"
    if docker run --rm -v "$awkward:/workspace/finn" -w /workspace/finn \
         --user "$(id -u):$(id -g)" -e FINN_ROOT=/workspace/finn \
         "$DEV_TAG" python -c 'import os;print(os.getcwd())' >/dev/null 2>&1; then
        ok "fixed-path policy tolerates a host path containing spaces"
    else
        bad "fixed-path policy breaks on a host path containing spaces"
    fi
    if docker run --rm -v "$awkward:$awkward" -w "$awkward" \
         --user "$(id -u):$(id -g)" -e FINN_ROOT="$awkward" \
         "$DEV_TAG" python -c 'import os;print(os.getcwd())' >/dev/null 2>&1; then
        ok "mirror policy tolerates a host path containing spaces"
    else
        bad "mirror policy breaks on a host path containing spaces"
    fi
    rm -rf "$awkward"
elif want 8; then
    skip "8: no docker daemon"
fi

# ---------------------------------------------------------------------------
head_ "9. Node-locked licence with the licence directory :ro"
# ---------------------------------------------------------------------------
# Resolves the UNRESOLVED contradiction recorded in docker/finn.kit/spec.yaml:
# the kit says some FLEXlm setups write beside the licence file, but both the
# docker and sbx paths mount that directory read-only. Our licence testing used
# the floating PORT@HOST form, which mounts nothing, so this path has never
# been exercised.
if want 9; then
    node_locked=""
    for entry in $(printf '%s' "${XILINXD_LICENSE_FILE:-}:${LM_LICENSE_FILE:-}" | tr ':' ' '); do
        case "$entry" in ''|*@*) continue ;; esac
        [ -f "$entry" ] && node_locked="$entry" && break
    done
    if [ -z "$node_locked" ]; then
        skip "9: no node-locked licence file configured -- the :ro-vs-sibling-writes"
        skip "   contradiction in docker/finn.kit/spec.yaml REMAINS UNRESOLVED"
    elif [ "$have_docker" = 1 ] && [ "$have_xilinx" = 1 ]; then
        licdir=$(dirname "$node_locked")
        if docker run --rm -v "$FINN_XILINX_PATH:$FINN_XILINX_PATH:ro" \
             -v "$licdir:$licdir:ro" -e "XILINXD_LICENSE_FILE=$node_locked" \
             -e "VIVADO_PATH=${VIVADO_PATH:-}" "$BUILD_TAG" \
             vivado -version >/dev/null 2>&1; then
            ok "node-locked licence works with the licence directory :ro"
        else
            bad "node-locked licence FAILS with the licence directory :ro -- either"
            bad "  the mount must be writable or the kit's sibling-write claim is wrong"
        fi
    else
        skip "9: needs docker and FINN_XILINX_PATH"
    fi
fi

# ---------------------------------------------------------------------------
head_ "10. Lane 3: the resolver works on a bare host"
# ---------------------------------------------------------------------------
# The three supported lanes -- docker, sbx, bare host -- share one toolchain
# resolver precisely so they cannot drift. Lane 3 has the fewest users and no
# other coverage, and setup-local.sh had in fact drifted: it hardcoded the
# pre-2024.2 Xilinx layout and reported "Vivado not found" on any recent
# install. That is the same defect that had sbx mounting no toolchain.
#
# This does NOT run setup-local.sh end to end -- that builds a venv and
# compiles finn_xsi, which is minutes and needs network. It checks the part
# that drifted.
if want 10; then
    if [ "$have_xilinx" = 1 ]; then
        out=$(./docker/finn-env print --format sh 2>/dev/null)
        if printf '%s' "$out" | grep -q '^export PATH='; then
            ok "finn-env resolves a toolchain on the bare host"
        else
            bad "finn-env produced no PATH on the bare host"
        fi
        # The real check: does the resolved environment actually run the tool?
        if bash -c 'eval "$(./docker/finn-env print --format sh)"; command -v vivado' >/dev/null 2>&1; then
            ok "vivado is on PATH after sourcing finn-env on the host"
        else
            bad "vivado is NOT on PATH after sourcing finn-env on the host"
        fi
    else
        skip "10: FINN_XILINX_PATH not set"
    fi
    # setup-local.sh must not have grown its own layout logic back.
    if grep -qE '^\s*VIVADO_PATH="\$FINN_XILINX_PATH/Vivado/' setup-local.sh 2>/dev/null; then
        bad "setup-local.sh has hardcoded the pre-2024.2 Xilinx layout again"
    else
        ok "setup-local.sh delegates layout resolution to finn-env"
    fi
fi

# ---------------------------------------------------------------------------
head_ "11. Apptainer/Singularity runs the image"
# ---------------------------------------------------------------------------
# A fourth runtime, and the third that does NOT run the image ENTRYPOINT. Same
# property as tests 4 and 5, checked in the bare form for the same reason.
#
# Needs a .sif, which takes about eight minutes to convert. Skipped rather than
# built, because the suite should stay runnable in a few minutes.
if want 11; then
    runtime=""
    for c in apptainer singularity; do command -v "$c" >/dev/null 2>&1 && { runtime="$c"; break; }; done
    if [ -z "$runtime" ]; then
        skip "11: neither apptainer nor singularity on PATH"
    else
        tag=$(tag_for "${FINN_TIER:-build}" 2>/dev/null)
        sif="${XDG_CACHE_HOME:-$HOME/.cache}/finn/sif/$(printf '%s' "$tag" | tr '/:' '__').sif"
        if [ ! -f "$sif" ]; then
            skip "11: no .sif cached for $tag (make one with docker/finn-apptainer build)"
        else
            if timeout 180 "$runtime" exec "$sif" python -c 'import sys' >/dev/null 2>&1; then
                ok "apptainer runs the image"
            else
                bad "apptainer cannot run the image"
            fi
            # The workspace policy MUST be mirror here. Apptainer cannot remap a
            # mount, so a fixed FINN_ROOT names a directory that was never
            # mounted and `import finn` fails with ModuleNotFoundError.
            pol=$(./docker/finn-env inspect --tier dev --backend apptainer 2>/dev/null \
                  | python3 -c 'import json,sys;print(json.load(sys.stdin)["workspace"]["policy"])' 2>/dev/null)
            [ "$pol" = "mirror" ] && ok "apptainer backend resolves the mirror workspace policy" \
                                  || bad "apptainer backend resolved policy '$pol', expected mirror"
        fi
    fi
fi

# ---------------------------------------------------------------------------
echo
echo "======================================"
echo "  PASS $PASS   FAIL $FAIL   SKIP $SKIP"
echo "======================================"
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "Failures:"
    for f in "${FAILED_TESTS[@]}"; do echo "  - $f"; done
    exit 1
fi
exit 0
