#!/bin/bash
# Copyright (c) 2020-2022, Xilinx, Inc.
# Copyright (C) 2022-2025, Advanced Micro Devices, Inc.
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

# Fail fast so a partial deps/ tree is caught early
# `-u` is deliberately omitted: optional FINN_SKIP_BOARD_FILES flag may be unset.
set -eo pipefail

# absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

# Pins live in deps.env so the Dockerfile's wheel build reads the same values;
# see the header there for the env-override and branch-ref contract.
# shellcheck source=deps.env
. "$SCRIPTPATH/deps.env"

# Which half of deps/ to fetch. `python` is the three importable packages,
# `data` is the HLS headers and board files used by FPGA build flows.
FETCH_GROUP="${1:-all}"
case "$FETCH_GROUP" in
    all)    FETCH_NAMES="$FINN_DEP_GROUP_PYTHON $FINN_DEP_GROUP_DATA" ;;
    python) FETCH_NAMES="$FINN_DEP_GROUP_PYTHON" ;;
    data)   FETCH_NAMES="$FINN_DEP_GROUP_DATA" ;;
    *)
        echo "usage: $0 [all|python|data]" >&2
        echo "  all     both groups (default)" >&2
        echo "  python  qonnx, finn-experimental, brevitas" >&2
        echo "  data    finn-hlslib and board files" >&2
        exit 2
        ;;
esac

# Retry a command with exponential back-off (github 5xx, DNS blips, rate-limit).
retry() {
    local n=0
    local max=5
    local delay=4
    until "$@"; do
        n=$((n+1))
        if (( n >= max )); then
            echo "fetch-repos: command failed after $n attempts: $*" >&2
            return 1
        fi
        echo "fetch-repos: attempt $n/$max failed for: $* (retrying in ${delay}s)" >&2
        sleep "$delay"
        delay=$((delay*2))
    done
}

# Clone fresh, dropping any leftover first so an interrupted clone never makes
# the retry refuse a non-empty target or reuse a tree with no resolvable HEAD.
clone_repo() {
    rm -rf "$2"
    git clone "$1" "$2"
}

fetch_repo() {
    # URL for git repo to be cloned
    local REPO_URL=$1
    # git ref for repo: a SHA, a tag, or a branch name
    local REPO_REF=$2
    # directory to clone to under deps/
    local REPO_DIR=$3
    # absolute path for the repo local copy
    local CLONE_TO=$SCRIPTPATH/deps/$REPO_DIR

    # (re-)clone when the dir is missing or has no resolvable HEAD (what an
    # interrupted clone leaves behind, and what the checkout below needs)
    if [ ! -d "$CLONE_TO" ] || ! git -C "$CLONE_TO" rev-parse HEAD >/dev/null 2>&1; then
        retry clone_repo "$REPO_URL" "$CLONE_TO"
    fi

    # Never move a dep out from under work in progress. qonnx and brevitas are
    # co-developed often enough that a silent checkout here would discard real
    # edits; warn and leave the tree exactly as the developer left it.
    if [ -n "$(git -C "$CLONE_TO" status --porcelain)" ]; then
        echo "fetch-repos: $REPO_DIR has uncommitted changes, leaving it untouched" >&2
        echo "fetch-repos:   at $(git -C "$CLONE_TO" rev-parse --short HEAD), wanted $REPO_REF" >&2
        echo "fetch-repos:   commit or stash them to let fetch-repos manage this dep again" >&2
        return 0
    fi

    # Resolve the ref to a commit so branches and tags work, not just SHAs. A
    # local resolve is tried first to keep the common no-op case offline; only
    # an unknown ref costs a network round trip.
    local WANT_COMMIT
    if ! WANT_COMMIT=$(git -C "$CLONE_TO" rev-parse --verify --quiet "$REPO_REF^{commit}"); then
        retry git -C "$CLONE_TO" fetch --tags --force
        if ! WANT_COMMIT=$(git -C "$CLONE_TO" rev-parse --verify --quiet "$REPO_REF^{commit}"); then
            echo "fetch-repos: ERROR: $REPO_DIR has no ref '$REPO_REF'" >&2
            return 1
        fi
    fi

    local CURRENT_COMMIT
    CURRENT_COMMIT=$(git -C "$CLONE_TO" rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$WANT_COMMIT" ]; then
        # fetch+checkout instead of pull: working copy is a detached HEAD.
        retry git -C "$CLONE_TO" fetch --tags --force
        # Re-resolve after fetching: a branch ref may now point further ahead.
        WANT_COMMIT=$(git -C "$CLONE_TO" rev-parse --verify "$REPO_REF^{commit}")
        git -C "$CLONE_TO" checkout "$WANT_COMMIT"
    fi

    CURRENT_COMMIT=$(git -C "$CLONE_TO" rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$WANT_COMMIT" ]; then
        echo "fetch-repos: ERROR: $REPO_DIR is at $CURRENT_COMMIT, expected $WANT_COMMIT" >&2
        return 1
    fi
    if [ "$REPO_REF" = "$CURRENT_COMMIT" ]; then
        echo "Successfully checked out $REPO_DIR at commit $CURRENT_COMMIT"
    else
        echo "Successfully checked out $REPO_DIR at $REPO_REF ($CURRENT_COMMIT)"
    fi
}

fetch_board_files() {
    echo "Downloading and extracting board files..."
    mkdir -p "$SCRIPTPATH/deps/board_files"
    OLD_PWD=$(pwd)
    cd "$SCRIPTPATH/deps/board_files"
    retry wget -qO pynq-z1.zip https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip
    retry wget -qO pynq-z2.zip https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip
    unzip -q pynq-z1.zip
    unzip -q pynq-z2.zip
    cp -r $SCRIPTPATH/deps/$AVNET_BDF_DIR/* $SCRIPTPATH/deps/board_files/
    cp -r $SCRIPTPATH/deps/$XIL_BDF_DIR/boards/Xilinx/rfsoc2x2 $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$RFSOC4x2_BDF_DIR/board_files/rfsoc4x2 $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$KV260_SOM_BDF_DIR/boards/Xilinx/kv260_som $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$AUPZU3_BDF_DIR/board-files/aup-zu3-8gb $SCRIPTPATH/deps/board_files/;
    cd $OLD_PWD
}

# Indirect through the group lists from deps.env so adding a dep means editing
# one file, and so `python` / `data` stay in sync with what is fetched.
for DEP in $FETCH_NAMES; do
    eval "fetch_repo \"\$${DEP}_URL\" \"\$${DEP}_COMMIT\" \"\$${DEP}_DIR\""
done

# Board files are assembled from the BDF repos above, so they belong to the
# data group and are skipped entirely when only Python deps were requested.
if [ "$FETCH_GROUP" = "python" ]; then
    exit 0
fi

# Can skip downloading of board files entirely if desired
if [ "$FINN_SKIP_BOARD_FILES" = "1" ]; then
    echo "Skipping download and verification of board files"
else
    # download extra board files and extract if needed
    if [ ! -d "$SCRIPTPATH/deps/board_files" ]; then
        fetch_board_files
    else
        cd $SCRIPTPATH
        BOARD_FILES_MD5=$(find deps/board_files/ -type f -exec md5sum {} \; | sort -k 2 | md5sum | cut -d' ' -f 1)
        if [ "$BOARD_FILES_MD5" = "$EXP_BOARD_FILES_MD5" ]; then
            echo "Verified board files folder content md5: $BOARD_FILES_MD5"
        else
            echo "Board files folder md5: expected $BOARD_FILES_MD5 found $EXP_BOARD_FILES_MD5"
            echo "Board files folder content mismatch, removing and re-downloading"
            rm -rf deps/board_files/
            fetch_board_files
        fi
    fi
fi
