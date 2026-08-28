#!/bin/bash
# Copyright (c) 2021, Xilinx
# Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
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

# This entrypoint does only what genuinely depends on mounted state. Everything
# that can be a cached image layer is one:
#
#   - the four `pip install -e` calls are gone; FINN and its deps resolve from
#     $FINN_ROOT via finn_paths.py in site-packages (see docker/finn_paths.py)
#   - the qonnx pyproject.toml mv/trap is gone with them; nothing here writes to
#     the mounted workspace any more
#   - finn_xsi compiles into $FINN_BUILD_DIR on demand, not into the source tree
#     on every start
#
# What remains is sourcing the Xilinx tools, which depends on a read-only mount
# that only exists at runtime.

set -e

# Respect an inherited HOME rather than overwriting it. An orchestrator may have
# provisioned a writable HOME holding state it expects to find again, and the
# unconditional `export HOME=/tmp/home_dir` this used to do discarded it.
#
# "Respect" has to mean "if it is usable", not merely "if it is set". Docker
# sets HOME=/ for a --user uid with no passwd entry, which is set but not
# writable - and things that fall back to ~/.cache (torch.hub, HF, pip) then try
# to create /.cache and fail with EACCES. Test writability, not definedness.
if [ -z "$HOME" ] || [ "$HOME" = "/" ] || ! mkdir -p "$HOME" 2>/dev/null || [ ! -w "$HOME" ]; then
  export HOME=/tmp/home_dir
fi
mkdir -p "$HOME" 2>/dev/null || true

# Give the running uid a resolvable name.
#
# Dropping the baked useradd (identity is a runtime concern) also removed the
# passwd entry it happened to provide, and `docker run --user 1234:1234` maps to
# no entry at all. Anything calling getpass.getuser() then raises
# "getpwuid(): uid not found" - which includes pytest's tmp_path fixture, so
# every test using it fails.
#
# getpass consults LOGNAME/USER/LNAME/USERNAME before the passwd database, so
# exporting a name is enough and needs no root. Prefer the real name when there
# is an entry; fall back to a fixed one when there is not.
if [ -z "$USER" ]; then
  export USER="$(id -un 2>/dev/null || echo finn)"
fi
export LOGNAME="${LOGNAME:-$USER}"

# LIMITATION(finn-root-absolute): FINN has no fixed workspace path, so the image
# cannot know where the source will be until it is running. Derive rather than
# require, so the same image works under host-path mirroring (run-docker.sh,
# sbx) and at a fixed path. See docker/finn_paths.py for the full statement of
# the limitation and the migration if the path ever becomes fixed.
# WORKSPACE_DIR before $PWD: sbx sets it and starts PID 1 with cwd=/, so $PWD
# alone resolves to the wrong place there. run-docker.sh sets neither and relies
# on -w, which $PWD picks up.
export FINN_ROOT="${FINN_ROOT:-${WORKSPACE_DIR:-$PWD}}"

# Match the defaults finn.util.basic applies, so generated Tcl and g++ include
# flags resolve even in a shell that never imports FINN. Already set as ENV in
# the build/build-xrt tiers, where the data lives outside the workspace.
export FINN_HLSLIB_PATH="${FINN_HLSLIB_PATH:-$FINN_ROOT/deps/finn-hlslib}"
export FINN_BOARD_FILES_PATH="${FINN_BOARD_FILES_PATH:-$FINN_ROOT/deps/board_files}"

# Scratch dir for generated HLS/Vivado projects. Derived for the same reason
# FINN_ROOT is: requiring it makes `docker run <image> quicktest.sh` fail with
# nothing but a KeyError, because finn.util.basic reads os.environ["FINN_BUILD_DIR"]
# with no default and pytest turns that into 7 collection errors.
#
# run-docker.sh still passes an explicit FINN_HOST_BUILD_DIR and mounts it, so
# artifacts survive the container there. This default only covers the bare
# `docker run` / sbx case, where an in-container tmpdir is the right scope: a
# build tree that outlives the sandbox has nowhere to live anyway.
export FINN_BUILD_DIR="${FINN_BUILD_DIR:-/tmp/finn_build_$(id -u)}"
mkdir -p "$FINN_BUILD_DIR" 2>/dev/null || true

# colorful terminal output, only for interactive shells
if [ -t 1 ]; then
  export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '
fi

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

yecho () {
  echo -e "${YELLOW}WARNING: $1${NC}"
}

gecho () {
  echo -e "${GREEN}$1${NC}"
}

recho () {
  echo -e "${RED}ERROR: $1${NC}"
}

# Missing FINN source is a WARNING, not a fatal error.
#
# FINN_ROOT is derived from $PWD, and not every launcher starts PID 1 in the
# workspace: sbx starts the container detached with cwd=/, then lands its own
# `exec` sessions in the workspace afterwards. Exiting here killed the container
# before those sessions could ever run, which surfaced only as sbx's opaque
# "failed to run sandbox container".
#
# So warn and carry on. The command still runs, `sbx exec` still lands in the
# workspace with a correct FINN_ROOT, and a genuinely missing mount fails at the
# first import with a clearer message than this check produced.
if [ ! -d "${FINN_ROOT}/src/finn" ]; then
  yecho "No FINN source at ${FINN_ROOT} (FINN_ROOT derived from \$PWD)."
  yecho "If this is a docker run, pass -v <path-to-finn-repo>:<path-to-finn-repo> and -w the same path."
  yecho "Under sbx this is expected at startup; exec sessions land in the workspace."
fi

# ---------------------------------------------------------------------------
# Xilinx tools. Everything below depends on the read-only host mount, which is
# why it cannot move into a layer.
# ---------------------------------------------------------------------------

# Workaround for a FlexLM issue, scoped to the case where a licensed tool is
# actually going to run. See:
# https://community.flexera.com/t5/InstallAnywhere-Forum/Issues-when-running-Xilinx-tools-or-Other-vendor-tools-in-docker/m-p/245820#M10647
if [ -n "$VITIS_PATH" ] || [ -n "$VIVADO_PATH" ] || [ -n "$HLS_PATH" ]; then
  export LD_PRELOAD="${LD_PRELOAD:-/lib/x86_64-linux-gnu/libudev.so.1}"
fi

# The dev tier deliberately has no Xilinx mount, so the detailed "did you mean
# to set VITIS_PATH?" advice is noise there - and with the paths unset it used to
# print "Unable to find /settings64.sh", which reads like a bug. Say it once.
if [ -z "$VITIS_PATH" ] && [ -z "$VIVADO_PATH" ] && [ -z "$HLS_PATH" ]; then
  gecho "No Xilinx tools configured. Vivado, Vitis, HLS and rtlsim are unavailable;"
  gecho "everything else - transformations, ONNX execution, brevitas export, tests"
  gecho "not marked vivado/vitis/board - works. Use the build tier if you need them."
  export PATH=$PATH:$HOME/.local/bin
  exec "$@"
fi

if [ -f "$VITIS_PATH/settings64.sh" ];then
  # source Vitis env.vars
  export XILINX_VITIS=$VITIS_PATH
  export XILINX_XRT=/opt/xilinx/xrt
  # env scripts may return non-zero, so do not let that abort the container
  source "$VITIS_PATH/settings64.sh" || true
  gecho "Found Vitis at $VITIS_PATH"
  if [ -f "$XILINX_XRT/setup.sh" ];then
    # source XRT
    source "$XILINX_XRT/setup.sh" || true
    gecho "Found XRT at $XILINX_XRT"
  else
    # NOT fatal. The `build` tier ships Vivado/HLS support without XRT on
    # purpose -- that is the whole reason it exists as a tier separate from
    # build-xrt -- so a VITIS_PATH pointed at it lands here legitimately.
    # Exiting killed the container outright, which under sbx surfaces only as
    # "failed to run sandbox container" with no cause.
    #
    # Warn and carry on: HLS synthesis and rtlsim need Vivado, never XRT, and
    # those still work. A genuine Vitis/Alveo flow fails later with a message
    # about the thing it actually could not find.
    yecho "XRT not found at $XILINX_XRT."
    yecho "Vitis/Alveo flows need the build-xrt tier; dev and build do not ship XRT."
    yecho "Vivado-only flows (HLS synthesis, rtlsim) are unaffected."
  fi
else
  yecho "Unable to find $VITIS_PATH/settings64.sh"
  yecho "Functionality dependent on Vitis will not be available."
  yecho "If you need Vitis, ensure VITIS_PATH is set correctly and mounted into the Docker container."
  if [ -f "$VIVADO_PATH/settings64.sh" ];then
    # source Vivado env.vars
    export XILINX_VIVADO=$VIVADO_PATH
    source "$VIVADO_PATH/settings64.sh" || true
    gecho "Found Vivado at $VIVADO_PATH"
  else
    yecho "Unable to find $VIVADO_PATH/settings64.sh"
    yecho "Functionality dependent on Vivado will not be available."
    yecho "If you need Vivado, ensure VIVADO_PATH is set correctly and mounted into the Docker container."
  fi
fi

if [ -z "${XILINX_VIVADO}" ]; then
  yecho "finnxsi will be unavailable since Vivado was not found"
else
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/:${XILINX_VIVADO}/lib/lnx64.o
fi
# finn_xsi is NOT compiled here. It is built on demand by finn.xsi.setup, into
# $FINN_BUILD_DIR rather than into the source tree, so a container start no
# longer deposits a build artifact in the mounted workspace - and no longer
# pays for a compile it may never use.

if [ -f "$HLS_PATH/settings64.sh" ];then
  # source Vitis HLS env.vars
  source "$HLS_PATH/settings64.sh" || true
  gecho "Found Vitis HLS at $HLS_PATH"
else
  yecho "Unable to find $HLS_PATH/settings64.sh"
  yecho "Functionality dependent on Vitis HLS will not be available."
  yecho "Please note that FINN needs at least version 2020.2 for Vitis HLS support. Our recommendation is to use version 2022.2"
  yecho "If you need Vitis HLS, ensure HLS_PATH is set correctly and mounted into the Docker container."
fi

# Beta-device Tcl init scripts. Sentinel-guarded so repeated container starts
# against the same HOME are a no-op instead of recopying every time.
if [ -d "$FINN_ROOT/.Xilinx" ] && [ ! -f "$HOME/.Xilinx/.finn-seeded" ]; then
  mkdir -p "$HOME/.Xilinx"
  if [ -f "$FINN_ROOT/.Xilinx/HLS_init.tcl" ]; then
    cp "$FINN_ROOT/.Xilinx/HLS_init.tcl" "$HOME/.Xilinx/"
    gecho "Found HLS_init.tcl and copied to $HOME/.Xilinx/HLS_init.tcl"
  else
    yecho "Unable to find $FINN_ROOT/.Xilinx/HLS_init.tcl"
  fi

  if [ -f "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" ]; then
    mkdir -p "$HOME/.Xilinx/Vivado/"
    cp "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" "$HOME/.Xilinx/Vivado/"
    gecho "Found Vivado_init.tcl and copied to $HOME/.Xilinx/Vivado/Vivado_init.tcl"
  else
    yecho "Unable to find $FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl"
  fi
  touch "$HOME/.Xilinx/.finn-seeded" 2>/dev/null || true
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VITIS_PATH/lnx64/tools/fpo_v7_1:$HLS_PATH/lnx64/tools/fpo_v7_1"

export PATH=$PATH:$HOME/.local/bin

# NOTE: this script deliberately does NOT write /etc/sandbox-persistent.sh.
#
# That file is sbx-managed. An earlier revision wrote it here, and it was dead
# code: sbx replaces the file after the entrypoint runs, so a sandbox always
# showed it 0 bytes despite this running as root with the file writable.
#
# Env for sessions that never run this script is delivered by the FINN kit
# instead (docker/finn.kit): `environment.variables` for FINN_ROOT and
# FINN_BUILD_DIR, which become real process env and so reach even a bare
# `sbx exec <cmd>` with no shell, and `commands.startup` for anything that has
# to be computed at start. Python additionally self-heals via finn_paths, so
# the image still works under plain `docker run` with no kit at all.

exec "$@"
