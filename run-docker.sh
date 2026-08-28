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

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# green echo
gecho () {
  echo -e "${GREEN}$1${NC}"
}

# red echo
recho () {
  echo -e "${RED}$1${NC}"
}

# Identity is applied at `docker run` time via --user, never baked into the
# image. DOCKER_GNAME and DOCKER_PASSWD are gone with the useradd/groupadd they
# fed: a per-user image made the tag - which encodes only git describe and the
# XRT version - fail to identify its own contents.
DOCKER_GID=$(id -g)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_INST_NAME="finn_dev_${DOCKER_UNAME}"
# ensure Docker inst. name is all lowercase
DOCKER_INST_NAME=$(echo "$DOCKER_INST_NAME" | tr '[:upper:]' '[:lower:]')
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

# the settings below will be taken from environment variables if available,
# otherwise the defaults below will be used
: ${JUPYTER_PORT=8888}
: ${JUPYTER_PASSWD_HASH=""}
: ${NETRON_PORT=8081}
: ${LOCALHOST_URL="localhost"}
: ${NUM_DEFAULT_WORKERS=4}
: ${FINN_SSH_KEY_DIR="$SCRIPTPATH/ssh_keys"}
: ${PLATFORM_REPO_PATHS="/opt/xilinx/platforms"}
# Dependency profile: which (Ubuntu, Python, pin set) combination to build.
#
#   py310  Ubuntu 22.04 / Python 3.10 - the currently validated configuration
#   py312  Ubuntu 24.04 / Python 3.12 - tracks upstream PR 1603
#
# Switching is a single variable:
#
#   FINN_PROFILE=py312 ./run-docker.sh quicktest
#
# The profile carries the XRT package, and the Ubuntu release is derived from
# that further down - so one setting moves OS, Python and every pin together
# rather than leaving them to be kept consistent by hand.
: ${FINN_PROFILE="py310"}
if [ ! -f "$SCRIPTPATH/docker/profiles/$FINN_PROFILE/profile.env" ]; then
  recho "Unknown FINN_PROFILE '$FINN_PROFILE'. Available:"
  for p in "$SCRIPTPATH"/docker/profiles/*/profile.env; do
    [ -f "$p" ] || continue
    d=$(basename "$(dirname "$p")")
    recho "  $d - $(. "$p"; echo "$FINN_PROFILE_DESC")"
  done
  exit 2
fi
. "$SCRIPTPATH/docker/profiles/$FINN_PROFILE/profile.env"

# The profile supplies defaults; an explicit environment value still wins, so a
# one-off XRT override does not require editing or forking a profile.
: ${XRT_DEB_VERSION="$FINN_PROFILE_XRT"}
# Verify what gets installed as root rather than trusting an unauthenticated
# download - but only when the package is still the profile's own. Applying a
# profile's checksum to an overridden XRT_DEB_VERSION would fail the build on a
# checksum that was never meant to describe it. Overriding without a matching
# FINN_XRT_SHA256 falls back to an unverified download, and the build prints the
# observed sha256 so it can be pinned into the profile.
if [ "$XRT_DEB_VERSION" = "$FINN_PROFILE_XRT" ]; then
  : ${FINN_XRT_SHA256="$FINN_PROFILE_XRT_SHA256"}
fi
: ${V80PP_DEB_PACKAGE=""}
: ${FINN_HOST_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}

# Which image tier to build and run:
#
#   dev        no XRT, no Xilinx mount, no build data - the sandboxed-agent
#              image. Runs quicktest.sh and every import-level task.
#   build      + finn-hlslib and board files, for Vivado / Vitis HLS flows.
#   build-xrt  + XRT and v80++, for Vitis / Alveo / V80 targets.
#
# Defaults to build-xrt so an unqualified `./run-docker.sh` behaves exactly as
# it did before the split, which is what Jenkins relies on.
: ${FINN_DOCKER_TARGET="build-xrt"}

# Tag per tier. build-xrt keeps the historical shape - git describe --dirty plus
# the XRT package name - because that is the tag the Jenkins publish path keys
# on and provenance there has to be exact.
#
# dev deliberately drops --dirty: under an agent workflow the tree is dirty by
# definition, so a dirty-tagged image would rebuild on the first edit.
finn_git_describe () {
  ( cd "$SCRIPTPATH" && git describe --always --tags $1 2>/dev/null )
}
#
# The profile is part of the tag for dev and build, because two profiles produce
# materially different images and a tag that did not distinguish them would be
# exactly the unsoundness §2.1 describes. build-xrt needs no such marker: its tag
# already carries XRT_DEB_VERSION, which differs per profile (22.04 vs 24.04),
# and keeping its historical shape is what the Jenkins publish path relies on.
finn_compute_tag () {
  case "$1" in
    dev)       echo "xilinx/finn:dev-$FINN_PROFILE-$(finn_git_describe)" ;;
    build)     echo "xilinx/finn:build-$FINN_PROFILE-$(finn_git_describe --dirty)" ;;
    build-xrt) echo "xilinx/finn:$(finn_git_describe --dirty).$XRT_DEB_VERSION" ;;
    *)         return 1 ;;
  esac
}

: ${FINN_DOCKER_PREBUILT="0"}
: ${FINN_DOCKER_RUN_AS_ROOT="0"}
: ${FINN_DOCKER_EXTRA=""}
: ${FINN_DOCKER_BUILD_EXTRA=""}
: ${FINN_SKIP_DEP_REPOS="0"}
: ${FINN_SKIP_BOARD_FILES="0"}
: ${NVIDIA_VISIBLE_DEVICES=""}
: ${DOCKER_BUILDKIT="1"}
: ${FINN_SINGULARITY=""}
: ${FINN_SKIP_XRT_DOWNLOAD=""}
: ${FINN_XRT_PATH=""}
: ${FINN_DOCKER_NO_CACHE="0"}

if ! FINN_DOCKER_TARGET_TAG=$(finn_compute_tag "$FINN_DOCKER_TARGET"); then
  recho "FINN_DOCKER_TARGET must be one of: dev, build, build-xrt (got '$FINN_DOCKER_TARGET')"
  exit 2
fi
: ${FINN_DOCKER_TAG="$FINN_DOCKER_TARGET_TAG"}

# print-tag emits the Docker image tag and exits, so the Jenkins publish step
# has one source of truth for the tag (FINN_DOCKER_TAG). Placed before any
# side effects so the invocation is read-only.
#
# An optional tier argument lets a caller ask for a tier other than the one it
# would run, without a second copy of the tag rule:
#
#   ./run-docker.sh print-tag dev
if [ "$1" = "print-tag" ]; then
  if [ "$#" -gt 2 ]; then
    echo "Usage: $0 print-tag [dev|build|build-xrt]" >&2
    exit 2
  fi
  if [ "$#" -eq 2 ]; then
    if ! finn_compute_tag "$2"; then
      echo "Usage: $0 print-tag [dev|build|build-xrt]" >&2
      exit 2
    fi
  else
    echo "$FINN_DOCKER_TAG"
  fi
  exit 0
fi

DOCKER_INTERACTIVE=""

# Catch FINN_DOCKER_EXTRA options being passed in without a trailing space
FINN_DOCKER_EXTRA+=" "

# The dev tier is defined by the absence of all of this: no Xilinx mount, no
# licence, no platform repos. Warning about unset Xilinx variables there would
# be noise about a deliberate property of the image.
if [ "$FINN_DOCKER_TARGET" != "dev" ]; then
  if [ -z "$FINN_XILINX_PATH" ];then
    recho "Please set the FINN_XILINX_PATH environment variable to the path to your Xilinx tools installation directory (e.g. /opt/Xilinx)."
    recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
  fi

  if [ -z "$FINN_XILINX_VERSION" ];then
    recho "Please set the FINN_XILINX_VERSION to the version of the Xilinx tools to use (e.g. 2022.2)"
    recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
  fi
fi

# Vitis/Alveo-only prerequisites: only build-xrt can use them at all.
if [ "$FINN_DOCKER_TARGET" = "build-xrt" ]; then
  if [ -z "$PLATFORM_REPO_PATHS" ];then
    recho "Please set PLATFORM_REPO_PATHS pointing to Vitis platform files (DSAs)."
    recho "This is required to be able to use Vitis-based Alveo PCIe cards."
  fi

  if [ -z "$V80PP_DEB_PACKAGE" ];then
    recho "Please set V80PP_DEB_PACKAGE pointing to the SLASH v80++ .deb package."
    recho "This is required to be able to use the Alveo V80 card."
  fi
fi

# Mirror the Jenkinsfile's local-fallback banner, but only inside a real
# Jenkins run (JENKINS_URL + BUILD_NUMBER) so unrelated CI systems and
# developer shells that happen to export BUILD_NUMBER stay quiet.
if [ -n "$JENKINS_URL" ] && [ -n "$BUILD_NUMBER" ] \
   && [ -z "$FINN_CI_NFS_ROOT" ] && [ -z "$FINN_DOCKER_SHARED_IMAGE_DIR" ]; then
  recho "FINN_CI_NFS_ROOT and FINN_DOCKER_SHARED_IMAGE_DIR are unset. Running in local-fallback mode."
  recho "  - no shared Docker image cache (this agent will build locally)"
  recho "  - no build-to-HW artifact handoff (the HW pipeline cannot test this build)"
  recho "Set FINN_CI_NFS_ROOT in the Jenkins job DSL to enable the shared cache."
fi

# `build` stops after producing the image, for callers that need the image to
# exist without running anything in it:
#
#   FINN_DOCKER_TARGET=dev ./run-docker.sh build
BUILD_ONLY="0"
if [ "$1" = "sbx" ]; then
  # ./run-docker.sh sbx [dev|build|build-xrt]
  if [ -n "$2" ]; then
    FINN_DOCKER_TARGET="$2"
    if ! FINN_DOCKER_TAG=$(finn_compute_tag "$FINN_DOCKER_TARGET"); then
      recho "Usage: $0 sbx [dev|build|build-xrt]"
      exit 2
    fi
  fi
  gecho "sbx mode, tier $FINN_DOCKER_TARGET"
  FINN_SBX_MODE="1"
  DOCKER_CMD="true"
elif [ "$1" = "build" ]; then
  if [ -n "$2" ]; then
    FINN_DOCKER_TARGET="$2"
    if ! FINN_DOCKER_TAG=$(finn_compute_tag "$FINN_DOCKER_TARGET"); then
      recho "Usage: $0 build [dev|build|build-xrt]"
      exit 2
    fi
  fi
  gecho "Building $FINN_DOCKER_TARGET image only"
  BUILD_ONLY="1"
  DOCKER_CMD="true"
elif [ "$1" = "test" ]; then
  gecho "Running test suite (all tests)"
  DOCKER_CMD="pytest"
elif [ "$1" = "quicktest" ]; then
  gecho "Running test suite (non-Vivado, non-slow tests)"
  DOCKER_CMD="quicktest.sh"
elif [ "$1" = "notebook" ]; then
  gecho "Running Jupyter notebook server"
  if [ -z "$JUPYTER_PASSWD_HASH" ]; then
    JUPYTER_PASSWD_ARG=""
  else
    JUPYTER_PASSWD_ARG="--NotebookApp.password='$JUPYTER_PASSWD_HASH'"
  fi
  DOCKER_CMD="jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port $JUPYTER_PORT $JUPYTER_PASSWD_ARG notebooks"
  FINN_DOCKER_EXTRA+="-e JUPYTER_PORT=$JUPYTER_PORT "
  FINN_DOCKER_EXTRA+="-e NETRON_PORT=$NETRON_PORT "
  if [ -z "$FINN_SINGULARITY" ]; then
    FINN_DOCKER_EXTRA+="-p $JUPYTER_PORT:$JUPYTER_PORT "
    FINN_DOCKER_EXTRA+="-p $NETRON_PORT:$NETRON_PORT "
  fi
elif [ "$1" = "build_dataflow" ]; then
  BUILD_DATAFLOW_DIR=$(readlink -f "$2")
  FINN_DOCKER_EXTRA+="-v $BUILD_DATAFLOW_DIR:$BUILD_DATAFLOW_DIR "
  DOCKER_INTERACTIVE="-it"
  #FINN_HOST_BUILD_DIR=$BUILD_DATAFLOW_DIR/build
  gecho "Running build_dataflow for folder $BUILD_DATAFLOW_DIR"
  DOCKER_CMD="build_dataflow $BUILD_DATAFLOW_DIR"
elif [ "$1" = "build_custom" ]; then
  BUILD_CUSTOM_DIR=$(readlink -f "$2")
  FLOW_NAME=${3:-build}
  FINN_DOCKER_EXTRA+="-v $BUILD_CUSTOM_DIR:$BUILD_CUSTOM_DIR -w $BUILD_CUSTOM_DIR "
  DOCKER_INTERACTIVE="-it"
  #FINN_HOST_BUILD_DIR=$BUILD_DATAFLOW_DIR/build
  gecho "Running build_custom: $BUILD_CUSTOM_DIR/$FLOW_NAME.py"
  DOCKER_CMD="python -mpdb -cc -cq $FLOW_NAME.py ${@:4}"
elif [ -z "$1" ]; then
   gecho "Running container only"
   DOCKER_CMD="bash"
   DOCKER_INTERACTIVE="-it"
else
  gecho "Running container with passed arguments"
  DOCKER_CMD="$@"
fi

# ensure build dir exists locally
mkdir -p $FINN_HOST_BUILD_DIR
mkdir -p $FINN_SSH_KEY_DIR

gecho "Docker container is named $DOCKER_INST_NAME"
gecho "Docker tag is named $FINN_DOCKER_TAG"
gecho "Mounting $FINN_HOST_BUILD_DIR into $FINN_HOST_BUILD_DIR"
gecho "Mounting $FINN_XILINX_PATH into $FINN_XILINX_PATH"
gecho "Port-forwarding for Jupyter $JUPYTER_PORT:$JUPYTER_PORT"
gecho "Port-forwarding for Netron $NETRON_PORT:$NETRON_PORT"

# Ensure git-based deps are checked out at correct commit.
#
# The dev tier gets the Python packages only: finn-hlslib and the board files
# are ~900 MB of data it has no way to use, and the build tiers now carry their
# own baked copies rather than reading them from the workspace.
if [ "$FINN_SKIP_DEP_REPOS" = "0" ]; then
  if [ "$FINN_DOCKER_TARGET" = "dev" ]; then
    ./fetch-repos.sh python || exit 1
  else
    ./fetch-repos.sh || exit 1
  fi
fi

# Dependency pins, so the commits baked into the image are the same ones the
# host checkout is on. Overriding QONNX_COMMIT et al. reaches both.
. "$SCRIPTPATH/deps.env"

# Derive the Ubuntu release from the XRT package rather than pinning it
# separately. The package name encodes three coordinates:
#
#   xrt_202420.2.18.179_24.04-amd64-xrt
#       └ version ──┘ └ OS ┘ └arch┘
#
# and setting a 24.04 package against a Jammy base installs a Noble package on
# the wrong OS - which apt will generally resolve rather than refuse, yielding a
# subtly wrong runtime instead of a clean failure. The Dockerfile re-checks this
# from inside the image and fails the build on disagreement.
#
# The codename maps to a date-pinned snapshot tag, not the rolling one. Canonical
# republishes `ubuntu:jammy` regularly, so building from it would make the image
# contents drift under an unchanged FINN tag - the same class of unsoundness as
# the per-user images this redesign removed, just slower moving. Bump these
# deliberately.
XRT_OS_VERSION=$(echo "$XRT_DEB_VERSION" | sed -n 's/.*_\([0-9][0-9]\.[0-9][0-9]\)-.*/\1/p')
case "$XRT_OS_VERSION" in
  20.04) UBUNTU_CODENAME="focal"; UBUNTU_TAG="focal-20240427" ;;
  22.04) UBUNTU_CODENAME="jammy"; UBUNTU_TAG="jammy-20230126" ;;
  24.04) UBUNTU_CODENAME="noble"; UBUNTU_TAG="noble-20240801" ;;
  *)
    if [ "$FINN_DOCKER_TARGET" = "build-xrt" ] && [ -z "$FINN_SKIP_XRT_DOWNLOAD" ]; then
      recho "Cannot derive an Ubuntu release from XRT_DEB_VERSION='$XRT_DEB_VERSION'"
      recho "Expected xrt_<version>_<os>-<arch>-xrt with os one of 20.04, 22.04, 24.04."
      exit 1
    fi
    # dev and build never install XRT, so an unparseable package name is not
    # their problem; fall back to the release those tiers are developed against.
    UBUNTU_CODENAME="jammy"; UBUNTU_TAG="jammy-20230126"
    ;;
esac

# If xrt path given, copy .deb file to this repo. Gate on the .deb
# itself, not the dir. Otherwise an empty cache dir trips LOCAL_XRT=1
# without producing a build-context .deb, and the docker build then
# fails because the wget branch is also skipped.
if [ -f "$FINN_XRT_PATH/$XRT_DEB_VERSION.deb" ]; then
  cp "$FINN_XRT_PATH/$XRT_DEB_VERSION.deb" .
  export LOCAL_XRT=1
fi

# If v80++ deb package given, copy it to repo root for docker build
if [ -n "$V80PP_DEB_PACKAGE" ] && [ -f "$V80PP_DEB_PACKAGE" ]; then
  cp "$V80PP_DEB_PACKAGE" ./v80pp.deb
fi

if [ "$FINN_DOCKER_NO_CACHE" = "1" ]; then
  FINN_DOCKER_BUILD_EXTRA+="--no-cache "
fi

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
  if [ -f "$SHARED_IMG" ] && [ -f "$SHARED_TAG_FILE" ]; then
    gecho "Loading Docker image from shared storage ($SHARED_DIR)..."
    SHARED_TAG=$(cat "$SHARED_TAG_FILE")
    if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ "$SHARED_TAG" != "$FINN_DOCKER_TAG" ]; then
      recho "Shared Docker tag $SHARED_TAG does not match requested tag $FINN_DOCKER_TAG"
      exit 1
    fi
    # local /tmp lock to serialise concurrent loads on the same host
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
  fi
  if [ "$SHARED_LOADED" != "1" ] && [ "$FINN_DOCKER_PREBUILT" != "1" ]; then
    gecho "WARNING: No usable shared Docker image found at FINN_DOCKER_SHARED_IMAGE_DIR=$SHARED_DIR. Falling back to local build"
  fi
  if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ "$SHARED_LOADED" != "1" ]; then
    recho "FINN_DOCKER_PREBUILT=1 but no usable shared Docker image at FINN_DOCKER_SHARED_IMAGE_DIR=$SHARED_DIR (expected finn-docker-image.tar.gz and finn-docker-tag.txt)"
    exit 1
  fi
fi

# Build the FINN Docker image
if [ "$FINN_DOCKER_PREBUILT" = "0" ] && [ -z "$FINN_SINGULARITY" ]; then
  # Need to ensure this is done within the finn/ root folder:
  OLD_PWD=$(pwd)
  cd $SCRIPTPATH
  # Export DOCKER_BUILDKIT to enable BuildKit features
  export DOCKER_BUILDKIT
  # No identity build args. The image is user-agnostic: identity is applied at
  # runtime via --user below, or by whatever orchestrator runs it. Baking a user
  # made two developers produce materially different images under an identical
  # tag, which made every tag-keyed path - the Jenkins publish step,
  # FINN_DOCKER_SHARED_IMAGE_DIR, any registry push - unsound.
  docker build \
    -f docker/Dockerfile.finn \
    --target=$FINN_DOCKER_TARGET \
    --build-arg FINN_PROFILE=$FINN_PROFILE \
    --build-arg UBUNTU_TAG=$UBUNTU_TAG \
    --build-arg XRT_DEB_VERSION=$XRT_DEB_VERSION \
    --build-arg XRT_DEB_SHA256=$FINN_XRT_SHA256 \
    --build-arg SKIP_XRT=$FINN_SKIP_XRT_DOWNLOAD \
    --build-arg LOCAL_XRT=$LOCAL_XRT \
    --build-arg V80PP_DEB_PACKAGE=$V80PP_DEB_PACKAGE \
    --build-arg QONNX_COMMIT=$QONNX_COMMIT \
    --build-arg FINN_EXP_COMMIT=$FINN_EXP_COMMIT \
    --build-arg BREVITAS_COMMIT=$BREVITAS_COMMIT \
    --build-arg HLSLIB_COMMIT=$HLSLIB_COMMIT \
    --build-arg AVNET_BDF_COMMIT=$AVNET_BDF_COMMIT \
    --build-arg XIL_BDF_COMMIT=$XIL_BDF_COMMIT \
    --build-arg RFSOC4x2_BDF_COMMIT=$RFSOC4x2_BDF_COMMIT \
    --build-arg KV260_BDF_COMMIT=$KV260_BDF_COMMIT \
    --build-arg AUPZU3_BDF_COMMIT=$AUPZU3_BDF_COMMIT \
    --tag=$FINN_DOCKER_TAG $FINN_DOCKER_BUILD_EXTRA \
    . || { recho "docker build failed"; exit 1; }
  cd $OLD_PWD
fi

# Remove local xrt.deb file from repo
if [ ! -z "$LOCAL_XRT" ];then
  rm $XRT_DEB_VERSION.deb
fi

# Remove local v80pp.deb file from repo
if [ -f "./v80pp.deb" ]; then
  rm ./v80pp.deb
fi

# Launch container with current directory mounted
# important to pass the --init flag here for correct Vivado operation, see:
# https://stackoverflow.com/questions/55733058/vivado-synthesis-hangs-in-docker-container-spawned-by-jenkins
DOCKER_BASE="docker run -t --rm $DOCKER_INTERACTIVE --tty --init --hostname $DOCKER_INST_NAME "
# Only genuine host bindings are passed as -e. SHELL, LOCALHOST_URL,
# NUM_DEFAULT_WORKERS, XILINX_LOCAL_USER_DATA, VIVADO_IP_CACHE and the locale are
# ENV in the image now. That is not cosmetic: sbx has no create-time env
# injection seam, so anything left in this tier simply cannot be delivered to a
# microVM. LD_PRELOAD moved into the entrypoint, which applies it only when it
# has actually found Xilinx tools to source.
# LIMITATION(finn-root-absolute): the workspace is mounted at its host path, so
# FINN_ROOT differs per developer and cannot be baked. This mirrors what sbx
# does - it attaches every mount at its host path and offers no container-side
# remapping - so the two agree, but neither can offer a fixed path. See
# docker/finn_paths.py for the migration if that ever changes.
DOCKER_EXEC="-w $SCRIPTPATH "
DOCKER_EXEC+="-v $SCRIPTPATH:$SCRIPTPATH "
DOCKER_EXEC+="-v $FINN_HOST_BUILD_DIR:$FINN_HOST_BUILD_DIR "
DOCKER_EXEC+="-e FINN_BUILD_DIR=$FINN_HOST_BUILD_DIR "
DOCKER_EXEC+="-e FINN_ROOT="$SCRIPTPATH" "
# Non-default values still have to be forwarded; the ENV default only covers the
# case where the caller did not override it.
if [ "$LOCALHOST_URL" != "localhost" ]; then
  DOCKER_EXEC+="-e LOCALHOST_URL=$LOCALHOST_URL "
fi
if [ "$NUM_DEFAULT_WORKERS" != "4" ]; then
  DOCKER_EXEC+="-e NUM_DEFAULT_WORKERS=$NUM_DEFAULT_WORKERS "
fi
if [ -n "$FINN_DEPS" ]; then
  DOCKER_EXEC+="-e FINN_DEPS=$FINN_DEPS "
fi
# Optional host cache for torch.hub / huggingface weights to avoid CDN 504s
# on parallel CI runs. Bind target is /finn_cache (NOT $HOME, because docker
# creates bind parents as root and that would break pip install --user).
: ${FINN_DOCKER_CACHE_DIR=""}
if [ -n "$FINN_DOCKER_CACHE_DIR" ]; then
  mkdir -p "$FINN_DOCKER_CACHE_DIR/torch" "$FINN_DOCKER_CACHE_DIR/huggingface"
  DOCKER_EXEC+="-v $FINN_DOCKER_CACHE_DIR:/finn_cache "
  DOCKER_EXEC+="-e TORCH_HOME=/finn_cache/torch "
  DOCKER_EXEC+="-e HF_HOME=/finn_cache/huggingface "
fi
if [ "$FINN_DOCKER_RUN_AS_ROOT" = "0" ] && [ -z "$FINN_SINGULARITY" ];then
  DOCKER_EXEC+="-v $FINN_SSH_KEY_DIR:$HOME/.ssh "
  DOCKER_EXEC+="--user $DOCKER_UID:$DOCKER_GID "
else
  DOCKER_EXEC+="-v $FINN_SSH_KEY_DIR:/root/.ssh "
fi
if [ ! -z "$IMAGENET_VAL_PATH" ];then
  DOCKER_EXEC+="-v $IMAGENET_VAL_PATH:$IMAGENET_VAL_PATH "
  DOCKER_EXEC+="-e IMAGENET_VAL_PATH=$IMAGENET_VAL_PATH "
fi
# The Xilinx install is 267 GB and cannot be baked; it is mounted from the host.
# The dev tier never mounts it - that, plus no licence variable, is what allows
# the dev sandbox to run on a closed network.
if [ ! -z "$FINN_XILINX_PATH" ] && [ "$FINN_DOCKER_TARGET" != "dev" ];then
  if [[ "$FINN_XILINX_VERSION" =~ ^20([0-9]{2})\.(1|2)$ ]]; then
    year="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"

    # Convert to integers for comparison
    year=$((10#$year))
    minor=$((10#$minor))

    if (( year > 24 )) || { (( year == 24 )) && (( minor > 2 )); }; then
      VIVADO_PATH="$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vivado"
      VITIS_PATH="$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vitis"
      HLS_PATH="$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vitis"
    else
      VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"
      VITIS_PATH="$FINN_XILINX_PATH/Vitis/$FINN_XILINX_VERSION"
      HLS_PATH="$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION"
    fi
  else
    echo "FINN_XILINX_VERSION ($FINN_XILINX_VERSION) is not in the correct format (YYYY.1 or YYYY.2)"
  fi
  DOCKER_EXEC+="-v $FINN_XILINX_PATH:$FINN_XILINX_PATH "
  if [ -d "$VIVADO_PATH" ];then
    DOCKER_EXEC+="-e "XILINX_VIVADO=$VIVADO_PATH" "
    DOCKER_EXEC+="-e VIVADO_PATH=$VIVADO_PATH "
  fi
  if [ -d "$HLS_PATH" ];then
    DOCKER_EXEC+="-e HLS_PATH=$HLS_PATH "
  fi
  if [ -d "$VITIS_PATH" ];then
    DOCKER_EXEC+="-e VITIS_PATH=$VITIS_PATH "
  fi
  if [ -d "$PLATFORM_REPO_PATHS" ];then
    DOCKER_EXEC+="-v $PLATFORM_REPO_PATHS:$PLATFORM_REPO_PATHS "
    DOCKER_EXEC+="-e PLATFORM_REPO_PATHS=$PLATFORM_REPO_PATHS "
  fi
fi

# Licensing. Previously not passed at all, so a container could find the tools
# and then fail to check out a licence.
#
# XILINXD_LICENSE_FILE (and the older LM_LICENSE_FILE) accept a colon-separated
# list whose entries are either form:
#
#   PORT@HOST   a floating FLEXlm server. Nothing to mount; the sandbox needs
#               raw TCP egress to that host:port. Note that is NOT expressible
#               in a domain-oriented egress policy - see docs/containerization.md.
#   /path/to.lic  a node-locked file, which must be readable INSIDE the
#               container. Mounted read-only at its own host path, so the
#               variable's value needs no rewriting.
#
# Both forms are passed through verbatim; only the path form implies a mount.
finn_license_mounts () {
  local var_value=$1 entry dir
  local IFS=':'
  for entry in $var_value; do
    case "$entry" in
      ''|*@*) continue ;;          # empty, or PORT@HOST - server, no mount
    esac
    # Mount the containing directory rather than the file: Vivado writes
    # sibling lock/state files next to some licences, and a single-file bind
    # mount cannot accommodate that.
    dir=$(dirname "$entry")
    [ -d "$dir" ] || { yecho "licence path $entry not found on the host; not mounted"; continue; }
    printf -- '-v %s:%s:ro ' "$dir" "$dir"
  done
}

for _lic_var in XILINXD_LICENSE_FILE LM_LICENSE_FILE; do
  eval "_lic_val=\${$_lic_var:-}"
  [ -z "$_lic_val" ] && continue
  DOCKER_EXEC+="-e $_lic_var=$_lic_val "
  DOCKER_EXEC+="$(finn_license_mounts "$_lic_val")"
done
unset _lic_var _lic_val

# This part is used for internal ci for finn-examples
# if using build verification for finn-examples ci, set up the necessary Docker variables
if [ "$VERIFICATION_EN" = 1 ]; then
  if [ -z "$FINN_EXAMPLES_ROOT" ]; then
    recho "FINN_EXAMPLES_ROOT path has not been set."
    recho "Please set FINN_EXAMPLES_ROOT path to enable verification."
    exit -1
  elif [ ! -d "${FINN_EXAMPLES_ROOT}/ci" ]; then
    recho "ci folder not found in ${FINN_EXAMPLES_ROOT}."
    recho "Please ensure the FINN-examples repo has been set up correctly, and FINN_EXAMPLES_ROOT path is set correctly, to enable verification."
    exit -1
  elif [ -z "$VERIFICATION_IO" ]; then
    recho "VERIFICATION_IO paths has not been set."
    recho "Please ensure the path to the input and expected output files has been set correctly to eneable verification."
    exit -1
  elif [ ! -d "$VERIFICATION_IO" ]; then
    recho "${VERIFICATION_IO} is not a directory."
    recho "Please ensure the VERIFICATION_IO path has been set to the directory containing the input and expected output files for verification."
    exit -1
  else
    DOCKER_EXEC+="-e VERIFICATION_EN=$VERIFICATION_EN "
    DOCKER_EXEC+="-e FINN_EXAMPLES_ROOT=$FINN_EXAMPLES_ROOT "
    DOCKER_EXEC+="-e VERIFICATION_IO=$VERIFICATION_IO "
    FINN_DOCKER_EXTRA+="-v $FINN_EXAMPLES_ROOT/ci:$FINN_EXAMPLES_ROOT/ci "
    FINN_DOCKER_EXTRA+="-v $VERIFICATION_IO:$VERIFICATION_IO "
  fi
fi


DOCKER_EXEC+="$FINN_DOCKER_EXTRA "

if [ "$BUILD_ONLY" = "1" ]; then
  gecho "Built $FINN_DOCKER_TAG"
  exit 0
fi

# ---------------------------------------------------------------------------
# sbx mode: create-or-attach a Docker Sandboxes microVM.
#
# Placed here so it inherits everything resolved above -- the tier tag, the
# Xilinx paths, and the licence handling -- instead of duplicating any of it.
#
# The point is parity with the docker path: one command, repeatable. Without
# this, using sbx meant hand-running docker save, sbx template load, a long
# sbx create, and sbx run, and redoing the first three on every commit because
# the tag embeds `git describe`.
# ---------------------------------------------------------------------------
if [ "$FINN_SBX_MODE" = "1" ]; then
  command -v sbx >/dev/null || { recho "sbx not found on PATH"; exit 1; }

  # One sandbox per (tier, checkout). Lowercased; sbx requires it.
  SBX_NAME=${FINN_SBX_NAME:-"finn-$FINN_DOCKER_TARGET-$(basename "$SCRIPTPATH")"}
  SBX_NAME=$(echo "$SBX_NAME" | tr '[:upper:]' '[:lower:]')

  if sbx ls 2>/dev/null | awk '{print $1}' | grep -qx "$SBX_NAME"; then
    if [ "$FINN_SBX_RECREATE" = "1" ]; then
      gecho "Removing existing sandbox $SBX_NAME (FINN_SBX_RECREATE=1)"
      sbx rm --force "$SBX_NAME" >/dev/null 2>&1
    else
      gecho "Attaching to existing sandbox $SBX_NAME"
      gecho "(FINN_SBX_RECREATE=1 to rebuild it against the current image)"
      exec sbx run --name "$SBX_NAME"
    fi
  fi

  # sbx keeps its own image store and cannot see the local docker one, so the
  # image has to be exported and loaded. Skipped when the tag is already there,
  # which is what makes repeat runs fast -- only a new tag pays the ~2 min.
  if sbx template ls 2>/dev/null | awk '{print $1":"$2}' | grep -qx "docker.io/$FINN_DOCKER_TAG"; then
    gecho "Template $FINN_DOCKER_TAG already loaded"
  else
    gecho "Loading $FINN_DOCKER_TAG into the sbx template store (first time is slow)"
    SBX_TAR=$(mktemp -t finn-sbx-XXXXXX.tar)
    docker save -o "$SBX_TAR" "$FINN_DOCKER_TAG" || { recho "docker save failed"; rm -f "$SBX_TAR"; exit 1; }
    sbx template load "$SBX_TAR" || { recho "sbx template load failed"; rm -f "$SBX_TAR"; exit 1; }
    rm -f "$SBX_TAR"
  fi

  # Extra workspaces are POSITIONAL in sbx, and every mount lands at its host
  # path -- which is what FINN needs anyway, since generated Vivado projects
  # embed absolute paths (see docs/containerization.md).
  SBX_ARGS=("$SCRIPTPATH")
  SBX_ENV=()

  # The dev tier gets NO toolchain, NO licence and NO extra egress. That is the
  # entire point of it: the containment boundary is dev vs the rest, so leaking
  # a licence variable in here because it happens to be set in the caller's
  # shell would quietly widen the profile that exists to be narrow.
  if [ "$FINN_DOCKER_TARGET" = "dev" ]; then
    gecho "dev tier: no Xilinx mount, no licence, no toolchain egress"
  else
  [ -d "$FINN_XILINX_PATH/$FINN_XILINX_VERSION" ] && \
    SBX_ARGS+=("$FINN_XILINX_PATH/$FINN_XILINX_VERSION:ro")

  for v in VIVADO_PATH VITIS_PATH HLS_PATH PLATFORM_REPO_PATHS \
           XILINXD_LICENSE_FILE LM_LICENSE_FILE NUM_DEFAULT_WORKERS; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && SBX_ENV+=(-e "$v=$val")
  done
  # Node-locked licences are files that must exist inside the sandbox; mount
  # their directory read-only at its own path, as the docker path does.
  for lv in XILINXD_LICENSE_FILE LM_LICENSE_FILE; do
    eval "lval=\${$lv:-}"
    [ -z "$lval" ] && continue
    ( IFS=':'; for entry in $lval; do
        case "$entry" in ''|*@*) continue ;; esac
        d=$(dirname "$entry"); [ -d "$d" ] && echo "$d"
      done ) | sort -u | while read -r d; do SBX_ARGS+=("$d:ro"); done
  done

  fi

  gecho "Creating sandbox $SBX_NAME from $FINN_DOCKER_TAG"
  sbx create "${FINN_SBX_AGENT:-shell}" "${SBX_ARGS[@]}" \
    --name "$SBX_NAME" \
    --template "$FINN_DOCKER_TAG" \
    --kit "$SCRIPTPATH/docker/finn.kit" \
    --no-share-skills \
    "${SBX_ENV[@]}" || { recho "sbx create failed"; exit 1; }

  # A floating licence needs raw TCP egress to PORT@HOST. sbx grants that
  # narrowly, per sandbox, so this does not require an open posture.
  for lv in XILINXD_LICENSE_FILE LM_LICENSE_FILE; do
    [ "$FINN_DOCKER_TARGET" = "dev" ] && continue
    eval "lval=\${$lv:-}"
    [ -z "$lval" ] && continue
    ( IFS=':'; for entry in $lval; do
        case "$entry" in *@*) ;; *) continue ;; esac
        port=${entry%@*}; host=${entry#*@}
        gecho "Allowing licence server egress: $host:$port"
        sbx policy allow network --sandbox "$SBX_NAME" "$host:$port" >/dev/null 2>&1
      done )
  done

  gecho "Sandbox $SBX_NAME ready"
  exec sbx run --name "$SBX_NAME"
fi

if [ -z "$FINN_SINGULARITY" ];then
  CMD_TO_RUN="$DOCKER_BASE $DOCKER_EXEC $FINN_DOCKER_TAG $DOCKER_CMD"
else
  SINGULARITY_BASE="singularity exec"
  # Replace command options for Singularity
  SINGULARITY_EXEC="${DOCKER_EXEC//-e /--env }"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//-v /-B }"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//-w /--pwd }"
  CMD_TO_RUN="$SINGULARITY_BASE $SINGULARITY_EXEC $FINN_SINGULARITY /usr/local/bin/finn_entrypoint.sh $DOCKER_CMD"
  gecho "FINN_SINGULARITY is set, launching Singularity container instead of Docker"
fi

echo $CMD_TO_RUN
$CMD_TO_RUN
