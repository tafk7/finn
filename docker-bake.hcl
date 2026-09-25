# FINN's build matrix: what images exist, what they are called, and what goes
# into them.
#
#   docker buildx bake -f docker-bake.hcl                 the FINN image
#   docker buildx bake -f docker-bake.hcl finn-xrt        + XRT
#   docker buildx bake -f docker-bake.hcl supported       everything CI gates on
#   docker buildx bake -f docker-bake.hcl --print finn    resolved config, no build
#
# Always pass -f docker-bake.hcl. Without it, bake also auto-loads compose.yaml
# and fails on its interpolation before any target builds.
#
# One image, with two orthogonal variants:
#
#   axis            values                        appears in the tag as
#   runtime target  a set: xrt, slash, slashkit   .slash.xrt  (sorted, dot-joined)
#   sbx contract    boolean                       sbx- prefix
#
# plus the release image, which installs a FINN wheel for read-only use.
#
# The tag is FINN_IMAGE_REVISION, a hash of docker/image-inputs.txt computed by
# docker/lib.sh. Package versions come from uv.lock and runtime versions from
# docker/runtimes/*.env; neither is restated here.

variable "FINN_IMAGE_REVISION" { default = "unresolved" }
variable "FINN_SOURCE_REVISION" { default = "unknown" }
variable "FINN_SOURCE_DIRTY" { default = "unknown" }

variable "REGISTRY" { default = "xilinx/finn" }

# Runtime set for the parameterized targets.
variable "FINN_RUNTIMES" { default = "" }

# The Ubuntu base. Date-pinned, never the rolling tag.
variable "UBUNTU_TAG" { default = "noble-20240605" }

# The runtime part of the tag is a function of the SET, so `xrt,slash` and
# `slash,xrt` are one image.
function "runtime_set" {
  params = [runtimes]
  result = runtimes == "" ? "" : join(",", distinct(sort(compact(split(",", runtimes)))))
}

function "runtime_suffix" {
  params = [runtimes]
  result = runtime_set(runtimes) == "" ? "" : ".${join(".", split(",", runtime_set(runtimes)))}"
}

function "tag" {
  params = [runtimes, sbx]
  result = "${REGISTRY}:${sbx ? "sbx-" : ""}${FINN_IMAGE_REVISION}${runtime_suffix(runtimes)}"
}

target "_common" {
  dockerfile = "docker/Dockerfile.finn"
  context    = "."
  # XRT and Vivado are x86-64 only; so is the LSB loader alias in the image.
  platforms = ["linux/amd64"]
  args = {
    UBUNTU_TAG           = UBUNTU_TAG
    FINN_SOURCE_REVISION = FINN_SOURCE_REVISION
    FINN_SOURCE_DIRTY    = FINN_SOURCE_DIRTY
  }
}

# Descriptive labels. What a launch needs is resolved by docker/config.py; these
# say what the image is.
function "labels" {
  params = [runtimes, sbx]
  result = {
    "org.opencontainers.image.title"       = "FINN"
    "org.opencontainers.image.description" = "FINN dataflow compiler, Ubuntu 24.04 / Python 3.12"
    "org.opencontainers.image.source"      = "https://github.com/Xilinx/finn"
    "org.opencontainers.image.version"     = FINN_IMAGE_REVISION
    "dev.finn.image-revision"              = FINN_IMAGE_REVISION
    "dev.finn.runtimes"                    = runtime_set(runtimes)
    # NOPASSWD root inside the container (not host privilege). The reason the
    # sbx variant is a separate target.
    "dev.finn.in-container-root" = sbx ? "true" : "false"
  }
}

target "finn" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = "" }
  labels   = labels("", false)
  tags     = [tag("", false)]
}

# Parameterized over FINN_RUNTIMES, for any manifest set.
target "finn-runtime" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = runtime_set(FINN_RUNTIMES) }
  labels   = labels(FINN_RUNTIMES, false)
  tags     = [tag(FINN_RUNTIMES, false)]
}

target "finn-xrt" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = "xrt" }
  labels   = labels("xrt", false)
  tags     = [tag("xrt", false)]
}

target "finn-sbx" {
  inherits = ["_common"]
  target   = "sbx"
  args     = { FINN_RUNTIMES = "" }
  labels   = labels("", true)
  tags     = [tag("", true)]
}

target "finn-sbx-runtime" {
  inherits = ["_common"]
  target   = "sbx"
  args     = { FINN_RUNTIMES = runtime_set(FINN_RUNTIMES) }
  labels   = labels(FINN_RUNTIMES, true)
  tags     = [tag(FINN_RUNTIMES, true)]
}

target "finn-sbx-xrt" {
  inherits = ["_common"]
  target   = "sbx"
  args     = { FINN_RUNTIMES = "xrt" }
  labels   = labels("xrt", true)
  tags     = [tag("xrt", true)]
}

# Deliberately outside every group. SLASH is SOURCE=supply: it needs
# docker/packages/slash.deb, which FINN does not ship and CI cannot produce.
target "finn-slash-xrt" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = "slash,xrt" }
  labels   = labels("slash,xrt", false)
  tags     = [tag("slash,xrt", false)]
}

target "finn-slashkit-xrt" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = "slash,slashkit,xrt" }
  labels   = labels("slash,slashkit,xrt", false)
  tags     = [tag("slash,slashkit,xrt", false)]
}

# FINN installed from wheels built from this checkout; the source for SIF
# exports. Tagged by the source revision, since FINN's code is part of it.
target "finn-release" {
  inherits = ["_common"]
  target   = "release"
  args     = { FINN_RUNTIMES = runtime_set(FINN_RUNTIMES) }
  labels   = labels(FINN_RUNTIMES, false)
  tags     = ["${REGISTRY}:release-${substr(FINN_SOURCE_REVISION, 0, 12)}${FINN_SOURCE_DIRTY == "1" ? "-dirty" : ""}${runtime_suffix(FINN_RUNTIMES)}"]
}

# Everything that must build on any machine, with no supplied packages.
group "default" {
  targets = ["finn"]
}

group "supported" {
  targets = ["finn", "finn-xrt", "finn-sbx", "finn-sbx-xrt"]
}

group "docker" {
  targets = ["finn", "finn-xrt"]
}

group "sbx" {
  targets = ["finn-sbx", "finn-sbx-xrt"]
}

group "all" {
  targets = ["supported"]
}
