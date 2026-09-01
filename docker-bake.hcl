# FINN's build matrix. The authoritative definition of what images exist, what
# they are called, and what goes into them.
#
#   docker buildx bake -f docker-bake.hcl                 the base image
#   docker buildx bake -f docker-bake.hcl finn-xrt        + XRT
#   docker buildx bake -f docker-bake.hcl supported       everything CI gates on
#   docker buildx bake -f docker-bake.hcl --print finn    resolved config, no build
#
# Always pass -f docker-bake.hcl. Without it, bake also auto-loads compose.yaml
# and fails on its interpolation before any target builds.
#
# THE SHAPE
# ---------
# One base image, plus two orthogonal layers:
#
#   axis            values                     appears in the tag as
#   base            one                        <git>
#   runtime target  a set: xrt, slash, v80pp   .xrt.slash   (sorted, dot-joined)
#   sbx contract    boolean                    sbx- prefix
#
# There is no profile axis and no tier axis. The profile axis had one member and
# threaded a variable through ten files to select from it. The tier axis encoded
# "does this user have Vivado", which is a host fact that docker/config
# resolves at launch -- see the Dockerfile header.
#
# Use `.` and not `+` to join runtime names. A Docker tag accepts
# [\w][\w.-]{0,127}; `+` gives "invalid reference format".
#
# ENUMERATE SUPPORT; PARAMETERIZE EVERYTHING ELSE
# ------------------------------------------------
# Named targets below declare the combinations FINN keeps green. The generic
# finn-runtime / finn-sbx-runtime targets accept any manifest set without
# generating a powerset of named targets:
#
#   FINN_RUNTIMES=xrt,v80pp docker buildx bake -f docker-bake.hcl finn-runtime
#
# WHAT IS *NOT* HERE
# ------------------
# Dependency commits. deps.env is COPYed into the build context and read by
# fetch-repos.sh inside the image, so it is already authoritative and a change
# to it already invalidates the layer. The *_COMMIT args below exist only as
# one-off overrides and default to empty; do not populate them from a second
# copy of the pins.
#
# Runtime package versions. Those live in docker/runtimes/<name>.env, which the
# build reads directly. HCL has no file() function, so duplicating them here is
# the codegen-goes-stale trap that removed the profile matrix in the first
# place. The tag carries the runtime NAME; the label carries the version.

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Injected by the launchers / CI from `git describe`. HCL cannot shell out, so a
# bare `docker buildx bake` produces the "local" fallback rather than a
# provenance-bearing tag. CI must always pass this.
#
# No --dirty variant. The old `build` tags carried one and `dev` did not, which
# was the right call for `dev` and is now the right call for everything: FINN's
# source is MOUNTED, not baked, so an edited working tree does not change the
# image. A dirty tag would rebuild on every edit under an agent workflow, which
# is the normal state there. The trade is that editing the Dockerfile or
# deps.env does not move the tag on its own; CI builds from a clean tree, so
# this only ever bites locally.
variable "GIT_DESCRIBE" { default = "local" }

variable "REGISTRY" { default = "xilinx/finn" }

# Runtime set for the two generic launcher targets. Named, supported targets
# below remain fixed so CI can build the supported matrix in one invocation.
variable "FINN_RUNTIMES" { default = "" }

# The Ubuntu base. Date-pinned, never the rolling tag.
variable "UBUNTU_TAG" { default = "jammy-20230126" }

# One-off dependency overrides. Empty means "use deps.env from the context".
variable "QONNX_COMMIT" { default = "" }
variable "FINN_EXP_COMMIT" { default = "" }
variable "BREVITAS_COMMIT" { default = "" }
variable "HLSLIB_COMMIT" { default = "" }
variable "AVNET_BDF_COMMIT" { default = "" }
variable "XIL_BDF_COMMIT" { default = "" }
variable "RFSOC4x2_BDF_COMMIT" { default = "" }
variable "KV260_BDF_COMMIT" { default = "" }
variable "AUPZU3_BDF_COMMIT" { default = "" }

# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

# runtimes: a comma-separated list, or "". sbx: true or false.
#
#   ""          -> xilinx/finn:<git>
#   "xrt"       -> xilinx/finn:<git>.xrt
#   "xrt,slash" -> xilinx/finn:<git>.xrt.slash
#   sbx         -> xilinx/finn:sbx-<git>[...]
#
# The runtime part is a function of the SET, so sort before joining. Otherwise
# `xrt,slash` and `slash,xrt` are the same image under two names.
function "runtime_set" {
  params = [runtimes]
  result = runtimes == "" ? "" : join(",", distinct(sort(compact(split(",", runtimes)))))
}

function "tag" {
  params = [runtimes, sbx]
  result = join("", [
    "${REGISTRY}:",
    sbx ? "sbx-" : "",
    GIT_DESCRIBE,
    runtime_set(runtimes) == "" ? "" : ".${join(".", split(",", runtime_set(runtimes)))}",
  ])
}

# ---------------------------------------------------------------------------
# Common target body
# ---------------------------------------------------------------------------

target "_common" {
  dockerfile = "docker/Dockerfile.finn"
  context    = "."
  # linux/amd64 only, and declared rather than assumed. XRT and Vivado are
  # x86-64 only, and parts of the image (the LSB loader symlink, the ncurses 5
  # backport) carry explicit x86-64 assumptions.
  platforms = ["linux/amd64"]
  args = {
    UBUNTU_TAG          = UBUNTU_TAG
    QONNX_COMMIT        = QONNX_COMMIT
    FINN_EXP_COMMIT     = FINN_EXP_COMMIT
    BREVITAS_COMMIT     = BREVITAS_COMMIT
    HLSLIB_COMMIT       = HLSLIB_COMMIT
    AVNET_BDF_COMMIT    = AVNET_BDF_COMMIT
    XIL_BDF_COMMIT      = XIL_BDF_COMMIT
    RFSOC4x2_BDF_COMMIT = RFSOC4x2_BDF_COMMIT
    KV260_BDF_COMMIT    = KV260_BDF_COMMIT
    AUPZU3_BDF_COMMIT   = AUPZU3_BDF_COMMIT
  }
}

# OCI labels, so the image describes itself to anything that inspects it.
#
# These are descriptive, NOT a requirements protocol. There is no OCI convention
# for declaring egress allowlists or mount requirements, and FINN should not
# invent labels that imply a standard exists. What a LANE needs is resolved at
# launch by `finn-env inspect`; what these say is what the image IS.
#
# dev.finn.runtimes carries the names the tag also carries. The exact package
# versions are NOT here, because they live in docker/runtimes/*.env and HCL
# cannot read a file -- restating them would be the stale-copy trap. Read the
# manifest at the recorded revision, or `dpkg -l` inside the image.
function "labels" {
  params = [runtimes, sbx]
  result = {
    "org.opencontainers.image.title"       = "FINN"
    "org.opencontainers.image.description" = "FINN dataflow compiler, Ubuntu 22.04 / Python 3.10"
    "org.opencontainers.image.source"      = "https://github.com/Xilinx/finn"
    "org.opencontainers.image.revision"    = GIT_DESCRIBE
    "dev.finn.runtimes"                    = runtime_set(runtimes)
    # Whether the image grants NOPASSWD root INSIDE the container. This is not
    # host privilege -- no devices, no capabilities, no privileged mode -- but
    # it is a real difference between two otherwise identical images, and the
    # whole reason the sbx variant is a separate target.
    "dev.finn.in-container-root" = sbx ? "true" : "false"
  }
}

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

target "finn" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = "" }
  labels   = labels("", false)
  tags     = [tag("", false)]
}

# Generic targets used by launchers for arbitrary manifest combinations. Bake
# remains authoritative for args, labels and the complete image tag; launchers
# no longer manufacture a target name from a runtime string.
target "finn-runtime" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = runtime_set(FINN_RUNTIMES) }
  labels   = labels(runtime_set(FINN_RUNTIMES), false)
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
  labels   = labels(runtime_set(FINN_RUNTIMES), true)
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
# docker/packages/slash.deb, which FINN does not ship and CI cannot produce, so
# a group containing this target would fail on any machine without the file.
# Build it explicitly once you have the package.
target "finn-slash-xrt" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = "slash,xrt" }
  labels   = labels("slash,xrt", false)
  tags     = [tag("xrt,slash", false)]
}

# Compatibility alias for the pre-canonical target spelling.
target "finn-xrt-slash" {
  inherits = ["finn-slash-xrt"]
}

target "finn-v80pp-xrt" {
  inherits = ["_common"]
  target   = "runtime"
  args     = { FINN_RUNTIMES = "v80pp,xrt" }
  labels   = labels("v80pp,xrt", false)
  tags     = [tag("xrt,v80pp", false)]
}

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

# The base image, not an accelerator variant. The old default was Jenkins
# history: the largest tier with the widest host exposure, for work that mostly
# does not need it.
group "default" {
  targets = ["finn"]
}

# Everything that must build, on any machine, with no supplied packages.
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
