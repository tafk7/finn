# FINN's build matrix. The authoritative definition of what images exist, what
# they are called, and what goes into them.
#
#   docker buildx bake dev-py310
#   docker buildx bake supported          # everything CI must keep green
#   docker buildx bake --print dev-py310  # resolved config, no build
#
# This replaces the tag computation and the sixteen --build-arg flags that lived
# in run-docker.sh. Tags are derived HERE and nowhere else, so CI references a
# target by name instead of re-deriving a tag string and hoping it matches.
#
# WHY THE PROFILE MATRIX MOVED HERE FROM docker/profiles/*/profile.env
# --------------------------------------------------------------------
# bake's HCL has no file() function and does not read .env, so the only native
# inputs are variables and override files. Keeping the profile values in a
# shell file would have meant either a codegen step (which goes stale silently)
# or a mandatory wrapper (so `docker buildx bake dev-py312` alone would
# silently build with py310's XRT package -- a footgun worse than the problem).
# A build matrix is what this file is for; the values live here.
#
# WHAT IS *NOT* HERE
# ------------------
# Dependency commits. deps.env is COPYed into the build context and read by
# fetch-repos.sh inside the image, so it is already authoritative and a change
# to it already invalidates the layer. The *_COMMIT args below exist only as
# one-off overrides and default to empty; do not populate them from a second
# copy of the pins.

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Injected by run-docker.sh / CI from `git describe`. HCL cannot shell out, so a
# bare `docker buildx bake` produces the "local" fallback rather than a
# provenance-bearing tag. CI must always pass this.
variable "GIT_DESCRIBE" { default = "local" }
variable "GIT_DESCRIBE_DIRTY" { default = "local" }

variable "REGISTRY" { default = "xilinx/finn" }

# One-off overrides. Empty means "use deps.env from the build context".
variable "QONNX_COMMIT" { default = "" }
variable "FINN_EXP_COMMIT" { default = "" }
variable "BREVITAS_COMMIT" { default = "" }
variable "HLSLIB_COMMIT" { default = "" }
variable "AVNET_BDF_COMMIT" { default = "" }
variable "XIL_BDF_COMMIT" { default = "" }
variable "RFSOC4x2_BDF_COMMIT" { default = "" }
variable "KV260_BDF_COMMIT" { default = "" }
variable "AUPZU3_BDF_COMMIT" { default = "" }

# Local XRT .deb in the build context instead of a download, and the escape
# hatch for building an XRT tier with no XRT at all.
variable "LOCAL_XRT" { default = "" }
variable "SKIP_XRT" { default = "" }
variable "V80PP_DEB_PACKAGE" { default = "" }

# ---------------------------------------------------------------------------
# Profiles
#
# A profile is one coherent (Ubuntu release, Python version, pin set, XRT
# package) combination. The Ubuntu release is DERIVED from the XRT package name
# rather than set independently, because a 24.04 package on a jammy base is
# something apt will generally resolve rather than refuse -- yielding a subtly
# wrong runtime instead of a clean failure. The Dockerfile re-checks this from
# inside the image.
#
# ubuntu_tag is a date-pinned snapshot, never the rolling tag: Canonical
# republishes `ubuntu:jammy` regularly, which would let image contents drift
# under an unchanged FINN tag. Bump deliberately.
#
# `status` is retained for when a second profile returns. py312 lived here
# until it was removed: it tracked an unfinished upstream PR, never built, and
# cost ~350 lines across six files plus a FINN_PROFILE axis threaded through all
# of them. Restoring it is this block plus a directory under docker/profiles.
#   supported     must build and must pass its declared tests
#   experimental  smoke-built; failures do not gate
# ---------------------------------------------------------------------------

profiles = {
  py310 = {
    status      = "supported"
    description = "Ubuntu 22.04 / Python 3.10 (current, 1970 tests pass)"
    ubuntu_tag  = "jammy-20230126"
    xrt_deb     = "xrt_202220.2.14.354_22.04-amd64-xrt"
    xrt_sha256  = "00f62fbf3e3b4972df96eb0a73191765f9a004c9cf86df9dff70450b32ccdbe0"
  }
}

# ---------------------------------------------------------------------------
# Tag rules
#
# Preserved exactly from run-docker.sh's finn_compute_tag, so this migration
# does not itself trigger a rebuild of every image.
#
#   dev        no --dirty. Under an agent workflow the tree is dirty by
#              definition, so a dirty-tagged dev image would rebuild on the
#              first edit.
#   build      --dirty, profile in the tag.
#   build-xrt  --dirty plus the XRT package, and NO profile marker: the XRT
#              package name already differs per profile (22.04 vs 24.04), and
#              this shape is what the Jenkins publish path keys on.
#   sbx-*      the same lineage plus NOPASSWD sudo. A distinct tag rather than
#              a build argument, so the privilege model is visible in the tag
#              instead of hidden inside identically-named images.
# ---------------------------------------------------------------------------

function "tag" {
  params = [tier, profile]
  result = (
    tier == "dev"           ? "${REGISTRY}:dev-${profile}-${GIT_DESCRIBE}" :
    tier == "build"         ? "${REGISTRY}:build-${profile}-${GIT_DESCRIBE_DIRTY}" :
    tier == "build-xrt"     ? "${REGISTRY}:${GIT_DESCRIBE_DIRTY}.${profiles[profile].xrt_deb}" :
    tier == "sbx-dev"       ? "${REGISTRY}:sbx-dev-${profile}-${GIT_DESCRIBE}" :
    tier == "sbx-build"     ? "${REGISTRY}:sbx-build-${profile}-${GIT_DESCRIBE_DIRTY}" :
    tier == "sbx-build-xrt" ? "${REGISTRY}:sbx-${GIT_DESCRIBE_DIRTY}.${profiles[profile].xrt_deb}" :
    "${REGISTRY}:${tier}-${profile}-${GIT_DESCRIBE_DIRTY}"
  )
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
}

function "common_args" {
  params = [profile]
  result = {
    FINN_PROFILE        = profile
    UBUNTU_TAG          = profiles[profile].ubuntu_tag
    XRT_DEB_VERSION     = profiles[profile].xrt_deb
    XRT_DEB_SHA256      = profiles[profile].xrt_sha256
    SKIP_XRT            = SKIP_XRT
    LOCAL_XRT           = LOCAL_XRT
    V80PP_DEB_PACKAGE   = V80PP_DEB_PACKAGE
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
# These are descriptive, NOT a requirements protocol. There is no OCI
# convention for declaring egress allowlists or mount requirements, and FINN
# should not invent labels that imply a standard exists. What a tier needs is
# resolved at launch by `finn-env inspect`; what these say is what the image IS.
function "common_labels" {
  params = [tier, profile]
  result = {
    "org.opencontainers.image.title"       = "FINN ${tier}"
    "org.opencontainers.image.description" = profiles[profile].description
    "org.opencontainers.image.source"      = "https://github.com/Xilinx/finn"
    "org.opencontainers.image.revision"    = GIT_DESCRIBE_DIRTY
    "dev.finn.tier"                        = tier
    "dev.finn.profile"                     = profile
    "dev.finn.profile-status"              = profiles[profile].status
    # Whether the image grants NOPASSWD root INSIDE the container. This is not
    # host privilege -- no devices, no capabilities, no privileged mode -- but
    # it is a real difference between two otherwise identical images, and the
    # whole reason the sbx variants are separate targets.
    "dev.finn.in-container-root" = can(regex("^sbx-", tier)) ? "true" : "false"
  }
}

# ---------------------------------------------------------------------------
# Targets: tier x profile, generated from one shape.
# ---------------------------------------------------------------------------

target "tiers" {
  inherits = ["_common"]
  name     = replace("${tier}-${profile}", ".", "-")
  matrix = {
    tier    = ["dev", "build", "build-xrt", "sbx-dev", "sbx-build", "sbx-build-xrt"]
    profile = ["py310"]
  }
  target = tier
  args   = common_args(profile)
  labels = common_labels(tier, profile)
  tags   = [tag(tier, profile)]
}

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

# `dev`, not `build-xrt`. The old default was Jenkins history: the largest tier
# with the widest host exposure, for work that mostly does not need it.
group "default" {
  targets = ["dev-py310"]
}

group "supported" {
  targets = ["dev-py310", "build-py310", "build-xrt-py310",
             "sbx-dev-py310", "sbx-build-py310", "sbx-build-xrt-py310"]
}

group "docker" {
  targets = ["dev-py310", "build-py310", "build-xrt-py310"]
}

group "sbx" {
  targets = ["sbx-dev-py310", "sbx-build-py310", "sbx-build-xrt-py310"]
}

group "all" {
  targets = ["supported"]
}
