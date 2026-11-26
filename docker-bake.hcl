# FINN's images: `docker buildx bake [finn|release]`. docker/run builds them when
# they are missing.
#
#   finn     the development image: Ubuntu and FINN's locked dependencies, plus the
#            runtimes in FINN_RUNTIMES ("xrt", "xrt,slash"); FINN is the mounted checkout
#   release  FINN installed from its wheel, on the same base
#
# Tags are local (finn:dev, finn:dev-xrt, finn:release); a registry or CI tags
# them itself (--set '*.tags=...').

variable "FINN_RUNTIMES" { default = "" }
variable "FINN_VERSION" { default = "" }
variable "UBUNTU_TAG" { default = "noble-20240605" }

function "runtime_suffix" {
  params = [runtimes]
  result = join("", [for r in sort(compact(split(",", runtimes))) : "-${r}"])
}

group "default" { targets = ["finn"] }

target "_common" {
  dockerfile = "docker/Dockerfile"
  context    = "."
  platforms  = ["linux/amd64"]
  args       = { UBUNTU_TAG = UBUNTU_TAG, FINN_RUNTIMES = FINN_RUNTIMES, FINN_VERSION = FINN_VERSION }
  labels     = {
    "org.opencontainers.image.title"  = "FINN"
    "org.opencontainers.image.source" = "https://github.com/Xilinx/finn"
  }
}

target "finn" {
  inherits = ["_common"]
  target   = "finn"
  tags     = ["finn:dev${runtime_suffix(FINN_RUNTIMES)}"]
}

target "release" {
  inherits = ["_common"]
  target   = "release"
  tags     = ["finn:release${runtime_suffix(FINN_RUNTIMES)}"]
}
