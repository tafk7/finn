# FINN Docker-built environment

FINN has two development setups: native (`setup-local.sh`), or the Docker-built
environment described here. Both use the same `uv.lock`; see
[installation](../docs/installation.md) and [the design](../docs/environment.md).

The image is always built with Docker Buildx. Docker runs it directly,
Apptainer consumes an exported SIF, and Docker Sandboxes builds its `sbx` stage
as FINN's workload kit:

```text
docker/Dockerfile.finn
├── Docker Compose              default
├── SIF export                  standard Apptainer/Singularity execution
└── sbx workload kit (finn.yaml) agent isolation; built by sbx from the checkout
```

## Run

```bash
./docker/run --name finn-test -- bash scripts/check-kernels.sh
./docker/run --fpga -- vivado -version
./docker/run --fpga --runtime xrt -- build_dataflow project/
```

With no command, `docker/run` opens an interactive shell. `/opt/venv` is active,
with FINN installed editable from the checkout when the container starts. The
image's system cache (`/opt/finn/resources`) holds finn-hlslib and the board files;
`finn-resources list` shows them. Use `--print` to see the normalized request without building or
running anything. `-n NAME` and `--name NAME` assign the Docker container name;
`--volume` adds a mount, for example a co-developed QONNX checkout.

## Prepare artifacts

```bash
./docker/build
./docker/build --runtime xrt
./docker/build --release
./docker/build --export-sif ./finn.sif
./docker/build --runtime xrt --export-sif ./finn-xrt.sif
```

Docker preparation is automatic when a run needs it. `--no-build`
requires an existing Docker image, while `--rebuild` rebuilds
without using the BuildKit cache. Ordinary runs and SIF export reuse a prepared
Docker image with the selected environment tag; explicit `docker/build` refreshes
its cached build. SIF export always writes the requested path.

No registry is assumed. Docker images remain in the local daemon and SIF files
are explicit build outputs. Run a SIF with the standard tool, for example:

```bash
apptainer exec --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" \
  ./finn.sif python -c 'from finn.util.resources import resource_path; print(resource_path("xsi"))'
```

Image tags are ``img-<hash>`` of the files in ``docker/image-inputs.txt``, which
include ``pyproject.toml`` and ``uv.lock`` but not FINN's sources: a change to the
lock produces a new image, an edit to FINN does not, nor does one to the tool
configuration (``.ruff.toml``, ``.mypy.ini``, ``.pytest.ini``), which is kept out
of ``pyproject.toml`` for that reason. The release image (and the
SIF exported from it) installs FINN from wheels and is tagged by source revision.
The mounted checkout's commit, description and dirty state are passed separately
as ``FINN_SOURCE_*`` runtime provenance. The immutable identity of a concrete build
remains its Docker image digest or exported SIF checksum.

## Configuration

The Xilinx installation and licence server are this machine's, in
`~/.config/finn/xilinx.env` ([configure a machine](../docs/installation.md#configure-a-machine));
`FINN_XILINX_VERSION=2026.1 ./docker/run --fpga …` selects another installed
version for one container. `FINN_RESOURCES_*` directories are mounted at their
own paths.

`docker/config.py` is the single executable Python host resolver for Docker and
native installation. It reads the machine file (through FINN's reader,
`src/finn/util/machine_file.py`, which `xilinx_install.py` loads and the sbx
workload shares) and preserves path/layout probing, licence
classification, UID/GID handling, shell output and Compose output:

```bash
./docker/config.py inspect --tier dev
./docker/config.py inspect --tier build
./docker/config.py compose --tier build --service build
./docker/config.py compose --tier auto --inputs-only --service dev   # the Dev Container's inputs
```

Reported network requirements are declarative. Docker does not enforce them.

## Docker Sandboxes

FINN ships a workload kit (`finn.yaml` at the repository root, building the
`sbx` stage from this checkout) and a mixin for a mounted Xilinx installation
and licence server (`docker/sbx/xilinx/`). Neither contains a coding agent,
machine paths or credentials: those are kits and arguments of whoever creates
the sandbox. See [the sbx guide](sbx/README.md). The sbx Bake targets build the
same stage as a Docker image, for inspection and tests.

`compose.yaml` and `docker-bake.hcl` remain usable directly for debugging and
advanced workflows. A direct Bake invocation must set
`FINN_IMAGE_REVISION` explicitly; the supported `docker/build` path computes it
from `docker/image-inputs.txt` automatically.

## Implementation boundaries

```text
docker/build + Docker/SIF consumers
    -> docker/lib.sh: finn_prepare_image
        -> docker-bake.hcl -> docker/Dockerfile.finn

docker/config.py                            host only
    -> shell assignments / Compose overrides

finn.yaml + docker/sbx/xilinx + the user's kits and mounts
    -> sbx builds and composes the kits; sbx owns the sandbox lifecycle

image                                      guest only
    -> Dockerfile tool list + toolchain-shim
    -> finn_entrypoint.sh / finn-bashenv.sh / finn-toolchain.sh
    -> installed FINN distribution and generated console entry points
```

Host rendering code and sandbox defaults are not image inputs. Editing them
does not change image identity. The Dockerfile declares the tool list and sbx
contract inline. The image does not install the host resolver.

Repository callers use the public interface; the old runtime scripts have been
removed. Use `--fpga` instead of legacy grant-tier selection,
`--runtime xrt` to select XRT, and `docker/build --print-tag` for image references.
Commands after `--` are passed verbatim: use `pytest` or a gate script explicitly.
For external dataflow directories, supply an explicit mount through
`FINN_DOCKER_EXTRA`; the launcher no longer interprets workload command names.

Python dependencies come from `pyproject.toml` and `uv.lock`, the same lock native
development uses. The image installs them into `/opt/venv` at build time.

The guest entrypoint provides writable home/scratch directories, installs the
mounted checkout into `/opt/venv` (never fatal; `FINN_SYNC=0` skips it) and writes
`/tmp/finn-ready`. Bash and bare vendor commands retain their respective
hooks because those paths bypass normal startup.

## Command migration

| Removed interface | Replacement |
| --- | --- |
| `docker/run-docker` | `docker/run -- COMMAND` |
| `docker/run-sbx`, `docker/finn-sbx`, `docker/run --sbx` | `sbx create` with FINN's kits ([sbx guide](sbx/README.md)) |
| `docker/build --sbx`, `sbxenv.yaml`, `docker/sbx/*.sbxenv.yaml` | FINN's workload kit `finn.yaml` and the `docker/sbx/xilinx` kit |
| `run-docker.sh` | Explicit CI image preparation, then `docker/run` |
| `docker/export-sif PATH` | `docker/build --export-sif PATH` |
| `test` / `quicktest` launch shortcuts | Explicit `pytest` or gate (`scripts/check-*.sh`) commands |
| `build_custom` shortcut | Explicit directory mount, working directory, and Python command |
| `build-xrt` grant/image spelling | `--fpga --runtime xrt` |
| Bake target `finn-xrt-slash` | `finn-slash-xrt` |
| `--dependencies`, `--venv`, `--deps`, `FINN_DEPS` | The dev image installs the mounted checkout at start |
| `fetch-repos.sh`, `deps.env` | `pyproject.toml`/`uv.lock`; finn-hlslib and board files as external resources (`finn-resources`) |
| `FINN_BOARD_FILES_PATH` | `FINN_RESOURCES_<NAME>` per board resource |
| `FINN_HLSLIB_PATH` | `FINN_RESOURCES_HLSLIB` |

`docker/config` and `docker/finn-env` are removed; use `docker/config.py`.
The resolver's `sbx` subcommand and `inspect --sbx`, and the launcher's `--sbx`
and `--remove`, now fail as unknown interfaces.

## Development environment

The dev image runs the mounted checkout, not an installed FINN. At container start
the entrypoint runs `uv sync --frozen --inexact` against `FINN_ROOT`: FINN is
installed editable, plus any difference between the checkout's `uv.lock` and the
image, offline where possible. uv's cache
is kept in `$FINN_BUILD_DIR/.uv-cache`, so later starts reuse the editable builds.
`docker exec` skips the entrypoint but joins a container where it has run;
scripts that exec into a new container can wait for `/tmp/finn-ready`. In a
sandbox the same script runs as the workload's startup hook.

The Dev Container uses the same image and entrypoint at the fixed path
`/workspace/finn`; its interpreter is `/opt/venv/bin/python`.

Generic images have no Bash activation hook. Only the sbx variant sources native
persistent environment configuration and composes `XILINXD_LICENSE_FILE` from
the xilinx kit's licence host and port. Site Tcl initialization uses explicit mounts
into the chosen user's `.Xilinx` directory. Bare vendor shims and global libudev
preload remain pending actual licensed/native validation; FINN vendor operations
use scoped argv/cwd/environment execution.
