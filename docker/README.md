# FINN environments

FINN uses uv 0.10.0, Python 3.12 and one committed `uv.lock`. Native development,
Docker, documentation and CI install from the dependency groups in `pyproject.toml`.
The development image contains the full locked environment, including notebook,
export, reporting and benchmark packages.

`docker/build` and `docker/run` are thin wrappers around `docker/launcher.py`, a
standard-library Python program (Python 3.7 or later on the host) that owns every
host decision: tools, licences, mounts, image identity and image transport.
Building needs Docker Buildx with Bake (`docker buildx bake --help`; install
`docker-buildx-plugin` if it is missing). Running a published, verified image
does not.

## Start an environment

```bash
./docker/build                          # prepare the development image
./docker/run                            # shell; Xilinx tools mounted if configured
./docker/run --fpga -- python -m pytest -m util
./docker/run --fpga --runtime xrt -- build_dataflow /path/in/the/checkout
./docker/run --notebook                 # Jupyter token URL, ports 8888 and 8081
```

The Xilinx tools are mounted whenever they are configured (see below).
`--fpga` makes them a requirement: the launcher fails before building or starting
anything if no Vivado, Vitis or HLS installation is found. `--no-tools` never
mounts them. The image is the same either way.

`docker/run` builds a missing image; `--no-build` requires an existing one and
`--rebuild` refreshes all layers. `FINN_UV_NO_CACHE=1` skips the intermediate uv
build cache on storage-constrained builders. Commands, mounts and environment
values keep their argument boundaries, including spaces:

```bash
model_dir="/absolute/path/model with spaces"
./docker/run --fpga --volume "$model_dir:$model_dir" --workdir "$model_dir" -- python build.py
```

`--env NAME[=VALUE]`, `--volume SOURCE:TARGET[:MODE]`, `--workdir DIR` and
`--name NAME` add Docker options. Site Docker flags in `FINN_DOCKER_EXTRA` are
shell-tokenized without executing substitutions. Terminals are attached when
present, so `docker/run -- ipython` is interactive while CI runs are not.

Containers run as the caller's UID/GID, which own all outputs. Private NSS files
(through `libnss_wrapper`) give that user a passwd/group identity, including for
OpenSSH and direct `docker exec`. Site overrides of `LD_PRELOAD` must keep the
image's `libnss_wrapper` and licence workaround libraries. The container's
`HOME` is throwaway; persistent state lives in `FINN_HOME`. Startup installs
only FINN itself, editable and without dependencies, after checking that the
checkout's `uv.lock` matches the image. A failed startup step stops the command.

### Native use

Install the system prerequisites (`sudo ./scripts/install-system-deps.sh`), then:

```bash
source /tools/Xilinx/2025.2/Vivado/settings64.sh   # or the site's Vivado module
./setup-local.sh
source scripts/activate.sh
./scripts/quicktest-local.sh
```

Setup bootstraps the uv version in `.uv-version` into `~/.cache/finn/uv-tool/`
if needed; it never installs compiler packages with pip. It builds XSI when the
shell has Vivado (`--skip-xsi` skips it). `FINN_VENV` and `FINN_PYTHON` select the
environment (default `.venv` in the checkout) and the Python 3.12 interpreter.
Activation installs nothing: it activates the environment and sets `FINN_ROOT`.

FINN reads the AMD tools' own environment, as a release's `settings64.sh` or a
site's module sets it: `XILINX_VIVADO`, `XILINX_HLS`, `XILINX_VITIS` and the tools
on `PATH`. It builds none of it, so switching releases is sourcing another
`settings64.sh` in a new shell, and one FINN process uses one release. FINN refuses
an environment whose variables or tools come from different releases. With no AMD
tools set up, a step that needs them names the newest release installed under
`/tools/Xilinx`, `/opt/Xilinx` or `~/Xilinx` and the `settings64.sh` to source; it
never uses it by itself. `VIVADO_PATH`, `VITIS_PATH` and `HLS_PATH`, which only
`run-docker.sh` set, are still read in place of the AMD names, with a warning.

## Host tools and licences

A container has no shell to source `settings64.sh` in, so the launcher decides
what to mount and passes the AMD variables in. It takes, in this order:

1. `FINN_XILINX_PATH` and `FINN_XILINX_VERSION` in the environment (Jenkins sets
   these);
2. the AMD variables of the calling shell (a sourced `settings64.sh` or a module);
3. the host file `~/.config/finn/xilinx.env`.

The host file is the machine description for launchers without a login shell:
`docker/run`, Dev Containers, CI agents and sandboxes (an sbx kit reads it as its
argument file). FINN itself never reads it. Environment variables override its
values; `FINN_XILINX_ENV` selects another file, and an empty value disables it.
Write plain `NAME=value` lines, without expansion or quotes:

```text
FINN_XILINX_PATH=/opt/Xilinx
FINN_XILINX_VERSION=2025.2
XILINXD_LICENSE_FILE=2100@licence-server
```

Both the `$ROOT/Vivado/$VERSION` and `$ROOT/$VERSION/Vivado` layouts are detected.
Tools and licence files are mounted read-only. FLEXlm file lists and servers
(`PORT@HOST`) are supported, as is `LM_LICENSE_FILE`; alternatively set
`FINN_LICENSE_HOST` and `FINN_LICENSE_PORT`. `FINN_LICENSE_VENDOR_PORT` records a
pinned vendor-daemon port. Network requirements reported by `inspect` are
declarations, not firewall rules.

`PLATFORM_REPO_PATHS` takes colon-separated directories and defaults to
`/opt/xilinx/platforms` where that exists. Tool directories outside both layouts
are given as the AMD variables themselves (`XILINX_VIVADO`, `XILINX_HLS`,
`XILINX_VITIS`); `VIVADO_PATH`, `VITIS_PATH` and `HLS_PATH` are deprecated
aliases. `FINN_TOOL_DIR_OVERRIDE` keeps site tool
interception, including LSF routing, through `resolve_xilinx_tool()`; wrapper
directories are not prepended to `PATH` for bare commands. A host directory is
mounted read-only; a path that exists only in the container, such as a wrapper
kit mounted through `FINN_DOCKER_EXTRA`, is passed on with a warning. Keep the build
directory and `FINN_HOME` on storage the farm sees at the same absolute path,
since generated Tcl embeds them. `.Xilinx/` Tcl files in the checkout are copied
into the container's home.

Inspect the resolved host request, or render a Compose override from it:

```bash
./docker/launcher.py inspect
FINN_IMAGE=$(./docker/build --print-tag) ./docker/launcher.py compose --fpga
```

Both take `--fpga` and `--no-tools` as `docker/run` does; without either, the tools
are mounted where they are configured. The checkout is mirrored at its host path;
`--workspace-policy fixed` mounts it at `/workspace/finn` instead, for a remote
Docker daemon, and only with `--no-tools`.

`compose --inputs-only` renders only tool, licence and resource inputs for a
Dev Container that owns its workspace and state.

## Where FINN keeps files

| Variable | Default | Holds |
| --- | --- | --- |
| `FINN_ROOT` | the checkout | the checkout, for notebooks, tutorials and the launcher; FINN itself does not read it |
| `FINN_HOME` | `~/.finn` | persistent, content-keyed state: everything below without its own variable |
| `FINN_RESOURCES_DIR` | `$FINN_HOME/resources` | fetched resources and assembled board files |
| `FINN_XSI_PATH` | `$FINN_HOME/xsi` | XSI builds, one per source, compiler, Python ABI and Vivado installation |
| `FINN_GOLDEN_DIR` | `$FINN_HOME/golden/<board>-<key>` | golden references; when set, a site directory holding `<board>/`, used as supplied |
| (none) | `$FINN_HOME/identity/<uid>` | the container user's passwd/group files |
| `FINN_BUILD_DIR` | `/tmp/finn_build_<uid>` | disposable build output; nothing in it is reused |
| `FINN_HOST_BUILD_DIR` | `FINN_BUILD_DIR` | the host build directory `docker/run` mirrors into containers |
| `FINN_RESOURCES_SYSTEM_CACHE` | `/opt/finn/resources` | the image's read-only resource cache |
| `FINN_RESOURCES_<NAME>` | unset | a local directory replacing one resource, unverified |
| `FINN_BOARD_FILES_PATH` | assembled in `FINN_RESOURCES_DIR` | an already assembled site board tree |
| `FINN_DOCKER_CACHE_DIR` | unset | Torch, Hugging Face and uv download caches for containers |
| `VIVADO_IP_CACHE` | `/tmp/vivado_ip_cache` in the image | Vivado's IP cache |

Deleting `FINN_HOME` costs only rebuilds and downloads; deleting the build
directory loses only build output. `docker/run` creates `FINN_HOME`,
`FINN_RESOURCES_DIR`, `FINN_XSI_PATH` and the build directory on the host and
mounts each at its own path. `finn-resources clean` removes cached resources,
XSI builds and golden references; `--unused` keeps those the current pins,
checkout and Vivado installation use. A site `FINN_GOLDEN_DIR` is never cleaned.

FINN reads its own trees (`finn-rtllib/`, `custom_hls/`, `finn_xsi/` and
`src/finn/qnn-data/`) as resources: in place in a checkout, from the package in
an installed wheel (see below). Nothing generated is written there by FINN
itself. The exceptions are explicit: the native `.venv`,
the editable install's metadata, the finn-experimental adapter's build
directory, staged supplied packages (`docker/packages/`, `docker/runtime-cache/`)
in the Docker build context, and Jenkins' `image-metadata/`.

## Dependencies

Use `uv add`, `uv lock --upgrade-package NAME` and `uv sync --locked` for Python
updates. QONNX and Brevitas are pinned to upstream's Git selections. The groups
`test`, `ci`, `notebooks`, `docs` and `dev` resolve together.

PyTorch is the CPU build by default (the `cpu` group). FINN does not use CUDA;
for a GPU environment, `uv sync --locked --no-group cpu --group cuda` selects the
CUDA 12.6 build from the same lock.

For native co-development of a dependency, use uv's local source support
(`uv add --editable /absolute/checkout`), and restore the committed selection
before submitting unrelated work. In containers, mount the checkout and use a
per-command overlay instead:

```bash
./docker/run --volume /absolute/qonnx:/absolute/qonnx -- uv run --locked --with-editable /absolute/qonnx python your_test.py
```

Like the Brevitas candidate job, an overlay resolves separately unless constrained
with a generated `uv export`.

The pinned finn-experimental repository declares `packages=find:` despite a
namespace-only source root, so its wheel contains no Python modules. The
`packaging/finn-experimental` adapter builds its verified, unchanged sources as a
namespace wheel; it can go when upstream fixes the package. Sphinx 5.0.2 needs
docutils below 0.19, so the lock selects 0.18.1.

## External resources

Every directory tree FINN uses that is not Python code is a resource declared in
`src/finn/resources.toml`, and FINN reaches it by name or kind through
`finn.resources`, never by a path into a checkout:

- FINN's own trees, used in place: `rtllib` (`finn-rtllib/`), `finn-xsi`
  (`finn_xsi/`), `custom-hls` (`custom_hls/`) and `qnn-data`
  (`src/finn/qnn-data/`). A checkout declares the first three as `path`
  sources; the wheel build copies them into `finn/_sources/` and declares them
  as `package` sources there, so an installed FINN needs no checkout.
- Fetched trees, each at a Git commit (or an archive URL and its sha256) with a
  content digest: finn-hlslib, the five upstream Vivado board repositories, the
  PYNQ golden sources and finn-experimental.

`finn-resources` lists, checks, fetches, verifies and updates them:

```bash
finn-resources list
finn-resources check
finn-resources fetch --kind hls-include --kind vivado-boards
finn-resources verify
finn-resources update hlslib --ref FULL_COMMIT_OR_REF
finn-resources clean --unused
```

Caches are content-addressed and published atomically under file locks.
`FINN_RESOURCES_<NAME>` (uppercase, `-` as `_`, e.g. `FINN_RESOURCES_HLSLIB`)
replaces a resource with a local directory, unverified.
`FINN_RESOURCES_<NAME>_URL` fetches it from another repository, such as a site
mirror, verified against the same digest. `FINN_RESOURCES_OFFLINE=1` forbids
downloads, including explicit fetches; to prepare an offline machine, fetch into
a directory with `--dest PATH` elsewhere and select it with `FINN_RESOURCES_DIR`.

### Your own resources

Declare resources in the same format at three levels:

- a project: `[tool.finn.resources.NAME]` in the nearest `pyproject.toml`, or
  `[resources.NAME]` in files listed in `FINN_RESOURCES_FILES`. A project may
  add resources and redefine FINN's (another finn-hlslib pin, a fork of
  finn-rtllib);
- an installed package: an entry point in the `finn.resources` group naming a
  module that contains a `resources.toml`. A package may only add resources, for
  example the RTL of its custom operators as a `package` source;
- a session: `FINN_RESOURCES_<NAME>=/dir` replaces any resource.

```toml
[tool.finn.resources.acme-hls]
kind = ["hls-include"]
path = "hls/include"          # relative to the declaring file

[tool.finn.resources.acme-boards]
kind = ["vivado-boards"]
git = "https://example.com/acme/boards.git"
commit = "FULL_COMMIT"
digest = "sha256:..."         # `finn-resources digest DIR` of that commit's tree
```

Every `hls-include` resource is an include directory for HLS C++ simulation and
IP generation, and every `vivado-boards` resource joins the assembled board
repository, so resources of these kinds take effect without code. Custom
operators reach any other resource with `finn.resources.path(NAME)` or
`finn.resources.join(NAME, ...)`.

The board repositories are assembled into the upstream flat layout in the
resource cache. Locally built images contain the board sources, which may not
be redistributed; the release target omits them.

XSI builds are keyed by their sources, Python ABI, compiler, compile and link
settings, pybind11 version and Vivado installation. `FINN_VCK190_GOLDEN=1` builds
the VCK190 golden reference from the pinned PYNQ sources with the selected
Vivado; the pinned Tcl declares Vivado 2025.2.

## Runtime packages and read-only images

Runtime content is independent of the tools. `--runtime xrt` installs upstream's
XRT 2.18.179 userspace; `runtimes/*.env` pins downloads and names supplied
SLASH/slashkit packages, which must be present in `docker/packages/`. Kernel
modules and host daemons remain host responsibilities.

```bash
./docker/build --release --runtime xrt
./docker/build --export-sif /absolute/finn.sif --runtime xrt
```

The release image holds FINN as `pip install` gives it: the build compiles a
wheel of the checkout and installs that, with no source tree and no `FINN_ROOT`,
so it needs no startup installation. The development image instead installs the
mounted checkout in editable mode at start. SIF export needs Apptainer or
Singularity. Run the SIF
with `apptainer run` (which runs the entrypoint), binding tool, licence and
scratch directories, and set `FINN_BUILD_DIR` and `FINN_HOME` (default
`~/.finn`) to writable host storage. Supply board caches for offline use.

## Image identity and CI transport

An image's tag is `img-<hash>` of the files in `docker/image-inputs.txt`, the
supplied packages of the selected runtimes and the Ubuntu base, plus a
`.<runtime>` suffix per runtime. FINN's own sources are not inputs: containers
install the mounted checkout, so source edits reuse the image. Modes are hashed
as Git records them, so differing Jenkins umasks do not change identity.

Jenkins builds one image, publishes it to shared storage with
`ci/scripts/publish_docker_image.sh` (gzip by `pigz` on all cores where
installed) and every shard loads it with `FINN_DOCKER_SHARED_IMAGE_DIR` and
`FINN_DOCKER_PREBUILT=1`. A publication records the image ID, environment
revision, lock and resource checksums and the archive checksum; loads verify
them, run by immutable ID, and serialize per host (`FINN_DOCKER_LOAD_LOCK`). A
broken publication never falls back to a local build when `FINN_DOCKER_PREBUILT=1`.
See [CI integration](../ci/README.md).

## Interface migration

`run-docker.sh` remains only as a stub that prints the equivalent new command
and fails. The launcher warns about retired variables it sees.

| Previous input | Replacement |
| --- | --- |
| `run-docker.sh` | `docker/run [--fpga] [--runtime NAME] -- COMMAND...` |
| `run-docker.sh print-tag` | `docker/build --print-tag` |
| `run-docker.sh test`, `quicktest` | `docker/run --fpga -- python -m pytest`, `docker/run -- bash docker/quicktest.sh [MODE]` |
| `run-docker.sh build_dataflow DIR` | `docker/run --fpga --volume DIR:DIR -- build_dataflow DIR` |
| `run-docker.sh build_custom DIR FLOW` | `docker/run --fpga --volume DIR:DIR --workdir DIR -- python FLOW.py` |
| `run-docker.sh notebook` | `docker/run --notebook` |
| `scripts/finn-env.sh` | `scripts/activate.sh`, with the AMD tools from a sourced `settings64.sh` |
| `VIVADO_PATH`, `VITIS_PATH`, `HLS_PATH` | `XILINX_VIVADO`, `XILINX_VITIS`, `XILINX_HLS` (the old names are read, with a warning) |
| `fetch-repos.sh`, Python clones in `deps/` | `uv.lock` and verified resources |
| `deps/pynq_golden` | `$FINN_HOME/golden`, or `FINN_GOLDEN_DIR` |
| `requirements.txt`, `docs/requirements.txt` | dependency groups and `uv.lock`; `uv export` if needed |
| `FINN_DOCKER_TAG` | input-derived tags; `FINN_IMAGE` selects a prepared image |
| `FINN_DOCKER_NO_CACHE` | `docker/build --rebuild` |
| `FINN_DOCKER_BUILD_EXTRA` | Bake options, e.g. `--set *.args.UBUNTU_TAG=...` (also set `UBUNTU_TAG`) |
| `FINN_SKIP_XRT_DOWNLOAD`, `XRT_DEB_VERSION` | `--runtime xrt` with the pinned manifest; `SOURCE=supply` for site packages |
| `FINN_SKIP_DEP_REPOS`, `FINN_SKIP_BOARD_FILES` | a prepared resource cache and `FINN_RESOURCES_OFFLINE=1` |
| `DOCKER_INTERACTIVE` | terminal detection; Docker flags in `FINN_DOCKER_EXTRA` |
| `FINN_INST_NAME` | `docker/run --name NAME` |
| `FINN_SINGULARITY` | `docker/build --export-sif PATH`, then Apptainer/Singularity |
| `FINN_DOCKER_PREBUILT`, `FINN_DOCKER_SHARED_IMAGE_DIR`, `FINN_DOCKER_CACHE_DIR`, `FINN_DOCKER_RUN_AS_ROOT`, `FINN_SSH_KEY_DIR` (mounted read-only; default `ssh_keys/` in the checkout when it exists, no longer created), `VERIFICATION_*`, `FINN_XRT_PATH` | retained |
| `SLASHKIT_DEB_PACKAGE` | retained: staged into `docker/packages/` and selects the `slashkit` runtime |

If `SLASHKIT_DEB_PACKAGE` is set, its file must be readable wherever the launcher
runs, including where prebuilt images are loaded, so identical bytes are hashed.
Old two-file image publications must be rebuilt.
