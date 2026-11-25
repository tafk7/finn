# FINN environments

FINN is a Python package that reads AMD's environment. Natively, a venv holds FINN
and its locked dependencies, and the shell that starts it has sourced a release's
`settings64.sh`. The container is the same thing on a known Ubuntu: an image of
FINN's locked dependencies, started like a shell that sourced `settings64.sh`. It
serves deployment, a reproducible distro, agent sandboxes and CI; it is not the only
way to run FINN.

FINN uses uv 0.10.0, Python 3.12 and one committed `uv.lock`; native development,
the image, documentation and CI install from the dependency groups in
`pyproject.toml`.

## Native use

Install the system prerequisites (`sudo ./scripts/install-system-deps.sh`), then:

```bash
source /tools/Xilinx/2025.2/Vivado/settings64.sh   # or the site's Vivado module
./setup-local.sh
source scripts/activate.sh
finn doctor                  # what FINN finds, and what it can run here
./scripts/quicktest-local.sh
```

Setup bootstraps the uv version in `.uv-version` into `~/.cache/finn/uv-tool/`
if needed; it never installs compiler packages with pip. Resources are fetched,
and the XSI binding is built for the sourced Vivado, the first time FINN needs
them; `--prefetch` fetches every declared resource ahead of use offline
(`finn resources fetch --all` does the same for a pip install), and
`python -m finn.xsi.setup` prebuilds XSI. `FINN_VENV` and `FINN_PYTHON` select the
environment (default `.venv` in the checkout) and the Python 3.12 interpreter.
Activation installs nothing: it activates the environment and sets `FINN_ROOT`.

FINN reads the AMD tools' own environment, as a release's `settings64.sh` or a
site's module sets it: `XILINX_VIVADO`, `XILINX_HLS`, `XILINX_VITIS` and the tools
on `PATH`. It builds none of it, so switching releases is sourcing another
`settings64.sh` in a new shell, and one FINN process uses one release. FINN refuses
an environment whose variables or tools come from different releases. With no AMD
tools set up, a step that needs them names the newest release installed under
`/tools/Xilinx`, `/opt/Xilinx` or `~/Xilinx` and the `settings64.sh` to source; it
never uses it by itself.

`finn doctor` reports, and changes nothing: the release in the environment and
whether its tools agree, other installed releases, the compiler, Python headers,
licence settings, the XSI binding, each resource's source, and what FINN can run
with what is set up. It also prints what FINN knows about the release in use that
the installation does not show (`finn.util.releases`: a release's own behaviour and
what FINN does about it). It exits 1 for a broken environment, such as tools of two
releases, and 0 without tools; `--json` is for CI.

## The container

```bash
source /tools/Xilinx/2025.2/Vivado/settings64.sh   # optional: AMD tools, as natively
export XILINXD_LICENSE_FILE=2100@licence-server
./docker/run                                       # a shell
./docker/run --fpga -- python -m pytest -m util
./docker/run --fpga --runtime xrt -- build_dataflow /path/in/the/checkout
./docker/run --notebook                            # Jupyter, ports 8888 and 8081
```

The container is defined in `compose.yaml` (the checkout, the working directory,
`FINN_HOME` and the build directory, mounted at their host paths; the caller's uid)
and `compose.tools.yaml` (the AMD install root of `XILINX_VIVADO`, read-only, and
AMD's own variables). `docker/run` derives their variables from the host and runs
`docker compose run`; `--dry-run` prints them and the command. Dev Containers and
agent sandboxes use the same files with the variables set in their own configuration.

AMD tools follow the shell, as natively: with `XILINX_VIVADO` set (a sourced
`settings64.sh`, a site module, a CI job), the install root is mounted and the
container sources that release's `settings64.sh` at start, in login shells and,
through `BASH_ENV`, in every bash, so `docker exec` sessions have the tools too.
`--fpga` requires the tools, `--no-tools` leaves them out. Licences and platforms
use AMD's names: `XILINXD_LICENSE_FILE` and `LM_LICENSE_FILE` (licence files among
them are mounted read-only; `port@host` servers need nothing) and
`PLATFORM_REPO_PATHS`. `FINN_TOOL_DIR_OVERRIDE`, a directory of site wrappers (an
LSF dispatcher), is mounted and passed on; keep the build directory and `FINN_HOME`
on storage the farm sees at the same paths, since generated Tcl embeds them.

The container runs as the caller, with a passwd/group identity from private
`libnss_wrapper` files, including for OpenSSH and `docker exec`. Its `HOME` is
throwaway; persistent state lives in `FINN_HOME`. The development image installs
the mounted checkout at start, editable and without dependencies, after checking
that its `uv.lock` matches the image's; the release image has FINN installed.
`ssh_keys/` in the checkout, when present, becomes the container's `~/.ssh`.

`-v SOURCE:TARGET[:MODE]` and `-e NAME[=VALUE]` add mounts and variables; further
Compose files (a site's or CI's) go in `COMPOSE_FILE` and apply after FINN's.

### Images

```bash
docker buildx bake                       # finn:dev
FINN_RUNTIMES=xrt docker buildx bake     # finn:dev-xrt
docker buildx bake release               # finn:release: FINN from its wheel
```

`docker/run` builds the image it needs when it is missing (`--build` rebuilds,
`--no-build` refuses; `FINN_IMAGE` names another). The image's stages
(`docker/Dockerfile`) are `base` (Ubuntu, the locked dependencies, finn-hlslib in
a read-only resource cache, the entrypoint), `finn` (plus the runtimes in
`FINN_RUNTIMES`) and `release` (plus FINN's wheel, no source tree). Building needs
Docker Buildx with Bake (`docker buildx bake --help`; install
`docker-buildx-plugin` if it is missing). For Apptainer or Singularity, build a
SIF from the release image: `apptainer build finn.sif docker-daemon://finn:release`.

Runtimes are independent of the AMD tools. `xrt` installs upstream's XRT 2.18.179
userspace from the pinned, checksummed URL in `docker/runtimes/xrt.env`, kept in
BuildKit's cache between builds; `slash` and `slashkit` name packages you supply in
`docker/packages/`. Kernel modules and host daemons remain host responsibilities.

## Where FINN keeps files

| Variable | Default | Holds |
| --- | --- | --- |
| `FINN_HOME` | `~/.finn` | persistent, content-keyed state: everything below without its own variable |
| `FINN_RESOURCES_DIR` | `$FINN_HOME/resources` | fetched resources and assembled board files |
| `FINN_XSI_PATH` | `$FINN_HOME/xsi` | XSI builds, one per source, compiler, Python ABI and Vivado installation |
| `FINN_GOLDEN_DIR` | `$FINN_HOME/golden/<board>-<key>` | golden references; when set, a site directory holding `<board>/`, used as supplied |
| (none) | `$FINN_HOME/identity/<uid>` | the container user's passwd/group files |
| `FINN_BUILD_DIR` | `/tmp/finn_build_<uid>` | disposable build output; nothing in it is reused |
| `FINN_RESOURCES_SYSTEM_CACHE` | `/opt/finn/resources` | the image's read-only resource cache |
| `FINN_RESOURCES_<NAME>` | unset | a local directory replacing one resource, unverified |
| `FINN_BOARD_FILES_PATH` | assembled in `FINN_RESOURCES_DIR` | an already assembled site board tree |

Deleting `FINN_HOME` costs only rebuilds and downloads; deleting the build
directory loses only build output. `finn resources clean` removes cached
resources, XSI builds and golden references; `--unused` keeps those the current
pins, installation and Vivado use. A site `FINN_GOLDEN_DIR` is never cleaned.
FINN writes nothing into a checkout but the native `.venv`, the editable install's
metadata and the finn-experimental adapter's build directory.

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

- FINN's own trees: `rtllib` (`finn-rtllib/`), `finn-xsi` (`finn_xsi/`),
  `custom-hls` (`custom_hls/`) and `qnn-data` (`src/finn/qnn-data/`). They are
  package data under `finn/_sources`: in a checkout symbolic links to the trees,
  in a wheel the files themselves, so an installed FINN needs no checkout.
- Fetched trees, each at a Git commit with a content digest: finn-hlslib, the
  five upstream Vivado board repositories, the PYNQ golden sources and
  finn-experimental.

```bash
finn resources list
finn resources fetch --all              # ahead of offline use
finn resources path hlslib
finn resources clean --unused
```

Caches are content-addressed and published atomically under file locks.
`FINN_RESOURCES_<NAME>` (uppercase, `-` as `_`, e.g. `FINN_RESOURCES_HLSLIB`)
replaces a resource with a local directory, unverified. `FINN_RESOURCES_OFFLINE=1`
forbids downloads; to prepare an offline machine, fetch into a directory with
`--dest PATH` elsewhere and select it with `FINN_RESOURCES_DIR`. Maintainers move
a pin with `scripts/update-resource.py NAME --ref REF`.

### Your own resources

An installed package adds resources through an entry point in the
`finn.resources` group, naming a module that contains a `resources.toml`, for
example the RTL of its custom operators:

```toml
# acme_finn/resources.toml; pyproject.toml: [project.entry-points."finn.resources"] acme = "acme_finn"
[resources.acme-hls]
kind = ["hls-include"]
package = "acme_finn"
subdir = "hls/include"

[resources.acme-boards]
kind = ["vivado-boards"]
git = "https://example.com/acme/boards.git"
commit = "FULL_COMMIT"
digest = "sha256:..."         # scripts/update-resource.py --digest DIR of that tree
```

A package may only add resources; `FINN_RESOURCES_<NAME>=/dir` replaces any
resource for a session (another finn-hlslib, a fork of finn-rtllib).

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

## Migrating from run-docker.sh

| Before | Now |
| --- | --- |
| `run-docker.sh` | `docker/run [--fpga] [--runtime NAME] -- COMMAND...` |
| `run-docker.sh test`, `quicktest` | `docker/run --fpga -- python -m pytest`, `docker/run -- bash docker/quicktest.sh [MODE]` |
| `run-docker.sh build_dataflow DIR` | `docker/run --fpga -v DIR:DIR -- build_dataflow DIR` |
| `run-docker.sh notebook` | `docker/run --notebook` |
| `scripts/finn-env.sh` | `scripts/activate.sh`, with the AMD tools from a sourced `settings64.sh` |
| `FINN_XILINX_PATH`, `FINN_XILINX_VERSION` | `XILINX_VIVADO`, from a sourced `settings64.sh` or set to `<root>/<release>/Vivado` |
| `VIVADO_PATH`, `VITIS_PATH`, `HLS_PATH` | `XILINX_VIVADO`, `XILINX_VITIS`, `XILINX_HLS` |
| `fetch-repos.sh`, Python clones in `deps/` | `uv.lock` and verified resources |
| `deps/pynq_golden` | `$FINN_HOME/golden`, or `FINN_GOLDEN_DIR` |
| `requirements.txt`, `docs/requirements.txt` | dependency groups and `uv.lock`; `uv export` if needed |
| `FINN_DOCKER_TAG`, `FINN_DOCKER_NO_CACHE` | `FINN_IMAGE`; `docker/run --build` |
| `FINN_SKIP_XRT_DOWNLOAD`, `XRT_DEB_VERSION` | `--runtime xrt` with the pinned manifest; `SOURCE=supply` for site packages |
| `FINN_SKIP_DEP_REPOS`, `FINN_SKIP_BOARD_FILES` | resources on first use; `FINN_RESOURCES_OFFLINE=1` with a prepared cache |
| `FINN_DOCKER_EXTRA`, `DOCKER_INTERACTIVE`, `FINN_INST_NAME` | `-v`/`-e`, or a Compose file in `COMPOSE_FILE`; terminals are detected |
| `FINN_SINGULARITY` | `apptainer build finn.sif docker-daemon://finn:release` |
