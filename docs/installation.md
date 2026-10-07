# Installing and developing FINN

FINN is a Python package. The Python side is managed with standard tools
(`pyproject.toml` and `uv.lock`); the containers add what pip cannot provide, such
as the OS libraries Xilinx tools need and XRT. The design, and the reasoning behind
it, is in [environment.md](environment.md).

## Use FINN

```bash
pip install finn          # Python 3.11-3.12, Linux x86-64
```

Hardware flows additionally need Vivado/Vitis (by sourcing AMD's `settings64.sh`)
and a licence; in a checkout, [configure the machine](#configure-a-machine) once
instead.
The finn-hlslib HLS library and Vivado board files are fetched from their pinned
upstream commits on first use and cached (see
[external resources](#external-resources)); fetching them uses git, or GitHub's
archives where git is missing. The `finn_xsi` simulation extension is built
against your Vivado on first use.

## Develop FINN

Every modality uses the same lock and ends up with the same thing: an active venv
with FINN installed editable, and every other dependency at its locked version.

**Native host:**

```bash
git clone https://github.com/Xilinx/finn.git && cd finn
./setup-local.sh              # checks, then `uv sync` into .venv, optional XSI build
source scripts/activate.sh    # activates .venv and the Xilinx toolchain, if configured
```

`setup-local.sh` wraps `uv sync`; that is all a Python-only developer needs. With
Vivado configured, it also fetches finn-hlslib and the board files, so later builds
can run offline. After pulling a change to `uv.lock`, run `uv sync` again.

**Docker:**

```bash
./docker/run                           # shell; /opt/venv is active
./docker/run -- pytest -m util
./docker/run --fpga -- vivado -version # with the machine configured (below)
```

The image holds the locked dependencies in `/opt/venv`, active for every process.
When a container starts, its entrypoint installs the mounted checkout editable
(`uv sync --frozen --inexact` against `FINN_ROOT`, about a second with a warm
cache). If `uv.lock` has changed since the image was built, the start installs the
difference, or `docker/run` builds a new image, since the lock is an image input.
Editing FINN never requires a new image.

**Dev Container:** open the repository in VS Code and reopen in the container. The
interpreter is `/opt/venv/bin/python`; there is no setup step. With the machine
configured, the container also gets the Xilinx tools and licence, as with
`docker/run --fpga`.

**sbx:** see [the sbx guide](../docker/sbx/README.md). The same entrypoint installs
the workspace checkout when the sandbox starts (as the workload kit's startup
hook), or, in clone mode, once sbx has cloned it. FINN's workload has no coding
agent; add one as a kit.

`docker exec` and `sbx exec` do not wait for the entrypoint. A script that execs
into a container it has just started can wait for `/tmp/finn-ready`.

## Configure a machine

What is specific to a machine is where its Xilinx tools are and how it reaches
its licence server. Keep that in one file, outside every repository:

```text
# ~/.config/finn/xilinx.env
FINN_XILINX_PATH=/opt/Xilinx
FINN_XILINX_VERSION=2025.2
FINN_LICENSE_HOST=10.0.0.5
FINN_LICENSE_PORT=2100
FINN_LICENSE_VENDOR_PORT=2101
```

| Setting | Meaning |
|---|---|
| `FINN_XILINX_PATH` | The installation root. Both AMD layouts are found under it (`Vivado/2024.2` up to 2024.2, `2025.1/Vivado` after) |
| `FINN_XILINX_VERSION` | The version to use from that root |
| `FINN_LICENSE_HOST`, `FINN_LICENSE_PORT` | The FlexLM server, by host name or IPv4 address, and its port: `XILINXD_LICENSE_FILE=PORT@HOST` |
| `FINN_LICENSE_VENDOR_PORT` | The vendor daemon's (xilinxd) port, which sbx must allow too; see [the sbx guide](../docker/sbx/README.md#vivado-and-the-licence-server) for finding it |
| `PLATFORM_REPO_PATHS` | Vitis platforms (optional) |

Native activation, `docker/run --fpga`, the Dev Container and FINN's sbx kit all
read it, and so does FINN itself when it launches a Xilinx tool: a process that
never sourced `scripts/activate.sh` still gets the licence. The format is that of an sbx argument file, since the file is one:
`NAME=value` lines and `#` comment lines, absolute paths, no quotes, no trailing
comments, and only the names above. `XILINXD_LICENSE_FILE` in your environment
still takes precedence over the host and port, for licence files or several
servers. `FINN_XILINX_ENV` names another file (empty: none).

**One container, another version.** The file holds defaults. A variable of the
same name overrides it for one shell, container or sandbox, so installed
versions can run side by side:

```bash
FINN_XILINX_VERSION=2026.1 source scripts/activate.sh
FINN_XILINX_VERSION=2026.1 ./docker/run --fpga -- vivado -version
sbx create … --kit-args-file ~/.config/finn/xilinx.env \
    --kit-arg FINN_XILINX_VERSION=2026.1 …
```

An installation under another root is `FINN_XILINX_PATH` the same way (and, in
sbx, that root mounted). Combinations you use often can be further files, named
with `FINN_XILINX_ENV` or passed to `--kit-args-file`. Nothing machine-wide
changes: the image is the same for every version, and `finn_xsi` is built once
per Vivado installation.

Everything else is not machine configuration: the build directory, worker
counts and ports are ordinary per-run variables (`FINN_HOST_BUILD_DIR`,
`NUM_DEFAULT_WORKERS`, …; in sbx, `-m`/`--cpus`), and your own HLS/RTL sources
belong to your project, below.

## Your own HLS or RTL sources

Libraries, board files and kernels your designs need are
[external resources](#external-resources) declared in your project's
`pyproject.toml` (or a file in `FINN_RESOURCES_FILES`): pinned, fetched and
cached the same way in every modality. Working on one, point FINN at your
checkout with `FINN_RESOURCES_<NAME>=/path` ([overrides](#overrides-and-co-development)).

In containers:

* `docker/run` and the Dev Container mount every directory these variables name
  (`FINN_RESOURCES_<NAME>`, `FINN_RESOURCES_DIR` writable; the directories of
  `FINN_RESOURCES_FILES` read-only) at the same path, and pass the variables in.
  A relative path is taken from the checkout.
* An sbx sandbox reaches only what it mounts and its network policy allows:
  mount the directory and set the variable (`-e FINN_RESOURCES_<NAME>=/path`,
  and the path as an extra workspace). Resources fetched from the network need
  an allow rule, or fetch them on the host into a directory you mount:
  `finn-resources fetch --all --dest DIR`, then `-e FINN_RESOURCES_DIR=DIR -e
  FINN_RESOURCES_OFFLINE=1` (see [offline use](#offline-use)).

## Co-develop QONNX, Brevitas, finn-hlslib or another dependency

Point the dependency's source at your checkout, locally (do not commit this):

```toml
# pyproject.toml
[tool.uv.sources]
qonnx = { path = "../qonnx", editable = true }
```

Then run `uv sync` (natively, or inside a running container). In Docker, mount the
checkout at the same relative place, e.g. `./docker/run --volume "$PWD/../qonnx:$PWD/../qonnx"`.
uv also updates `uv.lock`; do not commit either change.

finn-hlslib, board files and other [external resources](#external-resources) are
not Python dependencies. Point FINN at a checkout with an environment variable
instead: `FINN_RESOURCES_HLSLIB=../finn-hlslib`.

## Add a dependency

Everything goes through `pyproject.toml` and `uv.lock`; no Dockerfile or setup
script changes.

* **Python package:** add it to `[project] dependencies` (or a dependency group),
  with a `[tool.uv.sources]` git entry if it is unreleased, then `uv lock`.
* **Build data** (HLS or RTL libraries, board files, Tcl libraries): declare it as
  an [external resource](#external-resources) in `src/finn/resources.toml`
  and look it up by kind with `finn.resources.paths(kind)`. Say whether FINN may
  redistribute it (`redistributable`); the images follow that.

## Inspect an environment

```bash
python -m finn.util.installation finn qonnx brevitas
finn-resources list
```

The first reports import locations, versions and installation metadata; FINN wheel
provenance preserves the Git revision and dirty state through its sdist. The second
lists the external resources, whether each is cached, overridden or missing, and
where.

## Package data and external resources

FINN's own RTL and HLS sources are packages in the wheel: `finn/rtllib` and
`finn/custom_hls` (both expected to give way to FinnLib), and the XSI bridge
sources in `finn/xsi/src`. Look them up by resource name, not location:
`finn.resources.path("rtllib")`, `"custom-hls"` or `"xsi"`; inside FINN,
`finn.util.resources.resource_path(family, *parts)` does the same. Deployment
data (the Vitis driver descriptor, PYNQ driver templates) is in `finn/deploy/data`.
The Python XSI driver lives at `src/finn_xsi/`.
Editable installations observe changes to these directly. Generated RTL, driver
files and compiled XSI extensions belong in writable build storage, never in the
installed distribution.

## External resources

FINN uses directory trees it does not contain: the finn-hlslib HLS library and
Vivado board files. Each is an *external resource*: a named tree declared with a
pinned source and a content digest, fetched on first use, verified and cached.
Your own board files and RTL or HLS libraries are declared the same way.

```bash
finn-resources list              # declarations; cached, overridden or missing
finn-resources list --boards     # the boards the board files provide
finn-resources fetch --all       # fetch now rather than on first use
finn-resources verify            # re-check the digests of cached copies
finn-resources check             # git, caches and overrides
```

`python -m finn.resources` is the same command. In Python,
`finn.resources.path("hlslib")` returns one resource's directory and
`finn.resources.paths("vivado-boards")` those of every resource of a kind.

### FINN's resources

Declared in `src/finn/resources.toml`, which ships in the wheel:

| Name | Kind | Source |
|---|---|---|
| `hlslib` | `hls-include` | [finn-hlslib](https://github.com/Xilinx/finn-hlslib) |
| `avnet-boards` | `vivado-boards` | [Avnet/bdf](https://github.com/Avnet/bdf) |
| `rfsoc2x2-boards`, `kv260-som-boards` | `vivado-boards` | one board each from [XilinxBoardStore](https://github.com/Xilinx/XilinxBoardStore) |
| `rfsoc4x2-boards`, `aup-zu3-boards` | `vivado-boards` | RealDigital's board support repositories |
| `finnlib` | `kernel-sources` | FinnLib, the RTL/HLS component library of `finn.kernels` (private; SSH access) |

The HLS include path is the `hlslib` resource. Every `vivado-boards` resource is
added to Vivado's board repository paths. Board files are third-party files FINN
does not redistribute: images built locally contain them, and the release image
and `pip install finn` fetch them on first use.

FinnLib changes together with FINN, so it is never baked into an image. Work
against a clone: `FINN_RESOURCES_FINNLIB=../finnlib` (in sbx, mount the clone and set it with `-e`; see
[the sbx guide](../docker/sbx/README.md)). The pin records the commit
a FINN revision was validated against; fetching it needs SSH access, and
`finn-resources update finnlib --ref BRANCH` moves it once that commit is pushed.

### Caches

FINN keeps its per-user state under `FINN_HOME` (default `~/.finn`): fetched
resources in `resources/`, the `finn_xsi` builds in `xsi/`, and, natively, the
build directory in `build/`. `docker/run` and the Dev Container set `FINN_HOME`
inside their build directory, which outlives the container.

Resources are looked up in two places; the first complete copy wins:

1. `FINN_RESOURCES_DIR` (default `$FINN_HOME/resources`): writable. Point it at a
   site-managed directory or one carried to an offline machine.
2. The system cache, `FINN_RESOURCES_SYSTEM_CACHE` (default `/opt/finn/resources`,
   if it exists): read-only, filled when an image is built.

A fetch goes into the first. Git sources are fetched as a single
commit, with only the declared subdirectory's files. The tree is built in a
temporary directory, checked against its digest and renamed into place under a
file lock, so concurrent first uses fetch once and nothing unverified is ever
used. Entries are named `<name>-<digest prefix>`: a new pin is a new entry, never
a stale one. `finn-resources clean --unused` removes entries that no current
declaration uses.

### Offline use

Where the network is available, fetch into a directory:

```bash
finn-resources fetch --all --dest /media/finn-resources
```

and on the offline machine, use it as the resource directory:

```bash
export FINN_RESOURCES_DIR=/media/finn-resources FINN_RESOURCES_OFFLINE=1
finn-resources check
```

With `FINN_RESOURCES_OFFLINE=1`, a resource missing from every cache is an error
that names the command to fetch it, never a network access. A site mirror can
stand in for a source: `FINN_RESOURCES_HLSLIB_URL=https://git.example/finn-hlslib.git`
replaces the declared URL. A declaration can also list `mirrors`, which are tried
in order after the source; every source is verified against the same digest.

### Overrides and co-development

`FINN_RESOURCES_<NAME>=/dir` (the name upper-cased, `-` written as `_`) uses a
local directory instead, without a digest check, for example
`FINN_RESOURCES_HLSLIB=../finn-hlslib` or
`FINN_RESOURCES_AVNET_BOARDS=$HOME/bdf`. `FINN_HLSLIB_PATH` still works as an
alias for the first. `FINN_BOARD_FILES_PATH` is no longer used: override each
board resource instead (FINN warns if it is set).

### Moving a pin

```bash
finn-resources update hlslib --ref main    # a branch, tag or commit
```

This resolves the ref to a commit, fetches it, computes the new digest and
rewrites `commit` and `digest` in the file that declares the resource. It edits
only those two values and checks that the file still parses to the same thing
otherwise, or leaves it untouched. For a declaration inside an installed package
it prints the new values instead, for you to put in your project.

### Declaring your own

A project declares resources in `[tool.finn.resources]` of the nearest
`pyproject.toml`, searched upward from the working directory, or in TOML files
(with `[resources.NAME]` tables) listed in `FINN_RESOURCES_FILES`. It may add
resources, and it may redefine FINN's, for example to try a newer finn-hlslib;
FINN logs each redefinition.

For example, board files for a custom carrier board and an RTL library shipped
as an archive:

```toml
# pyproject.toml
[tool.finn.resources.acme-boards]
description = "ACME carrier board files"
kind = ["vivado-boards"]               # added to Vivado's board paths
git = "https://github.com/acme/board-files.git"
commit = "0123456789abcdef0123456789abcdef01234567"
subdir = "boards/acme_carrier"         # only this directory is fetched...
into = "acme_carrier"                  # ...and placed here in the resource root
digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

[tool.finn.resources.acme-rtl]
description = "ACME RTL library 1.2"
kind = ["rtl"]
url = "https://downloads.acme.example/acme-rtl-1.2.tar.gz"
sha256 = "<sha256sum of the archive>"
subdir = "rtl"
digest = "sha256:<finn-resources digest of the rtl directory>"
```

For a git source, start with any well-formed digest (as above) and let
`finn-resources update acme-boards --ref v1.0` fill in `commit` and `digest`. For
an archive, set `sha256` from `sha256sum`, unpack it and run
`finn-resources digest acme-rtl-1.2/rtl`. (An archive whose content is a single
top-level directory is unpacked into that directory's place, as source archives
usually are.) Then `finn-resources list --boards` shows the new board, and
`finn.resources.paths("rtl")` returns the library's directory.

A library you keep yourself needs no pin: a `path` source is used in place, with
no fetch and no digest. A relative path is relative to the declaring file.

```toml
[tool.finn.resources.my-rtl]
kind = ["rtl"]
path = "../my-rtl"
```

| Field | Meaning |
|---|---|
| `git` and `commit`, `url` and `sha256`, `package`, or `path` | The source: a full commit id; a tar or zip archive and its checksum; a module whose installed data it is; or a local directory |
| `subdir` | Only this directory of the source (not for `path`) |
| `into` | Where to place it inside the resource root |
| `digest` | The tree digest of the resource root (not for `package` or `path`) |
| `kind` | Tags consumers look resources up by |
| `redistributable` | Whether FINN may put it into published images (default `false`) |
| `mirrors` | Alternative URLs, tried in order after the source |
| `description` | Free text |

Kinds are free-form. FINN itself uses `hls-include` and `vivado-boards`; other
kinds are for your own code to look up.

FINN's own sources (`rtllib`, `custom-hls`, `xsi`) are declared the same way, as
`package` resources, so `FINN_RESOURCES_RTLLIB=/path/to/rtllib` or a project
declaration replaces one.

A Python package can ship declarations too: put a `resources.toml` with
`[resources.NAME]` tables in one of its modules and name that module in the
`finn.resources` entry-point group:

```toml
# the package's pyproject.toml
[project.entry-points."finn.resources"]
acme = "acme_finn"
```

Its resources usually use `package = "acme_finn"` with `subdir = "rtl"`. Packages may add
resources but not redefine FINN's or another package's; only the project may.

Saved projects contain absolute installed-resource and intermediate-artifact
paths. Keep the selected installation, external data and build directories in
place. Moving directories, replacing/upgrading FINN or changing toolchains can
invalidate generated projects, compiled simulations and checkpoints. Regenerate
those artifacts rather than assuming a checkpoint is a self-contained export.

## RTL simulation (finn_xsi)

RTL simulation builds the `finn_xsi` extension against the selected Vivado the
first time it is needed, into `$FINN_HOME/xsi/<abi>-<key>` (one build
per Vivado installation and Python ABI; `FINN_XSI_BUILD_DIR` selects an exact
directory). Concurrent first uses build once. Without a usable toolchain,
simulation fails with the missing prerequisites. `python -m finn.xsi.setup`
builds ahead of time; `--check` only checks prerequisites.

## Tools and build boundaries

`build_dataflow project/` remains the ordinary public build command. Pending P6
integration of the private build-engine branch, the archived implementation runs the
existing model/configuration in a fresh child, with that directory as cwd and
legacy build/loader settings prepared before Python starts. Workers inherit that
environment. The API `build_dataflow_directory` uses the same boundary.
`build_dataflow_cfg` remains in-process for custom Python callbacks: callers using
XSI there must supply its loader environment before interpreter startup. Concurrent
in-process builds still share legacy configuration, stdout and logger state; use
separate build processes. See [the compatibility ledger](legacy-build-env-ledger.md).

`finn.util.toolchain.Selection` supports explicit local settings scripts or
an explicitly accepted configured environment, a site command directory, and a
launcher argv prefix. `Selection.prepare()` snapshots the mapping. With local
settings and no base mapping, it uses system PATH plus HOME/user/locale/temp/display
and licence inputs and `LD_PRELOAD` (the image preloads `libudev.so.1`, without
which Vivado's licence library crashes in `udev_enumerate_scan_devices`). To accept additional base variables, pass a mapping explicitly;
FINN does not attempt to unsource a previously activated installation. Bash startup
hooks and exported functions are removed in children. A prepared environment that
names no licence gets the machine file's (`XILINXD_LICENSE_FILE=PORT@HOST`), except
on a launcher route, whose site owns it; `launch_process_helper` without an
environment launches in `Selection().prepare()`'s.

HLS and stitched-IP Vivado commands execute with argv, child env and cwd. Probes
use the identical route and a bounded timeout. `CallHLS(toolchain=...)` and
`CreateStitchedIP(..., toolchain=...)` accept prepared internal selections. A
dataflow build prepares one toolchain (the selection its configuration names,
`DataflowBuildConfig.toolchain`, on the first step that runs a tool; by default
the environment as configured, with Vitis HLS) and passes it to every HLS
synthesis, stitching, FIFO-sizing, shell-build, link and driver step;
`build_dataflow_directory` prepares its build process's environment from the same
selection; `ZynqBuild`, `PrepareForLinking` and
`InsertAndSetFIFODepths` likewise pass theirs to the tool steps they run.
`FINN_TOOL_DIR_OVERRIDE` continues to select site wrappers; launcher prefixes
preserve those names and do not substitute local absolute vendor executables.
The site owns remote activation, path visibility and remote cancellation.

A build configuration names its toolchain, the HLS frontend included. With
Vivado/Vitis 2025.x that is `vitis-run`: a selection never guesses its frontend,
and the default, `vitis_hls`, is refused on 2025.x. A kernel-path build of TFC
for Ultra96 (`dataflow_build_config.json`; the environment as configured, so
`settings` stays empty):

```json
{
  "output_dir": "output",
  "synth_clk_period_ns": 5.0,
  "board": "Ultra96",
  "shell_flow_type": "vivado_zynq",
  "steps": ["phase_kernel_path", "phase_generate_outputs"],
  "kernel_exploration": [
    {"strategy": "target_throughput", "fps": 1000000},
    {"strategy": "size_fifos"},
    {"strategy": "placeholder"}
  ],
  "generate_outputs": ["bitfile", "pynq_driver", "deployment_package"],
  "toolchain": {"hls_frontend": "vitis-run"}
}
```

`kernel_exploration` lists the strategies that choose the KernelOps' open choices
through the DSE seam, run as written, each with its own parameters:
`target_throughput` folds the least parallelism that meets `fps` at the target's
clock, `pinned` commits a `kernel_choices.json` (`"path"`), `size_fifos` sizes
each channel's FIFO from both ends' beat patterns at the bottleneck (`direct`
where none is needed; `"margin"` words added to a FIFO it places, default 0), and
`placeholder` takes every choice left by a fixed rank (the default list is just
it). A choice
left open after the list is refused by name. The chosen choices are written to
`kernel_choices.json`, and what the exploration found (per-member cycles and
buffering, the bottleneck, attempts and time per strategy) to
`report/kernel_choices.json`.

Choose `FINN_HLS_FRONTEND=vivado_hls`, `vitis_hls`, or `vitis-run` for legacy entry
points; explicit selections, and a dataflow build's configuration, name the
frontend directly. Compatibility checks
retain the old-HLS (through 2020.1), standalone Vitis HLS (2020.1–2024.2), and
unified HLS (2025.1+) code-generation boundaries. Unified HLS must also advertise
HLS in its help. Legacy selection without a frontend retains the existing
Vivado-path version convention; capability discovery never chooses a newer tool.
Fake tests validate dispatch and rejection, not support for synthesizing every
release. Version/help probes establish neither synthesis nor licence success.

`run_process` supports a timeout, a cancellation event, and KeyboardInterrupt;
these kill the local process group and reap the child. Site wrappers must propagate
cancellation to their remote jobs. Logs include command, route, selected settings,
probed identity, cwd, elapsed time and status; environment snapshots are not logged
or persisted. Replay shell files require the selected environment to be prepared.

## Native simulation sessions and remaining gates

The internal `finn.xsi._session.run_session` accepts a concrete `SessionRequest`,
packed input streams, a selected child environment, an existing output root and a
required wall-clock limit. It executes a fresh Python image, loads the selected
bridge/kernel/design inside that process and exchanges pickle-free NumPy arrays
of hexadecimal words (including streams wider than 64 bits). Each session has
independent logs, outputs, optional waveforms and an atomic success record.
The initial built-in testbench uses FINN's `ap_clk`/optional `ap_clk2x` and active-low
`ap_rst_n` interface, a configurable reset duration, stream suffix and output
watchdogs. Custom testbenches are explicit Python files defining
`run(sim, io, request) -> metrics`, with JSON arguments; parent closures and native
handles never cross the process boundary.

The `finn_xsi` bridge is built against the selected Vivado on first use (see
above). Bridge/design compilation writes adjacent `.finn.json` records of tool identity,
source/header hashes, ABI where applicable and compile arguments. Sessions reject
changed or incompatible artifacts and recheck after exec. Older artifacts need a
rebuild to supply these records. Site execution requires an explicit site Python
command and shared absolute paths; the wrapper owns remote cancellation.

This is a directly testable mechanism, not yet integrated with the private build
engine. Actual AMD output/cycle/trace equivalence, AXI initialization/readback,
external memory, MLO, characterization, native concurrent reuse and performance
measurements remain open. Keep the current C++ harness and native execution paths
until those gates pass. Forced termination can leave partial waveforms; timeout and
cancellation retain partial stdout/stderr beside replay/session artifacts.

Only the sbx image variant reads native persistent Bash environment configuration.
Generic images have no FINN Bash startup hook. Bare vendor shims and global libudev
preload remain pending licensed/native validation. Site Tcl initialization is an
explicit read-only mount into the selected user's `.Xilinx` directory (for example
`--volume /site/xilinx-init:/home/agent/.Xilinx:ro` for that user's home), with no
copy or repair performed by the entrypoint.
