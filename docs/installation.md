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

## Parts and custom devices

A kernel-path build names its part (or a board, which names one), and FINN reads
the part's facts from its part catalog, `finn.platform.catalog`: every part of
Zynq-7000, UltraScale, UltraScale+ and Versal that Vivado 2025.2 installs, its
device, and the device's LUTs, flip-flops, RAMB18s, UltraRAMs and DSPs per SLR.
Devices with identical resources share one record, and each device states the
others it shares with. A device sold on a larger die (an XCZU2EG on the XCZU3EG's)
states its own totals. Where it spans several SLRs, Vivado does not say how they
split: each SLR states its site capacity, an upper bound for that SLR alone, and the
device's totals cap their sum (`capped`), so not every SLR can be full at once.
Which of those caps Vivado was seen to enforce, and which are inferred, is the
device's `cap_evidence`. What FINN builds on a device (its fabric, DSP block and
whether an UltraRAM takes initial contents) is the rule for Vivado's
`(ARCHITECTURE, FAMILY)` pair, `finn.platform.architectures`. A part the catalog
does not have is refused, `unknown-part`, with close names and none chosen.

```python
from finn.platform import device, part

found = part("xcvu9p-flga2104-2L-e")
found.device.resources      # the device's totals: the sum of its SLRs'
found.device.slrs           # three SLRs, each its own Resources
device("xczu3eg").shared_with
```

The catalog is generated from the installed Vivado's part database and committed
(`src/finn/platform/data/`). `python -m finn.platform.generate check` extracts it
again and reports every part, device or field that differs (exit 0: identical);
`write` replaces it. Both need Vivado and take about an hour (`--jobs`, Vivado
processes at once, 8 by default).

**A device FINN does not ship** (a new part, an engineering sample, a board with a
custom part) is added without editing FINN: an overlay file, named by the machine
setting `FINN_PLATFORM_CATALOG` (read once, when the catalog is first queried). It
holds records of the committed data's schema, each list optional, and is validated
when it is loaded (`catalog-overlay-invalid` names what is wrong):

```json
{
  "source": "Acme's data sheet for the XCZU3EG-ES1, revision 0.3",
  "devices": [
    {
      "name": "xczu3eg_es1",
      "architecture": "zynquplus",
      "family": "zynquplus",
      "resources": {"slrs": [{"lut": 70000, "ff": 141120, "bram18": 432, "uram": 0, "dsp": 360}]}
    }
  ],
  "parts": [
    {"name": "xczu3eg_es1-sbva484-1-e", "device": "xczu3eg_es1",
     "package": "sbva484", "speed": "-1", "temperature": "E"}
  ]
}
```

```bash
FINN_PLATFORM_CATALOG=~/acme/parts.json build_dataflow ...
```

- A device's `resources` is its record, or the `digest` of one the catalog ships
  (`device("xczu3eg").digest`); a part may sit on a shipped device. A record states
  each SLR (`slrs`, as above), or, for a device of one SLR, only its `totals`; one of
  several SLRs whose split is not known states `totals` and `slr_count`, and a
  reduced die states each SLR's site capacity with its `totals` and `"capped": true`.
- Its fabric and DSP block are its pair's rule. Where the pair has none, or the
  device is built otherwise, the device states `fabric` (`series7`, `ultrascale`,
  `versal`), `dsp` (`DSP48E1`, `DSP48E2`, `DSP58`) and, optionally, `uram_init`.
- Every entry carries a `source`, its own or the file's. The part's `source`
  states it with the overlay's path, and so does the exploration report
  (`kernel_exploration.json`, `resources.part.source`).
- An entry whose name is a shipped part or device is refused, unless it says
  `"overrides": true`; `overrides` on a name the catalog does not ship is refused.

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

FINN's own sources in the wheel are the XSI bridge's, in `finn/xsi/src`; the kernels'
RTL and HLS are FinnLib's (the `finnlib` resource). Look them up by resource name,
not location: `finn.resources.path("xsi")`; inside FINN,
`finn.xsi.paths.resource_path(family, *parts)` does the same. The pynq shell's
data (its PYNQ driver files and design description) is in `finn/shells/pynq/data`.
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
`FINN_RESOURCES_AVNET_BOARDS=$HOME/bdf`. The retired `FINN_HLSLIB_PATH` and
`FINN_BOARD_FILES_PATH` select nothing.

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

FINN's own sources (`xsi`) are declared the same way, as `package` resources, so
`FINN_RESOURCES_XSI=/path/to/src` or a project declaration replaces one.

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

`build_dataflow project/` remains the ordinary public build command. It builds in
process, starting no child, as `build_dataflow_directory` and `build_dataflow_cfg` do,
and changes no working directory: the configuration's relative paths (`output_dir`,
`verify_input_npy`, the preparation's `preprocessing`, a `pinned` strategy's `path`,
the toolchain's settings scripts and command directory) name files in the project
directory, resolved when `read_build_config` reads it, and a relative `FINN_BUILD_DIR`
is relative to the directory the command runs from. Every tool runs in the
environment of the configuration's toolchain (the machine's when it names none),
prepared once for the build; the process's own environment is left as it is. The
command exits 0 when the build completes, 255 when a step fails
(`build_dataflow_directory` returns -1), and 1 with the traceback when the
configuration is refused (it raises). A caller that loads XSI in process supplies its
loader environment before interpreter startup (`docker/finn-toolchain.sh` does for a
shell). Concurrent in-process builds still share stdout and logger state; use
separate build processes. See [the machine settings and remaining obligations](legacy-build-env-ledger.md).

What a build leaves in its output directory: `build_dataflow.log` (for a reader:
every record with its level, step and logger), upstream's `time_per_step.json` and
`config_check_report.txt`/`.json`, the steps' reports under `report/`, each tool's
transcript under `logs/<step>/` (`<run>.stdout.log`, `.stderr.log`, written as the
tool runs), the intermediate models, and the outputs asked. `report/build.json` is
the build's manifest, the index of the run a program reads after the exit code, and
never the log: the build's identity (FINN's revision where it is known, the
toolchain's selection, the Vivado release, the target, the model), the configuration
checks that failed, each step with its phase, status (`running` while it runs, then
`completed` or `failed`), seconds, the reports it wrote with their findings by code,
its tool runs with their transcripts, status, exit and messages by severity, and its
verifications; each verification asked with its verdict (`pass`, `fail`, `error`,
`not-run`); and the `outcome`: `completed`, `failed` (the step, the error's kind:
`refusal`, `tool-unavailable`, `tool-failed`, `verdict` or `bug`, the error, a
refusal's findings by code) or `refused` (the configuration, before the first step).
It is rewritten as the build goes, so a build that stopped says where. The model
records only what a later step consumes, in `finn.outputs`: the packaged IP (`ip`,
`vlnv`, `interfaces`, `made_from`) on the partition's body, the bitfile (`bitfile`,
`hwh`, `bitfile_made_from`, `bitfile_options`) on the parent graph; the bitfile's
reports are beside it under `report/`.

`finn.toolchain.launcher.Selection` supports explicit local settings scripts or
an explicitly accepted configured environment, a site command directory, and a
launcher argv prefix. `Selection.prepare()` snapshots the mapping. With local
settings and no base mapping, it uses system PATH plus HOME/user/locale/temp/display
and licence inputs and `LD_PRELOAD` (the image preloads `libudev.so.1`, without
which Vivado's licence library crashes in `udev_enumerate_scan_devices`). To accept additional base variables, pass a mapping explicitly;
FINN does not attempt to unsource a previously activated installation. Bash startup
hooks and exported functions are removed in children. A prepared environment that
names no licence gets the host file's (`XILINXD_LICENSE_FILE=PORT@HOST`), except
on a launcher route, whose site owns it.

Vendor commands execute with argv, child env and cwd. Probes use the identical
route and a bounded timeout. Every function that runs a tool requires a
prepared toolchain (`toolchain=`). Only the entry points a user calls default to
one: `PackagePartition`, `ElaboratePartition` and the XSim executor when they are
made, the pynq shell's `BuildPynqBitfile`, and the command lines
(`python -m finn.platform.generate`, `python -m finn.xsi.setup`). Each prepares the
machine's once and passes it down,
`finn.toolchain.launcher.host_toolchain()`: the environment as configured (no
settings script is sourced), under the site command directory
`FINN_TOOL_DIR_OVERRIDE` names, if any, with the HLS frontend of the machine
file's `FINN_XILINX_VERSION` (`vitis-run` from 2025.1, else `vitis_hls`). A
build prepares one toolchain (the selection its configuration names,
`DataflowBuildConfig.toolchain`, or the machine's when it names none, on the
first step that runs a tool) and passes it to every step that runs a tool
(HLS synthesis, packaging, simulation, the shell's build). A stated selection is laid over the
machine's (`host_selection(stated=...)`): each field it states wins, and a field
it leaves `null` (`command_dir`, `hls_frontend`, `vivado_jobs`) is the machine's, so
stating one field drops none of the others. How many runs Vivado launches at once is
the machine setting `FINN_VIVADO_JOBS` (unset: the machine's cores, at most 16), which
a kernel-path build may state over as `toolchain.vivado_jobs`. Launcher prefixes
preserve tool names and do not substitute local absolute vendor executables. The
site owns remote activation, path visibility and remote cancellation.

A build configuration may name its toolchain, the HLS frontend included; the
frontend is never guessed from the executables found: it is the one stated, or the
host file's release's, and `vitis_hls` is refused on 2025.x. A kernel-path build has a configuration of its own,
`finn.builder.build_dataflow_config.DataflowBuildConfig`, written as
`dataflow_build_config.json` beside `model.onnx`, which `build_dataflow` builds. A
kernel-path build of TFC for Ultra96 in the
Zynq shell, its preprocessing beside the export (the tool environment as configured, so
the configuration states no `toolchain`):

```json
{
  "output_dir": "output",
  "target": {"period_ns": 5.0, "board": "Ultra96", "shell": "pynq"},
  "generate_outputs": ["bitfile", "pynq_driver", "deployment_package"],
  "preprocessing": "preproc.onnx",
  "input_datatype": "UINT8",
  "topk": 1,
  "strategies": [
    {"strategy": "target_throughput", "fps": 1000000},
    {"strategy": "size_fifos"}
  ]
}
```

`target` states the clock period, a `board` or a `part` (a board gives its part; a
part beside it is an assertion) and the `shell`: `ip` unless one is stated, the
packaged IP its user integrates, whose target states the part alone; `pynq`, the
Zynq block design for a board. `stitched_ip` and `ooc_synth` are the partition's
own outputs, on any shell: `stitched_ip/` holds the packaged IP, its interface
description `interface.json` (the part and period, each clock's `FREQ_HZ`, each
AXI-Stream port with its element type, lanes, beats and frame shape, each AXI-Lite
bus with its register map) and an XSim testbench (`testbench/run.sh`, one frame
of `verify_input_npy` or a generated one, checked against the partition in
Python); `ooc_synth` synthesizes the IP out of context. Every kernel-path build
writes `report/resources.json`: each member of the shell (the partition, each end,
each IP of the static region) and their total, as the model states them, as Vivado
synthesized each IP out of context (`pynq`'s per-IP runs with `bitfile`; `ip`'s
packaged partition with `ooc_synth`), and as placed (`pynq` with `bitfile`); with
`ooc_synth`, also the partition per member, what synthesis flattened
`unattributed`. The shell outputs (`bitfile`, `pynq_driver`, `deployment_package`)
need a shell that integrates the partition, so they are refused on `ip`. The default steps are the build's phases, each a list of
steps: `phase_prepare_model` (the export cleaned, its pre- and post-processing added,
its Quant nodes lowered, then tidied, streamlined and made channels-last),
`phase_convert_to_hardware` (the KernelOps), `phase_create_accelerator` (the
partition), `phase_optimize_hardware` (their choices and the partition's
verification), `phase_emit_hardware` (the partition's IP) and
`phase_generate_outputs` (what the shell makes), in
`finn.builder.build_dataflow_steps.build_dataflow_phases`. `steps` replaces that list;
its entries, and those of an edit, are phase or step names, import paths
(`package.module.name`) of a step function or of a qonnx `Transformation` class, run with
its defaults, or in Python the function, `functools.partial` or `Transformation` itself.
`step_edits` changes the list in order, without restating it:
`{"after": "step_streamline", "insert": ["finn.transformation.streamline.extract_multithreshold_scale_bias.ExtractMultiThresholdScaleBias"]}`,
`{"before": ..., "insert": ...}`, `{"replace": ..., "with": ...}` and
`{"remove": "step_verify_partition"}`. A phase as target is the phase as a whole; any
other name is a step, wherever it runs, and entries placed beside it join its phase.
Each target must name exactly one place. `start_step` and `stop_step` name a phase or a
step.

The configuration's fields say how the build runs (`output_dir`, `target`, `toolchain`,
`generate_outputs`, `verify_steps`, `verify_input_npy`, `steps`, `step_edits`,
`start_step`, `stop_step`, `save_intermediate_models`, `enable_build_pdb_debug`,
`verbose`). Every other top-level key is a setting (in Python, `settings={...}`): the
value of the keyword-only parameter of that name of the steps that declare it
(`def step_build_bitfile(model, cfg, *, hw_debug: bool = False)`), decoded by its
annotation; a path setting is relative to the build directory. A setting no step of
the build declares is refused, naming the settings its steps read; steps that declare
the same name share its value. The preparation steps declare `override_inpsize`
(`step_cleanup`), `max_multithreshold_bit_width` (`step_qonnx_to_finn`, default 8),
and `preprocessing`, `input_datatype` and `topk` (`step_add_pre_post_processing`);
`step_explore` `strategies`; `step_build_bitfile` `hw_debug` (integrated logic
analyzers on the ends' streams, refused on `ip`). The build's manifest records each
step it runs with its settings, and each setting with its value or default and its
readers (`plan`). JSON is read strictly: a key a nested object does not declare, a null
where a field is not optional, a wrong type and an unknown value are refused, naming
their place (`configuration.target: ...`).

`step_explore`'s `strategies` lists the strategies that choose the KernelOps' choices
through the DSE seam, run as written, each with its own parameters. Every choice the
partition's body leaves open first takes its baseline, the first viable case its
kernel lists (the least parallelism; each memory in the explicit style the size of
what it holds orders first, `auto` last; every transport `direct`); each strategy
then reopens the choices it decides, explores, and settles the rest back to their
earlier values (or their baselines where those are no longer valid), and every value
is saved. `target_throughput` folds the least parallelism that meets `fps` at the
target's clock (trying each case of an unordered choice, such as a MatMul's core on a
DSP58 part, and stating the bottleneck reached and whether `fps` is met),
`max_throughput` the fewest cycles a frame, of every point it tried, whose resources
(the shell's: its partition, ends and static region) stay within `"within"`, a
required `{resource: fraction}` of the part's totals (`{"lut": 0.5}`); where a member
states no resources (a memory pinned `auto`: placed by Vivado, not stated), it
budgets what is stated, a lower bound, and warns, naming it. `pinned` sets the
values of a `kernel_choices.json` (`"path"`) and holds them for the rest of the
list: write it first, so the strategies after it fold around the pins.
`size_fifos` sizes each channel's FIFO from both ends' beat patterns at the
bottleneck (`direct` where none is needed; `"margin"` words added to a FIFO it
places, default 0; its memory style FinnLib's `auto` unless `"ram_style"` names
one); it goes after the last strategy that changes folding, and a list that folds
after it is refused, as is a list that changes the design around FIFOs already
placed without sizing them again. `baseline` resets every value not pinned to its
baseline; `placeholder`, for debugging, sets each open required choice to its first
case. The default list is `[{"strategy": "size_fifos"}]`: the baseline folding, its
FIFOs sized; a list you write replaces it. A required choice left open (a FIFO's
depth) is refused by name when the partition is built. Every value is written to
`kernel_choices.json`, by graph name
(`{"MatMul_0_param0": {"transport": "direct"}, "MatMul_0": {"compute.packed.pe": 16}}`:
a kernel's choices under its node, a channel's under the tensor it carries, in graph
order: each node's input channels, the node, its output channels), and a log of the
exploration (each strategy with the values it changed, reopened, kept and
re-baselined, attempts and time; every value with who set it last; the pins; whether
FIFOs were sized; per-member cycles and buffering, the bottleneck, and the shell's
resources against the part's) to `report/kernel_exploration.json`. A point that uses more of a resource
than the part has is a warning naming that resource, never a refusal.

Explicit selections, and a build's configuration, name the frontend
(`vivado_hls`, `vitis_hls` or `vitis-run`) directly; the machine's follows the
host file's release. Compatibility checks retain the old-HLS (through 2020.1),
standalone Vitis HLS (2020.1–2024.2), and unified HLS (2025.1+) code-generation
boundaries. Unified HLS must also advertise HLS in its help; capability discovery
never chooses a newer tool.
Fake tests validate dispatch and rejection, not support for synthesizing every
release. Version/help probes establish neither synthesis nor licence success.

`run_process` supports a timeout, a cancellation event, and KeyboardInterrupt;
these kill the local process group and reap the child. Site wrappers must propagate
cancellation to their remote jobs. Logs include command, route, selected settings,
probed identity, cwd, elapsed time and status; environment snapshots are not logged
or persisted. Replay shell files require the selected environment to be prepared.

## Native integration

Only the sbx image variant reads native persistent Bash environment configuration.
Generic images have no FINN Bash startup hook. Bare vendor shims and global libudev
preload remain pending licensed/native validation. Site Tcl initialization is an
explicit read-only mount into the selected user's `.Xilinx` directory (for example
`--volume /site/xilinx-init:/home/agent/.Xilinx:ro` for that user's home), with no
copy or repair performed by the entrypoint.
