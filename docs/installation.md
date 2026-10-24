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
./docker/run -- pytest tests/util
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

A build names its part (or a board, which names one), and FINN reads
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
from finn.platform.catalog import device, part

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
  states it with the overlay's path, and so does each strategy step's report
  (`report/step_size_fifos.json`, `resources.part.source`).
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
resources in `resources/`, the `finn_xsi` builds in `xsi/`, HLS products in `hls/`,
packaged IPs in `ip/` (each keyed by what the tool is given and the tool, and copied
into the build that asks for it), and, natively, the build directory in `build/`. `docker/run` and the Dev Container set `FINN_HOME`
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
the preparation's `preprocessing`, the toolchain's settings scripts and command directory) name files in the project
directory (`read_build_config` reads it with the directory as its `base_dir`: a field's
path is joined to it when read, a path setting's, given or by default, when the build decodes it), and a relative `FINN_BUILD_DIR`
is relative to the directory the command runs from. Every tool runs in the
environment of the configuration's toolchain (the machine's when it names none),
prepared once for the build; the process's own environment is left as it is. The
command exits 0 when the build completes, 255 when a step fails
(`build_dataflow_directory` returns -1), and 1 when the configuration is refused,
its findings on the console and in the log, one line each, without a traceback: before
the first step, one `ConfigurationRefused` carries
every rejection found, the target's, the plan's and the configuration checks', each a
finding with its code, owned by its place in the configuration
(`configuration.target.board: unknown-board: ...`). A setting a step cannot take
is refused when that step runs, as a `refusal` of the step. A caller that loads XSI in process supplies its
loader environment before interpreter startup (`docker/finn-toolchain.sh` does for a
shell). Concurrent in-process builds still share stdout and logger state; use
separate build processes. See [the machine settings and remaining obligations](legacy-build-env-ledger.md).

What a build leaves in its output directory: `build_dataflow.log` (for a reader:
every record with its level, step and logger), the
steps' reports under `report/`, each tool's
transcript under `logs/<step>/` (`<run>.stdout.log`, `.stderr.log`, written as the
tool runs), the intermediate models, and the outputs its steps made. `report/build.json` is
the build's manifest, the index of the run a program reads after the exit code, and
never the log: the build's identity (FINN's revision where it is known, the
toolchain's selection, the target and the model), what checking the
configuration found (its rejections and the limitations warned of), the plan (the steps
it runs, and each setting with its value or default and its readers), each step with
its status (`running` while it runs, then
`completed` or `failed`), seconds, the reports it wrote (by path), its tool runs with
their transcripts, status, exit and messages by severity, the stored products it used
(an entry of the packaged-IP store or the HLS cache, `built` in this run or `reused`),
and the findings it noted (a
limitation the run goes on with, such as the delivered clock's shortfall); and the
`outcome`: `completed`, `failed` (the step, the error's kind: `refusal`,
`tool-unavailable`, `tool-failed` or `bug`, the error, a refusal's findings by code) or
`refused` (the configuration, before the first step). It is rewritten as the build
goes, so a build that stopped says where. Its reports, tool runs and noted findings
reach it as records of one logger, `finn.core.reporting.manifest`, whose level
the build sets: a caller that quiets another of FINN's loggers keeps them. The model
records only what a later step consumes, in `finn.outputs`: the packaged IP (`ip`,
`vlnv`, `interfaces`, `ip_key`, `ip_options`) on the partition's body, the bitfile
(`bitfile`, `hwh`, `bitfile_made_from`, `bitfile_ip_key`, `bitfile_options`) on the
parent graph; the bitfile's reports are beside it under `report/`.

What a build logs, at which level, and what reaches the console
(`finn.core.reporting.log`):

| Level | When | Console |
|---|---|---|
| ERROR | the build stops: a refusal (one line per finding), a tool unavailable or failed, a bug (with its traceback) | yes; a traceback only for a bug |
| WARNING | the build goes on, giving less than asked or than the user would assume: a noted limitation (a budget, the delivered clock, a setting no step reads, a choice stale at the cut) | yes |
| INFO | what the build did: its paths and steps, each step's summary, the reports written, the tool runs | the build's own lines (the loggers under `finn.builder`: the runner's and each step's summary); the rest in the log only |
| DEBUG | per-item detail | no; in the log only with `verbose` |

Python's warnings (qonnx's, NumPy's, streamlining's) go to the log at WARNING; the
console ends with one line, `N Python warnings, in <the log>`. With `verbose` the console
shows everything the log has but DEBUG, each Python warning included.

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
`FINN_TOOL_DIR_OVERRIDE` names, if any, with the HLS frontend of the host
file's `FINN_XILINX_VERSION` (`vitis-run` from 2025.1, else `vitis_hls`). A
build prepares one toolchain (the selection its configuration names,
`DataflowBuildConfig.toolchain`, or the machine's when it names none, on the
first step that runs a tool) and passes it to every step that runs a tool
(HLS synthesis, packaging, simulation, the shell's build). A stated selection is laid over the
machine's (`host_selection(stated=...)`): each field it states wins, and a field
it leaves `null` (`command_dir`, `hls_frontend`, `vivado_jobs`) is the machine's, so
stating one field drops none of the others. How many runs Vivado launches at once is
the machine setting `FINN_VIVADO_JOBS` (unset: the machine's cores, at most 16), which
a build may state over as `toolchain.vivado_jobs`. Launcher prefixes
preserve tool names and do not substitute local absolute vendor executables. The
site owns remote activation, path visibility and remote cancellation.

A build configuration may name its toolchain, the HLS frontend included; the
frontend is never guessed from the executables found: it is the one stated, or the
host file's release's, and `vitis_hls` is refused on 2025.x. A build's configuration is a
`finn.builder.build_dataflow_config.DataflowBuildConfig`, written as
`dataflow_build_config.json` beside `model.onnx`, which `build_dataflow` builds. A
build of TFC for Ultra96 in the
Zynq shell, its preprocessing beside the export (the tool environment as configured, so
the configuration states no `toolchain`):

```json
{
  "output_dir": "output",
  "target": {"period_ns": 5.0, "board": "Ultra96", "shell": "pynq"},
  "preprocessing": "preproc.onnx",
  "input_datatype": "UINT8",
  "topk": 1,
  "steps": [
    "phase_prepare_model",
    "phase_convert_to_hardware",
    "phase_create_accelerator",
    "step_target_throughput",
    "phase_optimize_hardware",
    "phase_generate_outputs"
  ],
  "target_fps": 1000000
}
```

`target` states the clock period, a `board` or a `part` (a board gives its part; a
part beside it is an assertion) and the `shell`: `ip` unless one is stated, the
packaged IP its user integrates, whose target states the part alone; `pynq`, the
Zynq block design for a board.

`boundary` states, for the `ip` shell, what its integrator presents or takes at each
port of the partition, by the partition's port names (`s_axis_<i>` for its i-th input,
`m_axis_<j>` for its j-th output): the pace, a beat every `k` cycles (`{"every": k}`) or
each beat's cycle over a frame and the frame's span (`{"times": [...], "span": n}`), and
optionally a `"row_major"` `order`, one pass a frame (`passes`) and the `lanes` a beat,
the port's word. Beat times state `lanes` and `passes` too, and their count must be the
beats those make of the port's tensor. What a statement leaves out is as the partition
presents it (an output as produced), and a port it leaves out is unstated. FIFO sizing
reads the stated pace, the partition's cycles a frame are no fewer than its span (the
partition runs no faster than its integrator supplies an input or takes an output), and
stated lanes set the port's AXIS width. On `pynq` the shell's movers state every port,
and a statement there is refused before the first step:

```json
"boundary": {
  "s_axis_0": {"pace": {"every": 2}},
  "m_axis_0": {"order": "row_major", "passes": 1, "lanes": 4,
               "pace": {"times": [0, 3, 6], "span": 12}}
}
```

A build makes what its list's steps make: each output is a step, and listing it makes
the output. `step_package_partition` packages the partition, on any shell, into
`stitched_ip/`: the IP, its interface description `interface.json` (the part and
period, each clock's `FREQ_HZ`, each AXI-Stream port with its element type, lanes,
beats and frame shape, each AXI-Lite bus with its register map) and an XSim testbench
(`testbench/run.sh`, one frame drawn from the partition's input datatypes, checked
against the partition in Python; the build does not run it); its setting `ooc_synth`
(default false) synthesizes the IP out of context. `step_build_bitfile` builds the
shell around the partition to a bitfile (packaging the partition itself unless
`step_package_partition` packaged it), `step_make_driver` writes the PYNQ driver for
that bitfile, and `step_deployment_package` packages both for the board, in
`deploy/`. These three need a shell with a bitfile build (`pynq`), and each is refused
at its turn on another, such as `ip`. An output made from another is listed after it:
the driver after the bitfile, the deployment after both; a step whose input is missing
is refused at its turn, naming the step to list before it. `step_resources_report`
writes `report/resources.json`: each member of the shell (the partition, each mover,
each IP of the static region) and their total, as the model states them, as Vivado
synthesized each IP out of context (`pynq`'s per-IP runs with the bitfile; `ip`'s
packaged partition with `ooc_synth`), and as placed (`pynq` with the bitfile); with
`ooc_synth`, also the partition per member, what synthesis flattened `unattributed`.

A phase is a name for its steps: `phase_prepare_model` (the export cleaned, its pre- and
post-processing added, its Quant nodes lowered, then tidied, streamlined and made
channels-last), `phase_convert_to_hardware` (the KernelOps), `phase_create_accelerator`
(the partition), `phase_optimize_hardware` (their choices, one strategy step at a time,
and the partition's verification), `phase_emit_hardware` (the packaged IP) and
`phase_generate_outputs` (the bitfile, the driver, the resources report and the
deployment package), in `finn.builder.build_dataflow_steps.build_dataflow_phases`. The
premade lists, in `build_dataflow_lists` beside them, are names for a list from the
export to an output, each the first four phases' steps, then:

| List | Then | Makes |
|---|---|---|
| `list_partition` (the default) | `step_resources_report` | the partition, its choices (`report/kernel_choices.json`) and its modelled resources |
| `list_ip` | `step_package_partition`, `step_resources_report` | and the packaged IP, its interface description and testbench |
| `list_bitfile` | `step_build_bitfile`, `step_resources_report` | and the bitfile (`pynq`) |
| `list_deployment` | `step_build_bitfile`, `step_make_driver`, `step_resources_report`, `step_deployment_package` | and the bitfile, its driver and the deployment package (`pynq`) |

`steps` is the list the build runs in place of the default one: names of phases or
premade lists (each its steps), step names, import paths (`package.module.name`) of a
step function, or in Python the step function itself. A qonnx `Transformation` runs in a
step function of its own (`def step_name(model, cfg): return model.transform(...)`). A
list names a phase for the steps it does not change, as above; a premade list stands
alone (`"steps": ["list_ip"]`); in Python,
`finn.builder.build_dataflow_steps.default_steps()` is the default list by step name, to
change and give as `steps`. `start_step` and `stop_step` name a step once in the list,
or a phase or premade list: its first step to start from, its last to stop after.

The configuration's fields say how the build runs (`output_dir`, `target`, `boundary`,
`toolchain`, `steps`, `start_step`, `stop_step`, `save_intermediate_models`, `enable_build_pdb_debug`,
`verbose`). Every other top-level key is a setting (in Python, `settings={...}`): the
value of the keyword-only parameter of that name of the steps that declare it
(`def step_build_bitfile(model, cfg, *, hw_debug: bool = False)`), decoded by its
annotation; a path setting is relative to the build directory. A setting no step of
the build declares is refused, naming the settings its steps read; steps that declare
the same name share its value, and a step that names a parameter like a field is
refused. The preparation steps declare `override_inpsize`
(`step_cleanup`), `max_multithreshold_bit_width` (`step_qonnx_to_finn`, default 8),
and `preprocessing`, `input_datatype` and `topk` (`step_add_pre_post_processing`);
the strategy steps theirs (below); `step_package_partition` `ooc_synth`;
`step_build_bitfile` `hw_debug` (integrated logic analyzers on the movers' streams); its
clock report's objective
is the `target_fps` the partition's body records where `step_target_throughput` folded
it (`finn.choices/objective_fps`), and none where a later step that may change the folding
(a hold, a maximum throughput, going back to the defaults) explored it since. The build's manifest records the steps it runs, and
each setting with its value or default and its readers (`plan`). JSON is read strictly: a key a nested object does not declare, a null
where a field is not optional, a wrong type and an unknown value are refused, naming
their place (`configuration.target: ...`), as a `ConfigurationRefused`. A build
directory's configuration that does not decode is refused in the output directory it
states, when it states one as a string: logged there and recorded in a new manifest, as
`refused`; otherwise on the console only.

Every choice is given its value where it is made, and the design-space engine sets
none itself. Conversion gives each KernelOp's choices their default policies, once: a
value the kernel's author fixes, or the first valid value its kernel lists (the least
parallelism; each memory in the explicit style the size of what it holds orders first,
`auto` last), or none for a required choice (a FIFO's depth). The cut makes each
channel's choices the same way, first valid (every transport `direct`): an interim rule,
until the ends are designed. The strategy steps then choose the KernelOps' choices
through the DSE seam, one strategy each; a build lists them like any step, in `steps`.
Each opens the partition's body, runs its strategy, which reopens the choices it
decides, explores, and gives the rest back their earlier values (setting one by its
policy where its earlier value is no longer valid, and saying so), and saves every
value in the body. A change that breaks a kernel's rule is refused, naming the rule.

| Step | Settings | What it does |
|---|---|---|
| `step_target_throughput` | `target_fps` (required), `relax_target_fps` (default true) | folds the least parallelism that meets `target_fps` at the target's clock |
| `step_max_throughput` | `max_throughput_within` (required: `{"lut": 0.5}`) | the fewest cycles a frame whose resources stay within the fractions of the part's |
| `step_size_fifos` | `fifo_margin` (default 0), `fifo_ram_style` (default none: FinnLib's `auto`) | sizes each channel's FIFO at the folding before it |
| `step_default_choices` | none | sets every value not held back to its default: conversion's policies for the kernels', the interim rule for the channels' |
| `step_placeholder_choices` | none | for debugging, sets each open required choice to its first possible value |

`step_target_throughput` tries each case of an unordered choice, such as a MatMul's
core on a DSP58 part, and states the bottleneck reached and whether `target_fps` is
met; with `relax_target_fps`, a throughput the folding cannot reach is relaxed to the
fastest it reaches. `step_max_throughput` keeps, of every point it tried, the fastest
whose resources (the shell's: its partition, movers and static region) stay within
`max_throughput_within`; where a member states no resources (a memory pinned `auto`:
placed by Vivado, not stated), it budgets what is stated, a lower bound, and warns,
naming it. A hold is recorded in the partition's body (`finn.choices/held`, the held
keys; the values are the ones the body states), so every later strategy step of the build
keeps it, going back to the defaults included. The build has no step that fixes values: a user fixes them with
`Hold` in a step of their own, listed before the strategies that should fold around them:

```python
from finn.kernels.explore import Hold
from finn.transformation.kernels import explore_partition

def step_my_choices(model, cfg):
    explore_partition(model, Hold({"MatMul_0": {"simd": 8},
                                   "MatMul_0_param0": {"source.ram_style": "block"}}))
    return model
```

(`steps=[..., step_my_choices, "step_target_throughput", "step_size_fifos", ...]`; the
values are `{graph name: {key: value}}`, a kernel's under its node, a channel's under its
tensor.) A value the partition does not take, or a key an earlier hold keeps at another
value, is refused, named. The trade-off: a build described by its JSON alone cannot fix a
choice without Python, since the step is named there by its import path
(`package.module.step_my_choices`).

`step_size_fifos` sizes each
channel from both ends' beat patterns at the bottleneck (`direct` where none is
needed), `fifo_margin` words added to a FIFO it places. The default phase is
`[step_size_fifos, step_verify_partition]`: conversion's folding, its FIFOs sized; a
target throughput is `step_target_throughput` listed before it, with `target_fps` (the
example above). FIFO depths are sized at the folding before them: run `step_size_fifos`
after the last step that may change the folding. The FIFOs are checked against their
streams once, by `step_verify_partition`, at the bottleneck the partition reached (the
model `step_size_fifos` sizes by), and in a strategy step only where its strategy calls
the check. A FIFO that no longer keeps its producer within its idle time (a fold anywhere
may shorten the bottleneck, and a depth edited by hand may be too small) is refused
there, named, unless you held it: a held one is kept with a warning, and so is a
`direct` that throttles its producer. Packaging, the shell's integration and the
testbench take only the design `step_verify_partition` verified: a partition it never
verified is refused (run `step_verify_partition`), and so is one changed since (verify
again), so a list that makes an output lists `step_verify_partition` after its last
strategy step. Simulating or elaborating the partition only builds it, without the
check, so a simulation can run a FIFO the model calls too small. A required choice left
open (a FIFO's depth) is refused by name when the partition is built.

Each strategy step writes its report, `report/<step>.json`, listed under it in the
build's manifest: its strategy with the values it changed, reopened, kept and set by
their policy, attempts and time; every value with who set it (saved in the body, the
strategy, the strategy by the choice's policy, or the interim channel rule); the holds; whether it sized FIFOs and how its FIFOs were judged
(`fifos_judged`: `model` where it sized or checked them, else `not checked`; where it
checked, what the check found, `fifo_check`); per-member cycles and buffering, the
bottleneck, and the shell's resources against the part's. A point that uses more of a
resource than the part has is a warning naming that resource, never a refusal.
`step_verify_partition` writes its check's record, `report/step_verify_partition.json`
(`fifos_judged`, and `fifo_check`: the bottleneck period, the channels checked, those
the model does not read, and the warnings), and every value to `report/kernel_choices.json`,
by graph name (`{"MatMul_0_param0": {"transport": "direct"}, "MatMul_0": {"pe": 16}}`:
a kernel's choices under its node, a channel's under the tensor it carries, in graph
order: each node's input channels, the node, its output channels), a log FINN never
reads, which makes two builds easy to compare.

A build makes the partition and its outputs; checking them is for tests and for
utilities a user runs over a finished build's output directory, none of them a step.
`finn.builder.build_dataflow_verification.verify_partition_python(output_dir,
inputs_file)` executes the last model the build saved, the parent graph with its
partition running the KernelOps, and the model it saved before `step_convert_to_hw`
(`intermediate_models/`, so `save_intermediate_models` on), on each input of an `.npy`
file (one input a row), and compares the partition's outputs value for value; it
returns the graph comparison's result (`finn.analysis.comparison`: the runs, and a
`values-differ` finding for each output that differs). The packaged IP's testbench runs
with `finn.builder.build_dataflow_testbench.run_testbench` on its directory
(`stitched_ip/testbench` in the output directory), which raises `SimulationFailed` on a
mismatch. The partition's RTL compiles and elaborates in XSim when its body
(`partition/partition.onnx` in the output directory) is transformed by
`finn.transformation.kernels.package.ElaboratePartition`.

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
