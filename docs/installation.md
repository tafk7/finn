# Installed and editable FINN

FINN resources ship in its wheel. Normal imports and resource generation need
neither a checkout nor `FINN_ROOT`. The reference image/native setup uses Ubuntu
22.04 and Python 3.10; the package metadata permits Python 3.9+. Installations must
be unpacked, as pip normally installs them.

## Application and dependency artifacts

`./docker/build` builds the installed application. `./docker/build --dependencies`
builds the development dependency artifact. Both support `--runtime NAME`, `--sbx`
and SIF export. `--print-tag` resolves the tag through Bake without building.
Application tags include FINN source/resource content; dependency tags exclude it.
The actual image ID and application wheel checksum identify the built artifacts.
Changing a mounted checkout does not change the installed application.

The dependency artifact provides `/opt/finn/wheels`, a checksum-pinned
`/opt/finn/development-requirements.txt` and `/opt/finn/wheelhouse.json`, with exact
Git dependency revisions and wheel checksums. The manifest contains no FINN.
It includes pip, setuptools, wheel, build and the supported editable backends.
The application installs the same resolved wheels but does not retain the wheelhouse.
The pinned finn-experimental wheel is metadata-only due to its upstream package
configuration; select its checkout explicitly if you need its Python modules.

## Select editable checkouts with one command

In a prepared, writable Python environment, run:

```bash
./scripts/prepare-editables
```

The command uses the Python selected on PATH and reads `editable-requirements.txt`
from the current directory. The checked-in file selects FINN; add any other local
projects using ordinary pip requirements syntax:

```text
-e .
-e ../qonnx
```

Paths resolve relative to the requirements file. Quote paths containing spaces.
An optional filename selects a different list: `./scripts/prepare-editables my-editables.txt`.
The helper applies the pip options for editable builds using the environment's
existing build tools, disables package indexes and dependency installation, and
runs `pip check`. It prevents pip configuration from redirecting installation to
a different environment or user site. An incompatible dependency baseline causes
a failure; the operation is not transactional and may already have installed the
selected editables before the compatibility check fails.

The same command works in native, Docker and sbx environments. Container paths
must refer to mounted checkouts, and the selected Python environment must be
writable. Each of three FINN/QONNX pairs gets its own environment and requirements
file. Code edits need no preparation; rerun after changing the selected checkouts,
package metadata or entry points. Removing an entry does not restore its baked
package; recreate the environment to return to the baseline. Ordinary startup
never runs this command.

## Docker development: prepare once, reuse after container removal

Create a separate directory as the same UID/GID that will run the container:

```bash
CHECKOUT=/absolute/path/to/finn
ENV_DIR="$HOME/.venvs/finn-docker-deps"
mkdir -p "$ENV_DIR"
cd "$CHECKOUT"
./docker/build --dependencies
./docker/run --dependencies --venv "$ENV_DIR" -- bash -c '
  python -m venv /env/venv &&
  /env/venv/bin/python -m pip install --no-index --find-links /opt/finn/wheels \
    -r /opt/finn/development-requirements.txt &&
  /env/venv/bin/python "$1/scripts/prepare-editables" "$1/editable-requirements.txt"
' finn-prepare "$CHECKOUT"
# The first container was removed. Reuse both mounts, with no installation:
./docker/run --dependencies --venv "$ENV_DIR" -- /env/venv/bin/python -m finn.util.installation
./docker/run --dependencies --venv "$ENV_DIR" -- /env/venv/bin/build_dataflow --help
```

The launcher mirrors the explicit checkout's absolute path. `--venv` only mounts
an already-created host directory at `/env/venv`; it never installs or repairs
ownership. Keep this directory separate from a native-host `.venv`. Use `--volume`
for explicitly selected additional source mounts, including paths containing spaces.
For a network-disabled validation run, use a Compose network override or direct
`docker run --network none` with the same source/environment mounts.

Add QONNX to `editable-requirements.txt`, then prepare this same environment with
the checkout mounted at its stable path:

```bash
QONNX=/absolute/path/to/qonnx
./docker/run --dependencies --venv "$ENV_DIR" --volume "$QONNX:$QONNX" -- \
  /env/venv/bin/python "$CHECKOUT/scripts/prepare-editables" "$CHECKOUT/editable-requirements.txt"
```

The same requirements file can select Brevitas and finn-experimental checkouts.
Their build requirements are prepared already. Repeat every additional source
mount on later runs; editable dependency metadata can remain present even when
its source is unmounted and cannot be imported. `fetch-repos.sh` fetches the pinned
sources in `deps.env`; fetching and mounting never select Python imports.

## Dev Container, sbx and native host

The Dev Container resolves/reuses its dependency image through Bake, prepares
`/home/agent/.venvs/finn-dev` during creation, and selects that interpreter for
editor and terminal commands. Reopening runs no pip operations. Rebuilding the
container recreates its private environment. It never shares a host-native `.venv`.

Native sbx uses the dependency sbx variant, native environment files stored outside
all mounted workspaces, and `$HOME/.venvs/finn-dev` inside the sandbox. See the
[concrete native preparation and repeated-exec commands](../docker/sbx/README.md).
Sandbox deletion deletes that venv unless you explicitly mount separate persistent
storage. Native lifecycle, networking, credentials and editor/agent settings remain
site-owned; no FINN installer runs from a shell hook.

`setup-local.sh` creates a native isolated venv (`FINN_VENV`, default `.venv`),
installs the reference requirements and explicitly selected checkouts, and builds
XSI when requested and available. Normal XSI setup checks an existing extension or
builds a missing one and verifies it; `python -m finn.xsi.setup --check` checks only
prerequisites. `scripts/activate.sh` selects the environment and existing legacy
vendor shell setup without installing or creating scratch directories.

Native/offline preparation uses ordinary package commands, with a wheelhouse and
manifest built for that host's Python/platform (container wheels are Python 3.10
Linux x86-64):

```bash
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$MANIFEST"
"$VENV/bin/python" "$CHECKOUT/scripts/prepare-editables" "$CHECKOUT/editable-requirements.txt"
```

SIF applications run from the installed read-only image. Editable use needs a
separately prepared writable environment, source and scratch mounts at stable
paths; SIF execution has not been validated here.

## Environment lifetime and inspection

Keep the image/platform, Python minor version, source mount paths and venv mount
path stable. Recreate the venv after incompatible dependency/image changes. Each
venv has its own explicit editable selection: opening another checkout does not
select it. Atomic code/resource edits are visible; reinstall after metadata or
entry-point changes and rebuild native extensions after source, ABI or tool changes.
There are no import modes, automatic installers or source-discovery hooks.

```bash
/path/to/venv/bin/python -m finn.util.installation finn qonnx brevitas finn-experimental
```

This reports selected import locations, distribution versions and installation
metadata. FINN wheel provenance preserves Git revision/dirty state through its
sdist. Archive-only builds report unknown revision. `VERSION` is the package
version. Preserve image/wheel digests to distinguish immutable artifacts from
subsequently edited application code.

## Package resources and external build data

`finn.util.resources.resource_path(family, *parts)` resolves stable read-only
paths in `rtllib`, `custom_hls`, `xsi`, or `qnn-data`. Their conventional package
anchor is `finn._data`, under `src/finn/_data/` in a checkout. The Python XSI driver
lives at `src/finn_xsi/`; it keeps its existing import name. Editable installations
observe changes directly in these source/resource directories.
Test vectors, testbenches, generated binaries and compilation caches are excluded.
Generated RTL, driver files and compiled XSI extensions belong in writable build
storage, never in the installed distribution.

`finn-hlslib` and board definitions remain external, versioned inputs. Their pins
and the assembled board-data checksum are in `deps.env`. Set
`FINN_HLSLIB_PATH=/absolute/path/to/finn-hlslib` and
`FINN_BOARD_FILES_PATH=/absolute/path/to/board_files`. Reference images install
these at `/opt/finn-src/deps/finn-hlslib` and `/opt/finn-src/deps/board_files`.
Legacy checkout builds may still use `FINN_ROOT/deps/...` through the internal
adapter. Generated HLS/Tcl references resolve these inputs when generated.

Saved projects contain absolute installed-resource and intermediate-artifact
paths. Keep the selected installation, external data and build directories in
place. Moving directories, replacing/upgrading FINN or changing toolchains can
invalidate generated projects, compiled simulations and checkpoints. Regenerate
those artifacts rather than assuming a checkpoint is a self-contained export.
Resource packaging is not a relocation guarantee.

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

The internal `_toolchain.Selection` supports explicit local settings scripts or
an explicitly accepted configured environment, a site command directory, and a
launcher argv prefix. `Selection.prepare()` snapshots the mapping. With local
settings and no base mapping, it uses system PATH plus HOME/user/locale/temp/display
and licence inputs. To accept additional base variables, pass a mapping explicitly;
FINN does not attempt to unsource a previously activated installation. Bash startup
hooks and exported functions are removed in children.

HLS and stitched-IP Vivado commands execute with argv, child env and cwd. Probes
use the identical route and a bounded timeout. `CallHLS(toolchain=...)` and
`CreateStitchedIP(..., toolchain=...)` accept prepared internal selections.
`FINN_TOOL_DIR_OVERRIDE` continues to select site wrappers; launcher prefixes
preserve those names and do not substitute local absolute vendor executables.
The site owns remote activation, path visibility and remote cancellation.

Choose `FINN_HLS_FRONTEND=vivado_hls`, `vitis_hls`, or `vitis-run` for legacy entry
points; explicit selections name the frontend directly. Compatibility checks
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

Bridge/design compilation now writes adjacent `.finn.json` records of tool identity,
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
