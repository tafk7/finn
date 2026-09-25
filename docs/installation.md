# Installing and developing FINN

FINN is a Python package. The Python side is managed with standard tools
(`pyproject.toml` and `uv.lock`); the containers add what pip cannot provide, such
as the OS libraries Xilinx tools need and XRT. The design, and the reasoning behind
it, is in [environment.md](environment.md).

## Use FINN

```bash
pip install finn          # Python-only use (Python 3.10-3.12, Linux x86-64)
pip install "finn[hw]"    # plus finn-hlslib, for HLS simulation and synthesis
```

Hardware flows additionally need Vivado/Vitis (selected with `FINN_XILINX_PATH`
and `FINN_XILINX_VERSION`, or by sourcing AMD's `settings64.sh`) and a licence.
Board files are fetched from their upstream repositories on first use and cached
(see [external build data](#external-build-data)). The `finn_xsi` simulation
extension is built against your Vivado on first use.

## Develop FINN

Every modality uses the same lock and ends up with the same thing: an active venv
with FINN and its workspace members (`packages/*`, currently finn-hlslib) installed
editable, and every other dependency at its locked version.

**Native host:**

```bash
git clone --recurse-submodules https://github.com/Xilinx/finn.git && cd finn
./setup-local.sh              # checks, then `uv sync` into .venv, optional XSI build
source scripts/activate.sh    # activates .venv and the Xilinx toolchain, if configured
```

`setup-local.sh` wraps `git submodule update --init && uv sync`; that is all a
Python-only developer needs. After pulling a change to `uv.lock` or a submodule,
run `git submodule update --init && uv sync` again.

**Docker:**

```bash
./docker/run                           # shell; /opt/venv is active
./docker/run -- pytest -m util
./docker/run --fpga -- vivado -version # with FINN_XILINX_PATH/VERSION set
```

The image holds the locked dependencies in `/opt/venv`, active for every process.
When a container starts, its entrypoint installs the mounted checkout editable
(`uv sync --frozen --inexact` against `FINN_ROOT`, about a second with a warm
cache). If `uv.lock` has changed since the image was built, the start installs the
difference, or `docker/run` builds a new image, since the lock is an image input.
Editing FINN never requires a new image.

**Dev Container:** open the repository in VS Code and reopen in the container. The
interpreter is `/opt/venv/bin/python`; there is no setup step.

**sbx:** see [the sbx guide](../docker/sbx/README.md). The same entrypoint installs
the workspace checkout when the sandbox starts.

`docker exec` and `sbx exec` do not wait for the entrypoint. A script that execs
into a container it has just started can wait for `/tmp/finn-ready`.

## Co-develop QONNX, Brevitas or another dependency

Point the dependency's source at your checkout, locally (do not commit this):

```toml
# pyproject.toml
[tool.uv.sources]
qonnx = { path = "../qonnx", editable = true }
```

Then run `uv sync` (natively, or inside a running container). In Docker, mount the
checkout at the same relative place, e.g. `./docker/run --volume "$PWD/../qonnx:$PWD/../qonnx"`.
uv also updates `uv.lock`; do not commit either change.

## Add a dependency

Everything goes through `pyproject.toml` and `uv.lock`; no Dockerfile or setup
script changes.

* **Python package:** add it to `[project] dependencies` (or a dependency group),
  with a `[tool.uv.sources]` git entry if it is unreleased, then `uv lock`.
* **Build data** (headers, board files, Tcl libraries): wrap it as a data-only
  Python package and treat it as above; register its lookup in
  `finn.util.external`.
* **Developed in lockstep with FINN** (always editable for everyone): add its
  repository as a git submodule under `packages/`, give it a `pyproject.toml`
  there if upstream has none (see `packages/finn-hlslib`), and mark it
  `{ workspace = true }` in `[tool.uv.sources]`. Each member must also be
  released to PyPI for `pip install finn` users.

## Inspect an environment

```bash
python -m finn.util.installation finn qonnx brevitas finn-hlslib
```

This reports import locations, versions and installation metadata. FINN wheel
provenance preserves the Git revision and dirty state through its sdist.

## Package resources and external build data

`finn.util.resources.resource_path(family, *parts)` resolves stable read-only
paths in `rtllib`, `custom_hls`, `xsi`, or `qnn-data`, under `finn._data`
(`src/finn/_data/` in a checkout). The Python XSI driver lives at `src/finn_xsi/`.
Editable installations observe changes to these directly. Generated RTL, driver
files and compiled XSI extensions belong in writable build storage, never in the
installed distribution.

### External build data

* **finn-hlslib** is the `finn-hlslib` package (`finn[hw]`), a workspace member
  wrapping the upstream repository as a submodule at its pinned commit.
* **Board files** come from third-party repositories that FINN does not
  redistribute. `finn.util.external` fetches the pinned commits on first use
  (sparse, single-commit fetches), verifies a content digest and caches the result
  under `${XDG_CACHE_HOME:-~/.cache}/finn`. Images and `setup-local.sh` (with
  Vivado) fetch them ahead of time; `python -m finn.util.external fetch-boards`
  does so explicitly.

`FINN_HLSLIB_PATH` and `FINN_BOARD_FILES_PATH` override either location.

Saved projects contain absolute installed-resource and intermediate-artifact
paths. Keep the selected installation, external data and build directories in
place. Moving directories, replacing/upgrading FINN or changing toolchains can
invalidate generated projects, compiled simulations and checkpoints. Regenerate
those artifacts rather than assuming a checkpoint is a self-contained export.

## RTL simulation (finn_xsi)

RTL simulation builds the `finn_xsi` extension against the selected Vivado the
first time it is needed, into `$FINN_BUILD_DIR/finn_xsi/<abi>-<key>` (one build
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
