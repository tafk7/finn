# Installed and editable FINN

FINN resources ship in the distribution. Normal imports, RTL code generation and
Python driver generation need neither a checkout nor `FINN_ROOT`. Python 3.9+
and an unpacked installation are required. Use pip 24.3+ and setuptools (64+
for editable installs). With older pip, pass `--use-pep517` to ensure the
editable backend is used. A dependency-image venv uses `--without-pip` to inherit
the prepared pip/setuptools instead of shadowing them with ensurepip's older copies. The repository's `requirements.txt`, `deps.env` and
Docker constraints define the reference dependency set; model-zoo tests also
need the selected Brevitas/torchvision packages.

## Install or develop

In a prepared virtual environment, install an application wheel or checkout:

```bash
python -m pip install /path/to/finn.whl
# Or build/install the selected source:
python -m pip install /absolute/path/to/finn
```

For FINN plus QONNX development, prepare once, then activate as often as needed:

```bash
python3 -m venv /path/to/finn-dev
source /path/to/finn-dev/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e /absolute/path/to/qonnx -e /absolute/path/to/finn
```

Select each additional co-developed package explicitly, for example
`python -m pip install -e /path/to/brevitas -e /path/to/finn-experimental`.
`fetch-repos.sh` prepares pinned checkouts but does not select imports. Use the
QONNX and other revisions in `deps.env` for the reference configuration. Opening
another checkout or setting `FINN_ROOT` never selects its Python code. Each venv
has its own package selection; normal startup performs no install or directory
repair. `FINN_DEPS` and `docker/run --deps` have been removed.

Offline, first provision dependencies and build requirements from a prepared
wheelhouse (including the selected QONNX/Brevitas wheels). Then use:

```bash
python -m pip install --no-index --find-links /wheelhouse -r prepared-requirements.txt
python -m pip install --use-pep517 --no-index --no-deps --no-build-isolation -e /path/to/finn
# The same flags apply to explicitly selected dependency checkouts, provided
# their own build requirements are already installed.
```

`setup-local.sh` remains the full native preparation workflow. Its activation
script selects that environment and explicitly configures legacy native tooling;
it does not install packages. For ordinary Python work, sourcing the venv's
`bin/activate` is sufficient.

Inspect the actual environment from any directory:

```bash
python -m finn.util.installation finn qonnx brevitas finn-experimental
```

This prints distribution versions, selected import locations and pip's installation
record. Wheel provenance includes the source revision and dirty flag when Git is
available, preserved through the sdist; archive-only builds report unknown revision.
`VERSION` is the release version. A dirty flag is not a content hash; preserve the
wheel checksum or application image digest for exact identity.

## Package resources and external build data

`finn.util.resources.resource_path(family, *parts)` resolves stable read-only
paths in `rtllib`, `custom_hls`, `xsi`, or `qnn-data`. These use private resource
packages, retaining the existing checkout layout for editable RTL development.
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

`build_dataflow project/` remains the ordinary public build command. It runs the
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

## Docker, sbx and HPC

The Dockerfile `base` target is a reusable dependency environment; `application`
and the public runtime/sbx targets install the selected FINN distribution.
Application image identity now includes FINN source/resources. Mounted checkouts
are available for work but do not shadow installed packages. For development,
create a writable venv with `--system-site-packages` and explicitly install the
selected checkout with `--no-deps --no-build-isolation -e` once. The Dev Container
performs that preparation in `postCreateCommand` and selects `.venv/bin/python`.

Use the same preparation inside a native sbx environment, then invoke the prepared
venv's Python/console scripts in subsequent `sbx env exec` calls. Composition,
mounts, credentials, network and lifecycle remain native/site-owned. No FINN
sandbox launcher or Cardinal integration is introduced.

Ordinary `apptainer exec finn.sif python ...` uses installed code in the read-only
image. Editable HPC development requires a deliberately writable venv/overlay
with prepared dependencies; do not install on every invocation or modify the SIF.
Keep its interpreter and all referenced installations available for saved projects.

The sbx Bash hook only reads sbx's persistent environment. Bare vendor commands
retain the tool shim as an interactive convenience for callers not yet migrated.
Global image loader workarounds remain pending actual licensed/native simulation
validation. Optional site Tcl initialization is deliberate: pass
`FINN_SITE_TCL_DIR` containing `HLS_init.tcl` and/or `Vivado/Vivado_init.tcl` to
copy those scripts into the writable home at container entry.

When overlaying an installed image with a `--system-site-packages` venv, use
`--config-settings editable_mode=strict` for FINN as shown above. This puts the
selected editable tree ahead of the inherited wheel (the default setuptools
finder may otherwise lose to that wheel). Existing code/resource edits are live;
after adding/removing files, reinstall to refresh the strict editable link tree.
Keep the checkout and its `build/__editable__*` directory available.
