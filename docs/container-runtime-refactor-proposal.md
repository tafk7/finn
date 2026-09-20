# FINN container and runtime refactor proposal

Status: direction approved; not a description of a completed implementation.
Execution is tracked in [the implementation plan](container-runtime-implementation-plan.md).
Date: 2026-09-20.

This consolidates the high-value runtime plan and the subsequent XSI investigation.
It replaces the whole-build Python worker as an architectural direction. Build
path/configuration/logging cleanup is owned by a separate private branch; this
work proceeds in parallel and integrates that implementation rather than
duplicating it or guessing its API. Earlier
implementation/validation records describe that earlier working-tree snapshot;
they are not release evidence for everything proposed here. Brainsmith and other
projects are reference material, not compatibility targets. No merge or push is
part of this proposal.

**1. Ownership and execution boundaries**

```text
Docker / native sbx / native host / Apptainer
    |
    +-- Runtime owns identity, mounts, storage, network and credentials
    |
    +-- Explicit installed or editable Python environment
            |
            +-- FINN build orchestration (current Python process)
                    |
                    +-- Installed package resources
                    +-- Explicit build paths and selected external data
                    |
                    +-- Vendor executable process
                    |       +-- argv, cwd, selected child environment
                    |
                    +-- XSI simulation session process
                            +-- selected environment before exec
                            +-- one kernel/design runtime per session
                            +-- complete testbench runs here
                            +-- bulk outputs, metrics and artifact paths
```

FINN owns package contents, code generation, tool invocation and native simulation
lifetimes. Container runtimes and sites own access to the machine. The build engine
does not infer which runtime launched it. No container mechanism chooses Python
imports or initializes FINN on every command.

**2. Image construction**

Keep one shared Dockerfile with a small set of independently useful targets:

```text
system + Python
    |
    +-- dependencies
            |   pinned Python dependencies and build requirements
            |   versioned HLS headers / board data
            |   no installed FINN application
            |
            +-- application
            |       installed FINN wheel + package entry points
            |       |
            |       +-- runtime additions, when selected (e.g. XRT)
            |
            +-- development uses dependencies directly
                    explicit writable environment prepared once

Native sbx integration is a thin variant of the selected image.
Application images can be exported as SIF artifacts.
```

The dependency build target can retain the existing `base` alias while gaining a
clear documented identity. Expose dependency and application artifacts through
existing Buildx/Bake and `docker/build` entry points. This does not require a new
launcher or image lifecycle tool. The application remains the normal installed
FINN artifact; the dependency image is the default preparation base for developers.

Proposed launcher behavior: `docker/build` and `docker/run` continue to select the
application by default. An explicit `--dependencies` option selects the dependency
artifact for development preparation; it never triggers editable installation.
Existing `--sbx` build/export and `--runtime` selections remain orthogonal. Dev
Containers select the dependency artifact in their configuration. These option
semantics are proposed, not present in the current implementation.

Build FINN into a wheel, then install that wheel in the application layer. Use the
same packaging path tested outside Docker. Keep source checkouts, test fixtures,
editable metadata and build outputs out of the application installation.

AMD's proprietary tool installation remains an explicit external mount, ordinarily
read-only. Network licences, device access and runtime packages are selected only
for workflows that need them. Development without FPGA tools needs no vendor mount
or licence configuration. Hardware runtime packages remain separate from toolchain
selection; selecting XRT does not select a Vivado/HLS installation.

**3. Image and application identity**

Dependency-image identity covers its OS/Python inputs, dependency pins, dependency
build recipes and external-data revisions. Editing FINN source does not change that
identity. Application identity additionally covers the FINN wheel/content digest.
Runtime additions and the sbx-specific variant participate in their own artifact
identity. Runtime mounts, credentials and licence values are not image inputs.

Record resolved dependency revisions and meaningful FINN version/source provenance.
Use the built image digest and wheel checksum for exact artifact identity. For an
editable environment, report the selected source path and revision/dirty state;
do not present its base image digest as the identity of edited application code.
The current single input manifest must be split by artifact responsibility.

**4. Package resources**

Retain the small `importlib.resources` helper and the migrated consumer calls.
Prefer a conventional package data layout under the Python source tree, with one
real package anchor for resource lookup. Remove the need for top-level resource
package mappings where that simplifies ordinary editable installation. FINN-owned
RTL, HLS/C++ support, Tcl, driver templates, licences and XSI sources belong in the
wheel and sdist; generated code, simulator binaries and test data do not.

FINN-owned data is resolved from the selected installation. Keep finn-hlslib and
board definitions explicitly versioned and external, with image defaults and
explicit overrides. Generated Tcl/shell files contain correctly quoted resolved
paths or deliberate build-owned copies. Resource lookup performs no extraction
with a lifetime shorter than its consumer and never writes into the installation.

This does not promise project or checkpoint relocation. Stable installation paths
remain an acceptable dependency of generated projects. Moving/replacing an
installation or changing incompatible code/tools requires regeneration.

**5. Installed and editable workflows**

An installed user runs the FINN application image directly. The package and its
entry points are already installed; no writable Python environment is necessary.
The workflow below is for developers editing FINN or its dependencies.

The development dependency image supplies Python/system prerequisites, an offline
wheelhouse at `/opt/finn/wheels`, and a fully resolved development dependency
manifest at `/opt/finn/development-requirements.txt`. Include pip, setuptools,
wheel and any selected editable dependency's build requirements. The manifest
excludes FINN itself. These paths define the proposed image contract; they are not
available in the current images yet. The application image can install from the
same dependency build stage without retaining the development wheelhouse.

The default development environment is an isolated writable venv populated from
those wheels, followed by explicit editable installs. This avoids inheriting a
second FINN installation and makes selecting an editable dependency a normal
replacement within that one environment. A system-site-packages overlay remains
an explicit alternative, not the main example.

```text
Prepared dependency image              Writable development state
    Python + system prerequisites         venv + installed metadata
    pinned wheels/build requirements      explicit editable links
             |                                      |
             +---------- one-time preparation ------+
                                                    |
                                                    v
                                     selected mounted checkout(s)
```

There are two independent locations: the source checkout and the venv. Source may
be shared across environments; a venv belongs to one Python/platform/dependency
configuration. Native, Docker and sbx development get different environments.
Editable links and script shebangs contain runtime paths; keep those paths stable
while reusing the environment. Do not use a container venv as a host venv.

For ordinary Docker, `docker/run` currently uses `compose run --rm`, so environment
state must survive removal of that disposable container. Use a dedicated writable
bind mount or explicitly prepared named volume, separate from the source checkout.
For the bind-mount example, prepare the host directory as the invoking user and
mount it at `/env`, with the selected FINN checkout at `/workspace/finn`.

The following are illustrative commands for the proposed dependency image:

```bash
# Host setup; DEPS_IMAGE is the prepared dependency-image reference.
CHECKOUT=/absolute/path/to/finn
ENV_DIR="$HOME/.local/share/finn/dev-envs/docker-main"
mkdir -p "$ENV_DIR"

# Prepare once. The container uses the host UID/GID to own writable files.
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$CHECKOUT,dst=/workspace/finn" \
  --mount "type=bind,src=$ENV_DIR,dst=/env" \
  "$DEPS_IMAGE" bash -e -c '
    python -m venv /env/venv
    /env/venv/bin/python -m pip install --no-index \
      --find-links /opt/finn/wheels -r /opt/finn/development-requirements.txt
    /env/venv/bin/python -m pip install --use-pep517 --no-index \
      --no-deps --no-build-isolation -e /workspace/finn
  '

# A later disposable container reuses the selected environment.
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$CHECKOUT,dst=/workspace/finn" \
  --mount "type=bind,src=$ENV_DIR,dst=/env" \
  "$DEPS_IMAGE" /env/venv/bin/python -m finn.util.installation finn qonnx
```

The same mounted venv supplies `/env/venv/bin/build_dataflow` and
`/env/venv/bin/python -m pytest`. Starting another container does not rerun pip.
A named volume is equally valid, but its ownership must be established explicitly
at preparation; do not rely on startup chown/repair. A persistent Docker container
may keep its venv in its writable layer instead, with container-lifetime persistence.

For FINN-plus-QONNX development, explicitly mount the selected QONNX checkout at a
stable location such as `/workspace/qonnx`, then run the same venv's pip with
`--use-pep517 --no-index --no-deps --no-build-isolation -e /workspace/qonnx` once.
Its required build tools must already be in the prepared set. Additional packages
follow the same rule. Nearby directories are never discovered automatically.

Dev Containers use the dependency image and an environment-specific writable
location. Creation-time preparation populates the venv; the editor's selected
interpreter and terminal environment point to it. Reopening a prepared container
just selects that environment. Do not share `.venv` silently with native-host work.

For native sbx, create a sandbox from the dependency-image sbx variant through the
existing native environment files. Keep the configuration outside mounted
workspaces and retain the same arguments/files for native lifecycle commands.
The workspace is mounted at its native absolute path; do not assume Docker's
`/workspace/finn` alias exists. The default venv lives in sandbox-private writable
home, for example `$HOME/.venvs/finn-dev`.

After creation, perform one explicit preparation operation:

```bash
# ARGS/FILES are the same native environment arguments and copied configuration
# used for creation; CHECKOUT is the absolute mounted workspace path.
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- bash -e -c '
  VENV="$HOME/.venvs/finn-dev"
  python -m venv "$VENV"
  "$VENV/bin/python" -m pip install --no-index \
    --find-links /opt/finn/wheels -r /opt/finn/development-requirements.txt
  "$VENV/bin/python" -m pip install --use-pep517 --no-index \
    --no-deps --no-build-isolation -e "$1"
' finn-prepare "$CHECKOUT"

# Later commands only execute the prepared interpreter.
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- bash -c '
  exec "$HOME/.venvs/finn-dev/bin/python" -m finn.util.installation finn qonnx
'
```

Each exec gets a fresh process, but uses the same sandbox filesystem. The venv
survives ordinary exec sessions. Treat it as sandbox-lifetime state; prepare again
after deleting/recreating the sandbox. Do not promise persistence across template
replacement. If persistence across recreation is needed, explicitly mount a
separate environment directory through native configuration at a stable path,
and apply the same interpreter/image-compatibility rules as Docker. Selecting
editable QONNX requires that checkout to be explicitly accessible in the sandbox.

For interactive shells or coding agents, select the prepared interpreter in editor
configuration and put its bin directory first in PATH using the native runtime
environment configuration. Supplying PATH/interpreter selection performs no
installation and does not depend on shell startup scripts. The explicit venv
executable examples above also work when interactive shell activation is absent.

Existing Python/resource edits become visible through the selected editable
installation. Reinstall after relevant dependency/package-metadata or entry-point
changes; rebuild native components after relevant ABI/source/toolchain changes.
Recreate the venv when its base Python/platform/dependency set changes. A source edit
alone does not require an image rebuild. The same source can be used by independent
venvs selecting different dependency revisions.

Native host setup follows the same explicit preparation and environment-selection
model. Vendor selection is separate from Python installation and is applied by
operations. Keep the installation inspection command to verify actual versions,
import locations and editable records in every runtime.

**6. Build engine: owned by the private branch, integrated here**

The user is already implementing build path/configuration/logging cleanup in a
separate private branch. Do not duplicate that implementation, seek inaccessible
code, or invent its API. This lane should maximize work independent of that branch:
package resources, images/identity, development environments, scoped process/tool
execution and an independently exercisable XSI session implementation.

The integration outcome remains direct build orchestration with explicit paths,
configuration and logging, without parent cwd/environment/stdout mutation. The
whole-build Python worker from the earlier working-tree changes is slated for
removal/reconciliation when the private branch is integrated. It is not a required
interface for any new component, and this lane should not expand it or encode it
as the permanent expected behavior in acceptance tests.

Keep new resource/process interfaces small and testable directly. Preserve artifact
layouts, checkpoints and current allocation policy. Pass selection descriptions to
existing workers rather than capturing process-global environments. The concrete
plumbing into the private branch is an integration task once its real interfaces
are available; no speculative public build-context or workspace framework is
needed in advance.

At integration, review the actual branch interfaces, adapt the narrow resource/tool/
simulation call sites, remove obsolete compatibility and whole-build activation,
and test both direct and directory-based builds. Include custom steps, error/logging
behavior, checkpoint reuse under existing path constraints and the supported
concurrency cases. Do not infer thread-safety for arbitrary user callbacks.

**7. A single narrow vendor execution implementation**

Extend the current internal toolchain module rather than adding another EDA
framework. Selection identifies the intended installation/frontend separately from
routing. Support explicitly selected settings scripts, an explicitly accepted
already-configured base environment, and site-owned execution routes.

Local settings are sourced by child Bash with positional arguments, controlled
startup inputs (including BASH_ENV), and NUL-delimited environment capture. Use a
defined base environment; never attempt to unsource a previous installation.
Snapshot mappings defensively and copy them for per-operation overrides.

All FINN-owned Vivado, HLS, Vitis linker and xelab invocations migrate to this
implementation. C++ simulation compilation also uses explicit compiler/header/
library inputs and the shared process execution primitive. Remove shell command
construction where argv is available. Preserve useful generated replay scripts,
while documenting their selected-environment requirement.

Probe versions/capabilities on demand through the actual route, with timeouts.
Frontend selection is explicit, executable availability is separate from FINN
code-generation compatibility, and no successful banner is treated as licence or
synthesis validation. Preserve command-directory overrides when applying a site
launcher, and avoid substituting local absolute executable paths into remote jobs.

Log command, cwd, route, selected identity, duration and result, retaining useful
output and failures. Never log or persist captured environments or credentials.
Serialize selection descriptions to workers rather than secret-bearing prepared
environments. Define timeout/cancellation for local process groups. Sites own
remote cancellation; FINN does not gain scheduler semantics.

**8. XSI owns a simulation-session boundary**

Keep compilation/discovery as ordinary operations. Pure Python compilation helpers
must not require importing a native bridge merely to launch xelab. Load the bridge
and AMD libraries only inside the simulation session that uses them.

Use the existing standalone C++ harness for the workloads it already supports.
For functional simulation and Python-driven testbenches, start an internal Python
session process with its selected loader environment supplied at exec. Keep the
Python driver, native kernel, compiled design and all per-cycle work together.
A session can include multiple frames, resets, initialization, readback and
characterization for one selected design/runtime. Do not create one process per
clock tick or forward port access through IPC.

Inputs describe the selected design, buffers/arrays, stream and clock configuration,
AXI initialization, tracing and execution limits. Results contain outputs, cycle
counts, requested readbacks/traces, status and artifact paths. Use simple explicit
job data and existing files first; optimize bulk transport only from measurements.
Native handles never escape the worker. Custom testbench code executes within the
session with explicit input/result data; arbitrary parent closures are not silently
serialized. Shape this interface for FINN's own workloads, with no Brainsmith
compatibility requirement.

Close designs deterministically on normal and exception paths. Enforce an external
wall-clock deadline as well as simulation-cycle watchdogs. Report crashes and
partial trace artifacts explicitly. Process exit supplies native lifetime
separation, without relying on complete unloading of globally linked libraries.
Do not use a fork of an interpreter that already loaded the vendor runtime.

Check bridge/design compatibility before reuse (source, relevant ABI, selected
headers/tool identity and compile inputs). This can use small adjacent records
without redesigning cache layouts or checkpoint schemas. File existence alone is
not sufficient. Introduce no resident service, generic worker pool or RPC framework.
Measure startup and transport costs before considering reuse beyond one session.
A site-routed session must use that site's valid Python/driver command and paths;
a local absolute sys.executable is not assumed valid remotely.

**9. Reduce container startup to runtime responsibilities**

The generic image entrypoint handles only justified non-root/arbitrary-UID home
requirements and process execution. Use runtime-private writable locations; it
performs no pip operation, source discovery, FINN build-directory allocation,
vendor activation or checkout-derived Tcl copying. Runtime execution that bypasses
the entrypoint must still support installed FINN operations.

Make site Tcl configuration explicit through mounts or a preparation action. Keep
credentials and network configuration site-owned. Native sbx's persistent shell
environment remains integrated only in its image variant. Generic Docker/native
execution does not acquire an unnecessary FINN BASH_ENV hook.

Remove global vendor LD_LIBRARY_PATH/LD_PRELOAD settings after required workarounds
are validated as tool/session-specific child inputs. This includes testing the
existing FLEXlm/libudev workaround on actual affected operations; ordinary Python
and unrelated commands should not inherit it.

Drop transparent bare-tool interception. Interactive users can deliberately source
vendor settings or use their site's wrapper. FINN's documented application commands
work through explicit internal execution regardless of shell activation. This
removes the recursion and process-tree activation-latch machinery.

**10. Concrete cleanup destinations**

| Existing mechanism | Proposed end state |
| --- | --- |
| `docker/finn_paths.py`, `docker/finn-live.pth` | Deleted; installed/editable package metadata chooses imports. |
| Handwritten `docker/build_dataflow` | Deleted; package-generated entry point. |
| Whole-build child in `build_dataflow_directory` | Removed; direct path/configuration fixes. |
| `docker/toolchain-shim` | Deleted after all FINN tool callers migrate; bare-tool usage documented explicitly. |
| `docker/finn-toolchain.sh` and `FINN_ENV_APPLIED` | Deleted after scoped tool/session environments replace consumers. |
| `docker/finn-bashenv.sh` | Generic FINN hook removed; preserve only native sbx's required integration in its variant. |
| `docker/finn_entrypoint.sh` | Small UID/home/exec responsibility, or deleted if equivalent runtime behavior is verified. |
| Dockerfile global loader workarounds | Scoped to affected vendor/session children, validated with real operations. |
| Broad `_legacy_build_env` behavior | Eliminate internal reliance. Optional old-input parsing stays at the outer API boundary only where justified; no execution/activation layer. |
| One shared image-input hash | Separate dependency and application content identities. |

Temporary migration code must have an actual consumer and deletion condition. It
must not become the end-state architecture or a way to avoid fixing owned code.
No public legacy wrapper command is introduced.

**11. Runtime-specific behavior**

Docker/Compose retain their existing execution surface. The host configuration
resolver determines explicit mounts, UID/GID, storage, selected tools, platform
repos and licences; it does not select Python code or establish a fake checkout.
Optional discovery remains host-side and never overrides explicit user selection.
Preserve current mount policies where required by existing absolute artifacts;
do not claim that source-path mirroring alone makes projects portable.

Native sbx keeps native composition, approval and lifecycle. Supply a dependency
or application template plus examples for workspace/storage and optional FPGA/site
access. No FINN sandbox runner, launcher plugin or Cardinal-specific integration
is introduced. FINN owns no credential registration or machine network policy.

Native host setup uses the same package and execution contracts. Setup prepares
software once; activation selects the environment. Correct the current native XSI
setup regression: `--check` checks prerequisites and must not be treated as proof
that an extension has been built.

Ordinary SIF execution remains read-only and uses the installed application.
Editable HPC development requires deliberately writable storage/environment and
prepared dependencies. Scratch is a site-selected writable location. The caller
selects required mounts, network and credentials using native runtime mechanisms.

**12. Storage and reproducibility limits**

Retain current output, scratch and checkpoint layouts. Allocate scratch on demand
through the operation that needs it, using an explicit root and the existing
allocator. Mount writable output/cache locations deliberately. Installed packages
and external vendor/data inputs remain read-only during normal execution.

Saved Tcl, HDL memory references, native binaries and model metadata may retain
absolute paths. Reuse requires those paths and compatible installations. Changing
FINN, Python, source inputs or tools can require regeneration. Neither the image
nor an environment adapter repairs stale artifacts or makes exports self-contained.
Build relocation/export redesign is outside this work.

**13. Parallel delivery and integration**

```text
Private branch                         This lane
    build paths/config/logging             package resources and installs
    build-engine cleanup                   dependency/application images
            |                              Docker/sbx development preparation
            |                              scoped tool/process execution
            |                              XSI session implementation and tests
            |                                          |
            +---------------- integration -------------+
                                  |
                                  +-- connect to actual build interfaces
                                  +-- remove whole-build worker / old activation
                                  +-- remove now-redundant container hooks
                                  +-- complete runtime/vendor validation
```

This lane can deliver packaging and image/development-environment changes without
waiting for the private branch. It can also implement/test scoped vendor execution
and simulation-session startup directly, keeping changes to overlapping build
entry points and configuration plumbing minimal. Migrate non-overlapping callers
where their interfaces are already known. Caller adaptation touching the private
branch's owned code is deferred to integration rather than implemented twice.

The XSI work still has its own installation-backed correctness and performance
gate. Private-branch availability is not a reason to delay independent simulation
research, and merging the private branch is not evidence that XSI isolation works.
Remove individual obsolete hooks once their actual consumers have replacements;
remove broad build activation only after integrating the build-engine changes.

Acceptance must include wheel-only operation with the checkout unavailable,
wheel-from-sdist parity, unrelated cwd, resource/code edits in selected editable
installs, FINN-plus-QONNX development, independent environments, and no startup
installation/repair. Exercise package-generated entry points and generated recipes.

Vendor acceptance is distinct: selected-route probes, real synthesis/linking,
licence use, real XSI outputs/cycle counts/waveforms, failures and cleanup,
sequential/concurrent sessions, and each claimed supported tool release. Test
Docker with arbitrary UID, native setup, native sbx and SIF/HPC where available.
Synthetic tests establish mechanisms, not vendor flow support.

**14. Current working-tree status**

Keep the resource packaging foundation, migrated resource consumers, explicit
installation work, package entry points, inspection command, scoped process/tool
helper, and relevant tests. The prior 157-test and Docker/sbx record remains useful
baseline evidence, not proof of this revised design.

Rework the resource layout where it simplifies packaging, dependency/application
identity separation, development image selection and remaining direct vendor
consumers. Build cwd/configuration/logging handling belongs to the private branch.
Integrate it to remove the whole-build Python worker and obsolete build activation. Fix the known XSI setup check. XSI session isolation and complete
removal of vendor startup/loader hooks are not implemented or vendor-validated yet.
The Linux loader probes support the architectural choice, while actual AMD tests
remain necessary. Packaging/tool improvements can be delivered separately without
calling the unfinished XSI/container cleanup complete.
