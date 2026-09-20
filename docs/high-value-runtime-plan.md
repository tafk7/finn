# FINN: high-value packaging and toolchain improvements

Status: proposed; this document does not implement the changes.
Date: 2026-09-16.
Working branch: `experiment/container-runtime-boundaries`.

This is the proposed replacement for `runtime-information-redesign-plan.md`.
Retain that document and the runtime-information inventory as historical design
inputs. The scope here is deliberately smaller: independently useful improvements
for FINN consumers and active developers, followed by removal of proven redundancy.
Do not merge or push as part of implementing this plan without a separate request.

## Outcomes that matter

1. Installing FINN installs the resources its supported operations need. Normal
   imports and resource access work without a source checkout or FINN_ROOT.
2. Developers explicitly install FINN and selected dependencies editably. The
   imported code and package metadata describe the same installation.
3. Relevant vendor operations select and execute tools predictably without changing
   the parent process environment or working directory.
4. Remaining legacy requirements live in a small internal compatibility module,
   with named consumers and deletion conditions. Ordinary users do not learn a new
   compatibility command.

Container simplification is a consequence of those outcomes, not the main deliverable.
Keep Docker, native sbx, native host, and HPC workflows usable during migration.

## Boundaries and exclusions

```text
installed FINN / explicitly editable FINN
                |
       existing public operations
                |
       +--------+-------------------+
       |                            |
package resources          internal toolchain execution
                                    |
                        local tools or site-owned route

Existing legacy build entry points
       |
small internal compatibility adapter
       |
existing build behavior and artifact formats
```

Do not add an EDA framework dependency, generic scheduler, launcher plugin system,
public legacy-wrapper command, or new container lifecycle manager. Keep sbx
composition and lifecycle native; site configuration owns mounts, network and
credentials. Cardinal remains an external consumer with no FINN-specific contract.

Do not redesign build allocation, checkpoint/model schemas, artifact relocation,
cache layouts, or self-contained project exports. An independent build-packaging
redesign is unavailable: do not seek it out or guess its API. Do not require a
pyproject.toml migration merely to implement the packaging fixes.

## Workstream 1: package resources properly

Start here. This is the largest direct improvement for ordinary FINN consumers.

- Use the existing resource inventory to identify FINN-owned RTL/templates, C++/HLS
  support, Tcl, driver templates and XSI sources. Exclude tests and generated output;
  retain licence files and notices.
- Include the required assets in wheels and sdists using the existing packaging
  machinery where possible. Define meaningful package version/provenance metadata.
- Add one small resource helper using standard-library package-resource facilities.
  Avoid a generalized resource registry.
- Migrate coherent consumer families, starting with resource operations that can
  be exercised without proprietary tools. Migrate generated Tcl/shell resource
  references as well as Python lookup code.
- Treat finn-hlslib and board definitions as explicitly versioned external inputs
  with documented installed locations; do not silently make them FINN-owned data.
- Never write generated output or compilation caches into the installed package.

Resource lifetime is part of correctness. Invocation-scoped temporary materialization
is suitable only for consumers that finish within that lifetime. Saved Tcl/projects
must reference stable installed paths or deliberate build-owned copies. Stable
absolute installation paths are acceptable for now, with their lifetime documented.
Packaging does not promise that a generated project survives moving or replacing
its installation.

Acceptance: build a wheel from a checkout and a wheel from its sdist, compare relevant
contents, install in isolation, hide the checkout, unset root/resource variables,
and run from an unrelated directory. Read resources and generate representative
RTL/driver outputs; inspect generated HLS/Tcl references. Repeat the supported
resource operations with an editable install. Do not label these relocation tests.

## Workstream 2: explicit installation for users and developers

Land alongside resource packaging where useful; do not wait for all vendor migration.

- Support an installed FINN distribution and an editable FINN checkout using ordinary
  package-tool commands and real package-generated console entry points.
- Document a short FINN-plus-QONNX editable workflow, and an explicit equivalent for
  other supported co-developed dependencies. No automatic nearby-checkout discovery.
- Provide a simple documented way to inspect selected package versions and import
  paths. Import behavior and installed metadata must agree.
- Replace hidden live/frozen/auto import selection with explicit installation
  choices. Remove the old selection mechanism only after its users are migrated.
- Distinguish a reusable dependency image from an installed FINN application image
  or environment. Application identity includes the selected FINN source/package;
  dependency-image identity can remain independent of the checkout.
- Make editable setup an explicit preparation step in a writable environment, never
  an action on every command, shell startup or Python import. Document offline setup
  using prepared dependencies/build requirements where supported.
- Preserve native activation as environment selection. Keep ordinary SIF execution
  read-only; editable HPC development needs an explicitly writable environment.

Acceptance: fresh installation, editable FINN, and editable FINN plus QONNX all work
from arbitrary cwd. Editing a selected dependency affects the intended environment;
opening an unrelated checkout does not. Two prepared development environments stay
independent. Normal Python startup neither installs packages nor repairs directories.

## Workstream 3: a narrow internal AMD/Xilinx execution boundary

Adopt implementation patterns, not another project's flow architecture. The supplied
research recommends no dependency on Edalize, FuseSoC, hls4ml, Xeda, PYNQ, pyvivado,
or cocotb-vivado for this purpose. Treat its source claims as research input; before
copying nontrivial code, inspect the relevant primary source at a pinned revision
and check its licence. The supplied report's opaque citation IDs are not a local,
reproducible source inventory.

Extend existing useful seams, particularly `resolve_xilinx_tool` and
`launch_process_helper`. Keep the permanent implementation compact, preferably in
one internal toolchain module with adapters into existing callers. Do not introduce
public framework objects before concrete callers demonstrate a need.

### Selection and routing

- Separate intended installation/configuration from how commands reach the tools.
- Support local-settings selection and an explicitly accepted already-configured
  environment. Keep discovery optional; preserve explicit user/site selection.
- For site-managed execution, let the site wrapper own remote activation and paths.
  A local environment snapshot is not a guarantee of remote environment propagation.
- Preserve FINN's existing site command-directory override when adapting callers.
  A launcher prefix must not bypass that override or replace remote tool names with
  absolute local executable paths.
- Probe identity through the same route used for the real operation. Bound probes
  with timeouts and actionable diagnostics; avoid broad eager probing on import.
- Record tool versions independently from capability evidence. Capability availability
  validates a requested frontend; it does not automatically select the newest one.
- Keep FINN operation/codegen compatibility constraints separate from executable
  availability. Validate old HLS, standalone Vitis HLS, and unified HLS deliberately.

### Environment and execution

- Capture vendor settings in a child Bash process, with positional arguments and
  NUL-delimited environment output. Do not reimplement settings64.sh in Python.
- Explicitly control inherited Bash startup inputs, especially BASH_ENV. Using
  --noprofile and --norc alone does not disable that noninteractive hook.
- Define the accepted base environment. Do not attempt a general-purpose "unsource"
  of a different installation; distinguish explicit selection from legacy ambient use.
- Snapshot caller-provided mappings defensively; a frozen dataclass containing a
  mutable dict is not immutable. Copy before per-operation overrides.
- Use argv, explicit cwd and child-scoped env. Do not call os.chdir or mutate
  os.environ as an execution convenience, including temporarily around parallel work.
- Preserve status and useful logs. Log the selected route, identity, command, cwd,
  duration and result; never dump captured environments or secrets.
- Define cancellation for the local process group and descendants. Remote cancellation
  remains a site-wrapper contract; FINN must not acquire scheduler semantics.
- Keep serializable selection descriptions for workers. Do not persist secret-bearing
  environment snapshots as ordinary provenance records.

Acceptance: fake settings scripts/tools test awkward paths, BASH_ENV interference,
argv fidelity, route-preserving probes, failures, timeouts, cancellation, environment
isolation, and concurrent toolchain selections. Validate one representative Vivado
operation and one HLS path before migrating further callers. Actual supported flows
need installation-backed and, separately, licensed-operation tests. Banner success
is not evidence of synthesis or licence success.

## Workstream 4: keep legacy compatibility internal and removable

Do not add `finn-legacy-build-env` as a public command. Existing explicit FINN build
entry points should arrange the temporary compatibility they still need.

- Confine interpretation of old FINN root/build/tool-selection conventions to one
  clearly named internal module, or at most two if an unavoidable native-loader
  boundary merits separation. Avoid overlapping build-env and tool-env adapters.
- Keep a short `docs/legacy-build-env-ledger.md`: assignment, actual consumer,
  enabling input, tests, and deletion condition. Artifact-location limitations belong
  in the ledger but are not something an environment adapter can repair.
- Translate legacy inputs once into the permanent resource/tool execution interfaces
  where possible. New code must not depend on legacy-variable interpretation.
- Invoke compatibility at known build boundaries, not generic container startup,
  Python startup, shell startup, or command-name detection in a launcher.
- Explicit arguments take precedence over conflicting ambient legacy roots. Do not
  reconstruct a fake checkout to satisfy consumers that have not yet migrated.
- Establish compatibility for the relevant execution/worker process tree. Do not
  hide global environment mutation inside a helper. Where an in-process legacy
  consumer prevents isolation, record that exact remaining limitation and migrate
  or isolate that entry path before claiming concurrent isolation.
- Preserve existing artifact layouts and checkpoint formats. Remove assignments
  progressively as their consumers migrate.

Acceptance: the ordinary documented build path remains usable without asking users
to prepend a new wrapper. Test worker inheritance, precedence, missing requirements,
and isolation. Ordinary imports/resource operations do not activate compatibility.
Use one searchable compatibility marker and an allowlist of real interpretation
sites; do not ban legitimate vendor-variable forwarding or documentation references.

## XSI: a bounded experiment, not a prerequisite for the first wins

Prototype a fresh Python worker for one representative XSI simulation. Compare it
with the existing integration for correct results, startup/runtime cost, cleanup,
cancellation and the required pre-start loader environment. Never substitute a
fork of a process that already loaded vendor libraries for a clean interpreter.

If feasible, make the worker an internal implementation detail. With a site-managed
route, do not pass the local absolute sys.executable unless it is valid there.
Do not build an RPC service or redesign simulation caches/artifact locations.

If real XSI validation is unavailable or isolation is not yet suitable, preserve
the exact existing requirement behind the bounded legacy path. Document unsupported
version switching and concurrency rather than broadening global startup activation.
Packaging and ordinary tool execution can land without declaring XSI solved.

## Follow-on cleanup: delete mechanisms after their obligations move

Audit and remove only what is now redundant:

| Existing mechanism | Gate for removal or reduction |
| --- | --- |
| `docker/finn-live.pth`, `docker/finn_paths.py` | Explicit installed/editable workflows cover imports, resources and required build setup. |
| `docker/build_dataflow` | Installed metadata supplies the console entry point. |
| `docker/finn-bashenv.sh` | FINN activation is explicit; native sbx-required integration is preserved. |
| `docker/toolchain-shim` | FINN operations use explicit dispatch; bare vendor convenience is intentionally documented or dropped. |
| `docker/finn-toolchain.sh` | Required vendor setup is provided by scoped execution or a documented legacy path. |
| `docker/finn_entrypoint.sh` | Preserve justified arbitrary-UID/home/storage behavior; make Tcl site initialization deliberate. |
| Dockerfile global loader variables | Scoped or legacy replacements pass actual relevant execution/loader tests. |

Update image inputs, Docker/Compose, native sbx examples, native activation and CI
with the corresponding behavior changes. No startup repair may simply move into
another universally invoked hook. Keep native sbx files as the integration surface;
this work must not resurrect a FINN sandbox runner or Cardinal adapter.

## Delivery and release gates

Prefer independently reviewable changes:

1. Resource packaging, small resource API, and initial consumer families.
2. Explicit installed/editable workflows and package identity verification.
3. Internal legacy adapter/ledger where migration actually needs them; no public CLI.
4. Narrow tool execution boundary and representative Vivado/HLS caller migrations.
5. Further consumer migrations and the optional XSI experiment.
6. Removal of proven-redundant hooks and updated runtime/CI instructions.

Steps may interleave to keep each commit usable; no empty framework or adapter is
needed up front. Inventory exact touched files per commit instead of promising a
speculative repository-wide rewrite.

Release gates must exercise complete user journeys, not only isolated helpers:

- Fresh installed FINN: hidden checkout, unrelated cwd, meaningful resource operation.
- Editable FINN: change code/resources and observe the intended installation.
- Editable FINN plus QONNX: verify selected code and metadata without hidden switching.
- Representative existing FPGA build: public entry point, selected tools, expected
  output; no extra user-facing legacy activation layer.
- Checkpoint reuse under documented installation/path constraints. Explicitly state
  what upgrading FINN, changing toolchains or moving directories invalidates; do not
  promise relocation or reuse semantics that the current build format cannot provide.
- Relevant runtime paths: Docker, native setup and native sbx; verify SIF/HPC where
  available and report unavailable coverage separately.

Most infrastructure tests should run without AMD software. Keep synthetic tests,
installed-tool probes, licensed operations and native simulations distinct in the
validation record. Missing hardware does not block package/resource improvements;
it does limit which vendor behavior can be claimed supported.

## Completion criteria

Each delivered change must make a supported user/developer workflow better on its
own. The permanent boundaries work without knowing about the compatibility adapter.
Remaining legacy obligations have named consumers and removal conditions. No new
build-workspace framework, scheduler abstraction, startup repair mechanism or public
compatibility command is introduced. Unverified hardware/runtime behavior is explicit.
