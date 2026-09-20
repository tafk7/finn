# FINN container/runtime implementation plan

Date: 2026-09-20.
Status: architecture approved; implementation work and integration remain.

This plan executes the approved [container/runtime direction](container-runtime-refactor-proposal.md).
It incorporates the [XSI investigation](xsi-process-boundary-investigation.md).
The [Brainsmith comparison](brainsmith-xsi-comparison.md) is reference material only;
compatibility with Brainsmith or another orchestration library is not a requirement.
Earlier design documents and validation records remain historical inputs.

Writing this plan does not implement the work packages below. No merge, push or
publication is part of this documentation task.

**1. Fixed decisions**

- Normal FINN build orchestration runs in the calling Python process.
- A separate private branch owns build path/configuration/logging/global-state
  cleanup. Integrate its implementation; do not duplicate it, seek inaccessible
  code, or guess its API.
- FINN and selected dependencies are installed explicitly. Python import, shell
  startup and ordinary commands never install packages or select nearby checkouts.
- Dependency and application images are distinct artifacts with distinct identities.
- The default development environment is an isolated writable venv populated from
  prepared wheels, followed by explicit editable installations.
- Vendor commands receive argv, cwd and a selected child environment through one
  narrow internal execution implementation.
- Native XSI state belongs to a simulation-session process. The whole-build Python
  worker introduced in the earlier working tree is slated for removal at integration.
- Docker, native sbx, native host and HPC runtimes own machine access, mounts,
  writable storage, network and credentials. Keep sbx lifecycle/composition native.
- Preserve current output/checkpoint formats and allocation policy. This work does
  not redesign build workspaces, cache layouts, relocation or self-contained exports.
- Add no scheduler, EDA framework dependency, public compatibility wrapper, generic
  RPC service, new container lifecycle manager or Cardinal-specific integration.

```text
                            Runtime/site configuration
                              mounts, identity, storage
                                         |
                            Explicit Python installation
                                         |
                              FINN build orchestration
                              (current Python process)
                                  /              \
                     Vendor command process     XSI session process
                     argv + cwd + child env     kernel + design + testbench
                                                bulk inputs/results
```

**2. Ownership and parallel execution**

| Area | Owner | Work allowed before private-branch integration |
| --- | --- | --- |
| Build paths, configuration, logging and process-global cleanup | Private build-engine branch | Define observable integration acceptance; keep implementation here unchanged except separately agreed fixes. |
| Resources, packaging and installation inspection | This work | Implement and validate independently. |
| Images, artifact identity, CLI selection and development preparation | This work | Implement and validate independently. |
| Process/tool execution and non-overlapping callers | This work | Implement helpers, direct operation tests and callers whose interfaces are known. |
| XSI session implementation | This work | Implement/test directly; defer overlapping build plumbing to integration. |
| Connecting runtime inputs to the cleaned build API | Integration | Use the private branch's actual interfaces once available. |
| Whole-build worker and broad activation removal | Integration and final cleanup | Do not expand or make new components depend on them. |

Treat `src/finn/builder/` and shared configuration/allocation/logging plumbing as
integration-sensitive. `src/finn/util/basic.py`, simulation call sites and some
transformations may also overlap; keep edits narrow and identify touched files in
each delivered change. Source/package moves that affect imports require the same
coordination. Do not reserve an invented build-context class or new model fields
in anticipation of the private branch.

```text
P0 baseline
   |
   +-- P1 package/resources -- P2 dependency/application images -- P3 development
   |                                                          Docker/sbx/native
   |
   +-- P4 scoped tools ------ P5 XSI session mechanisms and validation
   |         |                            |
   |         +---- independent tests -----+
   |
Private build-engine branch ------------------+
                                              |
                                  P6 integration
                                              |
                                  P7 remaining hook removal
                                              |
                                  P8 release validation
```

P1-P4 can deliver useful changes without waiting for the private branch. P5 can
advance with synthetic tests and direct session entry points; actual AMD validation
is a separate prerequisite for declaring XSI support. Work-package numbering does
not serialize independent work. Individual obsolete mechanisms may be removed as
soon as their own replacement gates pass.

**3. P0 — Establish the baseline and integration boundaries**

The local checkpoint and initial triage are recorded in
[the implementation handoff](container-runtime-handoff.md).

Deliverables:

- [x] Record the starting revision, working-tree changes and existing untracked
  design inputs. Preserve the user's work; do not reset the repository. See the
  archive checkpoint and exact inventory in the handoff.
- [ ] Classify existing changes as retain, revise, integrate later or delete.
  Retain the resource consumer migrations, package entry-point work, inspection
  command and scoped execution foundation where their tests support them.
- [ ] Inventory remaining root/resource interpretation, vendor subprocesses,
  native imports and startup side effects. Record actual consumers in the existing
  ledger; distinguish legitimate vendor-variable forwarding from interpretation.
- [ ] Identify shared files with the private branch when that information becomes
  available. Avoid speculative plumbing until then.
- [ ] Record available infrastructure: Docker, native sbx, AMD installations,
  licence access and Apptainer/Singularity. Keep unavailable coverage explicit.

Current known corrections include the whole-build worker, the single image-input
identity, remaining shell/tool hooks and the native XSI setup check. In
`setup-local.sh`, `finn.xsi.setup --check` checks prerequisites; it does not establish
that a compiled extension exists. Correct that independent setup regression without
reimplementing build-engine work.

Exit: a concrete touched-file inventory and current validation baseline exist.
The previous 157-test/Docker/sbx record is historical evidence, not an automatic
pass for subsequent changes.

**4. P1 — Finalize package resources and installation behavior**

Primary files: `setup.py`, `setup.cfg`, `MANIFEST.in`, `VERSION`,
`src/finn/util/resources.py`, `src/finn/util/installation.py`, FINN-owned resource
sources and their consumers; `tests/util/test_runtime_installation.py` and
`tests/util/test_runtime_codegen.py`.

- [ ] Consolidate FINN-owned data into a conventional package data tree, with a
  real package anchor (proposed location: `src/finn/_data/`). Preserve the small
  `resource_path(family, ...)` interface while simplifying package mappings.
- [ ] Include RTL, custom HLS/C++, Tcl, driver templates, XSI sources and required
  notices. Exclude test fixtures, generated files and compiled native artifacts.
- [ ] Update Python consumers, generated recipes, source-build references and tests
  together. Do not retain a synthetic checkout or resource-discovery hook.
- [ ] Keep finn-hlslib and board definitions as versioned external inputs with
  explicit locations and documented image defaults.
- [ ] Preserve meaningful package version/provenance through wheel and sdist builds.
  Inspect actual import locations and metadata for installed and editable packages.
- [ ] Ensure generated work never writes into installed resource directories.
  Document the lifetime of absolute resource references in saved projects.

Acceptance:

- [ ] Compare a checkout-built wheel with a wheel built from its sdist, including
  relevant resources, version and provenance.
- [ ] Install in isolation; make the source checkout unavailable; unset root/resource
  variables; run from unrelated cwd. Generate representative RTL and driver outputs
  and inspect generated HLS/Tcl references.
- [ ] Repeat resource operations with an editable install. Observe code/resource
  edits and verify selected code agrees with installed metadata.
- [ ] Ordinary imports create no scratch directory, install nothing and do not
  modify process environment. Test read-only package resources.

Do not describe these as checkpoint/project relocation tests.

**5. P2 — Build dependency and application artifacts**

Primary files: `docker/Dockerfile.finn`, `docker-bake.hcl`, `.dockerignore`,
`docker/image-inputs.txt` and its target-specific replacements, `docker/lib.sh`,
`docker/build`, `docker/run`, `deps.env`, dependency requirement/constraint files,
`ci/scripts/build-images.sh` and associated CLI/identity tests.

- [ ] Produce the resolved dependency wheel set, including pip, setuptools, wheel
  and build requirements for supported editable dependencies. Record exact source
  revisions for dependencies built from Git and provide wheel checksums.
- [ ] Inspect dependency closure so it cannot silently install FINN transitively
  into the development base. Arrange packages requiring FINN at the appropriate
  application/development installation step.
- [ ] Ship the development dependency artifact with Python/system prerequisites,
  `/opt/finn/wheels`, and `/opt/finn/development-requirements.txt`.
  The resolved manifest excludes FINN itself and is usable without network access.
- [ ] Build FINN into a wheel and install it in the application artifact. Install
  dependencies from the same prepared set; avoid retaining the development
  wheelhouse in the final application image where build-stage mounts suffice.
- [ ] Retain optional accelerator-runtime additions and a thin native-sbx variant
  for the selected artifact. AMD installations remain external read-only inputs.
- [ ] Split dependency/application identities. Dependency inputs exclude FINN
  application code/resources; application identity additionally includes its
  wheel/content digest. Runtime additions and sbx variations identify themselves.
- [ ] Add `--dependencies` to existing build/run entry points. Default remains the
  installed application. Preserve `--runtime`, sbx preparation and SIF export
  semantics; image selection never triggers editable installation.
- [ ] Make Bake authoritative for tags/targets and update CI provenance accordingly.
  Distinguish the immutable image from edited application code at runtime.

Acceptance:

- [ ] Editing FINN code/resources changes application identity without changing
  dependency identity. Dependency pin changes affect the relevant artifacts.
- [ ] From the dependency artifact, prepare an isolated venv entirely offline.
  No FINN application is selected until explicitly installed.
- [ ] Run the application image without a checkout mount; imports, packaged
  resources and generated console commands work.
- [ ] Verify relevant combinations of artifact kind, runtime additions and sbx
  variant. Keep supplied/proprietary runtime packages an explicit prerequisite.

**6. P3 — Make development environments concrete and persistent**

Primary files: `compose.yaml`, `.devcontainer/devcontainer.json`,
`.devcontainer/compose.yaml`, `docker/run`, `docker/config.py`, `docker/sbx/` examples,
`setup-local.sh`, `scripts/activate.sh`, installation/runtime documentation and
container conformance tests.

Default isolated-venv preparation, inside the selected dependency environment:

```bash
python -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-index \
  --find-links /opt/finn/wheels -r /opt/finn/development-requirements.txt
"$VENV/bin/python" -m pip install --use-pep517 --no-index \
  --no-deps --no-build-isolation -e "$CHECKOUT"
```

The prepared dependency manifest must make the packaging flags above valid on the
selected Python/platform. Use standard package commands; introduce no environment
manager. Optional overlay environments require their own import-precedence tests.

| Runtime | Source location | Default venv location | Lifetime |
| --- | --- | --- | --- |
| Disposable Docker run | Explicit checkout bind, e.g. `/workspace/finn` | Dedicated host bind/volume at `/env/venv` | Survives `run --rm`; belongs to selected dependency image and stable mount paths. |
| Persistent Docker container | Explicit checkout bind | Writable container location or dedicated mount | Container lifetime unless separately mounted. |
| Dev Container | Declared workspace mount | Environment-specific writable location | Prepared during creation; selected by editor/terminal afterwards. |
| Native sbx | Native absolute mounted workspace path | `$HOME/.venvs/finn-dev` inside sandbox | Reused by exec sessions; recreated after sandbox deletion unless explicitly mounted persistently. |
| Native host | Explicit checkout path | Host-specific venv | Independent of container/sbx environments. |
| SIF application | No checkout required | Installed application in image | Read-only; editable use needs separately prepared writable storage. |

Docker deliverables:

- [ ] Explicitly prepare the host environment directory/volume with suitable
  ownership. Use the selected UID/GID; do not repair ownership on every command.
- [ ] Provide one preparation command and later-run examples that reuse the same
  source/environment mounts. Later commands invoke `/env/venv/bin/python` or its
  console scripts and perform no pip operations.
- [ ] Wire the existing launcher/Compose surface to pass explicit mounts without
  command-name detection or automatic source selection.
- [ ] Select the dependency artifact in Dev Containers. Prepare once at creation
  and select its interpreter; do not silently share a host-native `.venv`.
- [ ] Remove stale import-mode settings, including the current `FINN_DEPS` entry
  in the Dev Container Compose override.

Native sbx deliverables:

- [ ] Use the dependency-image sbx variant with native environment creation and
  user-owned configuration outside mounted workspaces.
- [ ] After creation, explicitly prepare the sandbox-private venv against the
  actual mounted checkout path. Do not assume Docker's `/workspace/finn` alias.
- [ ] Later native exec commands reuse the venv. Use explicit executable paths
  or native PATH/editor configuration; no installer runs from a shell hook.
- [ ] Document deletion/recreation behavior and the optional separately mounted
  environment directory. Keep source, environment and artifact mounts explicit.
- [ ] Preserve native composition, lifecycle, network and credential ownership.
  Do not add a FINN sandbox runner or automatic site setup.

Common deliverables and acceptance:

- [ ] Select editable QONNX/Brevitas/other supported checkouts explicitly, with
  their build requirements already provisioned for offline preparation.
- [ ] Document stable runtime paths, environment recreation after incompatible
  base changes, metadata/entry-point reinstall requirements and native rebuilds.
- [ ] Verify a Docker environment prepared in one disposable container works in
  a second without reinstalling; verify multiple sbx exec sessions likewise.
- [ ] Verify FINN-plus-QONNX editable behavior, unrelated cwd, atomic source edits
  and two independent environments. Opening another checkout does not select it.
- [ ] Verify shell, noninteractive exec and editor/agent interpreter selection.
  Reopening a prepared environment does not install packages or repair directories.

**7. P4 — Complete scoped tool execution**

Primary files: `src/finn/util/_toolchain.py`, the process/resolution seams in
`src/finn/util/basic.py`, `src/finn/util/hls.py`, concrete vendor callers and
`tests/util/test_runtime_toolchain.py`. Avoid private-branch-owned build plumbing.

- [ ] Keep selection separate from prepared environment and execution route.
  Support explicit local settings, explicitly accepted configured environments,
  command-directory overrides and site launcher prefixes.
- [ ] Define the local base environment. Capture vendor settings in child Bash
  using positional arguments and NUL-delimited output; control BASH_ENV and other
  inherited startup inputs. Never mutate the caller or attempt to unsource tools.
- [ ] Defensively copy mappings. Pass serializable selections to workers; keep
  secret-bearing snapshots out of ordinary provenance.
- [ ] Use argv, explicit cwd/env, useful captured output and reliable status.
  Define timeout/cancellation of local process groups and the remote wrapper's
  responsibility. Document treatment of partial logs on cancellation.
- [ ] Probe identities through the same route as real operations with bounded
  timeouts. Keep frontend selection, capability evidence, codegen compatibility
  and licensed-operation success distinct.
- [ ] Migrate the known remaining callers: Zynq project creation, Alveo/Vitis
  packaging/linking, xelab and C++ simulation compilation/execution. Audit the rest
  of the source for additional real consumers before retiring tool shims.
- [ ] Preserve correctly quoted replay scripts and site command-directory routing;
  never replace remote tool names with local executable paths.

Known caller files include `make_zynq_proj.py`, `alveo_build.py`, `hlsbackend.py`,
`rtlbackend.py`, `finn_xsi/finn_xsi/adapter.py` and `CppBuilder`. `CallHLS` and
`CreateStitchedIP` are existing representative migrations to retain and review.
Other supported site/runtime commands must be accounted for before removing their
old dispatch path. Defer overlapping constructor/configuration plumbing to P6.

Acceptance:

- [ ] Fake tools/settings exercise spaces, substitution characters, argv fidelity,
  inherited Bash hooks, failures, timeout, cancellation and concurrent selections.
- [ ] Verify site override plus launcher composition and route-preserving probes.
- [ ] Direct operation tests preserve parent cwd/environment and do no eager probing
  on import. Record logs without environment/credential dumps.
- [ ] Validate actual installations and representative licensed Vivado/HLS/linking
  operations separately. Only claim tool versions whose relevant flows pass.

**8. P5 — Implement the native simulation-session boundary**

Primary files: `src/finn/xsi/`, the Python driver under `finn_xsi/finn_xsi/`,
`finn_xsi/xsi_finn.*`, `finn_xsi/xsi_bind.cpp`, the C++ harness,
`src/finn/core/rtlsim_exec.py`, and dedicated session tests. Inspect overlap with
private-branch changes before changing shared orchestration.

- [ ] Separate pure compilation/discovery helpers from native bridge loading.
  Launching xelab or inspecting paths must not require an imported XSI extension.
- [ ] Remove import-time native availability decisions from migrated simulation
  consumers. Check actual prerequisites when the operation is requested.
- [ ] Specify the smallest concrete internal request/result contract for a first
  functional simulation: selected compiled design/tool identity, input buffers,
  streams/clock/reset configuration, tracing and execution limits; output buffers,
  metrics, status and artifact paths. Avoid a generic simulation framework.
- [ ] Start a fresh executable image with the selected environment supplied before
  exec. Do not use a forked vendor-loaded interpreter or change loader paths only
  after a Python worker has started.
- [ ] Keep the complete testbench, Python driver, kernel, design and port handles
  inside the session. Exchange data in bulk; begin with existing file/array formats
  and a small control/result record, not per-cycle RPC.
- [ ] Use independent session outputs/logs so concurrent runs do not overwrite each
  other. Preserve compiled artifact locations and validate any vendor requirements
  for working-directory-relative design inputs on real installations.
- [ ] Keep stdout/stderr logs separate from structured results. Publish a successful
  result only after the session completes; reject stale/incomplete result files.
- [ ] Guarantee deterministic close on recoverable exceptions. Add an external
  wall-clock limit alongside cycle watchdogs; report native crashes and potentially
  partial waveforms on forced termination.
- [ ] Keep the current C++ harness for supported workloads; do not treat its
  dummy-input-only implementation as complete functional-simulation support.
- [ ] Cover FINN's own AXI initialization/readback, external memory, MLO, stream
  characterization and tracing as subsequent concrete session cases. Custom
  testbench code executes inside the session with explicit arguments/results;
  do not silently serialize arbitrary parent closures or preserve another
  library's live-object API as a requirement.
- [ ] Verify native bridge/design compatibility before reuse, using small adjacent
  records if needed: relevant sources, ABI, selected headers/tool identity and
  compile inputs. Do not redesign cache layouts or checkpoint schemas.
- [ ] For site-routed execution, require a valid site Python/driver command and
  accessible paths; do not assume local absolute sys.executable is usable remotely.

Acceptance and decision gate:

- [ ] Synthetic tests prove process isolation, selection, data round-trip, failure
  propagation, timeout/cancellation and cleanup without importing AMD libraries
  into the parent. Existing loader experiments are supporting evidence only.
- [ ] Compare a representative real simulation with the current engine for outputs,
  cycle counts and traces. Then cover AXI read/write, MLO and characterization.
- [ ] Test repeated sessions, supported concurrent sessions, compatible compiled
  design reuse, and rejection of incompatible artifacts/tool selections.
- [ ] Measure cold startup, small-node and large stitched workloads, and bulk data
  transfer. Record measurements before considering batching beyond a session or
  persistent workers. A session can already contain multiple frames and resets.
- [ ] Do not label XSI complete without installation-backed validation. Missing
  AMD access leaves this gate open while other work packages continue.

**9. P6 — Integrate the private build-engine branch**

Dependency: the actual private-branch changes are available for integration.
Integrate the resource/tool/session interfaces that are ready, then incorporate
remaining work incrementally. Completion of every P1-P5 gate is not required to
start P6; in particular, missing AMD access must not prevent integrating independent
build-engine fixes. Unvalidated simulation paths remain explicitly open, and their
activation/loader mechanisms are retired only when replacement gates pass.
No inferred branch API is an implementation prerequisite for P1-P5.

- [ ] Review the real branch interfaces and resolve overlaps using the ownership
  inventory. Adopt its path/configuration/logging implementation.
- [ ] Connect explicit resource/data/tool/scratch inputs at the real operation and
  worker boundaries. Avoid duplicate environment adapters or speculative model
  fields. Keep current artifact layout and checkpoint semantics.
- [ ] Remove/reconcile this working tree's whole-build subprocess in
  `build_dataflow_directory` and its broad environment preparation. Do not preserve
  that worker merely to avoid updating callers or tests.
- [ ] Remove obsolete internal legacy-variable interpretation where the new branch
  and explicit runtime inputs have replaced it. Any retained old-input parsing is
  confined to an outer boundary with a real consumer and deletion condition.
- [ ] Update tests to assert observable build behavior, not the existence of a
  whole-build worker. Retire tests of removed compatibility mechanisms.

Acceptance:

- [ ] Both package-generated CLI and direct Python builds work with intended path
  semantics, custom steps and error handling.
- [ ] FINN-supported paths do not change caller cwd/environment/stdout/stderr as
  execution conveniences. Check logging separation in supported concurrent cases.
- [ ] Resume a representative checkpoint under documented installation/path
  constraints; missing or incompatible artifacts fail clearly.
- [ ] Exercise a build using the selected tools and simulation boundary. An
  integration merge is not evidence of vendor correctness without that execution.

**10. P7 — Delete obsolete runtime mechanisms**

Delete mechanisms incrementally after their actual consumer gates pass. Packaging
and installation removals need not wait for XSI. Loader/global activation removal
requires the relevant native and licensed-operation evidence.

| Mechanism | Required replacement/gate |
| --- | --- |
| `docker/finn_paths.py`, `docker/finn-live.pth` | Explicit installed/editable journeys; no hidden source selection. Already deleted in the earlier working tree; verify no remaining references. |
| Handwritten `docker/build_dataflow` | Installed package-generated entry point; already removed in the earlier tree. |
| Whole-build Python worker | P6 integration of direct build-engine fixes. |
| `docker/toolchain-shim` | All supported FINN vendor callers use explicit dispatch; document deliberate vendor-shell/site-wrapper usage for bare tools. |
| `docker/finn-toolchain.sh`, `FINN_ENV_APPLIED` | Scoped tool/session environments replace every actual consumer. |
| Generic `docker/finn-bashenv.sh` activation | No FINN activation at shell startup; preserve only required native sbx persistent-environment integration in its variant. |
| Dockerfile global loader settings | Tool/session-scoped replacements pass actual relevant licence/native tests, including the existing libudev workaround where applicable. |
| Broad `_legacy_build_env` activation | Explicit inputs and integrated build engine; no public replacement wrapper. |
| Entrypoint FINN/vendor/Tcl behavior | Explicit preparation or mounts. Retain only justified UID/home/exec behavior. |

- [ ] Verify installed operations work through execution paths that bypass the
  entrypoint, including Docker exec and native sbx exec.
- [ ] Keep arbitrary-UID home behavior bounded to runtime-private writable storage.
  Do not install packages, repair build directories or source tools there.
- [ ] Provide site Tcl initialization through explicit mounts/preparation. Keep
  credentials and network configuration site-owned.
- [ ] Update image inputs, Compose, Dev Containers, native setup/activation, sbx
  examples and CI together. Remove stale comments and environment assignments;
  do not move universal startup repair into another hook.

**11. P8 — Complete user-journey validation and release record**

Keep separate results for synthetic tests, installed-tool probes, licensed builds,
native simulation and runtime integration. Record commands, artifact identities,
selected package/tool versions, outcomes and unavailable coverage without secrets.

| Journey | Required evidence |
| --- | --- |
| Installed FINN | Hidden checkout, unrelated cwd, unset resource roots, representative resource generation and real entry points. |
| Editable FINN | Selected code/resources change; package metadata and imports agree. |
| Editable FINN plus QONNX | Explicit dependency source selection and isolation from another prepared environment. |
| Docker development | Prepare once, remove container, reuse mounted venv in another container without installing; arbitrary UID and stable mount paths. |
| Dev Container | Correct dependency image and interpreter; creation versus reopen behavior. |
| Native sbx development | Native create, explicit preparation, repeated exec, selected agent/editor environment, cleanup and documented recreation behavior. |
| Native host | Explicit setup/activation and working installed resources; XSI setup check corrected. |
| FPGA build | Existing public entry point, selected route/tools, expected output, useful failures and no extra user-facing compatibility wrapper. |
| XSI | Functional and measurement equivalence, lifecycle, failure/cancellation behavior and measured overhead. |
| Checkpoint reuse | Original path/installation constraints honored; invalid reuse fails clearly. |
| SIF/HPC | Read-only installed execution; explicitly writable editable environment and scratch where supported. |

- [ ] Re-run the relevant focused tests after changes, then the complete journeys
  for affected release artifacts. Do not keep repeating unaffected broad suites.
- [ ] Update `docs/installation.md`, runtime guides, CI instructions,
  `docs/legacy-build-env-ledger.md`, `docs/runtime-validation.md` and the delivered
  file inventory to describe the final implementation.
- [ ] Remove stale claims that the earlier whole-build worker is the permanent
  architecture or that synthetic tests establish licensed/native support.
- [ ] Mark work complete only when its gates pass. Explicitly distinguish delivered
  packaging/development improvements from any remaining vendor-validation gate.

**12. Delivery discipline and first actions**

Deliver small, independently reviewable changes aligned with P1-P8. Each change
records the actual files touched, behavior, checks, and remaining limitations.
Source/resource moves should be mechanical and separated from unrelated behavior
changes. Do not turn the file inventory into a promise of speculative repository-
wide changes. Preserve historical design inputs and update current instructions
when behavior changes.

Begin with P0, then run packaging/image/development work alongside scoped tool and
XSI work. The first concrete artifacts are the final package/resource layout, the
resolved offline wheelhouse and manifests, and the dependency/application image
selection tests. Build-engine implementation remains with the private branch.
Integration is a distinct work package, not a reason to block independent progress.
