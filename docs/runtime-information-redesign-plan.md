# Experimental containerization refactor: explicit runtime boundaries

Status: proposed implementation plan; implementation has not started.
Updated: 2026-09-16.
Target branch: `experiment/container-runtime-boundaries`.
Base: `refactor/container-stack` at `8a05d50a61e2ab4fa7aa28ee3a70f86ea7b8b10d`.

This branch experiments with the containerization refactor already on its parent.
Do not merge or push as part of executing this plan without a separate request.

## 1. Objective and scope

Make FINN's resources, installation and tool execution explicit. Segment remaining
build-environment debt behind a runtime-neutral, explicitly invoked compatibility
command. Remove universal startup repair as its responsibilities move to these
boundaries.

A private branch is independently redesigning build packaging. Its code and
interfaces are unavailable. **This work must proceed without access, coordination,
branch identification or predictions about that implementation.** The integration
contract is removal of known legacy obligations, not conformity to a guessed
future build API.

This replaces the previous plan's phase 0 and core build-layout adapter proposal.
In particular, do not introduce a new allocator API, build-layout configuration
flag or checkpoint abstraction into the build system for this experiment.

Inputs:

- [Runtime information trace and design analysis](runtime-information-obligations.md).
- [Source inventory and probe results](runtime-information-obligations.json).

### Permanent improvements

- Installed packages own and locate resources without a checkout-root variable.
- Explicit installation selects FINN and editable dependencies.
- Toolchain execution has explicit inputs and scoped process configuration.
- Runtime startup owns only justified filesystem/identity preparation.

### Temporary compatibility

- Current build-directory defaults and `FINN_BUILD_DIR` consumers.
- Remaining `FINN_ROOT`-dependent build/codegen behavior during resource migration.
- Existing absolute artifact paths, checkpoint fields and replay assumptions.
- Build-wide vendor activation still required by unmigrated execution paths.

### Non-goals

Do not redesign build workspaces, artifact manifests, model path fields, cache
layouts, checkpoint migration, project relocation, or self-contained IP exports.
Do not rewrite `ReplaceVerilogRelPaths` to solve relocation here. Do not add FINN
sbx lifecycle management, change site network policy, or register credentials.
A packaging-tool migration is not required merely to change configuration formats.

## 2. Target execution model

```text
Docker / native sbx / native host / HPC
    supplies identity, mounts, storage and network access
                          |
                 installed environment
                          |
            +-------------+----------------+
            |                              |
       ordinary command          explicit legacy wrapper
       no startup repair         temporary build environment
            |                              |
            +-------------+----------------+
                          |
                  FINN operation
              /                     \
     package resources       toolchain execution boundary
                             argv / cwd / scoped environment
```

Normal imports and resource operations must not require compatibility activation.
Legacy builds may require the wrapper until their remaining obligations disappear.
Bare vendor commands after arbitrary runtime `exec` are a convenience choice,
not an automatic acceptance requirement. Document their explicit activation path.

## 3. Phase A — Establish a baseline and compatibility ledger

Use the current public tree only:

1. Refresh the environment/resource consumer inventory against the implementation
   starting revision. Trace Python accesses and generated shell/Tcl expressions.
2. Capture representative installed/import, codegen, allocation, checkpoint reuse,
   tool dispatch and native simulation behavior.
3. Create `docs/legacy-build-env-ledger.md` with each compatibility assignment,
   its current consumers, enabling input, test coverage and deletion condition.
4. Distinguish correctness tests from historical convenience tests. Mark assumptions
   such as mirrored paths and transparent bare-tool invocation for explicit review.
5. Record available vendor installations and runtime versions. Missing hardware
   must not block synthetic boundary tests; record unverified operations separately.

Initial ledger:

| Behavior | Current consumers | Removal condition |
| --- | --- | --- |
| `FINN_ROOT` | Unmigrated resources, generated recipes and XSI source lookup | Those consumers use installed resources or explicit inputs |
| `FINN_BUILD_DIR` | Current allocation helper and build entry | Consumers obtain allocation through replacement build behavior |
| HLS/board resource variables | Existing compiler/Tcl recipes | Recipes receive explicit versioned resource inputs |
| Global tool activation | Unmigrated vendor commands/native loading | Scoped tool execution or validated native-runtime integration covers them |
| Absolute artifact references | Current model/checkpoint/generated-project formats | Replacement build implementation defines the supported artifact semantics |

The last row is a limitation to track, not something the wrapper can repair.
No ledger item requires a link to the private branch. The owner is this experiment's
compatibility layer until an independently validated replacement removes the need.

## 4. Phase B — Add one explicit legacy execution boundary

Proposed public spelling:

```bash
finn-legacy-build-env --build-dir /work/build -- COMMAND

# Required only while a selected legacy operation needs a source checkout:
finn-legacy-build-env --build-dir /work/build --source-root /checkout -- COMMAND

# An explicitly activated interactive session:
finn-legacy-build-env --build-dir /work/build --source-root /checkout -- bash
```

Package the command normally, with its implementation in a clearly marked
compatibility module, for example `src/finn/compat/legacy_build_env.py`. The command
name may change during implementation, but its explicit invocation must remain.

### Responsibilities

- Accept and validate explicit input locations; resolve relative arguments once
  against the invocation cwd and preserve paths containing spaces.
- Set only the legacy FINN variables recorded in the ledger for the child command.
- Create the selected legacy scratch directory where required by existing behavior.
- Accept `--source-root` only as a bridge for recorded consumers. Do not discover a
  checkout from cwd, reconstruct a fake repository tree, or turn this into the
  permanent resource API. If a legacy operation requires a checkout, document the
  requirement and fail clearly when it is unavailable.
- Override conflicting inherited legacy variables with explicit arguments; prevent
  ambient FINN values from silently selecting different roots. Preserve ordinary
  unrelated process environment and site inputs needed by the command.
- Compose with the toolchain boundary when legacy build-wide activation is still
  required. Vendor settings/library knowledge belongs to that boundary, not a
  second implementation inside the compatibility command.
- Emit one concise compatibility diagnostic to stderr, keeping stdout usable.
- Preserve argv exactly, return the child exit status, and forward signals correctly
  (prefer process replacement where practical).

### Exclusions

The wrapper must not install packages, rewrite models, migrate checkpoints, change
mounts, manage sandboxes, contact a scheduler on its own, or modify the parent shell.
It must not run automatically from Python startup, Bash startup, the image
entrypoint, or command-name detection in `docker/run`.

Do not modify every allocation/path call site just to add this layer. Initially,
the current build code can consume the child environment it already understands.
Existing multiprocessing descendants inherit it through the ordinary process tree;
verify that behavior rather than inventing a new core context interface now.

A separate Docker/sbx `exec` process does not inherit activation from a previously
opened shell. It must explicitly invoke the wrapper if it runs a legacy build.
Update examples and CI commands to make that visible.

### Acceptance

Test explicit precedence, missing inputs, awkward paths, argv, clean stdout,
exit status, signals, child/worker inheritance and isolation between invocations.
Verify that ordinary installed Python does not invoke the wrapper. Run representative
legacy build/codegen operations under it without changing their artifact schema.

## 5. Phase C — Package resources and retire root-based lookup

### Distribution contract

- Explicitly include FINN-owned RTL, custom HLS, C++ support, Tcl, driver templates
  and XSI sources/adapter in suitable package locations.
- Separate runtime assets from tests/examples/generated output; retain licences.
- Keep upstream finn-hlslib and board definitions explicitly versioned, with a
  documented installed-data boundary rather than an assumed checkout layout.
- Validate wheel contents from a Git checkout and from a generated sdist. A Git
  archive is a separate input type and must not be confused with an sdist.
- Establish meaningful package version/provenance metadata.

### Consumer migration

Introduce a small package-resource API and migrate coherent families:

1. Shared RTL/templates and RTL operators.
2. FINN C++/HLS compilation resources.
3. HLS/Vivado Tcl resource references and shared IP repositories.
4. Driver/board templates and data.
5. XSI source/template lookup.

Replace direct `FINN_ROOT` reads and generated `$FINN_ROOT` /
`$::env(FINN_ROOT)` resource expansion. Migrate the import-time HLS/board resource
variable defaults to their explicit resource owner.

External tools need resource paths that remain valid for their use. Do not leave
persistent scripts referencing expired temporary extraction paths. During this
experiment, stable installed resource paths are acceptable and may remain absolute.
State their installation-lifetime requirement. Durable source snapshots and project
relocation remain deferred to the build-packaging refactor.

Keep current generated output locations, node attributes and checkpoint metadata.
Remove individual legacy assignments as their consumers disappear; the wrapper
must shrink as migration progresses.

### Acceptance

Install a wheel, hide the checkout, unset FINN-root/resource variables, and use an
unrelated cwd. Read resources and generate representative RTL/driver files. Inspect
HLS/C++/Tcl recipes for complete resource references. Verify wheel-from-sdist parity
and meaningful version metadata. Do not claim a relocatable build from these tests.

## 6. Phase D — Scope toolchain selection and execution

Extend useful existing seams such as `resolve_xilinx_tool` and
`launch_process_helper`; do not build a parallel scheduler or generic framework.

- Describe selected installation roots, explicit/validated version, capabilities
  and command dispatch. Keep descriptions serializable for worker processes.
- Replace repeated version parsing from directory names, including code-generation
  decisions as well as executable selection. Different feature thresholds must
  retain separate tests; they are not necessarily the same vendor transition.
- Execute with argv, cwd, scoped environment, logs and checked return codes.
- Preserve the existing site command-directory override as a compatibility input
  to explicit dispatch configuration. Absolute local binaries must not bypass
  site-owned remote wrappers.
- Scope licence variables, platform paths, vendor settings and loader workarounds
  to relevant executions. Network permissions and mounts remain site/runtime inputs.
- Migrate generated `cd`/`PWD` command sequences to explicit cwd where appropriate;
  retain useful replay recipes with deliberate quoting and required inputs.
- Permit a vendor settings script to configure a child execution. Do not rewrite
  vendor scripts or add environment caching without measured need.

While internal call sites remain unmigrated, the legacy wrapper may request
build-wide toolchain activation for its child process tree. Record that bridge in
the ledger. Do not retain a universal `FINN_ENV_APPLIED` latch as the permanent
solution to selecting or switching toolchains.

### XSI boundary and deferral

XSI loads vendor libraries inside Python and needs separate validation. Prototype
an isolated simulation worker process versus an explicit in-process loader contract
using a small representative simulation. Choose based on observed correctness and
integration cost. A shell wrapper alone is not a complete native-loader redesign.

Retain existing artifact locations and conservative compatibility/rebuild behavior.
Do not invent a new XSI cache layout to preempt the private build-packaging work.
If a replacement loader cannot yet be validated, retain that specific requirement
inside documented legacy execution and mark the remaining gate honestly. It must
not force universal activation for unrelated installed Python.

### Acceptance

Fake tools verify dispatch, argv, cwd, scoped environment and failures. Test
non-versioned installation paths and supported capability transitions. Demonstrate
no stale activation between two toolchain selections. Exercise real HLS/Vivado and
XSI where available; distinguish executable/version checks from licensed operations.

## 7. Phase E — Make development installation explicit

Support ordinary package-tool workflows for:

- Installed FINN and pinned installed dependencies.
- Editable FINN and an explicitly selected list of editable dependencies.

Do not preserve `FINN_DEPS=auto` as hidden source selection. Replace live/frozen
bootstrap behavior with explicit installation choices, retaining dependency
constraints and correct package metadata/console scripts.

The existing dependency image can remain a development building block. Decide
whether installed FINN is an additional image target or an explicitly installed
layer/environment. Installed application identity must include FINN package/source
identity; the reusable dependency image may continue to exclude mounted source.

For arbitrary checkout paths, install editably into a writable environment during
an explicit setup step. This may happen once per user-owned sandbox/environment,
but never on every container command, shell or Python start. Use prepared local
build requirements/dependencies for an offline setup path where supported.

Integration:

- Docker: execute an already prepared installation; preserve FPGA opt-in and notebooks.
- Native setup: installation prepares packages; activation selects that environment.
- Native sbx: copied examples use installed FINN or documented explicit editable
  setup. Agent installation, credentials and lifecycle remain native/user-owned.
- SIF/HPC: use installed FINN or a deliberately writable external development
  environment; ordinary execution must not mutate the SIF.
- CI: select/install source explicitly, retain shared-image loading and record
  application/source identity separately from dependency-image identity.

### Acceptance

Installed/editable FINN works from arbitrary cwd without root discovery. Selected
dependency code and metadata agree; unrelated checkouts do not shadow packages.
Ordinary Python startup has no FINN filesystem side effects or source-selection
failures. Prepared commands do not implicitly install packages. Parallel development
environments can remain independent.

## 8. Phase F — Remove obsolete guest hooks

Retire each mechanism only after its covered use cases have moved to explicit
installation, runtime setup, scoped tool execution or the documented legacy wrapper.

| Mechanism | Intended disposition |
| --- | --- |
| `finn-live.pth` / `finn_paths.py` | Delete startup import selection and scratch/root repair |
| Manually baked `docker/build_dataflow` console shim | Replace with packaging-generated entry point |
| `finn-bashenv.sh` | Remove FINN activation; preserve sbx-required integration through native-supported behavior |
| `toolchain-shim` | Delete after internal migration, unless deliberately retained as optional vendor-command convenience |
| `finn-toolchain.sh` | Reduce/rehome as a scoped vendor helper if still needed; no universal activation |
| `finn_entrypoint.sh` | Retain only justified writable-state/site initialization, or delete if runtime supplies it |
| Global library/preload settings | Remove only when scoped replacements or an explicit legacy execution path are validated |

Audit Dockerfile ENV/COPY, symlinks, image-input manifests, Compose, native examples,
activation, CI and docs together. Preserve ownership/arbitrary-UID behavior if it
remains supported. Treat optional `.Xilinx` Tcl initialization as an explicit
site/runtime concern rather than accidentally dropping it during import cleanup.

Normal execution and legacy execution must have separate conformance tests. Do not
quietly move startup repair into a renamed hook.

## 9. Implementation and review sequence

Suggested commits, each internally usable:

1. Compatibility ledger, baseline fixtures and explicit legacy command.
2. Resource distribution/API, followed by coherent resource-consumer migrations.
3. Explicit toolchain description/execution and incremental caller migrations.
4. XSI experiment and selected implementation or precisely bounded deferral.
5. Explicit installation workflows and runtime/image integration.
6. Hook retirement, documentation, CI invocation migration and final evidence.

Installation/resource prototypes can precede their broad integration. Keep adapters
thin; do not migrate core build internals merely to fit the experiment. Add exact
changed-file inventories to the completion record.

Expected areas: packaging configuration/resource trees; resource and tool helpers;
operator/templates and driver consumers; XSI integration; a compatibility command;
Dockerfile/Bake/image inputs and guest hooks; setup/activation; native examples;
CI invocation/setup; focused tests and docs.

No work item requires the private branch. If an overlap becomes apparent later,
prefer dropping a now-unnecessary adapter over preserving experimental abstractions.

## 10. Completion and eventual gap closure

Completion of the experiment means:

- Installed FINN finds its resources and imports without a checkout-root hook.
- Development source selection is an explicit installation action.
- Migrated tool operations use explicit execution configuration.
- Remaining legacy build requirements are confined to explicit invocation and a
  finite ledger, not scattered runtime startup behavior.
- Current artifact/checkpoint formats were not redesigned here.
- Removed hooks have no hidden replacement. Retained compatibility/vendor exceptions
  have named consumers, tests and removal conditions.
- Validation records distinguish synthetic checks, real vendor checks and unavailable
  runtime/hardware coverage.

Use one searchable marker, such as `LEGACY_BUILD_ENV_COMPAT`, in the compatibility
module, its tests and ledger. New code must not acquire dependencies on that module
or expand its environment contract without a documented existing legacy consumer.

When the independent build refactor becomes available:

1. Run its supported builds without the compatibility command.
2. Identify which ledger consumers and assignments have disappeared.
3. Remove those assignments and update conformance expectations.
4. Validate replacement behavior under its own checkpoint/artifact guarantees.
5. Delete the command, marker and legacy-only tests when no obligations remain.

The public package-resource, installation and tool-execution boundaries must remain
usable through this deletion. There is no promised mapping to an unknown future
allocator or manifest schema.
