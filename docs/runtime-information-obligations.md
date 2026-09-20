# Runtime information obligations: trace and redesign inputs

Date: 2026-09-15. Source revision: `8a05d50a6` on `refactor/container-stack`.
Status: analysis and design input; no production implementation changes.

## 1. Conclusions

The six guest startup files compensate for **four separate missing or incomplete
interfaces**:

1. An installable FINN distribution that owns and locates its resources.
2. A build workspace with explicit inputs, artifacts and persistence semantics.
3. A toolchain integration boundary that owns version/capabilities, execution and
   library loading without configuring every process.
4. An explicit development installation workflow, distinct from using an installed
   FINN distribution.

Container startup is downstream of these problems. Combining the existing hooks
would leave most obligations intact. Standardizing a mount path would make some
lookups easier but would not fix package contents, build relocation or compiled
artifact compatibility.

The largest semantic obligation is **persistent build state**, not the number of
startup scripts. FINN writes filesystem locations into ONNX attributes, metadata,
Tcl, Verilog, compiler scripts and source lists. Those locations subsequently
control reuse, simulation, checkpoint resume and export. A package resource API
is necessary but insufficient to make those artifacts portable.

The recommended direction is to fix resource ownership first, establish explicit
build and toolchain interfaces next, and retire the startup compensation only as
each underlying dependency disappears. Retain vendor-required environment variables
at the vendor boundary; do not promote FINN's current ambient variables into the
new public interface by default.

## 2. Scope, evidence and meaning of “weight”

The static inventory covers all **184 tracked Python files** under `src/finn` and
`finn_xsi` at the revision above. It records literal Python environment accesses,
selected helper calls, and artifact field access sites. Shell, Docker, native sbx,
packaging, CI documentation and representative tests were traced separately.

[Machine-readable inventory and probe results](runtime-information-obligations.json)
contains exact source paths and line numbers. AST counts exclude accesses through
dynamic keys, aliases and external dependency code. Textual mentions include
comments and generated shell/Tcl strings; they are not execution counts.

| Signal | Observed scope | Interpretation |
| --- | --- | --- |
| `FINN_ROOT` | 72 literal Python accesses; 110 textual mentions across 29 production Python files | Broad resource/codegen coupling, not one removable getter |
| `get_finn_root()` | 2 call sites, additional to direct environment reads | Replacing this helper alone leaves most consumers untouched |
| `FINN_BUILD_DIR` | 3 literal accesses; `make_build_dir()` called at 18 sites | Small configuration surface with broad transitive consequences |
| Artifact path/directory fields | 250 literal get/set calls across 18 selected fields | Build state is distributed across operators, transformations and metadata |
| `XILINX_VIVADO` | 9 literal accesses across 6 files | Used for resources and version-dependent behavior as well as execution |
| `resolve_xilinx_tool()` | 10 calls | Existing execution seam worth extending |
| `FINN_DEPS` | No mentions in the scoped production Python | Current live/frozen/auto policy belongs to image/development bootstrap |
| Licence variables / `LD_PRELOAD` | No direct reads in the scoped production Python | Mostly external tool/loader obligations, implemented around FINN |

These counts measure change surface, not runtime cost. No new timing benchmark of
vendor initialization was performed. Historical latency claims in comments are
not used as evidence for a redesign.

### Experiments performed

All mutations were confined to temporary source copies/build outputs and disposable
Docker containers. No vendor tools, credentials, global sandbox policy or user
installation were changed.

- Built wheels with `pip wheel --no-deps --no-build-isolation` from both a Git
  archive and an isolated Git checkout, using the existing
  `xilinx/finn:env-29ed58ffeb42c45f` image.
- Inspected wheel membership and exercised resource lookup with the startup hook
  disabled (`python -S`, explicitly supplied dependency search paths).
- Ran the real `ReplaceVerilogRelPaths` transformation against a synthetic ONNX
  node and HDL/data directory, saved the checkpoint, moved the project, removed
  the original, and inspected the surviving references.
- Loaded the real Python startup module without a workspace and observed its
  directory/environment side effects.
- Sourced the real toolchain script against two synthetic settings scripts to
  test changing tool installations in one process environment.

Raw evidence: `/tmp/finn-obligations-wupmmre_/`, including `wheel-build.log`,
`wheel-git-build.log`, wheel membership files, `probes.log`, and the synthetic
project. Probe scripts: `/tmp/finn-obligations-probe.py` and
`/tmp/finn-obligations-inventory.py`. The JSON companion preserves the essential
results independently of those temporary files.

## 3. Resource ownership: what FINN_ROOT actually stands for

### 3.1 Python import location

The image installs QONNX, Brevitas and finn-experimental, but intentionally does
**not install FINN**. `finn-live.pth` imports `finn_paths.py` at every normal Python
startup; that module finds a workspace and prepends its `src` directory.
`docker/build_dataflow` manually supplies the console script that a FINN
installation would otherwise generate.

This is an image product decision, not a requirement of FINN's algorithms. It
keeps the dependency image independent of source commits. A redesign must decide
whether that remains a development-only product, whether to add an installed
application image, or whether to require an explicit installation into a sandbox.
Simply deleting the hook would currently remove FINN from the image's import path.

Sources: [Python bootstrap](../docker/finn_paths.py),
[image packaging](../docker/Dockerfile.finn), [manual console script](../docker/build_dataflow),
[packaging entry point](../setup.cfg).

### 3.2 Resources used directly by Python

Many operators construct paths such as `FINN_ROOT/finn-rtllib/...`, read templates,
and copy or enumerate HDL. Examples include FIFO, thresholding, convolution input
generators, matrix-vector engines and the loop operator. `HWCustomOp` also reads
shared wrapper templates. These are package resources, not user-supplied projects.

Driver generation reads templates from `FINN_ROOT/src/finn/qnn-data/templates`,
then copies selected Python dependencies using their actual package locations.
The latter already demonstrates that a checkout root is not intrinsically needed.
`finn.util.test` also already uses a package resource API for test data.

Sources: [shared operator resources](../src/finn/custom_op/fpgadataflow/hwcustomop.py),
[FIFO](../src/finn/custom_op/fpgadataflow/rtl/streamingfifo_rtl.py),
[thresholding](../src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py),
[driver generation](../src/finn/transformation/fpgadataflow/make_driver.py),
[existing resource lookup](../src/finn/util/test.py).

### 3.3 Resources handed to compilers and Tcl

A representative C++ simulation chain is:

```text
PrepareCppSim
  -> make_build_dir()
  -> node.code_gen_dir_cppsim
  -> operator code generation
CompileCppSim
  -> HLSBackend.compile_singlenode_code()
  -> CppBuilder writes compile.sh
       $FINN_ROOT/src/finn/qnn-data/cpp
       $FINN_ROOT/custom_hls
       $FINN_HLSLIB_PATH
       vendor include/library directories and rpaths
  -> Bash expands environment variables
  -> g++ produces node_model
  -> node.executable_path stores its location
```

Here environment inheritance compensates for a generated recipe that does not
record its complete inputs. Replacing the Python-side lookup alone would not fix
the shell variable references in `compile.sh`.

`finn.util.basic` also initializes the HLS/board data roots through import-time
`os.environ.setdefault`. Those defaults do not track a later change of workspace
in a long-lived interpreter, adding another reason to separate resource ownership
from ambient initialization.

HLS synthesis similarly writes Tcl that reads `FINN_HLSLIB_PATH` and
`FINN_ROOT/custom_hls`. Vivado stitching refers to shared IP repositories and
simulation RTL through `$::env(FINN_ROOT)`. Board setup uses
`$::env(FINN_BOARD_FILES_PATH)`. These scripts remain dependent on the caller's
environment when rerun outside the original Python process.

Sources: [C++ compilation](../src/finn/custom_op/fpgadataflow/hlsbackend.py),
[compiler helper](../src/finn/util/basic.py),
[HLS Tcl templates](../src/finn/custom_op/fpgadataflow/templates.py),
[stitching](../src/finn/transformation/fpgadataflow/create_stitched_ip.py),
[board Tcl](../src/finn/transformation/fpgadataflow/templates.py).

### 3.4 Resource size and distribution boundaries

Tracked files at the analyzed revision:

| Resource tree | Files | Bytes |
| --- | ---: | ---: |
| `finn-rtllib` | 136 | 987,566 |
| `custom_hls` | 4 | 15,562 |
| `finn_xsi` | 12 | 84,991 |
| `src/finn/qnn-data` | 29 | 963,172 |

These counts include tests/examples within those trees; they are not a proposed
package manifest. FINN-owned resource volume is modest. Separate upstream build
data in the prepared image measured approximately 0.78 MB for finn-hlslib and
20 MB for board definitions (`du -sb`), not the much larger historical size in
Dockerfile comments. Those dependencies need their own version/resource ownership;
they should not be silently copied wholesale into an undifferentiated resource bag.

### 3.5 Packaging probe: important distinction

The archive-built wheel had 187 members and no non-Python data under `finn/`.
The Git-checkout-built wheel had 211 members, including 24 non-Python `qnn-data`
assets. Neither included the top-level `finn-rtllib`, `custom_hls` or `finn_xsi`
trees. Both were named `finn-0.0.0-py3-none-any.whl` in this environment.

The archive was `git archive`, **not a setuptools-generated sdist**. Therefore
this is evidence of source-form-dependent resource inclusion in the tested build,
not proof that the project's sdist-to-wheel path fails. That path needs its own
acceptance test. The version result likewise motivates a version/provenance audit,
not a claim about every historical distribution.

Without `FINN_ROOT`, installed `fifo_rtl_files()` raises an error telling the user
to launch Docker correctly. Setting it to the installed package root yields a
nonexistent FIFO path. This demonstrates that changing the value of the variable
cannot repair missing package resources.

**Destination:** define package-owned resource families and explicit inclusion,
then expose resource access/materialization without a checkout root. For tools
that retain input paths after the Python call, stage resources into the build
workspace; a temporary resource extraction lifetime is not a durable build input.

## 4. Build workspaces: persistence, reuse and relocation

`DataflowBuildConfig.output_dir` already identifies final outputs. Separately,
`build_dataflow_cfg()` directly reads `FINN_BUILD_DIR` for intermediate work.
`make_build_dir()` allocates random directories beneath it and is used across
codegen, IP projects, simulation and driver generation.

The useful requirement is that intermediates survive long enough for debugging,
reuse and resume. The global environment variable is the current implementation.
Its error message describes a Docker launch requirement even for native callers.

The paths are persisted in the model. The largest selected fields are:

- `code_gen_dir_ipgen`: 93 get/set sites.
- `code_gen_dir_cppsim`: 36.
- `ipgen_path`: 28; `ip_path`: 23.
- `vivado_stitch_proj`: 16; `rtlsim_so`: 11; `executable_path`: 9.
- Other fields include wrapper/bitfile paths, driver directories and sidecar data.

These counts include producers and consumers and do not represent distinct build
products. The exact fields and call-site lists are in the JSON inventory.

`PrepareIP` and `PrepareCppSim` reuse existing directories. `HLSSynthIP` checks
existing paths before synthesis. Checkpoint resume loads an intermediate model;
it does not reconstruct an independent artifact graph from the final output
folder. Copying only a checkpoint cannot promise resumability.

There is an explicit opposing transformation: `ReplaceVerilogRelPaths` converts
relative memory-file references to absolute paths to support the current simulation
and stitching contexts. The synthetic probe produced:

```text
Before:     $readmemh("./memory.dat", ...)
After:      $readmemh("/original-project/memory.dat", ...)
Checkpoint: ipgen_path = /original-project

Move to /relocated-project and remove original:
  memory.dat exists in the new location
  HDL and checkpoint still reference the old location
```

This is confirmed by running the actual transformation, not inferred from its
name. No vendor execution was needed. Separately, the stitched-IP export copies
the project directory but has an explicit TODO about copying all IP sources.
That copy is not evidence of a self-contained exported build.

Sources: [build entry](../src/finn/builder/build_dataflow.py),
[build configuration](../src/finn/builder/build_dataflow_config.py),
[PrepareIP](../src/finn/transformation/fpgadataflow/prepare_ip.py),
[PrepareCppSim](../src/finn/transformation/fpgadataflow/prepare_cppsim.py),
[HLSSynthIP](../src/finn/transformation/fpgadataflow/hlssynth_ip.py),
[path rewriting](../src/finn/transformation/fpgadataflow/replace_verilog_relpaths.py),
[export](../src/finn/builder/build_dataflow_steps.py).

**Destination:** a build workspace owns source/resource snapshots, generated files,
logs and artifact references. Resolve relative artifact identifiers against an
explicit workspace root. Decide separately which deliverables must be relocatable:
final deployment files, exported source projects, resumable checkpoints and native
compiled caches have different requirements. Do not promise that one copying rule
makes all four portable.

**Weight:** high. This crosses serialization, reuse/invalidation, codegen and
parallel transformations. `PrepareCppSim` uses multiprocessing; other preparation
steps use QONNX `NodeLocalTransformation`. A future explicit build configuration
must reach workers through supported arguments/state, not a thread-local object
that happens to work in one execution mode. Old checkpoints need a compatibility
reader, an explicit invalidation policy, or a documented migration boundary.

## 5. Toolchain information: selection is more than PATH

`XILINX_VIVADO` currently carries at least four facts:

- An installation root for executable setup.
- Locations of resources such as `glbl.v` and XSI headers.
- A version encoded in the directory name.
- A root for native simulation libraries.

Version parsing affects more than tool launch. `CallHLS` selects `vitis_hls` versus
`vitis-run`; C++ simulation selects include/library paths; XSI selects a simulation
kernel name; `outer_shuffle` uses version-dependent pipeline assumptions in a
buffer-size computation. A toolchain design that contains only executable paths
would miss algorithm/code-generation inputs.

Several sites parse the version directly from an installation path even though
`get_vivado_version()` exists. Missing/non-versioned paths are not robustly handled
at every site. The feature thresholds differ; that is not by itself proof of a
bug, because they describe different vendor changes. Represent version/capabilities
explicitly and test each feature transition rather than mechanically unifying them.

There is an existing seam: `resolve_xilinx_tool()` allows a site-owned command
directory through `FINN_TOOL_DIR_OVERRIDE`. CI documents its use for compute-farm
wrappers. Absolute local executable paths must not silently remove remote dispatch.
`launch_process_helper()` already accepts an explicit environment and working
directory. Many older flows instead write unquoted shell command strings and
inherit the full process environment; seven literal reads of `PWD` help generate
scripts that change into a build directory and then back again.

Sources: [tool helpers](../src/finn/util/basic.py),
[HLS invocation](../src/finn/util/hls.py),
[C++ backend](../src/finn/custom_op/fpgadataflow/hlsbackend.py),
[version-dependent algorithm](../src/finn/custom_op/fpgadataflow/outer_shuffle.py),
[site dispatch contract](../ci/README.md).

**Destination:** explicit toolchain selection/version/capabilities and a command
execution boundary carrying argv, cwd, scoped environment and logs. Local tool
execution and site dispatch can implement that boundary. Keep human-invoked bare
vendor commands as a separate convenience decision; FINN's internal correctness
should not depend on transparent shell interception.

## 6. Native simulation is a separate integration obligation

The XSI path combines:

1. FINN-owned C++ sources and a Python adapter under the top-level `finn_xsi` tree.
2. Python and pybind11 headers, a C++ compiler and Vivado XSI headers.
3. A compiled `xsi.so`, currently looked up by path/existence/importability.
4. Vendor simulation kernels loaded by name through `dlopen(..., RTLD_GLOBAL)`.
5. Per-design compiled simulation objects and generated source lists.

The Python adapter and C++ helper are not an ordinary installed package in the
current FINN wheel. Setup and loading modify `sys.path` to combine source and
artifact directories. `FINN_XSI_BUILD_DIR` overrides the artifact location; otherwise
it uses `FINN_BUILD_DIR/finn_xsi`, with `/tmp/finn_xsi` as another fallback. The
legacy source-tree binary is still accepted.

Moving the binary out of the source tree was useful, but its filename/location
is not a compatibility key. The reuse check does not explicitly compare source
revision, Python ABI, compiler flags or selected Vivado headers. Importability
alone cannot establish that a cached extension matches a new toolchain.

The standalone C++ RTL-simulation flow already scopes `LD_LIBRARY_PATH` to its
simulation command in a generated script. The Python XSI flow loads vendor
libraries inside the interpreter. These are different boundaries: a generic
vendor-executable wrapper cannot prepare an already-running Python loader. A
future design must choose an isolated simulation worker process, a tested explicit
loader strategy, or another well-defined native-runtime interface.

Sources: [XSI paths](../src/finn/xsi/paths.py),
[extension setup](../src/finn/xsi/setup.py),
[Python loading](../src/finn/xsi/__init__.py),
[adapter](../finn_xsi/finn_xsi/adapter.py),
[C++ loader](../finn_xsi/xsi_finn.cpp),
[simulation execution](../src/finn/core/rtlsim_exec.py).

**Weight:** medium source surface, high compatibility/testing burden. Packaging
sources and designing a correctly keyed build cache can be separate work items.
Real vendor validation is required before removing loader workarounds.

## 7. Genuine external inputs and runtime concerns

### Licence and platform configuration

No literal application reads of `XILINXD_LICENSE_FILE` or `LM_LICENSE_FILE` were
found in the scoped Python. The Docker resolver classifies/passes them, mounts
licence directories, and reports network requirements; native examples use explicit
site configuration. Vendor processes consume the licence settings. Network policy
remains a runtime/site responsibility and is independent of how those values reach
a process.

Vitis checks for `VITIS_PATH`, `PLATFORM_REPO_PATHS` and `XILINX_XRT`; linking also
passes the selected platform as a command option. These are related but distinct:
platform selection, platform search locations and installed accelerator runtime.
The current presence checks do not constitute a complete capability validation.

Sources: [host configuration](../docker/config.py),
[native FPGA example](../docker/sbx/fpga.sbxenv.yaml),
[Alveo build](../src/finn/transformation/fpgadataflow/alveo_build.py),
[configuration checks](../src/finn/builder/build_dataflow_checks.py).

### Writable storage and identity

No direct `HOME`, `USER` or `LOGNAME` reads appeared in the scoped application
inventory. Writable home, ownership and caches still matter to shells, notebooks,
agents and vendor software. This is runtime setup, not a FINN resource-discovery
requirement. Keep it explicit and independently test arbitrary-UID support if that
is a supported product contract.

The entrypoint defaults `HOME`, scratch and root, appends a user bin directory to
PATH and seeds optional `.Xilinx` files behind a marker. That is filesystem
preparation plus environment mutation. The marker also makes Tcl seeding a one-time
copy rather than reconciliation of current site configuration. Treat that feature
as explicit site/vendor initialization in a redesign.

### Execution controls

Worker count, xelab thread count, simulation timeouts and trace depth are real job
or algorithm controls. They should not be conflated with resource-path repair.
An explicit job/build configuration can carry them while a compatibility adapter
continues to accept existing environment inputs during transition.

## 8. What the six files are currently compensating for

| Mechanism | Actual obligation | Candidate fate after prerequisites |
| --- | --- | --- |
| `finn_entrypoint.sh` | Writable runtime state, defaults, optional site Tcl | Reduce to necessary filesystem preparation or delegate to runtime |
| `finn-bashenv.sh` | sbx persistent environment plus global toolchain activation | Preserve native sbx integration as needed; remove FINN activation once internal execution is explicit |
| `finn-toolchain.sh` | Vendor settings, library additions and compatibility state | Contain at vendor/native installation boundary; remove responsibilities demonstrated unnecessary |
| `toolchain-shim` | Make bare vendor commands work without activation | Optional interactive convenience, removable if deliberately unsupported |
| `finn-live.pth` | Register Python startup interception | Remove after explicit FINN/dependency installation works |
| `finn_paths.py` | Mounted source selection plus root/scratch repair | Replace installation behavior; move build/resource obligations to their owners |

The toolchain script uses a process-tree-wide `FINN_ENV_APPLIED=1` latch. In the
synthetic probe, switching the selected installation and sourcing again retained
the first setup (`one|1`). It is idempotence for one environment, not a configuration
cache or support for multiple toolchains in a long-lived process.

The Python probe confirmed that loading `finn_paths.py` without a workspace still
sets and creates a default scratch directory. Conversely, an explicitly supplied
nonexistent build directory is not created by `ensure_build_dir()`. Its name and
comments therefore promise more than the function does in that case.

Other comments also need scrutiny: Python startup does not apply FPGA settings;
`BASH_ENV` is not a universal interactive-shell hook; “fixed workspace path makes
this module unnecessary” overlooks development dependency selection and incomplete
packaging. These comments are historical explanations, not acceptance criteria.

## 9. Proposed boundaries to take into design

These are responsibilities, not a requirement to introduce a framework or a large
new configuration object.

```text
Installed FINN + versioned resource dependencies
       |
       +-- package-relative lookup for immediate reads
       +-- stage durable resource inputs into build workspace

Explicit build request
       |
       +-- model + configuration + selected toolchain
       +-- workspace root and artifact identifiers
       +-- generated recipes/logs/checkpoints/export manifest
       |
       +-- tool execution: argv + cwd + scoped environment
       +-- native simulation integration / compatible artifact cache

Development installation
       +-- explicit editable source/dependency selection

Container/native runtime
       +-- identity, mounts, writable storage, network policy, credentials
```

The strongest first design experiment is a **wheel-only, no-checkout resource
operation followed by a small generated project that can move**. It attacks the
actual obligations instead of optimizing bootstrap paths.

Avoid these premature conclusions:

- A fixed `/workspace/finn` path does not fix packaging or persisted artifacts.
- Moving all variables into a JSON file does not fix ownership or hidden lookup.
- A single activation script cannot initialize independent runtime exec processes.
- Packaging resources does not make references to generated intermediates portable.
- Removing all environment variables is not a useful goal: vendor programs and
  loaders have real environment interfaces. Scope them rather than reinvent them.
- An installed application image and a reusable development dependency image need
  not have the same source identity or setup contract.

## 10. Work packages, deferrals and removal gates

| Work package | Relative weight | Exit evidence / removal gate |
| --- | --- | --- |
| Explicit resource ownership and wheel contents | Medium; broad but largely mechanical call-site migration | Wheel from Git and sdist locates required resources outside checkout; no `FINN_ROOT` for resource lookup |
| Generated recipe inputs and resource staging | Medium-high | C++/Tcl/HDL recipes declare inputs; replay from another cwd without FINN environment variables |
| Build workspace and persisted artifact references | High; crosses models, checkpoints, workers and reuse | Move a workspace, resume representative steps, verify sidecars and references; define old-checkpoint policy |
| Toolchain selection/capabilities/execution | Medium-high | Versioned local and fake/site-dispatch tests; scoped env; two toolchains do not contaminate each other |
| XSI packaging and compatibility-keyed cache | High validation burden | Concurrent build safety, ABI/toolchain change invalidation and real XSI simulation |
| Explicit development installation | Medium | Editable FINN and selected editable dependencies work from arbitrary cwd; imported code and metadata agree |
| Startup-hook retirement | Low implementation after gates above | Required Docker/native/sbx/SIF invocation matrix passes without hidden FINN startup initialization |

Reasonable deferrals have explicit limits:

- Keep legacy absolute checkpoint paths readable for a defined transition; do not
  promise relocation for old checkpoints or silently reuse missing artifacts.
- Keep an optional bare-tool shim until the internal execution boundary is complete;
  do not make its transparent behavior an unquestioned permanent requirement.
- Retain the vendor loader workaround until tested across supported versions; source
  comments alone do not justify removing or globally preserving it.
- Defer native compiled-cache relocation while making source/build manifests portable;
  invalidate incompatible binaries rather than pretending they are portable data.
- Retain a development dependency image while designing installed application images;
  avoid sacrificing source/image provenance merely to eliminate a file quickly.

No duration estimates are justified yet. The counts support relative scope, but
vendor compatibility and checkpoint semantics dominate the uncertainty.

## 11. Validation needed before implementation decisions

Existing tests cover host discovery, shell quoting, toolchain idempotence, tool
resolution, image identity and native/Docker invocation. Those tests protect the
current interface; some assertions, such as forced FPGA path mirroring, will need
intentional revision when the underlying behavior changes.

New design acceptance cases should cover:

1. Installed wheel outside the source checkout; resources available with no FINN
   startup hook or root environment variable.
2. Wheel-from-sdist resource/version checks, separately from Git checkout builds.
3. Explicit editable installs from different directories; dependency code/metadata
   agreement; no source mutations during normal Python startup.
4. Replay generated scripts from a different cwd with no inherited FINN variables.
5. Move a build workspace, remove its original location, then resume and simulate;
   include HDL memory files, source lists, model sidecars and stitched IP inputs.
6. Two build workspaces in one process and multiprocessing workers; no shared global
   scratch or toolchain state leaking across builds.
7. Local and site-dispatched tool commands, explicit versions and feature transitions,
   failures attributed to the actual command rather than a missing output later.
8. XSI subprocess and in-process loading, source/ABI/toolchain cache invalidation,
   and actual licensed vendor operations where applicable.
9. Required runtime entry paths and arbitrary UID behavior, selected deliberately
   after internal correctness no longer depends on shell initialization.

These experiments did not establish vendor project relocation, a working wheel-only
FPGA build, real licence checkout, XSI ABI compatibility, SIF behavior or the contents
of site-owned dispatch wrappers. Those remain explicit design/validation work.
