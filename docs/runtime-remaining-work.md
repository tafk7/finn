# Task spec: container/runtime work from P6 onward

Written 2026-09-26, on `feature/external-resources` (8 commits on
`refactor/container-runtime-implementation` at `09fb86df7`). It covers what is
left after the environment, packaging, toolchain, simulation-boundary and
external-resource work: integration with the private build engine (P6),
deleting compatibility mechanisms (P7), validation that needs AMD tools or other
infrastructure (P8), publishing, and preparing the history for upstream review.

Current state: [CONTAINER_RUNTIME_STATUS.md](../CONTAINER_RUNTIME_STATUS.md).
Obligations and their gates: [legacy-build-env-ledger.md](legacy-build-env-ledger.md).
Evidence so far: [runtime-validation.md](runtime-validation.md).

## Working rules

These carry over from the earlier phases.

* One or more commits per task, each leaving the tests passing, so work can stop
  at any task boundary.
* Nothing is pushed, merged or proposed upstream without explicit approval.
* Every validation result, including a failure, goes into
  `runtime-validation.md` with the date, host, revision and exact tools.
  `CONTAINER_RUNTIME_STATUS.md` is updated when a phase finishes.
* A mechanism is removed only when its gate in the ledger has real evidence.
  Evidence from fakes or synthetic tests does not count for gates that name AMD
  tools, licences or devices.
* `src/finn/builder/` and `_legacy_build_env.py` stay unchanged until P6 works
  from the build engine's real interfaces; no API is inferred.

## Order and dependencies

```text
 D0 decide external resources ──► P6 build-engine integration ──► P7 deletions
                                        ▲                             ▲
 P8 validation (AMD tools, CI, SIF) ────┴── evidence for gates ───────┘
                                                                      │
                                 R history reorganization ◄───────────┘
                                 U publishing (independent; external releases)
```

P8 needs no code from P6 and can start as soon as a host with the tools is
available. Several P7 deletions are gated on P8 evidence as well as on P6.

## D0. Decide on external resources

`feature/external-resources` replaces the finn-hlslib submodule and workspace
package with `finn.resources` (plan and records in
[external-resources-plan.md](external-resources-plan.md)).

* **Accept:** merge it into `refactor/container-runtime-implementation`, or keep
  it as its own group in the reorganized history (R).
* **Abandon:** carry this spec over to the base branch and drop the resource
  items below.

Its one unverified assumption, Vivado with several board repository paths, is
P8.1. Accepting before that check is reasonable, because the fallback (one
directory of links to the resource roots) is local to `make_zynq_proj.py`.

## P6. Integrate the private build-engine branch

**Precondition:** access to the private build-engine branch. It is not available
on the host used so far, and nothing here may assume its API.

**Goal:** builds run in-process through the build engine, with explicit inputs
at its real boundaries, instead of through today's compatibility layer.

**What exists to connect to:**

| Input | Permanent interface today | Legacy path it should replace |
| --- | --- | --- |
| FINN's own data | `finn.util.resources.resource_path(family, ...)` | none left |
| External resources | `finn.resources.path(name)`, `paths(kind)` | none left |
| Tool selection | `finn.util._toolchain.Selection` / `Toolchain`, passed to `CallHLS`, `CreateStitchedIP`, the Zynq/Vitis/SLASH operations and FINNLoop | `_legacy_build_env.toolchain()` translating `XILINX_*`, `*_PATH`, `FINN_HLS_FRONTEND` |
| Build scratch | `make_build_dir`, `FINN_BUILD_DIR` | `_legacy_build_env.build_directory()` |
| Simulation | `finn.xsi._session.run_session` (separate process, explicit request) | the live-object simulation and C++ harness paths |
| Whole build | none yet | `build_dataflow_directory` starting a fresh interpreter with `build_environment()` |

**Tasks:**

1. Read the build engine's actual API and write down, in the ledger, which of
   the inputs above it takes where. Record any mismatch instead of adapting to
   it silently.
2. Connect the explicit inputs at those boundaries: resources, tool selection,
   scratch and simulation. Operation and board registration by kind
   (`hls-include`, `vivado-boards`, `rtl` and others) belongs to the builder;
   use `finn.resources.paths(kind)` for it rather than new lookups.
3. Remove or reconcile the whole-build subprocess in
   `build_dataflow_directory` (`src/finn/builder/build_dataflow.py`). It exists
   only to isolate the working directory and the native loader environment
   before Python starts. Keep a process boundary only where the loader
   environment still requires one (see
   [xsi-process-boundary-investigation.md](xsi-process-boundary-investigation.md)).
4. Retire the internal legacy-variable interpretation the engine replaces. The
   modules that import `_legacy_build_env` are the list to work through:
   `builder/build_dataflow.py`, `custom_op/fpgadataflow/hlsbackend.py`,
   `custom_op/fpgadataflow/rtl/finn_loop.py`, and in
   `transformation/fpgadataflow/`: `alveo_build.py`, `create_stitched_ip.py`,
   `make_driver.py`, `make_zynq_proj.py`; and `util/basic.py`, `util/hls.py`,
   `xsi/compile.py`, `xsi/paths.py`. Each removal updates the ledger row and its
   test.
5. Keep `build_dataflow project/` and `build_dataflow_cfg` working for existing
   users, and document any behaviour change in `installation.md`.

**Acceptance:**

* A CLI build (`build_dataflow project/`) and a Python build
  (`build_dataflow_cfg`) both complete through the selected tools and the
  simulation-session boundary.
* A checkpoint resumes under the documented constraints (same installation,
  build tree and toolchain).
* No FINN module outside the adapter reads `FINN_ROOT`, and the
  root-interpretation allowlist test still passes.
* The ledger lists what remains and why.

The end-to-end parts of these checks need a Vivado host, so they are recorded in
P8.

## P7. Delete the remaining compatibility mechanisms

One commit per mechanism, each with the evidence that satisfied its gate cited
in the commit message and in the ledger.

| Mechanism | Where | Gate |
| --- | --- | --- |
| Whole-build Python worker | `build_dataflow_directory` | P6 task 3 |
| Broad legacy activation | `src/finn/util/_legacy_build_env.py` (`checkout_root`, `build_directory`, `toolchain`, `build_environment`) | Every consumer takes explicit inputs (P6 task 4) |
| Bare-tool shims | `docker/toolchain-shim`, `docker/finn-toolchain.sh`, `FINN_ENV_APPLIED` | All FINN vendor callers use explicit dispatch; bare vendor commands in containers documented; real licensed runs (P8.2) |
| Global libudev preload | `ENV LD_PRELOAD` in `docker/Dockerfile.finn` | A scoped replacement passes a real licence checkout (P8.2) |
| Legacy simulation paths | live-object simulation, the C++ harness in `core/rtlsim_exec.py` | Session equivalence, lifecycle and overhead on real Vivado (P8.3) |

`FINN_HLSLIB_PATH` (the alias for `FINN_RESOURCES_HLSLIB`) can be removed in the
same phase if no user depends on it; announce it one release ahead, as for
`FINN_BOARD_FILES_PATH`.

**Acceptance:** the ledger's "Remaining input/mechanism" table holds only items
that are deliberately kept, each with a reason; the unit and conformance suites
pass.

## P8. Validation needing AMD tools or other infrastructure

Needed: a Linux host with Vivado/Vitis 2024.2 and a licence (node-locked and a
floating server, if possible), an FPGA board or Alveo card for device checks,
Apptainer, and access to run the GitHub and Jenkins pipelines.

| # | Check | Pass condition |
| --- | --- | --- |
| 8.1 | Zynq project with five board repository paths (`resources.paths("vivado-boards")`) | Vivado lists the boards from each resource and builds a project for a KV260 or AUP-ZU3 board part. If it fails, switch to one directory of links to the resource roots. |
| 8.2 | HLS C++ simulation and synthesis with the fetched finn-hlslib; licence checkout, natively and in the Ubuntu 24.04 image | `quicktest-local.sh vivado` and the HLS layer tests pass; checkout works with and without the libudev preload |
| 8.3 | `finn_xsi` first-use build, RTL simulation, session equivalence, lifecycle and overhead | Identical outputs and cycle counts to the legacy path on the layer tests; concurrent first uses build once |
| 8.4 | Vivado 2024.2 on 24.04 without the ncurses 5 libraries | Synthesis and simulation run; otherwise restore the libraries in the image |
| 8.5 | XRT 24.04 package install and a device run | The XRT image variant builds and runs an Alveo test |
| 8.6 | SIF export and read-only execution | `docker/build --export-sif` produces a SIF that runs `build_dataflow --help` and a resource lookup under Apptainer |
| 8.7 | Dev Container through VS Code | Opens with uid remapping; `/opt/venv/bin/python` imports FINN from the workspace |
| 8.8 | GitHub workflows and Jenkins as configured | Package, quicktest and Jenkins stages pass on the branch, including the wheel-only hlslib fetch and the native setup stage's persistent resource cache |
| 8.9 | P6 end-to-end acceptance | The builds and checkpoint resume from P6 on real tools |

## U. Publishing to PyPI

* A QONNX release that contains the four commits past 1.0.0 that FINN uses (the
  MaxPool `ceil_mode` fix). Then drop the git source for QONNX and relock.
* A decision on redistributing the board files (XilinxBoardStore, Avnet,
  RealDigital). If they may be redistributed, set `redistributable = true` on
  those declarations; published images then include them. No code change is
  needed either way.
* The Package workflow's published-metadata check passes against the release.

## R. Reorganize the history for upstream review

Rebuild the history on a new branch; the existing branches stay as records.

1. The move of FINN's own sources into packages (`finn.rtllib`, `finn.custom_hls`, `finn.xsi`, `finn.deploy`) as its own rename commit.
2. One commit, or a short series, per migration step: packaging, dependencies
   and lock, image, entrypoint, toolchain selection, simulation session, CI.
3. External resources as its own group, if D0 accepted it.
4. The upstream `dev` merge resolved inside the rebuilt history rather than
   kept as a merge commit, unless reviewers prefer the merge.

**Acceptance:** each commit builds and passes the unit suite. The final tree is
identical to the branch it was rebuilt from (`git diff` is empty).

## Open questions for the owner

1. How and when will the private build-engine branch be available, and is its
   API stable enough to integrate against?
2. Which host and licence setup should P8 use, and who runs the Jenkins
   pipeline?
3. Should P7 announce `FINN_HLSLIB_PATH` as deprecated now, or keep the alias
   indefinitely?
4. For R, one upstream pull request or a series?
