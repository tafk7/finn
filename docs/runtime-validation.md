# Runtime implementation validation

## External resources (2026-09-26)

Branch `feature/external-resources` from `09fb86df7`: finn-hlslib and the board
files as external resources (`finn.resources`), replacing the finn-hlslib
workspace package, its submodule and `finn.util.external`. Same host as below.

| Check | Result |
| --- | --- |
| Pins | The six declarations fetched from GitHub: hlslib 187 files (the submodule's `git ls-files` count), board repositories 227 + 6 + 6 + 14 + 5 files. The five board trees together: 258 files, digest `89ebe804...`, identical to the previous assembled tree. |
| Without git | `fetch --all` with no `git` on PATH: all six from GitHub's commit archives in 26 s, every tree matching its declared digest. |
| Native | `tests/util` and `tests/transformation` without Vivado: 2063 passed, 11 skipped, 4 xfailed, 1 xpassed. `tests/util/test_resources.py` (39 tests, no network): local git repositories and archives, sparse `subdir`/`into`, digest mismatch publishing nothing, cache order with a read-only system cache, overrides and compatibility variables, four processes' first use fetching once, redefinition rules, entry points in a temporary venv, `update` against a local repository, mirrors, the git-less fallback, and `finn.resources` importing only the standard library (`python -S`). |
| Lock | `uv lock` removes only `finn-hlslib`. |
| Wheel | Contains `finn/_data/resources.toml` and `finn/resources`; installed with no dependencies at all into a bare venv, `finn-resources list` shows the declarations and `finn-resources path hlslib` fetches it. |
| Offline | `fetch hlslib --dest DIR`; then with a fresh home, `FINN_RESOURCES_OFFLINE=1` alone is an error naming the fetch command, and with `FINN_RESOURCES_CACHE=DIR` resolves without network. |
| Images | Default target `dev` (`img-501a4001f1ac58da`, 3.65 GB): all six resources in `/opt/finn/resources`. `release` (3.63 GB): only hlslib. sbx builds on `dev`. With `--network none` as uid 1001, the dev image resolves and verifies all six from the system cache; the release image resolves hlslib, and board files fail with a network error (fetched on first use). |
| Containers | Through `docker/run`: FINN from the checkout, resources from the system cache; runtime, resource, installation and configuration tests inside the image: 131 passed, 1 skipped. Conformance suite against real Docker and sbx: 14 passed, 4 skipped (Xilinx toolchain, node-locked licence, Apptainer). |

Not validated here: Vivado finding boards across the five board repository paths
(each at the same depth as in the previous single directory); HLS with the fetched
finn-hlslib; the GitHub and Jenkins pipelines as configured.

## Upstream `dev` merge: Python 3.12 / Ubuntu 24.04 (2026-09-25)

Merge of `upstream/dev` at `b507fec58` (134 commits, including the Python 3.12 /
Ubuntu 24.04 upgrade) as `06939f7c5`, then `e307fec77` and the documentation
commit. Same host as below.

| Check | Result |
| --- | --- |
| Lock | Reproduces upstream's pinned versions (numpy 1.26.4, onnx 1.22.0, onnxruntime 1.28.0, scipy 1.11.4, pytest 7.4.4, pytest-html 4.1.1, pandas 2.1.4, QONNX `4c1f9b44`, ...), torch 2.8.0 CPU; unchanged when the temporary pins are removed. |
| Python range | 3.10 does not resolve (QONNX caps onnx at 1.17 on 3.10; onnxruntime 1.28 requires 3.11). On 3.11 at the locked versions the unit suites pass (2028 passed). Chosen: `>=3.11,<3.13`. |
| Native, 3.12 | `tests/util` and `tests/transformation` without Vivado: 2028 passed. The whole suite (33410 tests) collects. |
| Image | `ubuntu:noble-20240605`, Python 3.12.3, g++ 13.3, no ncurses 5; 3.65 GB (3.82 GB with XRT); build-time check over 191 packages; `agent` at uid 1000 with the base's `ubuntu` user removed. |
| Container | uid 1001: FINN editable from the checkout, hlslib from the submodule, the moved `requantf` library found; runtime tests inside the image: 115 passed. |
| Found and fixed | `onnxoptimizer` restored (Brevitas export imports it; dropped earlier because the transformation tests had not been run); three tests that still used `FINN_ROOT/finn-rtllib`; the portable-RTL export's detection of RTL library files; `tcl_quote` in upstream's new `add_files` code. |

Container conformance suite on the 24.04 image, against real Docker and sbx: 14
passed, 4 skipped (Xilinx toolchain, node-locked licence, Apptainer). This builds
the XRT variant: the 24.04 XRT package (2.18.179) downloads, matches its checksum
and installs.

Not run here: anything needing AMD tools (including Vivado 2024.2 on 24.04
without ncurses 5, and XRT against a device), and the CI pipelines themselves
(the new 3.11 job's steps were run locally).

## Environment simplification (2026-09-25)

Migration from the dependency/application image split and offline wheelhouse to
`pyproject.toml` + `uv.lock`, one image with an active venv, and a startup editable
install; design in [environment.md](environment.md). Commits `33a749f7a` through
the documentation commit on `refactor/container-runtime-implementation`, after the
checkpoint `5dd9df9bc`. Host: Ubuntu 24.04 VM, 2 CPUs, Docker 29.8.0, sbx v0.42.1,
uv 0.10.0; no AMD tools, Apptainer or Dev Container CLI.

| Check | Result |
| --- | --- |
| Lock versus the previous image | `uv.lock` reproduces every version installed in `xilinx/finn:app-a86b006396bd9277`; the only differences are packages nothing imports, removed on purpose (finn-experimental, deap, mip, gspread, psutil, toposort, vcdvcd, onnxoptimizer, torchaudio, pyscaffold, setupext-janitor, build tools). |
| Native | `setup-local.sh` on the checkout and on a fresh clone (about 12 s with a warm uv cache; `.venv` 1.5 GB); FINN and finn-hlslib editable; `scripts/activate.sh`. |
| Wheels | finn wheel: 328 files, 132 under `finn/_data`, build provenance, `build_dataflow` entry point. finn-hlslib wheel: 191 files, no git metadata. Both used from a clean lock-synced venv with no checkout (resource smoke test, entry point, `hlslib_path`). Published metadata resolves from PyPI (dry run); note it resolves QONNX 1.0.0, not the development commit. |
| Board files | Sparse single-commit fetches in about 6 s; the assembled tree is byte-identical to the one `fetch-repos.sh` produced (258 files, digest `89ebe804...`). |
| Image | 3.62 GB (previous application image 5.02 GB). Build-time `uv pip check` over 199 packages, import smoke test and pytest probe. |
| Container start | uid 1001 (no passwd entry): FINN editable from the mounted checkout, hlslib from the submodule, board files from the image. Start with `FINN_SYNC=0` 0.4 s; with sync 4 s cold, 0.6-1.1 s with the uv cache in `FINN_BUILD_DIR`. Changed `uv.lock`: installs the difference online; offline warns and starts. No checkout: note, starts. Uninitialized submodule: warning at start, actionable error at use. |
| Dev Container | Compose service from `initialize.sh` (without the extension's uid remapping): ready in about 4 s, `/opt/venv/bin/python`, FINN from `/workspace/finn`. |
| Tests, native | `tests/util`: all pass except tests needing Vivado/HLS (npy stream C++ tests, end-to-end `build_dataflow`, the FPGA flow tutorial, C++ simulation). New: external data, XSI first-use build (including four concurrent first uses building once). |
| Tests, containers | Conformance suite against real Docker and sbx: 14 passed, 4 skipped (need a Xilinx toolchain or Apptainer). Runtime tests inside the image pass (a host-only `docker buildx` test excepted). |

Not validated here: anything needing AMD tools (HLS, synthesis, XSI build and
simulation, licences), SIF export and execution, the Dev Container through VS Code,
the GitHub/Jenkins/Read the Docs pipelines themselves (their steps were run
locally), and the Brevitas CI job.

## Approved implementation: P0 baseline (2026-09-20)

Starting revision `97eb24645`, clean worktree on
`refactor/container-runtime-implementation`. The exact 92-path prototype disposition
is in `docs/container-runtime-inventory.md` (removed; in history at `5dd9df9bc`); the actual
consumer audit is in [legacy-build-env-ledger.md](legacy-build-env-ledger.md).
Private build-engine interfaces remain unavailable; no builder implementation was changed.

Infrastructure checked now: Docker daemon 29.8.0 responds; native sbx v0.42.1
(`cc6e400a4a3ce3ce5e0b2b77b8ee352aac854c64`) is installed and `/dev/kvm` exists.
Existing `finn-runtime-plan:validation` (`9f27e41796af`) and `:sbx`
(`f80eed01a233`) images are historical artifacts, not newly validated deliverables.
No AMD executable is on PATH, no configured vendor/licence variable is set, and
`/opt/Xilinx`, `/tools/Xilinx`, `/opt/amd`, `/tools/amd` are absent.
Apptainer/Singularity are absent. Licensed builds, native XSI and SIF remain open.

Baseline uses the existing isolated `/tmp/finn-runtime-venv` with CPython 3.12.3,
FINN 0.11.0.dev0 editable at this checkout, QONNX
`f5c9819bd00f01f41e70639b8461c8e4b39432f7`, pytest 9.1.1, setuptools 84.0.0,
wheel 0.48.0 and build 1.6.1. This is supplemental host coverage, not the supported
Python 3.10 image environment. No implicit dependency provisioning occurred.

```bash
/tmp/finn-runtime-venv/bin/python -m pytest -q \
  tests/util/test_runtime_toolchain.py tests/util/test_runtime_codegen.py \
  tests/util/test_xsi_pkg_ordering.py \
  tests/util/test_container_config.py tests/util/test_container_cli.py \
  tests/util/test_ci_container_transport.py
```

150 passed in 41.73 s (`/tmp/finn-p0-focused.log`). The pre-existing suite includes
prototype worker assertions: passing them does not endorse that architecture, and
new components must not depend on it. Installation journeys initially passed four
tests; the explicitly selected `deps/qonnx` checkout did not exist. Provisioned
`/tmp/finn-implementation-qonnx` at the exact revision above for the editable test.
The initial command used two nonexistent test filenames and collected no tests;
the corrected command above is the measured baseline.

Independent setup correction: `setup-local.sh` now invokes normal XSI setup, which
checks/reuses or builds the extension and verifies it. `--check` only tests
prerequisites. Removed obsolete `FINN_DEPS` from Dev Container Compose.

## Independent implementation delivery (2026-09-20)

P0 inventory is complete; exact private-branch overlap is still conditional on
that branch becoming available. The corrected QONNX baseline retry passed in
11.23 s, for 155 baseline tests in total. P1 then passed 86 focused tests, including
wheel/sdist resource, metadata and Git-provenance equality, hidden source copies,
read-only package data, unrelated cwd and explicit editable FINN/QONNX selection.

The final combined run passed **181 tests in 115.70 s**: runtime installation,
codegen/toolchain, resolver, XSI source ordering and session tests, SLASH dispatch,
wheelhouse closure, Docker configuration/CLI, CI transport, plus two non-vendor
container conformance checks. After removing the duplicate Compose build surface,
the affected Docker/CI suites passed **117 tests in 40.60 s**. The final read-only
inspection/activation fix passed **65 resolver tests in 3.55 s**, including the
new no-scratch-allocation regression. This is 182 distinct tested cases overall. Ruff 0.15.10,
Black 23.3.0, isort 5.12.0, shell syntax, ShellCheck and whitespace checks were used
on delivered implementation files. Logs are local evidence, not shipped artifacts:
`/tmp/finn-implementation-final-tests.log` and
`/tmp/finn-implementation-final-compose-tests.log` and
`/tmp/finn-implementation-inspection-tests.log`. A real host g++ compilation and
execution with literal spaces/substitution characters in paths also passed with
unchanged parent cwd/environment.

```bash
FINN_TEST_QONNX_CHECKOUT=/tmp/finn-implementation-qonnx \
/tmp/finn-runtime-venv/bin/python -m pytest -q \
  tests/util/test_runtime_installation.py tests/util/test_runtime_toolchain.py \
  tests/util/test_runtime_codegen.py \
  tests/util/test_xsi_pkg_ordering.py tests/util/test_xsi_session.py \
  tests/util/test_slash_link.py tests/util/test_wheelhouse_manifest.py \
  tests/util/test_container_config.py tests/util/test_container_cli.py \
  tests/util/test_ci_container_transport.py \
  tests/container/test_container_conformance.py::test_05b_dependency_sbx_identity_ignores_mounted_source_commit \
  tests/container/test_container_conformance.py::test_12_bake_owns_runtime_tags_and_custom_flavors
```

### Built artifacts and actual identities

Bake built the application, application+XRT, dependencies and dependency+sbx targets.
The input revision tags are not immutable build outputs; the IDs below are.
A final manifest EOF normalization changed only identity metadata: Docker inspection
confirmed identical rootfs layers and runtime environment to the runtime-tested
`app-928cd39692c62436[.xrt]`, `deps-928baf60bf600cfd` and
`sbx-deps-928baf60bf600cfd` images. Evidence:
`/tmp/finn-final-artifact-equivalence.json`. Plain Docker exec was also tested
directly against the final application tag (`/tmp/finn-final-docker-exec.log`).

| Image | Image ID |
| --- | --- |
| `xilinx/finn:app-0bf7ac24bf131246` | `sha256:2266bc3edcdb99f7b9abd9489f480239814695e21334bfd91db48ce38ed65523` |
| `xilinx/finn:app-0bf7ac24bf131246.xrt` | `sha256:01c2af859b675edecc778596529f05bcfd554e9181189a174dc526ca8a3c4502` |
| `xilinx/finn:deps-1d3775474e8454c3` | `sha256:91655e6103251f2cb9753c313c36118c1a0703da3a1a69dc6e9e74c9ae38ccdf` |
| `xilinx/finn:sbx-deps-1d3775474e8454c3` | `sha256:6880d087228ad59d1702a756d529da8acbea8b8ddc5eda842105e46cc35f8776` |

`ci/scripts/build-images.sh finn /tmp/finn-delivery-provenance` completed and
recorded the actual artifact/wheel/dependency provenance. The application wheel SHA256
is `4e1ba446eb52317fece62801089b987bd673d4a7ae372b5e5f40703bbdf567c0`.
The wheelhouse contains **220 wheels**, checksum-pinned in the offline development
manifest, including pip 24.3.1, setuptools 68.2.2, wheel 0.45.1, build 1.2.2.post1
and setuptools-scm 8.1.0. Python is 3.10 in these images. FINN is absent from the
dependency closure; finn-experimental's pinned wheel remains metadata-only due to
its existing upstream package configuration. Exact Python Git source revisions:

- `brevitas`: `aad4d5a293db6f2ec622a92a5d3278e47072453e`
- `dataset_loading`: `5b9faa226e5f7c857579d31cdd9acde8cdfb816f`
- `finn-experimental`: `0724be21111a21f0d81a072fccc1c446e053f851`
- `qonnx`: `f5c9819bd00f01f41e70639b8461c8e4b39432f7`

### Runtime journeys

- Installed application: final image, no checkout mount, network disabled,
  read-only root filesystem, writable `/tmp`, UID/GID 12345, unrelated cwd, and
  unset root/resource variables. FIFO RTL and Python drivers generated correctly.
  Entrypoint-bypassing Python inspection and generated `build_dataflow --help`
  worked; the application has no `/opt/finn/wheels`. Evidence:
  `/tmp/finn-implementation-release-smoke.log`.
- Docker development: created a user-owned `/tmp/finn-implementation-docker-venv`,
  mounted at `/env/venv`, and prepared it with ordinary package commands under
  `docker run --network none`. FINN was absent before explicit editable install.
  Installed FINN plus pinned QONNX offline; subsequent disposable containers reused
  both mounts without installation. The actual `docker/run --dependencies
  --no-build --venv ... --volume ...` path also resolved both selected packages.
  An omitted QONNX source mount produced metadata with no import path, as intended;
  restoring that explicit mount restored imports without reinstalling.
  Evidence: `/tmp/finn-p3-offline.log`, `/tmp/finn-p3-launcher-reuse.log`.
- Dev Container: tested with Dev Containers CLI 0.89.0, Docker Compose 5.5.1 and
  Node 24.21.0. Creation prepares `/home/agent/.venvs/finn-dev`; native CLI exec
  selects that exact interpreter and `/workspace/finn/src/finn/_data` resources.
  Reopening reused the same container with no pip/postCreate operation. The initial
  test exposed Compose rebuilding the Bake image: Compose now consumes prepared
  references and contains no competing build definition. Source `.venv` is not used.
  Evidence: `/tmp/finn-p3-devcontainer-final-reopen.log` and final-image creation
  `/tmp/finn-implementation-release-devcontainer.log`, with interpreter and final
  reopen evidence in `/tmp/finn-implementation-release-devcontainer-exec.log` and
  `/tmp/finn-delivery-devcontainer-reopen.log`.
- Native sbx: native plan/create, explicit sandbox-private venv preparation,
  repeated exec with selected Python/console scripts, resource generation and
  removal passed. Recreated from `sbx-deps-928baf60bf600cfd`, confirmed that its
  private venv was gone, then explicitly prepared FINN plus pinned QONNX offline.
  Native PATH/VIRTUAL_ENV configuration selected the correct interpreter in plain
  `sbx env exec -- python` and later Bash/console-script exec. Both editable import
  paths, unrelated-cwd resource generation and console entry points passed.
  Configurations stay under `/tmp/finn-implementation-sbx-config`, outside both
  explicitly mounted source trees. No custom lifecycle manager or automatic
  agent/credential/network setup was introduced. Evidence:
  `/tmp/finn-p3-sbx-final-prepare.log`, `/tmp/finn-p3-sbx-final-reuse.log`.
- XRT: application variant installed XRT 2.18.179 (2024.2,
  build `3ade2e671e5ab463400813fc2846c57edf82bb10`). An actual scoped settings capture
  and `xclbinutil --version` operation preserved the parent environment. This is an
  installed-runtime probe, not FPGA/licence validation.
  Evidence: `/tmp/finn-implementation-release-xrt-probe.log`.
- Native host: setup now uses isolated venvs and ordinary prepared PEP 517 backends;
  the XSI prerequisite/build decision is corrected. This host's Python 3.12 tests
  supplement the reference Python 3.10 image journeys; full Ubuntu 22.04 native
  provisioning and vendor extension building were not performed on this host.

### Open integration and native gates

The direct XSI session mechanism passed synthetic isolation, wide-word data
round-trip, concurrent output separation, lifecycle, failure/crash, cancellation,
wall timeout, stale-result and source/tool/ABI compatibility tests. It loads the
bridge/kernel/design only after exec with the selected environment and uses explicit
custom testbench files, not parent closures. No actual AMD XSI session was run.
Real output/cycle/trace equivalence, AXI/external-memory/MLO/characterization cases,
concurrent native reuse and workload/startup/transfer measurements remain open.

`src/finn/builder/` and `_legacy_build_env.py` remain unchanged from the handoff.
P6 must integrate the actual private build-engine implementation and retire/reconcile
the whole-build worker. The existing C++ performance harness and its legacy command/
loader seam remain integration-sensitive; broad path/config/allocation/logging
cleanup was not duplicated. Bare vendor shims and libudev preload remain until
licensed/native replacement gates pass. No AMD Vivado/HLS/licence access,
FPGA hardware or Apptainer/Singularity was available. The complete runtime/optional
proprietary package matrix and P8 release approval remain open.

Test-created sandboxes and Dev Containers were removed after validation. Prepared
local images and `/tmp` logs/environments remain available as explicit validation
artifacts; no user sandbox or pre-existing image was removed.

The delivered-file list is
`docs/container-runtime-implementation-files.txt` (removed; in history at `5dd9df9bc`).
The archive and historical inputs remain preserved. No commit, merge or push was
performed; new-file intent entries only make the resource renames reviewable in
`git diff`.

## Historical prototype validation (superseded architecture)

The record below describes the archived prototype. Inherited-package editable
overlays and the whole-build worker are not the approved final architecture.

Date: 2026-09-16. Implements `high-value-runtime-plan.md` (removed; in history at `5dd9df9bc`); the earlier redesign
and information inventories remain historical inputs. No merge or push performed.
The working-tree delivery inventory was `runtime-changed-files.txt` (removed).

### Delivered behavior

- Explicit wheel/sdist data for FINN-owned RTL, custom HLS/C++, XSI sources and
  adapter, Tcl and driver templates; notices retained and test/generated data
  excluded. `VERSION`, wheel provenance and pip installation metadata describe
  the selected installation. Standard package resources supply stable paths.
- FINN-owned resource consumers, generated HLS/Tcl and relevant shell resource
  references migrated away from `FINN_ROOT`. External HLS/board data stays pinned
  separately. Intermediate/checkpoint formats and allocation policy retained.
- Installed application images and explicit editable FINN/dependency workflows;
  removed `finn_paths.py`, `finn-live.pth`, the handwritten console shim, and
  live/frozen/auto import selection. No startup installation or repair replacement.
- Scoped settings capture and AMD execution; representative `CallHLS` and
  `CreateStitchedIP` migrations, route-preserving identity/capability checks,
  timeouts/cancellation, defensive environment copies and actionable failures.
- Internal compatibility at the existing directory build boundary, child cwd and
  pre-start loader environment, worker inheritance, documented in-process limits.
  The sbx hook keeps native integration only. Site Tcl copying is opt-in.

### Checks and results

Host validation used Python 3.12, an explicit editable FINN install and QONNX
revision `f5c9819bd00f01f41e70639b8461c8e4b39432f7`. The image uses the existing
Python 3.10 dependency pins and now supplies pip 24.3.1 for explicit editable setup.

- 151 focused regression tests passed: runtime toolchain/codegen, tool resolver,
  XSI source ordering, Docker configuration/CLI, CI transport and driver helpers.
- A separate non-vendor checkpoint test passed: resume from the original
  intermediate path succeeds, and missing intermediates fail explicitly.
- Five installation acceptance tests passed, including wheel-from-source versus
  wheel-from-sdist content equality, removing both build source trees before
  installed use, unrelated cwd, unset root/resource variables, FIFO RTL and driver
  generation, real console-script invocation, two independent editable FINN
  environments, and editable FINN plus the actual pinned QONNX source. Startup
  does not mutate the environment or create scratch. These are installed-resource
  tests, not relocation tests.
- Synthetic AMD tests cover awkward settings/argument paths, BASH_ENV interference,
  concurrent selections, route-preserving probes and site directory overrides,
  captured failures, timeout/cancellation of descendants, deliberate old/standalone/
  unified HLS frontend validation, generated HLS references and a complete fake
  stitched-IP Vivado operation. Public directory builds preserve parent cwd/env and
  report nonzero failures. No synthetic test establishes licensed synthesis success.
- Docker runtime and sbx image targets built. The runtime image generated RTL and
  Python driver outputs from `/tmp`, without a checkout mount, as arbitrary UID
  12345. Explicit editable preparation was tested separately. This exposed and
  fixed both Ubuntu's older pip and the inherited-wheel precedence problem: use
  the prepared pip/backend and strict editable mode in system-site-package venvs.
- Native sbx 0.42.1: loaded the sbx image, planned and created a disposable
  shell environment from the copied base example, generated FIFO/driver outputs,
  prepared an editable overlay, verified selected import/resource paths and an
  atomic RTL edit, then removed the sandbox. No additional workspace, credentials
  or site policy were added.
- Direct Git-checkout wheel and wheel rebuilt from its sdist matched all 325
  members excluding RECORD, including source revision/dirty provenance. Static
  Ruff, Python compilation, shell syntax and whitespace checks passed.

### Explicit editable preparation helper

`scripts/prepare-editables` was checked with three focused tests using real pip:
wheel replacement with live/atomic source edits and console scripts, manifest
paths containing spaces from an unrelated working directory, ignored pip install
redirection settings, dependency incompatibility, and a missing manifest.
All three passed on host Python 3.12. Formatting and lint checks passed.

A disposable Python 3.10 container with networking disabled additionally installed
regular FINN/QONNX wheels into a writable isolated environment, then replaced both
through the helper. Import paths and editable metadata selected the mounted
checkouts, `pip check` passed, and `build_dataflow --help` worked. This used the
uv spike's experimental dependency image as a fixture; it does not migrate the
production images to a writable baked environment or change their storage model.

### Coverage limits and release gates

No AMD installations, licence server, FPGA hardware or Apptainer/Singularity are
available here. Installation-backed AMD probes, licensed Vivado/HLS operations,
actual XSI simulations and SIF/HPC execution remain unverified. The optional XSI
fresh-worker experiment is deferred under the plan's stated fallback; no native
isolation/performance claims are made. Bare-tool shims, explicit native shell setup
and image loader workarounds remain until their real execution gates pass.

The generic HLS runner validates frontend command syntax, reported identity and
unified-HLS capability evidence separately. That does not certify FINN's generated
HLS directives for every older or newer tool release. Validate actual generated
projects against each supported installation before declaring release support.

No claim is made that generated projects, exported directories or checkpoints
survive installation replacement, upgrades, toolchain changes or relocation.
Checkpoint reuse still requires original absolute artifact/resource paths and
compatible generated/native code. Concurrent in-process legacy/XSI builds remain
unsupported as detailed in `legacy-build-env-ledger.md`; directory builds isolate
those settings in fresh interpreter trees.
