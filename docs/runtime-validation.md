# Runtime implementation validation

Date: 2026-09-16. Implements `high-value-runtime-plan.md`; the earlier redesign
and information inventories remain historical inputs. No merge or push performed.
The working-tree delivery inventory is in `runtime-changed-files.txt`.

## Delivered behavior

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

## Checks and results

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

## Coverage limits and release gates

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
