# Container/runtime status

Updated: 2026-09-26. Branch: `feature/external-resources`, an exploration based on
`refactor/container-runtime-implementation` at `09fb86df7` (after merging upstream
`dev` at `b507fec58`); plan and records in
[docs/external-resources-plan.md](docs/external-resources-plan.md).

Committed locally; nothing has been merged or pushed. The original prototype is on
`archive/container-runtime-prototype`; the dependency/application-image
implementation this replaced is preserved at the checkpoint `5dd9df9bc`.

## Implemented

- **Packaging:** resources in `finn.bundled`; installed FINN works without a checkout.
- **Dependencies:** `pyproject.toml` (runtime ranges, dependency groups, uv sources
  for unreleased commits) and `uv.lock`, reproducing the previous image's versions.
- **External resources:** finn-hlslib and the board files (five repositories) are
  declared with pinned commits and tree digests in `finn/bundled/resources.toml`,
  fetched on first use and cached by digest (`finn.resources`, `finn-resources`);
  projects and installed packages declare their own. No submodules, no uv
  workspace, no `finn[hw]` extra.
- **Environments:** one image with `/opt/venv` active and a never-fatal startup
  editable install of the mounted checkout (Docker, Dev Container, sbx); native
  `setup-local.sh` wraps `uv sync`; release image for SIF/HPC. The default `dev`
  image carries all resources; `release` only the redistributable ones.
- **Tools and simulation:** scoped vendor command execution; XSI session process;
  `finn_xsi` built on first use (keyed, concurrency-safe).
- **CI:** wheel build and clean-environment use, including fetching finn-hlslib;
  lock-keyed caches.

Design: [docs/environment.md](docs/environment.md). Instructions:
[docs/installation.md](docs/installation.md). Evidence:
[docs/runtime-validation.md](docs/runtime-validation.md).

## Remaining work

The task spec, with order, acceptance criteria and open questions, is
[docs/runtime-remaining-work.md](docs/runtime-remaining-work.md).

**P6: integrate the private build-engine branch** using its actual API: connect
explicit resource/tool/scratch inputs at its real boundaries, remove or reconcile
the whole-build subprocess in `build_dataflow_directory`, and retire internal
legacy-variable interpretation it replaces. Acceptance: CLI and Python builds,
checkpoint resume under documented constraints, and a build through the selected
tools and simulation boundary.

**P7: delete remaining compatibility mechanisms** once their gates pass:

| Mechanism | Gate |
| --- | --- |
| Whole-build Python worker | P6 integration |
| `docker/toolchain-shim`, `docker/finn-toolchain.sh` | All FINN vendor callers use explicit dispatch; bare-tool use documented |
| Global `LD_PRELOAD` (libudev) in the image | Scoped replacement passes real licence checkout |
| Broad `_legacy_build_env` activation | Explicit inputs and the integrated build engine |

**P8: validation needing AMD tools or other infrastructure**, none of which this
host has:

- HLS C++ simulation, synthesis and licence checkout, native and in the Ubuntu 24.04
  image (without ncurses 5); the XRT 24.04 package install.
- `finn_xsi` first-use build and RTL simulation against a real Vivado; XSI session
  equivalence, lifecycle and overhead.
- SIF export and read-only execution (Apptainer).
- Vivado finding boards across several board repository paths (one per board
  resource, at the same depth as the previous single directory), in a real Zynq
  project build.
- The Dev Container through VS Code (uid remapping); the GitHub, Jenkins and Read
  the Docs pipelines as configured.

**Before publishing to PyPI:** a QONNX release that includes the four commits past
1.0.0 that FINN uses, and a decision on redistributing board files (which would
make them `redistributable` and part of published images).

**Python and OS:** Python 3.12 on Ubuntu 24.04 (lock, image, development), with
3.11 also allowed and unit-tested in CI. Vivado/Vitis 2024.2 is the minimum. The
ncurses 5 compatibility libraries are no longer installed; confirm Vivado 2024.2
runs without them (P8).

**Review:** the branch history is one checkpoint commit, one commit per migration
step, then the merge of upstream `dev` (conflict resolutions recorded by git
rerere). Reorganize into reviewable commits (resource move as its own
pure-rename commit) before proposing it upstream.
