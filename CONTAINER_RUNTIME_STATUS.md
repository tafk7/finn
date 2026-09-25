# Container/runtime status

Updated: 2026-09-25 (after merging upstream `dev` at `b507fec58`). Branch: `refactor/container-runtime-implementation`.

Committed locally; nothing has been merged or pushed. The original prototype is on
`archive/container-runtime-prototype`; the dependency/application-image
implementation this replaced is preserved at the checkpoint `5dd9df9bc`.

## Implemented

- **Packaging:** resources in `finn._data`; installed FINN works without a checkout.
- **Dependencies:** `pyproject.toml` (runtime ranges, dependency groups, uv sources
  for unreleased commits) and `uv.lock`, reproducing the previous image's versions.
  finn-hlslib is a workspace member (`packages/finn-hlslib`, upstream as a
  submodule); board files are fetched on first use from pinned commits.
- **Environments:** one image with `/opt/venv` active and a never-fatal startup
  editable install of the mounted checkout (Docker, Dev Container, sbx); native
  `setup-local.sh` wraps `uv sync`; release image for SIF/HPC.
- **Tools and simulation:** scoped vendor command execution; XSI session process;
  `finn_xsi` built on first use (keyed, concurrency-safe).
- **CI:** wheel build and clean-environment use; lock-keyed caches; submodules.

Design: [docs/environment.md](docs/environment.md). Instructions:
[docs/installation.md](docs/installation.md). Evidence:
[docs/runtime-validation.md](docs/runtime-validation.md).

## Remaining work

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
- The Dev Container through VS Code (uid remapping); the GitHub, Jenkins and Read
  the Docs pipelines as configured.

**Before publishing to PyPI:** a QONNX release that includes the four commits past
1.0.0 that FINN uses, finn-hlslib published alongside FINN, and a decision on
redistributing board files.

**Python and OS:** Python 3.12 on Ubuntu 24.04 (lock, image, development), with
3.11 also allowed and unit-tested in CI. Vivado/Vitis 2024.2 is the minimum. The
ncurses 5 compatibility libraries are no longer installed; confirm Vivado 2024.2
runs without them (P8).

**Review:** the branch history is one checkpoint commit, one commit per migration
step, then the merge of upstream `dev` (conflict resolutions recorded by git
rerere). Reorganize into reviewable commits (resource move as its own
pure-rename commit) before proposing it upstream.
