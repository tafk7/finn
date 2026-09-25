# Container/runtime status

Updated: 2026-09-23. Branch: `refactor/container-runtime-implementation`.

The independent implementation is largely complete and tested locally, but is
not integrated or release-ready. Changes remain uncommitted; nothing has been
merged or pushed. The original prototype is preserved on
`archive/container-runtime-prototype`.

## Implemented

- **P0–P1:** Prototype audit and conventional wheel/resource packaging. Installed
  resource use works without a source checkout.
- **P2:** Separate application/dependency images, distinct artifact identities,
  and a checksummed offline development wheelhouse.
- **P3:** Explicit writable development environments with editable FINN and
  selected dependencies; Docker persistence, Dev Container setup and native sbx
  reuse validated. Native setup updated; full native provisioning remains open.
- **P4:** Scoped vendor command execution and concrete caller migrations.
- **P5:** XSI simulation-session process mechanism with synthetic isolation,
  lifecycle, failure and data-transfer coverage; actual AMD simulation unvalidated.

Current development flow: prepared dependency image/wheels → explicit writable
venv preparation → editable source installation → ordinary Python execution.
Each independent FINN/QONNX pair gets its own environment. Startup does not install
packages, and mounting a checkout alone does not select it for imports.

## Development setup decision still open

- **pip:** Current implementation uses ordinary pip. The new
  `scripts/prepare-editables` helper reads `editable-requirements.txt`, supplies
  the required options and runs `pip check`. It does not provision environments
  or mounts. The helper is used in Dev Container setup and documented examples.
- **uv:** A successful spike exercised three real FINN/QONNX pairs, native/Docker/
  sbx workflows, and an image-prepared venv seeded into a Docker volume. No main
  implementation migration to uv has occurred.
- Installing wheels first and replacing selected packages with editables was
  validated. A fully baked writable development environment has not replaced the
  current image/environment setup. The pip/helper/uv choice remains reversible.

## Remaining work

- **P6:** Integrate the separate private build-engine branch using its actual API.
  Builder code and `_legacy_build_env.py` remain unchanged from the handoff.
- **P7:** Reconcile/remove the prototype whole-build worker and remaining
  compatibility hooks after integration and required execution checks.
- **P8:** Real Vivado/HLS/XSI, FPGA and SIF/HPC validation and release review.
- Organize the working-tree implementation into reviewable commits; no merge or
  push is authorized.

## Evidence

Recorded validation includes 182 distinct runtime test cases, actual Docker,
Dev Container and native sbx journeys, the uv spike, and three additional helper
tests plus actual FINN/QONNX wheel-to-editable replacement with networking disabled.
These are prior results, not a fresh full-suite run on this status date.

See [implementation plan](docs/container-runtime-implementation-plan.md),
[validation record](docs/runtime-validation.md),
[current setup instructions](docs/installation.md), and
[uv spike](docs/experiments/uv-spike/README.md).
