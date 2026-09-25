# Sol worker findings

The requested `gpt-5.6-sol` worker authored and executed `worker_probe.py`.
This report is compiled by the parent from the worker's final summary and its
recorded command/results files. The user permission-change interruption paused
the worker; it resumed to deliver the conclusions below.

## Tested scope

- uv binary: `/home/tkeller787/.local/bin/uv`, version **0.10.0**.
- Host interpreter: `/tmp/finn-runtime-venv/bin/python`, **3.12.3**.
- Three synthetic FINN/QONNX-style source pairs with a small self-contained
  PEP 517/660 backend and local external wheels.
- Offline commands, explicit project selection and separate absolute
  `UV_PROJECT_ENVIRONMENT` paths.
- Scratch/results: `/tmp/finn-uv-spike/worker/`.

These dependency-light fixtures test uv semantics. They are not actual FINN or
AMD simulation tests. The parent-owned runtime probes supply the real FINN/QONNX
evidence described in [README.md](README.md).

## Observed behavior

1. uv 0.10.0 supports `--no-install-local`, `--no-install-workspace`,
   `--no-emit-local`, `UV_PROJECT_ENVIRONMENT` and `uv run --no-sync`.
2. Three independent pairs with identical relative layouts produced byte-identical
   lockfiles and external dependency exports.
3. `uv sync --locked --no-install-local` installed the external closure while
   excluding both local editable distributions.
4. `--no-install-workspace` did **not** exclude ordinary local path dependencies;
   it is not interchangeable with `--no-install-local` for the wrapper approach.
5. Full sync installed the selected editable sources. Source-code edits became
   visible immediately and did not require re-locking or alter the external export.
6. Relocating the complete relative-layout fixture preserved its lock. Changing a
   declared local source path made the lock stale; re-locking changed that lock
   while preserving the external-only export.
7. `uv run --no-sync` preserved a prepared partial environment without installing
   omitted editables. Normal `uv run --locked` installed them. After project
   metadata changed, `--no-sync` still ran the old prepared environment while
   normal `--locked` execution rejected the stale lock.
8. Absolute `UV_PROJECT_ENVIRONMENT` settings kept environments outside the source
   project; no default project `.venv` was created in those cases.
9. Locking a wrapper around the real FINN/QONNX sources failed offline against the
   tiny synthetic wheelhouse, as expected. A lockfile is not a substitute for
   provisioning the real dependency/build artifacts.

The synthetic common lock SHA256 was
`6aabfdb1e565199dbe2b1d101dc488abc9801ddf2cc9eca14b218f25d907edcc`.
The common external export SHA256 was
`c14e952ab8a560829946f648366d8e9d73f1f8f4064305f5ae1a9cf912c55e42`.
Changing a local source declaration produced lock SHA256
`46c793cd792081ae8c9963e25e12e5041a5abd2005770fc84bd756c950e57475`
without changing the external export.

The successful synthetic relocation test does not establish portability of the
real offline lock: the parent's real probe found location-sensitive relative
wheelhouse registry paths. Those are separate cases and both are recorded.

## Worker recommendation

Use one small non-package wrapper project per FINN/QONNX pair, with declarative
relative editable sources and one explicitly selected environment per pair.
Prepare dependency artifacts with `--no-install-local`; use the prepared
interpreter directly or `uv run --no-sync` for ordinary execution. Keep lock
validation and synchronization explicit.

Preserve exact dependency pins, CPU Torch selection, supported Linux target
constraints, existing package build backends and offline build requirements.
Use the external-only export when identifying reusable dependency content, and
retain wheel checksums/provenance. Neither mounts nor sandbox lifecycle need to
move into a new FINN environment manager.
