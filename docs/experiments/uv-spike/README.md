# uv development-environment spike

Date: 2026-09-21. Experimental evidence only; the current FINN runtime implementation
and private build-engine boundary are unchanged. No repository commit, merge or push.

The requested Sol worker owns `worker_probe.py` and `worker-report.md`. The parent
agent owns the real-package/runtime probes in this directory. Together these test
whether ordinary uv project operations can replace package-specific preparation
flags and custom editable-selection logic.

## Result

**uv 0.10.0 can implement the proposed package policy.** A virtual development
project declares editable paths using `tool.uv.sources`; `uv sync
--no-install-local` prepares the external dependency closure, and a subsequent
explicit full sync registers the selected local projects. No custom FINN/QONNX
CLI switches are necessary.

This worked with actual FINN and the pinned QONNX source, including QONNX linked
worktrees, with all existing external wheel versions preserved. The generic
semantics also passed the Sol worker's independent, dependency-light host probe.

| Experiment | Result |
| --- | --- |
| Three real source pairs in Docker | Each partial venv excludes FINN/QONNX; full sync selects its own FINN/QONNX sources and metadata. An atomic source edit affects only the selected pair. All three pass `pip check`. |
| Native host | An explicitly provisioned, task-local CPython 3.10.19 environment passes the real package lock/partial/full-sync/import/resource/console checks. Package operations are offline after provisioning Python and copying the prepared wheels. |
| Native sbx | The same real project/lock works with a sandbox-private environment; partial exclusion, editable attachment, later native exec, resources, entry point and `pip check` pass. |
| Source-excluded image | Built from the cached Ubuntu system stage, with no FINN/QONNX source copied. Its isolated `/opt/finn/venv` and 219 external wheels omit both local distributions. |
| Docker persistent volume | Docker seeds a fresh named volume at `/opt/finn/venv` from the image. Explicit sync builds/installs only FINN and QONNX; a second disposable container reuses that environment without syncing. |
| External dependency projection | `uv export --frozen --no-emit-local --no-header --no-annotate` produces identical 219-line requirements exports for all three real pairs. |

Actual versions: uv 0.10.0; Docker/sbx Python 3.10.12; task-local native Python
3.10.19. The worker's synthetic native probe uses Python 3.12.3. The prepared
wheelhouse was taken from `xilinx/finn:deps-1d3775474e8454c3`, image ID
`sha256:91655e6103251f2cb9753c313c36118c1a0703da3a1a69dc6e9e74c9ae38ccdf`.
Its 220-wheel reference set includes QONNX; the source-excluded experiment retains
219 external wheels and obtains QONNX from the selected checkout.

## What this changes

Use one development project per independent FINN/QONNX pair. A virtual wrapper
project can use the existing FINN `setup.py`/`setup.cfg` and QONNX build backend;
this spike did not migrate either package's build backend.

```toml
# Source-binding portion of the wrapper pyproject.toml; not its complete lock inputs.
[tool.uv]
package = false
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]

[tool.uv.sources]
finn = { path = "./finn", editable = true }
qonnx = { path = "./qonnx", editable = true }
```

The wrapper also declares FINN/QONNX dependencies and the supported Python range.
For this experiment, the ordinary package versions were generated directly from
the existing wheelhouse manifest to avoid changing the dependency contract.

```text
One dependency policy / external package set
                 |
        uv partial installation
                 |
     immutable image with isolated venv
     (FINN and QONNX absent)
                 |
         runtime writable storage
          /          |          \
       env A       env B       env C
         |           |           |
       FINN A      FINN B      FINN C
       QONNX A     QONNX B     QONNX C
```

- **Native:** uv constructs an isolated host venv from the wheel set for that
  host/Python. It then registers the selected editable sources.
- **Docker:** an image can contain an already-populated isolated venv at a fixed
  path. A fresh named volume mounted at that same path is initialized from the
  image. Each pair gets its own volume; source attachment installs two editables
  rather than reinstalling all external packages. An empty bind mount does not
  provide Docker's volume initialization behavior and needs explicit provisioning.
- **sbx:** uv prepares a sandbox-private environment using native source mounts.
  The tested route creates the venv from wheels. A writable copy of an image's
  prepopulated venv could reduce that work, but that particular sbx image/storage
  combination was not tested here. Native sbx continues to own lifecycle and policy.

No system-site-packages overlay is needed for the tested image/volume route.
The venv is constructed and consumed at the same `/opt/finn/venv` path. It is not
copied from a host-native interpreter or relocated to a different guest prefix.
Independent volumes/environments preserve per-pair source metadata and selection.

## Constraints discovered empirically

1. **Universal resolution needs an explicit supported target.** Initially uv
   requested Windows-only `colorama` while resolving the Linux-only wheelhouse.
   Specifying Python 3.10 alone was insufficient. The supported Linux/x86-64
   environment marker made the complete offline lock resolve.
2. **Local wheelhouse paths affect lock portability.** uv recorded the wheelhouse
   path relative to the project. Moving the lock into a shallower image path failed
   normalization even with `--frozen`. Preserving the project's path made the build
   work. A production design should use a deliberate artifact layout or index and
   test its relocatability; copying arbitrary local locks into an image is unsafe.
3. **Use the external projection for dependency-image reuse.** The real exports
   are identical across all three source pairs (SHA256
   `39748da47cd3cbc001c91cce42c9eb6ff5f315360349f04d1ba5b2c0968ba32a`).
   The worker also verified that changing a local source binding changes the full
   lock while leaving its external projection unchanged. Base identity should
   reflect the external closure, build/runtime inputs and artifact hashes.
4. **Keep artifact checksums/provenance.** In this local-flat-wheelhouse experiment,
   the lock records wheel paths without wheel content hashes. The existing
   checksummed wheel manifest still supplies information that this lock does not.
5. **`--no-install-workspace` is insufficient for ordinary path dependencies.**
   The worker verified that those packages are installed under that option;
   `--no-install-local` excludes them as intended.
6. **Normal `uv run` can mutate the environment.** The worker demonstrated that
   it installs intentionally omitted local packages. Use an explicitly prepared
   interpreter or `uv run --no-sync` for ordinary work. `--locked` prevents a lock
   update, not installation; `--no-sync` deliberately does not validate freshness.
7. **Offline build requirements must be present.** A lock does not supply wheels,
   backends, headers or Python. All real package operations here use the prepared
   wheel set; native Python download is a separate explicit provisioning step.
8. **Worktree Git metadata must be visible.** The real QONNX worktrees and their
   shared repository metadata are mounted together under stable absolute paths.

Measured observations, not a performance benchmark: the first three isolated
Docker partial installs took 17.5/11.4/14.5 seconds and editable attachment took
1.67/1.65/1.89 seconds, with shared cache state. The seeded-volume preparation took
20.4 seconds including Docker's initial filesystem copy and container startup;
uv reported building two packages in 466 ms and installing those two in 2 ms.

## Reproducing the evidence

The runtime script only creates task-owned scratch data, fixture worktrees,
experimental images and temporary containers/volumes/sandboxes. It needs the
existing prepared image and `/tmp/finn-implementation-qonnx` source from the earlier
validation, a Python 3.11+ host harness, Docker access, and native sbx access for that mode. Review these inputs
before running it on another machine.

```bash
python3 docs/experiments/uv-spike/worker_probe.py
python3 docs/experiments/uv-spike/runtime_probe.py inventory
python3 docs/experiments/uv-spike/runtime_probe.py docker
python3 docs/experiments/uv-spike/runtime_probe.py image
python3 docs/experiments/uv-spike/runtime_probe.py native
python3 docs/experiments/uv-spike/runtime_probe.py sbx
python3 docs/experiments/uv-spike/runtime_probe.py export
```

`native` explicitly downloads a Python interpreter into scratch with `--no-bin`;
it does not change the host interpreter or install a global Python command.
`image` builds a Python-only experiment, not a replacement vendor runtime image.
The image mode must be run after `docker`, which creates the fixture locks.

Machine-readable results and command logs live under `/tmp/finn-uv-spike/runtime/`:
`docker-results.json`, `native-results.json`, `sbx-results.json`,
`image-results.json`, `export-results.json`. The two initial failures are preserved
as `initial-unconstrained-lock.log` and `initial-relocated-lock-image.log`.
Worker results are under `/tmp/finn-uv-spike/worker/` and summarized in
[worker-report.md](worker-report.md).
The compact result records are also preserved in [results.json](results.json).
Temporary containers, named volumes and the sbx sandbox were removed. Bulk
throwaway Docker venvs and installer caches were cleaned after recording results;
source fixtures, locks, the native venv and logs remain under the scratch root.
The two experimental Docker images remain local for inspection.

## Recommendation and limits

Adopt uv project/source declarations and explicit sync as the candidate replacement
for custom editable-package policy. Retain standard FINN wheel packaging, runtime
ownership of mounts/storage, exact artifact provenance and scoped vendor execution.
Avoid adding a second independent source/dependency manifest beside uv's project
model. Choose and validate one supported lock/index/wheelhouse layout before migration.

This spike supports the architecture, not a production migration claim. No current
launcher, Dockerfile, dependency pin or build-engine implementation was changed.
The real tests cover three environments sequentially, not simultaneous native FPGA
builds. No AMD simulation, licence, FPGA, SIF/HPC or complete Dev Container migration
was tested. Native package success with a task-local Python is not certification of
the complete native OS/vendor-tool contract.
