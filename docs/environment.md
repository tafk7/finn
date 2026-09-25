# FINN's environment: design

How FINN's Python environment, build data, toolchain and containers fit together,
and why. For instructions, see [installation.md](installation.md). The design
history behind this (the dependency/application image split, the offline
wheelhouse and the explicit venv preparation it replaced) is in the repository
history at commit `5dd9df9bc`.

## Three things and a cache

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 3 PYTHON   pip install finn[hw]                          (users)          │
 │            uv sync -> active venv: FINN + workspace members editable,     │
 │            everything else at uv.lock                    (developers)    │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 2 VIVADO   the user's, never installed by FINN; selected by environment  │
 │            (FINN_XILINX_PATH/VERSION or settings64.sh) plus a licence     │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 1 SYSTEM   OS packages and XRT/SLASH: the host, or the image              │
 └──────────────────────────────────────────────────────────────────────────┘
   caches   finn_xsi (built on first use), board files (fetched on first use)
```

* **Python** is standard packaging. `pyproject.toml` declares FINN's runtime
  dependencies as ranges, dependency groups for development, and uv sources for
  unreleased commits; `uv.lock` is the exact environment for development, CI and
  the images. Build data is packaged too: finn-hlslib is the `finn-hlslib` package.
* **Vivado** is always the user's. `docker/config.py` locates it on the host (both
  AMD install layouts) and `docker/finn-toolchain.sh` applies it, natively
  (`scripts/activate.sh`) and in the image (entrypoint and tool shims).
* **System** packages are what pip cannot provide: the libraries Xilinx tools need
  (ncurses 6, the LSB loader, the libudev preload for FLEXlm) and XRT/SLASH.
* **Caches** are built or fetched by FINN when first needed, never installed:
  `finn_xsi` per Vivado installation and Python ABI, board files per pinned digest.

## Per modality

```text
                 SYSTEM                 VIVADO                       PYTHON
               ┌──────────────────────┬────────────────────────────┬──────────────────────────────┐
 User          │ host (hw only)       │ host (hw only)             │ pip install finn[hw]         │
 Native dev    │ host                 │ host; scripts/activate.sh  │ uv sync; scripts/activate.sh │
 Docker        │ image                │ mounted by docker/run      │ /opt/venv, always active;    │
 Dev Container │ image                │ (none, or a mount)         │ the entrypoint installs the  │
 sbx           │ image + sbx stage    │ sbx mount + licence policy │ checkout at container start  │
 Release/SIF   │ image                │ mounted                    │ FINN wheels, installed       │
               └──────────────────────┴────────────────────────────┴──────────────────────────────┘
```

The Python column is the same `uv sync` everywhere. The image runs the expensive
part ahead of time.

## One image

```text
 system ──► python ──────────────────────────► runtime ────────────► sbx
 apt,       uv; /opt/venv from uv.lock           XRT/SLASH            NOPASSWD sudo,
 ncurses6,  (no FINN, no workspace members);     (FINN_RUNTIMES)      BASH_ENV, npm,
 LSB,       active via ENV; board files;             │                proxy env_keep
 libudev    entrypoint                               └──► release: FINN + finn-hlslib wheels
```

* The tag hashes `docker/image-inputs.txt`: the Dockerfile, `pyproject.toml`,
  `uv.lock`, the board pins in `finn/util/external.py`, the container scripts and
  the runtime manifests. FINN's sources are not an input.
* `/opt/venv` is active through image `ENV`, so `docker exec`, `sbx exec`,
  non-interactive shells and editors see it, not only login shells. It is
  writable by any uid: containers run as the caller's uid, and each container's
  writable layer is its own.
* The release image installs wheels built from the checkout. It is for read-only
  use (SIF/HPC), where a startup install is impossible, and is what
  `docker/build --export-sif` exports.

## Container start

```text
 docker/run · Dev Container · sbx start
      │
      ▼
 tini ─► finn_entrypoint.sh
      ├─ no checkout at FINN_ROOT ──► note; image environment only ─────────┐
      ▼                                                                      │
 uv sync --frozen --inexact --project $FINN_ROOT                             │
   (offline first; online only if uv.lock needs packages the image lacks)   │
      ├─ ok ──► FINN + workspace members editable, + any lock difference ───┤
      └─ fails ──► warning with the fix; image environment kept ────────────┤
                                                                             ▼
                                               touch /tmp/finn-ready; exec the command
```

* Never fatal: a fatal entrypoint once killed sandboxes, which start PID 1 before
  any workspace is used.
* uv's cache lives in `$FINN_BUILD_DIR/.uv-cache`, which outlives the container, so
  starts after the first take about a second instead of several.
* This is a startup install, which the earlier design avoided. It is acceptable
  because it is deterministic: exactly the committed lock, plus a link to the
  checkout that was mounted. Nothing is discovered or resolved.
* Rejected alternatives: baking an editable FINN at a fixed path (`--fpga` mirrors
  the checkout's host path, so no single path works), and a `.pth` hook reading
  `FINN_ROOT` (hidden source discovery).

## Dependencies

* **Ordinary dependency:** `[project] dependencies` or a dependency group; a git
  source in `[tool.uv.sources]` while unreleased. Occasional co-development: a
  local, uncommitted path source.
* **Build data:** a data-only package, found through `importlib.resources`, with
  an environment-variable override (`finn.util.external`).
* **Developed in lockstep:** a workspace member under `packages/`, a git submodule
  whose commit is the pin, always editable. It lives inside the checkout, so every
  container sees it through the existing mount. Members are never baked into the
  image (`--no-install-workspace`).
* **Not redistributable:** fetched from upstream on first use, pinned and verified
  by digest, cached. The board files are this case until their licences are
  confirmed for redistribution in a wheel.

## Guards

* The image build fails on an inconsistent environment: `uv pip check`, an import
  smoke test and a pytest run (plugin import failures have no metadata conflict).
* The Package workflow builds the wheels and uses them from a clean environment
  with no checkout, because editable installs hide packaging mistakes. It also
  checks that the published metadata resolves from PyPI.
* A manually triggered job tests the lowest versions the ranges allow.

## Open items

* **PyPI:** FINN is developed against QONNX 1.0.0 plus four commits (the MaxPool
  `ceil_mode` fix); the published requirement is `qonnx>=1.0.0`, which PyPI has.
  Publishing needs a QONNX release that includes those commits, and `finn-hlslib`
  published alongside FINN.
* **Board files:** confirm whether XilinxBoardStore, Avnet and RealDigital files
  may be redistributed in a wheel; if so, a `finn-boards` package can replace the
  on-demand fetch.
* **Python range:** the lock, the image and development use Python 3.12 on
  Ubuntu 24.04. `requires-python` allows 3.11-3.12: 3.10 cannot resolve (QONNX caps
  onnx at 1.17 there, and onnxruntime 1.28 needs 3.11), numpy<2 has no wheels
  beyond 3.12, and a Package workflow job runs the unit tests on 3.11.
