# FINN's environment: design

How FINN's Python environment, build data, toolchain and containers fit together,
and why. For instructions, see [installation.md](installation.md). The design
history behind this (the dependency/application image split, the offline
wheelhouse and the explicit venv preparation it replaced) is in the repository
history at commit `5dd9df9bc`.

## Three things and a cache

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 3 PYTHON   pip install finn                              (users)          │
 │            uv sync -> active venv: FINN editable,                         │
 │            everything else at uv.lock                    (developers)    │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 2 VIVADO   the user's, never installed by FINN; selected per machine     │
 │            (~/.config/finn/xilinx.env; a variable overrides it)           │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 1 SYSTEM   OS packages and XRT/SLASH: the host, or the image              │
 └──────────────────────────────────────────────────────────────────────────┘
   caches   finn_xsi (built on first use), external resources such as
            finn-hlslib and board files (fetched on first use)
```

* **Python** is standard packaging. `pyproject.toml` declares FINN's runtime
  dependencies as ranges, dependency groups for development, and uv sources for
  unreleased commits; `uv.lock` is the exact environment for development, CI and
  the images. Build data FINN does not contain (finn-hlslib, board files) is not
  Python: it is declared as [external resources](#external-resources).
* **Vivado** is always the user's, described once per machine in
  `~/.config/finn/xilinx.env`, read by one reader, `finn.util.machine_file`.
  `docker/xilinx_install.py` locates the installation (both AMD install layouts)
  for `docker/config.py` on the host and for the sbx workload's startup hook,
  FINN's tool launches take the licence from it, and `docker/finn-toolchain.sh`
  applies it, natively
  (`scripts/activate.sh`) and in the image (entrypoint and tool shims). A
  variable overrides the file for one shell, container or sandbox, so installed
  versions run side by side from the same image.
* **System** packages are what pip cannot provide: the libraries Xilinx tools need
  (ncurses 6, the LSB loader, the libudev preload for FLEXlm) and XRT/SLASH.
* **Caches** are built or fetched by FINN when first needed, never installed:
  `finn_xsi` per Vivado installation and Python ABI, external resources per
  pinned digest.

## Per modality

```text
                 SYSTEM                 VIVADO                       PYTHON
               ┌──────────────────────┬────────────────────────────┬──────────────────────────────┐
 User          │ host (hw only)       │ host (hw only)             │ pip install finn             │
 Native dev    │ host                 │ host; scripts/activate.sh  │ uv sync; scripts/activate.sh │
 Docker        │ image                │ mounted by docker/run      │ /opt/venv, always active;    │
 Dev Container │ image                │ mounted, as by docker/run  │ the entrypoint installs the  │
 sbx           │ workload kit (sbx)   │ sbx mount + xilinx kit     │ checkout at sandbox start    │
 Release/SIF   │ image                │ mounted                    │ FINN wheel, installed        │
               └──────────────────────┴────────────────────────────┴──────────────────────────────┘
```

The Python column is the same `uv sync` everywhere. The image runs the expensive
part ahead of time.

## One image

```text
 system ──► python ─────────────────────► runtime ──────────► dev ──────────► sbx
 apt,       uv; /opt/venv from uv.lock      XRT/SLASH          board files     NOPASSWD sudo,
 ncurses6,  (no FINN); active via ENV;      (FINN_RUNTIMES)    (not redistri-  BASH_ENV, npm,
 LSB,       finn-hlslib (redistributable        │              butable; local  proxy env_keep,
 libudev    resources); entrypoint              │              images only)    shell launch; no agent
                                                └──► release: FINN wheel; board files on first use
```

* The tag hashes `docker/image-inputs.txt`: the Dockerfile, `pyproject.toml`,
  `uv.lock`, the pins of the resources the image bakes in (from
  `finn/resources.toml`; FinnLib's is never baked, so moving it changes
  nothing) and `finn.resources`, which fetches them, the container scripts and the
  runtime manifests. FINN's other sources are not an input, nor is the tool
  configuration (`.ruff.toml`, `.mypy.ini`, `.pytest.ini`), kept out of
  `pyproject.toml` so that editing it leaves the image as it is.
* `dev` is the default target and the base of `sbx`. Only images built locally
  contain third-party board files; `release` builds on `runtime`, so it carries
  only resources declared `redistributable`.
* `/opt/venv` is active through image `ENV`, so `docker exec`, `sbx exec`,
  non-interactive shells and editors see it, not only login shells. It is
  writable by any uid: containers run as the caller's uid, and each container's
  writable layer is its own.
* The release image installs a wheel built from the checkout. It is for read-only
  use (SIF/HPC), where a startup install is impossible, and is what
  `docker/build --export-sif` exports.

## Container start

```text
 docker/run · Dev Container · sbx start
      │
      ▼
 tini ─► finn_entrypoint.sh
      ├─ sbx clone mode: FINN_ROOT still being cloned ──► the steps below run
      │  in the background once git has written the checkout; exec at once
      ├─ no checkout at FINN_ROOT ──► note; image environment only ─────────┐
      ▼                                                                      │
 uv sync --frozen --inexact --project $FINN_ROOT                             │
   (offline first; online only if uv.lock needs packages the image lacks)   │
      ├─ ok ──► FINN editable, + any lock difference ───────────────────────┤
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
* **Build data** (HLS or RTL libraries, board files): an external resource,
  below. Co-development: `FINN_RESOURCES_<NAME>` pointing at a checkout.

## External resources

```text
 declarations                         caches (first complete copy wins)
 1 finn/resources.toml          1 FINN_RESOURCES_DIR          writable
 2 packages: entry points  ─ merge ─►   (default ~/.finn/resources)
   "finn.resources" (add only)        2 /opt/finn/resources         read-only, image
 3 project: pyproject.toml                 │
   [tool.finn.resources], or               ▼ missing: fetch (locked) ─► verify digest ─► rename
   FINN_RESOURCES_FILES (may redefine)       git commit (sparse) · archive + sha256
                                             · GitHub archive without git · mirrors
 override: FINN_RESOURCES_<NAME>=/dir (unverified)
```

* **One mechanism** for FINN's resources, packages' and users': a named
  directory tree with a pinned source (`git`+`commit`, `url`+`sha256`, or data in
  an installed `package`) and a tree digest. Consumers ask by kind
  (`paths("vivado-boards")`), so adding a board repository or an RTL library is a
  declaration, not a code change.
* **Why not packages:** finn-hlslib used to be a workspace package from a git
  submodule. That needed the submodule in every checkout and CI job, a uv
  workspace, a `finn[hw]` extra, a second wheel and a PyPI release for
  `pip install finn` users, and it could not cover board files, which may not be
  redistributed. Declarations ship in FINN's wheel; resources are fetched on
  first use.
* **Reproducible and safe:** exact pins; the digest is checked before a tree is
  published, under a file lock, with an atomic rename. An entry is named by its
  digest, so a new pin can never reuse a stale tree.
* **Standard library only**, so the image build runs it from the mounted sources
  before FINN is installed, and a wheel-only install needs nothing else.
* **Redistribution:** each declaration says whether FINN may bake it into published
  images. The `python` stage fetches the redistributable ones, `dev` the rest.

## Running tools on LSF

FINN runs every Xilinx tool through a selected toolchain, whose command directory
(`finn.util.toolchain.Selection.command_dir`) is an interception hook: the machine
running FINN still drives the build and pytest, but each `vivado` / `v++` /
`vitis_hls` / `vitis-run` / `xelab` / `g++` invocation is a deployment-specific shim
in that directory, which may delegate the heavy subprocess to a compute farm (IBM
LSF's `bsub`, or another HPC model). FINN itself knows nothing of the farm.

The site states the shim directory once, as the machine setting
`FINN_TOOL_DIR_OVERRIDE`: the machine's toolchain
(`finn.util.toolchain.machine_selection`) takes it as its command directory, and
every transformation called without a toolchain, as the tests call them, and every
build whose configuration names none, runs by that toolchain. A build configuration
that states its own `toolchain` is laid over the machine's: a `command_dir` it states
(`{"command_dir": ...}`) wins over the setting, and one it does not state is the
setting's. A site wrapper owns remote activation, path visibility and the
cancellation of its remote jobs (`tests/util/test_tool_route.py` checks the route
with fake tools).

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
  Publishing needs a QONNX release that includes those commits.
* **Board files:** confirm whether XilinxBoardStore, Avnet and RealDigital files
  may be redistributed; if so, marking them `redistributable = true` puts them in
  published images too.
* **Several board paths:** each board resource is now a separate Vivado board
  repository path (at the same depth as before, where all shared one directory).
  This still needs a real Vivado run; the fallback is one directory of links to
  the resource roots.
* **Python range:** the lock, the image and development use Python 3.12 on
  Ubuntu 24.04. `requires-python` allows 3.11-3.12: 3.10 cannot resolve (QONNX caps
  onnx at 1.17 there, and onnxruntime 1.28 needs 3.11), numpy<2 has no wheels
  beyond 3.12, and a Package workflow job runs the unit tests on 3.11.
