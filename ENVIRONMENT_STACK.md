# FINN environment stack, simplified

Rewritten 2026-09-24. Replaces the earlier 11-layer version of this file and
`ENVIRONMENT_STACK_DIAGRAMS.md`.

## The idea

The 11 layers weren't really 11 things. Most of them were one of three things split up by
who happened to install them. Merge them, and let standard tools (uv, pip, apt) do the
work FINN was doing by hand:

```text
  BEFORE (11 layers)                        AFTER (3 things + a cache)

  L10 session config  ──────────────┐
  L1  Vivado + licence ─────────────┴──►  ┌──────────────────────────────────────────────┐
                                          │ 2  VIVADO   yours, always. FINN never        │
                                          │             installs it. Selected by env     │
                                          │             vars (FINN_XILINX_PATH, licence) │
                                          └──────────────────────────────────────────────┘
  L8  sister overlays ──────────────┐
  L7  FINN ─────────────────────────┤
  L6  third-party deps ─────────────┤
  L5  venv ─────────────────────────┼──►  ┌──────────────────────────────────────────────┐
  L4  Python ───────────────────────┤     │ 3  PYTHON   pip install finn   (users)       │
  L3  hlslib + board files ─────────┘     │             uv sync + active venv (devs)     │
                                          │             one lock, hlslib/boards inside   │
                                          └──────────────────────────────────────────────┘
  L2  XRT / SLASH ──────────────────┐
  L0  OS packages ──────────────────┴──►  ┌──────────────────────────────────────────────┐
                                          │ 1  SYSTEM   OS packages + XRT: your host,    │
                                          │             or the image                     │
                                          └──────────────────────────────────────────────┘
  L9  XSI extension ────────────────────►   (cache) built by FINN on first use, like the
                                            cppsim binaries it already compiles
```

Three merges do the work:

1. **Build data becomes Python packages.** finn-hlslib is **1.2 MB**. The assembled board
   files are **20 MB** (measured in the current image). The image also carries about
   **500 MB** of full board-repo clones they're copied out of. Turn both into small
   data-only Python packages, so they're ordinary Python dependencies. hlslib becomes a
   workspace member so it's always editable; the boards become a companion wheel. See
   "Adding a dependency" below. Then `deps.env` and `fetch-repos.sh` disappear, and "fetch
   the data" stops being a step.
2. **Python, venv, deps, FINN and overlays become one thing, managed by uv.** `uv sync`
   already means "make the environment match the lock and install this project editable".
   That's the whole dev-env, editable-step and prepare-editables machinery, as one standard
   command. The venv is then simply active: you type `python` or `pytest`, not a wrapper.
3. **The XSI extension becomes a cache, not an install step.** FINN already compiles C++
   for cppsim on demand. The XSI bridge is the same kind of artifact: build it on first
   rtlsim into a cache keyed by toolchain version and Python ABI.
   `python -m finn.xsi.setup` stays, but only for pre-building.

Session config (L10) folds into Vivado (2), because it only ever existed to say *which*
Vivado and *which* licence.

## Per modality

```text
                 1 SYSTEM               2 VIVADO                     3 PYTHON
               ┌──────────────────────┬────────────────────────────┬──────────────────────────────┐
 Native user   │ your OS (hw only)    │ yours (hw only)            │ pip install finn             │
               ├──────────────────────┼────────────────────────────┼──────────────────────────────┤
 Native dev    │ your OS              │ yours                      │ uv sync, then activate .venv │
               ├──────────────────────┼────────────────────────────┼──────────────────────────────┤
 Docker        │ the image            │ mounted by docker/run      │ /opt/venv, always active;    │
 Dev Container │ the image            │ (none, or a mount)         │ uv sync runs at container    │
 sbx           │ the image + sbx stage│ sbx mount + licence policy │ start (entrypoint)           │
               └──────────────────────┴────────────────────────────┴──────────────────────────────┘
```

The Python column is **the same `uv sync` everywhere**. Natively you run it; in a container
the image and the entrypoint run it for you:

```text
  image build:       uv sync --frozen --no-install-workspace third-party packages -> /opt/venv
                     ENV VIRTUAL_ENV=/opt/venv                the venv is active for every
                         PATH=/opt/venv/bin:$PATH             process, not just login shells
                         UV_PROJECT_ENVIRONMENT=/opt/venv

  container start:   uv sync --frozen --inexact              adds editable FINN (and any
                         --project "$FINN_ROOT"               workspace members) from the
                                                             mounted checkout (~1 s, offline)
                     (entrypoint)                            plus any lock difference

  then:              python, pytest, build_dataflow, ...     no wrapper, no activation step
```

**Why `ENV` and not sourcing `activate` in `.bashrc`.** Activation only sets
`VIRTUAL_ENV` and `PATH` (plus a prompt). Setting them as image `ENV` makes the venv active
for `docker exec`, `sbx exec`, non-interactive shells, CI steps and editors. A `.bashrc`
reaches none of those. It's the same reason the toolchain shims exist.

**Why the entrypoint.** It's the one hook that `docker/run`, the Dev Container (Compose)
and sbx (sandbox start) all go through, so the editable link is made in one place instead
of three. `docker exec`/`sbx exec` skip the entrypoint, but they join a container where it
has already run. This does bring back "sync at startup", which the branch removed. It's
reasonable now because it's deterministic: it installs exactly what the committed lock
says, plus a link to the checkout you mounted. Nothing is discovered or resolved.

The entrypoint step has to be **non-fatal**. sbx starts PID 1 from `/`, possibly before
the workspace is there, and a fatal entrypoint check killed sandboxes before. So:

- no `pyproject.toml` at `$FINN_ROOT`: skip, with a one-line note
- sync fails (for example the lock needs a package the image lacks and there's no
  network): warn with the fix (`docker/build`, or `uv sync` with network), and start
  anyway on the image's environment
- `--inexact`: packages installed ad hoc in a long-lived sandbox survive a restart

```text
 docker/run   ·   Dev Container (Compose)   ·   sbx sandbox start
                          │
                          ▼
            tini ──► finn_entrypoint.sh                       venv is already active (image ENV)
                          │
                          ├── no pyproject.toml at $FINN_ROOT ──► note: "image environment only" ────┐
                          │                                                                          │
                          ▼                                                                          │
            uv sync --frozen --inexact --project "$FINN_ROOT"                                        │
                          │                                                                          │
                          ├── ok ──► editable FINN, + any uv.lock difference,                        │
                          │          + local [tool.uv.sources] (e.g. ../qonnx) ──────────────────────┤
                          │                                                                          │
                          └── fails ──► warning with the fix (docker/build, or uv sync               │
                                        with network); image environment kept ───────────────────────┤
                                                                                                     ▼
                                                                exec the command: bash, pytest, sleep infinity, ...

   Never fatal: a fatal entrypoint has killed sandboxes before.
   docker exec / sbx exec skip the entrypoint, but join a container where it already ran.
```

If `uv.lock` has changed since the image was built, the startup sync installs just the
difference. So **the image only needs rebuilding when the system layer changes**, not when
Python dependencies do. A stale image means a slightly slower start, not a wrong
environment. Offline sites rebuild the image after a lock change.

**Native** is the standard uv workflow: `uv sync`, then `source .venv/bin/activate` (or a
one-line direnv `.envrc`, so it activates on `cd`). Re-run `uv sync` after pulling a lock
change.

```text
 FIRST TIME
 ┌────────────────┐   ┌──────────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
 │ git clone finn │──►│ uv sync              │──►│ activate           │──►│ hardware only        │
 │                │   │  .venv: locked deps  │   │  source .venv/bin/ │   │  FINN_XILINX_PATH    │
 │                │   │  + editable FINN     │   │    activate        │   │  + licence env vars  │
 │                │   │                      │   │  or direnv .envrc  │   │                      │
 └────────────────┘   └──────────────────────┘   └────────────────────┘   └──────────────────────┘

 EVERY DAY
 ┌────────────────────────────────────────────────┐
 │ edit FINN code       nothing to do (editable)  │
 │ pulled a uv.lock     uv sync                   │
 │   change                                       │
 │ new Vivado version   nothing: XSI rebuilds     │
 │                      itself on first use       │
 └────────────────────────────────────────────────┘
```

**Rejected alternatives,** so they don't come back:
- *Baking editable FINN at a fixed path at image build* (so there's no startup step):
  `--fpga` mirrors the checkout's host path, so no single baked path works.
- *A `.pth` import hook reading `$FINN_ROOT`*: that's exactly the hidden source discovery
  the branch deliberately removed.

## One image

```text
 ┌────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────┐   ┌────────────────┐
 │ system             │──►│ python                       │──►│ runtime          │──►│ sbx            │
 │  apt packages      │   │  uv, Python 3.10             │   │  XRT / SLASH     │   │  NOPASSWD sudo │
 │  ncurses 5,        │   │  uv sync --frozen            │   │  (FINN_RUNTIMES) │   │  BASH_ENV      │
 │  LSB loader,       │   │    --no-install-workspace    │   │                  │   │                │
 │  libudev preload   │   │    -> /opt/venv              │   │                  │   │                │
 │                    │   │  ENV VIRTUAL_ENV, PATH,      │   │                  │   │                │
 │                    │   │    UV_PROJECT_ENVIRONMENT    │   │                  │   │                │
 │                    │   │  chmod a+rwX /opt/venv       │   │                  │   │                │
 │                    │   │  ENTRYPOINT: startup sync    │   │                  │   │                │
 └────────────────────┘   └──────────────────────────────┘   └─────────┬────────┘   └────────────────┘
                                                                       │            xilinx/finn:sbx-<id>[.xrt]
                                                                       │  xilinx/finn:<id>[.xrt]   <- the one dev image
                                                                       │
                                                                       ▼  optional, release tags only
                                                                     ┌────────────────────────────┐
                                                                     │ release                    │
                                                                     │  uv pip install finn==X    │──► docker/build --export-sif
                                                                     │  (no startup sync needed)  │
                                                                     └────────────────────────────┘

 <id> = hash(Dockerfile, runtime manifests)   optionally + uv.lock, to keep /opt/venv fresh
        FINN source is not an input: editing FINN never produces a new image.
```

No dependency/application split, no wheelhouse, no board-repo clones, and FINN source
isn't baked in. A release/SIF image is one optional stage (`uv pip install finn==X`), and
only worth maintaining if you actually publish images.

## Docker at run time

```text
 HOST                                                     CONTAINER  (docker/run, --rm, --user $UID:$GID)
 ┌────────────────────────────────┐                      ┌──────────────────────────────────────────────────┐
 │ ~/src/finn        (checkout)   │══ rw, same path ════►│ ~/src/finn     <- editable FINN (entrypoint)     │
 │ ~/src/qonnx       (optional)   │══ rw, --volume ═════►│ ~/src/qonnx    <- via [tool.uv.sources]          │
 │ /tools/Xilinx                  │══ ro ═══════════════►│ /tools/Xilinx                                    │
 │ licence file / port@server     │── ro / network ─────►│ XILINXD_LICENSE_FILE                             │
 │ /tmp/finn_build_$UID           │══ rw ═══════════════►│ FINN_BUILD_DIR <- outputs, XSI cache             │
 └────────────────────────────────┘                      │                                                  │
   ▲                                                     │ baked in the image:                              │
   │ docker/config.py resolves                           │   /opt/venv    locked packages, active via ENV   │
   │ all of the above                                    │   XRT/SLASH    if FINN_RUNTIMES selected them    │
                                                         │   (hlslib: editable from the checkout's         │
                                                         │    packages/; boards: a wheel in /opt/venv)      │
                                                         └──────────────────────────────────────────────────┘
```

## Co-developing QONNX or Brevitas

This uses uv's own mechanism, with no FINN tooling. Point the source at your checkout
locally:

```toml
# pyproject.toml, local edit, don't commit
[tool.uv.sources]
qonnx = { path = "../qonnx", editable = true }
```

```text
 pyproject.toml  (local edit)                                 resulting venv
 ┌──────────────────────────────────────┐                    ┌────────────────────────────────────┐
 │ [tool.uv.sources]                    │      uv sync       │ finn      editable   ./            │
 │ qonnx = { path = "../qonnx",         │ ─────────────────► │ qonnx     editable   ../qonnx      │
 │           editable = true }          │  (natively, or a   │ brevitas  from uv.lock             │
 └──────────────────────────────────────┘  new container)    │ numpy...  from uv.lock             │
   uv re-locks too: commit neither file                      └────────────────────────────────────┘
   In Docker, ../qonnx must be mounted at the same relative place.
```

Then `uv sync` (natively, or inside a running container; a new container picks it up at
start). This works in every modality as long as `../qonnx` is visible there (in Docker,
mount the parent directory). The rough edge: uv re-locks, so `uv.lock` also shows as
modified. It's rare enough that "don't commit those two files" is acceptable, and it's
much cheaper than the overlay system it replaces.

## Adding a dependency

The test of this design: **adding a dependency touches only `pyproject.toml` and
`uv.lock`** (plus, for one route, a git submodule). No Dockerfile, `docker/run`, sbx or
setup-script change, in any modality.

### First rule: build data becomes a Python package

Header libraries, board files, Tcl/IP libraries and so on don't get their own mechanism.
Wrap them as a **data-only Python package**: a `pyproject.toml` plus the files as package
data, like `certifi` or `tzdata`. From then on they're just a Python dependency, which
means locking, the image, editable co-development and PyPI all work with no extra
machinery. FINN finds the files with `importlib.resources`, and the environment variable
stays as an override:

```python
# src/finn/util/external.py (replaces the deps/ lookup in _legacy_build_env.external_path)
EXTERNAL = {
    "hlslib": ("FINN_HLSLIB_PATH", "finn_hlslib"),
    "boards": ("FINN_BOARD_FILES_PATH", "finn_boards"),
    "newlib": ("FINN_NEWLIB_PATH", "finn_newlib"),   # adding one = adding one line
}

def external_path(kind):
    variable, package = EXTERNAL[kind]
    return os.environ.get(variable) or str(importlib.resources.files(package))
```

So "a new build-data dependency" and "a new Python dependency" reduce to the same
question: **does it need to be editable?**

### Which route

```text
                              new dependency (Python, or data wrapped as a package)
                                                   │
                        ┌──────────────────────────┴───────────────────────────┐
                        │ developed in lockstep with FINN, so everyone should   │
                        │ always have it editable?                              │
                        └──────────────┬───────────────────────────┬───────────┘
                                   yes │                           │ no
                                       ▼                           ▼
                      ┌──────────────────────────────┐   ┌──────────────────────────────────────┐
                      │ B  WORKSPACE MEMBER          │   │ A  ORDINARY DEPENDENCY               │
                      │    git submodule under       │   │    PyPI version, or a pinned git     │
                      │    packages/, committed,     │   │    source in [tool.uv.sources].      │
                      │    editable for everyone     │   │    Anyone who occasionally wants it  │
                      │    in every modality         │   │    editable: a local path source,    │
                      │                              │   │    as for QONNX today                │
                      │    e.g. finn-hlslib          │   │    e.g. onnx, qonnx, brevitas        │
                      └──────────────────────────────┘   └──────────────────────────────────────┘

             third-party data FINN can't add a pyproject to (board repos): FINN owns a tiny
             wrapper package (finn-boards) built from pinned upstream commits  →  route A

             not redistributable, or too big for a wheel: the one remaining "fetch" case.
             FINN downloads it on first use into a cache keyed by a pinned checksum; the
             image pre-fetches it at build time
```

### A. Ordinary dependency (the default)

```toml
[project]
dependencies = [ …, "newdep>=1.2" ]

# only if it isn't on PyPI yet:
[tool.uv.sources]
newdep = { git = "https://github.com/org/newdep.git", rev = "<sha>" }
```

Then `uv lock` and commit both files. For occasional co-development, a developer edits the
source locally to `{ path = "../newdep", editable = true }` and doesn't commit it, exactly
like QONNX today.

### B. Workspace member (always editable, for everyone)

For something developed in lockstep with FINN, the same way the RTL library already is.
**finn-hlslib is the obvious first case**: HLS kernel changes often land together with
FINN changes.

```bash
git submodule add https://github.com/Xilinx/finn-hlslib.git packages/finn-hlslib
# finn-hlslib gets a small pyproject.toml of its own (AMD owns the repo)
```

```toml
# FINN's pyproject.toml
[project]
dependencies = [ …, "finn-hlslib>=2026.1" ]      # what a PyPI user gets

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
finn-hlslib = { workspace = true }                # what developers get: editable
```

Then `uv lock` and commit.

- **Pinning:** the submodule commit is the pin. Bumping hlslib is `git -C
  packages/finn-hlslib checkout <sha>` plus a commit, reviewed like any FINN change.
- **Every modality for free:** the member lives inside the checkout, so the mount that
  already brings FINN into a container brings it too. There's no extra `--volume`, and
  the entrypoint's `uv sync` installs it editable.
- **Image:** the pre-sync is `uv sync --frozen --no-install-workspace`, which skips FINN
  and every member. It needs only the root `pyproject.toml` and `uv.lock` (with
  `--frozen`, uv doesn't read member manifests). Members are never baked in; they're
  linked from the checkout at start, like FINN.
- **Publishing:** workspace sources aren't written into the published metadata. PyPI users
  get `finn-hlslib>=2026.1`, so every member has to be released to PyPI alongside FINN.
- **Failure mode:** a clone without `--recurse-submodules` has an empty `packages/<x>`, and
  `uv sync` fails naming the missing member. The fix is `git submodule update --init`.
  The entrypoint's warning should say exactly that.

### Picking it up after someone adds a dependency

```text
                   what you do after pulling                 image rebuild needed?
 ┌───────────────┬────────────────────────────────────────┬──────────────────────────────────┐
 │ Native        │ uv sync  (+ git submodule update       │ n/a                              │
 │               │   --init, for a new member)            │                                  │
 ├───────────────┼────────────────────────────────────────┼──────────────────────────────────┤
 │ Docker        │ nothing: the next docker/run syncs     │ no. The first start installs the │
 │               │   at start                             │ difference; rebuild only for     │
 ├───────────────┼────────────────────────────────────────┤ offline use or to make the first │
 │ Dev Container │ uv sync in the terminal, or rebuild    │ start fast again                 │
 ├───────────────┼────────────────────────────────────────┤                                  │
 │ sbx           │ uv sync in the sandbox                 │                                  │
 └───────────────┴────────────────────────────────────────┴──────────────────────────────────┘
```

## CI

```text
              ┌─► dev image + checkout (entrypoint sync) ────────────────────────────► full test suite
              │
 PR / push ───┼─► uv build ─► clean python:3.10 venv ─► pip install finn-*.whl ──────► import + resource smoke,
              │                                                                        small test subset
              │
              └─► (optional) uv sync --resolution lowest-direct ─────────────────────► unit tests at the
                                                                                       bottom of the ranges
```

The wheel job is the only guard against packaging mistakes that editable installs hide
(a file missing from the wheel, a broken entry point).

## What this deletes from the current branch

| Gone | Replaced by |
|---|---|
| `deps.env`, `fetch-repos.sh`, ~500 MB of board clones in the image | data-only packages: finn-hlslib (workspace member), finn-boards (wheel) |
| `requirements.txt`, `docker/requirements-*.txt`, `pip-constraints.txt`, `pip-torch.txt` | `pyproject.toml` + `uv.lock` |
| `wheelhouse.py`, `development-requirements.txt`, both input manifests | `uv.lock` |
| dependency/application images, dual revisions, `--dependencies`, `--venv` | one image |
| `prepare-editables`, `editable-requirements.txt`, the planned `scripts/dev-env` | `uv sync` + `[tool.uv.sources]` |
| most of `setup-local.sh`, `scripts/activate.sh` | `uv sync` + active venv (image `ENV`; natively `.venv/bin/activate` or direnv) |
| `python -m finn.xsi.setup` as a required step | built on first use |
| `FINN_HLSLIB_PATH` / `FINN_BOARD_FILES_PATH` as required settings | optional overrides |

**What FINN still maintains:** `pyproject.toml` + `uv.lock`, one Dockerfile,
`docker/run` + `docker/config.py` (Vivado mount, licence, uid, path mirroring), and the
sbx configuration.

## Checks before committing to this

- **Board file licences.** XilinxBoardStore, Avnet and RealDigital files must be
  redistributable in a PyPI wheel. hlslib is AMD's own and BSD-licensed, so it's fine. If
  some boards can't be redistributed, those few get fetched on demand into a cache when
  that board is requested; the rest still ship.
- **Building XSI on first use has to be concurrency-safe.** pytest-xdist workers will race
  for it, so use a file lock and an atomic rename. It should fail loudly, not silently,
  when no toolchain is selected.
- **`/opt/venv` must be writable** by the `--user $UID:$GID` that `docker/run` uses, because
  the entrypoint sync writes to it. One `chmod` in the image; the writable layer is
  discarded with `--rm`.
- **The entrypoint sync must never be fatal** (see above), and `python -m
  finn.util.installation` should report when the venv doesn't match `uv.lock`, so a
  skipped or failed sync is visible.
- **uv becomes required for developers,** not for users. pip developers can still use
  `pip install -e . --group dev`, with pip ≥ 25.1.
