# Python dependencies: a simpler model

Written 2026-09-23 as a follow-up to `CONTAINER_RUNTIME_REVIEW.md`.

## Short answer

Yes, the current design is overcomplicated, and mostly for the Python half. The container
should be the **editable development environment**. FINN is always installed editable
from the mounted checkout, and third-party packages come from a committed lock and are
pre-installed in the image. You don't need a separate non-editable "application"
architecture. The one real consumer of a non-editable image is read-only images: SIF/HPC
and published release tags. That can be a single extra stage built from the same recipe,
not a parallel system with its own identities.

The reason: the container's job is everything pip *can't* do. That's the Vivado/Vitis
mounts, licences, system libraries (ncurses 5, the LSB loader, the libudev preload),
XRT/SLASH, g++, and the hlslib/board data. Python packaging is the solved problem. Right
now the design reinvents it: a wheelhouse manifest, two artifact identities, input-file
hashing, `--dependencies` flags and a custom editables helper.

## A correction first: "native install is just a pip install" isn't true yet

It's close, but three things are in the way:

1. **QONNX is pinned to a git commit** (`deps.env`). PyPI rejects packages whose metadata
   has direct git dependencies, so `pip install finn` from PyPI needs a QONNX release to
   depend on.
2. **finn-hlslib and the board files are data, not Python.** A pip install doesn't bring
   them, and they're still reached through `FINN_HLSLIB_PATH` / `FINN_BOARD_FILES_PATH`.
3. **The pins are exact** (`numpy==1.24.1`, `onnx==1.17.0`, ...). That's right for a lock
   and wrong for a library's metadata.

Everything else is fine. What `pip install` can't handle (Vivado, licences, the XSI
extension built against the user's Vivado) is a host concern whichever way you go.

## Finding: FINN's real Python dependencies are much smaller than the setup assumes

I grepped `src/` and `tests/` for imports:

| Package | Imported in `src/` | Imported in `tests/` | Actually used for |
|---|---|---|---|
| qonnx | yes | yes | Runtime dependency |
| torch | 3 files | 39 files | Runtime (small) and tests |
| brevitas | **none** | 36 files | Tests and notebooks only |
| dataset_loading | **none** (host) | 1 end2end test | On-board PYNQ driver template, notebooks |
| finn-experimental | only `util/installation.py` (reporting) | none | Nothing I can find |
| deap / mip / networkx | **none** | **none** | Nothing I can find. They were only there because finn-experimental pulls them in |

So of the "sister repos", **only QONNX is a runtime dependency of FINN.** Brevitas and
dataset_loading are test/notebook dependencies. finn-experimental and its solver stack look
entirely unused. Confirm that (notebooks and CI scripts might still use it), then drop it.
That removes a git dependency, a broken package (the metadata-only wheel noted in the
Dockerfile) and three pinned transitive dependencies.

Co-development still matters for QONNX and Brevitas. But it's a development-environment
concern, not something the FINN package has to express.

## Proposed shape

### One source of truth: `pyproject.toml` plus a committed `uv.lock`

```toml
[project]
name = "finn"
requires-python = ">=3.10"
dependencies = [          # library-style ranges, not exact pins
  "qonnx>=…",
  "onnx>=1.17,<1.18",
  "numpy>=1.24,<2",
  # …
]

[dependency-groups]
test      = ["pytest…", "pytest-xdist…", "brevitas…", "dataset_loading…"]
notebooks = ["jupyter…", "matplotlib…", "netron"]
docs      = ["sphinx…", "sphinx_rtd_theme…"]
dev       = [{include-group = "test"}, {include-group = "notebooks"}, "pre-commit"]

[tool.uv.sources]
# Only used by uv to lock/sync the dev environment; never published metadata.
qonnx = { git = "https://github.com/fastmachinelearning/qonnx.git", rev = "f5c9819…" }
torch = { index = "pytorch-cpu" }
```

- `uv.lock` is the exact, hashed, tested environment for dev and CI. That fixes the
  "the lock isn't a lock" problem from the review for free.
- `requirements.txt`, `docker/requirements-*.txt`, `docker/pip-constraints.txt`,
  `docker/pip-torch.txt`, `setup.cfg` `install_requires`, and the Python half of
  `deps.env`/`fetch-repos.sh` all go away. `deps.env` keeps only the data repos.
- pip users still work: `pip install -e . --group dev` (dependency groups need pip ≥ 25.1),
  or `uv export` to a hashed requirements file for pip-only or offline sites.

### Workflows

| Who | What they run |
|---|---|
| User | `pip install finn`, then fetch the hlslib/board data, then Vivado for hardware flows |
| Native developer | `uv sync` (creates `.venv` with editable FINN plus the locked dependencies) |
| Co-developing QONNX/Brevitas | `uv sync`, then overlay `uv pip install --no-deps -e ../qonnx` |
| Docker/sbx/Dev Container | Same lock, same overlay, pre-built in the image |
| CI | Same as the container, **plus** one wheel job (below) |

The overlay step is what `scripts/prepare-editables` already does. It's the right idea;
it just becomes a thin wrapper, e.g. `scripts/dev-env`:
`uv sync --frozen && uv pip install --no-deps -e <each line of editable-requirements.txt>`.
One sharp edge: a later `uv sync` reinstalls QONNX from the lock and removes the overlay.
The wrapper makes re-overlaying automatic, which is why it's worth having a wrapper at all.

### Container

```
system  →  venv (uv sync --frozen --no-install-project → /opt/venv)  →  runtime (XRT/SLASH)  →  sbx
                                                                      ↘  release: + FINN wheel (tags only)
```

- **No system-Python install and no wheelhouse.** The dependencies live once, in
  `/opt/venv`. That fixes the "stored three times" problem.
- **FINN is editable from the mounted checkout.** `docker/run` puts
  `uv pip install --no-deps -e "$FINN_ROOT"` (plus overlays) in front of the command. It's
  offline, deterministic and takes about a second. That does break the "startup does not
  install" rule, but that rule was aimed at hidden, networked installs that change which
  code gets imported. An explicit, offline, no-dependency link of the checkout you just
  mounted is much closer to setting `PYTHONPATH`. The alternative is a persisted venv
  volume, as now, but created by `uv sync` from the image's uv cache. That also works; it
  just brings back the stale-venv problem.
- **sbx and the Dev Container** do the same step once at create time, since those
  environments live a long time anyway.
- **Release/SIF image:** `FROM runtime` plus `uv pip install finn-X.whl`. Apptainer images
  are read-only, so the startup overlay can't work there; that's the one real consumer of
  a baked FINN. Build it on release tags only.
- **Image identity** becomes a hash of `uv.lock`, the Dockerfile and the runtime manifests.
  FINN source is no longer baked into the dev image, so a source edit no longer means a new
  image tag.

### What editable-everywhere would hide, and the one CI job that covers it

With FINN always editable, packaging mistakes go unnoticed: a missing `package_data` file,
a broken entry point, a resource read that only works from a source tree. This is the real
value the application image was providing. You can keep that value with a single CI job:
`uv build` → install the wheel into a clean venv → run an import/resource smoke test and a
small test subset. That's much cheaper than a whole second image family.

## What survives from the current branch, and what goes

**Keep:** the `finn._data` resource move (it's what makes all of this possible), the
toolchain runner and the fixed callers, `docker/config.py` host resolution, the runtimes
manifests, the sbx stage, the XSI session work, `resource_path`, and the idea behind
`prepare-editables`.

**Remove or replace:** the dependency/application artifact split and dual revisions,
`docker/dependency-inputs.txt` and `docker/image-inputs.txt`, `docker/wheelhouse.py`,
`development-requirements.txt`, `--dependencies`/`--venv` flags,
`.devcontainer/initialize.sh`'s image-tag dance, the requirements/constraints files, and
most of the Python steps in `setup-local.sh`.

This partly undoes P2. P1 (packaging) is what makes the simpler model possible, and it
stays. The P2 split was solving a problem the simpler model doesn't have.

## Trade-offs to decide consciously

- **uv adoption upstream.** Contributors need uv for the locked dev flow. Users and pip-only
  sites don't, because of `uv export` and dependency groups in pip ≥ 25.1.
- **Loosening pins to ranges** exposes FINN to dependency versions it hasn't been tested
  against, which the exact pins used to hide. The lock keeps dev/CI exact. An optional CI
  job resolving `--resolution lowest-direct` tests the bottom of the ranges.
- **A QONNX release** is needed before FINN can be published to PyPI. Until then, the git
  source in `[tool.uv.sources]` covers development.
- **hlslib/board data** needs its own answer. Either ship the hlslib headers as package data
  (check licence and size), or add a `finn` command that fetches pinned commits into a user
  cache directory, with images pre-populating it. Keep the environment-variable overrides
  either way.
- **Python version.** `setup.cfg` says `>=3.9`, but only 3.10 is built and tested. Declare
  what CI actually covers.

## Suggested order

1. Confirm finn-experimental and deap/mip/networkx are unused, then drop them. Move Brevitas
   and dataset_loading into the `test` group.
2. Write `[project]` and the dependency groups in `pyproject.toml`, generate `uv.lock`, and
   delete the requirements/constraints files.
3. Restructure the Dockerfile to system → venv → runtime → sbx, with a release stage only
   on tags.
4. Replace `--dependencies`/`--venv` with the editable overlay step, and have one
   `scripts/dev-env` used by native, Docker, sbx and the Dev Container.
5. Add the wheel-build CI job.
6. Delete the wheelhouse, manifest and dual-revision machinery.
