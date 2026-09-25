# Migration plan: current branch → simplified environment stack

Written 2026-09-25. Target design: `ENVIRONMENT_STACK.md`. Starting point: the uncommitted
working tree on `refactor/container-runtime-implementation`.

## Overview

Seven workstreams. Each one lands as its own reviewable commits and leaves every modality
working, so the migration can stop at any point without leaving a broken tree.

```text
  0 checkpoint ──► 1 pyproject + uv.lock ──┬──► 2 build data as packages ──┐
                                           ├──► 3 one image + entrypoint ──┼──► 6 CI ──► 7 delete + docs
                                           └──► 4 native path ─────────────┘
                   5 XSI built on first use  (independent; can run in parallel with 1-4)
```

**Already done, and kept as-is:** the `finn._data` resource move, `resource_path`, the
toolchain runner and the migrated callers, the XSI session work, host resolution in
`docker/config.py`, the runtime manifests, the toolchain shims and the sbx stage.

**Removed by the end:** about 1,200 lines across `deps.env`, `fetch-repos.sh`,
`wheelhouse.py`, both input manifests, six requirements/constraints files,
`prepare-editables`, most of `setup-local.sh`/`activate.sh`, and their tests. On top of
that, the dependency/application code paths in `docker/lib.sh`, `docker/run`,
`docker/build`, `docker-bake.hcl` and `ci/scripts/build-images.sh`.

**External blockers,** listed now so they can start early:

| Blocker | Blocks | Owner |
|---|---|---|
| Board file licences (XilinxBoardStore, Avnet, RealDigital) allow redistribution in a wheel | workstream 2, boards part | legal / board owners |
| finn-hlslib accepts a `pyproject.toml` (data-only package) | workstream 2, hlslib part | AMD, hlslib maintainers |
| Upstream agrees to uv for development | all of it | FINN maintainers |
| A QONNX release to depend on | publishing FINN to PyPI (not development) | QONNX maintainers |

---

## 0. Checkpoint the current tree

Nothing has been committed yet, and several of the files below get deleted. Make one WIP
commit on the branch first, so the dependency/application implementation stays
recoverable. The prototype is already on `archive/container-runtime-prototype`. Squash or
reorganize for review at the end (workstream 7).

---

## 1. `pyproject.toml` + `uv.lock`

Everything else depends on this.

**Add `pyproject.toml`:**
- `[project]`:
  - name; `dynamic = ["version"]`, with `[tool.setuptools.dynamic] version = {file = "VERSION"}`
  - `requires-python = ">=3.10"` (what's actually tested; `setup.cfg` claims 3.9)
  - `dependencies`: the **runtime** subset of `requirements.txt` plus `setup.cfg`
    `install_requires`, as ranges
  - `[project.scripts]` from `setup.cfg` `[options.entry_points]`
- `[dependency-groups]`:
  - `test`: pytest stack, brevitas, dataset_loading
  - `notebooks`: jupyter, matplotlib, netron
  - `docs`: from `docs/requirements.txt` and the `docs` extra, reconciled (they disagree
    today)
  - `dev`: the other groups plus pre-commit
- `[tool.uv]`: a pytorch-cpu index for torch/torchvision/torchaudio. Git sources for
  qonnx, brevitas and dataset_loading at the commits `deps.env` pins today.

**First lock = today's versions.** Make the ranges include the current exact pins, and
constrain the first `uv lock` to them, so this step changes *nothing* about which
versions get installed. Loosening and upgrading come later, as separate reviewable lock
bumps. Mixing a refactor with upgrades is how the jupyter/anyio breakage happened.

**Drop finn-experimental** (and with it deap, mip, networkx): nothing in `src/` or
`tests/` imports them. Grep the notebooks first; if one needs it, it goes in `notebooks`.
Also drop `pre-commit`, `sphinx`, `pyscaffold` and `setupext-janitor` from the runtime
dependencies.

**Keep `setup.py`**, trimmed to the two build hooks (`BuildPy` / `Sdist` provenance) and
the `package_data` for `finn._data`. The metadata moves to `pyproject.toml`. From
`setup.cfg`, delete `[metadata]`, `[options]`, `[options.extras_require]`,
`[options.entry_points]`, `[aliases]`, `[bdist_wheel]`, `[build_sphinx]`,
`[devpi:upload]` and `[pyscaffold]`. Move `[tool:pytest]` to `[tool.pytest.ini_options]`.
Keep `[flake8]`, which can't read `pyproject.toml`.

**Delete:** `requirements.txt`, `docker/requirements-dev.txt`,
`docker/requirements-build.txt`, `docker/pip-constraints.txt`, `docker/pip-torch.txt`,
`docs/requirements.txt`, and `.travis.yml` (dead). Update `.readthedocs.yaml` to install
the `docs` group, using a `build.jobs` step with uv.

**Done when:**
- `uv lock` resolves to the same versions as today's image; diff `uv pip freeze` against
  the current image
- a native `uv sync` then the unit tests pass
- `pip install .` into a clean venv works
- the wheel from `uv build` contains `finn/_data/**` (`tests/util/runtime_resource_smoke.py`
  already exists for this)

At this point the images still build the old way. Nothing else has changed yet.

---

## 2. Build data as packages

**finn-hlslib, as a workspace member:**
- Upstream PR to finn-hlslib: a `pyproject.toml` making it a data-only package
  (`finn_hlslib`, with the headers as package data). Until it merges, point at a fork
  branch.
- `git submodule add … packages/finn-hlslib`. In `pyproject.toml`: `[tool.uv.workspace]
  members = ["packages/*"]`, `finn-hlslib = { workspace = true }`, and `finn-hlslib` in
  `dependencies`.

**finn-boards, as an ordinary dependency** (after the licence check):
- A small repo containing the **assembled** board files (the 20 MB that `fetch-repos.sh`
  builds today) and a `pyproject.toml`. Bumping a board = a commit there plus a release.
  Committing the assembled files beats a fetch-at-build-time hook: the diffs are
  reviewable and wheel builds need no network.
- Any board that can't be redistributed gets a cached, checksum-pinned download on first
  use instead.

**The lookup in FINN:**
- New `src/finn/util/external.py`: an environment-variable override, otherwise
  `importlib.resources.files(package)`. This replaces `external_path` in
  `_legacy_build_env.py`.
- Update the callers: `custom_op/fpgadataflow/hlsbackend.py`,
  `transformation/fpgadataflow/make_zynq_proj.py`.
- Update `tests/util/test_runtime_codegen.py` and `test_runtime_toolchain.py`.

**Delete:** `deps.env`, `fetch-repos.sh`, and `/deps/` from `.gitignore`.
`ci/Jenkinsfile_Brevitas` currently `sed`s `BREVITAS_COMMIT` into `fetch-repos.sh`. Change
that to overriding the brevitas source rev in `pyproject.toml`, then
`uv lock --upgrade-package brevitas`.

**Done when:**
- C++ simulation compiles against hlslib found through the package
- a Zynq project build finds the board files
- `FINN_HLSLIB_PATH` still overrides
- a clone without submodules gives a clear error

---

## 3. One image, plus the entrypoint sync

**`docker/Dockerfile.finn`:**
- Stages become `system → python → runtime → sbx`, plus an optional `release`.
- Delete the `wheels`, `dependencies`, `application-wheel` and `application` stages, the
  system-wide pip install in `base`, the `fetch-repos.sh data` step (with its ~500 MB of
  clones), and `PIP_BREAK_SYSTEM_PACKAGES`.
- The `python` stage:
  - installs uv at a pinned version
  - runs `uv sync --frozen --no-install-workspace`, with bind mounts of `pyproject.toml`
    and `uv.lock`
  - sets `ENV VIRTUAL_ENV=/opt/venv`, `PATH=/opt/venv/bin:…` and
    `UV_PROJECT_ENVIRONMENT=/opt/venv`
  - runs `chmod -R a+rwX /opt/venv`
  - **keeps the build-time checks** (`uv pip check`, the import smoke test, the pytest
    probe). Those are worth having.
- Move the sbx-only settings (the npm prefix, the `.claude` directories, `NO_PROXY` for
  the Docker subnet) into the `sbx` stage.
- Rewrite the historical comments so they describe the current state.

**`docker/finn_entrypoint.sh`:** add the non-fatal sync, i.e. `uv sync --frozen --inexact
--project "$FINN_ROOT"`.
- No checkout: print a note and carry on.
- Sync fails: warn with the fix and carry on.
- Missing submodule: say `git submodule update --init`.
- Always `exec "$@"` at the end.

**Build and run scripts:**
- `docker-bake.hcl`: remove the `finn-dependencies-*` targets, the
  `FINN_DEPENDENCY_REVISION`/`FINN_APPLICATION_REVISION` variables and labels, and the
  `*_COMMIT` arguments. Simplify `tag()`. Add an optional `finn-release` target.
- `docker/lib.sh`: one image-input list (Dockerfile, runtime manifests, entrypoint, shims,
  optionally `uv.lock`) instead of two manifests. Remove the dependencies branch from
  `finn_bake_target` and simplify `finn_set_provenance`.
- `docker/run`, `docker/build`: remove `--dependencies`, `--venv`, the
  `FINN_DEPS`/`FINN_FETCH_DEPS` handling, and the related `--print` fields.
- `docker/config.py`: remove the `FINN_DEV_ENVIRONMENT` → `/env/venv` mount and the
  artifact/revision pass-through.

**Dev Container:**
- `initialize.sh` resolves the single image.
- `devcontainer.json`: remove `postCreateCommand` and `remoteEnv`, since the entrypoint
  and the image `ENV` now do their jobs. `python.defaultInterpreterPath` becomes
  `/opt/venv/bin/python`.

**sbx:** `docker/sbx/sbxenv.yaml` doesn't set `FINN_ROOT`. Pass the workspace path through
to it, or the entrypoint must fall back to the sandbox workspace. Remove the venv
preparation steps from `docker/sbx/README.md`.

**CI image publishing:** `ci/scripts/build-images.sh` records provenance as the `uv.lock`
hash plus the image digest, not `wheelhouse.json` and the application wheel's sha256.

**Done when** each of these works:
- `docker/run -- pytest -m util`
- `docker/run --fpga` running a C++ simulation test
- as a host uid other than 1000
- with `uv.lock` changed since the image was built (installs the difference)
- with no checkout mounted (still starts)
- Dev Container: opens with `/opt/venv` active and FINN imported from the checkout
- sbx: creates, and `sbx exec python -c 'import finn; print(finn.__file__)'` points at the
  checkout

---

## 4. Native path

- `setup-local.sh`: shrink to a system-dependency check (reuse
  `scripts/install-system-deps.sh`), `uv sync`, and a hardware readiness report. Or
  delete it and document the three commands.
- `scripts/activate.sh`: delete. Document `source .venv/bin/activate` or a one-line
  direnv `.envrc`. The Vivado settings it used to source are selected through
  `FINN_XILINX_PATH` / the toolchain runner now.
- `.github/workflows/quicktest-local.yml`: `uv sync`, then run the quicktest.
- `ci/Jenkinsfile`'s setup-local stage: base the cache key on `uv.lock`, not
  `requirements.txt`.
- **Delete:** `scripts/prepare-editables`, `editable-requirements.txt`,
  `tests/util/test_prepare_editables.py`.

**Done when:** a fresh clone, then `uv sync`, then the unit tests pass, and co-developing
QONNX with a local path source works.

---

## 5. XSI built on first use (independent)

- Add a `finn.xsi.ensure_built()` that builds into a cache keyed by tool identity and
  Python ABI (`<cache>/<tool>-<soabi>/xsi.so`), with an `fcntl` file lock and an atomic
  rename. It fails loudly when no toolchain is selected.
- Call it where the code now uses `find_xsi_so()` / `is_available()`: `util/rtlsim.py`,
  `xsi/__init__.py`, and the rtlsim paths in `hlsbackend.py` and `rtlbackend.py`.
- `is_available()` changes meaning to "can be built": a toolchain is present. Update the
  test skips in `tests/fpgadataflow/test_fpgadataflow_{mvau,thresholding_runtime,shuffle,convinputgenerator_rtl_dynamic}.py`.
- `python -m finn.xsi.setup` stays, for pre-building.

**Done when:** a synthetic test with parallel builders racing (in the style of the
existing session tests) passes, and a real rtlsim run works with Vivado (part of P8).

---

## 6. CI

- **New job** (GitHub workflow): `uv build`, then a clean `python:3.10` venv, then
  `pip install finn-*.whl`, then `tests/util/runtime_resource_smoke.py` and a small test
  subset.
- **Optional job:** `uv sync --resolution lowest-direct`, then the unit tests.
- Jenkins needs no structural change. It already mounts source rather than baking it, and
  `ci/Jenkinsfile_CI` calls `docker/run -- pytest …`, which keeps working.

---

## 7. Delete what's left, and the docs

**Delete:**
- `docker/wheelhouse.py`
- `docker/dependency-inputs.txt`, replaced by the single list from workstream 3
- `tests/util/test_wheelhouse_manifest.py`
- the dependency/application and venv cases in `tests/util/test_container_config.py`,
  `test_container_cli.py`, `tests/container/test_container_conformance.py` and
  `test_runtime_installation.py`

**Docs:**
- Rewrite `docs/installation.md` around the three commands.
- Update `docker/README.md`, `docker/sbx/README.md` and `docs/finn/getting_started.rst`.
- Consolidate the 14 design notes in `docs/` into one, with `ENVIRONMENT_STACK.md` as its
  basis.
- Update `CONTAINER_RUNTIME_STATUS.md`.
- Remove my review files (`CONTAINER_RUNTIME_REVIEW.md`, `PYTHON_DEPS_PROPOSAL.md`, this
  file) once their content has moved into the real docs.

**Then** reorganize the branch into reviewable commits, one or more per workstream, with
the resource move as its own pure-rename commit.

---

## Where this fits the original plan

- **P6 (build-engine integration)** is independent of this. It can go before or after.
- **P7 (removing compatibility hooks)** gets smaller, because workstreams 3 and 7 remove
  most of what P7 was going to reconcile.
- **P8 (real Vivado/HLS/XSI, FPGA, SIF validation)** should run *after* workstreams 3 and
  5. Those are the changes that most need it, and running P8 before them would mean doing
  it twice.

## Suggested first move

Workstream 1, on its own. It's the foundation, it changes no installed versions, it can be
fully tested without Vivado, and it doesn't depend on any of the external blockers. Start
the licence question and the finn-hlslib `pyproject.toml` PR at the same time, since those
take calendar time rather than effort.
