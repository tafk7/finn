# Review: container runtime and dependency/environment management

Reviewed 2026-09-23 on `refactor/container-runtime-implementation` (uncommitted working tree).

**Overall:** the architecture is sound. It fixes the right problems, in roughly the right
order. The main weaknesses are in dependency management, not in the container plumbing.
Right now the setup looks more reproducible than it is, and the everyday developer
workflow involves a lot of ritual. Fix those two things before putting more polish on the
rest.

## What's working well

- **Moving resources into `finn._data` is the most important change.** An installed wheel
  now works without a source checkout, and that's what makes the application/dependency
  split, SIF images and native installs possible. `resource_path` also refuses paths that
  escape their resource family.
- **Hidden behaviour is gone.** Nothing installs packages at startup, `FINN_DEPS`/`--deps`
  fail loudly, and mounting a checkout no longer means it gets imported. The failure modes
  are much easier to understand.
- **The image build checks itself.** `pip check`, an import smoke test and a real pytest
  run are all done during the build (`docker/Dockerfile.finn:257`). That catches the
  jupyter/anyio breakage that `pip check` alone passes.
- **The new way of running vendor tools fixes real bugs.** Commands are now argument lists
  rather than shell strings, the whole process group is killed on timeout, and exit codes
  are checked. Before this, `CppBuilder.build` and `exec_precompiled_singlenode_model`
  ignored failures: a g++ error went unnoticed until something later broke.
- **Host facts are resolved in one place**, `docker/config.py`. Bake is the only source of
  image tags.
- **Running XSI simulation in a separate process is the right model** for a native library
  that can crash or leak, and checking artifact hashes at both ends is a good touch.

## Concerns, most important first

### 1. The lock file isn't really a lock

`docker/wheelhouse.py` writes a hash-pinned `development-requirements.txt`, but it's
generated inside the build from inputs that aren't fixed:

- `onnxoptimizer`, `netron` and `setupext-janitor` have no version pin.
- Every transitive dependency is unpinned.
- The build runs `apt-get upgrade`.

The `deps-<hash>` tag is a hash of those inputs, not of what actually got resolved. So two
machines can hold different environments under the same tag, and `finn_prepare_image`
(`docker/lib.sh`) will reuse whichever one is already present. The Dockerfile comment
admits "the digest is the identity", but the tag reads as a promise.

**Fix:** a lock file committed to the repo, which the wheels stage installs from. Then the
input hash really does describe the environment.

### 2. Version pins are spread across seven places

They live in `requirements.txt`, `docker/requirements-dev.txt`,
`docker/requirements-build.txt`, `docker/pip-constraints.txt`, `docker/pip-torch.txt`,
`setup.cfg` `install_requires` (unpinned, and a different set) and `deps.env`. Some
problems that come out of this:

- `setuptools` is pinned twice.
- The `docs` extra conflicts with `requirements.txt`: `sphinx_rtd_theme` 0.5.0 vs 2.0.0,
  and `qonnx@main`.
- Runtime `requirements.txt` includes `pre-commit`, `sphinx` and `pyscaffold`.

Most of this existed before the refactor, but this refactor is the natural time to
consolidate.

### 3. The environment is stored three times

The base stage installs everything into the system Python. The dependencies stage then
adds the wheelhouse: the local `sbx-deps` image is 5.93 GB against 5.02 GB for the
application image. After that, every development workflow builds a separate venv and
installs every wheel again, so torch and the rest end up in the image twice and in the
venv a third time.

Either the dependencies image shouldn't also install into the system Python, or
development venvs should build on that install instead of repeating it.

### 4. Day-to-day development is clunky

- The documented Docker setup (`docs/installation.md:69`) is a five-line `bash -c`
  command.
- There are four venv locations: `/env/venv`, `/home/agent/.venvs/finn-dev` (Dev
  Container), `$HOME/.venvs/finn-dev` in sbx, and `.venv` natively.
- `scripts/prepare-editables` only covers the last step.
- Nothing notices when a venv kept on the host outlives the dependency image it was built
  from.

**Suggestion:** a single command that creates the venv, installs the wheels, adds the
editable checkouts, and records the dependency revision in the venv. Later runs can then
warn when that recorded revision no longer matches the image.

### 5. Some behaviour changes haven't been validated

- `src/finn/custom_op/fpgadataflow/hlsbackend.py` used to choose `HLS_PATH` or
  `VITIS_PATH` by Vivado version. It now uses whichever of `XILINX_HLS`, `HLS_PATH`,
  `XILINX_VITIS` or `VITIS_PATH` is set first. For 2024.2 in particular that can pick a
  different install than before.
- `finnxsi = xsi` replaced a real availability check.
- The Python adapter is now a top-level `finn_xsi` package in the wheel, and the worker
  loads the bridge as `sys.modules["xsi"]`. Both are generic top-level names that could
  collide with other packages.
- `finn_xsi/testcase` is still at the repo root, and `MANIFEST.in` still prunes the old
  paths.

### 6. Many comments describe history rather than the current state

Examples: "used to", "an earlier revision", "§3.4", and "That check is now a removed" at
`docker/Dockerfile.finn:376`. The comment describing the runtime stage
(`docker/Dockerfile.finn:431-444`) now sits above the `dependencies` stage. There are also
14 design documents (about 4k lines) in `docs/`. Upstream reviewers will pay for all of
this. Comments should say what the code does now, and the design notes should be cut down
to one document.

### 7. sbx-specific details leak into the shared base image

`NPM_CONFIG_PREFIX`, the `.claude` directories under `/home/agent` and a `NO_PROXY` for the
Docker subnet all apply to every image, not just sbx. Separately, setting
`PIP_BREAK_SYSTEM_PACKAGES` pulls against the move to venvs.

## On the open pip vs uv decision

Go with **uv**, mainly because it gives you a committed lock file, which is the fix for
concern 1; speed is a bonus. You can keep a pip-compatible path for offline HPC and SIF use
by exporting the lock (`uv export --format requirements-txt`) and feeding it to the
existing wheelhouse step. The spike already covered the hard cases. If uv is politically
difficult upstream, `pip-compile --generate-hashes` gets you most of the benefit with plain
pip.

## Order to tackle the rest in

1. Run one real HLS C simulation, one real XSI simulation and one real synthesis (the
   minimum of P8) before P6. The toolchain changes touched `alveo_build` and
   `make_zynq_proj`, and so far none of it has run against actual vendor tools.
2. Commit a lock file, then collapse the pin files into it.
3. Add the single environment-setup command with the revision check.
4. Split the history into commits. Make the resource move a pure-rename commit on its own:
   at the moment the index shows the `rtllib` files as additions rather than renames,
   which hides that their content is unchanged.

## Limits of this review

- No tests were re-run. The host has no Python environment with pytest, and building
  images just for this review didn't seem worth it, so the 182-test figure in
  `CONTAINER_RUNTIME_STATUS.md` is still the latest result.
- The sbx, CI and native-provisioning scripts weren't read in depth.
