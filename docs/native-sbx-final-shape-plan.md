# Implementation plan: native sbx examples, Docker-only launcher

Status: implemented and validated; see the completion record below.
Written: 2026-09-15.
Branch: `refactor/container-stack`, following `0e78d9dd8` and `a6665654b`.

## Objective

FINN provides a reproducible development image, convenient Docker execution,
and native sbx configuration examples. Users and their sites own instantiated
sandbox configuration, agent selection, credentials, mounts, and network policy.
Native sbx owns composition, approval, execution, and sandbox lifecycle.
Cardinal consumes these artifacts independently; FINN carries no Cardinal contract.

Keep the existing Python host-discovery behavior for Docker and native installation.
Make `docker/config.py` its single executable entry point. Remove the sbx runner,
sbx configuration generation, and forwarding aliases. Do not replace Python with
shell, migrate packaging to pyproject.toml, or redesign guest initialization here.

## Intended interfaces

```text
FINN image preparation
    docker/build
        --sbx                 build/import a compatible template
        --print-tag           report the selected image reference
        --export-sif PATH     export a SIF

Docker execution
    docker/run -- COMMAND
    docker/run --fpga -- COMMAND
        -> docker/config.py inspect / compose
        -> Compose

Native installation
    setup-local.sh + scripts/activate.sh
        -> docker/config.py inspect --format sh

sbx execution
    copy FINN's native examples outside mounted workspaces
    edit site configuration / supply native arguments
        -> sbx env plan / create / run / exec / rm
```

`docker/build --sbx` remains image preparation only. It must not create a sandbox,
register credentials, configure machine policy, or generate a user environment.
`docker/run` loses `--sbx` and `--remove`; removed options fail normally rather
than forwarding to a compatibility implementation.

## 1. Establish the native example contract

Before implementation, refresh a local checkout of the official sbx documentation
and record its revision together with installed client/server versions. The last
validated implementation used sbx 0.42.1; do not describe that as the latest version
without checking. Validate examples using native commands, not a local YAML merger.

Keep `docker/sbx/sbxenv.yaml` as the portable development example. It selects the
prepared template and accepts native arguments for sandbox name, checkout path,
and agent. Default to shell; keep FINN's scratch setting and disabled shared
writable skills. Do not add custom agent definitions or duplicate dependency
installation in a kit.

Add `docker/sbx/fpga.sbxenv.yaml` as an optional native overlay. Demonstrate explicit
read-only toolchain mounts and explicit Vivado/Vitis/HLS environment paths. Include
a clearly documented floating-licence example. Describe licence-file and external
platform-directory mounts as deliberate site additions, not automatic discovery.

Add `docker/sbx/site-license/spec.yaml` as a minimal example of a site-owned native
network mixin. Use example domains or required native arguments, never real sites
or a universal allow rule. Demonstrate the licence-manager and pinned vendor-daemon
ports. Explain the host-wide alternative for an unpinned vendor port without
silently selecting it. Do not claim successful real licence checkout from policy
readback or lmstat alone.

Document copying the base, selected overlay, and optional kit together outside
all mounted workspaces. Native file references must still work after copying.
The optional kit must not be included by the development example. State that
native lists concatenate, and that users must pass the same files and arguments
for subsequent lifecycle commands. Keep personal agent settings and credentials
in user-owned native configuration.

## 2. Make the Python resolver a single executable

Set the executable bit on `docker/config.py`, update its argparse program name and
module description, and migrate all repository callers to that path.

Delete `docker/config` and `docker/finn-env` after migrating their callers.
Keep direct Python imports in tests; no replacement wrapper or package is needed.

Remove `cmd_sbx`, its parser command, bundle-writing code, and generator-specific
imports. Remove `inspect --sbx` and internal sbx-only branches once their consumers
are gone. Retain the existing Docker/native path discovery, licence classification,
version/layout probing, UID/GID handling, and shell/Compose output behavior.
Keep any remaining network description explicitly declarative; it must not claim
that Docker enforces the reported permissions.

Do not bundle a broader resolver redesign into this migration. Changes to default
path behavior, inspection side effects, shell/Compose quoting, and automatic host
discovery require separate behavior changes and tests if pursued later.

## 3. Remove FINN's sbx runtime orchestration

Delete `run_sbx()` and its option handling from `docker/run`. Remove sandbox-name
selection, sbx-specific printed fields, generated state-directory handling,
existence queries, approval/lifecycle calls, and sandbox removal logic.

Retain Docker command forwarding, explicit FPGA access, notebook support, image
preparation/reuse, and existing Docker configuration behavior.

Retain `finn_prepare_sbx()` in `docker/lib.sh` for the build command only. Remove
branches that existed solely for runtime reuse of an already-imported template
if they no longer have a caller. Keep explicit preparation/import idempotent and
preserve useful error handling. Do not remove templates or sandboxes as cleanup
for unrelated user resources.

Do not delete existing generated FINN environment directories. They are ordinary
native files and may describe active user sandboxes. Document how to keep using
those files directly with sbx, or replace them with user-owned copies of the new
examples. No automatic state migration is necessary.

## 4. File-by-file implementation scope

| File | Planned change |
| --- | --- |
| `docker/run` | Make Docker-only; remove sbx runtime options and lifecycle code; invoke `config.py`. |
| `docker/build` | Retain sbx template preparation, image-reference output, and SIF export; clarify help. |
| `docker/lib.sh` | Keep shared image preparation and build-only sbx import; remove runtime-only branches. |
| `docker/config.py` | Make executable; retain Docker/native discovery; remove sbx inspection/generation surfaces. |
| `docker/config` | Delete forwarding wrapper. |
| `docker/finn-env` | Delete compatibility alias. |
| `docker/sbx/sbxenv.yaml` | Finalize portable, copyable native development example. |
| `docker/sbx/fpga.sbxenv.yaml` | Add explicit site-parameterized FPGA overlay example. |
| `docker/sbx/site-license/spec.yaml` | Add optional example licence-network mixin. |
| `scripts/activate.sh` | Invoke `config.py` directly; preserve activation behavior. |
| `setup-local.sh` | Invoke `config.py` directly; preserve native setup behavior. |
| `compose.yaml` | Update resolver references in comments; preserve service definitions. |
| `docker-bake.hcl` | Update obsolete resolver references in comments; preserve image targets. |
| `README.md` | Present Docker convenience versus direct native sbx workflows. |
| `docker/README.md` | Replace sbx-runner/generator instructions with preparation, copying, composition, and migration. |
| `docker/sbx/README.md` | Add concise base/FPGA example instructions and site-owned configuration guidance. |
| `docs/finn/getting_started.rst` | Update new-user Docker and sbx instructions, including agent plus FPGA use. |
| `docs/finn/developers.rst` | Document final ownership boundaries and single Python entry point. |
| `docker/runtimes/README.md` | Update command references while preserving accelerator-image guidance. |
| `docker/runtimes/slash.env` | Update resolver references in comments if present. |
| `ci/README.md` | Update configuration references; preserve explicit shared-image loading. |
| `tests/util/test_container_cli.py` | Replace sbx-runner tests with Docker-only CLI and template-preparation checks. |
| `tests/util/test_container_config.py` | Migrate executable path; retain discovery tests; remove sbx-renderer-only expectations. |
| `tests/container/test_container_conformance.py` | Exercise copied native examples directly and preserve Docker/SIF coverage. |
| `docs/native-sbx-final-shape-plan.md` | Record completion, final validation evidence, and any deviations. |

Run a repository-wide caller scan, including notebooks, CI definitions, shell
scripts, and tests. Any additional real caller discovered must be migrated; extend
this inventory with its exact path. Historical descriptions and negative tests
may mention removed names, but runnable examples and active callers may not.

No changes are planned to dependency declarations, Dockerfile guest behavior,
retained guest hooks, Dev Container configuration, Jenkins orchestration, or
Cardinal itself.

## 5. Validation and acceptance

- Run focused Python resolver, public CLI, and CI transport/configuration tests.
- Verify `config.py` can be executed directly and imported by tests.
- Verify removed config aliases, sbx generation, and sbx runner flags fail rather
  than invoking a hidden compatibility path.
- Check shell syntax, Python lint/formatting, and whitespace errors.
- Verify `docker/build --sbx --print-tag` and explicit template preparation still
  work without discovering FPGA paths or configuring a sandbox.
- Copy native examples into temporary user-owned configuration outside mounted
  directories. Use an isolated checkout and unique sandbox names.
- Test native development plan/create/exec/reuse/rm with shell, then at least one
  supported coding agent. Verify agent selection and FPGA overlay composition
  together; agent authentication limitations must be reported explicitly.
- Use a synthetic FPGA tree and example licence host to verify read-only mounts,
  declared tool paths, and native network-rule readback. This is not proof of
  hardware operation or a real licence checkout.
- Check that development configuration has no optional FPGA mounts or site kit.
  Do not infer closed networking from the absence of FINN network grants.
- Preserve a Docker smoke test with FINN imports and explicit FPGA discovery.
  Check native activation/setup callers without modifying the user's installation.
- Clean up only test-owned resources. Keep global sbx policy and unrelated
  credentials, templates, sandboxes, and generated user configurations intact.

Completion means no FINN-owned sbx runtime or generator remains, all active callers
use the surviving interfaces, and the native examples support choosing an agent
and adding explicit FPGA configuration in the same environment.

## 6. Commit and review sequence

1. Add and validate the native development/FPGA examples and site-configuration guide.
2. Migrate the Python executable and callers; remove wrappers and sbx generation.
3. Remove sbx execution from `docker/run`; update conformance tests and migration docs.

Keep each commit internally usable and revise tests alongside behavior. Perform
work on `refactor/container-stack`; do not merge or push as part of implementing
this plan. Record tested versions, commands, evidence locations, and unverified
agent authentication, FPGA licensing, SIF, or Jenkins behavior in the completion
record.


## 7. Completion record — 2026-09-15

Implemented on `refactor/container-stack`. No push or merge was performed.

- `e63033350`: copyable native development/FPGA examples, optional parameterized
  site licence mixin, and native usage guide.
- `b5690426b`: executable Python resolver, migrated callers, removed aliases and
  generation, Docker-only launcher, build-only template import, docs and tests.
- This completion record is committed separately.

The resolver still handles Docker/native discovery and shell/Compose output.
There is no FINN sbx runtime or configuration generator. Removed options are
unknown arguments, and removed wrappers do not exist. Existing generated user
configuration directories were neither migrated nor deleted.

### Native contract and tested versions

Refreshed the official documentation checkout with `git fetch origin main` and
`git merge --ff-only origin/main` before implementation:

- Source: <https://github.com/docker/docs>
- Local checkout: `/tmp/sbxc-audit-2026-09-12-docker-docs`
- Revision: `083f66104491efe09ed5347367a6e674eaac214d`
- Read `content/manuals/ai/sandboxes/configuration/environment-files.md`,
  `customize/kit-reference.md`, `customize/templates.md`, and agent guidance.
- `sbx version --json`: client and server `v0.42.1`, revision
  `cc6e400a4a3ce3ce5e0b2b77b8ee352aac854c64`; server API `0.28.0`.
  These are tested installed versions, not a latest-release claim.
- Docker client/server `29.8.0`; host Python `3.12.3`, pytest `9.1.1`,
  Ruff `0.16.7`, Black `23.3.0`.
- User-owned installation in the test sandboxes reported Claude Code `2.1.236`.
  Its vendor installer was read from `https://claude.ai/install.sh`, following
  <https://code.claude.com/docs/en/setup>.

### Validation evidence

The evidence paths below are local temporary logs on the implementation host.
The tests and public commands remain in the repository for reproduction.

| Check | Result | Evidence |
| --- | --- | --- |
| Resolver, CLI, CI transport, CI synchronization and CI configuration | 151 passed | `/tmp/finn-final-focused.log` |
| Copied shell example: plan/create/exec/run detached/reuse/rm; source-only image identity; Bake tags | 3 passed | `/tmp/finn-final-native-shell-artifact.log` |
| Copied Claude development and Claude + FPGA + site kit: native lifecycle, executable, imports, mounts and policy readback | 2 passed | `/tmp/finn-final-agent.log` |
| SIF runtime check | Skipped: neither Apptainer nor Singularity installed | `/tmp/finn-final-native-shell-artifact.log` |
| Initial copied native plans, including resolved relative kit reference and agent + FPGA composition | Passed | `/tmp/finn-final-native-gpyouhaa/dev.log`, `/tmp/finn-final-native-gpyouhaa/fpga-agent.log` |
| Explicit sbx image build/import and repeated import | Passed; no template removal | `/tmp/finn-final-sbx-build.log`, `/tmp/finn-final-sbx-repeat.log` |
| Docker image preparation | Passed | `/tmp/finn-final-docker-build.log` |
| Public Docker launcher: FINN/QONNX/Brevitas imports and explicit synthetic FPGA discovery/read-only mounts | Passed | `/tmp/finn-final-docker-u5lklab7/smoke.log` |
| Shell syntax, launcher/library ShellCheck, Python lint/Black, whitespace | Passed | `/tmp/finn-final-shellcheck.log` (empty on success), terminal checks |

Focused suite command:

```bash
/tmp/finn-sbx-test-venv/bin/pytest -q \
  tests/util/test_container_config.py tests/util/test_container_cli.py \
  tests/util/test_ci_container_transport.py tests/util/test_ci_config_sync.py \
  tests/util/test_finn_ci_config.py
```

Native/runtime commands (run as two focused selections):

```bash
FINN_TEST_SBX_TEMPLATE=xilinx/finn:sbx-env-29ed58ffeb42c45f \
  /tmp/finn-sbx-test-venv/bin/pytest -q -s \
  tests/container/test_container_conformance.py -k '05_copied and claude'
FINN_TEST_SBX_TEMPLATE=xilinx/finn:sbx-env-29ed58ffeb42c45f \
  /tmp/finn-sbx-test-venv/bin/pytest -q -rs \
  tests/container/test_container_conformance.py -k 'shell or 05b or 11 or 12'
```

Prepared references: `xilinx/finn:env-29ed58ffeb42c45f` and
`xilinx/finn:sbx-env-29ed58ffeb42c45f`. `docker/build --sbx --print-tag`
reported the latter directly. Repeated explicit preparation also passed with
`FINN_XILINX_PATH=/invalid/must-not-discover` and inherited
`FINN_CONTAINER_NO_BUILD=1`, demonstrating that the build path does not perform
FPGA discovery and remains an explicit preparation request.

Native tests copied the checked-in examples and site kit outside every mounted
directory, used separate local clones and unique sandbox names, and exercised
native composition rather than a local YAML merger. The development plans had
no optional mounts or site kit. FPGA checks used synthetic 2022.2 directories,
explicit environment aliases, rejected writes and readback of
`license.example.com:2100` / `license.example.com:2101`. These checks do not
establish closed networking, hardware operation or a real licence checkout.

Native activation was sourced from an isolated installation fixture; the actual
setup script's resolver expression was executed separately without running
installation steps. Unit tests also cover direct executable use and Python imports,
removed-interface rejection, SIF export transport and template-only preparation.

Final readback with `sbx ls --json` showed no remaining test sandboxes. No global
policy, credentials, unrelated templates or user-generated environment files
were changed. Explicitly prepared FINN images/templates were retained.

### Additional inventory and deviations

Additional references found and migrated beyond the planned table:

- `.gitignore`: obsolete resolver name in a comment.
- `docker/finn-toolchain.sh`: host resolver reference in a comment only; guest
  behavior was unchanged.

The final scan included hidden CI/Dev Container files, notebooks, scripts and
tests. Old executable names remain only in migration prose, historical material
and negative assertions. No dependency declarations, Dockerfile behavior,
Dev Container configuration, Jenkins orchestration or Cardinal files changed.

Steps 2 and 3 were committed together because the old sbx launcher depended on
the removed generator; separate removal commits would leave a broken launcher.
Git lacked an author identity. The user supplied Thomas Keller
`github@tafk.dev`; commits used command-local Git configuration, leaving global
and repository identity settings untouched.

Live validation uncovered two native prerequisites now documented and tested:

1. sbx 0.42.1 `env exec` requires argument flags before positional file paths.
   The examples use separate Bash arrays in that order for every command.
2. Selecting a native agent does not install its executable into FINN's generic
   template. The guide demonstrates user-owned installation via native `env exec`
   and a sandbox-local symlink onto the existing PATH. No custom agent definition,
   dependency-install kit, image dependency change or guest initialization redesign
   was introduced. Site-owned preinstalled templates remain another option.

Early validation failures (missing host NumPy, an unsuitable initial Docker test
invocation, native argument order, missing agent executable and PATH) were resolved
before the successful runs above. NumPy and Black were installed only in the
existing temporary host test virtual environment.

### Unverified behavior

Authenticated agent inference was not exercised; no user credentials were
registered or changed. The agent checks verified installation/version, native
selection and detached launch/reuse. Real FPGA tools, hardware and licence
checkout were not tested. No SIF was exported/executed on this host because its
runtime is absent. Jenkins execution and the complete accelerator-image build
matrix were not run; focused CI transport/configuration and Bake checks passed.
