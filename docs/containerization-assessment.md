# FINN containerization assessment

Date: 2026-08-28

## Executive assessment

The image architecture is mostly sound; the orchestration architecture is not.

FINN should keep the three image tiers, profile-specific dependency sets, baked
dependency wheels, runtime identity handling, and thin entrypoint. It should
stop treating `run-docker.sh` as the authoritative build, runtime, CI, and
sandbox specification. The launcher has accumulated too many unrelated
responsibilities, so requirements are neither readily discoverable nor fully
consistent between Docker and sbx.

The target structure should be:

- `docker/Dockerfile.finn` for generic image contents.
- `docker-bake.hcl` for the build matrix, arguments, tags, and CI outputs.
- `compose.yaml` for ordinary human Docker workflows.
- `.devcontainer/devcontainer.json` for generic development and agent tooling.
- `docker/finn.kit/spec.yaml` as an sbx adapter, not the primary contract.
- A much smaller `run-docker.sh` retained temporarily as a compatibility
  wrapper.

## What is already right

The `dev -> build -> build-xrt` split is the correct capability boundary.
`dev` needs only the Python environment; `build` adds HLS headers and board
data; `build-xrt` adds XRT and V80 tooling. The repository correctly established
that RTL simulation needs Vivado/XSim, not XRT. That is more useful than a vague
small/full split.

Other good decisions include:

- Profiles bind Ubuntu, Python, XRT, and Python constraints into coherent
  configurations rather than letting them drift independently.
- `deps.env` is a shared source of dependency revisions for both host checkouts
  and image builds.
- Dependencies are installed into the image, so container startup is offline
  and does not run `pip install`.
- The live-source shadowing in `docker/finn_paths.py` is unusual, but it is a
  pragmatic response to arbitrary workspace locations and frequent qonnx and
  brevitas co-development. It should remain until FINN's resource and path
  packaging is redesigned.
- `finn_xsi` is built on demand into the build directory rather than modifying
  the checkout.
- The `.dockerignore` allowlist is the right approach for this repository.
- The XRT checksum and OS/architecture consistency checks are valuable
  supply-chain and correctness gates.
- Host UID/GID is runtime state. Removing per-developer identities from image
  tags was correct.
- There is no genuine need for `--privileged`, host devices, Docker socket
  access, or a GPU for compilation. XRT in this image does not imply direct
  access to an FPGA device.

## Genuine constraints versus incidental history

| Item | Assessment |
| --- | --- |
| Writable source workspace | Genuine for development, because FINN itself is not baked into the image. |
| `FINN_ROOT` | Currently genuine: many Python, C++, and Tcl paths resolve repository resources through it. |
| Workspace at the identical host path | Not a general FINN requirement. It is required by sbx's no-remapping behavior and useful when generated projects are opened on the host or dispatched to an NFS-visible compute farm. Ordinary Docker, CI, and Dev Containers can use a fixed `/workspace/finn`. |
| Arbitrary workspace-path support | Genuine while sbx is supported. This justifies retaining `finn_paths.py` as a compatibility seam. |
| Xilinx installation mounted from the host | Genuine for Vivado/Vitis flows: it is huge, proprietary, versioned, and impractical to bake. It should be read-only. |
| Platform repository mount | Genuine only for Vitis/Alveo flows. |
| License file or license-server reach | Genuine only for licensed tool flows. A floating license needs access to the license host and its vendor-daemon port, so in practice this is a host-level egress grant. |
| Persistent host build directory | Useful for CI, debugging, and artifact retention, but not required for the default agent environment. Ephemeral `/tmp` is appropriate there. |
| Dependency downloads during container launch | Incidental. The image already carries the pinned wheels. |
| Automatic SSH-key mount | Optional deployment functionality, not part of the base development contract. |
| Defaulting to `build-xrt` | Historical CI compatibility, not the right human or agent default. |
| `FINN_DOCKER_EXTRA` | A compatibility escape hatch, not a discoverable contract. |
| `linux/amd64` | A genuine current limitation for Xilinx-backed tiers; parts of the image also contain explicit x86-64 assumptions. It should be declared in the build and runtime manifests. |

The important workspace distinction is that FINN needs a correct in-container
`FINN_ROOT`; it does not intrinsically need that path to equal the host path.
The current Docker launcher mirrors it because that supports host-side
Vivado/GUI and LSF workflows, while sbx forces it. A default Compose or Dev
Container environment should use a fixed path, with an explicit host-path
bridge profile for external-tool workflows.

## Docker and sbx are not currently equivalent

They share an image, but they have meaningfully different runtime contracts:

- Docker creates a disposable container, mounts a persistent host build
  directory, runs as the host UID/GID, mounts SSH keys, supports cache and
  dataset mounts, and uses Docker's broad default networking.
- sbx creates or reuses a persistent microVM, imports the image into a separate
  template store, uses its `agent` identity model, keeps the build directory
  inside the VM, and applies license-host egress after creation.
- Docker invokes the image entrypoint for each new container. sbx invokes it
  for PID 1, but subsequent `sbx exec` commands bypass it.
- Ordinary `docker exec` also bypasses the entrypoint. The documentation
  recommends `docker exec` for another terminal, so this is not merely
  theoretical.

The entrypoint now does reasonable runtime work: it repairs `HOME` and user
variables, derives workspace and build paths, sources mounted Xilinx settings,
applies the FLEXlm workaround, and seeds Tcl files. It no longer installs
software or builds into the source tree, which is good.

But not every consumer runs it. `finn_paths.py` repairs Python execution after
an entrypoint bypass, and the kit reproduces enough environment for sbx. A
generic long-running agent container followed by `docker exec`, however, does
not get the dynamically sourced Xilinx environment.

Essential correctness therefore should not depend solely on entrypoint exports.
For the FPGA tiers, FINN should provide an idempotent `finn-env exec <command>`
or tool-shim mechanism that sources the mounted toolchain for each invocation.
The entrypoint can call it, but agents and CI can call it directly as well.

There are also concrete Docker/sbx parity problems:

- Docker mounts the entire Xilinx root read-write, despite comments calling it
  read-only. It should be read-only, with explicit writable cache and state
  mounts.
- For the default 2022.2 layout, Docker derives paths such as
  `$ROOT/Vivado/2022.2`, but sbx only attempts to mount `$ROOT/2022.2`. That
  appears structurally wrong unless a site happens to provide an extra umbrella
  directory or symlink.
- sbx passes `PLATFORM_REPO_PATHS` as environment but does not mount that
  directory.
- Docker mounts SSH keys, caches, ImageNet, and persistent scratch; sbx does
  not. Some differences are intentional, but they should be explicit profiles
  rather than consequences of two separate code paths.

## Recommended artifact structure

| Artifact | Role |
| --- | --- |
| `docker/Dockerfile.finn` | Generic FINN image stages only: `dev`, `build`, and `build-xrt`. Keep the current layering. |
| `docker-bake.hcl` | Authoritative build matrix for tier x profile, build arguments, tags, OCI labels, cache, registry or OCI output, and platform. Replaces the build and tag computation in `run-docker.sh`. |
| `compose.yaml` | Human-facing runtime services such as `dev`, `fpga`, `vitis`, and `notebook`. Declares mounts, environment, working directory, ports, user, and read-only toolchain access. |
| `.devcontainer/devcontainer.json` | Portable default agent/developer contract referencing the Compose `dev` service. Declares the workspace, non-root user, writable home, platform, and host resource guidance. |
| Optional `.devcontainer/fpga/devcontainer.json` or `compose.fpga.yaml` | Explicit capability escalation for the toolchain, platform repository, license, and persistent build output. |
| `docker/Dockerfile.sbx` plus `docker/finn.kit/spec.yaml` | Thin sbx-specific adaptation adding the reverse-engineered `agent` home, sudo, NPM, and startup behavior. It should not live in every generic image. |
| `run-docker.sh` | Temporary compatibility wrapper translating old modes to Compose and Bake. It should no longer compute or own the contract. |
| `docs/containerization.md` | An architecture decision record explaining decisions and measurements, not the live executable specification. |

A normal human workflow would then use recognizable commands:

```bash
docker compose build dev
docker compose run --rm dev
docker compose run --rm dev quicktest.sh
docker compose up notebook
```

The default should be `dev`. CI should explicitly request `build` or
`build-xrt`; Jenkins history should not dictate the developer default.

The Dev Container Specification is the best established contract for generic
agent tooling. It covers the image or build target, workspace, mounts,
environment, user, and host requirements. There is no universal OCI convention
for egress allowlists, so FINN should not invent image labels that pretend to
solve this. Instead:

- Make the portable `dev` environment require no FINN runtime egress, secrets,
  privileges, or devices.
- Represent licensed FPGA work as a separate, explicit capability profile.
- Keep network-policy translation in backend adapters such as the sbx kit.

## Launcher changes

The launcher currently fetches dependency repositories before almost every
invocation, even though the image already contains the pinned Python wheels and
the build tiers contain their own HLS and board data. That should become an
explicit dependency-development command, not container startup behavior. CI
and generic agents should normally use `FINN_DEPS=frozen`; developers editing
qonnx or brevitas can explicitly synchronize the live checkouts.

Other launcher changes should include:

- Stop mounting `ssh_keys` automatically. Make SSH agent forwarding or a
  read-only secret mount opt-in.
- Replace `FINN_DOCKER_EXTRA` in documented workflows with Compose override
  files. Retain it only as a legacy or site escape hatch.
- Remove unconditional TTY allocation from automated runs.
- Use the image's `tini`; do not also add Docker `--init`.
- Put the sbx-specific user, NOPASSWD sudo, populated home, and NPM prefix in an
  sbx adapter image. The generic image should have a safe non-root default user,
  while Compose may override it to the host UID/GID.

## CI changes

CI currently builds by asking `run-docker.sh` to launch an `echo`, deletes the
machine-wide image cache, saves the resulting tagged image to NFS, and has
shards reload it. It should instead:

- Build with `docker buildx bake`.
- Publish to an internal registry by immutable digest, or produce an OCI
  archive plus its digest if NFS transport is mandatory.
- Pass the digest, rather than a recomputed mutable tag, to test shards.
- Run tests through a non-interactive Compose CI service or direct
  `docker run`.
- Add an `imageTier` or capability field to the CI stage matrix: Python-only
  tests use `dev`; Vivado/HLS tests use `build`; Vitis/Alveo tests use
  `build-xrt`.
- Test both lifecycle forms: a fresh `docker run` and a long-running container
  followed by `docker exec`. The latter is currently the missing contract test.

## Documentation and conformance

`docs/containerization.md` is a useful engineering record but has already
drifted from the implementation. Examples include:

- Its sbx schema discussion conflicts with the checked-in schema.
- It says the entrypoint writes `sandbox-persistent.sh`, while the entrypoint
  explicitly says it does not.
- Its license-network descriptions disagree about whether an open network
  posture is necessary.

The document should be split into stable architecture decision records and
short user documentation derived from the actual Compose, Dev Container, and
Bake contracts.

Conformance tests should validate:

1. Every advertised tier and supported profile builds.
2. The default `dev` environment runs with only the workspace mounted.
3. A fresh `docker run` works.
4. Commands executed later through `docker exec` work.
5. A plain sbx create plus sbx exec works without the launcher.
6. Xilinx and platform mounts are present and read-only in FPGA profiles.
7. The default agent profile has no FINN-specific network, secret, device, or
   privilege requirement.

## Conclusion

FINN should preserve the current image engineering, make `dev` the universal
safe default, express runtime needs in standard manifests, isolate licensed
FPGA and sbx behavior as explicit adapters, and reduce `run-docker.sh` from the
architecture to a compatibility convenience.
