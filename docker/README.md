# FINN Docker-built environment

FINN has two setup paths: native installation through `setup-local.sh`, or the
Docker-built reference environment described here.

The image is always built with Docker Buildx. Docker runs it directly, sbx
imports a specialized image variant, and Apptainer consumes an exported SIF:

```text
Docker image
├── Docker Compose              default
├── sbx template import         agent isolation
└── SIF export                  standard Apptainer/Singularity execution
```

## Run

```bash
./docker/run -- quicktest.sh
./docker/run --name finn-test -- pytest -m util
./docker/run --fpga -- vivado -version
./docker/run --fpga --runtime xrt -- build_dataflow project/
```

With no command, `docker/run` opens an interactive shell. Use `--print` to see
the normalized request without building or running anything. `-n NAME` and
`--name NAME` assign the Docker container name.

## Prepare artifacts

```bash
./docker/build
./docker/build --runtime xrt
./docker/build --sbx
./docker/build --export-sif ./finn.sif
./docker/build --runtime xrt --export-sif ./finn-xrt.sif
```

Docker preparation is automatic when a run needs it. `--no-build`
requires an existing Docker image, while `--rebuild` rebuilds
without using the BuildKit cache. Ordinary runs and SIF export reuse a prepared
Docker image with the selected environment tag; explicit `docker/build` refreshes
its cached build. SIF export always writes the requested path.

No registry is assumed. Docker images remain in the local daemon, sbx images
are transferred to the sbx image store, and SIF files are explicit build
outputs. Run a SIF with the standard tool, for example:

```bash
apptainer exec --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" \
  ./finn.sif python -c 'from finn.util.basic import fifo_rtl_files; print(fifo_rtl_files())'
```

Image tags contain an ``env-<hash>`` revision derived from the declared image
inputs in ``docker/image-inputs.txt``, including the FINN code and resources
installed in the application. The mounted checkout's commit, description and dirty
state are passed separately as
``FINN_SOURCE_*`` runtime provenance. The immutable identity of a concrete
build remains its Docker image digest or exported SIF checksum.

## Configuration

`docker/config.py` is the single executable Python host resolver for Docker and
native installation. It preserves path/layout probing, licence classification,
UID/GID handling, shell output and Compose output:

```bash
./docker/config.py inspect --tier dev
./docker/config.py inspect --tier build
./docker/config.py compose --tier build --service build
```

Reported network requirements are declarative. Docker does not enforce them.

## Native sandbox environments

Prepare a template with `./docker/build --sbx`; obtain its reference with
`./docker/build --sbx --print-tag`. Preparation only builds/imports an image.
It does not create sandboxes, register credentials or configure machine policy.

Follow [the native example guide](sbx/README.md) to copy the base, selected FPGA
overlay and optional site kit together outside all mounted workspaces. Use
native `sbx env plan / create / run / exec / rm` with the same files and arguments
for every command. The base defaults to shell; `--env-arg agent=claude` selects
a coding agent and composes with the explicit FPGA overlay. The generic image
has no coding-agent executable; follow the guide's user-owned installation step
before attaching, or provide your own prepared template. Lists concatenate
under native composition. Personal agent settings and credentials remain in
user-owned native configuration.

FINN supplies images and examples. Users and sites own instantiated environments,
mounts and network policy; sbx owns composition, approval and lifecycle. FINN has
no Cardinal contract. The development example has no optional FPGA mounts or
site kit, and disables shared writable skills. Effective networking still depends
on machine/organization policy and the selected agent. Use a real licensed tool
operation to validate FPGA licensing; policy readback or `lmstat` is insufficient.

The examples were validated with client/server 0.42.1, which is a tested version,
not a claim about the latest release. Native environment and kit interfaces are
experimental; see the [completion record](../docs/native-sbx-final-shape-plan.md)
for documentation revision and validation limits.

`compose.yaml` and `docker-bake.hcl` remain usable directly for debugging and
advanced workflows. A direct Bake invocation must set
`FINN_IMAGE_REVISION` explicitly; the supported `docker/build` path computes it
from `docker/image-inputs.txt` automatically.

## Implementation boundaries

```text
docker/build + Docker/sbx/SIF consumers
    -> docker/lib.sh: finn_prepare_image
        -> docker-bake.hcl -> docker/Dockerfile.finn

docker/config.py                            host only
    -> shell assignments / Compose overrides

user-owned copies of docker/sbx examples
    -> native sbx env composition / approval / lifecycle

image                                      guest only
    -> Dockerfile tool list + toolchain-shim
    -> finn_entrypoint.sh / finn-bashenv.sh / finn-toolchain.sh
    -> installed FINN distribution and generated console entry points

Jenkins common helper -> shared-image loader -> docker/run
```

Host rendering code and sandbox defaults are not image inputs. Editing them
does not change image identity. The Dockerfile declares the tool list and sbx
contract inline. The image does not install the host resolver.

CI loads and verifies shared images before invoking `docker/run --no-build`.
Repository callers use the public interface; the old runtime scripts and Jenkins
bridge have been removed. Use `--fpga` instead of legacy grant-tier selection,
`--runtime xrt` to select XRT, and `docker/build --print-tag` for image references.
Commands after `--` are passed verbatim: use `pytest` and `quicktest.sh` explicitly.
For external dataflow directories, supply an explicit mount through
`FINN_DOCKER_EXTRA`; the launcher no longer interprets workload command names.

Python dependencies use FINN's `requirements.txt` plus `docker/requirements-dev.txt`.
`docker/pip-torch.txt` retains the CPU wheel source and pins; the shared
`docker/pip-constraints.txt` applies to all installation steps. Native installation
uses the same declarations. No packaging-format migration is required.

The small guest entrypoint provides writable home/scratch directories and optional
mounted Tcl initialization. Bash and bare vendor commands retain their respective
hooks because those paths bypass normal startup.

## Command migration

| Removed interface | Replacement |
| --- | --- |
| `docker/run-docker` | `docker/run -- COMMAND` |
| `docker/run-sbx`, `docker/finn-sbx`, `docker/run --sbx` | Copied native examples and `sbx env` |
| `run-docker.sh` | Explicit CI image preparation, then `docker/run` |
| `docker/export-sif PATH` | `docker/build --export-sif PATH` |
| Internal `print-tag sbx-dev` | `docker/build --sbx --print-tag` |
| `test` / `quicktest` launch shortcuts | Explicit `pytest` / `quicktest.sh` commands |
| `build_custom` shortcut | Explicit directory mount, working directory, and Python command |
| `build-xrt` grant/image spelling | `--fpga --runtime xrt` |
| Bake target `finn-xrt-slash` | `finn-slash-xrt` |

`docker/config` and `docker/finn-env` are removed; use `docker/config.py`.
The resolver's `sbx` subcommand and `inspect --sbx`, and the launcher's `--sbx`
and `--remove`, now fail as unknown interfaces.

Existing generated environment directories under
`${XDG_STATE_HOME:-$HOME/.local/state}/finn/sbxenv` are left intact, since they
may describe active sandboxes. Use their `finn.sbxenv.yaml` directly with native
`sbx env` commands, retaining any original overlays and arguments. For example:

```bash
sbx env plan /existing/environment/finn.sbxenv.yaml
sbx env exec /existing/environment/finn.sbxenv.yaml -- python -c 'import finn'
sbx env rm /existing/environment/finn.sbxenv.yaml --force
```

Alternatively copy the new examples into a user-owned directory. No automatic
state migration, sandbox removal or global policy/credential changes occur.

## Explicit development preparation

Images run installed FINN. A mounted checkout does not select imports, and
`--deps`/`FINN_DEPS` import modes have been removed. In a writable container or
native sbx session, prepare a venv once:

```bash
python -m venv --system-site-packages --without-pip /writable/finn-dev
/writable/finn-dev/bin/python -m pip install --use-pep517 --config-settings editable_mode=strict --no-deps --no-build-isolation -e /path/to/finn
# Optional explicit co-development:
/writable/finn-dev/bin/python -m pip install --use-pep517 --config-settings editable_mode=strict --no-deps --no-build-isolation -e /path/to/qonnx
```

Use that venv's executables for subsequent commands. The Dev Container prepares
`.venv` in its creation step. Read-only SIF execution uses installed code; editable
HPC work requires an explicitly writable environment. See
[installation and tool selection](../docs/installation.md) for offline preparation,
metadata inspection, resource lifetimes and remaining native loader requirements.
