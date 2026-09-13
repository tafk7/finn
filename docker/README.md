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
./docker/run --sbx --name agent-1 -- pytest -m util
```

With no command, `docker/run` opens an interactive shell. Use `--print` to see
the normalized request without building or running anything. `-n NAME` and
`--name NAME` assign the Docker container name or the persistent sbx sandbox
name.

## Prepare artifacts

```bash
./docker/build
./docker/build --runtime xrt
./docker/build --sbx
./docker/build --export-sif ./finn.sif
./docker/build --runtime xrt --export-sif ./finn-xrt.sif
```

Docker and sbx preparation is automatic when a run needs it. `--no-build`
requires an existing Docker image or sbx template, while `--rebuild` rebuilds
without using the BuildKit cache. Ordinary runs and SIF export reuse a prepared
Docker image with the selected environment tag; explicit `docker/build` refreshes
its cached build. SIF export always writes the requested path.

No registry is assumed. Docker images remain in the local daemon, sbx images
are transferred to the sbx image store, and SIF files are explicit build
outputs. Run a SIF with the standard tool, for example:

```bash
apptainer exec --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" \
  --env FINN_ROOT="$PWD" ./finn.sif python -c 'import finn'
```

Image tags contain an ``env-<hash>`` revision derived from the declared image
inputs in ``docker/image-inputs.txt``. Mounted FINN source is not an image
input: its commit, description and dirty state are passed separately as
``FINN_SOURCE_*`` runtime provenance. The immutable identity of a concrete
build remains its Docker image digest or exported SIF checksum.

## Configuration

`docker/config.py` is the shared host resolver. The `docker/config` command can
inspect its result or render Docker and sbx configuration:

```bash
./docker/config inspect --tier dev
./docker/config inspect --tier build --sbx
./docker/config compose --tier build --service build
./docker/config sbx --tier build --output-dir /path/outside/workspace/finn-sbx
```

## Native sandbox environments

`./docker/run --sbx` is the public FINN sandbox entry point. It prepares a local
FINN template and writes a native environment bundle outside the mounted checkout.
Docker and sbx execution are functions in the same launcher. No workspace controller is required.

For direct sbx use, prepare the template, then render a bundle:

```bash
./docker/build --sbx
export FINN_SBX_TEMPLATE=$(./docker/build --sbx --print-tag)
export FINN_SBX_NAME=finn-native-dev
./docker/config sbx --tier dev --output-dir "$HOME/.local/state/finn/native-dev"
sbx env plan "$HOME/.local/state/finn/native-dev/finn.sbxenv.yaml"
sbx env create "$HOME/.local/state/finn/native-dev/finn.sbxenv.yaml"
sbx env exec "$HOME/.local/state/finn/native-dev/finn.sbxenv.yaml" -- python -c 'import finn'
sbx env run "$HOME/.local/state/finn/native-dev/finn.sbxenv.yaml"
sbx env rm "$HOME/.local/state/finn/native-dev/finn.sbxenv.yaml" --force
```

First use requires native sbx approval. For a noninteractive job, review
`sbx env plan` and explicitly use `sbx env create --auto-approve` before invoking
the FINN runner. The runner does not approve native changes automatically.

The renderer prints the environment as JSON (valid YAML) and writes the same file
plus local v2 kits into the output directory. It specializes the checked-in
`docker/sbx/sbxenv.yaml`, which is also directly usable with native environment
arguments (`name`, `workspace`, `template`, and optional `agent`). For example,
after preparing the template and setting `FINN_SBX_TEMPLATE` above:

```bash
sbx env run ./docker/sbx/sbxenv.yaml \
  --env-arg name=finn-native-dev --env-arg workspace="$PWD" \
  --env-arg template="$FINN_SBX_TEMPLATE"
```

Use the same file and arguments for subsequent native commands.
The native environment supplies FINN defaults; an optional generated licence kit supplies site-specific network
permissions. The dev tier omits toolchain mounts, licence configuration and FINN
network grants. Existing machine/organization policy and agent kits still determine
effective connectivity; absence of grants does not establish a closed network.

Use sbx **0.42.1 or later**; 0.42.1 is the tested integration baseline. For FPGA
work, set the existing host configuration variables and render `--tier build`.
FLEXlm grants cover the server host unless the vendor-daemon port is pinned,
in which case the kit requests the two required ports. Validate a real licence
checkout at your site; `lmstat` only verifies the licence-manager connection.

Remove and recreate a sandbox after changing its template, mounts or kit
permissions. Editing the rendered files does not revoke grants on an existing
sandbox. Keep site paths, licence settings, credentials, personal agent overlays,
and external controller configuration outside the FINN repository. Shared writable
agent skills are disabled by default. See Docker's
[native environments](https://docs.docker.com/ai/sandboxes/configuration/environment-files/)
and [kit reference](https://docs.docker.com/ai/sandboxes/customize/kit-reference/).

`compose.yaml` and `docker-bake.hcl` remain usable directly for debugging and
advanced workflows. A direct Bake invocation must set
`FINN_IMAGE_REVISION` explicitly; the supported `docker/build` path computes it
from `docker/image-inputs.txt` automatically.

## Implementation boundaries

```text
docker/build + Docker/sbx/SIF consumers
    -> docker/lib.sh: finn_prepare_image
        -> docker-bake.hcl -> docker/Dockerfile.finn

docker/config -> config.py                  host only
    -> Compose overrides / native sbx bundles

image                                      guest only
    -> Dockerfile tool list + toolchain-shim
    -> finn_entrypoint.sh / finn-bashenv.sh / finn-toolchain.sh
    -> finn-live.pth / finn_paths.py

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
| `docker/run-sbx`, `docker/finn-sbx` | `docker/run --sbx -- COMMAND` |
| `run-docker.sh` | Explicit CI image preparation, then `docker/run` |
| `docker/export-sif PATH` | `docker/build --export-sif PATH` |
| Internal `print-tag sbx-dev` | `docker/build --sbx --print-tag` |
| `test` / `quicktest` launch shortcuts | Explicit `pytest` / `quicktest.sh` commands |
| `build_custom` shortcut | Explicit directory mount, working directory, and Python command |
| `build-xrt` grant/image spelling | `--fpga --runtime xrt` |
| Bake target `finn-xrt-slash` | `finn-slash-xrt` |

The Python configuration interface remains unchanged for this pass:
`docker/config` and its `docker/finn-env` alias are retained.
