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
without using the BuildKit cache. SIF export always writes the requested path.

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
./docker/config sbx --tier build
```

## Workspace-controller profile

FINN publishes a provider-neutral development profile under `execution/`:

```bash
./execution/plan --tier dev
sbxc plan --canonical --config execution/sbx-compose.toml
```

`execution/plan` is the versioned machine-readable interface for workspace
controllers. It combines the existing host resolver with FINN's environment
and source identities, names the preparation command, and points to the public
sbxc manifest and agent. Secret values are never emitted; only required secret
names may appear. The initial public profile is deliberately the portable dev
tier: frozen dependencies, no Xilinx toolchain, no licence, and no FINN
workload egress grants.

Prepare the local sbx image/template before first attachment:

```bash
./docker/build --sbx
```

The public sbxc profile carries no provider configuration. Company LLM
gateways, certificates, licence servers, concrete toolchain paths, and model
policy belong in private adapter overlays.

`compose.yaml` and `docker-bake.hcl` remain usable directly for debugging and
advanced workflows. A direct Bake invocation must set
`FINN_IMAGE_REVISION` explicitly; the supported `docker/build` path computes it
from `docker/image-inputs.txt` automatically.

## Compatibility

The following historical commands remain available:

```text
docker/finn-env
docker/finn-sbx
```

New workflows should use `docker/run`, `docker/build`, and `docker/config`.
