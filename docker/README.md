# FINN Docker-built environment

FINN has two setup paths: native installation through `setup-local.sh`, or the
Docker-built reference environment described here.

The image is always built with Docker Buildx. It can then execute through three
backends:

```text
Docker image
├── Docker Compose              default
├── sbx template import         agent isolation
└── Apptainer SIF conversion    HPC execution
```

## Run

```bash
./docker/run -- quicktest.sh
./docker/run --name finn-test -- pytest -m util
./docker/run --fpga -- vivado -version
./docker/run --fpga --runtime xrt -- build_dataflow project/
./docker/run --backend sbx -- pytest -m util
./docker/run --backend sbx --name agent-1 -- pytest -m util
./docker/run --backend apptainer -- pytest -m util
```

With no command, `docker/run` opens an interactive shell. Use `--print` to see
the normalized request without building or running anything. `-n NAME` and
`--name NAME` assign the Docker container name or the persistent sbx sandbox
name. Apptainer does not have an equivalent named-instance object.

## Prepare artifacts

```bash
./docker/build
./docker/build --runtime xrt
./docker/build --backend sbx
./docker/build --backend apptainer
```

Preparation is automatic when an artifact is absent. `--no-build` requires an
existing artifact, while `--rebuild` replaces it without using the BuildKit
cache.

No registry is assumed. Docker images remain in the local daemon, sbx images
are transferred to the sbx image store, and Apptainer images are cached as SIF
files under the user's cache directory.

## Configuration

`docker/config.py` is the shared host resolver. The `docker/config` command can
inspect its result or render Docker and sbx configuration:

```bash
./docker/config inspect --tier dev
./docker/config compose --tier build --service build
./docker/config sbx --tier build
```

`compose.yaml` and `docker-bake.hcl` remain usable directly for debugging and
advanced workflows.

## Compatibility

The following historical commands remain available:

```text
docker/finn-env
docker/finn-sbx
docker/finn-apptainer
```

New workflows should use `docker/run`, `docker/build`, and `docker/config`.
