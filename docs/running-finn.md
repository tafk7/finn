# Running FINN

FINN runs in three ways. Each way has a different purpose. Select the way that
agrees with your task. No way is better than the others.

| | Way | Use it for | Isolation |
|---|---|---|---|
| **1** | **Docker container** | Development by a person | None. This is a development environment, not a security boundary. |
| **2** | **sbx sandbox** | Development by an autonomous agent | A microVM with its own kernel and a network policy |
| **3** | **Host system** | Development with no container | None |

The three ways use the same dependency versions, the same image tiers and the
same toolchain resolver. A result in one way is therefore correct in the others.

## What FINN gives you, and what you give FINN

FINN supplies the **software environment**: Python, the dependency set, the HLS
headers, the board files, and XRT on the largest tier. FINN does **not** supply
Vivado, Vitis or Vitis HLS.

This division is practical. Engineers use different tool versions. Some install
the tools on a local disk, others on network storage. AMD also has distribution
conditions for the tools. FINN therefore uses your installation, and this
document tells you how to connect it.

You supply:

| Item | Variable | Necessary for |
|---|---|---|
| Xilinx installation directory | `FINN_XILINX_PATH` | `build`, `build-xrt` |
| Xilinx version | `FINN_XILINX_VERSION` | `build`, `build-xrt` |
| A licence | `XILINXD_LICENSE_FILE` | All tool operations |
| Vitis platforms | `PLATFORM_REPO_PATHS` | `build-xrt`, Alveo only |

FINN mounts the installation **read-only**. The tools do not write into their
own installation. A real `synth_design` operation shows this. An agent with
write access to a 267 GB installation is the largest unnecessary risk in this
setup.

The two licence types both operate:

- `XILINXD_LICENSE_FILE=2100@licsrv.example` is a floating server. FINN mounts
  nothing. FINN permits network access to that **host**.

  FINN does not use `host:port`. FLEXlm gives a second port for the vendor
  daemon after the first connection, and that port usually changes. A rule with
  a port lets `lmutil lmstat` operate, but each subsequent licence request
  fails with an incorrect message.

- `XILINXD_LICENSE_FILE=/path/to/Xilinx.lic` is node-locked. FINN mounts the
  parent directory read-only, at the same path.

> **Node-locked licences in way 2 are not verified.** FLEXlm connects a
> node-locked licence to an Ethernet host ID. A sandbox does not show the host
> ID of the machine. The `docker run --mac-address` option exists, but sbx has
> no equivalent. Conformance test 9 examines this. The test has not been run.
> Do not assume a result.

## Tiers

The three tiers are the same in all three ways.

| Tier | Adds | Necessary from the host |
|---|---|---|
| `dev` | Python, FINN and its dependencies | **Nothing** but the repository |
| `build` | HLS headers, board files | Xilinx installation, licence |
| `build-xrt` | XRT, V80 support | Also the platform repository |

The `dev` tier is defined by what it does not have. It has no toolchain mount,
no licence, no secrets and no FINN network access. An agent or a new
contributor can therefore use it with no configuration. If you must add a host
mount to `dev`, use `build` instead.

RTL simulation uses the `xsim` tool of Vivado, not XRT. Use `build` for RTL
simulation.

---

## Way 1 — Docker, for development by a person

```bash
docker compose run --rm dev                      # a shell
docker compose run --rm dev quicktest.sh         # the fast tests
docker compose --profile fpga run --rm build     # Vivado and Vitis HLS
docker compose --profile notebook up             # Jupyter
```

The `dev` tier needs no configuration. For the tiers that use the Xilinx tools,
make the host settings one time:

```bash
export FINN_XILINX_PATH=/opt/Xilinx FINN_XILINX_VERSION=2025.2
./docker/finn-env inspect --tier build --format sh > .env
```

The `.env` file is a cache. One program writes it. Do not edit it. If it is not
correct, delete it and make it again.

### `run-docker.sh`, the older command

```bash
./run-docker.sh quicktest
FINN_DOCKER_TARGET=build-xrt ./run-docker.sh bash
```

This command continues to operate. It accepts all the variables that it always
accepted. It now translates its arguments into the commands above, and it holds
no configuration. Use Docker Compose for new work. You do not have to change
your scripts or your habits.

Three defaults are different from earlier versions:

- The default tier is `build`, not `build-xrt`. Vivado and Vitis HLS are
  available. XRT and the platform repository are not. If `FINN_XILINX_PATH` is
  not set, the command gives a warning and uses `dev`. An explicit
  `FINN_DOCKER_TARGET=build` still stops with an error, because that is a
  request and not a default. Docker Compose and Bake use `dev`, which is
  correct for a new contributor or an agent.
- Under `run-docker.sh`, the `dev` tier mounts the workspace at its host path.
  Under Docker Compose, it mounts the workspace at `/workspace/finn`.
- `FINN_DEPS` is `frozen`.

### This way is not a security boundary

The container uses the kernel of the host and has full network access. An
autonomous agent in the container can reach everything that your machine can
reach. The read-only toolchain mount and the small `dev` tier prevent
**accidents**. They do not stop an attacker.

Use way 2 for agent work.

---

## Way 2 — sbx, for development by an autonomous agent

```bash
docker/finn-sbx dev                      # only the repository
docker/finn-sbx build                    # also the toolchain (ro) and licence access
docker/finn-sbx build -- pytest -m util  # one command
docker/finn-sbx rm build                 # remove the sandbox
```

This way has a true boundary: a microVM with its own kernel. Network access is
denied, with small permissions for each sandbox. The `dev` tier has no
toolchain, no licence and no network permission. This is not because the
variables are empty. It is because the tier does not read the file that adds
them.

You must have sbx 0.39.0 or later, and you must be signed in.

### How it is assembled

`docker/sbxenv/base.sbxenv.yaml` declares what a FINN sandbox is.
`fpga.sbxenv.yaml` adds the toolchain for the larger tiers. The `sbx env run`
command creates the sandbox, or connects to it if it exists.

`docker/finn-sbx` does only the three tasks that the file format cannot do:

1. Put a locally built image into the image store of sbx.
2. Write the environment files to a directory outside all mounted workspaces.
3. Permit network access to the licence server.

Task 2 is necessary for safety. With a direct mount, the agent can write each
file that it can read. An environment file in the workspace is therefore a file
that the agent can change to make its next sandbox larger.

---

## Way 3 — host system, for development with no container

```bash
./setup-local.sh
```

This command makes a virtual environment, gets the dependencies, installs FINN,
finds your Xilinx toolchain and builds `finn_xsi`.

To set up the toolchain in a shell that you already have:

```bash
eval "$(./docker/finn-env print --format sh)"
```

`finn-env` is not only for containers. It finds a toolchain and sets it up
wherever it runs. This is why all three ways use it.

The shared code is necessary. `setup-local.sh` had its own copy of this logic.
That copy used the Xilinx directory structure from before 2024.2. It therefore
reported "Vivado not found" on each recent installation.

This way has no isolation and the least test coverage. Conformance test 10
examines the toolchain resolution. The other parts have no tests.

---

## Necessary tools

| Tool | Necessary for | Source |
|---|---|---|
| Docker Engine | Ways 1 and 2 | Any installation |
| **`docker buildx`** | All image builds | Docker Desktop, or the Docker packages |
| **`docker compose`** | Way 1 | The same |
| Python 3 | `finn-env`, all ways | Each Linux distribution |
| `sbx` 0.39.0 or later | Way 2 only | A separate installation |
| Xilinx tools | The `build` tiers only | Your own installation |

Buildx and Compose are CLI plugins. They are not part of the engine. Docker
Desktop and the Docker packages contain them. The distribution package
`docker.io` does not contain them. If you installed Docker with
`apt install docker.io`, do this:

```bash
sudo apt install docker-buildx-plugin docker-compose-plugin
```

Tested with docker 29.6.1, buildx 0.35.0, compose 5.3.1 and sbx 0.39.0.

## Where each part is

| Subject | File |
|---|---|
| Which images exist, their names and their build inputs | `docker-bake.hcl` |
| Your toolchain location, licence type, mounts and network access | `docker/finn-env` |
| Way 1 runtime | `compose.yaml` |
| Way 2 runtime | `docker/sbxenv/*.sbxenv.yaml` |
| Way 3 setup | `setup-local.sh` |
| Dependency versions | `deps.env` |
| The older command for way 1 | `run-docker.sh` |

`finn-env` finds each host fact one time only. This is a correctness
requirement, not tidiness. The defects that this structure replaced were all
examples of two programs that found the same fact and then disagreed.

For the reasons behind the design, read `containerization-decisions.md`. For
the mechanisms, read `containerization.md`.
