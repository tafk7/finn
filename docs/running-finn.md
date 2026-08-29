# Running FINN

FINN supports three ways to run, for three different purposes. Pick the one
that matches what you are doing; they are not ranked, and none is a fallback
for another.

| | Lane | Use it for | Isolation |
|---|---|---|---|
| **1** | **Docker container** | setup-and-go human development | none — a dev environment, not a boundary |
| **2** | **sbx sandbox** | autonomous agent development | microVM + egress policy |
| **3** | **Bare host** | unstructured development | none |

All three share the same dependency pins, the same tier definitions, and the
same toolchain resolver, so a result in one lane means the same thing in the
others.

## What FINN provides, and what you provide

FINN ships the **software environment**: Python, the dependency closure, HLS
headers, board files, and on the largest tier XRT. It does **not** ship Vivado,
Vitis or Vitis HLS.

That split is practical rather than ideological. Engineers run different tool
versions, install to local disk or to network storage, and are bound by AMD's
distribution terms for the tools themselves. So FINN expects you to have your
own installation and tells you how to connect it.

You provide:

| | Set with | Needed by |
|---|---|---|
| Xilinx install location | `FINN_XILINX_PATH` | `build`, `build-xrt` |
| Xilinx version | `FINN_XILINX_VERSION` | `build`, `build-xrt` |
| A licence | `XILINXD_LICENSE_FILE` | anything that runs the tools |
| Vitis platforms | `PLATFORM_REPO_PATHS` | `build-xrt`, Alveo only |

The install is always mounted **read-only**. The tools do not write into their
own installation — verified with a real `synth_design` — and an agent with write
access to a 267 GB install nobody wants to rebuild is the largest avoidable risk
in this setup.

Both licence forms work:

- `XILINXD_LICENSE_FILE=2100@licsrv.example` — a floating server. Nothing is
  mounted; the sandbox is granted egress to that **host**. Not `host:port`:
  FLEXlm hands back a second, usually ephemeral, vendor-daemon port after the
  initial handshake, so a port-scoped rule lets `lmutil lmstat` succeed while
  every real checkout fails with a misleading error.
- `XILINXD_LICENSE_FILE=/path/to/Xilinx.lic` — node-locked. The containing
  directory is mounted read-only at its own path.

> **Node-locked licences in lane 2 are unverified.** FLEXlm binds a node-locked
> licence to an Ethernet host ID, and a sandbox does not present the host's.
> `docker run --mac-address` exists; sbx has no equivalent. Whether a
> node-locked checkout succeeds in a sandbox is tracked as conformance test 9
> and has not been run. Do not assume either answer.

## Tiers

The same three tiers exist in every lane.

| Tier | Adds | Needs from the host |
|---|---|---|
| `dev` | Python, FINN and its dependency closure | **nothing** but the repo |
| `build` | HLS headers, board files | Xilinx install, licence |
| `build-xrt` | XRT, V80 support | + platform repository |

`dev` is defined by what it lacks. It has no toolchain mount, no licence, no
secrets and no FINN-specific network egress, which is what lets an agent or a
new contributor run it with no configuration at all. If you find yourself adding
a host mount to `dev`, that is the signal to use `build` instead.

Note that RTL simulation needs Vivado's `xsim`, not XRT — so `build`, not
`build-xrt`.

---

## Lane 1 — Docker, for human development

```bash
docker compose run --rm dev                      # a shell
docker compose run --rm dev quicktest.sh         # the fast test suite
docker compose --profile fpga run --rm build     # Vivado / Vitis HLS
docker compose --profile notebook up             # Jupyter
```

`dev` needs no configuration. For the FPGA tiers, generate the host-specific
values once:

```bash
export FINN_XILINX_PATH=/opt/Xilinx FINN_XILINX_VERSION=2025.2
./docker/finn-env inspect --tier build --format sh > .env
```

`.env` is a cache with a single producer. Never hand-edit it; delete and
regenerate it if it looks stale.

### `run-docker.sh` — the legacy entry point

```bash
./run-docker.sh quicktest
FINN_DOCKER_TARGET=build ./run-docker.sh bash
```

Still supported, still works, and every environment variable it ever took still
works. It is now a translator onto the commands above and owns no configuration
of its own. New work should target Compose directly; existing scripts and muscle
memory need not change.

Three defaults differ from older versions of the script:

- the default tier is `dev`, not `build-xrt`
- `dev` mounts the workspace at its host path under `run-docker.sh`, and at
  `/workspace/finn` under plain Compose
- `FINN_DEPS` defaults to `frozen`

### This lane is not a security boundary

The container shares the host kernel and has unrestricted network access.
Running an autonomous agent in it means the agent can reach anything your
machine can reach. The read-only toolchain mount and the narrow `dev` tier are
there to prevent **accidents**, not to contain an adversary.

For agent work, use lane 2.

---

## Lane 2 — sbx, for autonomous agent development

```bash
docker/finn-sbx dev                      # narrow: the repo, nothing else
docker/finn-sbx build                    # + toolchain (ro) + licence egress
docker/finn-sbx build -- pytest -m util  # one command
docker/finn-sbx rm build                 # tear down
```

This is the lane with an actual boundary: a microVM with its own kernel, and
egress denied by default with narrow per-sandbox allowances. `dev` gets no
toolchain, no licence and no egress grant at all — not because the variables
happen to be unset, but because the tier does not load the overlay that adds
them.

Requires `sbx` ≥ 0.39.0, signed in.

### How it is put together

`docker/sbxenv/base.sbxenv.yaml` declares what a FINN sandbox is; `fpga.sbxenv.yaml`
is the capability escalation layered over it for the toolchain tiers.
`sbx env run` handles create-or-attach. `docker/finn-sbx` does only the three
things that file format cannot express: load a locally built template into sbx's
image store, materialise the environment files outside every mounted workspace,
and grant licence-server egress.

The environment files are materialised outside the workspace deliberately. With
a direct mount the agent can write every file it can see, so an environment file
inside the workspace is a file the agent could edit to widen its own next
sandbox.

---

## Lane 3 — bare host, for unstructured development

```bash
./setup-local.sh
```

Creates a virtualenv, fetches dependencies, installs FINN editable, resolves
your Xilinx toolchain and builds `finn_xsi`.

To configure the toolchain in an existing shell:

```bash
eval "$(./docker/finn-env print --format sh)"
```

`finn-env` is not container-specific — it resolves and sources a toolchain
wherever it runs, which is why all three lanes share it. That sharing is not
cosmetic: `setup-local.sh` previously had its own copy of the layout logic,
hardcoded to the pre-2024.2 Xilinx directory structure, and reported "Vivado not
found" on every recent install.

No isolation, and the least CI coverage of the three. Conformance test 10 covers
the toolchain resolution; the rest is best-effort.

---

## Requirements

| Tool | Needed for | Where it comes from |
|---|---|---|
| Docker Engine | lanes 1 and 2 | any installation |
| **`docker buildx`** | every image build | Docker Desktop, or Docker's own apt/yum packages |
| **`docker compose`** | lane 1 | same |
| Python 3 | `finn-env`, all lanes | every Linux distribution |
| `sbx` ≥ 0.39.0 | lane 2 only | separate install |
| Xilinx tools | `build` tiers only | your own installation |

**buildx and compose are CLI plugins, not part of the engine.** Docker Desktop
and Docker's own packages include them; the distribution `docker.io` package
does **not**. If you installed with `apt install docker.io`:

```bash
sudo apt install docker-buildx-plugin docker-compose-plugin
```

Tested against: docker 29.6.1, buildx 0.35.0, compose 5.3.1, sbx 0.39.0.

## Where the pieces live

| Concern | Owner |
|---|---|
| What images exist, their names, their build inputs | `docker-bake.hcl` |
| Where your toolchain is, licence form, mounts, egress | `docker/finn-env` |
| Lane 1 runtime | `compose.yaml` |
| Lane 2 runtime | `docker/sbxenv/*.sbxenv.yaml` |
| Lane 3 setup | `setup-local.sh` |
| Dependency pins | `deps.env` |
| Legacy lane 1 entry point | `run-docker.sh` |

Every host fact is derived in exactly one place, `finn-env`. That is a
correctness property rather than tidiness: the defects this structure replaced
were all cases of two code paths computing the same fact and drifting apart.

For the design reasoning see `docs/containerization-decisions.md`; for the
mechanisms see `docs/containerization.md`.
