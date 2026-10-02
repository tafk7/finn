# FINN in Docker Sandboxes (sbx)

FINN ships two [kits](https://docs.docker.com/ai/sandboxes/customize/):

| Kit | Kind | Adds |
|---|---|---|
| `finn.yaml` (repository root) | workload | FINN's image, built from `docker/Dockerfile.finn` (stage `sbx`) with this checkout as context; the checkout installed editable at start. No coding agent |
| `docker/sbx/xilinx/` | mixin | `XILINX_VIVADO`/`XILINX_VITIS`/`XILINX_HLS` for a mounted installation, `XILINXD_LICENSE_FILE`, and network access to the licence server's two ports only |

Everything else is yours: which harness (Claude Code, Codex, …), your
instructions and skills, credentials, and this machine's paths. Add each as a
kit or a sandbox argument. Kits never mount host paths; mounts are arguments of
the sandbox.

Requires sbx 0.45 or later (v3 kits); validated with 0.46.0. sbx kits are
experimental and change between releases: check the
[release notes](https://docs.docker.com/ai/sandboxes/release-notes/) when upgrading.

## Prerequisite: a builder for kits

sbx builds kits from source with Docker BuildKit and needs its OCI exporter:
either Docker's containerd image store (the default on new Docker Engine 29
installations), or a `docker-container` builder selected with `BUILDX_BUILDER`:

```bash
docker buildx create --name sbx-kits --driver docker-container
export BUILDX_BUILDER=sbx-kits        # e.g. in your shell profile
```

Behind a corporate resolver, give that builder DNS servers it can reach
(`--buildkitd-config` with a `[dns] nameservers = [...]` section). The first
build of the FINN workload takes as long as `docker/build`; later sandboxes
reuse the builder's cache.

## Run FINN

From the checkout (the kit reference must name the directory, so use its path):

```bash
sbx create --name finn --skills off -m 96g \
    "$PWD" "$PWD"                              # the workload kit, then the workspace
sbx run --name finn                            # a shell in the sandbox
sbx exec finn -- python -m finn.util.installation
sbx rm --force finn
```

The sandbox gets every host CPU and half the host memory (at most 32 GiB)
unless `-m` says otherwise; the XSim sweep wants about 96 GiB.

When the sandbox starts, its startup hook installs the checkout editable into
`/opt/venv` and then writes `/tmp/finn-ready`, which scripts that `exec` right
after `create` can wait for. After pulling a change to `uv.lock`, run
`uv sync --inexact` in the sandbox or recreate it.

## Vivado and the licence server

Mount the installation read-only and add the xilinx kit with this machine's
values, kept in a file outside every repository:

```text
# ~/.config/finn-sbx/xilinx.args
vivado=/opt/Xilinx/2025.2/Vivado
vitis=/opt/Xilinx/2025.2/Vitis
hls=/opt/Xilinx/2025.2/Vitis
license_host=10.0.0.5
license_port=2100
vendor_port=2101
```

```bash
sbx create --name finn --skills off -m 96g \
    --kit "$PWD/docker/sbx/xilinx" --kit-args-file ~/.config/finn-sbx/xilinx.args \
    "$PWD" "$PWD" /opt/Xilinx:ro
```

Paths depend on the installation: older ones end in `Vivado/2024.2`, newer ones in
`2025.1/Vivado`. `XILINXD_LICENSE_FILE` is set in Bash (`BASH_ENV`), where FINN
runs its tools.

* **Use the server's IP address** for `license_host`. FlexLM connections are plain
  TCP, which sbx matches by address, and resolving a name needs DNS access,
  which a closed policy blocks. The kit refuses anything but an IPv4 address.
* **Finding the vendor port.** Unless the licence file pins it (`VENDOR xilinxd
  port=...`), the server chooses the xilinxd port. Create the sandbox with any
  value, run a licensed operation, and `sbx policy log SANDBOX` lists the blocked
  port; recreate with it.
* **Check** with `sbx policy check network --sandbox finn 10.0.0.5:2100` and a host
  that should be closed. Policy readback does not prove a checkout: validate with
  a licensed operation, such as synthesis for a Versal part.
* A node-locked licence may depend on a host ID the microVM does not have.
  Licence-file directories and platform repositories are further read-only
  mounts plus variables (`-e XILINXD_LICENSE_FILE=…`, `-e PLATFORM_REPO_PATHS=…`).

## FinnLib

`finn.kernels` and the MVAU/VVAU replay buffer compile against FinnLib. A
sandbox has no SSH credentials to fetch it, so mount your clone (writable) and
point FINN at it:

```bash
    -e FINN_RESOURCES_FINNLIB=$HOME/finnlib  …  "$PWD" "$PWD" $HOME/finnlib
```

An agent can edit and commit there; push from the host. The mount is your real
clone: keep uncommitted work of your own out of it while an agent runs.

## Coding agents

FINN's image contains no agent. Add a harness as a kit, for example Docker's
Claude Code or Codex mixins
([sandbox-kit-spec examples](https://github.com/docker/sandbox-kit-spec/tree/main/examples)),
and start it in the sandbox's shell:

```bash
sbx create … --kit <claude-mixin> "$PWD" "$PWD" …
sbx exec -it finn claude
```

The harness kit brings its own network rules and credential request; sbx
injects the credential through its proxy, so it never enters the sandbox
(`sbx secret set anthropic`, or a custom secret for a company gateway). Two
kits that provide the same tool are refused when the sandbox is created.

## Lanes: a private clone per sandbox

`--clone` gives the sandbox its own clone of the checkout, so its edits and
commits stay out of your working tree. Run it from a main clone (sbx refuses
`--clone` from a `git worktree`):

```bash
sbx create --clone --name finn-lane-1 … "$PWD" "$PWD" …
```

The clone appears at the checkout's path inside the sandbox; the startup hook
installs it once git has written it (`/tmp/finn-ready`, as before). sbx adds a
`sandbox-finn-lane-1` remote to your checkout: `git fetch sandbox-finn-lane-1`
brings the agent's branches back for review. Removing the sandbox removes the
clone.

## Network

FINN needs no network in a sandbox except the licence server. Whether
everything else is closed is decided by your sbx policy, not by FINN: close it
with `sbx policy init deny-all` on a new installation (on an existing one,
`sbx policy reset` first, which deletes all local rules and stops every running
sandbox). Each sandbox then reaches only what its kits allow. Under
organization-managed policy only organization allow rules grant access; ask
your administrator to allow the licence server's two ports.

## Environment files

An [environment file](https://docs.docker.com/ai/sandboxes/configuration/environment-files/)
would hold the command line above (kits, mounts, arguments) in one place. sbx
0.46.0 environment files select only built-in agents, not a workload kit; FINN
will ship an example once they can name one.

References: [kits](https://docs.docker.com/ai/sandboxes/customize/),
[local network policy](https://docs.docker.com/ai/sandboxes/governance/access-controls/local/),
[credentials](https://docs.docker.com/ai/sandboxes/configuration/credentials/).
