# FINN in Docker Sandboxes (sbx)

FINN supplies a prepared image, a base environment (`sbxenv.yaml` at the
repository root) and overlays in this directory. You and your site own the
machine's network policy, agents, credentials and site values. sbx owns
composition, approval and sandbox lifecycle.

Requires sbx 0.43 or later (0.43 renamed `shareSkills` to `skills` and fixed
allow rules for IP addresses); validated with 0.43.0. sbx environments and kits
are experimental and change between minor releases: check the
[release notes](https://docs.docker.com/ai/sandboxes/release-notes/) when upgrading.

| File | Adds |
|---|---|
| `sbxenv.yaml` (repository root) | The checkout as the workspace, FINN's image, no network grants |
| `fpga.sbxenv.yaml` | Your Xilinx installation, read-only, and `XILINXD_LICENSE_FILE` |
| `finnlib.sbxenv.yaml` | A FinnLib clone, writable, as the `finnlib` resource |
| `license.sbxenv.yaml` + `site-license/` | Network access to the licence server's two ports |

## Where the files live

The base stays in the checkout: sbx mounts it read-only, and a file at the
workspace root cannot be swapped out. Copy the overlays and the licence kit to a
directory you own, outside every workspace the sandbox mounts, and keep your site
values there too:

```bash
mkdir -p ~/.config/finn-sbx
cp -R docker/sbx/*.sbxenv.yaml docker/sbx/site-license ~/.config/finn-sbx/
```

An agent can write anywhere in the checkout. An overlay or kit read from there
could be changed to widen what the next sandbox may reach, and the plan sbx shows
before creating a sandbox lists a kit's source and arguments but not its
permissions. Re-copy after reviewing changes to these files.

Site values go in arguments files beside the copies, one per overlay, which no
repository contains. sbx rejects an argument no loaded file declares, so pass only
the files of the overlays you use:

```text
# ~/.config/finn-sbx/fpga.args
toolchain=/opt/Xilinx
vivado=/opt/Xilinx/2025.2/Vivado
vitis=/opt/Xilinx/2025.2/Vitis
hls=/opt/Xilinx/2025.2/Vitis
license_host=10.0.0.5
license_port=2100

# ~/.config/finn-sbx/license.args
vendor_port=2101

# ~/.config/finn-sbx/finnlib.args
finnlib=/home/you/finnlib
```

## Development

From the checkout, build the image and create a sandbox:

```bash
./docker/build --sbx
C=~/.config/finn-sbx
FILES=(sbxenv.yaml)
ARGS=(--env-arg template="$(./docker/build --sbx --print-tag)")
for overlay in fpga finnlib license; do   # the overlays you use
    FILES+=("$C/$overlay.sbxenv.yaml"); ARGS+=(--env-args-file "$C/$overlay.args")
done
sbx env plan "${ARGS[@]}" "${FILES[@]}"
sbx env create "${ARGS[@]}" "${FILES[@]}"
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- python -m finn.util.installation
sbx env run "${ARGS[@]}" "${FILES[@]}"
sbx env rm "${ARGS[@]}" "${FILES[@]}" --force
```

`license` needs `fpga`. `sbx env run` with no paths uses the base alone. Use the same files and arguments for every command, flags before file
paths. `--env-arg name=...` gives a second sandbox its own name (default `finn`);
`--env-arg agent=claude` selects a coding agent.

When the sandbox starts, the image entrypoint installs the checkout editable into
the active `/opt/venv`; finn-hlslib and the board files are already in the image.
It then writes `/tmp/finn-ready`, which scripts that `exec` right after `create`
can wait for. After pulling a change to `uv.lock`, run `uv sync --inexact` in the
sandbox or recreate it.

The FINN template contains no coding agent. After `create`, install one inside the
sandbox, which needs network access to the vendor's download, or build a derived
template that contains it. For Claude:

```bash
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- bash -o pipefail -c \
  'wget -qO- https://claude.ai/install.sh | bash -s -- stable'
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- sh -c \
  'sudo ln -sf "$HOME/.local/bin/claude" /usr/local/bin/claude'
```

## FPGA tools

`fpga.sbxenv.yaml` mounts the toolchain root read-only at the same path and sets
the Vivado, Vitis and HLS variables. Paths depend on the installation: older ones
end in `Vivado/2024.2`, newer ones in `2025.1/Vivado`. Licence-file directories and
platform repositories are site additions: add read-only `additionalWorkspaces`
entries and `XILINXD_LICENSE_FILE` / `PLATFORM_REPO_PATHS` in an overlay of your
own. A node-locked licence may depend on a host ID the microVM does not have.

## FinnLib

`finn.kernels` and the MVAU/VVAU replay buffer compile against FinnLib, a private
repository that changes together with FINN. A sandbox has no SSH credentials to
fetch it, so `finnlib.sbxenv.yaml` mounts your clone, writable, and FINN uses it
through `FINN_RESOURCES_FINNLIB`. An agent can edit and commit there; push from
the host. The mount is your real clone: keep uncommitted work of your own out of it
while an agent runs.

## Network and the licence server

FINN needs no network in a sandbox except the licence server. Whether everything
else is closed is decided by your sbx policy, not by FINN:

* **Close the network** with the global policy: `sbx policy init deny-all` on a new
  installation. On an existing one, `sbx policy reset` first, which deletes all
  local rules and stops every running sandbox. Each sandbox then reaches only
  what its kits allow: the licence ports from `license.sbxenv.yaml`, and the model
  API from a coding-agent kit. A per-sandbox "deny all except" is not possible:
  a deny rule overrides every allow.
* **Under organization-managed policy** only organization allow rules grant
  access; allows from kits (including `site-license`) and local rules are
  inactive. Ask your administrator to allow the licence server's two ports.
* **Use the server's IP address** for `license_host`. FlexLM connections are plain
  TCP, which sbx matches by address, and resolving a name needs DNS access,
  which a closed policy blocks. The kit refuses anything but an IPv4 address.
* **Finding the vendor port.** Unless the licence file pins it (`VENDOR xilinxd
  port=...`), the server chooses the xilinxd port. Create the sandbox with any
  value, run a licensed operation, and `sbx policy log SANDBOX` lists the blocked
  port; recreate with it.
* **Check** with `sbx policy check network --sandbox finn 10.0.0.5:2100` and a host
  that should be closed, such as `github.com:443`. Policy readback does not prove a
  checkout: validate with a licensed operation, such as synthesis for a Versal
  part.

The licence kit also tells the agent that a licence or download failure is a
network-policy block to report, not a design problem to work around.

References: [environment files](https://docs.docker.com/ai/sandboxes/configuration/environment-files/),
[local network policy](https://docs.docker.com/ai/sandboxes/governance/access-controls/local/),
[precedence](https://docs.docker.com/ai/sandboxes/governance/concepts/#precedence).
