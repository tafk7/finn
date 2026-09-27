# Native sbx examples

FINN supplies a prepared image and examples. You and your site own copied
configuration, agents, credentials, mounts and network policy. Native sbx owns
composition, approval and sandbox lifecycle. These files are JSON-form YAML.
Validated with sbx client/server 0.43.0 (0.43 replaced `shareSkills` with `skills`); native environments and kits are experimental.

## Development

From your FINN checkout, prepare the image and copy the examples to a new directory
outside **every mounted workspace**, including additional toolchain directories:

```bash
./docker/build --sbx
TEMPLATE=$(./docker/build --sbx --print-tag)
CHECKOUT=$PWD
ENV_DIR=$(mktemp -d "$HOME/finn-native.XXXXXX")
cp docker/sbx/sbxenv.yaml docker/sbx/fpga.sbxenv.yaml "$ENV_DIR/"
cp -R docker/sbx/site-license "$ENV_DIR/"
FILES=("$ENV_DIR/sbxenv.yaml")
ARGS=(--env-arg name=finn-dev \
  --env-arg workspace="$CHECKOUT" --env-arg template="$TEMPLATE")
sbx env plan "${ARGS[@]}" "${FILES[@]}"
sbx env create "${ARGS[@]}" "${FILES[@]}"
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- python -c 'import finn'
sbx env run "${ARGS[@]}" "${FILES[@]}"
sbx env rm "${ARGS[@]}" "${FILES[@]}" --force
```

These Bash arrays preserve the same files and arguments throughout the lifecycle.
Put argument flags before file paths: sbx `env exec` requires this order.
The default agent is `shell`. To select a supported coding agent,
append `--env-arg agent=claude` to `ARGS` before planning and creating a new sandbox.
The FINN template has no coding-agent executable. Selecting an agent does not
install it. After `create` and before `run`, install your selected client inside
the sandbox, or supply a user-owned derived template that already contains it.
For Claude, the [vendor installer](https://code.claude.com/docs/en/setup) is:

```bash
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- bash -o pipefail -c \
  'wget -qO- https://claude.ai/install.sh | bash -s -- stable'
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- sh -c \
  'sudo ln -sf "$HOME/.local/bin/claude" /usr/local/bin/claude'
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- claude --version
```

The sandbox-local symlink makes the client available on the template's existing
PATH. Site policy must permit the installer downloads. Repeat installation after
recreating a generic template sandbox; sites can pin a client version in their
own setup. FINN does not duplicate its Python dependencies in a kit.
Agent authentication belongs in user-owned native configuration; FINN neither
registers credentials nor defines custom agents. Pass personal overlay paths
explicitly: native commands with explicit paths skip `~/.sbxenv.yaml` defaults.

The base has no FPGA mounts, licence settings or optional site kit. It sets
`FINN_BUILD_DIR=/tmp/finn_build` and turns the shared skills store off. Absence of
FINN network grants does not establish closed networking: machine/organization
policy and selected agent kits determine connectivity. Provisioning can also
require package-repository access.

## FPGA and a coding agent together

The optional overlay requires explicit absolute paths. For an older installation,
paths might end in `Vivado/2024.2`, `Vitis/2024.2` and `Vitis_HLS/2024.2`. For a
newer installation they may end in `2025.1/Vivado` and `2025.1/Vitis` (also HLS).
Choose paths that exist at your site; this example performs no host discovery.

```bash
FILES=("$ENV_DIR/sbxenv.yaml" "$ENV_DIR/fpga.sbxenv.yaml")
ARGS=(--env-arg name=finn-fpga-agent --env-arg workspace="$CHECKOUT" \
  --env-arg template="$TEMPLATE" --env-arg agent=claude \
  --env-arg toolchain=/opt/Xilinx \
  --env-arg vivado=/opt/Xilinx/Vivado/2024.2 \
  --env-arg vitis=/opt/Xilinx/Vitis/2024.2 \
  --env-arg hls=/opt/Xilinx/Vitis_HLS/2024.2 \
  --env-arg license_host=license.example.com --env-arg license_port=2100)
sbx env plan "${ARGS[@]}" "${FILES[@]}"
```

The root is mounted read-only at the same absolute path; Vivado, Vitis and HLS
variables and their FINN aliases are explicit. `XILINXD_LICENSE_FILE` demonstrates
a floating licence (`2100@license.example.com`). Substitute your site's values.
Licence-file directories and external platform directories are deliberate site
additions: add read-only `additionalWorkspaces` entries and the matching
`XILINXD_LICENSE_FILE` / `PLATFORM_REPO_PATHS` values in a user-owned overlay.
A licence file may depend on a host ID that is unavailable in the microVM.

## Optional site network mixin

If your site needs explicit licence network grants, create
`$ENV_DIR/license.sbxenv.yaml` alongside the copied `site-license` directory:

```yaml
args:
  vendor_port:
    required: true
kits:
  - source: ./site-license
    args:
      host: "${{ env.args.license_host }}"
      manager_port: "${{ env.args.license_port }}"
      vendor_port: "${{ env.args.vendor_port }}"
```

Append this overlay and the **pinned** vendor-daemon port to the FPGA request:

```bash
FILES+=("$ENV_DIR/license.sbxenv.yaml")
ARGS+=(--env-arg vendor_port=2101)
sbx env plan "${ARGS[@]}" "${FILES[@]}"
sbx env create "${ARGS[@]}" "${FILES[@]}"
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- bash -o pipefail -c \
  'wget -qO- https://claude.ai/install.sh | bash -s -- stable'
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- sh -c \
  'sudo ln -sf "$HOME/.local/bin/claude" /usr/local/bin/claude'
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- vivado -version
sbx env run "${ARGS[@]}" "${FILES[@]}"
sbx env rm "${ARGS[@]}" "${FILES[@]}" --force
```

The mixin requests the licence-manager and vendor-daemon connections separately.
If the vendor daemon has an unpinned port, a site may deliberately replace the
port-specific rules in its copy with an exact host rule such as
`license.example.com`. That permits all ports on that host; this example does not
select it automatically. Policy readback or `lmstat` alone does not prove a real
licence checkout. Validate an actual licensed tool operation at your site.

Keep the base, chosen overlays and optional kit together when copying. Relative
kit references resolve beside the file declaring them. Native mappings merge,
lists concatenate, and later scalar values replace earlier ones; an empty list
does not remove earlier mounts or kits. Use the same files and arguments for
`plan`, `create`, `run`, `exec` and `rm`. Recreate after changing templates, mounts
or kit permissions. Review native plans before approval; unattended jobs may use
`create --auto-approve` after reviewing the concrete plan.

Existing generated FINN directories are ordinary native files and are left intact.
Keep using them with `sbx env plan /existing/path/finn.sbxenv.yaml`, then native
`create`, `run`, `exec` and `rm` with that same path (and any original overlays and
arguments). Alternatively, create user-owned copies of these examples. There is
no automatic state migration or sandbox removal.

References: [native environments](https://docs.docker.com/ai/sandboxes/configuration/environment-files/)
and [kit schema](https://docs.docker.com/ai/sandboxes/customize/kit-reference/).

## The development environment

There is nothing to prepare. The template's `/opt/venv` is active for every
command and already holds FINN's locked dependencies. When the sandbox starts, the
image entrypoint installs the workspace checkout (`FINN_ROOT`, set by
`sbxenv.yaml`) editable into it; finn-hlslib and the board files are already in
the image's resource cache. It then writes `/tmp/finn-ready`; scripts that `exec` into a sandbox
immediately after creating it can wait for that file.

```bash
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- python -m finn.util.installation
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- build_dataflow --help
```

After pulling a change to `uv.lock` or `pyproject.toml`, run `uv sync --inexact`
in the sandbox (or recreate it).

To co-develop QONNX or another dependency, point its entry in
`[tool.uv.sources]` at your checkout (locally; do not commit it), mount that path
at the same place, and run `uv sync --inexact`:

```yaml
additionalWorkspaces:
  - path: /absolute/path/to/qonnx
    readOnly: false
```

## FinnLib

`finn.kernels` and the MVAU/VVAU replay buffer compile against FinnLib, a private
repository that changes together with FINN. A sandbox has no SSH credentials to
fetch it, so mount a clone with the `finnlib` overlay; FINN then uses it through
`FINN_RESOURCES_FINNLIB`, and an agent can edit and commit in it (push from the
host):

```bash
cp docker/sbx/finnlib.sbxenv.yaml "$ENV_DIR/"
FILES+=("$ENV_DIR/finnlib.sbxenv.yaml")
ARGS+=(--env-arg finnlib=/absolute/path/to/finnlib)
sbx env exec "${ARGS[@]}" "${FILES[@]}" -- finn-resources list
```

Deleting the sandbox deletes its environment changes; the next sandbox starts from
the template again. See [installation](../../docs/installation.md).
