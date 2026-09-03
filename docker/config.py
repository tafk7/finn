#!/usr/bin/env python3
"""Resolve host capabilities for FINN's Docker-built environment.

This module is host-side only. It is the sole owner of workspace, toolchain,
licence, mount and egress discovery. ``compose.yaml`` holds static Docker
behavior; Compose overrides and complete sbx environment files are rendered here.
Diagnostics go to stderr because stdout is machine-readable data.
"""

import argparse
import json
import os
import re
import sys

# A tier is a GRANT profile, not an image. There is one image; what differs
# between these two is what the launcher mounts and allows.
#
# `build-xrt` used to be a third tier. It differed from `build` by one mount,
# and its real content -- XRT -- is now a runtime target baked into the image
# and named in its tag. See docker/runtimes/README.md.
TIERS = ("dev", "build")

# Tiers that get a toolchain, a licence and toolchain egress. `dev` deliberately
# gets none of it: that boundary is the reason the tier exists, and it is
# enforced here rather than left to whether the caller's shell happens to have
# FINN_XILINX_PATH set.
#
# This is now the ONLY place the boundary is drawn. It used to be drawn twice --
# here, and again by which image you pulled -- and the two could disagree.
FPGA_TIERS = ("build",)


def normalize_runtimes(runtimes=None):
    """Return the runtime set in its canonical, duplicate-free order."""
    if runtimes is None:
        runtimes = os.environ.get("FINN_RUNTIMES", "")
    return sorted(set(n for n in runtimes.replace(",", " ").split() if n))


# D4: the workspace path policy.
#
#   mirror  container path == host path. Required for anything that generates
#           Vivado projects, because add_files writes $::env(FINN_ROOT) into the
#           .xpr as an absolute path (LIMITATION(finn-root-absolute)), so a
#           project built under a fixed path cannot be opened in the host GUI.
#           Also forced by sbx, which has no mount remapping.
#   fixed   container path == FIXED_WORKSPACE. Better for remote daemons,
#           reproducible diagnostics and Dev Container config.
#
# dev defaults to "fixed" and the FPGA tiers to "mirror". dev is Python-only and
# generates no Vivado projects, so mirroring buys it nothing while costing
# remote-daemon support, reproducible diagnostic paths, and simple Dev Container
# config. The FPGA tiers generate projects and so must mirror.
#
# CONSEQUENCE: FINN_ROOT differs between dev and the FPGA tiers on one host, so
# the workspace path moves when you switch tiers. finn_paths.py resolves either
# correctly, but it surprises people; it is called out in compose.yaml and the
# docs for that reason.
FIXED_WORKSPACE = "/workspace/finn"
DEFAULT_DEV_WORKSPACE_POLICY = "fixed"

# Vendor executables that get a transparent shim (stage 3). Listed here because
# the shim needs the same toolchain resolution this file already performs.
SHIMMED_TOOLS = (
    "vivado",
    "vitis",
    "vitis_hls",
    "vitis-run",
    "v++",
    "slashkit",
    "xelab",
    "xsim",
    "xvlog",
    "xsc",
    "lmutil",
    "xsct",
)


def warn(msg):
    print("docker/config: %s" % msg, file=sys.stderr)


def die(msg, code=1):
    print("docker/config: error: %s" % msg, file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------
# Shared resolution. Used by both the host and container sides.
# --------------------------------------------------------------------------


def hostpath(value):
    """Expand ``~`` and make a host path absolute before it becomes a mount."""
    if not value:
        return value
    return os.path.abspath(os.path.expanduser(value))


def xilinx_layout(root, version):
    """Locate Vivado / Vitis / HLS within a Xilinx install.

    AMD reorganised the install tree after 2024.2:

        <= 2024.2   $ROOT/Vivado/2022.2   $ROOT/Vitis_HLS/2022.2
        >  2024.2   $ROOT/2025.1/Vivado   $ROOT/2025.1/Vitis

    Returning both candidates and probing is deliberate. Deciding from the
    version string alone means a site with a symlinked or non-standard tree gets
    a confidently wrong answer; probing means an unusual layout is detected
    rather than assumed. The version string only orders the candidates.
    """
    if not root or not version:
        return {}

    m = re.match(r"^(20\d\d)\.([12])$", version)
    if not m:
        warn(
            "FINN_XILINX_VERSION %r is not YYYY.1 or YYYY.2; probing both layouts anyway" % version
        )
        new_first = False
    else:
        new_first = (int(m.group(1)), int(m.group(2))) > (2024, 2)

    new = {
        "XILINX_VIVADO": os.path.join(root, version, "Vivado"),
        "XILINX_VITIS": os.path.join(root, version, "Vitis"),
        "XILINX_HLS": os.path.join(root, version, "Vitis"),
    }
    old = {
        "XILINX_VIVADO": os.path.join(root, "Vivado", version),
        "XILINX_VITIS": os.path.join(root, "Vitis", version),
        "XILINX_HLS": os.path.join(root, "Vitis_HLS", version),
    }

    found = {}
    for candidate in (new, old) if new_first else (old, new):
        for key, path in candidate.items():
            if key not in found and os.path.isdir(path):
                found[key] = path
    return found


def vendor_daemon_port(files):
    """The pinned FLEXlm vendor-daemon port, if the site pinned one.

    FLEXlm needs TWO connections. lmgrd on the advertised port is a directory
    service; it hands back a second port where the vendor daemon (xilinxd) does
    the actual checkout. That second port is EPHEMERAL by default, which is why
    the egress grant defaults to the whole host.

    It does not have to be ephemeral. A licence admin can pin it:

        DAEMON xilinxd /opt/xilinx/xilinxd port=2101

    When we can see a licence file, read it. When we cannot -- a bare PORT@HOST
    tells us nothing about the DAEMON line -- FINN_LICENSE_VENDOR_PORT lets the
    user state it.

    Returns None when unknown, and None means "grant the host", not "guess".
    """
    explicit = os.environ.get("FINN_LICENSE_VENDOR_PORT", "").strip()
    if explicit:
        if not explicit.isdigit():
            warn("FINN_LICENSE_VENDOR_PORT=%r is not a port number; ignoring" % explicit)
        else:
            return explicit

    for path in files or ():
        try:
            with open(hostpath(path), "r", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        m = re.search(r"^\s*DAEMON\s+\S+.*?\bport\s*=\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1)
    return None


def classify_license(value):
    """Split a FLEXlm licence variable into server and file entries.

    XILINXD_LICENSE_FILE / LM_LICENSE_FILE take a colon-separated list of
    either form:

        PORT@HOST     floating server. Nothing to mount; needs TCP egress.
        /path/to.lic  node-locked file. Must be readable inside the container.

    Egress defaults to the whole HOST, because the vendor-daemon port is
    ephemeral unless the site pinned it -- see vendor_daemon_port(). A
    port-scoped rule against an unpinned daemon lets `lmutil lmstat` succeed,
    because lmstat only talks to lmgrd, while every real checkout fails with
    "A valid license was not found" -- a thoroughly misleading error for a
    firewall problem.

    So: narrow ONLY when the vendor port is known, and say so out loud when you
    do. lmstat is not a test of the narrowed rule.
    """
    servers, files = [], []
    for entry in (value or "").split(":"):
        entry = entry.strip()
        if not entry:
            continue
        if "@" in entry:
            port, _, host = entry.partition("@")
            servers.append({"host": host, "advertised_port": port})
        else:
            files.append(entry)
    return servers, files


def resolve_tier(tier):
    """Resolve `auto` to a real tier, here rather than in every launcher.

    `auto` means "build if this machine has a toolchain, dev if it does not".
    That degrade used to live in the legacy launcher, which made it a fact derived in
    a launcher -- the exact shape of the four defects this program exists to
    prevent. It belongs here, once.

    `dev` and `build` stay EXPLICIT requests. An explicit `--tier build` with no
    toolchain is still a hard error: that is a request, not a default, and
    silently handing back a narrower environment than the caller asked for is
    how a CI shard ends up passing without testing anything.
    """
    if tier != "auto":
        return tier
    root = hostpath(os.environ.get("FINN_XILINX_PATH"))
    if root and os.path.isdir(root):
        return "build"
    warn(
        "no FINN_XILINX_PATH; resolving --tier auto to 'dev'. "
        "Vivado, Vitis, HLS and rtlsim will be unavailable."
    )
    return "dev"


def resolve_workspace(tier, backend, policy="auto"):
    """Resolve the host and container workspace paths."""
    workspace_host = hostpath(
        os.environ.get("FINN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    forced_mirror = backend == "sbx" or tier in FPGA_TIERS
    if policy == "auto":
        policy = "mirror" if forced_mirror else DEFAULT_DEV_WORKSPACE_POLICY
    elif policy == "fixed" and forced_mirror:
        die(
            "workspace policy 'fixed' is incompatible with backend %r / tier %r" % (backend, tier),
            2,
        )
    workspace_target = workspace_host if policy == "mirror" else FIXED_WORKSPACE
    return {"policy": policy, "source": workspace_host, "target": workspace_target}


def resolve_build_dir(create=False, fatal=False):
    """Resolve the per-user host build directory, optionally creating it."""
    path = hostpath(os.environ.get("FINN_HOST_BUILD_DIR", "/tmp/finn_build_%d" % os.getuid()))
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            message = "could not create the build directory %s (%s)" % (path, exc)
            if fatal:
                die(message, 3)
            warn(message)
    return path


def add_toolchain(out):
    """Add toolchain paths and their read-only root mount."""
    root = hostpath(os.environ.get("FINN_XILINX_PATH"))
    version = os.environ.get("FINN_XILINX_VERSION")
    if not root:
        die("FINN_XILINX_PATH is unset; tier %r requires it" % out["tier"], 3)
    if not os.path.isdir(root):
        die("FINN_XILINX_PATH=%s is not a directory" % root, 3)

    out["mounts"].append(
        {"source": root, "target": root, "mode": "ro", "reason": "xilinx-toolchain"}
    )
    layout = xilinx_layout(root, version)
    if not layout:
        warn(
            "no Vivado/Vitis/HLS found under %s for version %r; "
            "the toolchain mount will be present but empty of known tools" % (root, version)
        )
    out["env"].update(layout)
    aliases = {
        "XILINX_VIVADO": "VIVADO_PATH",
        "XILINX_VITIS": "VITIS_PATH",
        "XILINX_HLS": "HLS_PATH",
    }
    for source, alias in aliases.items():
        if source in layout:
            out["env"][alias] = layout[source]
    return root


def add_platform_repo(out, toolchain_root):
    """Add the optional platform repository without overlapping the toolchain."""
    platforms = hostpath(os.environ.get("PLATFORM_REPO_PATHS"))
    if not platforms or not os.path.isdir(platforms):
        return
    if not os.path.abspath(platforms).startswith(os.path.abspath(toolchain_root) + os.sep):
        out["mounts"].append(
            {"source": platforms, "target": platforms, "mode": "ro", "reason": "platform-repo"}
        )
    out["env"]["PLATFORM_REPO_PATHS"] = platforms


def add_licenses(out):
    """Add FLEXlm environment, file mounts and network grants."""
    for var in ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE"):
        value = os.environ.get(var)
        if not value:
            continue
        out["env"][var] = value
        servers, files = classify_license(value)
        vendor = vendor_daemon_port(files)
        for server in servers:
            grant = {
                "host": server["host"],
                "reason": "flexlm",
                "advertised_port": server["advertised_port"],
                "ports": [],
            }
            if vendor:
                grant["ports"] = sorted({server["advertised_port"], vendor})
                grant["vendor_port"] = vendor
            out["egress"].append(grant)
        for path in files:
            directory = os.path.dirname(hostpath(path))
            if not os.path.isdir(directory):
                warn("licence path %s not found on the host; not mounted" % path)
                continue
            if not any(m["source"] == directory for m in out["mounts"]):
                out["mounts"].append(
                    {
                        "source": directory,
                        "target": directory,
                        "mode": "ro",
                        "reason": "license-file",
                    }
                )


def add_optional_inputs(out):
    """Add explicitly requested, tier-independent runtime inputs."""
    for variable in ("NUM_DEFAULT_WORKERS", "FINN_XELAB_MT"):
        if os.environ.get(variable):
            out["env"][variable] = os.environ[variable]
    imagenet = hostpath(os.environ.get("IMAGENET_VAL_PATH"))
    if not imagenet:
        return
    out["env"]["IMAGENET_VAL_PATH"] = imagenet
    if os.path.isdir(imagenet):
        out["mounts"].append(
            {"source": imagenet, "target": imagenet, "mode": "ro", "reason": "imagenet"}
        )
    else:
        warn("IMAGENET_VAL_PATH=%s is not a directory; not mounted" % imagenet)


def resolve_host(tier, backend, workspace_policy="auto"):
    """Everything a launcher needs, derived from the host exactly once."""
    tier = resolve_tier(tier)
    if tier not in TIERS:
        die("unknown tier %r; expected one of %s" % (tier, ", ".join(TIERS)), 2)

    runtimes = normalize_runtimes()
    workspace = resolve_workspace(tier, backend, workspace_policy)

    out = {
        "tier": tier,
        "backend": backend,
        "runtimes": runtimes,
        "runtime_csv": ",".join(runtimes),
        "platform": "linux/amd64",
        "workspace": workspace,
        "build_dir": resolve_build_dir(create=False),
        "mounts": [],
        "egress": [],
        "egress_enforcement": "enforced" if backend == "sbx" else "declared",
        "env": {
            "FINN_ROOT": workspace["target"],
            "FINN_DEPS": os.environ.get("FINN_DEPS", "frozen").lower(),
        },
    }
    for variable in (
        "FINN_IMAGE_REVISION",
        "FINN_SOURCE_REVISION",
        "FINN_SOURCE_DESCRIBE",
        "FINN_SOURCE_DIRTY",
    ):
        if os.environ.get(variable):
            out["env"][variable] = os.environ[variable]
    add_optional_inputs(out)

    if tier == "dev":
        out["dev_contract"] = {
            "toolchain": False,
            "license": False,
            "egress": False,
        }
        return out

    root = add_toolchain(out)
    add_platform_repo(out, root)
    add_licenses(out)

    return out


def shquote(value):
    """Quote a value for both shell ``eval`` and a Compose env file."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def sh_assignments(data, create_build_dir=True):
    """Return the complete launcher/Compose environment as shell assignments."""
    build_dir = resolve_build_dir(create=create_build_dir)
    values = dict(data["env"])
    values.update(
        {
            "FINN_BUILD_DIR": build_dir,
            "FINN_HOST_BUILD_DIR": build_dir,
            "FINN_GID": str(os.getgid()),
            "FINN_RUNTIMES": data["runtime_csv"],
            "FINN_UID": str(os.getuid()),
            "FINN_WORKSPACE_SOURCE": data["workspace"]["source"],
            "FINN_WORKSPACE_TARGET": data["workspace"]["target"],
        }
    )
    if data["tier"] in FPGA_TIERS and os.environ.get("FINN_XILINX_PATH"):
        values["FINN_XILINX_PATH"] = hostpath(os.environ["FINN_XILINX_PATH"])
    if os.environ.get("FINN_IMAGE"):
        values["FINN_IMAGE"] = os.environ["FINN_IMAGE"]
    return "".join("%s=%s\n" % (key, shquote(value)) for key, value in sorted(values.items()))


def compose_bind(source, target, mode="rw"):
    """Return one long-form Compose bind mount."""
    mount = {
        "type": "bind",
        "source": source,
        "target": target,
        "bind": {"create_host_path": False},
    }
    if mode == "ro":
        mount["read_only"] = True
    return mount


def compose_override(data, services):
    """Render host-specific runtime configuration for Compose."""
    image = os.environ.get("FINN_IMAGE")
    if data["runtime_csv"] and not image:
        die(
            "FINN_IMAGE must be set when FINN_RUNTIMES is non-empty; "
            "use docker/run or resolve the tag through Bake",
            2,
        )
    build_dir = resolve_build_dir(create=True, fatal=True)
    volumes = [
        compose_bind(data["workspace"]["source"], data["workspace"]["target"]),
        compose_bind(build_dir, build_dir),
    ]
    seen_targets = {mount["target"] for mount in volumes}
    for mount in data["mounts"]:
        if mount["target"] in seen_targets:
            continue
        volumes.append(compose_bind(mount["source"], mount["target"], mount.get("mode", "rw")))
        seen_targets.add(mount["target"])

    environment = dict(data["env"])
    environment.update(
        {
            "FINN_BUILD_DIR": build_dir,
            "FINN_RUNTIMES": data["runtime_csv"],
        }
    )
    service = {
        "build": {"args": {"FINN_RUNTIMES": data["runtime_csv"]}},
        "environment": environment,
        "user": "%d:%d" % (os.getuid(), os.getgid()),
        "volumes": volumes,
        "working_dir": data["workspace"]["target"],
    }
    if image:
        service["image"] = image
    if os.environ.get("FINN_CONTAINER_NAME"):
        service["container_name"] = os.environ["FINN_CONTAINER_NAME"]
    return {"services": {name: dict(service) for name in services}}


def cmd_inspect(args):
    backend = "sbx" if args.sbx else "docker"
    data = resolve_host(args.tier, backend, args.workspace_policy)
    if args.format == "json":
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(sh_assignments(data))
    return 0


def cmd_compose(args):
    data = resolve_host(args.tier, "docker", args.workspace_policy)
    services = args.service or [data["tier"]]
    json.dump(compose_override(data, services), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_sbx(args):
    data = resolve_host(args.tier, "sbx", "mirror")
    name = os.environ.get("FINN_SBX_NAME")
    template = os.environ.get("FINN_SBX_TEMPLATE")
    agent = os.environ.get("FINN_SBX_AGENT", "shell")
    if not name or not template:
        die("FINN_SBX_NAME and FINN_SBX_TEMPLATE must be set", 2)
    config = {
        "schemaVersion": "1",
        "name": name,
        "agent": agent,
        "workspace": data["workspace"]["source"],
        "additionalWorkspaces": [
            {"path": mount["source"], "readOnly": mount["mode"] == "ro"} for mount in data["mounts"]
        ],
        "env": dict(data["env"]),
        "sandboxOptions": {
            "template": template,
            "pullPolicy": "never",
        },
    }
    config["env"]["FINN_BUILD_DIR"] = "/tmp/finn_build"
    json.dump(config, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def add_resolution_arguments(parser, include_sbx=True):
    parser.add_argument(
        "--tier", default=os.environ.get("FINN_DOCKER_TARGET", "auto"), choices=TIERS + ("auto",)
    )
    if include_sbx:
        parser.add_argument("--sbx", action="store_true", help="resolve sbx-specific policy")
    parser.add_argument("--workspace-policy", default="auto", choices=("auto", "fixed", "mirror"))


def main():
    parser = argparse.ArgumentParser(prog="docker/config", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="resolve host configuration")
    add_resolution_arguments(p)
    p.add_argument("--format", default="json", choices=("json", "sh"))
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("compose", help="render an ephemeral Compose override")
    add_resolution_arguments(p, include_sbx=False)
    p.add_argument(
        "--service", action="append", help="service to configure; repeat for multiple services"
    )
    p.set_defaults(func=cmd_compose)

    p = sub.add_parser("sbx", help="render a complete sbx environment file")
    p.add_argument("--tier", default="dev", choices=TIERS)
    p.set_defaults(func=cmd_sbx)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
