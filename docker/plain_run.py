# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""FINN's container as plain `docker run`, for docker/run on hosts without Docker
Compose (Ubuntu's docker.io ships none) and with FINN_DOCKER_EXTRA: a site's extra
`docker run` arguments (run-docker.sh's variable, which a CI job sets for its tool
wrappers' mounts, a licence and the network), passed through as given.

definition() is compose.yaml's service (with compose.tools.yaml's additions when the
AMD tools come in) as `docker compose config` renders it from the same variables.
tests/docker_scripts/test_plain_run.py compares the two wherever Compose is
installed, so that a change to either shows as a failure, not as drift.
"""
import shlex

VARIABLE = "FINN_DOCKER_EXTRA"

# compose.tools.yaml's environment: names the container takes from docker/run's.
TOOL_VARIABLES = (
    "XILINX_VIVADO",
    "FINN_XILINX_PATH",
    "FINN_XILINX_VERSION",
    "XILINXD_LICENSE_FILE",
    "LM_LICENSE_FILE",
    "PLATFORM_REPO_PATHS",
    "FINN_TOOL_DIR_OVERRIDE",
)


class PlainRunError(Exception):
    """FINN_DOCKER_EXTRA, or a definition, docker/run cannot start a container from."""


def split(value):
    """FINN_DOCKER_EXTRA's arguments, split as shell words."""
    try:
        return shlex.split(value)
    except ValueError as error:
        # shlex names the fault ("No closing quotation"), not the text.
        raise PlainRunError(f"{VARIABLE} does not split as shell words: {error}") from None


def explain(value, compose):
    """One line on how docker/run starts the container, without FINN_DOCKER_EXTRA's
    values. compose: whether `docker compose` is installed."""
    if value and value.strip():
        return (
            f"{VARIABLE}: {len(split(value))} arguments, appended as given to plain "
            f"`docker run` (docker compose {'present' if compose else 'absent'})"
        )
    if compose:
        return f"{VARIABLE}: not set; docker/run starts the container with docker compose"
    return (
        f"{VARIABLE}: not set; docker compose is absent, so docker/run starts the "
        "container with plain `docker run`"
    )


# --------------------------------------------------------------------------
# compose.yaml and compose.tools.yaml, as `docker compose config` renders them.
# --------------------------------------------------------------------------


def _value(env, name, default):
    """${name:-default}"""
    return env.get(name) or default


def _bind(path, read_only=False):
    volume = {"type": "bind", "source": path, "target": path, "bind": {}}
    if read_only:
        volume["read_only"] = True
    return volume


def definition(env, service="finn", tools=False):
    """The rendered service, from the variables docker/run gives Compose."""
    root = _value(env, "FINN_ROOT", env.get("PWD", ""))
    home = _value(env, "FINN_HOME", f"{env.get('HOME', '')}/.finn")
    build = _value(env, "FINN_BUILD_DIR", f"/tmp/finn_build_{_value(env, 'FINN_UID', '1000')}")
    rendered = {
        "command": None,
        "entrypoint": None,
        "environment": {
            "USER": _value(env, "USER", "finn"),
            "FINN_ROOT": root,
            "FINN_HOME": home,
            "FINN_BUILD_DIR": build,
        },
        "image": _value(env, "FINN_IMAGE", "finn:dev"),
        "init": False,
        "network_mode": "bridge",
        "platform": "linux/amd64",
        "user": f"{_value(env, 'FINN_UID', '1000')}:{_value(env, 'FINN_GID', '1000')}",
        "volumes": [_bind(root), _bind(home), _bind(build)],
        "working_dir": _value(env, "FINN_WORKDIR", root),
    }
    # Compose drops an empty hostname.
    if env.get("FINN_HOSTNAME"):
        rendered["hostname"] = env["FINN_HOSTNAME"]
    if tools:
        if not env.get("FINN_XILINX_ROOT"):
            raise PlainRunError("FINN_XILINX_ROOT is not set")
        rendered["environment"].update({name: env.get(name) for name in TOOL_VARIABLES})
        rendered["volumes"].append(_bind(env["FINN_XILINX_ROOT"], read_only=True))
    if service == "notebook":
        jupyter, netron = _value(env, "JUPYTER_PORT", "8888"), _value(env, "NETRON_PORT", "8081")
        rendered["command"] = [
            "jupyter",
            "notebook",
            "--no-browser",
            "--ip=0.0.0.0",
            "--port",
            jupyter,
            "notebooks",
        ]
        rendered["ports"] = [
            {"mode": "ingress", "protocol": "tcp", "published": port, "target": int(port)}
            for port in (jupyter, netron)
        ]
        rendered["profiles"] = ["notebook"]
    return rendered


# --------------------------------------------------------------------------
# docker run.
# --------------------------------------------------------------------------


def _volume_flag(volume):
    spec = f"{volume['source']}:{volume['target']}"
    return spec + ":ro" if volume.get("read_only") else spec


def _port_flag(port):
    host = f"{port['host_ip']}:" if port.get("host_ip") else ""
    published = f"{port['published']}:" if port.get("published") else ""
    return f"{host}{published}{port['target']}"


def service_flags(service):
    """docker run flags for a rendered service; a key without one stops the launch
    rather than being dropped."""
    argv = []
    for key, value in service.items():
        if key in ("image", "command", "profiles") or (key == "entrypoint" and value is None):
            continue
        if key == "environment":
            for name, setting in value.items():
                # None: named without a value, and unset in docker/run's environment.
                if setting is not None:
                    argv += ["--env", f"{name}={setting}"]
        elif key == "volumes":
            for volume in value:
                argv += ["--volume", _volume_flag(volume)]
        elif key == "ports":
            for port in value:
                argv += ["--publish", _port_flag(port)]
        elif key == "init":
            if value:
                argv.append("--init")
        elif key in ("platform", "user", "hostname"):
            argv += [f"--{key}", str(value)]
        elif key == "working_dir":
            argv += ["--workdir", str(value)]
        elif key == "network_mode":
            argv += ["--network", str(value)]
        else:
            raise PlainRunError(f"the service key {key!r} has no docker run flag in docker/run")
    return argv


def claimed(tokens):
    """The network flag and the mount targets in FINN_DOCKER_EXTRA. docker run takes the
    last of most flags, but joins every --network and refuses two mounts on one target,
    so docker/run's own definition yields these to it."""
    network, targets = False, set()
    for i, token in enumerate(tokens):
        name, attached, value = token.partition("=")
        if name in ("--network", "--net"):
            network = True
        elif name in ("-v", "--volume") and (attached or i + 1 < len(tokens)):
            parts = (value if attached else tokens[i + 1]).split(":")
            if len(parts) > 1:
                targets.add(parts[1])
    return network, targets


def docker_run(service, tokens, volumes, variables, command, terminal):
    """The docker run command, in run-docker.sh's order: docker/run's own definition,
    FINN_DOCKER_EXTRA as given, then what the caller of docker/run adds (its host name,
    --volume and --env, so these win), the image and the command."""
    network, targets = claimed(tokens)
    own = {k: v for k, v in service.items() if k != "hostname"}
    if network:
        own.pop("network_mode", None)
    own["volumes"] = [v for v in service["volumes"] if v["target"] not in targets]
    argv = ["docker", "run", "--rm", "--pull", "never", "--interactive"]
    if terminal:
        argv.append("--tty")
    argv += service_flags(own)
    argv += tokens
    if service.get("hostname"):
        argv += ["--hostname", service["hostname"]]
    for volume in volumes:
        argv += ["--volume", volume]
    for variable in variables:
        argv += ["--env", variable]
    argv.append(service["image"])
    argv += command or (service["command"] or [])
    return argv
