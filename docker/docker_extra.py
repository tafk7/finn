# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""FINN_DOCKER_EXTRA for docker/run: the extra `docker run` arguments a site's job
sets (run-docker.sh's variable), for the LSF tool wrapper's mounts, a licence, the
network and the variables the tool-dispatch layer reads.

When every argument is one Compose expresses, they become a Compose override that
docker/run adds after its own files and before COMPOSE_FILE's. Otherwise docker/run
starts the container with plain `docker run`, from the same rendered Compose
definition, and passes the arguments through as given: its own definition first,
then FINN_DOCKER_EXTRA, then what COMPOSE_FILE's files add or change (a pipeline's
host name and caches), so later ones win, as run-docker.sh ordered them; docker/run's
own network and mounts yield to FINN_DOCKER_EXTRA's, which docker run would not
replace but join or refuse.

The string is split as shell words. Explanations name flags, never their values.
"""
import hashlib
import json
import os
import shlex
from collections import Counter
from pathlib import Path

VARIABLE = "FINN_DOCKER_EXTRA"

# docker run flags that take a value, and the Compose key that holds it. List keys
# collect every occurrence; the others take the last.
VALUED = {
    "-v": "volumes",
    "--volume": "volumes",
    "-e": "environment",
    "--env": "environment",
    "--env-file": "env_file",
    "--network": "network_mode",
    "--net": "network_mode",
    "-h": "hostname",
    "--hostname": "hostname",
    "--device": "devices",
    "--add-host": "extra_hosts",
    "--cap-add": "cap_add",
    "--cap-drop": "cap_drop",
    "--group-add": "group_add",
    "--security-opt": "security_opt",
    "--dns": "dns",
    "--dns-search": "dns_search",
    "--tmpfs": "tmpfs",
    "--shm-size": "shm_size",
    "--ipc": "ipc",
    "--pid": "pid",
    "-l": "labels",
    "--label": "labels",
    "-w": "working_dir",
    "--workdir": "working_dir",
    "-u": "user",
    "--user": "user",
}
LISTS = {
    "volumes",
    "environment",
    "env_file",
    "devices",
    "extra_hosts",
    "cap_add",
    "cap_drop",
    "group_add",
    "security_opt",
    "dns",
    "dns_search",
    "tmpfs",
    "labels",
}
# Flags without a value, and the Compose key they set true.
SWITCHES = {"--privileged": "privileged", "--init": "init", "--read-only": "read_only"}
NETWORKS = ("host", "bridge", "none")


class ExtraError(Exception):
    """FINN_DOCKER_EXTRA, or a Compose definition, docker/run cannot launch with."""


def split(value):
    """FINN_DOCKER_EXTRA's arguments, split as shell words."""
    try:
        return shlex.split(value)
    except ValueError as error:
        # shlex names the fault ("No closing quotation"), not the text.
        raise ExtraError(f"{VARIABLE} does not split as shell words: {error}") from None


def flag_name(token):
    """A token's flag, without any value attached to it: safe to print."""
    if token.startswith("--"):
        return token.split("=", 1)[0]
    if token.startswith("-") and len(token) > 1:
        return token[:2]
    return "a bare argument"


def translate(tokens):
    """(Compose service keys, flags) when Compose expresses every argument, else
    (None, the reason, naming the flag)."""
    service = {}
    flags = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        name, attached, value = token.partition("=") if token.startswith("--") else (token, "", "")
        if name in SWITCHES and not attached:
            service[SWITCHES[name]] = True
        elif name in VALUED:
            if not attached:
                if i + 1 == len(tokens):
                    return None, f"{name} has no value"
                i += 1
                value = tokens[i]
            key = VALUED[name]
            if key == "volumes" and not value.startswith("/"):
                return None, f"{name} names a volume, not a host path"
            if key == "network_mode" and value not in NETWORKS and ":" not in value:
                return None, f"{name} names a user-defined network"
            if key in LISTS:
                service.setdefault(key, []).append(value)
            else:
                service[key] = value
        else:
            return None, f"{flag_name(token)} has no Compose translation here"
        flags.append(name if name.startswith("--") else long_name(name))
        i += 1
    return service, flags


def long_name(short):
    """A short flag's long name, so that an explanation counts -v and --volume as one."""
    key = VALUED[short]
    return next(f for f, k in VALUED.items() if k == key and f.startswith("--"))


def escape(value):
    """Compose interpolates $; docker run takes the string as it is."""
    if isinstance(value, list):
        return [escape(v) for v in value]
    return value.replace("$", "$$") if isinstance(value, str) else value


def override(service):
    """The Compose file for translated arguments, for both of compose.yaml's services."""
    keys = {key: escape(value) for key, value in service.items()}
    return {"services": {"finn": keys, "notebook": keys}}


def write_override(service, directory):
    """Write the override (JSON is YAML) under a name its content determines, readable
    only by the caller, since it may hold a licence or a token; return its path."""
    text = json.dumps(override(service), indent=2, sort_keys=True) + "\n"
    path = (
        Path(directory)
        / f".finn-docker-extra-{hashlib.sha256(text.encode()).hexdigest()[:16]}.yaml"
    )
    if not path.exists() or path.read_text() != text:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
    return path


def explain(value):
    """One line on how docker/run applies FINN_DOCKER_EXTRA, without its values."""
    if not value or not value.strip():
        return f"{VARIABLE}: not set"
    tokens = split(value)
    service, detail = translate(tokens)
    if service is None:
        return (
            f"{VARIABLE}: {len(tokens)} arguments, passed through to plain `docker run` "
            f"as given, because {detail}"
        )
    counts = Counter(detail)
    flags = ", ".join(f"{f} x{n}" if n > 1 else f for f, n in counts.items())
    return (
        f"{VARIABLE}: {len(tokens)} arguments, as a Compose override "
        f"(every flag has a Compose key: {flags})"
    )


# --------------------------------------------------------------------------
# Plain docker run, from the rendered Compose definition.
# --------------------------------------------------------------------------


def delta(full, base):
    """What full (every Compose file) adds to or changes in base (docker/run's own)."""
    changes = {}
    for key, value in full.items():
        if key == "environment":
            known = base.get(key) or {}
            changed = {k: v for k, v in (value or {}).items() if k not in known or known[k] != v}
            if changed:
                changes[key] = changed
        elif key in ("volumes", "ports"):
            added = [v for v in value or [] if v not in (base.get(key) or [])]
            if added:
                changes[key] = added
        elif key not in base or base[key] != value:
            changes[key] = value
    return changes


def volume_flag(volume):
    if volume.get("type") != "bind" or set(volume.get("bind") or {}) - {"create_host_path"}:
        raise ExtraError(
            f"a {volume.get('type')} mount with options {sorted(volume.get('bind') or {})} "
            "has no `docker run --volume` form here; express it in FINN_DOCKER_EXTRA instead"
        )
    spec = f"{volume['source']}:{volume['target']}"
    return spec + ":ro" if volume.get("read_only") else spec


def port_flag(port):
    host = f"{port['host_ip']}:" if port.get("host_ip") else ""
    published = f"{port['published']}:" if port.get("published") else ""
    protocol = f"/{port['protocol']}" if port.get("protocol", "tcp") != "tcp" else ""
    return f"{host}{published}{port['target']}{protocol}"


# Keys docker/run handles itself, or that do not affect a one-off container.
HANDLED = {"image", "command", "entrypoint", "profiles"}


def service_flags(service):
    """docker run flags for a rendered Compose service (`docker compose config --format
    json`); a key with no flag here stops the launch rather than being dropped."""
    argv = []
    for key, value in service.items():
        if key in HANDLED:
            if key == "entrypoint" and value is not None:
                raise ExtraError("a Compose entrypoint has no docker run translation here")
            continue
        if key == "environment":
            for name, setting in (value or {}).items():
                # None: named without a value and unset in docker/run's environment.
                if setting is not None:
                    argv += ["--env", f"{name}={setting}"]
        elif key == "volumes":
            for volume in value or []:
                argv += ["--volume", volume_flag(volume)]
        elif key == "ports":
            for port in value or []:
                argv += ["--publish", port_flag(port)]
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
            raise ExtraError(
                f"the Compose key {key!r} has no docker run translation in docker/run; "
                f"with {VARIABLE} set to flags Compose cannot express, COMPOSE_FILE's files "
                "may only set environment, volumes, ports, hostname, user, working_dir, "
                "network_mode and init"
            )
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


def docker_run(base, full, tokens, volumes, variables, command, terminal):
    """The docker run command: docker/run's own definition (base), FINN_DOCKER_EXTRA as
    given, what COMPOSE_FILE's files add (full minus base), then docker/run's own
    --volume and --env arguments; the image and the command last."""
    added = delta(full, base)
    network, targets = claimed(tokens)
    if network and "network_mode" in added:
        raise ExtraError(
            f"both {VARIABLE} and a COMPOSE_FILE file set the network; docker run would "
            "join both"
        )
    own = {k: v for k, v in base.items() if not (k == "network_mode" and network)}
    own["volumes"] = [v for v in base.get("volumes") or [] if v.get("target") not in targets]
    if "network_mode" in added:
        own.pop("network_mode", None)
    argv = ["docker", "run", "--rm", "--pull", "never", "--interactive"]
    if terminal:
        argv.append("--tty")
    argv += service_flags(own)
    argv += tokens
    argv += service_flags(added)
    for volume in volumes:
        argv += ["--volume", volume]
    for variable in variables:
        argv += ["--env", variable]
    argv.append(full["image"])
    argv += command or (full.get("command") or [])
    return argv
