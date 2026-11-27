# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""docker/docker_extra.py: FINN_DOCKER_EXTRA as a Compose override, or plain
`docker run` with the arguments as given."""
import pytest

import json
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docker"))
import docker_extra  # noqa: E402

pytestmark = pytest.mark.util

SECRET = "2100@licence.example.com"


def translate(value):
    return docker_extra.translate(docker_extra.split(value))


# --------------------------------------------------------------------------
# Translation: every flag class, in each form docker run takes it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("-v /lsf:/lsf:ro --volume /a:/b", {"volumes": ["/lsf:/lsf:ro", "/a:/b"]}),
        ("--volume=/a:/b", {"volumes": ["/a:/b"]}),
        ("-e A=1 --env B --env=C=x=y", {"environment": ["A=1", "B", "C=x=y"]}),
        ("--env-file /site/env", {"env_file": ["/site/env"]}),
        ("--network host", {"network_mode": "host"}),
        ("--net=none", {"network_mode": "none"}),
        ("--network container:lsf", {"network_mode": "container:lsf"}),
        ("-h agent1 --hostname agent2", {"hostname": "agent2"}),
        ("--device /dev/xclmgmt", {"devices": ["/dev/xclmgmt"]}),
        ("--add-host lic:10.0.0.1", {"extra_hosts": ["lic:10.0.0.1"]}),
        (
            "--cap-add SYS_PTRACE --cap-drop NET_RAW",
            {"cap_add": ["SYS_PTRACE"], "cap_drop": ["NET_RAW"]},
        ),
        (
            "--group-add 100 --security-opt seccomp=unconfined",
            {"group_add": ["100"], "security_opt": ["seccomp=unconfined"]},
        ),
        ("--dns 10.0.0.2 --dns-search corp", {"dns": ["10.0.0.2"], "dns_search": ["corp"]}),
        ("--tmpfs /scratch --shm-size 8g", {"tmpfs": ["/scratch"], "shm_size": "8g"}),
        ("--ipc host --pid host", {"ipc": "host", "pid": "host"}),
        ("-l a=b --label c=d", {"labels": ["a=b", "c=d"]}),
        ("-w /work -u 1000:1000", {"working_dir": "/work", "user": "1000:1000"}),
        ("--privileged --init --read-only", {"privileged": True, "init": True, "read_only": True}),
        ("", {}),
    ],
)
def test_each_flag_class_has_its_compose_key(value, expected):
    service, _ = translate(value)
    assert service == expected


@pytest.mark.parametrize(
    "value, reason",
    [
        ("-v cache:/cache", "-v names a volume, not a host path"),
        ("--network sitenet", "--network names a user-defined network"),
        ("--gpus all", "--gpus has no Compose translation here"),
        ("-p 8888:8888", "-p has no Compose translation here"),
        ("--mount type=bind,src=/a,dst=/b", "--mount has no Compose translation here"),
        ("-it", "-i has no Compose translation here"),
        ("-v=/a:/b", "-v has no Compose translation here"),
        ("--init=false", "--init has no Compose translation here"),
        ("-v /a:/a -e", "-e has no value"),
        (f"-v /a:/a {SECRET}", "a bare argument has no Compose translation here"),
    ],
)
def test_anything_else_falls_back_naming_the_flag_not_its_value(value, reason):
    service, why = translate(value)
    assert service is None and why == reason


def test_quoted_words_stay_whole_and_dollars_reach_the_container_as_written(tmp_path):
    service, _ = translate("-e 'MSG=a b' -e \"PRICE=$5\" -v '/with space:/with space'")
    assert service == {
        "environment": ["MSG=a b", "PRICE=$5"],
        "volumes": ["/with space:/with space"],
    }
    # Compose interpolates $, so the override doubles it.
    assert docker_extra.override(service)["services"]["finn"]["environment"][1] == "PRICE=$$5"


def test_a_string_that_does_not_split_stops_without_showing_it():
    with pytest.raises(docker_extra.ExtraError) as error:
        docker_extra.split(f"-e LIC='{SECRET}")
    assert "No closing quotation" in str(error.value) and SECRET not in str(error.value)


# --------------------------------------------------------------------------
# The override file and the explanation.
# --------------------------------------------------------------------------


def test_the_override_is_named_by_its_content_and_private(tmp_path):
    service, _ = translate(f"-e LM_LICENSE_FILE={SECRET} --network host")
    path = docker_extra.write_override(service, tmp_path)
    assert path.name.startswith(".finn-docker-extra-") and path.suffix == ".yaml"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    written = json.loads(path.read_text())
    assert written["services"]["finn"] == written["services"]["notebook"] == service
    assert docker_extra.write_override(service, tmp_path) == path
    other = docker_extra.write_override({"network_mode": "none"}, tmp_path)
    assert other != path


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", "FINN_DOCKER_EXTRA: not set"),
        (
            f"-v /lsf:/lsf -v /shim:/shim -e FINN_TOOL_DIR_OVERRIDE=/shim -e LIC={SECRET} "
            "--network host",
            "FINN_DOCKER_EXTRA: 10 arguments, as a Compose override (every flag has a Compose "
            "key: --volume x2, --env x2, --network)",
        ),
        (
            f"-e LIC={SECRET} --gpus all",
            "FINN_DOCKER_EXTRA: 4 arguments, passed through to plain `docker run` as given, "
            "because --gpus has no Compose translation here",
        ),
    ],
)
def test_the_explanation_names_the_path_and_flags_never_values(value, expected):
    assert docker_extra.explain(value) == expected


# --------------------------------------------------------------------------
# Plain docker run, from rendered Compose services.
# --------------------------------------------------------------------------

BASE = {
    "command": None,
    "entrypoint": None,
    "environment": {
        "FINN_ROOT": "/finn",
        "XILINX_VIVADO": "/tools/2025.2/Vivado",
        "LM_LICENSE_FILE": None,
    },
    "image": "finn:ci-0123",
    "init": False,
    "network_mode": "bridge",
    "platform": "linux/amd64",
    "user": "1000:1000",
    "volumes": [
        {"type": "bind", "source": "/finn", "target": "/finn", "bind": {}},
        {"type": "bind", "source": "/tools", "target": "/tools", "read_only": True, "bind": {}},
    ],
    "working_dir": "/finn",
}
# compose.ci.yaml's additions: the agent's host name and caches.
FULL = {
    **BASE,
    "environment": {**BASE["environment"], "TORCH_HOME": "/finn_cache/torch"},
    "hostname": "agent7",
    "volumes": BASE["volumes"]
    + [{"type": "bind", "source": "/nfs/c", "target": "/finn_cache", "bind": {}}],
}


def test_delta_is_what_compose_files_add_or_change():
    assert docker_extra.delta(FULL, BASE) == {
        "environment": {"TORCH_HOME": "/finn_cache/torch"},
        "hostname": "agent7",
        "volumes": [{"type": "bind", "source": "/nfs/c", "target": "/finn_cache", "bind": {}}],
    }


def position(argv, part):
    return next(i for i in range(len(argv)) if argv[i : i + len(part)] == part)


def test_docker_run_orders_own_definition_then_extra_then_pipeline():
    extra = ["--network", "host", "--hostname", "dsl", "--gpus", "all", "-e", "A=1"]
    argv = docker_extra.docker_run(
        BASE, FULL, extra, ["/w:/w"], ["B=2"], ["pytest", "-x"], terminal=False
    )
    assert argv[:6] == ["docker", "run", "--rm", "--pull", "never", "--interactive"]
    assert "--tty" not in argv
    at = position(argv, extra)
    own, after = argv[:at], argv[at + len(extra) :]
    # docker/run's own definition first; unset variables stay unset, and the network
    # yields to the DSL's (docker run would join both).
    assert own[6:] == [
        "--env", "FINN_ROOT=/finn",
        "--env", "XILINX_VIVADO=/tools/2025.2/Vivado",
        "--platform", "linux/amd64",
        "--user", "1000:1000",
        "--volume", "/finn:/finn",
        "--volume", "/tools:/tools:ro",
        "--workdir", "/finn",
    ]  # fmt: skip
    # Later wins: the pipeline's host name over the DSL's; docker/run's own
    # arguments, the image and the command last.
    assert after == [
        "--env", "TORCH_HOME=/finn_cache/torch",
        "--volume", "/nfs/c:/finn_cache",
        "--hostname", "agent7",
        "--volume", "/w:/w",
        "--env", "B=2",
        "finn:ci-0123", "pytest", "-x",
    ]  # fmt: skip


def test_docker_runs_own_network_and_mounts_yield_to_the_extras():
    argv = docker_extra.docker_run(
        BASE, BASE, ["--gpus", "all", "--volume=/site/tools:/tools"], [], [], [], False
    )
    assert argv[argv.index("--network") + 1] == "bridge"  # the DSL sets none
    assert "/tools:/tools:ro" not in argv and "/finn:/finn" in argv
    argv = docker_extra.docker_run(
        BASE, BASE, ["--gpus", "all", "-v", "/x:/finn"], [], [], [], False
    )
    assert "/finn:/finn" not in argv
    with pytest.raises(docker_extra.ExtraError, match="both FINN_DOCKER_EXTRA and a COMPOSE_FILE"):
        docker_extra.docker_run(
            BASE,
            {**BASE, "network_mode": "host"},
            ["--gpus", "all", "--net=host"],
            [],
            [],
            [],
            False,
        )
    # A COMPOSE_FILE network replaces docker/run's own.
    argv = docker_extra.docker_run(
        BASE, {**BASE, "network_mode": "host"}, ["--gpus", "all"], [], [], [], False
    )
    assert argv.count("--network") == 1 and argv[argv.index("--network") + 1] == "host"


def test_docker_run_on_a_terminal_and_with_the_services_command():
    notebook = {
        **BASE,
        "command": ["jupyter", "notebook"],
        "ports": [{"target": 8888, "published": "8888", "protocol": "tcp", "mode": "ingress"}],
    }
    argv = docker_extra.docker_run(notebook, notebook, ["--gpus", "all"], [], [], [], terminal=True)
    assert "--tty" in argv and argv[-3:] == ["finn:ci-0123", "jupyter", "notebook"]
    assert argv[argv.index("--publish") + 1] == "8888:8888"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"cap_add": ["SYS_ADMIN"]}, "the Compose key 'cap_add' has no docker run translation"),
        ({"entrypoint": ["sh"]}, "entrypoint has no docker run translation"),
        (
            {"volumes": [{"type": "volume", "source": "v", "target": "/v", "volume": {}}]},
            "a volume mount",
        ),
    ],
)
def test_compose_keys_without_a_flag_stop_the_launch(change, message):
    with pytest.raises(docker_extra.ExtraError) as error:
        docker_extra.docker_run(BASE, {**FULL, **change}, ["--gpus", "all"], [], [], [], False)
    assert message in str(error.value)
