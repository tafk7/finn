# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""docker/plain_run.py: FINN's container as plain `docker run`, from a definition that
must stay compose.yaml's, with FINN_DOCKER_EXTRA appended as given."""
import pytest

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docker"))
import plain_run  # noqa: E402

pytestmark = pytest.mark.util

SECRET = "2100@licence.example.com"


# --------------------------------------------------------------------------
# The definition is compose.yaml's: compared with `docker compose config`.
# --------------------------------------------------------------------------


@pytest.fixture
def compose():
    """Skips without Docker Compose, except in GitHub's workflows, which have it: there
    this comparison must run."""
    if shutil.which("docker") and not (
        subprocess.run(["docker", "compose", "version"], capture_output=True).returncode
    ):
        return
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.fail("docker compose is missing, so compose.yaml cannot be compared")
    pytest.skip("needs docker compose")


def rendered(env, service, tools):
    files = ["compose.yaml"] + (["compose.tools.yaml"] if tools else [])
    result = subprocess.run(
        ["docker", "compose", "--project-directory", str(REPO), "--profile", "notebook"]
        + ["config", "--format", "json"],
        env={**env, "COMPOSE_FILE": os.pathsep.join(files)},
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    definition = json.loads(result.stdout)["services"][service]
    # Compose releases differ in whether they spell out the default bind option.
    for volume in definition.get("volumes", []):
        if set(volume.get("bind") or {}) <= {"create_host_path"}:
            volume["bind"] = {}
    return definition


# docker/run's variables (see its main()), and none at all: compose.yaml's defaults.
DOCKER_RUN = {
    "FINN_IMAGE": "finn:ci-0123-xrt",
    "FINN_UID": "1234",
    "FINN_GID": "5678",
    "FINN_ROOT": "/work/finn",
    "FINN_WORKDIR": "/work/project",
    "FINN_HOME": "/nfs/agent7/finn_home",
    "FINN_BUILD_DIR": "/scratch/finn_build",
}
TOOLS = {
    "FINN_XILINX_ROOT": "/proj/xbuilds/SWIP/2024.2_1113_1001/installs/lin64",
    "FINN_XILINX_PATH": "/proj/xbuilds/SWIP/2024.2_1113_1001/installs/lin64",
    "FINN_XILINX_VERSION": "2024.2",
    "XILINXD_LICENSE_FILE": SECRET,
    "LM_LICENSE_FILE": "",
    "FINN_TOOL_DIR_OVERRIDE": "/opt/finn-lsf/shim",
}


@pytest.mark.parametrize("service", ["finn", "notebook"])
@pytest.mark.parametrize(
    "variables, tools",
    [
        ({}, False),
        ({"USER": "jenkins", "FINN_HOSTNAME": "agent7"}, False),
        (DOCKER_RUN, False),
        ({**DOCKER_RUN, **TOOLS, "FINN_HOSTNAME": "agent7"}, True),
        ({**DOCKER_RUN, **TOOLS, "XILINX_VIVADO": "/tools/2025.2/Vivado"}, True),
        ({**DOCKER_RUN, "JUPYTER_PORT": "9999", "NETRON_PORT": "9998"}, False),
    ],
)
def test_the_definition_is_what_compose_renders(compose, tmp_path, service, variables, tools):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PWD": str(REPO), **variables}
    # Everything: user, mounts, environment, working directory, init and entrypoint,
    # network, host name, ports and command.
    assert plain_run.definition(env, service, tools) == rendered(env, service, tools)


def test_the_tools_need_their_install_root():
    with pytest.raises(plain_run.PlainRunError, match="FINN_XILINX_ROOT"):
        plain_run.definition({}, tools=True)


# --------------------------------------------------------------------------
# FINN_DOCKER_EXTRA: split as shell words; explained without its values.
# --------------------------------------------------------------------------


def test_quoted_words_stay_whole():
    assert plain_run.split("-e 'MSG=a b' -v \"/with space:/w s\" --network=host") == [
        "-e",
        "MSG=a b",
        "-v",
        "/with space:/w s",
        "--network=host",
    ]


def test_a_string_that_does_not_split_stops_without_showing_it():
    with pytest.raises(plain_run.PlainRunError) as error:
        plain_run.split(f"-e LIC='{SECRET}")
    assert "No closing quotation" in str(error.value) and SECRET not in str(error.value)


@pytest.mark.parametrize(
    "value, compose, expected",
    [
        (
            "",
            True,
            "FINN_DOCKER_EXTRA: not set; docker/run starts the container with docker compose",
        ),
        (
            " ",
            False,
            "FINN_DOCKER_EXTRA: not set; docker compose is absent, so docker/run starts the "
            "container with plain `docker run`",
        ),
        (
            f"-v /lsf:/lsf -e LIC={SECRET} --network=host",
            False,
            "FINN_DOCKER_EXTRA: 5 arguments, appended as given to plain `docker run` "
            "(docker compose absent)",
        ),
    ],
)
def test_the_explanation_says_how_the_container_starts_never_values(value, compose, expected):
    assert plain_run.explain(value, compose) == expected


# --------------------------------------------------------------------------
# docker run.
# --------------------------------------------------------------------------

ENV = {**DOCKER_RUN, **TOOLS, "USER": "jenkins", "PWD": "/work/finn"}


def position(argv, part):
    return next(i for i in range(len(argv)) if argv[i : i + len(part)] == part)


def test_docker_run_orders_own_definition_then_extra_then_docker_runs_arguments():
    service = plain_run.definition({**ENV, "FINN_HOSTNAME": "agent7"}, tools=True)
    extra = ["--hostname", "dsl", "--gpus", "all", "-e", "A=1"]
    argv = plain_run.docker_run(service, extra, ["/w:/w"], ["B=2"], ["pytest", "-x"], False)
    assert argv[:6] == ["docker", "run", "--rm", "--pull", "never", "--interactive"]
    assert "--tty" not in argv
    at = position(argv, extra)
    own, after = argv[:at], argv[at + len(extra) :]
    # Unset variables stay unset; an empty one is passed empty, as Compose passes it.
    assert own[6:] == [
        "--env", "USER=jenkins",
        "--env", "FINN_ROOT=/work/finn",
        "--env", "FINN_HOME=/nfs/agent7/finn_home",
        "--env", "FINN_BUILD_DIR=/scratch/finn_build",
        "--env", "FINN_XILINX_PATH=/proj/xbuilds/SWIP/2024.2_1113_1001/installs/lin64",
        "--env", "FINN_XILINX_VERSION=2024.2",
        "--env", f"XILINXD_LICENSE_FILE={SECRET}",
        "--env", "LM_LICENSE_FILE=",
        "--env", "FINN_TOOL_DIR_OVERRIDE=/opt/finn-lsf/shim",
        "--network", "bridge",
        "--platform", "linux/amd64",
        "--user", "1234:5678",
        "--volume", "/work/finn:/work/finn",
        "--volume", "/nfs/agent7/finn_home:/nfs/agent7/finn_home",
        "--volume", "/scratch/finn_build:/scratch/finn_build",
        "--volume", f"{TOOLS['FINN_XILINX_ROOT']}:{TOOLS['FINN_XILINX_ROOT']}:ro",
        "--workdir", "/work/project",
    ]  # fmt: skip
    # Later wins: docker/run's host name over the extra's; then its own arguments, the
    # image and the command.
    assert after == [
        "--hostname", "agent7",
        "--volume", "/w:/w",
        "--env", "B=2",
        "finn:ci-0123-xrt", "pytest", "-x",
    ]  # fmt: skip


def test_docker_runs_own_network_and_mounts_yield_to_the_extras():
    service = plain_run.definition(ENV, tools=True)
    root = TOOLS["FINN_XILINX_ROOT"]
    argv = plain_run.docker_run(service, [f"--volume={root}:{root}"], [], [], [], False)
    assert argv[argv.index("--network") + 1] == "bridge"  # the extra sets none
    assert f"{root}:{root}:ro" not in argv and "/work/finn:/work/finn" in argv
    argv = plain_run.docker_run(service, ["-v", "/x:/work/finn", "--net=host"], [], [], [], False)
    assert "/work/finn:/work/finn" not in argv and "--network" not in argv
    argv = plain_run.docker_run(service, ["--network", "host"], [], [], [], False)
    assert "--network" not in argv[: argv.index("host") - 1]


def test_docker_run_on_a_terminal_and_with_the_services_command():
    notebook = plain_run.definition(ENV, "notebook")
    argv = plain_run.docker_run(notebook, ["--gpus", "all"], [], [], [], terminal=True)
    assert "--tty" in argv and argv[-8:] == ["finn:ci-0123-xrt"] + notebook["command"]
    assert notebook["command"][:2] == ["jupyter", "notebook"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--publish"] == [
        "8888:8888",
        "8081:8081",
    ]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"cap_add": ["SYS_ADMIN"]}, "the service key 'cap_add' has no docker run flag"),
        ({"entrypoint": ["sh"]}, "the service key 'entrypoint' has no docker run flag"),
    ],
)
def test_a_key_without_a_flag_stops_the_launch(change, message):
    service = {**plain_run.definition(ENV), **change}
    with pytest.raises(plain_run.PlainRunError, match=message):
        plain_run.docker_run(service, [], [], [], [], False)
