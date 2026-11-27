# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""docker/run: the variables and command it hands to `docker compose run`."""
import pytest

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

pytestmark = pytest.mark.util

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "docker" / "run"


def dry_run(tmp_path, *args, env=None, cwd=None):
    """docker/run --dry-run: (variables, compose argv), or the error text."""
    base = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "FINN_BUILD_DIR": str(tmp_path / "build"),
    }
    result = subprocess.run(
        [sys.executable, RUN, "--dry-run", *args],
        env={**base, **(env or {})},
        cwd=cwd or REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return result.stderr
    *assignments, command = result.stdout.strip().splitlines()
    variables = dict(line.split("=", 1) for line in assignments)
    return {k: shlex.split(v)[0] if v else "" for k, v in variables.items()}, shlex.split(command)


def release(tmp_path, layout="new", name="2025.2"):
    """A fake AMD install: <root>/<release>/Vivado, or <root>/Vivado/<release> before 2025.1."""
    root = tmp_path / "Xilinx"
    vivado = root / name / "Vivado" if layout == "new" else root / "Vivado" / name
    vivado.mkdir(parents=True)
    (vivado / "settings64.sh").write_text("")
    return root, vivado


def files(variables):
    return [Path(f).name for f in variables["COMPOSE_FILE"].split(os.pathsep)]


def volumes(argv):
    return [argv[i + 1] for i, token in enumerate(argv) if token == "--volume"]


def test_without_tools_only_the_container_is_defined(tmp_path):
    variables, argv = dry_run(tmp_path, "--", "pytest", "-x")
    assert files(variables) == ["compose.yaml"]
    assert variables["FINN_ROOT"] == str(REPO)
    assert variables["FINN_WORKDIR"] == str(REPO)
    assert variables["FINN_HOME"] == str(tmp_path / "home/.finn")
    assert variables["FINN_UID"] == str(os.getuid())
    assert variables["FINN_IMAGE"] == "finn:dev"
    assert argv[:2] == ["docker", "compose"] and argv[argv.index("run") :][:4] == [
        "run",
        "--rm",
        "--pull",
        "never",
    ]
    assert argv[argv.index("--project-directory") + 1] == str(REPO)
    assert argv[-3:] == ["finn", "pytest", "-x"]
    assert (tmp_path / "home/.finn").is_dir() and (tmp_path / "build").is_dir()


@pytest.mark.parametrize("layout", ["new", "old"])
def test_a_sourced_release_mounts_its_install_root(tmp_path, layout):
    root, vivado = release(tmp_path, layout)
    variables, _ = dry_run(tmp_path, env={"XILINX_VIVADO": str(vivado)})
    assert files(variables) == ["compose.yaml", "compose.tools.yaml"]
    assert variables["FINN_XILINX_ROOT"] == str(root)


def test_no_tools_and_fpga(tmp_path):
    _, vivado = release(tmp_path)
    variables, _ = dry_run(tmp_path, "--no-tools", env={"XILINX_VIVADO": str(vivado)})
    assert files(variables) == ["compose.yaml"] and "FINN_XILINX_ROOT" not in variables
    error = dry_run(tmp_path, "--fpga")
    assert "--fpga needs the AMD tools" in error and "settings64.sh" in error
    assert "not allowed with" in dry_run(tmp_path, "--fpga", "--no-tools")


def test_licence_files_and_platforms_are_mounted_servers_are_not(tmp_path):
    _, vivado = release(tmp_path)
    licence = tmp_path / "node.lic"
    licence.write_text("")
    platforms = tmp_path / "platforms"
    platforms.mkdir()
    env = {
        "XILINX_VIVADO": str(vivado),
        "XILINXD_LICENSE_FILE": f"2100@licence-server:{licence}",
        "PLATFORM_REPO_PATHS": str(platforms),
    }
    _, argv = dry_run(tmp_path, env=env)
    # a worktree checkout also mounts its git directories
    mounted = [v for v in volumes(argv) if "/.git" not in v]
    assert mounted == [f"{licence}:{licence}:ro", f"{platforms}:{platforms}:ro"]


def test_a_workspace_outside_the_checkout_is_mounted(tmp_path):
    work = tmp_path / "project"
    work.mkdir()
    variables, argv = dry_run(tmp_path, cwd=work)
    assert variables["FINN_WORKDIR"] == str(work)
    assert f"{work}:{work}" in volumes(argv)


def test_images_runtimes_and_extra_compose_files(tmp_path):
    variables, _ = dry_run(tmp_path, "--runtime", "xrt")
    assert variables["FINN_IMAGE"] == "finn:dev-xrt"
    variables, _ = dry_run(tmp_path, "--release")
    assert variables["FINN_IMAGE"] == "finn:release"
    variables, _ = dry_run(tmp_path, env={"FINN_IMAGE": "site/finn:abc", "COMPOSE_FILE": "ci.yaml"})
    assert variables["FINN_IMAGE"] == "site/finn:abc"
    assert files(variables) == ["compose.yaml", "ci.yaml"]


def test_notebook_publishes_its_ports_and_takes_no_command(tmp_path):
    _, argv = dry_run(tmp_path, "--notebook")
    assert "--service-ports" in argv and argv[-1] == "notebook"
    assert "--notebook takes no command" in dry_run(tmp_path, "--notebook", "--", "bash")


@pytest.mark.skipif(
    not shutil.which("docker")
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode,
    reason="needs docker compose",
)
def test_compose_renders_mirrored_paths_and_read_only_tools(tmp_path):
    root, vivado = release(tmp_path)
    variables, _ = dry_run(
        tmp_path, env={"XILINX_VIVADO": str(vivado), "XILINXD_LICENSE_FILE": "2100@s"}
    )
    rendered = subprocess.run(
        ["docker", "compose", "--project-directory", str(REPO), "config", "--format", "json"],
        env={
            **os.environ,
            **variables,
            "XILINX_VIVADO": str(vivado),
            "XILINXD_LICENSE_FILE": "2100@s",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    finn = json.loads(rendered.stdout)["services"]["finn"]
    mounts = {(v["source"], v["target"], v.get("read_only", False)) for v in finn["volumes"]}
    for path in (str(REPO), variables["FINN_HOME"], variables["FINN_BUILD_DIR"]):
        assert (path, path, False) in mounts
    assert (str(root), str(root), True) in mounts
    assert finn["environment"]["XILINX_VIVADO"] == str(vivado)
    assert finn["environment"]["XILINXD_LICENSE_FILE"] == "2100@s"
    # Unset AMD variables stay unset in the container.
    assert finn["environment"].get("PLATFORM_REPO_PATHS") is None
    assert finn["user"] == f"{os.getuid()}:{os.getgid()}"
    assert finn["working_dir"] == str(REPO)


def test_a_worktrees_git_directories_are_mounted(tmp_path):
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if common.returncode != 0 or Path(common.stdout.strip()).is_relative_to(REPO.resolve()):
        pytest.skip("not a worktree checkout")
    _, argv = dry_run(tmp_path)
    git_dir = common.stdout.strip()
    assert f"{git_dir}:{git_dir}" in volumes(argv)
