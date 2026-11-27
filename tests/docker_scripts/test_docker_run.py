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


def dry_run(tmp_path, *args, env=None, cwd=None, run=RUN):
    """docker/run --dry-run: (variables, compose argv), or the error text."""
    base = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "FINN_BUILD_DIR": str(tmp_path / "build"),
    }
    result = subprocess.run(
        [sys.executable, run, "--dry-run", *args],
        env={**base, **(env or {})},
        cwd=cwd or run.parents[1],
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


def test_a_release_selected_through_a_symlink_is_mounted_at_both_paths(tmp_path):
    root, vivado = release(tmp_path)
    alias = tmp_path / "site-tools"
    alias.symlink_to(root, target_is_directory=True)
    variables, argv = dry_run(tmp_path, env={"XILINX_VIVADO": str(alias / "2025.2/Vivado")})
    assert variables["FINN_XILINX_ROOT"] == str(root)
    assert f"{alias}:{alias}:ro" in volumes(argv)


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


def worktree(tmp_path, common):
    """A worktree of a repository at `common` holding docker/run: its launcher, and the
    git directories it must mount."""
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@localhost"]
    for source in (RUN, RUN.parent / "docker_extra.py", REPO / "src/finn/util/paths.py"):
        target = common / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    subprocess.run([*git, "init", "--quiet"], cwd=common, check=True)
    subprocess.run([*git, "add", "."], cwd=common, check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", "fixture"], cwd=common, check=True)
    checkout = tmp_path / "checkout"
    subprocess.run(
        [*git, "worktree", "add", "--quiet", "--detach", checkout], cwd=common, check=True
    )
    return checkout / "docker/run", [common / ".git", common / ".git/worktrees/checkout"]


def test_git_directories_with_spaces_are_mounted_whole(tmp_path):
    run, git_dirs = worktree(tmp_path, tmp_path / "repository with spaces")
    _, argv = dry_run(tmp_path, run=run)
    assert [v for v in volumes(argv) if "/.git" in v] == [f"{d}:{d}" for d in git_dirs]


def test_a_host_without_git_mounts_no_git_directories(tmp_path):
    run, _ = worktree(tmp_path, tmp_path / "repository")
    (tmp_path / "bin").mkdir()
    _, argv = dry_run(tmp_path, run=run, env={"PATH": str(tmp_path / "bin")})
    assert not [v for v in volumes(argv) if "/.git" in v]


# --------------------------------------------------------------------------
# A job's run-docker.sh variables: FINN_XILINX_PATH/VERSION and FINN_DOCKER_EXTRA.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["new", "old"])
def test_a_jobs_install_root_and_release_bring_the_tools_in(tmp_path, layout):
    root, _ = release(tmp_path, layout)
    job = {"FINN_XILINX_PATH": str(root), "FINN_XILINX_VERSION": "2025.2"}
    variables, argv = dry_run(tmp_path, "--fpga", env=job)
    assert files(variables) == ["compose.yaml", "compose.tools.yaml"]
    assert variables["FINN_XILINX_ROOT"] == str(root)
    # A symlinked root is mounted as the job names it, too.
    alias = tmp_path / "site-tools"
    alias.symlink_to(root, target_is_directory=True)
    variables, argv = dry_run(tmp_path, env={**job, "FINN_XILINX_PATH": str(alias)})
    assert variables["FINN_XILINX_ROOT"] == str(root)
    assert f"{alias}:{alias}:ro" in volumes(argv)


def test_xilinx_vivado_wins_and_a_missing_install_root_stops(tmp_path):
    root, vivado = release(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    variables, _ = dry_run(
        tmp_path,
        env={
            "XILINX_VIVADO": str(vivado),
            "FINN_XILINX_PATH": str(other),
            "FINN_XILINX_VERSION": "x",
        },
    )
    assert variables["FINN_XILINX_ROOT"] == str(root)
    error = dry_run(
        tmp_path, env={"FINN_XILINX_PATH": str(tmp_path / "none"), "FINN_XILINX_VERSION": "2025.2"}
    )
    assert "FINN_XILINX_PATH=" in error and "is not a directory" in error
    # Only the path, or only the release, is not enough.
    error = dry_run(tmp_path, "--fpga", env={"FINN_XILINX_PATH": str(root)})
    assert "--fpga needs the AMD tools" in error


def test_compose_compatible_extra_becomes_an_override_before_compose_file(tmp_path):
    shim = tmp_path / "shim"
    extra = f"-v {shim}:{shim} -e FINN_TOOL_DIR_OVERRIDE={shim} -e LM_LICENSE_FILE=2100@lic"
    extra += " --network host"
    variables, argv = dry_run(
        tmp_path, env={"FINN_DOCKER_EXTRA": extra, "COMPOSE_FILE": "site.yaml"}
    )
    names = files(variables)
    assert names[0] == "compose.yaml" and names[-1] == "site.yaml"
    assert names[1].startswith(".finn-docker-extra-") and len(names) == 3
    override = Path(variables["COMPOSE_FILE"].split(os.pathsep)[1])
    assert override.parent == tmp_path / "build"
    assert json.loads(override.read_text())["services"]["finn"] == {
        "environment": [f"FINN_TOOL_DIR_OVERRIDE={shim}", "LM_LICENSE_FILE=2100@lic"],
        "network_mode": "host",
        "volumes": [f"{shim}:{shim}"],
    }
    assert argv[:2] == ["docker", "compose"]


def test_explain_extra_runs_nothing_and_shows_no_values(tmp_path):
    def explain(value):
        result = subprocess.run(
            [sys.executable, RUN, "--explain-extra"],
            env={"PATH": os.environ["PATH"], "FINN_DOCKER_EXTRA": value},
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr

    code, out = explain("-e LIC=2100@secret --network host")
    assert code == 0 and "Compose override" in out and "secret" not in out
    code, out = explain("-e LIC=2100@secret --gpus all")
    assert code == 0 and "plain `docker run`" in out and "--gpus" in out and "secret" not in out
    code, out = explain("-e 'LIC=2100@secret")
    assert code == 1 and "does not split as shell words" in out and "secret" not in out


@pytest.mark.skipif(
    not shutil.which("docker")
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode,
    reason="needs docker compose",
)
def test_other_extra_launches_plain_docker_run_from_the_same_definition(tmp_path):
    # The pipeline's compose.ci.yaml: the agent's caches and host name.
    caches = tmp_path / "caches"
    caches.mkdir()
    root, vivado = release(tmp_path)
    extra = "--hostname dsl --network host --ulimit nofile=65536:65536 -e LIC=2100@lic"
    variables, argv = dry_run(
        tmp_path,
        "--fpga",
        "--",
        "pytest",
        env={
            "FINN_DOCKER_EXTRA": extra,
            "XILINX_VIVADO": str(vivado),
            "COMPOSE_FILE": str(REPO / "ci/compose.ci.yaml"),
            "FINN_CI_CACHES": str(caches),
            "NODE_NAME": "agent7",
        },
    )
    assert not any(Path(f).name.startswith(".finn-docker-extra-") for f in files(variables))
    assert argv[:2] == ["docker", "run"] and argv[-2:] == ["finn:dev", "pytest"]
    at = argv.index("--ulimit") - 4
    assert argv[at : at + 8] == shlex.split(extra)
    own, after = argv[:at], argv[at + 8 :]
    assert f"{REPO}:{REPO}" in own and f"{root}:{root}:ro" in own
    assert "--network" not in own  # the DSL's network replaces docker/run's
    assert own[own.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert f"XILINX_VIVADO={vivado}" in own
    # The pipeline's host name and caches come after the DSL's flags, as upstream's did.
    assert after[after.index("--hostname") + 1] == "agent7"
    assert f"{caches}:/finn_cache" in after and "TORCH_HOME=/finn_cache/torch" in after
