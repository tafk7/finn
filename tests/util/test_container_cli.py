"""Tests for the repository-local Docker environment entry points."""

from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "docker/run"
BUILD = REPO / "docker/build"


def invoke(path, *args):
    return subprocess.run([str(path), *args], cwd=REPO, capture_output=True, text=True)


def assignments(output):
    return dict(line.split("=", 1) for line in output.splitlines())


def test_default_is_the_docker_dev_environment():
    proc = invoke(RUN, "--print")
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["backend"] == "docker"
    assert data["tier"] == "dev"
    assert data["runtimes"] == ""
    assert data["deps"] == "frozen"
    assert data["operation"] == "run"


def test_common_options_are_normalized():
    proc = invoke(
        RUN,
        "--fpga",
        "--runtime",
        "xrt",
        "--runtime=slash",
        "--runtime",
        "xrt",
        "--deps",
        "live",
        "--print",
        "--",
        "pytest",
        "-m",
        "util",
    )
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["tier"] == "build"
    assert data["runtimes"] == "slash,xrt"
    assert data["deps"] == "live"
    assert data["bake_target"] == "finn-runtime"
    assert "pytest" in data["command"]


def test_sbx_uses_the_sbx_image_variant():
    proc = invoke(RUN, "--backend", "sbx", "--name", "agent-1", "--runtime", "slash", "--print")
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["backend"] == "sbx"
    assert data["bake_target"] == "finn-sbx-runtime"
    assert data["name"] == "agent-1"


def test_name_selects_a_docker_container_name():
    for option in ("-n", "--name"):
        proc = invoke(RUN, option, "docker-test", "--print")
        assert proc.returncode == 0, proc.stderr
        data = assignments(proc.stdout)
        assert data["backend"] == "docker"
        assert data["name"] == "docker-test"


def test_name_is_not_supported_by_apptainer():
    proc = invoke(RUN, "--backend", "apptainer", "--name", "nope", "--print")
    assert proc.returncode == 2
    assert "not Apptainer" in proc.stderr


def test_apptainer_without_fpga_reports_host_capabilities():
    proc = invoke(RUN, "--backend", "apptainer", "--print")
    assert proc.returncode == 0, proc.stderr
    assert assignments(proc.stdout)["tier"] == "auto"


def test_build_is_the_prepare_operation():
    proc = invoke(BUILD, "--backend", "apptainer", "--runtime", "xrt", "--print")
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["operation"] == "prepare"
    assert data["backend"] == "apptainer"
    assert data["runtimes"] == "xrt"


def test_build_rejects_a_command():
    proc = invoke(BUILD, "--", "pytest")
    assert proc.returncode == 2
    assert "does not accept a command" in proc.stderr


def test_build_rejects_runtime_only_fpga_grant():
    proc = invoke(BUILD, "--fpga")
    assert proc.returncode == 2
    assert "run-time grant" in proc.stderr


def test_build_rejects_runtime_dependency_mode():
    proc = invoke(BUILD, "--deps", "live")
    assert proc.returncode == 2
    assert "does not apply to docker/build" in proc.stderr


def test_build_rejects_runtime_instance_name():
    proc = invoke(BUILD, "--backend", "sbx", "--name", "agent-1")
    assert proc.returncode == 2
    assert "does not apply to docker/build" in proc.stderr


def test_invalid_backend_is_rejected():
    proc = invoke(RUN, "--backend", "unknown", "--print")
    assert proc.returncode == 2
    assert "Unknown backend" in proc.stderr


def test_notebook_is_docker_only():
    proc = invoke(RUN, "--backend", "sbx", "--notebook", "--print")
    assert proc.returncode == 2
    assert "only with the Docker backend" in proc.stderr


def test_rebuild_and_no_build_are_mutually_exclusive():
    proc = invoke(RUN, "--rebuild", "--no-build", "--print")
    assert proc.returncode == 2
    assert "mutually exclusive" in proc.stderr


def test_legacy_user_entrypoints_remain_thin():
    for rel in ("docker/finn-sbx", "docker/finn-apptainer", "docker/finn-env"):
        lines = (REPO / rel).read_text().splitlines()
        assert len(lines) <= 30, "%s has regained implementation logic" % rel


def test_run_docker_is_only_a_thin_jenkins_bridge():
    path = REPO / "run-docker.sh"
    lines = path.read_text().splitlines()
    assert len(lines) <= 12
    assert "Jenkins compatibility bridge" in lines[1]


def test_user_documentation_does_not_advertise_run_docker():
    user_docs = (
        "README.md",
        "docker/README.md",
        "docs/finn/getting_started.rst",
        "tutorials/fpga_flow/README.md",
        "notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb",
    )
    for rel in user_docs:
        assert "run-docker.sh" not in (REPO / rel).read_text(), rel
