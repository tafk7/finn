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
    assert data["runner"] == "docker"
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
    proc = invoke(RUN, "--sbx", "--name", "agent-1", "--runtime", "slash", "--print")
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["runner"] == "sbx"
    assert data["bake_target"] == "finn-sbx-runtime"
    assert data["name"] == "agent-1"


def test_name_selects_a_docker_container_name():
    for option in ("-n", "--name"):
        proc = invoke(RUN, option, "docker-test", "--print")
        assert proc.returncode == 0, proc.stderr
        data = assignments(proc.stdout)
        assert data["runner"] == "docker"
        assert data["name"] == "docker-test"


def test_backend_option_is_removed():
    proc = invoke(RUN, "--backend", "sbx", "--print")
    assert proc.returncode == 2
    assert "Unknown option: --backend" in proc.stderr


def test_build_defaults_to_the_docker_image():
    proc = invoke(BUILD, "--print")
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["artifact"] == "docker"
    assert data["bake_target"] == "finn"
    assert data["output"] == ""


def test_build_can_prepare_the_sbx_variant():
    proc = invoke(BUILD, "--sbx", "--runtime", "slash", "--print")
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["artifact"] == "sbx"
    assert data["bake_target"] == "finn-sbx-runtime"


def test_build_can_export_a_sif():
    proc = invoke(BUILD, "--runtime", "xrt", "--export-sif", "out/finn.sif", "--print")
    assert proc.returncode == 0, proc.stderr
    data = assignments(proc.stdout)
    assert data["artifact"] == "sif"
    assert data["bake_target"] == "finn-xrt"
    assert data["runtimes"] == "xrt"
    assert data["output"] == "out/finn.sif"


def test_build_rejects_sbx_sif_combination():
    proc = invoke(BUILD, "--sbx", "--export-sif", "finn.sif")
    assert proc.returncode == 2
    assert "mutually exclusive" in proc.stderr


def test_build_requires_a_sif_output_path():
    proc = invoke(BUILD, "--export-sif=")
    assert proc.returncode == 2
    assert "needs a path" in proc.stderr


def test_build_rejects_a_command():
    proc = invoke(BUILD, "--", "pytest")
    assert proc.returncode == 2
    assert "does not accept a command" in proc.stderr


def test_build_rejects_runtime_only_options():
    for option in ("--fpga", "--deps", "--name"):
        proc = invoke(BUILD, option, "value")
        assert proc.returncode == 2
        assert "Unknown option" in proc.stderr


def test_notebook_is_docker_only():
    proc = invoke(RUN, "--sbx", "--notebook", "--print")
    assert proc.returncode == 2
    assert "only with Docker" in proc.stderr


def test_remove_is_sbx_only():
    proc = invoke(RUN, "--remove", "--print")
    assert proc.returncode == 2
    assert "only with --sbx" in proc.stderr


def test_rebuild_and_no_build_are_mutually_exclusive():
    proc = invoke(RUN, "--rebuild", "--no-build", "--print")
    assert proc.returncode == 2
    assert "mutually exclusive" in proc.stderr


def test_legacy_user_entrypoints_remain_thin():
    for rel in ("docker/finn-sbx", "docker/finn-env"):
        lines = (REPO / rel).read_text().splitlines()
        assert len(lines) <= 30, "%s has regained implementation logic" % rel


def test_apptainer_runner_is_not_part_of_the_interface():
    assert not (REPO / "docker/run-apptainer").exists()
    assert not (REPO / "docker/finn-apptainer").exists()
    assert (REPO / "docker/export-sif").is_file()


def test_run_docker_is_only_a_thin_jenkins_bridge():
    path = REPO / "run-docker.sh"
    lines = path.read_text().splitlines()
    assert len(lines) <= 12
    assert "Jenkins compatibility bridge" in lines[1]


def test_user_documentation_does_not_advertise_retired_launchers():
    user_docs = (
        "README.md",
        "docker/README.md",
        "docs/finn/getting_started.rst",
        "tutorials/fpga_flow/README.md",
        "notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb",
    )
    for rel in user_docs:
        body = (REPO / rel).read_text()
        assert "run-docker.sh" not in body, rel
        assert "--backend" not in body, rel
