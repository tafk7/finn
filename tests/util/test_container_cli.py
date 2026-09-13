"""Tests for the repository-local Docker environment entry points."""

import json
import os
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
    assert data["image_revision"].startswith("env-")
    assert (
        data["source_revision"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    assert data["source_describe"]


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
    assert data["image_revision"].startswith("env-")
    assert data["source_revision"]


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
    for rel in ("docker/finn-env",):
        lines = (REPO / rel).read_text().splitlines()
        assert len(lines) <= 30, "%s has regained implementation logic" % rel


def test_apptainer_runner_is_not_part_of_the_interface():
    assert not (REPO / "docker/run-apptainer").exists()
    assert not (REPO / "docker/finn-apptainer").exists()
    assert "export_sif ()" in BUILD.read_text()


def test_only_public_container_entrypoints_remain():
    for path in ("run-docker.sh", "docker/run-docker", "docker/run-sbx", "docker/export-sif"):
        assert not (REPO / path).exists()


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
        assert "./run-docker.sh" not in body, rel
        assert "--backend" not in body, rel


def test_native_sbx_launcher_reuses_and_preserves_state_on_failed_remove(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake = binaries / "sbx"
    fake.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["FAKE_SBX_ROOT"])
with (root / "calls").open("a") as log:
    log.write(json.dumps(args) + "\\n")
active = root / "active"
if args[:2] == ["template", "ls"]:
    print(json.dumps({"images": [{"repository": "docker.io/xilinx/finn", "tag": "sbx-env-test"}]}))
elif args == ["ls", "--json"]:
    print(json.dumps({"sandboxes": [{"name": "native-test"}] if active.exists() else []}))
elif args[:2] == ["env", "create"]:
    active.touch()
elif args[:2] == ["env", "exec"]:
    assert active.exists()
elif args[:2] == ["env", "rm"]:
    if os.environ.get("FAIL_REMOVE"):
        sys.exit(7)
    active.unlink()
else:
    sys.exit("unexpected sbx call: " + str(args))
""")
    fake.chmod(0o755)
    env = {
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "FAKE_SBX_ROOT": str(tmp_path),
        "FINN_IMAGE_REVISION": "env-test",
    }
    command = [str(RUN), "--sbx", "--name", "native-test", "--no-build"]
    for _ in range(2):
        proc = subprocess.run(command + ["--", "true"], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert sum(call[:2] == ["env", "create"] for call in calls) == 1
    assert sum(call[:2] == ["env", "exec"] for call in calls) == 2
    bundle = tmp_path / "state/finn/sbxenv/native-test/finn.sbxenv.yaml"
    assert bundle.exists()
    env["FAIL_REMOVE"] = "1"
    proc = subprocess.run(command + ["--remove"], env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert bundle.exists()
    del env["FAIL_REMOVE"]
    proc = subprocess.run(command + ["--remove"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not bundle.exists()


def test_shared_image_preparation_reuse_rebuild_and_no_build(tmp_path):
    script = r"""
set -euo pipefail
. "$1/docker/lib.sh"
finn_bake_tag () { echo xilinx/finn:test; }
finn_bake_build () { printf '%s\n' "$*" >> "$CALLS"; }
docker () { [ "$*" = 'image inspect xilinx/finn:test' ] && [ "$PRESENT" = 1 ]; }
finn_prepare_image finn "$MODE"
[ "$FINN_IMAGE" = xilinx/finn:test ]
"""
    for case, present, mode, no_build, rebuild, success, expected in [
        ("reuse", "1", "ensure", "0", "0", True, ""),
        ("missing", "0", "ensure", "0", "0", True, "finn\n"),
        ("explicit", "1", "build", "0", "0", True, "finn\n"),
        ("explicit-overrides-inherited", "1", "build", "1", "0", True, "finn\n"),
        ("rebuild", "1", "ensure", "0", "1", True, "finn --no-cache\n"),
        ("require-present", "1", "ensure", "1", "0", True, ""),
        ("require-missing", "0", "ensure", "1", "0", False, ""),
    ]:
        calls = tmp_path / case
        result = subprocess.run(
            ["bash", "-c", script, "test", str(REPO)],
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ["PATH"],
                "CALLS": str(calls),
                "PRESENT": present,
                "MODE": mode,
                "FINN_CONTAINER_NO_BUILD": no_build,
                "FINN_CONTAINER_REBUILD": rebuild,
            },
        )
        assert (result.returncode == 0) == success, (case, result.stderr)
        assert (calls.read_text() if calls.exists() else "") == expected, case


def test_sif_export_uses_caller_directory_and_prepared_image(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text("""#!/usr/bin/env python3
import json, sys
from pathlib import Path
a = sys.argv[1:]
if a[:2] == ["buildx", "bake"] and "--print" in a:
    print(json.dumps({"target": {a[-1]: {"tags": ["xilinx/finn:test"]}}}))
elif a[:2] == ["image", "inspect"]:
    pass
elif a[0] == "save":
    Path(a[a.index("-o")+1]).write_text("archive")
else:
    sys.exit("unexpected docker call: " + str(a))
""")
    apptainer = binaries / "apptainer"
    apptainer.write_text("""#!/usr/bin/env python3
import sys
from pathlib import Path
assert sys.argv[1:3] == ["build", "--force"]
assert Path(sys.argv[4].removeprefix("docker-archive://")).read_text() == "archive"
Path(sys.argv[3]).write_text("sif")
""")
    for executable in (docker, apptainer):
        executable.chmod(0o755)
    proc = subprocess.run(
        [str(BUILD), "--export-sif", "output with spaces/finn.sif"],
        cwd=tmp_path,
        env={
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "FINN_IMAGE_REVISION": "env-test",
            "FINN_CONTAINER_NO_BUILD": "1",
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "output with spaces/finn.sif").read_text() == "sif"


def test_public_image_reference_matches_bake():
    proc = invoke(BUILD, "--sbx", "--runtime", "xrt", "--print-tag")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("xilinx/finn:sbx-env-")
    assert proc.stdout.strip().endswith(".xrt")
