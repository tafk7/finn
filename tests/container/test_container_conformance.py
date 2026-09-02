"""Runtime conformance tests for FINN's container artifacts and runners.

These tests intentionally exercise real Docker, sbx and SIF behavior.
Unavailable runtimes or host resources are reported as pytest skips. Static
resolver behavior belongs in ``tests/util/test_container_config.py``.
"""

from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


REPO = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO / "docker"
FINN_ENV = DOCKER_DIR / "config"
sys.path.insert(0, str(DOCKER_DIR))
import config as finn_env  # noqa: E402

pytestmark = pytest.mark.container


def run(argv, *, env=None, timeout=600, check=False, cwd=REPO):
    """Run one command with captured text output."""
    proc = subprocess.run(
        [str(arg) for arg in argv],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode:
        pytest.fail(
            "command failed (%d): %s\nstdout:\n%s\nstderr:\n%s"
            % (proc.returncode, " ".join(map(str, argv)), proc.stdout, proc.stderr)
        )
    return proc


def bake_config(target, *, runtimes="", git_describe="CONFORMANCE"):
    env = dict(os.environ)
    env.update({"FINN_RUNTIMES": runtimes, "GIT_DESCRIBE": git_describe})
    proc = run(["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--print", target], env=env)
    if proc.returncode:
        pytest.fail(proc.stderr or proc.stdout)
    start = proc.stdout.find("{")
    assert start >= 0, proc.stdout
    return json.loads(proc.stdout[start:])


def bake_tag(target="finn", *, runtimes="", git_describe="CONFORMANCE"):
    return bake_config(target, runtimes=runtimes, git_describe=git_describe)["target"][target][
        "tags"
    ][0]


def require_command(name):
    if not shutil.which(name):
        pytest.skip("%s is not installed" % name)


@pytest.fixture(scope="session")
def docker_daemon():
    require_command("docker")
    if run(["docker", "info"], timeout=30).returncode:
        pytest.skip("Docker daemon is unavailable")
    return True


@lru_cache(maxsize=None)
def ensure_image(target="finn", runtimes=""):
    tag = bake_tag(target, runtimes=runtimes)
    if run(["docker", "image", "inspect", tag], timeout=30).returncode:
        env = dict(os.environ)
        env.update({"FINN_RUNTIMES": runtimes, "GIT_DESCRIBE": "CONFORMANCE"})
        run(
            ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--load", target],
            env=env,
            timeout=3600,
            check=True,
        )
    return tag


def resolved(tier, backend="docker", policy="auto", env=None):
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    command = [sys.executable, FINN_ENV, "inspect", "--tier", tier]
    if backend == "sbx":
        command.append("--sbx")
    else:
        assert backend == "docker"
    command.extend(["--workspace-policy", policy])
    proc = run(command, env=child_env)
    if proc.returncode:
        pytest.fail(proc.stderr)
    return json.loads(proc.stdout)


def compose_override(tier, service, policy="auto", env=None):
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    proc = run(
        [
            sys.executable,
            FINN_ENV,
            "compose",
            "--tier",
            tier,
            "--workspace-policy",
            policy,
            "--service",
            service,
        ],
        env=child_env,
    )
    if proc.returncode:
        pytest.fail(proc.stderr)
    return json.loads(proc.stdout)


def compose_run(override, service, command, tag, tmp_path, env=None, timeout=600):
    override_path = tmp_path / "compose.override.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(json.dumps(override))
    child_env = dict(os.environ)
    child_env.update({"FINN_IMAGE": tag, "FINN_RUNTIMES": ""})
    if env:
        child_env.update(env)
    return run(
        [
            "docker",
            "compose",
            "-f",
            REPO / "compose.yaml",
            "-f",
            override_path,
            "run",
            "--rm",
            service,
        ]
        + list(command),
        env=child_env,
        timeout=timeout,
    )


def test_01_supported_targets_build(docker_daemon):
    """Every target in Bake's supported group builds."""
    config = bake_config("supported")
    for target in sorted(config["target"]):
        ensure_image(target)


def test_02_dev_runs_with_only_workspace_and_scratch(docker_daemon):
    """The portable dev contract needs no toolchain, licence or special egress."""
    data = resolved(
        "dev",
        env={"FINN_XILINX_PATH": "/opt/Xilinx", "XILINXD_LICENSE_FILE": "2100@example.invalid"},
    )
    assert data["mounts"] == []
    assert data["egress"] == []
    assert not any("XILINX" in key or "LICENSE" in key for key in data["env"])
    tag = ensure_image("finn")
    proc = run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            "%s:/workspace/finn" % REPO,
            "-w",
            "/workspace/finn",
            "--user",
            "%d:%d" % (os.getuid(), os.getgid()),
            "-e",
            "FINN_ROOT=/workspace/finn",
            tag,
            "python",
            "-c",
            "import finn",
        ],
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr


def test_03_fresh_mirrored_docker_run(docker_daemon):
    """A bare docker run works with the historical mirrored workspace policy."""
    tag = ensure_image("finn")
    proc = run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            "%s:%s" % (REPO, REPO),
            "-w",
            REPO,
            "--user",
            "%d:%d" % (os.getuid(), os.getgid()),
            "-e",
            "FINN_ROOT=%s" % REPO,
            tag,
            "python",
            "-c",
            "import finn",
        ],
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr


def test_04_bare_docker_exec(docker_daemon):
    """docker exec reaches Python and vendor tools without running ENTRYPOINT."""
    name = "finn-conformance-%d" % os.getpid()
    have_xilinx = os.path.isdir(os.environ.get("FINN_XILINX_PATH", ""))
    data = resolved("build" if have_xilinx else "dev")
    tag = ensure_image("finn")
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-v",
        "%s:%s" % (REPO, REPO),
        "-w",
        REPO,
        "--user",
        "%d:%d" % (os.getuid(), os.getgid()),
        "-e",
        "FINN_ROOT=%s" % REPO,
    ]
    for mount in data["mounts"]:
        args.extend(["-v", "%s:%s:%s" % (mount["source"], mount["target"], mount["mode"])])
    for key, value in data["env"].items():
        args.extend(["-e", "%s=%s" % (key, value)])
    args.extend([tag, "sleep", "infinity"])
    try:
        run(args, timeout=180, check=True)
        time.sleep(2)
        run(["docker", "exec", name, "python", "-c", "import finn"], timeout=180, check=True)
        if have_xilinx:
            run(["docker", "exec", name, "vivado", "-version"], timeout=300, check=True)
    finally:
        run(["docker", "rm", "-f", name], timeout=60)


def test_05_bare_sbx_exec(docker_daemon):
    """A stock sbx exec works after the wrapper provisions the sandbox."""
    require_command("sbx")
    name = "finn-conformance"
    child_env = dict(os.environ)
    command = [REPO / "docker/run", "--sbx", "--name", name]
    if os.environ.get("FINN_XILINX_PATH"):
        command.append("--fpga")
    command.extend(["--", "true"])
    try:
        run(
            ["timeout", "2400", *command],
            env=child_env,
            timeout=2450,
            check=True,
        )
        run(
            ["timeout", "180", "sbx", "exec", name, "python", "-c", "import finn"],
            timeout=200,
            check=True,
        )
        if "--fpga" in command:
            run(
                ["timeout", "300", "sbx", "exec", name, "vivado", "-version"],
                timeout=320,
                check=True,
            )
    finally:
        if os.environ.get("KEEP_SANDBOX") != "1":
            run(
                [REPO / "docker/run", "--sbx", "--name", name, "--remove"],
                timeout=120,
            )


def test_06_resolved_toolchain_mounts_are_read_only(docker_daemon):
    """Every host capability mount is declared read-only and enforced as such."""
    root = os.environ.get("FINN_XILINX_PATH")
    if not root or not os.path.isdir(root):
        pytest.skip("FINN_XILINX_PATH is not configured")
    data = resolved("build")
    assert data["mounts"]
    assert all(mount["mode"] == "ro" for mount in data["mounts"])
    tag = ensure_image("finn")
    proc = run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            "%s:%s:ro" % (root, root),
            tag,
            "sh",
            "-c",
            "touch %s/.conformance-write" % root,
        ],
        timeout=180,
    )
    assert proc.returncode != 0


def test_07_sbx_privilege_is_visible_in_the_image(docker_daemon):
    """Only the sbx image grants passwordless in-container root."""
    generic = ensure_image("finn")
    sbx = ensure_image("finn-sbx")
    assert (
        run(["docker", "run", "--rm", generic, "sh", "-c", "sudo -n true"], timeout=180).returncode
        != 0
    )
    assert (
        run(["docker", "run", "--rm", sbx, "sh", "-c", "sudo -n true"], timeout=180).returncode == 0
    )


@pytest.mark.parametrize("policy", ["fixed", "mirror"])
def test_08_compose_handles_awkward_workspace_paths(docker_daemon, tmp_path, policy):
    """The generated Compose override safely carries paths containing spaces."""
    workspace = tmp_path / "finn conformance"
    (workspace / "src/finn").mkdir(parents=True)
    env = {"FINN_ROOT": str(workspace), "FINN_HOST_BUILD_DIR": str(tmp_path / ("build-" + policy))}
    override = compose_override("dev", "dev", policy=policy, env=env)
    tag = ensure_image("finn")
    proc = compose_run(
        override,
        "dev",
        ["python", "-c", "import os; print(os.getcwd())"],
        tag,
        tmp_path / policy,
        env=env,
        timeout=180,
    )
    expected = str(workspace) if policy == "mirror" else finn_env.FIXED_WORKSPACE
    assert proc.returncode == 0, proc.stderr
    assert expected in proc.stdout


def test_09_node_locked_licence_mount_is_used_by_compose(docker_daemon, tmp_path):
    """The Docker lane consumes the resolver's node-locked licence mount."""
    root = os.environ.get("FINN_XILINX_PATH")
    value = os.environ.get("XILINXD_LICENSE_FILE") or os.environ.get("LM_LICENSE_FILE")
    _, files = finn_env.classify_license(value)
    licence = next((path for path in files if os.path.isfile(path)), None)
    if not root or not licence:
        pytest.skip("a Xilinx tree and node-locked licence are required")
    env = {"FINN_HOST_BUILD_DIR": str(tmp_path / "build")}
    override = compose_override("build", "build", env=env)
    licdir = os.path.dirname(os.path.abspath(licence))
    mounts = override["services"]["build"]["volumes"]
    assert any(m["target"] == licdir and m.get("read_only") for m in mounts)
    tag = ensure_image("finn")
    proc = compose_run(
        override, "build", ["vivado", "-version"], tag, tmp_path, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr


def test_10_bare_host_toolchain_resolution():
    """The bare-host lane finds and applies the same resolved toolchain."""
    root = os.environ.get("FINN_XILINX_PATH")
    if not root or not os.path.isdir(root):
        pytest.skip("FINN_XILINX_PATH is not configured")
    proc = run(
        [
            "bash",
            "-c",
            'eval "$(./docker/config inspect --tier build --format sh | sed "s/^/export /")"; '
            ". ./docker/finn-toolchain.sh; command -v vivado",
        ]
    )
    assert proc.returncode == 0, proc.stderr


def test_11_exported_sif_runs_with_the_standard_cli():
    """Apptainer/Singularity can execute an explicitly selected FINN SIF."""
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if not runtime:
        pytest.skip("neither apptainer nor singularity is installed")
    configured = os.environ.get("FINN_TEST_SIF")
    if not configured:
        pytest.skip("set FINN_TEST_SIF to exercise an exported SIF")
    sif = Path(configured)
    if not sif.is_file():
        pytest.fail("FINN_TEST_SIF does not exist: %s" % sif)
    run(
        [
            runtime,
            "exec",
            "--cleanenv",
            "--bind",
            "%s:%s" % (REPO, REPO),
            "--pwd",
            REPO,
            "--env",
            "FINN_ROOT=%s" % REPO,
            sif,
            "python",
            "-c",
            "import finn",
        ],
        timeout=180,
        check=True,
    )


def test_12_bake_owns_runtime_tags_and_custom_flavors():
    """Fixed and parameterized targets expose the runtime set in their tag."""
    require_command("docker")
    cases = [
        ("finn", "", "xilinx/finn:CONFORMANCE"),
        ("finn-xrt", "", "xilinx/finn:CONFORMANCE.xrt"),
        ("finn-slash-xrt", "", "xilinx/finn:CONFORMANCE.slash.xrt"),
        ("finn-runtime", "xrt,slash,xrt", "xilinx/finn:CONFORMANCE.slash.xrt"),
    ]
    for target, runtimes, expected in cases:
        config = bake_config(target, runtimes=runtimes)
        assert config["target"][target]["tags"][0] == expected
    proc = run(["bash", "-c", ". ./docker/lib.sh; finn_bake_target xrt,slash"])
    assert proc.stdout == "finn-runtime"


def test_13_missing_supplied_runtime_fails_with_its_path(docker_daemon):
    """A missing supplied package is never silently omitted from an image."""
    package = REPO / "docker/packages/slash.deb"
    if package.exists():
        pytest.skip("%s exists, so the absent case cannot be tested" % package)
    proc = run(
        ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--load", "finn-slash-xrt"],
        timeout=3600,
    )
    assert proc.returncode != 0
    assert "docker/packages/slash.deb" in proc.stdout + proc.stderr
