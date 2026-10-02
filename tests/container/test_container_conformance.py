"""Runtime conformance tests for FINN's container artifacts and runners.

These tests intentionally exercise real Docker, sbx and SIF behavior.
Unavailable runtimes or host resources are reported as pytest skips. Static
resolver behavior belongs in ``tests/util/test_container_config.py``.
"""

import pytest

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO / "docker"
FINN_ENV = DOCKER_DIR / "config.py"
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


def bake_config(target, *, runtimes="", image_revision="env-CONFORMANCE"):
    env = dict(os.environ)
    env.update({"FINN_RUNTIMES": runtimes, "FINN_IMAGE_REVISION": image_revision})
    proc = run(["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--print", target], env=env)
    if proc.returncode:
        pytest.fail(proc.stderr or proc.stdout)
    start = proc.stdout.find("{")
    assert start >= 0, proc.stdout
    return json.loads(proc.stdout[start:])


def bake_tag(target="finn", *, runtimes="", image_revision="env-CONFORMANCE"):
    return bake_config(target, runtimes=runtimes, image_revision=image_revision)["target"][target][
        "tags"
    ][0]


def wait_ready(exec_prefix, timeout=300):
    """Wait for the entrypoint's startup sync; exec does not wait for the entrypoint."""
    deadline = time.monotonic() + timeout
    while run([*exec_prefix, "test", "-e", "/tmp/finn-ready"], timeout=60).returncode:
        assert time.monotonic() < deadline, "container never became ready"
        time.sleep(1)


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
        env.update({"FINN_RUNTIMES": runtimes, "FINN_IMAGE_REVISION": "env-CONFORMANCE"})
        run(
            ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--load", target],
            env=env,
            timeout=3600,
            check=True,
        )
    return tag


def resolved(tier, policy="auto", env=None):
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    command = [sys.executable, FINN_ENV, "inspect", "--tier", tier]
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
    # Mirrored like docker/run, so FINN_ROOT names the mounted checkout.
    data = resolved("build" if have_xilinx else "dev", "mirror")
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
        wait_ready(["docker", "exec", name])
        run(["docker", "exec", name, "python", "-c", "import finn"], timeout=180, check=True)
        if have_xilinx:
            run(["docker", "exec", name, "vivado", "-version"], timeout=300, check=True)
    finally:
        run(["docker", "rm", "-f", name], timeout=60)


@pytest.mark.parametrize("fpga", [False, True])
def test_05_sbx_workload_and_xilinx_kit(tmp_path, fpga):
    """FINN's workload kit builds from a checkout and runs it; the xilinx kit adds tools.

    sbx builds the workload from the checkout (BuildKit with an OCI exporter:
    Docker's containerd image store, or BUILDX_BUILDER naming a
    docker-container builder). Set FINN_TEST_SBX_HARNESS_KIT to a harness mixin
    (with FINN_TEST_SBX_HARNESS naming its command) to also compose an agent;
    this does not verify authenticated inference. Synthetic FPGA paths and
    policy readback do not prove licence checkout.
    """
    require_command("sbx")
    name = "finn-conformance-" + uuid.uuid4().hex[:12]
    checkout = tmp_path / "finn"
    run(["git", "clone", "--shared", REPO, checkout], check=True)
    args = ["--name", name, "--skills", "off"]
    paths = [checkout, checkout]  # the workload kit, then the workspace
    if fpga:
        toolchain = tmp_path / "Xilinx"
        for tool in ("Vivado", "Vitis"):
            (toolchain / "2025.2" / tool).mkdir(parents=True)
        args += ["--kit", REPO / "docker/sbx/xilinx"]
        for key, value in {
            "vivado": toolchain / "2025.2/Vivado",
            "vitis": toolchain / "2025.2/Vitis",
            "hls": toolchain / "2025.2/Vitis",
            "license_host": "192.0.2.1",
            "license_port": "2100",
            "vendor_port": "2101",
        }.items():
            args += ["--kit-arg", "%s=%s" % (key, value)]
        paths.append("%s:ro" % toolchain)
    harness = os.environ.get("FINN_TEST_SBX_HARNESS_KIT")
    if harness:
        args += ["--kit", harness]
    exec_ = ["sbx", "exec", name, "--"]
    try:
        run(["sbx", "create", *args, *paths], timeout=3600, check=True)
        wait_ready(exec_)
        run([*exec_, "python", "-c", "import finn"], check=True)
        run(
            [*exec_, "sh", "-c", 'test "$FINN_BUILD_DIR" = /tmp/finn_build && echo seen > marker'],
            check=True,
        )
        assert (checkout / "marker").read_text().strip() == "seen"
        if not harness:  # FINN ships no agent
            assert run([*exec_, "sh", "-c", "command -v claude"]).returncode != 0
        else:
            command = os.environ.get("FINN_TEST_SBX_HARNESS", "claude")
            version = run([*exec_, command, "--version"], check=True)
            print("Harness version:", version.stdout.strip())
        if fpga:
            code = (
                "import os; from pathlib import Path; "
                "assert os.environ['XILINXD_LICENSE_FILE']=='2100@192.0.2.1', "
                "os.environ.get('XILINXD_LICENSE_FILE'); "
                "assert all(Path(os.environ[k]).is_dir() for k in "
                "('XILINX_VIVADO','XILINX_VITIS','XILINX_HLS'))"
            )
            # XILINXD_LICENSE_FILE is composed in bash (BASH_ENV), where tools run.
            run([*exec_, "bash", "-c", 'python -c "$0"', code], check=True)
            write = run([*exec_, "touch", toolchain / "must-not-write"])
            assert write.returncode != 0
            assert not (toolchain / "must-not-write").exists()
            policy = run(["sbx", "policy", "ls", name, "--json"], check=True).stdout
            assert "192.0.2.1:2100" in policy
            assert "192.0.2.1:2101" in policy
    finally:
        run(["sbx", "rm", "--force", name], timeout=120)
    inventory = json.loads(run(["sbx", "ls", "--json"], check=True).stdout)["sandboxes"]
    assert not any(item["name"] == name for item in inventory)


def test_05b_image_identity_ignores_mounted_source_commit(tmp_path):
    """Image preparation identity stays independent of mounted-source commits."""
    checkout = tmp_path / "checkout"
    run(["git", "clone", "--shared", REPO, checkout], check=True)
    # Compare two commits within the same isolated checkout; never set repository config.
    environment = {**os.environ, "FINN_SOURCE_ROOT": str(checkout)}
    command = [REPO / "docker/build", "--print"]
    before = run(command, env=environment, check=True)
    run(
        [
            "git",
            "-c",
            "user.name=FINN Conformance",
            "-c",
            "user.email=finn-conformance@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "source-only",
        ],
        cwd=checkout,
        check=True,
    )
    after = run(command, env=environment, check=True)
    before = dict(line.split("=", 1) for line in before.stdout.splitlines())
    after = dict(line.split("=", 1) for line in after.stdout.splitlines())
    assert before["image_revision"] == after["image_revision"]
    assert before["source_revision"] != after["source_revision"]


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
    probe = ["--entrypoint", "sh"]
    command = ["-c", "sudo -n true"]
    generic_run = run(["docker", "run", "--rm", *probe, generic, *command], timeout=180)
    sbx_run = run(["docker", "run", "--rm", *probe, sbx, *command], timeout=180)
    assert generic_run.returncode != 0
    assert sbx_run.returncode == 0


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
            'eval "$(./docker/config.py inspect --tier build --format sh | sed "s/^/export /")"; '
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
        ("finn", "", "xilinx/finn:env-CONFORMANCE"),
        ("finn-xrt", "", "xilinx/finn:env-CONFORMANCE.xrt"),
        ("finn-slash-xrt", "", "xilinx/finn:env-CONFORMANCE.slash.xrt"),
        (
            "finn-slashkit-xrt",
            "",
            "xilinx/finn:env-CONFORMANCE.slash.slashkit.xrt",
        ),
        ("finn-runtime", "xrt,slash,xrt", "xilinx/finn:env-CONFORMANCE.slash.xrt"),
    ]
    for target, runtimes, expected in cases:
        config = bake_config(target, runtimes=runtimes)
        assert config["target"][target]["tags"][0] == expected
        labels = config["target"][target]["labels"]
        assert labels["dev.finn.image-revision"] == "env-CONFORMANCE"
        assert "org.opencontainers.image.revision" not in labels
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


def test_installed_resources_without_checkout(docker_daemon):
    """The release image can generate RTL/driver outputs with no checkout."""
    tag = ensure_image("finn-release")
    script = Path(REPO) / "tests/util/runtime_resource_smoke.py"
    proc = run(
        [
            "docker",
            "run",
            "--rm",
            "--workdir",
            "/tmp",
            "--user",
            "12345:12345",
            "-v",
            f"{script}:/tmp/smoke.py:ro",
            tag,
            "python",
            "/tmp/smoke.py",
            "/tmp/rtl-output",
        ],
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"driver":' in proc.stdout
