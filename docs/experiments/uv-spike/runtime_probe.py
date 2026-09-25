#!/usr/bin/env python3
"""Isolated uv spike; only scratch files and disposable containers are mutated."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path("/tmp/finn-uv-spike/runtime")
IMAGE = "xilinx/finn:deps-1d3775474e8454c3"


def command(argv, name, check=True, env=None):
    ROOT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, env=env)
    (ROOT / (name + ".log")).write_text(result.stdout + result.stderr)
    if check and result.returncode:
        raise RuntimeError(f"{name} failed ({result.returncode}); see {ROOT / (name + '.log')}")
    return result


def inventory():
    guidance = []
    for directory in [REPO.parent, REPO, REPO / "docs", REPO / "docs/experiments"]:
        path = directory / "AGENTS.md"
        if path.exists():
            guidance.append({"path": str(path), "text": path.read_text()})
    uv = shutil.which("uv")
    facts = {
        "guidance": guidance,
        "uv": uv,
        "uv_version": command([uv, "--version"], "uv-version").stdout.strip(),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "image": json.loads(command(["docker", "image", "inspect", IMAGE], "image").stdout)[0][
            "Id"
        ],
    }
    result = command(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            IMAGE,
            "-c",
            "import json,sys; from pathlib import Path; "
            "print(json.dumps({'python':sys.version,'manifest':"
            "json.loads(Path('/opt/finn/wheelhouse.json').read_text())}))",
        ],
        "image-wheelhouse",
    )
    (ROOT / "image-wheelhouse.json").write_text(result.stdout)
    facts["wheel_count"] = len(json.loads(result.stdout)["manifest"]["wheels"])
    (ROOT / "inventory.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


def fixtures():
    if (ROOT / "fixtures-ready").exists():
        reset_markers()
        return
    source = Path("/tmp/finn-implementation-qonnx")
    base = ROOT / "qonnx-repository"
    shutil.copytree(
        source, base, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist")
    )
    for pair in ("a", "b", "c"):
        directory = ROOT / "pairs" / pair
        finn = directory / "finn"
        finn.mkdir(parents=True)
        for name in (
            "setup.py",
            "setup.cfg",
            "MANIFEST.in",
            "VERSION",
            "README.md",
            "LICENSE.txt",
            "deps.env",
        ):
            shutil.copy2(REPO / name, finn / name)
        shutil.copytree(
            REPO / "src",
            finn / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc", "*.so"),
        )
        command(
            ["git", "-C", base, "worktree", "add", "--detach", directory / "qonnx", "HEAD"],
            "worktree-" + pair,
        )
        for path in (
            finn / "src/finn/util/uv_spike_marker.py",
            directory / "qonnx/src/qonnx/uv_spike_marker.py",
        ):
            path.write_text(f"VALUE = {pair!r}\n")
    (ROOT / "fixtures-ready").write_text("Snapshots and task-owned QONNX worktrees\n")


def reset_markers():
    for pair in ("a", "b", "c"):
        for relative in (
            "finn/src/finn/util/uv_spike_marker.py",
            "qonnx/src/qonnx/uv_spike_marker.py",
        ):
            (ROOT / "pairs" / pair / relative).write_text(f"VALUE = {pair!r}\n")


def image_probe():
    """Bake a true source-excluded venv; attach editables in a persistent volume."""
    reset_markers()
    context = ROOT / "image-context"
    context.mkdir(exist_ok=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / "pairs/a" / name, context / name)
    shutil.copy2(ROOT / "uv", context / "uv")
    # The package names come from the declarative source table, not per-package CLI code.
    sources = tomllib.loads((context / "pyproject.toml").read_text())["tool"]["uv"]["sources"]
    (context / "local-packages.json").write_text(json.dumps(sorted(sources)))
    command(
        [
            "docker",
            "buildx",
            "bake",
            "-f",
            str(REPO / "docker-bake.hcl"),
            "--load",
            "--set",
            "finn.target=system",
            "--set",
            "finn.tags=finn-uv-spike:system",
            "finn",
        ],
        "system-build",
    )
    command(
        [
            "docker",
            "buildx",
            "build",
            "--load",
            "-t",
            "finn-uv-spike:source-excluded",
            "--build-arg",
            f"SPIKE_UID={os.getuid()}",
            "--build-arg",
            f"PROJECT_ROOT={ROOT / 'pairs/a'}",
            "-f",
            str(Path(__file__).with_name("Dockerfile")),
            context,
        ],
        "excluded-image-build",
    )
    image = "finn-uv-spike:source-excluded"
    command(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            image,
            "python",
            "-I",
            "-c",
            "import importlib.util; from pathlib import Path; "
            "assert importlib.util.find_spec('finn') is None; "
            "assert importlib.util.find_spec('qonnx') is None; "
            "assert not list(Path('/opt/finn/wheels').glob('qonnx-*.whl'))",
        ],
        "excluded-image-imports",
    )
    volume = "finn-uv-spike-" + uuid.uuid4().hex[:10]
    command(["docker", "volume", "create", volume], "volume-create")
    base = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        f"type=volume,src={volume},dst=/opt/finn/venv",
        "--mount",
        f"type=bind,src={ROOT},dst={ROOT}",
        image,
    ]
    try:
        started = time.monotonic()
        prep = command(
            [
                *base,
                "uv",
                "sync",
                "--project",
                str(ROOT / "pairs/a"),
                "--locked",
                "--offline",
                "--no-index",
                "--find-links",
                "/opt/finn/wheels",
            ],
            "volume-prepare",
        )
        elapsed = time.monotonic() - started
        probe = (
            "import json,sys; from importlib.metadata import distribution; "
            "import finn.util.uv_spike_marker as f,qonnx.uv_spike_marker as q; "
            "assert (f.VALUE,q.VALUE)==('a','a'); assert sys.prefix=='/opt/finn/venv'; "
            "print(json.dumps({'finn':f.__file__,'qonnx':q.__file__,'prefix':sys.prefix}))"
        )
        first = command([*base, "python", "-I", "-c", probe], "volume-reuse")
        command([*base, "python", "-m", "pip", "check"], "volume-pip-check")
        command([*base, "build_dataflow", "--help"], "volume-entrypoint")
        record = {
            "image": image,
            "prepare_seconds": elapsed,
            "selection": json.loads(first.stdout),
            "prepare_output": prep.stdout + prep.stderr,
            "persistent_volume_reuse": True,
        }
        (ROOT / "image-results.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
    finally:
        command(["docker", "volume", "rm", volume], "volume-remove", check=False)


def sbx_probe():
    """Use native sbx lifecycle and its own writable environment for pair C."""
    name = "finn-uv-spike-" + uuid.uuid4().hex[:8]
    config_dir = ROOT.parent / "sbx-config"
    config_dir.mkdir(exist_ok=True)
    config = config_dir / (name + ".sbxenv.yaml")
    prefix = "/home/agent/.venvs/finn-uv-spike"
    config.write_text(
        json.dumps(
            {
                "schemaVersion": "1",
                "name": name,
                "workspace": str(ROOT),
                "agent": "shell",
                "sandboxOptions": {
                    "template": "xilinx/finn:sbx-deps-928baf60bf600cfd",
                    "pullPolicy": "never",
                    "shareSkills": False,
                },
                "env": {
                    "UV_PROJECT_ENVIRONMENT": prefix,
                    "UV_CACHE_DIR": "/home/agent/.cache/finn-uv-spike",
                    "UV_PYTHON": "/usr/bin/python3",
                    "UV_PYTHON_DOWNLOADS": "never",
                    "UV_LINK_MODE": "copy",
                },
            },
            indent=2,
        )
        + "\n"
    )
    created = False
    try:
        command(["sbx", "env", "plan", config], "sbx-plan")
        command(["sbx", "env", "create", "--auto-approve", config], "sbx-create")
        created = True
        execute = ["sbx", "env", "exec", config, "--"]
        sync = [
            ROOT / "uv",
            "sync",
            "--project",
            ROOT / "pairs/c",
            "--locked",
            "--offline",
            "--no-index",
            "--find-links",
            "/opt/finn/wheels",
        ]
        command([*execute, *sync, "--no-install-local"], "sbx-partial")
        command(
            [
                *execute,
                prefix + "/bin/python",
                "-I",
                "-c",
                "import importlib.util; assert importlib.util.find_spec('finn') is None; "
                "assert importlib.util.find_spec('qonnx') is None",
            ],
            "sbx-excluded",
        )
        command([*execute, *sync], "sbx-editable")
        probe = (
            "import json,sys; import finn.util.uv_spike_marker as f,qonnx.uv_spike_marker as q; "
            "assert (f.VALUE,q.VALUE)==('c','c'); "
            f"assert sys.prefix=={prefix!r}; "
            "from finn.util.resources import resource_path; "
            "print(json.dumps({'finn':f.__file__,'qonnx':q.__file__,'rtl':resource_path('rtllib')}))"
        )
        result = command([*execute, prefix + "/bin/python", "-I", "-c", probe], "sbx-reuse")
        command([*execute, prefix + "/bin/build_dataflow", "--help"], "sbx-entrypoint")
        command([*execute, prefix + "/bin/python", "-m", "pip", "check"], "sbx-pip-check")
        record = {
            "template": "xilinx/finn:sbx-deps-928baf60bf600cfd",
            "prefix": prefix,
            "selection_output": result.stdout,
            "repeated_native_exec": True,
        }
        (ROOT / "sbx-results.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
    finally:
        if created:
            command(["sbx", "env", "rm", config, "--force"], "sbx-remove", check=False)


def native_probe():
    """Real host Python-package journey; all interpreter/data files are task-owned."""
    interpreter_root = ROOT / "native-python"
    command(
        [
            ROOT / "uv",
            "python",
            "install",
            "3.10",
            "--no-bin",
            "--install-dir",
            interpreter_root,
            "--cache-dir",
            ROOT / "native-cache",
        ],
        "native-python-install",
    )
    python = next(interpreter_root.glob("*/bin/python3.10"))
    wheels = ROOT / "native-wheels"
    if not wheels.exists():
        wheels.mkdir()
        container = command(["docker", "create", IMAGE], "wheel-copy-container").stdout.strip()
        try:
            command(
                ["docker", "cp", container + ":/opt/finn/wheels/.", wheels], "native-wheels-copy"
            )
        finally:
            command(["docker", "rm", container], "wheel-copy-remove", check=False)
    project = ROOT / "native-project"
    project.mkdir(exist_ok=True)
    source = (ROOT / "pairs/b/pyproject.toml").read_text()
    source = source.replace('path = "./finn"', 'path = "../pairs/b/finn"')
    source = source.replace('path = "./qonnx"', 'path = "../pairs/b/qonnx"')
    (project / "pyproject.toml").write_text(source)
    prefix = ROOT / "native-environment"
    environment = {
        **os.environ,
        "UV_PYTHON": str(python),
        "UV_PROJECT_ENVIRONMENT": str(prefix),
        "UV_CACHE_DIR": str(ROOT / "native-cache"),
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_LINK_MODE": "copy",
    }
    offline = ["--offline", "--no-index", "--find-links", wheels]
    command([ROOT / "uv", "lock", "--project", project, *offline], "native-lock", env=environment)
    command(
        [ROOT / "uv", "sync", "--project", project, "--locked", "--no-install-local", *offline],
        "native-partial",
        env=environment,
    )
    command(
        [
            prefix / "bin/python",
            "-I",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('finn') is None; "
            "assert importlib.util.find_spec('qonnx') is None",
        ],
        "native-excluded",
    )
    command(
        [ROOT / "uv", "sync", "--project", project, "--locked", *offline],
        "native-editable",
        env=environment,
    )
    result = command(
        [
            prefix / "bin/python",
            "-I",
            "-c",
            "import json,sys; import finn.util.uv_spike_marker as f,qonnx.uv_spike_marker as q; "
            "assert (f.VALUE,q.VALUE)==('b','b'); from finn.util.resources import resource_path; "
            "print(json.dumps({'python':sys.version,'prefix':sys.prefix,'finn':f.__file__,"
            "'qonnx':q.__file__,'rtl':resource_path('rtllib')}))",
        ],
        "native-reuse",
    )
    command([prefix / "bin/python", "-m", "pip", "check"], "native-pip-check")
    command([prefix / "bin/build_dataflow", "--help"], "native-entrypoint")
    record = json.loads(result.stdout)
    (ROOT / "native-results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def export_probe():
    records = {}
    base = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={ROOT},dst={ROOT}",
        "--entrypoint",
        str(ROOT / "uv"),
        IMAGE,
    ]
    for pair in ("a", "b", "c"):
        project = ROOT / "pairs" / pair
        result = command(
            [
                *base,
                "export",
                "--project",
                project,
                "--frozen",
                "--no-emit-local",
                "--no-header",
                "--no-annotate",
                "--offline",
                "--no-python-downloads",
                "--cache-dir",
                ROOT / "export-cache",
            ],
            pair + "-external-export",
        )
        (ROOT / (pair + "-external-requirements.txt")).write_text(result.stdout)
        records[pair] = {
            "sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "line_count": len(result.stdout.splitlines()),
        }
    assert len({record["sha256"] for record in records.values()}) == 1
    (ROOT / "export-results.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


def docker_probe():
    fixtures()
    binary = ROOT / "uv"
    if not binary.exists():
        shutil.copy2(shutil.which("uv"), binary)
    probe = Path(__file__).with_name("runtime_in_container.py")
    started = time.monotonic()
    result = command(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={ROOT},dst={ROOT}",
            "--mount",
            f"type=bind,src={probe},dst=/spike.py,readonly",
            "--entrypoint",
            "python",
            IMAGE,
            "/spike.py",
            str(ROOT),
        ],
        "docker-probe",
        check=False,
    )
    print(
        json.dumps(
            {
                "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "log": str(ROOT / "docker-probe.log"),
            }
        )
    )
    if result.returncode:
        print(result.stdout[-3000:] + result.stderr[-3000:])
        raise SystemExit(result.returncode)
    print((ROOT / "docker-results.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inventory", "docker", "image", "sbx", "native", "export"])
    args = parser.parse_args()
    {
        "inventory": inventory,
        "docker": docker_probe,
        "image": image_probe,
        "sbx": sbx_probe,
        "native": native_probe,
        "export": export_probe,
    }[args.mode]()


if __name__ == "__main__":
    main()
