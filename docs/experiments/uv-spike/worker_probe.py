#!/usr/bin/env python3
"""Bounded, offline uv semantics probe for FINN editable-source environments.

This script only replaces /tmp/finn-uv-spike/worker. It builds tiny synthetic
packages with a self-contained PEP 517 backend so that the test exercises uv's
project and editable behavior without relying on PyPI or the FINN wheelhouse.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

ROOT = Path("/tmp/finn-uv-spike/worker")
UV = Path("/home/tkeller787/.local/bin/uv")
PYTHON = Path("/tmp/finn-runtime-venv/bin/python")
LOG = ROOT / "worker_probe.log"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())


def run(
    argv: list[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    expected: int = 0,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in argv]
    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=effective_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with LOG.open("a") as stream:
        stream.write(f"\n$ (cd {cwd or Path.cwd()} && {' '.join(command)})\n")
        stream.write(result.stdout)
        stream.write(f"[exit {result.returncode}]\n")
    if result.returncode != expected:
        raise RuntimeError(
            f"expected exit {expected}, got {result.returncode}: {' '.join(command)}\n"
            f"{result.stdout}"
        )
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_wheel(name: str, version: str, module: str, marker: str) -> Path:
    normalized = name.replace("-", "_")
    wheel = ROOT / "wheelhouse" / f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    entries = {
        f"{module}.py": f"MARKER = {marker!r}\n",
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.3\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            "Requires-Python: >=3.10\n\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: worker_probe\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
        f"{dist_info}/RECORD": "",
    }
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, content in entries.items():
            archive.writestr(filename, content)
    return wheel


BACKEND = r"""
from pathlib import Path
import base64
import csv
import hashlib
import io
import tomllib
import zipfile


def _metadata():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    name = project["name"]
    version = project["version"]
    requires = project.get("dependencies", [])
    return name, version, requires


def _wheel(wheel_directory, editable):
    name, version, requires = _metadata()
    normalized = name.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.3\n"
        f"Name: {name}\n"
        f"Version: {version}\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requires)
        + "Requires-Python: >=3.10\n\n"
    )
    entries = {
        f"{dist_info}/METADATA": metadata.encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: worker_backend\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
    }
    if editable:
        entries[f"_{normalized}_editable.pth"] = (str(Path.cwd()) + "\n").encode()
    else:
        module = normalized
        entries[f"{module}/__init__.py"] = (Path(module) / "__init__.py").read_bytes()
    record_rows = []
    for filename, content in entries.items():
        value = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        record_rows.append((filename, f"sha256={value}", str(len(content))))
    record_rows.append((f"{dist_info}/RECORD", "", ""))
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(record_rows)
    entries[f"{dist_info}/RECORD"] = output.getvalue().encode()
    filename = f"{normalized}-{version}-py3-none-any.whl"
    target = Path(wheel_directory) / filename
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in entries.items():
            archive.writestr(path, content)
    return filename


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _wheel(wheel_directory, False)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _wheel(wheel_directory, True)


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []
"""


def create_package(path: Path, name: str, marker: str, dependencies: list[str]) -> None:
    normalized = name.replace("-", "_")
    requirements = ",\n    ".join(json.dumps(item) for item in dependencies)
    write(
        path / "pyproject.toml",
        f"""
        [build-system]
        requires = []
        build-backend = "worker_backend"
        backend-path = ["."]

        [project]
        name = "{name}"
        version = "1.0.0"
        requires-python = ">=3.10"
        dependencies = [
            {requirements}
        ]
        """,
    )
    write(path / "worker_backend.py", BACKEND)
    write(path / normalized / "__init__.py", f"MARKER = {marker!r}\n")


def create_pair(name: str) -> tuple[Path, Path]:
    pair = ROOT / "pairs" / name
    create_package(pair / "qonnx", "qonnx", f"qonnx-{name}", ["worker-external==1.0.0"])
    create_package(
        pair / "finn",
        "finn",
        f"finn-{name}",
        ["qonnx==1.0.0", "worker-external==1.0.0"],
    )
    project = pair / "project"
    write(
        project / "pyproject.toml",
        """
        [project]
        name = "finn-dev-wrapper"
        version = "0.0.0"
        requires-python = ">=3.10"
        dependencies = ["finn==1.0.0", "qonnx==1.0.0"]

        [tool.uv]
        package = false
        no-index = true
        find-links = ["/tmp/finn-uv-spike/worker/wheelhouse"]

        [tool.uv.sources]
        finn = { path = "../finn", editable = true }
        qonnx = { path = "../qonnx", editable = true }
        """,
    )
    environment = ROOT / "envs" / name
    return project, environment


def uv_args(project: Path, environment: Path) -> tuple[list[str], dict[str, str]]:
    args = [
        str(UV),
        "--project",
        str(project),
        "--offline",
        "--no-python-downloads",
    ]
    return args, {"UV_PROJECT_ENVIRONMENT": str(environment)}


def assert_imports(environment: Path, expected: dict[str, str | None]) -> None:
    expression = (
        "import importlib.util,json; out={}; "
        + "; ".join(
            f"s=importlib.util.find_spec({name!r}); "
            f"out[{name!r}]=None if s is None else __import__({name!r}).MARKER"
            for name in expected
        )
        + f"; assert out == {expected!r}, out; print(json.dumps(out,sort_keys=True))"
    )
    run([environment / "bin/python", "-c", expression])


def export_external(project: Path, environment: Path, destination: Path) -> str:
    base, env = uv_args(project, environment)
    run(
        base
        + [
            "export",
            "--locked",
            "--no-emit-local",
            "--no-header",
            "--no-annotate",
            "--output-file",
            str(destination),
        ],
        env=env,
    )
    content = destination.read_text()
    assert "worker-external==1.0.0" in content, content
    assert "finn" not in content and "qonnx" not in content, content
    return digest(destination)


def main() -> None:
    if ROOT.resolve() != Path("/tmp/finn-uv-spike/worker"):
        raise RuntimeError(f"unexpected scratch root: {ROOT}")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    LOG.touch()
    if not UV.is_file() or not PYTHON.is_file():
        raise RuntimeError("expected uv and validation Python binaries are unavailable")

    version = run([UV, "--version"]).stdout.strip()
    assert version == "uv 0.10.0", version
    create_wheel("worker-external", "1.0.0", "worker_external", "external-1")
    create_wheel("worker-added", "1.0.0", "worker_added", "added-1")

    pairs = {name: create_pair(name) for name in ("one", "two", "three")}
    initial_lock_hashes = {}
    initial_export_hashes = {}
    for name, (project, environment) in pairs.items():
        base, env = uv_args(project, environment)
        run(base + ["lock", "--python", PYTHON], env=env)
        initial_lock_hashes[name] = digest(project / "uv.lock")
        initial_export_hashes[name] = export_external(
            project, environment, ROOT / f"external-{name}.txt"
        )
    assert len(set(initial_lock_hashes.values())) == 1, initial_lock_hashes
    assert len(set(initial_export_hashes.values())) == 1, initial_export_hashes

    # Base-layer sync: external closure only; local editables remain absent.
    one_project, one_env = pairs["one"]
    base, env = uv_args(one_project, one_env)
    run(base + ["sync", "--locked", "--no-install-local", "--python", PYTHON], env=env)
    assert_imports(one_env, {"worker_external": "external-1", "finn": None, "qonnx": None})
    assert not (one_project / ".venv").exists()
    assert (one_env / "pyvenv.cfg").is_file()

    # Full sync adds both selected editables. Source-content edits are visible
    # immediately and leave both the lock and external export unchanged.
    run(base + ["sync", "--locked"], env=env)
    assert_imports(
        one_env,
        {"worker_external": "external-1", "finn": "finn-one", "qonnx": "qonnx-one"},
    )
    lock_before_edit = digest(one_project / "uv.lock")
    write(ROOT / "pairs/one/qonnx/qonnx/__init__.py", "MARKER = 'qonnx-one-edited'\n")
    assert_imports(
        one_env,
        {
            "worker_external": "external-1",
            "finn": "finn-one",
            "qonnx": "qonnx-one-edited",
        },
    )
    run(base + ["lock", "--check"], env=env)
    assert digest(one_project / "uv.lock") == lock_before_edit
    assert (
        export_external(one_project, one_env, ROOT / "external-one-edited.txt")
        == initial_export_hashes["one"]
    )

    # --no-install-workspace does not omit ordinary local path dependencies.
    three_project, three_env = pairs["three"]
    base3, env3 = uv_args(three_project, three_env)
    run(base3 + ["sync", "--locked", "--no-install-workspace", "--python", PYTHON], env=env3)
    assert_imports(
        three_env,
        {"worker_external": "external-1", "finn": "finn-three", "qonnx": "qonnx-three"},
    )

    # Normal uv run syncs the project; --no-sync executes the already prepared
    # environment without installing the omitted local packages.
    two_project, two_env = pairs["two"]
    base2, env2 = uv_args(two_project, two_env)
    run(base2 + ["sync", "--locked", "--no-install-local", "--python", PYTHON], env=env2)
    run(
        base2
        + [
            "run",
            "--no-sync",
            "python",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('finn') is None",
        ],
        env=env2,
    )
    assert_imports(two_env, {"worker_external": "external-1", "finn": None, "qonnx": None})
    run(base2 + ["run", "--locked", "python", "-c", "import finn,qonnx"], env=env2)
    assert_imports(
        two_env,
        {"worker_external": "external-1", "finn": "finn-two", "qonnx": "qonnx-two"},
    )

    # Relocating a complete pair preserves its relative source declarations and
    # lock. Changing a declaration is detected, but its external export is stable.
    relocated = ROOT / "relocated" / "one"
    shutil.copytree(ROOT / "pairs/one", relocated)
    relocated_project = relocated / "project"
    relocated_env = ROOT / "envs/relocated-one"
    relocated_base, relocated_vars = uv_args(relocated_project, relocated_env)
    run(relocated_base + ["lock", "--check", "--python", PYTHON], env=relocated_vars)
    assert digest(relocated_project / "uv.lock") == lock_before_edit
    assert (
        export_external(relocated_project, relocated_env, ROOT / "external-relocated-one.txt")
        == initial_export_hashes["one"]
    )

    qonnx_custom = ROOT / "pairs/three/qonnx-custom"
    shutil.copytree(ROOT / "pairs/three/qonnx", qonnx_custom)
    pyproject = three_project / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('../qonnx"', '../qonnx-custom"'))
    run(base3 + ["lock", "--check"], env=env3, expected=1)
    run(base3 + ["lock"], env=env3)
    changed_lock_hash = digest(three_project / "uv.lock")
    assert changed_lock_hash != initial_lock_hashes["three"]
    assert (
        export_external(three_project, three_env, ROOT / "external-three-repathed.txt")
        == initial_export_hashes["three"]
    )

    # An outdated project is deliberately ignored by --no-sync. A normal run
    # with --locked rejects it instead of silently changing the tested lock.
    with pyproject.open("a") as stream:
        stream.write('\n[dependency-groups]\ndev = ["worker-added==1.0.0"]\n')
    run(
        base3
        + [
            "run",
            "--no-sync",
            "python",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('worker_added') is None",
        ],
        env=env3,
    )
    run(base3 + ["run", "--locked", "python", "-c", "pass"], env=env3, expected=2)

    # Preserve a wrapper for the real sources. Offline locking is intentionally
    # left to a wheelhouse-backed runtime probe; host uv has no FINN wheelhouse.
    real_project = ROOT / "real-project"
    write(
        real_project / "pyproject.toml",
        """
        [project]
        name = "finn-real-dev-wrapper"
        version = "0.0.0"
        requires-python = ">=3.10"
        dependencies = ["finn", "qonnx"]

        [tool.uv]
        package = false
        no-index = true
        find-links = ["/tmp/finn-uv-spike/worker/wheelhouse"]

        [tool.uv.sources]
        finn = { path = "/home/tkeller787/finn", editable = true }
        qonnx = { path = "/tmp/finn-implementation-qonnx", editable = true }
        """,
    )
    real_base, real_env = uv_args(real_project, ROOT / "envs/real")
    real_lock = run(real_base + ["lock", "--python", PYTHON], env=real_env, expected=1)
    assert "not found" in real_lock.stdout.lower() or "failed" in real_lock.stdout.lower()

    results = {
        "uv": version,
        "uv_binary": str(UV),
        "python": run([PYTHON, "--version"]).stdout.strip(),
        "lock_hash_shared_by_three_pairs": next(iter(initial_lock_hashes.values())),
        "external_export_hash_shared_by_three_pairs": next(iter(initial_export_hashes.values())),
        "changed_local_source_path_lock_hash": changed_lock_hash,
        "scratch": str(ROOT),
        "log": str(LOG),
        "real_source_lock": "expected offline failure without wheelhouse",
    }
    write(ROOT / "results.json", json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
