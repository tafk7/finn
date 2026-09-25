"""Exercise preparation with real pip and a dependency-free PEP 660 test backend."""

import json
import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

PREPARE = Path(__file__).resolve().parents[2] / "scripts/prepare-editables"
BACKEND = """
import base64
import hashlib
import json
import zipfile
from pathlib import Path

def build(out, editable):
    root = Path(__file__).parent
    config = json.loads((root / "config.json").read_text())
    info = "prepare_demo-1.0.dist-info"
    files = {
        info + "/METADATA": "Metadata-Version: 2.1\\nName: prepare-demo\\nVersion: 1.0\\n"
            + "".join("Requires-Dist: " + dep + "\\n" for dep in config["requires"]),
        info + "/WHEEL": "Wheel-Version: 1.0\\nGenerator: test\\n"
            "Root-Is-Purelib: true\\nTag: py3-none-any\\n",
        info + "/entry_points.txt": "[console_scripts]\\nprepare-demo = prepare_demo:main\\n",
    }
    if editable:
        files["prepare_demo.pth"] = str(root) + "\\n"
    else:
        files["prepare_demo.py"] = (root / "prepare_demo.py").read_text()
    records = []
    for name, value in files.items():
        hashed = hashlib.sha256(value.encode()).digest()
        digest = base64.urlsafe_b64encode(hashed).rstrip(b"=").decode()
        records.append(name + ",sha256=" + digest + "," + str(len(value.encode())))
    files[info + "/RECORD"] = "\\n".join(records + [info + "/RECORD,,"]) + "\\n"
    filename = "prepare_demo-1.0-py3-none-any.whl"
    with zipfile.ZipFile(Path(out) / filename, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return filename

def build_wheel(out, config_settings=None, metadata_directory=None):
    return build(out, False)

def build_editable(out, config_settings=None, metadata_directory=None):
    return build(out, True)
"""


def run(argv, cwd, env=None, check=True):
    return subprocess.run(
        list(map(str, argv)), cwd=cwd, env=env, capture_output=True, text=True, check=check
    )


def environment(tmp_path):
    prefix = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True).create(prefix)
    return prefix / "bin/python"


def project(tmp_path):
    source = tmp_path / "source with spaces"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n'
    )
    (source / "backend.py").write_text(textwrap.dedent(BACKEND))
    (source / "config.json").write_text(json.dumps({"requires": []}))
    (source / "prepare_demo.py").write_text("VALUE='baked'\ndef main(): print(VALUE)\n")
    manifest = tmp_path / "editable-requirements.txt"
    manifest.write_text('-e "./source with spaces"\n')
    return source, manifest


def test_replaces_wheel_with_live_source_and_ignores_pip_redirection(tmp_path):
    python = environment(tmp_path)
    source, manifest = project(tmp_path)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    run(
        [
            python,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            wheels,
            source,
        ],
        tmp_path,
    )
    run(
        [python, "-m", "pip", "install", "--no-index", "--no-deps", next(wheels.glob("*.whl"))],
        tmp_path,
    )
    assert (
        run(
            [python, "-I", "-c", "import prepare_demo; print(prepare_demo.VALUE)"], tmp_path
        ).stdout.strip()
        == "baked"
    )
    (source / "prepare_demo.py").write_text("VALUE='editable'\ndef main(): print(VALUE)\n")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    target = tmp_path / "wrong-install-target"
    env = {**os.environ, "PIP_USER": "true", "PIP_TARGET": str(target)}
    result = run([python, PREPARE, manifest], unrelated, env)
    assert not target.exists()
    assert "No broken requirements found" in result.stdout
    info = run(
        [
            python,
            "-I",
            "-c",
            "import prepare_demo; from importlib.metadata import distribution; "
            "print(prepare_demo.__file__); "
            "print(distribution('prepare-demo').read_text('direct_url.json'))",
        ],
        unrelated,
    ).stdout.splitlines()
    assert info[0] == str(source / "prepare_demo.py")
    assert json.loads(info[1])["dir_info"]["editable"]
    assert run([python.parent / "prepare-demo"], unrelated).stdout.strip() == "editable"
    replacement = source / "replacement.py"
    replacement.write_text("VALUE='atomic source edit'\ndef main(): print(VALUE)\n")
    replacement.replace(source / "prepare_demo.py")
    assert run([python.parent / "prepare-demo"], unrelated).stdout.strip() == "atomic source edit"


def test_dependency_changes_fail_without_installing_dependencies(tmp_path):
    python = environment(tmp_path)
    source, manifest = project(tmp_path)
    (source / "config.json").write_text(
        json.dumps({"requires": ["finn-prepare-missing-dependency>=1"]})
    )
    result = run([python, PREPARE, manifest], tmp_path, check=False)
    assert result.returncode != 0
    assert "finn-prepare-missing-dependency" in result.stdout
    assert "compatible dependency baseline" in result.stderr


def test_missing_manifest_fails_before_pip(tmp_path):
    result = run([sys.executable, PREPARE, tmp_path / "missing.txt"], tmp_path, check=False)
    assert result.returncode == 2
    assert "Requirements file not found" in result.stderr
