"""Dependency closure/identity checks before an image can ship an offline wheelhouse."""

import pytest

import importlib.util
import json
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "wheelhouse", Path(__file__).resolve().parents[2] / "docker/wheelhouse.py"
)
wheelhouse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheelhouse)


def wheel(directory, name, requires=()):
    directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(directory / f"{name}-1.0-py3-none-any.whl", "w") as archive:
        archive.writestr(
            f"{name}-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
            + "".join(f"Requires-Dist: {value}\n" for value in requires),
        )
        archive.writestr("package/vendor/other.dist-info/METADATA", "vendored metadata")


def test_offline_manifest_is_pinned_and_checksummed(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("pip", "setuptools", "wheel", "build", "setuptools_scm"):
        wheel(tmp_path / "wheels", name)
    wheelhouse.manifest(tmp_path, sources)
    record = json.loads((tmp_path / "wheelhouse.json").read_text())
    assert set(record["wheels"]) == {"pip", "setuptools", "wheel", "build", "setuptools-scm"}
    requirements = (tmp_path / "development-requirements.txt").read_text()
    assert requirements.count("--hash=sha256:") == 5
    assert "setuptools-scm==1.0" in requirements


@pytest.mark.parametrize(
    "name,requires,message",
    [
        ("finn", [], "FINN or duplicate"),
        ("consumer", ["finn>=0.1"], "requires FINN"),
        ("consumer", ["missing>=1"], "Unsatisfied dependency"),
    ],
)
def test_dependency_artifact_rejects_finn_or_incomplete_closure(tmp_path, name, requires, message):
    wheel(tmp_path / "wheels", name, requires)
    with pytest.raises(ValueError, match=message):
        wheelhouse.manifest(tmp_path, tmp_path)
