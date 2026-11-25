# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A wheel carries FINN's own trees and finds them without a checkout."""
import pytest

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

pytestmark = pytest.mark.util

REPO = Path(__file__).resolve().parents[2]
# What the wheel is built from: the project files and the trees it ships.
INPUTS = ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE.txt", "src")
TREES = {"rtllib": "finn-rtllib", "finn-xsi": "finn_xsi", "custom-hls": "custom_hls"}
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.egg-info", "build")


def files(root):
    """The files under root, relative, without byte code or build output."""
    found = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP(directory, dirs)]
        found += [
            Path(directory, n).relative_to(root).as_posix()
            for n in names
            if n not in SKIP(directory, names)
        ]
    return found


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    source = tmp_path_factory.mktemp("source")
    for name in (*INPUTS, *TREES.values()):
        if (REPO / name).is_dir():
            shutil.copytree(REPO / name, source / name, ignore=SKIP, symlinks=True)
        else:
            shutil.copy2(REPO / name, source / name)
    out = tmp_path_factory.mktemp("dist")
    # The PEP 517 backend in this environment: no download, no build isolation.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools import build_meta; build_meta.build_wheel(sys.argv[1])",
            str(out),
        ],
        cwd=source,
        check=True,
        capture_output=True,
    )
    return next(out.glob("finn-*.whl"))


def test_the_wheel_ships_finn_trees(wheel):
    names = set(zipfile.ZipFile(wheel).namelist())
    for name, tree in TREES.items():
        expected = {f"finn/_sources/{name}/{path}" for path in files(REPO / tree)}
        assert expected <= names, sorted(expected - names)[:5]
    assert "finn/qnn-data/cpp/cnpy.cpp" in names
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]


def test_an_unpacked_wheel_needs_no_checkout(wheel, tmp_path):
    site = tmp_path / "site"
    zipfile.ZipFile(wheel).extractall(site)
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "FINN_HOME": str(tmp_path / "home")}
    probe = (
        f"import sys; sys.path.insert(0, {str(site)!r});"
        "import json; from finn import resources as r;"
        "print(json.dumps({n: [r.status(n).state, r.path(n)]"
        " for n in ('rtllib', 'finn-xsi', 'custom-hls', 'qnn-data')}))"
    )
    # Isolated and without site-packages: only the unpacked wheel provides finn,
    # and finn.resources needs nothing else.
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout)
    for name, (state, path) in found.items():
        assert state == "package", (name, state)
        assert Path(path).is_relative_to(site.resolve()), (name, path)
    assert (Path(found["rtllib"][1]) / "mvu").is_dir()
    assert (Path(found["finn-xsi"][1]) / "finn_xsi/adapter.py").is_file()
