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
INPUTS = ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE.txt", "src")
TREES = {"rtllib": "finn-rtllib", "finn-xsi": "finn_xsi", "custom-hls": "custom_hls"}


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    source = tmp_path_factory.mktemp("source")
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", *INPUTS, *TREES.values()],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    for name in filter(None, tracked):
        target = source / name.decode()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name.decode(), target)
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
        expected = {
            f"finn/_sources/{name}/{path}"
            for path in subprocess.run(
                ["git", "ls-files", "--", "."],
                cwd=REPO / tree,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        }
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
