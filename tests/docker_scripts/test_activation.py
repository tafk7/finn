# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tool activation in the container (docker/profile.sh), native use without
FINN's own scripts, container identity and the installers."""
import pytest

import json
import os
import re
import shutil
import subprocess
import sys
from packaging.requirements import Requirement
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILE_SH = str(REPO / "docker/profile.sh")
sys.path.insert(0, str(REPO / "docker"))

pytestmark = pytest.mark.util


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("FINN_", "XILINX_")) or name in (
            "BASH_ENV",
            "ENV",
            "SLASHKIT_DEB_PACKAGE",
            "PLATFORM_REPO_PATHS",
            "HLS_PATH",
            "VIVADO_PATH",
            "VITIS_PATH",
            "LD_PRELOAD",
            "XILINXD_LICENSE_FILE",
            "LM_LICENSE_FILE",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FINN_ROOT", str(REPO))
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "state"))


# --------------------------------------------------------------------------
# docker/profile.sh: the container's environment.
# --------------------------------------------------------------------------


def profile(tmp_path, probe, **env):
    """Source docker/profile.sh in a clean bash and run probe."""
    proc = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", f". {PROFILE_SH}; {probe}"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), **env},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_the_container_sources_the_release_it_is_given(tmp_path):
    vivado = tmp_path / "2025.2/Vivado"
    vivado.mkdir(parents=True)
    (vivado / "settings64.sh").write_text(
        f'export XILINX_HLS={tmp_path}/2025.2/Vitis; export PATH="{vivado}/bin:$PATH"\n'
        "false  # a failing line in AMD's script does not stop the container\n"
    )
    out = profile(tmp_path, 'echo "$XILINX_HLS|$PATH|$FINN_PROFILE"', XILINX_VIVADO=str(vivado))
    hls, path, marker = out.strip().split("|")
    assert hls == f"{tmp_path}/2025.2/Vitis" and path.startswith(f"{vivado}/bin:")
    assert marker == "1"
    # Once per process tree: a child shell inherits, it does not source again.
    again = profile(tmp_path, f'. {PROFILE_SH}; echo "$PATH"', XILINX_VIVADO=str(vivado))
    assert again.count(f"{vivado}/bin") == 1


def test_the_container_without_tools_sets_no_amd_variables(tmp_path):
    out = profile(tmp_path, 'echo "${XILINX_HLS:-none}|$FINN_HOME"', HOME="/")
    hls, home = out.strip().split("|")
    assert hls == "none"
    # A caller uid with no writable home gets one under /tmp.
    assert home == f"/tmp/finn-home-{os.getuid()}/.finn"


# --------------------------------------------------------------------------
# Native use: uv sync and the venv, nothing of FINN's own.
# --------------------------------------------------------------------------


def test_finn_finds_its_checkout_without_finn_root(monkeypatch):
    from finn.util.basic import get_finn_root  # noqa: PLC0415

    monkeypatch.delenv("FINN_ROOT", raising=False)
    assert Path(get_finn_root()) == REPO
    monkeypatch.setenv("FINN_ROOT", "/elsewhere")
    assert get_finn_root() == "/elsewhere"


def test_one_uv_version():
    """uv's own pin (required-version) is the one the image and Travis install."""
    pin = re.search(r'required-version = "==([0-9.]+)"', (REPO / "pyproject.toml").read_text())
    assert pin, "pyproject.toml pins uv"
    assert f"astral-sh/uv:{pin[1]} " in (REPO / "docker/Dockerfile").read_text()
    assert f"uv=={pin[1]}" in (REPO / ".travis.yml").read_text()


def test_installers_do_not_hardcode_layouts_or_pins():
    """The pre-2024.2 layout, hardcoded, reported "Vivado not found" at a path
    the user could see; Python pins live only in pyproject.toml and uv.lock."""
    pin = re.compile(r"[A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,.\-]+\])?==[0-9]")
    for rel in ("scripts/install-system-deps.sh", "docker/Dockerfile"):
        body = (REPO / rel).read_text()
        for number, line in enumerate(body.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            assert "$FINN_XILINX_PATH/Vivado/" not in line, "%s:%d" % (rel, number)
        assert pin.findall(body) == [], rel


# --------------------------------------------------------------------------
# Container identity.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("/usr/lib/x86_64-linux-gnu/libnss_wrapper.so").exists(),
    reason="libnss_wrapper is installed in the container",
)
def test_container_identity_is_visible_to_python_and_openssh(tmp_path):
    import identity  # noqa: PLC0415

    home = tmp_path / "home"
    home.mkdir()
    values = identity.prepare(tmp_path / "nss", os.getuid(), os.getgid(), "finn-test", str(home))
    env = dict(
        os.environ,
        **values,
        HOME=str(home),
        LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnss_wrapper.so",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,pwd; assert pwd.getpwuid(os.getuid()).pw_name == 'finn-test'",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["ssh", "-G", "-F", "/dev/null", "user@127.0.0.1"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# The runtime profile and CI scripts.
# --------------------------------------------------------------------------


def test_builder_imports_without_development_packages(clean_env):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is needed to export the runtime profile")
    exported = subprocess.check_output(
        [uv, "export", "--locked", "--no-default-groups", "--no-emit-project", "--no-hashes"],
        cwd=REPO,
        text=True,
    )
    allowed = {"finn"}
    for line in exported.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            allowed.add(Requirement(line).name.lower().replace("_", "-"))
    code = """
import importlib.metadata as md
from importlib.machinery import PathFinder
import json, sys
allowed = set(json.loads(sys.argv[1]))
packages = md.packages_distributions()
original = PathFinder.find_spec
def runtime_spec(fullname, path=None, target=None):
    distributions = packages.get(fullname.split('.')[0], [])
    if distributions and not any(n.lower().replace('_','-') in allowed for n in distributions):
        return None
    return original(fullname, path, target)
PathFinder.find_spec = staticmethod(runtime_spec)
assert PathFinder.find_spec('pytest') is None
from finn.builder import build_dataflow
"""
    subprocess.run([sys.executable, "-c", code, json.dumps(sorted(allowed))], cwd=REPO, check=True)


def test_candidate_failure_cannot_publish_previous_results(clean_env, tmp_path, monkeypatch):
    # Use an isolated checkout-shaped cwd so real reports are not removed.
    root = tmp_path / "candidate"
    root.mkdir()
    for name in ("brevitas_suite.xml", "brevitas_suite.html"):
        (root / name).write_text("old report")
    for name in ("brevitas-provenance", "coverage_brevitas_suite"):
        (root / name).mkdir()
        (root / name / "old.json").write_text("old evidence")
    monkeypatch.setenv("FINN_ROOT", str(root))
    monkeypatch.setenv("BREVITAS_COMMIT", "invalid")
    result = subprocess.run(
        ["bash", str(REPO / "ci/scripts/brevitas_candidate.sh")], capture_output=True
    )
    assert result.returncode == 2
    assert list(root.iterdir()) == []
