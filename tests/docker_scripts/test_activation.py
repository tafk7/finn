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

from finn.util import launcher

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


@pytest.mark.parametrize("layout", ["2024.2/Vivado", "Vivado/2024.2"])
def test_a_jobs_install_root_and_release_select_the_release(tmp_path, layout):
    vivado = tmp_path / "Xilinx" / layout
    vivado.mkdir(parents=True)
    (vivado / "settings64.sh").write_text(f'export PATH="{vivado}/bin:$PATH"\n')
    job = {"FINN_XILINX_PATH": str(tmp_path / "Xilinx"), "FINN_XILINX_VERSION": "2024.2"}
    out = profile(tmp_path, 'echo "$XILINX_VIVADO|$PATH|${finn_vivado-unset}"', **job)
    selected, path, scratch = out.strip().split("|")
    assert selected == str(vivado) and path.startswith(f"{vivado}/bin:")
    assert scratch == "unset"
    # XILINX_VIVADO, when the launcher passes it, wins.
    other = tmp_path / "other/Vivado"
    other.mkdir(parents=True)
    (other / "settings64.sh").write_text("")
    out = profile(tmp_path, 'echo "$XILINX_VIVADO"', XILINX_VIVADO=str(other), **job)
    assert out.strip() == str(other)


@pytest.mark.parametrize(
    "version, layout, hls",
    [
        # xbuilds' separate trees: Vitis HLS inside Vitis from 2024.2, its own before.
        ("2024.2", "{tool}/{version}", None),
        ("2023.2", "{tool}/{version}", "Vitis_HLS"),
        ("2025.2", "{version}/{tool}", None),
    ],
)
def test_a_jobs_release_gets_vitis_and_run_dockers_names(tmp_path, version, layout, hls):
    root = tmp_path / "Xilinx"
    trees = {}
    for tool in ("Vivado", "Vitis", "Vitis_HLS"):
        trees[tool] = root / layout.format(tool=tool, version=version)
        if tool != "Vitis_HLS" or hls:
            trees[tool].mkdir(parents=True)
            (trees[tool] / "settings64.sh").write_text(
                f'export PATH="{trees[tool]}/bin:$PATH" SOURCED="$SOURCED {tool}"\n'
            )
    job = {"FINN_XILINX_PATH": str(root), "FINN_XILINX_VERSION": version}
    probe = 'echo "$XILINX_VITIS|$VIVADO_PATH|$VITIS_PATH|$HLS_PATH|$SOURCED|${finn_vitis-unset}"'
    vitis, vivado_path, vitis_path, hls_path, sourced, scratch = (
        profile(tmp_path, probe, **job).strip().split("|")
    )
    assert vitis == vitis_path == str(trees["Vitis"])
    assert vivado_path == str(trees["Vivado"])
    assert hls_path == str(trees["Vitis_HLS"] if hls else trees["Vitis"])
    assert sourced.split() == ["Vivado", "Vitis"] + (["Vitis_HLS"] if hls else [])
    assert scratch == "unset"


def test_a_unified_release_is_not_sourced_twice_and_given_names_stay(tmp_path):
    # From 2025.1, Vivado's settings64.sh sets up Vitis as well.
    vivado = tmp_path / "Xilinx/2025.2/Vivado"
    vitis = tmp_path / "Xilinx/2025.2/Vitis"
    for tree in (vivado, vitis):
        tree.mkdir(parents=True)
    (vivado / "settings64.sh").write_text(f"export XILINX_VITIS={vitis} SOURCED=1\n")
    (vitis / "settings64.sh").write_text("SOURCED=$((SOURCED + 1))\n")
    out = profile(
        tmp_path,
        'echo "$SOURCED|$VITIS_PATH|$HLS_PATH|$VIVADO_PATH"',
        FINN_XILINX_PATH=str(tmp_path / "Xilinx"),
        FINN_XILINX_VERSION="2025.2",
        VIVADO_PATH="/site/vivado",
    )
    assert out.strip() == f"1|{vitis}|{vitis}|/site/vivado"
    # The launcher's XILINX_VIVADO gets the names too.
    out = profile(tmp_path, 'echo "$VIVADO_PATH|$VITIS_PATH"', XILINX_VIVADO=str(vivado))
    assert out.strip() == f"{vivado}|{vitis}"


def test_a_callers_bash_env_runs_after_finns_and_its_tools_come_first(tmp_path):
    # As the FINN job starts the container: an LSF kit's enable.sh as BASH_ENV, which
    # puts its shim directory on PATH (once), and FINN_TOOL_DIR_OVERRIDE naming it.
    kit = tmp_path / "finn-lsf"
    (kit / "shim").mkdir(parents=True)
    (kit / "enable.sh").write_text(
        'case ":$PATH:" in *":KIT/shim:"*) ;; *) export PATH="KIT/shim:$PATH" ;; esac\n'
        'echo "enable.sh" >&2\n'.replace("KIT", str(kit))
    )
    vivado = tmp_path / "Xilinx/Vivado/2024.2"
    vivado.mkdir(parents=True)
    (vivado / "settings64.sh").write_text(f'export PATH="{vivado}/bin:$PATH"\n')
    entrypoint = tmp_path / "entrypoint"
    entrypoint.write_text(
        f"set -eo pipefail\n. {PROFILE_SH}\n"
        "exec /bin/bash -c 'echo \"$BASH_ENV|$FINN_CALLER_BASH_ENV|$PATH\"'\n"
    )
    proc = subprocess.run(
        ["/bin/bash", str(entrypoint)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "BASH_ENV": str(kit / "enable.sh"),
            "FINN_TOOL_DIR_OVERRIDE": str(kit / "shim"),
            "FINN_XILINX_PATH": str(tmp_path / "Xilinx"),
            "FINN_XILINX_VERSION": "2024.2",
        },
    )
    assert proc.returncode == 0, proc.stderr
    bash_env, caller, path = proc.stdout.strip().split("|")
    # The image's profile stays BASH_ENV; the caller's runs after it, in every bash.
    assert bash_env == "/etc/profile.d/finn.sh" and caller == str(kit / "enable.sh")
    assert proc.stderr.count("enable.sh") >= 2
    # The kit's tools ahead of the release's, once.
    assert path.split(":")[:2] == [str(kit / "shim"), f"{vivado}/bin"]
    assert path.split(":").count(str(kit / "shim")) == 1


def test_a_jobs_tool_directory_wins_over_the_release_on_path(tmp_path, monkeypatch):
    # The LSF kit's shims name no release; a vivado of another release on PATH is not
    # what the launcher picks.
    shim, other = tmp_path / "finn-lsf/shim", tmp_path / "Xilinx/2025.2/Vivado/bin"
    for directory in (shim, other):
        directory.mkdir(parents=True)
        (directory / "vivado").write_text("#!/bin/sh\n")
        (directory / "vivado").chmod(0o755)
    monkeypatch.setenv("XILINX_VIVADO", str(tmp_path / "Xilinx/Vivado/2024.2"))
    monkeypatch.setenv("PATH", f"{other}:{os.environ['PATH']}")
    monkeypatch.setenv(launcher.TOOL_DIR_ENV, str(shim))
    launcher.xilinx_tool.cache_clear()
    try:
        assert launcher.xilinx_tool("vivado") == str(shim / "vivado")
    finally:
        launcher.xilinx_tool.cache_clear()


def test_a_jobs_missing_release_stops_the_container_start(tmp_path):
    (tmp_path / "Xilinx/2025.2/Vivado").mkdir(parents=True)  # no settings64.sh
    entrypoint = tmp_path / "entrypoint"
    entrypoint.write_text(f"set -eo pipefail\n. {PROFILE_SH}\necho started\n")
    # As in the image: BASH_ENV sources the profile before the entrypoint does.
    proc = subprocess.run(
        ["/bin/bash", str(entrypoint)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "BASH_ENV": PROFILE_SH,
            "FINN_XILINX_PATH": str(tmp_path / "Xilinx"),
            "FINN_XILINX_VERSION": "2024.2",
        },
    )
    assert proc.returncode == 1 and "started" not in proc.stdout
    assert "no settings64.sh for FINN_XILINX_VERSION=2024.2" in proc.stderr


# --------------------------------------------------------------------------
# Native use: uv sync and the venv, nothing of FINN's own.
# --------------------------------------------------------------------------


def test_finn_finds_its_checkout_without_finn_root(monkeypatch):
    import finn  # noqa: PLC0415
    from finn.util.basic import get_finn_root  # noqa: PLC0415

    # an installed wheel has no checkout to find (CI's no-tools job tests the wheel)
    if not Path(finn.__path__[0]).is_relative_to(REPO):
        pytest.skip("FINN is installed from a wheel, not run from this checkout")
    monkeypatch.delenv("FINN_ROOT", raising=False)
    assert Path(get_finn_root()) == REPO
    monkeypatch.setenv("FINN_ROOT", "/elsewhere")
    assert get_finn_root() == "/elsewhere"


def test_the_image_and_travis_install_the_pinned_uv():
    """uv's own pin (required-version) is the one the image and Travis install."""
    pin = re.search(r'required-version = "==([0-9.]+)"', (REPO / "pyproject.toml").read_text())
    assert pin, "pyproject.toml pins uv"
    assert f"astral-sh/uv:{pin[1]} " in (REPO / "docker/Dockerfile").read_text()
    assert f"uv=={pin[1]}" in (REPO / ".travis.yml").read_text()


def test_python_pins_live_only_in_pyproject_and_the_lock():
    pin = re.compile(r"[A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,.\-]+\])?==[0-9]")
    for rel in ("scripts/install-system-deps.sh", "docker/Dockerfile"):
        assert pin.findall((REPO / rel).read_text()) == [], rel


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
