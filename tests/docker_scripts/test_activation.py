# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tool activation shared by native use and containers: finn-toolchain.sh,
scripts/activate.sh, container identity and the installers."""
import pytest

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from packaging.requirements import Requirement
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLCHAIN_SH = str(REPO / "docker/finn-toolchain.sh")
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
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_ROOT", str(REPO))
    monkeypatch.setenv("FINN_HOST_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "state"))


# --------------------------------------------------------------------------
# finn-toolchain.sh: sourced natively, by the entrypoint, BASH_ENV and shims.
# --------------------------------------------------------------------------


def toolchain_env(env):
    """os.environ minus everything finn-toolchain.sh reads, plus `env`.

    These tests also run inside the container, where XILINX_VIVADO is set and a
    real settings64.sh would put Vivado's bin on PATH. BASH_ENV and ENV go too:
    bash reads them even with --noprofile --norc.
    """
    child = dict(os.environ)
    child.pop("FINN_ENV_APPLIED", None)
    for key in (
        "BASH_ENV",
        "VIRTUAL_ENV",
        "ENV",
        "XILINX_VIVADO",
        "XILINX_VITIS",
        "XILINX_HLS",
        "XILINX_XRT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONPATH",
    ):
        child.pop(key, None)
    child.update(env)
    return child


def apply_toolchain(env, probe='printf "%s" "$PATH"'):
    proc = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", ". %s; %s" % (TOOLCHAIN_SH, probe)],
        capture_output=True,
        text=True,
        env=toolchain_env(env),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_toolchain_dedupes_path_variables():
    """settings64.sh prepends unconditionally; nested shells must not grow PATH."""
    assert apply_toolchain({"PATH": "/a:/b:/a:/c:/b"}) == "/a:/b:/c"
    assert apply_toolchain({"PATH": "/a::/b:"}) == "/a:/b"
    assert (
        apply_toolchain({"PATH": "/a", "LD_PRELOAD": "/x:/x"}, 'printf "%s" "$LD_PRELOAD"') == "/x"
    )


@pytest.mark.skipif(
    not glob.glob("/lib/*-linux-gnu/libudev.so.1"), reason="no system libudev to preload"
)
def test_toolchain_preloads_libudev_on_merged_usr_hosts():
    """/lib is a symlink on merged-/usr hosts; discovery must follow it."""
    preload = apply_toolchain({}, 'printf "%s" "$LD_PRELOAD"')
    assert preload.endswith("-linux-gnu/libudev.so.1")


def test_toolchain_is_idempotent_and_silent():
    """BASH_ENV sources this before every `bash -c`, so a stray echo corrupts
    the output of every command in the image."""
    assert apply_toolchain({"PATH": "/a"}, probe="true") == ""
    proc = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            '. {0}; . {0}; printf "%s|%s" "$PATH" "$FINN_ENV_APPLIED"'.format(TOOLCHAIN_SH),
        ],
        capture_output=True,
        text=True,
        env=toolchain_env({"PATH": "/a:/b"}),
    )
    assert proc.stdout == "/a:/b|1", proc.stderr


def test_vendor_settings_tolerate_unset_variables_and_keep_selected_python(tmp_path):
    tool = tmp_path / "Vivado"
    tool.mkdir()
    (tool / "settings64.sh").write_text(
        'echo "${SITE_UNSET_SETTING}" >/dev/null\nexport PATH="/vendor/python:$PATH"\n'
    )
    env = toolchain_env({"XILINX_VIVADO": str(tool), "VIRTUAL_ENV": str(tmp_path / "venv")})
    result = subprocess.run(
        ["bash", "-eu", "-c", 'source "$1"; test "${PATH%%:*}" = "$VIRTUAL_ENV/bin"', "probe"]
        + [TOOLCHAIN_SH],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_farm_wrappers_do_not_shadow_bare_tools(clean_env, tmp_path, monkeypatch):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    for tool in ("make", "vivado"):
        (wrappers / tool).write_text("#!/bin/sh\nexit 99\n")
        (wrappers / tool).chmod(0o755)
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(wrappers))
    result = subprocess.run(
        ["bash", "-c", ". docker/finn-toolchain.sh; command -v make"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != str(wrappers / "make")
    from finn.util.basic import resolve_xilinx_tool  # noqa: PLC0415

    assert resolve_xilinx_tool("vivado") == str(wrappers / "vivado")


def test_slashkit_has_a_transparent_tool_shim():
    assert re.search(
        r"for t in [^;\n]*\bslashkit\b[^;\n]*; do \\\n"
        r'\s*ln -sf /usr/local/lib/finn/toolchain-shim /usr/local/lib/finn/toolchain-bin/"\$t"',
        (REPO / "docker/Dockerfile.finn").read_text(),
    )


# --------------------------------------------------------------------------
# Native activation.
# --------------------------------------------------------------------------


def standin_checkout(tmp_path):
    """Activation's own file, with a test-owned virtual environment."""
    checkout = tmp_path / "finn"
    (checkout / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / "scripts/activate.sh", checkout / "scripts/activate.sh")
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/activate").write_text(
        'PATH="%s:$PATH"  # test-owned virtual environment stand-in\n'
        % os.path.dirname(sys.executable)
    )
    return checkout


def activate(checkout, tmp_path, probe, **env):
    return subprocess.run(
        ["bash", "-c", 'source "$1/scripts/activate.sh" && ' + probe, "test", str(checkout)],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), **env},
        text=True,
        capture_output=True,
    )


def test_activation_takes_the_shells_tools_and_sets_nothing_else(tmp_path):
    """Native use is the venv plus AMD's own settings: activation neither selects
    tools nor allocates FINN's directories."""
    checkout = standin_checkout(tmp_path)
    vivado = "/tools/Xilinx/2025.2/Vivado"
    proc = activate(
        checkout,
        tmp_path,
        'test "$FINN_ROOT" = "$1" && test "$XILINX_VIVADO" = "%s" && env | sort' % vivado,
        XILINX_VIVADO=vivado,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    names = {line.split("=", 1)[0] for line in proc.stdout.splitlines()}
    assert {name for name in names if name.startswith("FINN_")} == {"FINN_ROOT"}
    assert "LD_LIBRARY_PATH" not in names and "LD_PRELOAD" not in names
    assert not (tmp_path / "home").exists()


def test_activation_without_tools_says_what_to_source(tmp_path):
    checkout = standin_checkout(tmp_path)
    proc = activate(checkout, tmp_path, "true")
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.startswith("FINN: no AMD tools in this shell. ")
    assert "settings64.sh" in proc.stderr


def test_activation_needs_the_environment(tmp_path):
    checkout = standin_checkout(tmp_path)
    (checkout / ".venv/bin/activate").unlink()
    proc = activate(checkout, tmp_path, "true")
    assert proc.returncode == 1
    assert "run %s/setup-local.sh" % checkout in proc.stderr


def test_installers_do_not_hardcode_layouts_or_pins():
    """The pre-2024.2 layout, hardcoded, reported "Vivado not found" at a path
    the user could see; Python pins live only in pyproject.toml and uv.lock."""
    pin = re.compile(r"[A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,.\-]+\])?==[0-9]")
    for rel in ("setup-local.sh", "scripts/activate.sh", "docker/Dockerfile.finn"):
        body = (REPO / rel).read_text()
        for number, line in enumerate(body.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            assert "$FINN_XILINX_PATH/Vivado/" not in line, "%s:%d" % (rel, number)
        assert pin.findall(body) == [], rel


def test_nothing_writes_to_the_checkout_by_default():
    """Persistent state belongs in FINN_HOME and scratch in FINN_BUILD_DIR."""
    for rel in ("setup-local.sh", "scripts/activate.sh", "scripts/ensure-uv.sh"):
        for line in (REPO / rel).read_text().splitlines():
            if not line.lstrip().startswith("#"):
                assert ".cache/" not in line, (rel, line)


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
