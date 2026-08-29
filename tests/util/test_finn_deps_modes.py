"""Tests for FINN_DEPS mode resolution in docker/finn_paths.py.

The point of these is the `live` case. Before the three modes existed, the
default was called `live` but silently fell back to the baked wheels when a
checkout was absent -- auto behaviour under a name that promises determinism.
An unattended run could resolve either way with nothing in the output saying
which, which is precisely the failure an image digest is supposed to rule out.
"""

import importlib
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "docker"))

import finn_paths  # noqa: E402


@pytest.fixture
def deps_env(monkeypatch):
    """Isolate FINN_DEPS and FINN_ROOT from the ambient container environment."""
    monkeypatch.delenv("FINN_DEPS", raising=False)
    monkeypatch.delenv("FINN_ROOT", raising=False)
    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    return monkeypatch


def _workspace(tmp_path, with_deps):
    """A synthetic FINN checkout, optionally with the dependency sources."""
    root = tmp_path / "finn"
    (root / "src" / "finn").mkdir(parents=True)
    if with_deps:
        for rel in finn_paths.DEP_SRC_DIRS:
            (root / rel).mkdir(parents=True)
    return str(root)


def test_default_is_auto(deps_env):
    assert finn_paths.deps_mode() == "auto"


@pytest.mark.parametrize("mode", ["frozen", "live", "auto"])
def test_modes_round_trip(deps_env, mode):
    deps_env.setenv("FINN_DEPS", mode)
    assert finn_paths.deps_mode() == mode


def test_mode_is_case_insensitive(deps_env):
    deps_env.setenv("FINN_DEPS", "FROZEN")
    assert finn_paths.deps_mode() == "frozen"


def test_unknown_mode_warns_and_falls_back(deps_env, capsys):
    deps_env.setenv("FINN_DEPS", "sometimes")
    assert finn_paths.deps_mode() == "auto"
    assert "not one of" in capsys.readouterr().err


def test_frozen_excludes_dep_sources(deps_env, tmp_path):
    root = _workspace(tmp_path, with_deps=True)
    deps_env.setenv("FINN_ROOT", root)
    deps_env.setenv("FINN_DEPS", "frozen")
    dirs = finn_paths.source_dirs()
    # FINN's own src is live in every mode -- it is never baked, so there is no
    # wheel for frozen to fall back to.
    assert dirs == [os.path.join(root, "src")]


def test_auto_includes_dep_sources_when_present(deps_env, tmp_path):
    root = _workspace(tmp_path, with_deps=True)
    deps_env.setenv("FINN_ROOT", root)
    deps_env.setenv("FINN_DEPS", "auto")
    dirs = finn_paths.source_dirs()
    assert len(dirs) == 1 + len(finn_paths.DEP_SRC_DIRS)


def test_auto_tolerates_missing_dep_sources(deps_env, tmp_path):
    """auto is the mode that means 'use whatever is there'."""
    root = _workspace(tmp_path, with_deps=False)
    deps_env.setenv("FINN_ROOT", root)
    deps_env.setenv("FINN_DEPS", "auto")
    finn_paths.source_dirs()   # must not raise


def test_live_fails_loudly_when_a_checkout_is_missing(deps_env, tmp_path):
    """The whole reason `live` is distinct from `auto`."""
    root = _workspace(tmp_path, with_deps=False)
    deps_env.setenv("FINN_ROOT", root)
    deps_env.setenv("FINN_DEPS", "live")
    with pytest.raises(finn_paths.DepsUnavailable) as excinfo:
        finn_paths.source_dirs()
    message = str(excinfo.value)
    # Name the missing thing and the way out, not just the fact of failure.
    assert "qonnx" in message
    assert "fetch-repos.sh" in message


def test_live_succeeds_when_checkouts_are_present(deps_env, tmp_path):
    root = _workspace(tmp_path, with_deps=True)
    deps_env.setenv("FINN_ROOT", root)
    deps_env.setenv("FINN_DEPS", "live")
    dirs = finn_paths.source_dirs()
    assert len(dirs) == 1 + len(finn_paths.DEP_SRC_DIRS)


def test_live_failure_is_not_swallowed_at_interpreter_startup(tmp_path):
    """finn_paths runs from a .pth, where a raise would normally be caught.

    Everything else in that module is deliberately swallowed so an unrelated
    Python process in the image cannot be broken by it. FINN_DEPS=live is the
    one case that must survive that handler -- otherwise asking for strictness
    gets you silence.
    """
    root = _workspace(tmp_path, with_deps=False)
    env = dict(os.environ)
    env.update({"FINN_ROOT": root, "FINN_DEPS": "live"})
    env.pop("WORKSPACE_DIR", None)
    proc = subprocess.run(
        [sys.executable, "-c", "import finn_paths"],
        capture_output=True, text=True, env=env,
        cwd=os.path.join(REPO, "docker"))
    assert proc.returncode == 1
    assert "FINN_DEPS=live" in proc.stderr


def test_deps_are_live_matches_mode(deps_env):
    deps_env.setenv("FINN_DEPS", "frozen")
    assert finn_paths.deps_are_live() is False
    deps_env.setenv("FINN_DEPS", "auto")
    assert finn_paths.deps_are_live() is True
    deps_env.setenv("FINN_DEPS", "live")
    assert finn_paths.deps_are_live() is True
