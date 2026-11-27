# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Where FINN keeps state: FINN_HOME (persistent, content-keyed) and the
build directory (disposable), with per-cache overrides."""
import pytest

import contextlib
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

from finn import xsi
from finn.resources import _cli, _store, prepare
from finn.util.basic import get_pynq_golden_dir, make_build_dir, tcl_string
from finn.util.paths import home, scratch
from finn.xsi import paths

pytestmark = pytest.mark.util


@pytest.fixture
def state(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("FINN_RESOURCES_", "XILINX_")) or name in (
            "FINN_HOME",
            "FINN_BUILD_DIR",
            "FINN_XSI_PATH",
            "FINN_GOLDEN_DIR",
            "FINN_BOARD_FILES_PATH",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FINN_RESOURCES_SYSTEM_CACHE", str(tmp_path / "no-system-cache"))
    return tmp_path / "home"


def vivado(tmp_path, name="2025.2", settings="# settings\n"):
    tool = tmp_path / name / "Vivado"
    tool.mkdir(parents=True)
    (tool / "settings64.sh").write_text(settings)
    return tool


def test_state_lives_under_finn_home_by_default(state, tmp_path, monkeypatch):
    monkeypatch.delenv("FINN_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert home() == tmp_path / ".finn"
    assert scratch() == Path("/tmp/finn_build_%d" % os.getuid())
    assert paths.root() == tmp_path / ".finn/xsi"
    assert prepare.golden_root() == tmp_path / ".finn/golden"


def test_build_directory_is_named_not_created(state, tmp_path, monkeypatch):
    assert scratch({"FINN_BUILD_DIR": str(tmp_path / "scratch")}) == tmp_path / "scratch"
    monkeypatch.chdir(tmp_path)
    assert scratch({"FINN_BUILD_DIR": "relative"}) == tmp_path / "relative"
    assert not (tmp_path / "scratch").exists() and not (tmp_path / "relative").exists()


def test_make_build_dir_needs_no_environment(state, tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "builds"))
    made = Path(make_build_dir("probe_"))
    assert made.parent == tmp_path / "builds" and made.name.startswith("probe_")
    monkeypatch.delenv("FINN_BUILD_DIR")
    made = Path(make_build_dir("probe_"))
    try:
        assert made.parent == scratch()
    finally:
        made.rmdir()


def test_xsi_builds_are_keyed_under_finn_home(state, tmp_path, monkeypatch):
    checkout = tmp_path / "source"
    (checkout / "finn_xsi").mkdir(parents=True)
    cpp = checkout / "finn_xsi/xsi_finn.cpp"
    cpp.write_text("// version one")
    monkeypatch.setenv("FINN_RESOURCES_FINN_XSI", str(checkout / "finn_xsi"))
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path)))
    original = paths.build_dir()
    assert original.parent == state / "xsi"
    cpp.write_text("// version two")
    changed_source = paths.build_dir()
    assert changed_source != original
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path, "2026.1")))
    assert paths.build_dir() != changed_source
    monkeypatch.setenv("FINN_XSI_PATH", str(tmp_path / "shared xsi"))
    assert paths.build_dir().parent == tmp_path / "shared xsi"
    assert not original.exists()


def test_xsi_key_ignores_the_checkout_location(state, tmp_path, monkeypatch):
    """Identical sources in workspace and workspace@2 share one build."""
    for name in ("checkout", "checkout@2"):
        (tmp_path / name / "finn_xsi").mkdir(parents=True)
        (tmp_path / name / "finn_xsi/xsi_finn.cpp").write_text("// same")
    monkeypatch.setenv("FINN_RESOURCES_FINN_XSI", str(tmp_path / "checkout/finn_xsi"))
    first = paths.build_dir()
    monkeypatch.setenv("FINN_RESOURCES_FINN_XSI", str(tmp_path / "checkout@2/finn_xsi"))
    assert paths.build_dir() == first


@pytest.mark.skipif(not shutil.which("tclsh"), reason="tclsh is not installed")
def test_resource_paths_reach_tcl_literally(tmp_path):
    path = str(tmp_path / 'odd $HOME [exec touch pwned] "q" \\ {b} ;x')
    script = tmp_path / "probe.tcl"
    script.write_text("set p %s\nputs -nonewline $p\n" % tcl_string(path))
    result = subprocess.run(["tclsh", str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert (result.returncode, result.stdout) == (0, path), result.stderr
    assert not (tmp_path / "pwned").exists()


@pytest.mark.skipif(not shutil.which("tclsh"), reason="tclsh is not installed")
def test_hls_include_paths_reach_the_compiler_whole(state, tmp_path, monkeypatch):
    from finn.custom_op.fpgadataflow import templates  # noqa: PLC0415
    from finn.custom_op.fpgadataflow.hlsbackend import (  # noqa: PLC0415
        hls_include_flags,
    )

    includes = [tmp_path / "hlslib", tmp_path / 'custom hls $HOME [x] "q" \\']
    for variable, include in zip(("FINN_RESOURCES_HLSLIB", "FINN_RESOURCES_CUSTOM_HLS"), includes):
        include.mkdir()
        monkeypatch.setenv(variable, str(include))
    line = next(line for line in templates.ipgentcl_template.splitlines() if "-cflags" in line)
    line = line.replace("$HLSINCLUDES$", hls_include_flags()).replace("$TOPFXN$", "top")
    stub = "set config_hwsrcdir src\nproc add_files {args} {puts -nonewline [lindex $args end]}\n"
    result = subprocess.run(["tclsh"], input=stub + line, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # Split as a GNU command line is: quotes group an argument, backslashes escape.
    assert shlex.split(result.stdout) == ["-std=c++14", *("-I" + str(i) for i in includes)]


def test_xsi_key_changes_with_flags_and_pybind_version(state, monkeypatch):
    from finn.xsi import _build_config  # noqa: PLC0415

    baseline = paths.build_dir()
    original = paths.get_build_paths
    monkeypatch.setattr(paths, "get_build_paths", lambda: (original()[0], original()[1], ["-O0"]))
    assert paths.build_dir() != baseline
    monkeypatch.setattr(paths, "get_build_paths", original)
    version = paths.importlib.metadata.version
    monkeypatch.setattr(
        paths.importlib.metadata,
        "version",
        lambda name: "different" if name == "pybind11" else version(name),
    )
    assert paths.build_dir() != baseline
    assert _build_config.LINK_ARGS


def test_missing_xsi_says_what_to_do(state, tmp_path, monkeypatch):
    monkeypatch.setattr(xsi, "_adapter_module", None)
    with pytest.raises(ImportError, match="XILINX_VIVADO is not set"):
        xsi.compile_sim_obj("top", [], str(tmp_path))
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path)))
    # On first use FINN builds the binding; against this stub installation that fails.
    monkeypatch.setattr(xsi, "_build_attempted", False)
    monkeypatch.setattr(xsi, "_build_failed", False)
    with pytest.raises(ImportError, match="did not build") as error:
        xsi.SimEngine()
    assert str(paths.build_dir() / "compile_xsi.stderr.log") in str(error.value)
    assert "python -m finn.xsi.setup --force" in str(error.value)


def test_golden_references_are_keyed_by_sources_and_vivado(state, tmp_path, monkeypatch):
    assert get_pynq_golden_dir("VCK190") == str(prepare.golden_dir("vck190"))
    unselected = prepare.golden_dir()
    assert unselected.parent == state / "golden" and unselected.name.startswith("vck190-")
    tool = vivado(tmp_path)
    monkeypatch.setenv("XILINX_VIVADO", str(tool))
    selected = prepare.golden_dir()
    assert selected != unselected
    (tool / "settings64.sh").write_text("# patched installation\n")
    assert prepare.golden_dir() != selected
    # A local PYNQ checkout is unverified, so its content keys the reference.
    pynq = tmp_path / "pynq"
    pynq.mkdir()
    monkeypatch.setenv("FINN_RESOURCES_PYNQ", str(pynq))
    first = prepare.golden_dir()
    (pynq / "golden.tcl").write_text("# edited\n")
    assert prepare.golden_dir() != first


def test_site_golden_directory_is_used_as_supplied(state, tmp_path, monkeypatch):
    site = tmp_path / "site golden"
    monkeypatch.setenv("FINN_GOLDEN_DIR", str(site))
    assert prepare.golden_dir("VCK190") == site / "vck190"
    (site / "vck190").mkdir(parents=True)
    for name in prepare.GOLDEN_FILES:
        (site / "vck190" / name).write_text("supplied")
    # Supplied files need neither Vivado nor the PYNQ sources.
    assert prepare.golden() == str(site / "vck190")


def test_clean_removes_keyed_state_but_not_site_state(state, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XILINX_VIVADO", str(vivado(tmp_path)))
    entries = {
        "resource": state / "resources/hlslib-0123456789abcdef",
        "xsi": paths.build_dir(),
        "stale xsi": state / "xsi" / ("0" * 20),
        "golden": prepare.golden_dir(),
        "stale golden": state / "golden/vck190-0123456789abcdef",
        "unrelated": state / "xsi/notes",
    }
    for entry in entries.values():
        entry.mkdir(parents=True)
    assert _cli.main(["clean", "--unused"]) == 0
    removed = capsys.readouterr().out.split("\n")
    assert sorted(line for line in removed if line) == sorted(
        "removed %s" % entries[name] for name in ("resource", "stale xsi", "stale golden")
    )
    assert entries["xsi"].is_dir() and entries["golden"].is_dir()
    site = tmp_path / "site"
    (site / "vck190").mkdir(parents=True)
    monkeypatch.setenv("FINN_GOLDEN_DIR", str(site))
    assert _cli.main(["clean"]) == 0
    assert not entries["xsi"].exists() and not entries["golden"].exists()
    assert entries["unrelated"].is_dir() and (site / "vck190").is_dir()


def test_concurrent_cleaners_remove_each_entry_once(state, capsys, monkeypatch):
    entry = state / "resources/hlslib-0123456789abcdef"
    entry.mkdir(parents=True)
    # Both cleaners have found the entry before either takes its lock.
    found = threading.Barrier(2)
    locked = _store.locked

    @contextlib.contextmanager
    def after_both_found(root, name):
        found.wait(timeout=10)
        with locked(root, name) as held:
            yield held

    monkeypatch.setattr(_store, "locked", after_both_found)
    outcomes = []

    def clean():
        try:
            outcomes.append(_cli.main(["clean"]))
        except Exception as error:
            outcomes.append(error)

    cleaners = [threading.Thread(target=clean) for _ in range(2)]
    for cleaner in cleaners:
        cleaner.start()
    for cleaner in cleaners:
        cleaner.join(30)
    assert outcomes == [0, 0]
    assert capsys.readouterr().out.count(f"removed {entry}\n") == 1
