# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
import subprocess
import sys
from pathlib import Path

from finn import doctor
from finn.util import launcher, releases

REPOSITORY = Path(__file__).resolve().parents[2]


def install(root, release, tools=("vivado", "xelab", "vitis-run", "vitis_hls", "v++")):
    """A fake AMD release in the 2025.1+ layout: <root>/<release>/{Vivado,Vitis}."""
    vivado, vitis = root / release / "Vivado", root / release / "Vitis"
    for directory in (vivado / "bin", vitis / "bin", vitis / "include"):
        directory.mkdir(parents=True, exist_ok=True)
    (vivado / "settings64.sh").write_text("")
    (vitis / "include" / "hls_stream.h").write_text("")
    for tool in tools:
        home = vivado if tool in ("vivado", "xelab") else vitis
        (home / "bin" / tool).write_text("#!/bin/sh\n")
        (home / "bin" / tool).chmod(0o755)
    return vivado, vitis


def source(monkeypatch, vivado, vitis):
    """What settings64.sh sets, as far as FINN reads it."""
    monkeypatch.setenv("XILINX_VIVADO", str(vivado))
    monkeypatch.setenv("XILINX_HLS", str(vitis))
    monkeypatch.setenv("XILINX_VITIS", str(vitis))
    monkeypatch.setenv("PATH", f"{vivado}/bin:{vitis}/bin:{os.environ['PATH']}")


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for name in (*launcher.AMD_VARIABLES, *launcher.DEPRECATED_NAMES.values()):
        monkeypatch.delenv(name, raising=False)
    for name in (launcher.TOOL_DIR_ENV, "PLATFORM_REPO_PATHS", "XILINX_XRT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FINN_XSI_PATH", str(tmp_path / "xsi"))
    monkeypatch.setattr(launcher, "INSTALL_ROOTS", (str(tmp_path / "Xilinx"),))
    launcher.xilinx_tool.cache_clear()
    yield tmp_path / "Xilinx"
    launcher.xilinx_tool.cache_clear()


def states(report, section=None):
    return {f.label: f.state for f in report.findings if section in (None, f.section)}


@pytest.mark.util
def test_without_tools_the_software_flow_is_reported_and_nothing_fails(clean):
    install(clean, "2025.2")
    report = doctor.diagnose()
    assert not report.failed()
    release = next(f for f in report.findings if f.label == "release")
    assert release.state == doctor.NOTE
    assert f"source {clean}/2025.2/Vivado/settings64.sh" in release.detail
    can = states(report, "FINN can run")
    assert can["transformations, estimates, code generation"] == doctor.OK
    assert {can["cppsim"], can["rtlsim"], can["Alveo builds"]} == {doctor.NOTE}


@pytest.mark.util
def test_a_sourced_release_is_reported_with_what_it_runs(clean, monkeypatch):
    vivado, vitis = install(clean, "2025.2")
    install(clean, "2024.2")
    source(monkeypatch, vivado, vitis)
    report = doctor.diagnose()
    assert not report.failed()
    assert report.release == (2025, 2)
    tools = states(report, "AMD tools")
    assert tools["vivado"] == tools["vitis-run"] == tools["HLS headers"] == doctor.OK
    assert "vitis_hls" not in tools
    assert next(f.detail for f in report.findings if f.label == "also installed") == "2024.2"
    can = states(report, "FINN can run")
    assert can["stitched IP, Zynq and Versal bitfiles"] == can["HLS synthesis"] == doctor.OK
    assert can["Alveo builds"] == doctor.NOTE


@pytest.mark.util
def test_releases_before_2025_1_run_vitis_hls(clean, monkeypatch):
    vivado, vitis = install(clean, "2024.2")
    source(monkeypatch, vivado, vitis)
    tools = states(doctor.diagnose(), "AMD tools")
    assert tools["vitis_hls"] == doctor.OK and "vitis-run" not in tools


@pytest.mark.util
def test_tools_of_two_releases_fail(clean, monkeypatch):
    vivado, _ = install(clean, "2025.2")
    _, vitis = install(clean, "2024.2")
    source(monkeypatch, vivado, vitis)
    report = doctor.diagnose()
    assert report.failed()
    assert states(report)["release"] == doctor.FAIL
    assert doctor.main([]) == 1


@pytest.mark.util
def test_missing_hls_headers_fail(clean, monkeypatch):
    vivado, vitis = install(clean, "2025.2")
    (vitis / "include" / "hls_stream.h").unlink()
    source(monkeypatch, vivado, vitis)
    assert states(doctor.diagnose())["HLS headers"] == doctor.FAIL


@pytest.mark.util
def test_the_release_table_is_printed_for_the_release_in_use(clean, monkeypatch):
    vivado, vitis = install(clean, "2026.2")
    source(monkeypatch, vivado, vitis)
    report = doctor.diagnose()
    notes = [f for f in report.findings if f.section == "Known differences of release 2026.2"]
    assert len(notes) == len(releases.notes_for((2026, 2)))
    assert any("sitecustomize.py" in f.detail for f in notes)


@pytest.mark.util
def test_finn_doctor_runs_as_a_command(clean):
    result = subprocess.run(
        [sys.executable, "-m", "finn.cli", "doctor", "--json"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert '"findings"' in result.stdout


@pytest.mark.util
def test_every_release_note_names_code_that_exists():
    """The table stays with the code that handles each entry."""
    for note in releases.NOTES:
        assert note.where
        for path, text in note.where:
            assert text in (REPOSITORY / path).read_text(), f"{path} no longer has {text!r}"


@pytest.mark.util
def test_release_ranges():
    note = releases.ReleaseNote("Vivado", None, (2024, 1), "", "", ())
    assert note.applies_to((2023, 2)) and note.applies_to((2024, 1))
    assert not note.applies_to((2024, 2))
    assert note.releases() == "up to 2024.1"
    note = releases.ReleaseNote("Vitis", (2024, 2), None, "", "", ())
    assert not note.applies_to((2024, 1)) and note.applies_to((2026, 2))
    assert note.releases() == "2024.2 and later"
