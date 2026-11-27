# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import subprocess
import sys
from pathlib import Path

from finn import doctor
from finn.util import launcher, releases
from finn_test_helpers import fake_amd_release, source_fake_release

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for name in launcher.AMD_VARIABLES:
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
    fake_amd_release(clean, "2025.2")
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
    vivado, vitis = fake_amd_release(clean, "2025.2")
    fake_amd_release(clean, "2024.2")
    source_fake_release(monkeypatch, vivado, vitis)
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
    vivado, vitis = fake_amd_release(clean, "2024.2")
    source_fake_release(monkeypatch, vivado, vitis)
    tools = states(doctor.diagnose(), "AMD tools")
    assert tools["vitis_hls"] == doctor.OK and "vitis-run" not in tools


@pytest.mark.util
def test_tools_of_two_releases_fail(clean, monkeypatch):
    vivado, _ = fake_amd_release(clean, "2025.2")
    _, vitis = fake_amd_release(clean, "2024.2")
    source_fake_release(monkeypatch, vivado, vitis)
    report = doctor.diagnose()
    assert report.failed()
    assert states(report)["release"] == doctor.FAIL
    assert doctor.main([]) == 1


@pytest.mark.util
def test_missing_hls_headers_fail(clean, monkeypatch):
    vivado, vitis = fake_amd_release(clean, "2025.2")
    (vitis / "include" / "hls_stream.h").unlink()
    source_fake_release(monkeypatch, vivado, vitis)
    assert states(doctor.diagnose())["HLS headers"] == doctor.FAIL


@pytest.mark.util
def test_the_release_table_is_printed_for_the_release_in_use(clean, monkeypatch):
    vivado, vitis = fake_amd_release(clean, "2026.2")
    source_fake_release(monkeypatch, vivado, vitis)
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
def test_every_release_note_is_used_by_code():
    """The table stays with the code: each entry's key appears where FINN acts on it."""
    sources = "".join(
        path.read_text()
        for path in (REPOSITORY / "src" / "finn").rglob("*.py")
        if path.name != "releases.py"
    )
    for key in releases.NOTES:
        assert key in sources, f"no code names the release note {key!r}"


@pytest.mark.util
def test_release_ranges():
    note = releases.ReleaseNote("Vivado", None, (2024, 1), "", "")
    assert note.applies_to((2023, 2)) and note.applies_to((2024, 1))
    assert not note.applies_to((2024, 2))
    assert note.releases() == "up to 2024.1"
    note = releases.ReleaseNote("Vitis", (2024, 2), None, "", "")
    assert not note.applies_to((2024, 1)) and note.applies_to((2026, 2))
    assert note.releases() == "2024.2 and later"
    assert releases.minimum("no-slash") == "2025.1" and releases.minimum("uram-init") == "2024.2"
    assert not releases.affects("uram-init", None)
