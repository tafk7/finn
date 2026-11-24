# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""FINN reads the AMD tools' own environment (XILINX_VIVADO, XILINX_HLS, XILINX_VITIS
and PATH, as a release's settings64.sh sets them) and never builds it."""
import pytest

import logging
import os

from finn.util import launcher
from finn.util.basic import get_vivado_version
from finn.util.launcher import (
    ToolchainMismatch,
    ToolUnavailable,
    amd_path,
    expected_release,
    hls_root,
    installed_releases,
    release_of,
    setup_hint,
    xilinx_tool,
)

pytestmark = pytest.mark.util

NAMES = ("XILINX_VIVADO", "XILINX_HLS", "XILINX_VITIS", "VIVADO_PATH", "HLS_PATH", "VITIS_PATH")


@pytest.fixture
def bare(monkeypatch, tmp_path):
    """An environment without AMD tools, with no install roots to find."""
    for name in NAMES + ("FINN_TOOL_DIR_OVERRIDE",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setattr(launcher, "INSTALL_ROOTS", (str(tmp_path / "no-root"),))
    monkeypatch.setattr(launcher, "_warned_names", set())
    xilinx_tool.cache_clear()
    yield
    xilinx_tool.cache_clear()


def release(root, version, new_layout=True):
    """A stub AMD release under root: settings64.sh, bin/vivado, bin/xelab, HLS headers."""
    vivado = root / version / "Vivado" if new_layout else root / "Vivado" / version
    vitis = root / version / "Vitis" if new_layout else root / "Vitis" / version
    for directory in (vivado / "bin", vitis / "bin", vitis / "include"):
        directory.mkdir(parents=True)
    (vivado / "settings64.sh").write_text("")
    (vitis / "include" / "hls_stream.h").write_text("")
    for tool in (vivado / "bin" / "vivado", vivado / "bin" / "xelab", vitis / "bin" / "v++"):
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
    return vivado, vitis


def source(monkeypatch, vivado, vitis):
    """What settings64.sh does for a release."""
    monkeypatch.setenv("XILINX_VIVADO", str(vivado))
    monkeypatch.setenv("XILINX_HLS", str(vitis))
    monkeypatch.setenv("XILINX_VITIS", str(vitis))
    monkeypatch.setenv("PATH", os.pathsep.join([str(vitis / "bin"), str(vivado / "bin")]))


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/tools/Xilinx/2025.2/Vivado", (2025, 2)),
        ("/opt/Xilinx/Vivado/2024.2", (2024, 2)),
        ("/proj/xbuilds/SWIP/2022.2_1014_8888/installs/lin64/Vivado/2022.2", (2022, 2)),
        ("/opt/Xilinx/Vivado/current", None),
        ("/opt/Xilinx/2025.21/Vivado", None),
    ],
)
def test_release_of(path, expected):
    assert release_of(path) == expected


def test_the_deprecated_names_are_read_with_a_warning(bare, monkeypatch, caplog):
    monkeypatch.setenv("VIVADO_PATH", "/opt/Xilinx/Vivado/2024.2")
    with caplog.at_level(logging.WARNING, logger="finn.util.launcher"):
        assert amd_path("XILINX_VIVADO") == "/opt/Xilinx/Vivado/2024.2"
        assert get_vivado_version() == (2024, 2)
    assert caplog.text.count("VIVADO_PATH is deprecated") == 1
    monkeypatch.setenv("XILINX_VIVADO", "/tools/Xilinx/2025.2/Vivado")
    assert amd_path("XILINX_VIVADO") == "/tools/Xilinx/2025.2/Vivado"


def test_installed_releases_are_found_in_both_layouts_newest_first(tmp_path):
    release(tmp_path / "new", "2025.2")
    release(tmp_path / "new", "2025.1")
    release(tmp_path / "old", "2024.2", new_layout=False)
    found = installed_releases([str(tmp_path / "new"), str(tmp_path / "old"), "/no/such/root"])
    assert [r for r, _ in found] == [(2025, 2), (2025, 1), (2024, 2)]
    assert found[0][1] == str(tmp_path / "new/2025.2/Vivado/settings64.sh")
    assert found[2][1] == str(tmp_path / "old/Vivado/2024.2/settings64.sh")


def test_the_hint_names_the_newest_release_and_never_uses_it(bare, monkeypatch, tmp_path):
    release(tmp_path / "root", "2024.2")
    release(tmp_path / "root", "2025.2")
    monkeypatch.setattr(launcher, "INSTALL_ROOTS", (str(tmp_path / "root"),))
    script = tmp_path / "root/2025.2/Vivado/settings64.sh"
    assert setup_hint() == f"To use Vivado 2025.2: source {script} (also installed: 2024.2)"
    with pytest.raises(ToolUnavailable, match="No AMD tools are set up.*source .*2025.2"):
        xilinx_tool("vivado")
    assert amd_path("XILINX_VIVADO") is None
    assert get_vivado_version() is None


def test_without_installs_the_hint_says_how_to_get_them(bare):
    assert "source a release's settings64.sh" in setup_hint()


def test_a_sourced_release_is_used(bare, monkeypatch, tmp_path):
    vivado, vitis = release(tmp_path, "2025.2")
    source(monkeypatch, vivado, vitis)
    assert expected_release() == (2025, 2)
    assert xilinx_tool("vivado") == str(vivado / "bin" / "vivado")
    assert hls_root() == str(vitis)


def test_variables_from_two_releases_are_refused(bare, monkeypatch, tmp_path):
    vivado, _ = release(tmp_path, "2025.2")
    _, vitis = release(tmp_path, "2024.2", new_layout=False)
    source(monkeypatch, vivado, vitis)
    with pytest.raises(ToolchainMismatch, match="different releases"):
        xilinx_tool("vivado")


def test_a_tool_from_another_release_on_path_is_refused(bare, monkeypatch, tmp_path):
    vivado, vitis = release(tmp_path, "2025.2")
    other, _ = release(tmp_path / "other", "2024.2")
    source(monkeypatch, vivado, vitis)
    monkeypatch.setenv("PATH", os.pathsep.join([str(other / "bin"), os.environ["PATH"]]))
    with pytest.raises(ToolchainMismatch, match="release 2024.2.*release 2025.2"):
        xilinx_tool("vivado")


def test_a_wrapper_without_a_release_is_accepted(bare, monkeypatch, tmp_path):
    vivado, vitis = release(tmp_path, "2025.2")
    source(monkeypatch, vivado, vitis)
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "vivado").write_text("#!/bin/sh\n")
    (shims / "vivado").chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(shims), os.environ["PATH"]]))
    assert xilinx_tool("vivado") == str(shims / "vivado")


def test_hls_headers_decide_between_xilinx_hls_and_xilinx_vitis(bare, monkeypatch, tmp_path):
    _, vitis = release(tmp_path, "2024.2", new_layout=False)
    hls_without_headers = tmp_path / "Vitis_HLS" / "2024.2"
    hls_without_headers.mkdir(parents=True)
    monkeypatch.setenv("XILINX_HLS", str(hls_without_headers))
    monkeypatch.setenv("XILINX_VITIS", str(vitis))
    assert hls_root() == str(vitis)
    monkeypatch.delenv("XILINX_VITIS")
    with pytest.raises(ToolUnavailable, match="no HLS headers"):
        hls_root()
    monkeypatch.delenv("XILINX_HLS")
    with pytest.raises(ToolUnavailable, match="XILINX_HLS and XILINX_VITIS are not set"):
        hls_root()
