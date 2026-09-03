# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ReplayBuffer source ownership across declarative and legacy RTL consumers."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
from finn.custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl import VVAU_rtl
from finn.custom_op.fpgadataflow.rtlbackend import get_finnlib_root
from finn.dataflow.kernels.replay_buffer import FINNLIB_ROOT, FINNLIB_SOURCES

FINN_ROOT = Path(__file__).parents[2]
REPLAY_PATH = Path("rtl/infra/replay_buffer.sv")


def _fake_operator(cls: type[Any], attributes: dict[str, object]) -> Any:
    operator = object.__new__(cls)
    operator.onnx_node = SimpleNamespace(name="dut")
    operator.get_nodeattr = lambda name: attributes[name]
    operator.get_verilog_top_module_intf_names = lambda: {"clk": ("ap_clk",)}
    return operator


def _assert_legacy_source_resolution(operator: Any, finnlib: Path) -> None:
    expected = finnlib / REPLAY_PATH
    absolute = tuple(Path(item) for item in operator.get_rtl_file_list(abspath=True))
    relative = tuple(operator.get_rtl_file_list(abspath=False))
    paths = tuple(Path(item) for item in operator.get_verilog_paths())
    commands: list[str] = []
    operator.instantiate_ip(commands)

    assert absolute.count(expected) == 1
    assert "replay_buffer.sv" in relative
    assert expected.parent in paths
    assert commands.count(f"add_files -norecurse {expected}") == 1
    assert all("finn-rtllib/mvu/replay_buffer.sv" not in item for item in commands)


def test_the_kernel_declares_one_finnlib_source() -> None:
    assert FINNLIB_ROOT == "finnlib"
    assert FINNLIB_SOURCES == (str(REPLAY_PATH),)


def test_legacy_mvu_and_vvau_resolve_the_caller_selected_finnlib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    finnlib = tmp_path / "selected-finnlib"
    monkeypatch.setenv("FINNLIB_ROOT", str(finnlib))
    monkeypatch.setenv("FINN_ROOT", str(FINN_ROOT))

    mvau = _fake_operator(
        MVAU_rtl,
        {
            "code_gen_dir_ipgen": str(tmp_path / "generated"),
            "gen_top_module": "mvau_top",
            "TH": 1,
            "mem_mode": "internal_embedded",
            "mlo_max_iter": 0,
            "pumpedCompute": 0,
        },
    )
    vvau = _fake_operator(
        VVAU_rtl,
        {
            "code_gen_dir_ipgen": str(tmp_path / "generated"),
            "gen_top_module": "vvau_top",
            "mem_mode": "internal_embedded",
        },
    )

    _assert_legacy_source_resolution(mvau, finnlib)
    _assert_legacy_source_resolution(vvau, finnlib)


def test_legacy_resolution_defaults_to_the_pinned_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FINNLIB_ROOT", raising=False)
    monkeypatch.setenv("FINN_ROOT", str(FINN_ROOT))
    assert Path(get_finnlib_root()) == FINN_ROOT / "deps/finnlib"


def test_only_finnlib_contains_the_live_replay_module() -> None:
    # The selected checkout is supplied by the caller for local branches and
    # defaults to FINN's pinned dependency in normal installations.
    selected = Path(get_finnlib_root())
    canonical = selected / REPLAY_PATH
    assert canonical.is_file()
    assert not (FINN_ROOT / "finn-rtllib/mvu/replay_buffer.sv").exists()

    declaration = re.compile(r"\bmodule\s+replay_buffer\b")
    definitions = [
        path
        for root in (FINN_ROOT / "finn-rtllib", selected / "rtl")
        for path in root.rglob("*.sv")
        if declaration.search(path.read_text())
    ]
    assert definitions == [canonical]
