# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each readable authoring example runs as one command on the real installed stack."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from finn.dataflow.descriptors import load
from finn.dataflow.logical import LogicalPlan

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("example", ("native", "compute"))
def test_one_command_emits_real_sources_and_an_untimed_logical_plan(
    example: str,
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "examples" / "untimed" / f"{example}.py"),
        "--out",
        str(tmp_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == json.loads((tmp_path / "summary.json").read_text())
    plan = load((tmp_path / "logical-plan.json").read_bytes())
    assert isinstance(plan, LogicalPlan)
    assert plan.digest == summary["logical_digest"]
    assert summary["timing"].startswith("not requested")
    expected = (
        {"rtl/shape/vpc.sv", "rtl/infra/fifo.sv"}
        if example == "native"
        else {"rtl/linalg/dotp_axi.sv", "rtl/infra/memstream_axi.sv", "rtl/shape/input_gen.sv"}
    )
    assert expected <= set(summary["sources"])
    assert all((tmp_path / "rtl" / name).is_file() for name in summary["sources"])
    assert all((tmp_path / "rtl" / name).is_file() for name in summary["data"])
    top = tmp_path / "rtl" / (summary["module"] + ".sv")
    assert top.is_file() and "module " in top.read_text()
    if example == "native":
        assert summary["native_fifo_capacities"]["buffer"] == {
            "requested_depth": 4,
            "realized_words": 5,
        }
        assert summary["packing"]["port/middle/read"]["payload_bits"] == 24
    else:
        assert summary["channel_fifo_capacities"]["x"] == {
            "requested_depth": 16,
            "realized_words": 16,
        }
        assert summary["requirements"]["x"]["steps"] == ["reorder", "markers"]
        assert "adapter.input_gen.input_gen" in summary["requirements"]["x"]["stages"]
        assert summary["data"]  # The actual known-weight memory image, not a placeholder.
