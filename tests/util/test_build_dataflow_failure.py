# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A failing build step, with the default ``enable_build_pdb_debug``: the postmortem
debugger starts only where stdin is a terminal, so a build under pytest or a batch job
reports its own error and fails, rather than pdb's failed read."""

import pytest

import io
import sys
from pathlib import Path

import finn.builder.build_dataflow as build
from finn.util.basic import make_build_dir

# finn.qnn-data's fixtures are not in the wheel: read them from this checkout.
QNN_DATA = Path(__file__).resolve().parents[2] / "src" / "finn" / "qnn-data"


class Stdin(io.StringIO):
    """A stdin that is a terminal or not, and that no one may read."""

    def __init__(self, terminal: bool):
        super().__init__()
        self.terminal = terminal

    def isatty(self):
        return self.terminal

    def read(self, *args):
        raise AssertionError("the build read stdin")

    readline = read


def step_fails(model, cfg):
    raise RuntimeError("the step's own error")


def failed_build(monkeypatch, terminal: bool) -> tuple[int, list]:
    """The build's status and the tracebacks the debugger was started on."""
    debugged = []
    monkeypatch.setattr(sys, "stdin", Stdin(terminal))
    monkeypatch.setattr(build.pdb, "post_mortem", debugged.append)
    cfg = build.DataflowBuildConfig(
        output_dir=make_build_dir("test_build_failure_"),
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        steps=[step_fails],
        generate_outputs=[],
    )
    assert cfg.enable_build_pdb_debug
    return build.build_dataflow_cfg(str(QNN_DATA / "build_dataflow" / "model.onnx"), cfg), debugged


@pytest.mark.util
def test_a_failed_step_without_a_terminal_reports_its_error(monkeypatch, capsys):
    status, debugged = failed_build(monkeypatch, terminal=False)
    assert status == -1 and debugged == []
    captured = capsys.readouterr()
    assert "RuntimeError: the step's own error" in captured.err
    assert "stdin is not a terminal" in captured.out and "Build failed" in captured.out


@pytest.mark.util
def test_a_failed_step_at_a_terminal_starts_the_debugger(monkeypatch):
    status, debugged = failed_build(monkeypatch, terminal=True)
    assert status == -1 and len(debugged) == 1
