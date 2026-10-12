# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A failing build step, with the default ``enable_build_pdb_debug``: the postmortem
debugger starts only where stdin is a terminal, so a build under pytest or a batch job
reports its own error and fails, rather than pdb's failed read. Each kind of error is
logged in its own form, and the build's log has every record with its step."""

import pytest

import io
import logging
import onnx
import onnx.helper as oh
import os
import sys
import warnings

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import ConfigurationRefused, DataflowBuildConfig
from finn.core.rtlsim_bench import SimulationFailed
from finn.core.space import Finding, FindingKind
from finn.platform import TargetRequest
from finn.toolchain.launcher import SELECTED_BY, ToolFailed, ToolUnavailable
from finn.util.basic import make_build_dir


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
    output = make_build_dir("test_build_failure_")
    model = f"{output}/model.onnx"
    onnx.save(oh.make_model(oh.make_graph([], "empty", [], [])), model)
    cfg = DataflowBuildConfig(
        output_dir=output,
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=[step_fails],
    )
    assert cfg.enable_build_pdb_debug
    return build.build_dataflow_cfg(model, cfg), debugged


@pytest.mark.util
def test_a_failed_step_without_a_terminal_reports_its_error(monkeypatch, capsys):
    status, debugged = failed_build(monkeypatch, terminal=False)
    assert status == -1 and debugged == []
    captured = capsys.readouterr()
    assert "RuntimeError: the step's own error" in captured.err
    assert "stdin is not a terminal" in captured.out and "ERROR: Build failed" in captured.err


@pytest.mark.util
def test_a_failed_step_at_a_terminal_starts_the_debugger(monkeypatch):
    status, debugged = failed_build(monkeypatch, terminal=True)
    assert status == -1 and len(debugged) == 1


class Refused(ValueError):
    """A refusal of no common base: an error carrying findings."""

    def __init__(self, findings):
        super().__init__(f"{len(findings)} reasons")
        self.findings = tuple(findings)


def blocker(code: str, message: str) -> Finding:
    return Finding(FindingKind.BLOCKER, code, "test", message)


def raising(error: BaseException):
    def step_fails(model, cfg):
        raise error

    return step_fails


def logged_failure(capsys, error: BaseException) -> tuple[str, str]:
    """A build whose one step raises ``error``: its console's stderr and its log file."""
    output = make_build_dir("test_build_failure_")
    model = f"{output}/model.onnx"
    onnx.save(oh.make_model(oh.make_graph([], "empty", [], [])), model)
    cfg = DataflowBuildConfig(
        output_dir=output,
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=[raising(error)],
        enable_build_pdb_debug=False,
    )
    capsys.readouterr()
    assert build.build_dataflow_cfg(model, cfg) == -1
    with open(f"{output}/{build.LOG_FILE}") as log:
        return capsys.readouterr().err, log.read()


@pytest.mark.util
def test_a_refusal_is_logged_as_its_findings_its_traceback_in_the_file_only(capsys):
    err, log = logged_failure(
        capsys,
        Refused([blocker("odd", "one is odd"), blocker("odd", "three is odd"), blocker("x", "x")]),
    )
    assert err.splitlines()[:3] == [
        "ERROR: step_fails refused: 3 findings: Refused: 3 reasons",
        "ERROR:   odd (blocker) 2: one is odd; and 1 more",
        "ERROR:   x (blocker) 1: x",
    ]
    assert "Traceback" not in err and "Traceback" in log
    assert "ERROR step_fails finn.builder.build_dataflow: step_fails refused: 3 findings" in log


@pytest.mark.util
def test_a_tool_that_failed_is_logged_with_its_transcript_and_errors_no_traceback(capsys):
    failed = ToolFailed(
        "vivado", 1, "/logs/package.sh.stdout.log", ("ERROR: [Common 17-1] no", "ERROR: two")
    )
    err, log = logged_failure(capsys, failed)
    assert err.splitlines()[:3] == [
        "ERROR: step_fails: ToolFailed: vivado failed with exit status 1; transcript "
        "/logs/package.sh.stdout.log",
        "  ERROR: [Common 17-1] no",
        "  ERROR: two",
    ]
    assert "Traceback" not in err and "Traceback" in log


@pytest.mark.util
def test_a_tool_unavailable_is_logged_naming_what_selects_it(capsys):
    err, _ = logged_failure(capsys, ToolUnavailable("vivado", "not on the route"))
    named = f"vivado: not on the route; {SELECTED_BY}"
    assert err.splitlines()[0] == f"ERROR: step_fails: ToolUnavailable: {named}"
    assert "Traceback" not in err


@pytest.mark.util
def test_a_verdict_is_logged_as_a_failed_verification(capsys):
    err, log = logged_failure(capsys, SimulationFailed("FAIL: 3 words differ"))
    assert err.splitlines()[:2] == [
        "ERROR: Verification for step_fails : FAIL",
        "ERROR: SimulationFailed: FAIL: 3 words differ",
    ]
    assert "Traceback" not in err and "Traceback" in log


@pytest.mark.util
def test_a_plain_assert_is_a_bug_with_its_traceback_on_the_console(capsys):
    err, _ = logged_failure(capsys, AssertionError("unreachable"))
    assert err.startswith("ERROR: step_fails failed: AssertionError: unreachable\nTraceback")


def step_says(model, cfg):
    print("a third party's print")
    warnings.warn("a step's warning", UserWarning)
    logging.getLogger("finn.transformation.kernels").info("a module's line")
    logging.getLogger("finn.transformation.kernels").warning("a module's warning")
    return model


@pytest.mark.util
def test_the_log_has_every_record_with_its_step_the_console_the_runner_and_warnings(capsys):
    """No stream is swapped: a print reaches the console as printed and not the log.
    The log file has every FINN record and the captured warnings, each with its level,
    step and logger; the console the runner's lines and the warnings. Afterwards the
    warnings hook and FINN's logger are as they were."""
    shown, finn = warnings.showwarning, logging.getLogger("finn")
    level, handlers, propagate = finn.level, list(finn.handlers), finn.propagate
    output = make_build_dir("test_build_log_")
    model = f"{output}/model.onnx"
    onnx.save(oh.make_model(oh.make_graph([], "empty", [], [])), model)
    cfg = DataflowBuildConfig(
        output_dir=output,
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=[step_says],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        assert build.build_dataflow_cfg(model, cfg) == 0
    printed = capsys.readouterr()
    with open(f"{output}/{build.LOG_FILE}") as f:
        log = f.read()
    assert "a third party's print" in printed.out and "a third party's print" not in log
    assert "Running step: step_says [1/1]" in printed.out
    assert "a module's line" not in printed.out + printed.err
    assert "WARNING: a module's warning" in printed.err
    assert "UserWarning: a step's warning" in printed.err
    assert "INFO step_says finn.transformation.kernels: a module's line" in log
    assert "WARNING step_says finn.transformation.kernels: a module's warning" in log
    assert "WARNING step_says py.warnings: " in log and "UserWarning: a step's warning" in log
    assert "INFO - finn.builder.build_dataflow: Completed successfully" in log
    assert warnings.showwarning is shown
    assert (finn.level, finn.handlers, finn.propagate) == (level, handlers, propagate)


@pytest.mark.util
def test_a_refused_configuration_is_logged_each_finding_once_then_raises(capsys):
    """Before the first step, each limitation the configuration checks find is a warning
    and each rejection an error, once each on the console and in the log, and the build
    raises one ConfigurationRefused; it writes no report of its own."""
    output = make_build_dir("test_build_failure_")
    model = f"{output}/model.onnx"
    onnx.save(oh.make_model(oh.make_graph([], "empty", [], [])), model)
    cfg = DataflowBuildConfig(
        output_dir=output,
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=[step_says],
        verify_steps=["stitched_ip_testbench"],
        generate_outputs=[],
    )
    refusal = "testbench-output: configuration.verify_steps: stitched_ip_testbench runs the"
    with pytest.raises(ConfigurationRefused, match=refusal):
        build.build_dataflow_cfg(model, cfg)
    err = capsys.readouterr().err
    with open(f"{output}/{build.LOG_FILE}") as f:
        log = f.read()
    warning = "verification-not-run: configuration.verify_steps: stitched_ip_testbench, which"
    assert err.splitlines()[:3] == [
        f"WARNING: {warning} step_package_partition runs, and the resolved build steps "
        "(steps/start_step/stop_step) leave it out: that verification does not run, and has "
        "no verdict",
        "ERROR: Configuration refused: 1 findings",
        f"ERROR:   {refusal} testbench stitched_ip writes, and generate_outputs does not ask "
        "for it",
    ]
    assert log.count(refusal) == 1 and log.count(warning) == 1
    assert "Traceback" not in log
    assert sorted(os.listdir(output)) == ["build_dataflow.log", "model.onnx", "report"]
