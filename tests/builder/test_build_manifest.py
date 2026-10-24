# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's manifest, ``report/build.json`` (finn.builder.build_dataflow_manifest):
the index of a run, which a machine reads after the exit code, never the log. It is
written from the build's start and rewritten as the build goes: each step with its
status, seconds, reports, tool runs and their transcripts, and noted findings; what
checking the configuration found; the outcome, by the failure's kind.
The steps and the tool are fakes; no Vivado."""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnx
import onnx.helper as oh
import pytest
from support.refusals import refused

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import ConfigurationRefused, DataflowBuildConfig
from finn.builder.build_dataflow_manifest import MANIFEST_FILE, finn_revision
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.reporting.manifest import RECORDS, note, reported
from finn.platform.request import TargetRequest
from finn.toolchain.launcher import Selection, Toolchain, ToolFailed, ToolUnavailable

pytestmark = pytest.mark.util

#: A tool printing Vivado's messages: one warning, one critical warning.
WARNS = """
import sys
print("WARNING: [Synth 8-1] a warning", flush=True)
print("CRITICAL WARNING: [Timing 38-2] a critical one", file=sys.stderr, flush=True)
"""


def fake_toolchain(directory: Path) -> Toolchain:
    """A toolchain over a bare PATH whose one tool, ``xvlog``, prints WARNS."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "xvlog"
    path.write_text("#!" + sys.executable + "\n" + WARNS)
    path.chmod(0o755)
    return Toolchain(Selection(command_dir=str(directory)), {"PATH": os.defpath})


def blocker(code: str) -> Finding:
    return Finding(FindingKind.BLOCKER, code, "test", f"{code} found")


def limitation(code: str) -> Finding:
    return Finding(FindingKind.LIMITATION, code, "test", f"{code} found")


def manifest_of(output: str) -> dict:
    with open(os.path.join(output, MANIFEST_FILE)) as f:
        return json.load(f)


def build_of(tmp_path: Path, steps, **fields) -> tuple[int, dict, DataflowBuildConfig]:
    """A build of an empty model through ``steps``: its status, its manifest and its
    configuration."""
    output = str(tmp_path / "output")
    model = str(tmp_path / "model.onnx")
    onnx.save(oh.make_model(oh.make_graph([], "empty", [], [])), model)
    fields.setdefault("target", TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"))
    cfg = DataflowBuildConfig(
        output_dir=output, steps=steps, enable_build_pdb_debug=False, **fields
    )
    cfg.prepared_toolchain = fake_toolchain(tmp_path / "tools")
    status = build.build_dataflow_cfg(model, cfg)
    return status, manifest_of(output), cfg


def step_reports(model, cfg):
    """Writes a report with two findings of one code, one of another, and says so; notes
    a limitation."""
    path = Path(cfg.output_dir) / "report" / "checked.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stated = [blocker(code).record() for code in ("odd", "odd", "even")]
    path.write_text(json.dumps({"section": {"findings": stated}}))
    reported(path)
    note(limitation("short"))
    return model


def step_tools(model, cfg):
    """Runs the fake tool twice, from two threads, as the HLS pool does."""
    tools = cfg.prepared_toolchain.transcripts_in(Path(cfg.output_dir) / "logs" / "step_tools")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: tools.run("xvlog", [], timeout=60), range(2)))
    return model


def step_looks(model, cfg):
    """Reads the manifest while it runs: itself running, no outcome yet."""
    seen = manifest_of(cfg.output_dir)
    assert seen["outcome"] is None
    assert seen["steps"][-1]["name"] == "step_looks"
    assert (seen["steps"][-1]["status"], seen["steps"][-1]["seconds"]) == ("running", None)
    return model


def test_a_completed_build_is_indexed_step_by_step(tmp_path):
    status, manifest, cfg = build_of(tmp_path, [step_reports, step_tools, step_looks])
    assert status == 0
    assert manifest["outcome"] == {"status": "completed"}
    identity = manifest["identity"]
    assert identity["model"] == str(tmp_path / "model.onnx")
    assert identity["output_dir"] == cfg.output_dir
    assert identity["target"] == {
        "part": "xczu3eg-sbva484-1-e",
        "shell": "ip",
        "board": None,
        "period_ns": 10.0,
    }
    assert set(identity["toolchain"]) == {
        "settings",
        "command_dir",
        "launcher",
        "hls_frontend",
        "vivado_jobs",
    }
    assert identity["finn"] == finn_revision()
    assert manifest["configuration_findings"] == []
    steps = {step["name"]: step for step in manifest["steps"]}
    assert list(steps) == ["step_reports", "step_tools", "step_looks"]
    assert all(step["status"] == "completed" for step in steps.values())
    assert all(step["seconds"] >= 0 for step in steps.values())
    # A report is listed by path; its findings stay in it. A noted finding is listed.
    assert steps["step_reports"]["reports"] == ["report/checked.json"]
    assert steps["step_reports"]["findings"] == [limitation("short").record()]
    assert steps["step_tools"]["findings"] == []
    assert "verifications" not in manifest
    runs = steps["step_tools"]["tools"]
    assert sorted(run["transcript"] for run in runs) == [
        "logs/step_tools/xvlog.2.stdout.log",
        "logs/step_tools/xvlog.stdout.log",
    ]
    for run in runs:
        assert (run["tool"], run["status"], run["exit"]) == ("xvlog", "completed", 0)
        assert run["messages"] == {"CRITICAL WARNING": 1, "WARNING": 1}
        assert run["seconds"] >= 0
    # The manifest holds each step's time; nothing half-written is left.
    assert not (Path(cfg.output_dir) / "time_per_step.json").exists()
    assert sorted(path.name for path in (Path(cfg.output_dir) / "report").iterdir()) == [
        "build.json",
        "checked.json",
    ]


def raising(error):
    def step_fails(model, cfg):
        raise error

    return step_fails


class Refused(Refusal):
    def __init__(self, findings):
        super().__init__(findings, f"{len(findings)} reasons")


def untyped(findings):
    """A plain ValueError carrying findings: a bug, since a refusal is told by its type."""
    error = ValueError(f"{len(findings)} reasons")
    error.findings = tuple(findings)
    return error


@pytest.mark.parametrize(
    "error, kind",
    [
        (Refused([blocker("odd"), blocker("odd"), blocker("x")]), "refusal"),
        (ToolFailed("vivado", 1, "/logs/package.sh.stdout.log", ("ERROR: no",)), "tool-failed"),
        (ToolUnavailable("vivado", "not on the route"), "tool-unavailable"),
        (AssertionError("unreachable"), "bug"),
        (untyped([blocker("odd")]), "bug"),
    ],
)
def test_a_failed_step_is_the_outcome_by_its_kind(tmp_path, error, kind):
    status, manifest, _ = build_of(tmp_path, [step_reports, raising(error)])
    assert status == -1
    outcome = manifest["outcome"]
    assert (outcome["status"], outcome["step"], outcome["kind"]) == ("failed", "step_fails", kind)
    assert outcome["error"] == {"type": type(error).__name__, "message": str(error)}
    failed = manifest["steps"][-1]
    assert (failed["name"], failed["status"]) == ("step_fails", "failed")
    assert failed["seconds"] >= 0
    # A refusal's findings are the outcome's.
    assert outcome["findings"] == ({"odd": 2, "x": 1} if kind == "refusal" else {})
    assert manifest["steps"][0]["status"] == "completed"


def test_a_refused_configuration_is_the_outcome_with_its_findings(tmp_path):
    # A board that requires a Vivado release; the fake toolchain has no Vivado.
    with refused("board-vivado", error=ConfigurationRefused) as caught:
        build_of(
            tmp_path,
            [step_reports, "step_explore"],
            target=TargetRequest(period_ns=10.0, board="AUP-ZU3_8GB", shell="pynq"),
            stop_step="step_reports",
            settings={"strategies": [{"size_fifos": {"margin": 2}}]},
        )
    manifest = manifest_of(str(tmp_path / "output"))
    assert manifest["outcome"] == {
        "status": "refused",
        "error": {"type": "ConfigurationRefused", "message": str(caught.error)},
    }
    assert manifest["steps"] == []
    found = [(each["code"], each["kind"]) for each in manifest["configuration_findings"]]
    assert found == [("board-vivado", "rejection"), ("setting-unread", "limitation")]


def test_an_unknown_step_is_refused_before_the_first(tmp_path):
    with refused(
        "step-unknown",
        error=ConfigurationRefused,
        owner="configuration.steps[0]",
        message="^'step_none' ",
    ):
        build_of(tmp_path, ["step_none"])
    manifest = manifest_of(str(tmp_path / "output"))
    assert manifest["outcome"]["status"] == "refused"
    assert manifest["outcome"]["error"]["type"] == "ConfigurationRefused"
    assert [each["code"] for each in manifest["configuration_findings"]] == ["step-unknown"]


def test_a_caller_quieter_than_info_still_gets_its_manifest(tmp_path, capsys):
    """The manifest is collected from INFO records of its own logger; a caller that set
    FINN's logger to WARNING keeps its log and console at WARNING."""
    finn = logging.getLogger("finn")
    level = finn.level
    finn.setLevel(logging.WARNING)
    try:
        status, manifest, cfg = build_of(tmp_path, [step_reports])
    finally:
        finn.setLevel(level)
    assert status == 0
    assert manifest["steps"][0]["reports"] == ["report/checked.json"]
    log = (Path(cfg.output_dir) / build.LOG_FILE).read_text()
    assert "wrote it" not in log and "Running step" not in log


def test_a_module_logger_a_caller_quiets_drops_nothing_from_the_manifest(tmp_path):
    """The reports and tool runs reach the manifest through one logger whose
    level the build sets: a caller that quiets the toolchain's and the builder's loggers
    drops none of them, and gets its own level back after the build."""
    quieted = [logging.getLogger(name) for name in ("finn.toolchain", "finn.builder")]
    records = RECORDS
    levels = [logger.level for logger in quieted]
    for logger in quieted:
        logger.setLevel(logging.WARNING)
    records.setLevel(logging.ERROR)
    try:
        status, manifest, _ = build_of(tmp_path, [step_reports, step_tools])
    finally:
        assert records.level == logging.ERROR
        records.setLevel(logging.NOTSET)
        for logger, level in zip(quieted, levels):
            logger.setLevel(level)
    assert status == 0
    steps = {step["name"]: step for step in manifest["steps"]}
    assert steps["step_reports"]["reports"] == ["report/checked.json"]
    assert [run["status"] for run in steps["step_tools"]["tools"]] == ["completed"] * 2
