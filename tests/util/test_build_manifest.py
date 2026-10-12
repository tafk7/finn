# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's manifest, ``report/build.json`` (finn.builder.build_dataflow_manifest):
the index of a run, which a machine reads after the exit code, never the log. It is
written from the build's start and rewritten as the build goes: each step with its
phase, status, seconds, reports and their findings, tool runs and their transcripts,
and verifications; the configuration checks; the outcome, by the failure's kind. The
steps and the tool are fakes; no Vivado."""

import pytest

import json
import logging
import numpy as np
import onnx
import onnx.helper as oh
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.builder.build_dataflow_manifest import (
    MANIFEST_FILE,
    REPORT,
    VERIFICATION,
    findings_by_code,
    finn_revision,
)
from finn.core.rtlsim_bench import SimulationFailed
from finn.core.space import Finding, FindingKind, finding_record
from finn.platform import TargetRequest
from finn.toolchain.launcher import Selection, Toolchain, ToolFailed, ToolUnavailable

pytestmark = pytest.mark.util

_STEPS = logging.getLogger("finn.builder.test_steps")

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


def manifest_of(output: str) -> dict:
    with open(os.path.join(output, MANIFEST_FILE)) as f:
        return json.load(f)


def build_of(tmp_path: Path, steps, **fields) -> tuple[int, dict, DataflowBuildConfig]:
    """A build of an empty model through ``steps``: its status, its manifest and its
    configuration."""
    output = str(tmp_path / "output")
    model = str(tmp_path / "model.onnx")
    onnx.save(oh.make_model(oh.make_graph([], "empty", [], [])), model)
    cfg = DataflowBuildConfig(
        output_dir=output,
        target=TargetRequest(period_ns=10.0, part="xczu3eg-sbva484-1-e"),
        steps=steps,
        enable_build_pdb_debug=False,
        **fields,
    )
    cfg.prepared_toolchain = fake_toolchain(tmp_path / "tools")
    status = build.build_dataflow_cfg(model, cfg)
    return status, manifest_of(output), cfg


def step_reports(model, cfg):
    """Writes a report with two findings of one code, one of another, and says so."""
    path = Path(cfg.output_dir) / "report" / "checked.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stated = [finding_record(blocker(code)) for code in ("odd", "odd", "even")]
    path.write_text(json.dumps({"section": {"findings": stated}}))
    _STEPS.info("wrote it", extra={REPORT: str(path)})
    _STEPS.info("verified", extra={VERIFICATION: {"name": "a_check", "passed": True}})
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
    assert identity["vivado"] is None  # the fake toolchain has no Vivado
    checks = manifest["configuration_checks"]
    assert checks["report"] == "config_check_report.json" and checks["failed"] == []
    steps = {step["name"]: step for step in manifest["steps"]}
    assert list(steps) == ["step_reports", "step_tools", "step_looks"]
    assert all(step["status"] == "completed" for step in steps.values())
    assert all(step["phase"] is None for step in steps.values())  # no phase lists them
    assert all(step["seconds"] >= 0 for step in steps.values())
    reported = steps["step_reports"]
    assert reported["reports"] == [
        {"path": "report/checked.json", "findings": {"even": 1, "odd": 2}}
    ]
    assert reported["findings"] == {"even": 1, "odd": 2}
    assert reported["verifications"] == {"a_check": "pass"}
    runs = steps["step_tools"]["tools"]
    assert sorted(run["transcript"] for run in runs) == [
        "logs/step_tools/xvlog.2.stdout.log",
        "logs/step_tools/xvlog.stdout.log",
    ]
    for run in runs:
        assert (run["tool"], run["status"], run["exit"]) == ("xvlog", "completed", 0)
        assert run["messages"] == {"CRITICAL WARNING": 1, "WARNING": 1}
        assert run["seconds"] >= 0
    # Upstream's file stays beside it; nothing half-written is left.
    assert list(json.loads((Path(cfg.output_dir) / "time_per_step.json").read_text())) == [
        "step_reports",
        "step_tools",
        "step_looks",
    ]
    assert sorted(path.name for path in (Path(cfg.output_dir) / "report").iterdir()) == [
        "build.json",
        "checked.json",
    ]


def raising(error):
    def step_fails(model, cfg):
        raise error

    return step_fails


class Refused(ValueError):
    def __init__(self, findings):
        super().__init__(f"{len(findings)} reasons")
        self.findings = tuple(findings)


@pytest.mark.parametrize(
    "error, kind",
    [
        (Refused([blocker("odd"), blocker("odd"), blocker("x")]), "refusal"),
        (ToolFailed("vivado", 1, "/logs/package.sh.stdout.log", ("ERROR: no",)), "tool-failed"),
        (ToolUnavailable("vivado", "not on the route"), "tool-unavailable"),
        (SimulationFailed("FAIL: 3 words differ"), "verdict"),
        (AssertionError("unreachable"), "bug"),
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
    # A refusal's findings are the outcome's; the step's are its reports' (none here).
    assert outcome["findings"] == ({"odd": 2, "x": 1} if kind == "refusal" else {})
    assert failed["findings"] == {}
    assert manifest["steps"][0]["status"] == "completed"


def named(name, step):
    step.__name__ = name
    return step


def test_each_verification_asked_has_a_verdict(tmp_path):
    """A verification asked is ``not-run`` until it concludes: ``pass`` or ``fail`` by its
    verdict, ``fail`` when its step fails with a verdict, ``error`` when its step fails
    otherwise, and ``not-run`` when its step never ran."""
    np.save(tmp_path / "input.npy", np.zeros((1, 2)))
    status, manifest, _ = build_of(
        tmp_path,
        [named("step_verify_partition", raising(SimulationFailed("differs")))],
        verify_steps=[
            VerificationStepType.PARTITION_PYTHON,
            VerificationStepType.STITCHED_IP_TESTBENCH,
        ],
        verify_input_npy=str(tmp_path / "input.npy"),
        generate_outputs=[DataflowOutputType.STITCHED_IP],
    )
    assert status == -1
    assert manifest["verifications"] == {
        "partition_python": "fail",
        "stitched_ip_testbench": "not-run",
    }
    assert manifest["steps"][0]["verifications"] == {"partition_python": "fail"}
    assert manifest["steps"][0]["phase"] == "phase_optimize_hardware"
    _, manifest, _ = build_of(
        tmp_path,
        [named("step_verify_partition", raising(ToolUnavailable("xelab", "missing")))],
        verify_steps=[VerificationStepType.PARTITION_ELABORATION],
    )
    assert manifest["verifications"] == {"partition_elaboration": "error"}


def test_a_refused_configuration_is_the_outcome_with_its_failed_checks(tmp_path):
    with pytest.raises(AssertionError, match="Configuration check failed"):
        build_of(tmp_path, [step_reports], verify_steps=["stitched_ip_testbench"])
    manifest = manifest_of(str(tmp_path / "output"))
    assert manifest["outcome"]["status"] == "refused"
    assert manifest["outcome"]["error"]["type"] == "AssertionError"
    assert manifest["steps"] == []
    failed = manifest["configuration_checks"]["failed"]
    assert {check["name"] for check in failed} >= {"testbench_output"}
    assert {check["severity"] for check in failed if check["name"] == "testbench_output"} == {
        "ERROR"
    }
    assert manifest["verifications"] == {"stitched_ip_testbench": "not-run"}


def test_an_unknown_step_is_refused_before_the_first(tmp_path):
    with pytest.raises(ValueError, match="steps\\[0\\]: 'step_none' is no step or phase"):
        build_of(tmp_path, ["step_none"])
    manifest = manifest_of(str(tmp_path / "output"))
    assert manifest["outcome"]["status"] == "refused"
    assert manifest["outcome"]["error"]["type"] == "ValueError"


def test_a_caller_quieter_than_info_still_gets_its_manifest(tmp_path, capsys):
    """The manifest is collected from INFO records; a caller that set FINN's logger to
    WARNING keeps its log and console at WARNING."""
    finn = logging.getLogger("finn")
    level = finn.level
    finn.setLevel(logging.WARNING)
    try:
        status, manifest, cfg = build_of(tmp_path, [step_reports])
    finally:
        finn.setLevel(level)
    assert status == 0
    assert manifest["steps"][0]["reports"][0]["path"] == "report/checked.json"
    log = (Path(cfg.output_dir) / build.LOG_FILE).read_text()
    assert "wrote it" not in log and "Running step" not in log


def test_findings_are_counted_at_any_depth_but_inside_a_finding():
    cause = finding_record(blocker("cause"))
    stated = finding_record(blocker("top"))
    stated["causes"] = [cause]
    report = {"a": [{"findings": [stated]}], "b": {"findings": [stated, stated]}, "c": 1}
    assert findings_by_code(report) == {"top": 3}
