# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A run's manifest outside a build: the run's sections first, then each step with the
reports, tool runs and noted findings its records carry, and the outcome. A noted
finding is one record, its line ``str(finding)``; outside a step it is logged, not
listed. Afterwards the records logger is as it was."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.reporting.log import running_step
from finn.core.reporting.manifest import (
    RECORDS,
    TOOL_RUN,
    Manifest,
    note,
    recorded_run,
    reported,
    write_report,
)

RUNNER = "finn.core.reporting.test_manifest"


def limitation(code: str) -> Finding:
    return Finding(FindingKind.LIMITATION, code, "check", f"{code} found")


def read(manifest: Manifest) -> dict[str, Any]:
    stated: dict[str, Any] = json.loads(manifest.path.read_text())
    return stated


def test_a_step_lists_its_reports_tool_runs_and_noted_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = Manifest(tmp_path / "run.json", tmp_path, {"identity": {"name": "probe"}})
    level = RECORDS.level
    with recorded_run(str(tmp_path / "run.log"), manifest, runner=RUNNER) as recording:
        assert recording is manifest
        note(limitation("outside"))
        with running_step("probe"):
            manifest.step_started("probe")
            write_report(tmp_path / "report" / "probe.json", {"findings": [{"code": "x"}]})
            reported(tmp_path / "report" / "other.txt")
            reported(tmp_path / "report" / "probe.json")
            stem = str(tmp_path / "logs" / "tool")
            started = {"run": 1, "tool": "tool", "transcript": f"{stem}.stdout.log"}
            RECORDS.info("started", extra={TOOL_RUN: {**started, "status": "running"}})
            ended = {"run": 1, "status": "completed", "exit": 0, "messages": {"WARNING": 2}}
            RECORDS.info("ended", extra={TOOL_RUN: ended})
            note(limitation("clock-below-asked"))
            manifest.step_ended("completed", 0.5)
        manifest.completed()
    stated = read(manifest)
    assert list(stated) == ["identity", "steps", "outcome"]
    (step,) = stated["steps"]
    assert step["reports"] == ["report/other.txt", "report/probe.json"]
    assert step["tools"] == [
        {
            "tool": "tool",
            "transcript": "logs/tool.stdout.log",
            "status": "completed",
            "exit": 0,
            "messages": {"WARNING": 2},
        }
    ]
    assert step["findings"] == [limitation("clock-below-asked").record()]
    assert stated["outcome"] == {"status": "completed"}
    # Each noted finding is one line, logged inside a step or out.
    err = capsys.readouterr().err.splitlines()
    assert err == [
        "WARNING: check: outside: outside found",
        "WARNING: check: clock-below-asked: clock-below-asked found",
    ]
    log = (tmp_path / "run.log").read_text()
    assert f"WARNING probe {RECORDS.name}: check: clock-below-asked: " in log
    assert RECORDS.level == level and not RECORDS.handlers


def test_a_failed_step_is_the_outcome_with_its_refusals_findings_by_code(tmp_path: Path) -> None:
    class Refused(ValueError):
        findings = (limitation("odd"), limitation("odd"), limitation("x"))

    manifest = Manifest(tmp_path / "run.json", tmp_path, {})
    manifest.step_started("probe")
    manifest.failed("probe", "refusal", Refused("no"))
    manifest.step_ended("failed", 0.0)
    stated = read(manifest)
    assert stated["outcome"] == {
        "status": "failed",
        "step": "probe",
        "kind": "refusal",
        "error": {"type": "Refused", "message": "no"},
        "findings": {"odd": 2, "x": 1},
    }
    assert stated["steps"][0]["status"] == "failed"


def test_a_caller_quieter_than_info_keeps_the_manifests_records(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "run.json", tmp_path, {})
    RECORDS.setLevel(logging.ERROR)
    try:
        with recorded_run(str(tmp_path / "run.log"), manifest, runner=RUNNER):
            manifest.step_started("probe")
            reported(tmp_path / "a.json")
            manifest.step_ended("completed", 0.0)
        assert RECORDS.level == logging.ERROR
    finally:
        RECORDS.setLevel(logging.NOTSET)
    assert read(manifest)["steps"][0]["reports"] == ["a.json"]
