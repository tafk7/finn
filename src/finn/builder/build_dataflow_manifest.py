# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's manifest, ``report/build.json``: the index of one run of
``build_dataflow_cfg``. A program reads the exit code, then the manifest, then the
reports it lists, and never the build's log.

The model and the manifest index different things: the model indexes the design and
the artifacts a later step uses (``finn.outputs``); the manifest indexes the run:

- ``identity``: what ran it: FINN's revision where it is known (``finn_revision``), the
  toolchain's selection, the Vivado release the configuration checks probed, the
  target, the model read and the output directory;
- ``plan``: the steps the build runs (finn.builder.build_dataflow_plan), each with its
  phase and the value of each setting it declares, and each setting with its value,
  whether the configuration gave it (else its default), and the steps that read it;
- ``configuration_checks``: the configuration check report
  (``config_check_report.json``), its counts, and each check that failed;
- ``steps``: each step run, in order, with its phase (None for a step no phase lists),
  its ``status`` (``running``, ``completed``, ``failed``), its ``seconds``, the
  ``reports`` it wrote (each with its findings by code), its tool runs (``tools``: each
  with its transcript, status, seconds, exit status and messages by severity), its
  ``findings`` by code (summed over its reports), and its ``verifications`` with their
  verdicts;
- ``verifications``: each verification the configuration asked for, by name, and its
  verdict: ``pass``, ``fail``, ``error`` (its step failed in another way before it
  concluded) or ``not-run``;
- ``outcome``: None while the build runs; then ``completed``; ``failed``, with the
  step, the error's ``kind`` (``refusal``, ``tool-unavailable``, ``tool-failed``,
  ``verdict``, ``bug``), its type and message, and a refusal's findings by code; or
  ``refused`` before the first step (the configuration), with the error.

The file is rewritten atomically (a temporary file renamed over it) whenever it
changes, so a build stopped at any point leaves the last state it reached; a killed
build's step stays ``running``. Steps report what they did in log records, which the
build's ``ManifestHandler`` collects: a report written (``REPORT``: ``reported``,
``write_report``), a verdict (``VERIFICATION``: ``verdict``), a tool run
(``finn.toolchain.launcher.TOOL_RUN``). Of a report, only its findings' counts are
copied. Paths inside the output directory are written relative to it.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import threading
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from finn.toolchain import vivado
from finn.toolchain.launcher import TOOL_RUN, transcript_files

#: The manifest's file, under the output directory.
MANIFEST_FILE = "report/build.json"

#: The ``extra`` key of a log record for a report a step wrote: its path.
REPORT = "finn_report"
#: The ``extra`` key of a log record for a verification's verdict: ``{"name",
#: "passed"}``.
VERIFICATION = "finn_verification"

PASS, FAIL, ERROR, NOT_RUN = "pass", "fail", "error", "not-run"

#: The log of what the steps report to the manifest: the reports they write and the
#: verdicts of their verifications.
_LOG = logging.getLogger(__name__)


def reported(path: str | os.PathLike[str]) -> None:
    """Log ``wrote <path>`` for a report the running step wrote: the manifest lists it
    under that step, with its findings (``REPORT``)."""
    _LOG.info(f"wrote {path}", extra={REPORT: os.fspath(path)})


def write_report(path: Path, stated: Any) -> None:
    """Write ``stated`` as JSON into the report ``path``, its directory made, and log it
    (``reported``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stated, indent=2))
    reported(path)


def verdict(name: str, passed: bool) -> None:
    """Log a verification's verdict, ``Verification for <name> : SUCCESS`` or ``FAIL``;
    the manifest records it (``VERIFICATION``). A FAIL is a warning, because the build
    continues."""
    stated = {VERIFICATION: {"name": name, "passed": passed}}
    if passed:
        _LOG.info(f"Verification for {name} : SUCCESS", extra=stated)
    else:
        _LOG.warning(f"Verification for {name} : FAIL", extra=stated)


@functools.cache
def finn_revision() -> dict[str, Any] | None:
    """The FINN revision running: the one an installed FINN recorded when it was built
    (``finn/_build_info.json``), else its git checkout's (``revision``, and ``dirty``:
    whether a tracked file differs); None if neither is known. Read once per process."""
    package = Path(__file__).resolve().parents[1]
    recorded = package / "_build_info.json"
    if recorded.is_file():
        info = json.loads(recorded.read_text())
        if info.get("revision"):
            return {"revision": info["revision"], "dirty": info.get("dirty")}
        return None

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(package), *args], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    if not revision:
        return None
    changed = git("status", "--porcelain", "--untracked-files=no")
    return {"revision": revision, "dirty": None if changed is None else bool(changed)}


def findings_by_code(value: Any) -> dict[str, int]:
    """A report's findings counted by code: each finding record
    (``finn.core.space.finding_record``) in a list under a ``findings`` key, at any depth
    of the report except inside another finding."""
    counts: Counter[str] = Counter()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                if key == "findings" and isinstance(item, list):
                    counts.update(
                        each["code"]
                        for each in item
                        if isinstance(each, Mapping) and isinstance(each.get("code"), str)
                    )
                else:
                    walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return dict(sorted(counts.items()))


def _report_findings(path: Path) -> dict[str, int]:
    """A JSON report's findings by code (``findings_by_code``); none for another file."""
    if path.suffix != ".json" or not path.is_file():
        return {}
    try:
        return findings_by_code(json.loads(path.read_text()))
    except ValueError:
        return {}


def _messages(transcript: Path) -> dict[str, int]:
    """A tool run's messages by severity (``finn.toolchain.vivado.messages``), from both
    files of its transcript, whose standard output's file is ``transcript``."""
    stem = str(transcript).removesuffix(".stdout.log")
    counts: Counter[str] = Counter()
    for each in transcript_files(stem):
        if each.is_file():
            read = vivado.messages(each.read_text(errors="replace"))
            for (severity, _), count in read.counts.items():
                counts[severity] += count
    return dict(sorted(counts.items()))


def _summed(*counts: Mapping[str, int]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for each in counts:
        total.update(each)
    return dict(sorted(total.items()))


class Manifest:
    """The ``report/build.json`` of the build into ``output_dir``, written whenever it
    changes (see the module docstring). ``identity`` is the build's identity so far;
    ``asked`` the verifications the configuration asks for, each with the step that
    concludes it."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        identity: Mapping[str, Any],
        asked: Mapping[str, str] | None = None,
    ) -> None:
        self.directory = Path(output_dir)
        self.path = self.directory / MANIFEST_FILE
        self._lock = threading.RLock()
        self._step: dict[str, Any] | None = None
        self._runs: dict[int, dict[str, Any]] = {}
        self._concluded_by = dict(asked or {})
        self.state: dict[str, Any] = {
            "identity": dict(identity),
            "plan": None,
            "configuration_checks": None,
            "steps": [],
            "verifications": {name: NOT_RUN for name in self._concluded_by},
            "outcome": None,
        }
        self.write()

    def relative(self, path: str | os.PathLike[str]) -> str:
        """``path`` relative to the output directory when it is inside it, else as
        given."""
        given = Path(path)
        try:
            return str(given.resolve().relative_to(self.directory.resolve()))
        except ValueError:
            return str(given)

    def write(self) -> None:
        """Write the manifest as it is now: a temporary file beside it, renamed over it."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # One writer per process (the lock); the file gets the process's umask mode.
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}")
            with open(temporary, "w") as f:
                json.dump(self.state, f, indent=2)
                f.write("\n")
            os.replace(temporary, self.path)

    def identify(self, **fields: Any) -> None:
        """Add ``fields`` to the build's identity."""
        with self._lock:
            self.state["identity"].update(fields)
            self.write()

    def planned(self, plan: Mapping[str, Any]) -> None:
        """Record the build's plan, as ``BuildPlan.record`` states it."""
        with self._lock:
            self.state["plan"] = dict(plan)
            self.write()

    def configuration_checks(self, report: Any, path: str) -> None:
        """Record the configuration checks' ``report`` (``build_dataflow_checks.Report``),
        saved as JSON beside ``path``: its counts and each check that failed."""
        failed = [check for check in report.checks if not check.passed]
        with self._lock:
            self.state["configuration_checks"] = {
                "report": self.relative(Path(path).with_suffix(".json")),
                "counts": dict(sorted(Counter(check.severity.value for check in failed).items())),
                "failed": [
                    {"name": c.name, "severity": c.severity.value, "message": c.message}
                    for c in failed
                ],
            }
            self.write()

    def step_started(self, name: str, phase: str | None) -> None:
        with self._lock:
            self._step = {
                "name": name,
                "phase": phase,
                "status": "running",
                "seconds": None,
                "reports": [],
                "tools": [],
                "findings": {},
                "verifications": {},
            }
            self.state["steps"].append(self._step)
            self.write()

    def step_ended(self, status: str, seconds: float) -> None:
        with self._lock:
            if self._step is not None:
                self._step["status"] = status
                self._step["seconds"] = round(seconds, 3)
            self._step = None
            self.write()

    def report(self, path: str | os.PathLike[str]) -> None:
        """Record a report the running step wrote, with its findings; a report written
        again replaces its entry."""
        with self._lock:
            step = self._step
            if step is None:
                return
            entry = {"path": self.relative(path), "findings": _report_findings(Path(path))}
            step["reports"] = [each for each in step["reports"] if each["path"] != entry["path"]]
            step["reports"].append(entry)
            step["findings"] = _summed(*(each["findings"] for each in step["reports"]))
            self.write()

    def verification(self, name: str, verdict: str) -> None:
        """Record a verification's verdict, in the running step."""
        with self._lock:
            if self._step is not None:
                self._step["verifications"][name] = verdict
            self.state["verifications"][name] = verdict
            self.write()

    def tool_run(self, fields: Mapping[str, Any]) -> None:
        """Record a tool run's start or end (``finn.toolchain.launcher.TOOL_RUN``) in the
        running step; at its end, add its messages by severity."""
        with self._lock:
            run = fields["run"]
            if fields.get("status") == "running":
                if self._step is None:
                    return
                entry = {key: value for key, value in fields.items() if key != "run"}
                if entry.get("transcript") is not None:
                    entry["transcript"] = self.relative(entry["transcript"])
                self._runs[run] = entry
                self._step["tools"].append(entry)
            elif run in self._runs:
                entry = self._runs.pop(run)
                entry.update({key: value for key, value in fields.items() if key != "run"})
                transcript = entry.get("transcript")
                if transcript is not None:
                    entry["messages"] = _messages(self.directory / transcript)
            else:
                return
            self.write()

    def completed(self) -> None:
        with self._lock:
            self.state["outcome"] = {"status": "completed"}
            self.write()

    def failed(
        self, step: str, kind: str, error: BaseException, findings: Mapping[str, int]
    ) -> None:
        """Record that the build failed in ``step`` with ``error`` of ``kind``. Each
        verification asked for that this step concludes, and that has no verdict yet,
        becomes ``fail`` for a verdict, else ``error``. A refusal's ``findings`` go into
        the outcome."""
        with self._lock:
            for name, verdict in self.state["verifications"].items():
                if verdict == NOT_RUN and self._concluded_by.get(name) == step:
                    self.verification(name, FAIL if kind == "verdict" else ERROR)
            self.state["outcome"] = {
                "status": "failed",
                "step": step,
                "kind": kind,
                "error": {"type": type(error).__name__, "message": str(error)},
                "findings": dict(sorted(findings.items())),
            }
            self.write()

    def refused(self, error: BaseException) -> None:
        """Record that the build was refused before its first step, with ``error``."""
        with self._lock:
            self.state["outcome"] = {
                "status": "refused",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            self.write()


class ManifestHandler(logging.Handler):
    """Collects into ``manifest`` what a build's log records carry: the reports written
    (``REPORT``), the verdicts (``VERIFICATION``) and the tool runs
    (``finn.toolchain.launcher.TOOL_RUN``)."""

    def __init__(self, manifest: Manifest) -> None:
        super().__init__(logging.NOTSET)
        self.manifest = manifest

    def emit(self, record: logging.LogRecord) -> None:
        try:
            path = getattr(record, REPORT, None)
            if path is not None:
                self.manifest.report(path)
            verdict = getattr(record, VERIFICATION, None)
            if verdict is not None:
                self.manifest.verification(verdict["name"], PASS if verdict["passed"] else FAIL)
            run = getattr(record, TOOL_RUN, None)
            if run is not None:
                self.manifest.tool_run(run)
        except Exception:
            self.handleError(record)


__all__ = [
    "ERROR",
    "FAIL",
    "MANIFEST_FILE",
    "NOT_RUN",
    "PASS",
    "REPORT",
    "VERIFICATION",
    "Manifest",
    "ManifestHandler",
    "finn_revision",
    "findings_by_code",
    "reported",
    "verdict",
    "write_report",
]
