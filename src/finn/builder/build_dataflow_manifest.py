# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's manifest, ``report/build.json``: the index of one run of
``build_dataflow_cfg``. A program reads the exit code, then the manifest, then the
reports it lists, never the build's log. The model indexes the design and what a later
step uses; the manifest indexes the run:

- ``identity``: what ran it: FINN's revision where it is known (``finn_revision``), the
  toolchain's selection, the target, the model read and the output directory;
- ``plan``: the steps the build runs (finn.builder.build_dataflow_plan), by name, and
  each setting with its value, whether the configuration gave it (else its default), and
  the steps that read it;
- ``configuration_findings``: what checking the configuration found before the first
  step (finn.builder.build_dataflow_checks), each a finding record
  (``finn.core.space.finding_record``): the rejections that refused it, and the
  limitations the build warned of;
- ``steps``: each step run, in order, with its ``status`` (``running``, ``completed``,
  ``failed``), its ``seconds``, the ``reports`` it wrote (each with its findings by
  code), and its tool runs (``tools``: each with its transcript, status, seconds, exit
  status and messages by severity);
- ``outcome``: None while the build runs; then ``completed``; ``failed``, with the
  step, the error's ``kind`` (``refusal``, ``tool-unavailable``, ``tool-failed``,
  ``bug``), its type and message, and a refusal's findings by code; or
  ``refused`` before the first step, with the error (a refused configuration's findings
  are its ``configuration_findings``).

The file is replaced whole whenever it changes, so a build stopped at any point leaves
the last state it reached; a killed build's step stays ``running``. Steps report what
they did in log records of one logger, ``finn.toolchain.launcher.RECORDS``, which the
build's ``ManifestHandler`` collects: a report written (``reported``, ``write_report``)
and a tool run (``finn.toolchain.launcher.TOOL_RUN``). Of a report, only its findings'
counts are copied. Paths inside the output directory are relative to it.
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

from finn.core.space import Finding, finding_record
from finn.toolchain import vivado
from finn.toolchain.launcher import RECORDS, TOOL_RUN, transcript_files

#: The manifest's file, under the output directory.
MANIFEST_FILE = "report/build.json"

#: The ``extra`` key of a log record for a report a step wrote: its path.
REPORT = "finn_report"


def reported(path: str | os.PathLike[str]) -> None:
    """Log a report the running step wrote, which the manifest lists under the step."""
    RECORDS.info(f"wrote {path}", extra={REPORT: os.fspath(path)})


def write_report(path: Path, stated: Any) -> None:
    """Write ``stated`` as JSON into the report ``path`` and log it (``reported``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stated, indent=2))
    reported(path)


@functools.cache
def finn_revision() -> dict[str, Any] | None:
    """The FINN revision running: the one an installed FINN recorded when it was built
    (``finn/_build_info.json``), else its git checkout's, with whether a tracked file
    differs (``dirty``); None if neither is known."""
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
    """A report's findings counted by code: each finding record in a list under a
    ``findings`` key, at any depth of the report except inside another finding."""
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
    """A tool run's messages by severity, from both files of the transcript whose
    standard output is ``transcript``."""
    stem = str(transcript).removesuffix(".stdout.log")
    counts: Counter[str] = Counter()
    for each in transcript_files(stem):
        if each.is_file():
            read = vivado.messages(each.read_text(errors="replace"))
            for (severity, _), count in read.counts.items():
                counts[severity] += count
    return dict(sorted(counts.items()))


class Manifest:
    """The manifest of the build into ``output_dir``, written whenever it changes;
    ``identity`` is the build's identity so far."""

    def __init__(self, output_dir: str | os.PathLike[str], identity: Mapping[str, Any]) -> None:
        self.directory = Path(output_dir)
        self.path = self.directory / MANIFEST_FILE
        self._lock = threading.RLock()
        self._step: dict[str, Any] | None = None
        self._runs: dict[int, dict[str, Any]] = {}
        self.state: dict[str, Any] = {
            "identity": dict(identity),
            "plan": None,
            "configuration_findings": None,
            "steps": [],
            "outcome": None,
        }
        self.write()

    def relative(self, path: str | os.PathLike[str]) -> str:
        """``path`` relative to the output directory when it is inside it."""
        given = Path(path)
        try:
            return str(given.resolve().relative_to(self.directory.resolve()))
        except ValueError:
            return str(given)

    def write(self) -> None:
        """Write the manifest as it is now: a temporary file beside it, renamed over it."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # The pid keeps processes' temporary files apart; the lock, a process's threads.
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
        """Record the build's plan, as ``Plan.record`` states it."""
        with self._lock:
            self.state["plan"] = dict(plan)
            self.write()

    def configured(self, findings: list[Finding]) -> None:
        """Record what checking the configuration found."""
        with self._lock:
            self.state["configuration_findings"] = [finding_record(each) for each in findings]
            self.write()

    def step_started(self, name: str) -> None:
        with self._lock:
            self._step = {
                "name": name,
                "status": "running",
                "seconds": None,
                "reports": [],
                "tools": [],
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
            self.write()

    def tool_run(self, fields: Mapping[str, Any]) -> None:
        """Record a tool run's start or end in the running step; at its end, with its
        messages by severity."""
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

    def failed(self, step: str, kind: str, error: BaseException) -> None:
        """Record that the build failed in ``step`` with ``error`` of ``kind``, with a
        refusal's findings by code."""
        codes = Counter(each.code for each in getattr(error, "findings", ()))
        with self._lock:
            self.state["outcome"] = {
                "status": "failed",
                "step": step,
                "kind": kind,
                "error": {"type": type(error).__name__, "message": str(error)},
                "findings": dict(sorted(codes.items())),
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
    """Collects into ``manifest`` the reports and tool runs the records of
    ``finn.toolchain.launcher.RECORDS`` carry."""

    def __init__(self, manifest: Manifest) -> None:
        super().__init__(logging.NOTSET)
        self.manifest = manifest

    def emit(self, record: logging.LogRecord) -> None:
        try:
            path = getattr(record, REPORT, None)
            if path is not None:
                self.manifest.report(path)
            run = getattr(record, TOOL_RUN, None)
            if run is not None:
                self.manifest.tool_run(run)
        except Exception:
            self.handleError(record)


__all__ = [
    "MANIFEST_FILE",
    "REPORT",
    "Manifest",
    "ManifestHandler",
    "finn_revision",
    "findings_by_code",
    "reported",
    "write_report",
]
