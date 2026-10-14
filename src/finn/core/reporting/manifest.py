# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A run's manifest: the index of one run, a JSON file a program reads after the exit
code, never the run's log. It holds the sections the run states itself first (a build's
identity and plan), then:

- ``steps``: each step run, in order, with its ``status`` (``running``, ``completed``,
  ``failed``), its ``seconds``, the ``reports`` it wrote (by path), its tool runs
  (``tools``: each with its transcript, status, seconds, exit status and messages by
  severity), the stored products it used (``stored``: each with its store, entry, key
  and ``status``, ``built`` in this run or ``reused``) and the ``findings`` noted while
  it ran (each ``Finding.record``);
- ``outcome``: None while the run goes on; then ``completed``; ``failed``, with the
  step, the error's ``kind``, its type and message, and a refusal's findings by code; or
  ``refused`` before the first step, with the error.

The file is replaced whole whenever it changes, so a run stopped at any point leaves the
last state it reached; a killed run's step stays ``running``. Paths inside the
manifest's directory are relative to it.

Steps say what they did in records of one logger, ``RECORDS``, which a run's manifest
collects (``recorded_run``): a report written (``reported``, ``write_report``), a tool
run (``TOOL_RUN``, logged by ``finn.toolchain.launcher``), a stored product used
(``stored``, logged by the stores of ``finn.toolchain``) and a finding stated without
refusing (``note``). A record outside a step is logged, not listed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

from finn.core.reporting.findings import Finding, Refusal
from finn.core.reporting.log import log_run

#: The logger of the records a run's manifest lists. A run that records sets its level,
#: so a caller that quiets another of FINN's loggers keeps them.
RECORDS = logging.getLogger(__name__)

#: The ``extra`` key of a log record for a report a step wrote: its path.
REPORT = "finn_report"

#: The ``extra`` key of a log record for a noted finding: the ``Finding``.
FINDING = "finn_finding"

#: The ``extra`` key of a tool run's log records: the run's number in this process
#: (``run``) and, when it starts, ``tool``, ``command``, ``cwd``, ``transcript`` and
#: ``status`` ``running``; when it ends, ``status`` (``completed``, ``failed``, or
#: ``stopped`` with the ``error``), ``seconds``, ``exit`` and, while a manifest records
#: it, its ``messages`` by severity.
TOOL_RUN = "finn_tool_run"

#: The ``extra`` key of a log record for a stored product a step used: its ``store``,
#: ``entry`` (its directory), ``key`` and ``status``, ``built`` or ``reused``.
STORED = "finn_stored"


def reported(path: str | os.PathLike[str]) -> None:
    """Log a report the running step wrote, which the manifest lists under the step."""
    RECORDS.info(f"wrote {path}", extra={REPORT: os.fspath(path)})


def write_report(path: Path, stated: Any) -> None:
    """Write ``stated`` as JSON into the report ``path`` and log it (``reported``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stated, indent=2))
    reported(path)


def stored(store: str, entry: str | os.PathLike[str], key: str, *, built: bool) -> None:
    """Log a product of the store ``store`` the running step used, its ``entry`` of the
    ``key``, ``built`` in this run or reused, which the manifest lists under the step."""
    status = "built" if built else "reused"
    fields = {"store": store, "entry": os.fspath(entry), "key": key, "status": status}
    RECORDS.info(f"{status} {entry}", extra={STORED: fields})


def note(finding: Finding, level: int = logging.WARNING) -> None:
    """State ``finding`` without refusing: one record at ``level``, its line
    ``str(finding)``, which the manifest lists under the running step."""
    RECORDS.log(level, "%s", finding, extra={FINDING: finding})


class Manifest:
    """The manifest file ``path``, written whenever it changes: the run's own
    ``sections`` first, then its steps and its outcome. Paths inside ``directory`` are
    stated relative to it."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        directory: str | os.PathLike[str],
        sections: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        self.directory = Path(directory)
        self._lock = threading.RLock()
        self._step: dict[str, Any] | None = None
        self._runs: dict[int, dict[str, Any]] = {}
        self.state: dict[str, Any] = {**sections, "steps": [], "outcome": None}
        self.write()

    def relative(self, path: str | os.PathLike[str]) -> str:
        """``path`` relative to the manifest's directory when it is inside it."""
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

    def section(self, name: str, value: Any) -> None:
        """State the run's own section ``name``."""
        with self._lock:
            self.state[name] = value
            self.write()

    def step_started(self, name: str) -> None:
        with self._lock:
            self._step = {
                "name": name,
                "status": "running",
                "seconds": None,
                "reports": [],
                "tools": [],
                "stored": [],
                "findings": [],
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
        """List a report the running step wrote; a report written again replaces its
        entry."""
        with self._lock:
            step = self._step
            if step is None:
                return
            listed = self.relative(path)
            step["reports"] = [*(each for each in step["reports"] if each != listed), listed]
            self.write()

    def tool_run(self, fields: Mapping[str, Any]) -> None:
        """Record a tool run's start or end in the running step."""
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
            else:
                return
            self.write()

    def stored(self, fields: Mapping[str, Any]) -> None:
        """List a stored product the running step used."""
        with self._lock:
            if self._step is None:
                return
            self._step["stored"].append({**fields, "entry": self.relative(fields["entry"])})
            self.write()

    def noted(self, finding: Finding) -> None:
        """List a finding noted in the running step."""
        with self._lock:
            if self._step is None:
                return
            self._step["findings"].append(finding.record())
            self.write()

    def completed(self) -> None:
        with self._lock:
            self.state["outcome"] = {"status": "completed"}
            self.write()

    def failed(self, step: str, kind: str, error: BaseException) -> None:
        """Record that the run failed in ``step`` with ``error`` of ``kind``, with a
        refusal's findings by code."""
        found = error.findings if isinstance(error, Refusal) else ()
        codes = Counter(each.code for each in found)
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
        """Record that the run was refused before its first step, with ``error``."""
        with self._lock:
            self.state["outcome"] = {
                "status": "refused",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            self.write()


class _Collector(logging.Handler):
    """Lists in ``manifest`` the reports, tool runs, stored products and noted findings
    that the records of ``RECORDS`` carry."""

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
            product = getattr(record, STORED, None)
            if product is not None:
                self.manifest.stored(product)
            finding = getattr(record, FINDING, None)
            if finding is not None:
                self.manifest.noted(finding)
        except Exception:
            self.handleError(record)


_Manifest = TypeVar("_Manifest", bound=Manifest)


@contextlib.contextmanager
def recorded_run(
    log: str, manifest: _Manifest, *, runner: str, verbose: bool = False
) -> Iterator[_Manifest]:
    """The run's log into the file ``log`` (``log_run``, with ``runner`` and ``verbose``)
    and its ``manifest``, filled from ``RECORDS``, for the ``with`` block; then the
    logging found, restored."""
    collector = _Collector(manifest)
    level = RECORDS.level
    with log_run(log, runner=runner, verbose=verbose):
        RECORDS.setLevel(logging.INFO)
        RECORDS.addHandler(collector)
        try:
            yield manifest
        finally:
            RECORDS.removeHandler(collector)
            RECORDS.setLevel(level)
            collector.close()


__all__ = [
    "FINDING",
    "RECORDS",
    "REPORT",
    "STORED",
    "TOOL_RUN",
    "Manifest",
    "note",
    "recorded_run",
    "reported",
    "stored",
    "write_report",
]
