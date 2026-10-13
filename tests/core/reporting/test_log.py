# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A run's log outside a build: the file has every FINN record with its step, the
console the run's own INFO lines, FINN's warnings and a count of Python's, and FINN's
logger is restored afterwards."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pytest

from finn.core.reporting.log import FILE_ONLY, log_run, running_step

RUNNER = "finn.core.reporting.test_log"


def test_a_run_logs_each_record_with_its_step_and_restores_the_logging(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = logging.getLogger(RUNNER)
    step = logging.getLogger(f"{RUNNER}.step")
    other = logging.getLogger("finn.transformation.kernels")
    finn = logging.getLogger("finn")
    state = finn.level, list(finn.handlers), finn.propagate
    path = tmp_path / "logs" / "run.log"
    with log_run(str(path), runner=runner.name):
        runner.info("before")
        with running_step("probe"):
            runner.info("inside")
            step.info("a step's summary")
            other.info("a module's line")
            other.warning("a module's warning")
            runner.error("the traceback", extra=FILE_ONLY)
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("a library's warning", UserWarning)
        runner.info("after")
    printed = capsys.readouterr()
    log = path.read_text()
    assert "INFO - finn.core.reporting.test_log: before" in log
    assert "INFO probe finn.core.reporting.test_log: inside" in log
    assert "INFO probe finn.core.reporting.test_log.step: a step's summary" in log
    assert "INFO probe finn.transformation.kernels: a module's line" in log
    assert "WARNING probe finn.transformation.kernels: a module's warning" in log
    assert "ERROR probe finn.core.reporting.test_log: the traceback" in log
    assert "WARNING probe py.warnings: " in log and "UserWarning: a library's warning" in log
    assert "INFO - finn.core.reporting.test_log: after" in log
    assert printed.out.splitlines() == [
        "before",
        "inside",
        "a step's summary",
        "after",
        f"1 Python warnings, in {path}",
    ]
    assert printed.err.splitlines() == ["WARNING: a module's warning"]
    assert (finn.level, finn.handlers, finn.propagate) == state


def test_verbose_shows_each_python_warning_and_logs_debug_to_the_file_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = logging.getLogger("finn.transformation.kernels")
    finn = logging.getLogger("finn")
    level = finn.level
    path = tmp_path / "run.log"
    with log_run(str(path), runner=RUNNER, verbose=True):
        other.debug("one item's detail")
        other.info("a module's line")
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("a library's warning", UserWarning)
    printed = capsys.readouterr()
    log = path.read_text()
    assert "DEBUG - finn.transformation.kernels: one item's detail" in log
    assert printed.out.splitlines() == ["a module's line"]
    assert "UserWarning: a library's warning" in printed.err
    assert "Python warnings" not in printed.out + log
    assert finn.level == level
