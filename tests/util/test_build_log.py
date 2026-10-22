# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import logging
import warnings

from finn.util.build_log import FILE_ONLY, log_run, running_step

RUNNER = "finn.util.test_build_log"


@pytest.mark.util
def test_a_run_logs_each_record_with_its_step_and_restores_the_logging(tmp_path, capsys):
    runner = logging.getLogger(RUNNER)
    step = logging.getLogger(f"{RUNNER}.step")
    other = logging.getLogger("finn.transformation.fpgadataflow")
    finn = logging.getLogger("finn")
    state = finn.level, list(finn.handlers), finn.propagate
    path = tmp_path / "logs" / "run.log"
    with log_run(str(path), runner=runner.name):
        runner.info("before")
        with running_step("probe"):
            runner.info("inside")
            step.info("a step's summary")
            other.info("a module's line")
            other.info("a tool's output", extra=FILE_ONLY)
            other.warning("a module's warning")
            runner.error("the traceback", extra=FILE_ONLY)
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("a library's warning", UserWarning)
        runner.info("after")
    printed = capsys.readouterr()
    log = path.read_text()
    assert f"INFO - {RUNNER}: before" in log
    assert f"INFO probe {RUNNER}: inside" in log
    assert f"INFO probe {RUNNER}.step: a step's summary" in log
    assert "INFO probe finn.transformation.fpgadataflow: a module's line" in log
    assert "INFO probe finn.transformation.fpgadataflow: a tool's output" in log
    assert "WARNING probe finn.transformation.fpgadataflow: a module's warning" in log
    assert f"ERROR probe {RUNNER}: the traceback" in log
    assert "WARNING probe py.warnings: " in log and "UserWarning: a library's warning" in log
    assert f"INFO - {RUNNER}: after" in log
    assert printed.out.splitlines() == [
        "before",
        "inside",
        "a step's summary",
        "after",
        f"1 Python warnings, in {path}",
    ]
    assert printed.err.splitlines() == ["WARNING: a module's warning"]
    assert (finn.level, finn.handlers, finn.propagate) == state


@pytest.mark.util
def test_verbose_shows_each_python_warning_and_logs_debug_to_the_file_only(tmp_path, capsys):
    other = logging.getLogger("finn.transformation.fpgadataflow")
    finn = logging.getLogger("finn")
    level = finn.level
    path = tmp_path / "run.log"
    with log_run(str(path), runner=RUNNER, verbose=True):
        other.debug("one item's detail")
        other.info("a module's line")
        other.info("a tool's output", extra=FILE_ONLY)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("a library's warning", UserWarning)
    printed = capsys.readouterr()
    log = path.read_text()
    assert "DEBUG - finn.transformation.fpgadataflow: one item's detail" in log
    assert printed.out.splitlines() == ["a module's line", "a tool's output"]
    assert "UserWarning: a library's warning" in printed.err
    assert "Python warnings" not in printed.out + log
    assert finn.level == level
