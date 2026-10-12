# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A run's log outside a build: the file has every FINN record with its step, the
console the runner's lines, and FINN's logger is restored afterwards."""

import pytest

import logging

from finn.util.run_log import FILE_ONLY, log_run, running_step


@pytest.mark.util
def test_a_run_logs_each_record_with_its_step_and_restores_the_logging(tmp_path, capsys):
    runner = logging.getLogger("finn.util.test_run_log")
    other = logging.getLogger("finn.transformation.kernels")
    finn = logging.getLogger("finn")
    state = finn.level, list(finn.handlers), finn.propagate
    path = tmp_path / "logs" / "run.log"
    with log_run(str(path), runner=runner.name):
        runner.info("before")
        with running_step("probe"):
            runner.info("inside")
            other.info("a module's line")
            runner.error("the traceback", extra=FILE_ONLY)
        runner.info("after")
    printed = capsys.readouterr()
    log = path.read_text()
    assert "INFO - finn.util.test_run_log: before" in log
    assert "INFO probe finn.util.test_run_log: inside" in log
    assert "INFO probe finn.transformation.kernels: a module's line" in log
    assert "ERROR probe finn.util.test_run_log: the traceback" in log
    assert "INFO - finn.util.test_run_log: after" in log
    assert printed.out.splitlines() == ["before", "inside", "after"]
    assert printed.err == ""
    assert (finn.level, finn.handlers, finn.propagate) == state
