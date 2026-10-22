# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The shadow run's slice of the kernel gate: one kernel module whose tests reach selectors,
channels and ``explain``, run with ``-p core.space._running_shadow`` (today's public API
answering every question both ways, failing a test on any disagreement) in a process of its
own, single process.

The gate's pytest takes no plugin, so the module runs here, in a subprocess. Nothing of the
plugin is imported: it is named on the subprocess's command line only.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).parent.parent

MODULE = "kernels/test_channel_source.py"
"""The kernel module run under the shadow (from ``tests``)."""

PLUGIN = "core.space._running_shadow"

COUNT = re.compile(r"running shadow: (\d+) questions compared \((.*)\); disagreements: (\d+)")
SUMMARY = re.compile(r"(\d+) (passed|failed|errors?|xfailed|xpassed|skipped|deselected)")


def test_a_kernel_module_passes_under_the_shadow() -> None:
    """The module's tests all pass under the shadow, with questions compared and none
    answered otherwise."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:xdist",
        "-p",
        "no:cacheprovider",
        "-p",
        PLUGIN,
        "-q",
        MODULE,
    ]
    environment = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    ran = subprocess.run(
        command,
        cwd=str(TESTS),
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    lines = ran.stdout.splitlines()
    tail = "\n".join([*lines[-40:], *ran.stderr.splitlines()[-20:]])
    summary = lines[-1] if lines else ""
    counts = {kind.rstrip("s"): int(number) for number, kind in SUMMARY.findall(summary)}
    assert ran.returncode == 0, tail
    assert counts.get("passed", 0) > 0, tail
    assert counts.get("failed", 0) == 0 and counts.get("error", 0) == 0, tail
    found = COUNT.search(ran.stdout)
    assert found is not None, tail
    assert int(found.group(1)) > 0 and int(found.group(3)) == 0, tail
