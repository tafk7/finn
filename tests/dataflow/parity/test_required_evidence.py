# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Deviation D5: a Gate run may not skip its own parity evidence.

A developer without the oracle worktree gets a skip, which is right -- the
oracle is not part of a checkout.  A Gate run gets a failure, which is also
right, and for a reason worth stating plainly: a suite that skipped its parity
tests and then reported "all checks passed" would be reporting a pass for a
question nobody asked.  The two behaviours are checked here in subprocesses,
because the mode is read from the environment at import time.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(*, oracle: str, require: bool) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["FINN_DATAFLOW_ORACLE"] = oracle
    environment["FINN_ROOT"] = str(ROOT)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "deps" / "qonnx" / "src")]
    )
    if require:
        environment["FINN_DATAFLOW_REQUIRE_PARITY"] = "1"
    else:
        environment.pop("FINN_DATAFLOW_REQUIRE_PARITY", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(
                ROOT / "tests/dataflow/parity/test_source_projection_parity.py::"
                "test_the_table_covers_every_oracle_field"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=environment,
        check=False,
    )


def test_a_missing_oracle_is_a_skip_by_default() -> None:
    completed = _run(oracle="/nonexistent/oracle", require=False)
    assert completed.returncode == 0, completed.stdout[-3000:]
    assert "skipped" in completed.stdout


def test_a_missing_oracle_is_a_failure_when_evidence_is_required() -> None:
    completed = _run(oracle="/nonexistent/oracle", require=True)
    assert completed.returncode != 0
    assert "parity evidence was required" in completed.stdout


def test_the_wrong_revision_is_also_a_failure_when_evidence_is_required(tmp_path: Path) -> None:
    """A worktree at another revision is not the oracle the table was written against."""

    worktree = tmp_path / "not-the-oracle"
    (worktree / "src" / "finn" / "dataflow").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "commit", "-q", "--allow-empty", "-m", "x"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@x",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@x",
        },
    )

    completed = _run(oracle=str(worktree), require=True)
    assert completed.returncode != 0
    assert "not the pinned" in completed.stdout
