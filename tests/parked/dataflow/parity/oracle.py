# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Running the previous implementation, and refusing to pretend we did.

Two rules, and the second is the one that matters.

**Two interpreters, never one.**  The oracle and this stack define modules with
the same names; importing both into one process would give whichever was
imported first, and a parity result produced that way would be one
implementation agreeing with itself.

**Absence is a failure when evidence was demanded.**  An ordinary developer run
may skip these -- the oracle worktree is not part of a checkout.  A Gate run may
not: ``FINN_DATAFLOW_REQUIRE_PARITY=1`` turns every reason-to-skip into a
failure, including a worktree at the wrong revision.  A gate that silently
skipped its own parity evidence would report a pass for a question nobody asked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

#: The revision the correspondence table was written against.  Pinned because
#: the table names fields: read against another revision, "every field is
#: accounted for" would be a claim about a different set of fields.
ORACLE_REVISION = "df2e42a20a2f4a2d852a861f2cf8af78e3e77cc5"

REQUIRE_PARITY = os.environ.get("FINN_DATAFLOW_REQUIRE_PARITY") == "1"

_PROBE = Path(__file__).with_name("oracle_probe.py")


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3].parent / "finn"


def oracle_root() -> Path:
    override = os.environ.get("FINN_DATAFLOW_ORACLE")
    return Path(override) if override else _default_root()


def _unavailable(reason: str) -> None:
    if REQUIRE_PARITY:
        raise AssertionError(
            f"parity evidence was required and could not be produced: {reason}.  "
            "Set FINN_DATAFLOW_ORACLE to the oracle worktree, or drop "
            "--require-parity and say so in the report"
        )
    pytest.skip(f"oracle parity unavailable: {reason}")


def _revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def require_oracle() -> Path:
    """The oracle worktree, at the pinned revision, or a skip/failure."""

    root = oracle_root()
    if not (root / "src" / "finn" / "dataflow").is_dir():
        _unavailable(f"no oracle worktree at {root}")
    revision = _revision(root)
    if revision is None:
        _unavailable(f"{root} is not a git worktree")
    if revision != ORACLE_REVISION:
        _unavailable(f"{root} is at {revision}, not the pinned {ORACLE_REVISION}")
    return root


@lru_cache(maxsize=1)
def _probe(payload: str) -> dict[str, Any]:
    root = require_oracle()
    environment = dict(os.environ)
    entries = [str(root / "src")]
    qonnx = root / "deps" / "qonnx" / "src"
    if qonnx.is_dir():
        entries.append(str(qonnx))
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["FINN_ROOT"] = str(root)
    completed = subprocess.run(
        [sys.executable, str(_PROBE)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(root),
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"the oracle probe failed (exit {completed.returncode}):\n{completed.stderr[-4000:]}"
        )
    return dict(json.loads(completed.stdout))


def oracle_report(fixtures: tuple[dict[str, Any], ...], build: dict[str, Any]) -> dict[str, Any]:
    """One probe run over these fixtures, memoized for the session."""

    payload = json.dumps({"fixtures": list(fixtures), "build": build}, sort_keys=True)
    return _probe(payload)


__all__ = [
    "ORACLE_REVISION",
    "REQUIRE_PARITY",
    "oracle_report",
    "oracle_root",
    "require_oracle",
]
