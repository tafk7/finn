# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a simulation runs on: FinnLib as FINN resolves it, and the revisions a run
prints first in its log. Whether the toolchain can simulate is XSim's driver's
(``finn.toolchain.xsim.available``).

Apart from ``finn.core.rtlsim_bench`` so that a job importing only these (a numeric sweep
driving XSI, an artifact test reading FinnLib) is not keyed by the testbench writer's
code (``scripts/emitted_text.py`` keys a job by the harness files it imports).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from finn.resources import finnlib_root


def print_identity() -> None:
    """Print the FINN and FinnLib revisions a run compiles, first in its log.

    ``finn <sha>[ dirty]`` and ``finnlib <revision> <path>``: a log that does not
    say what it compiled is not evidence about a commit. A working clone of
    FinnLib is named by its commit; the cached pin, which is not a repository,
    by the digest it was verified against.
    """

    def revision(directory: Path) -> str:
        def git(*args: str) -> str:
            done = subprocess.run(
                ["git", "-C", str(directory), *args], capture_output=True, text=True
            )
            return done.stdout.strip() if done.returncode == 0 else ""

        sha = git("rev-parse", "--short", "HEAD")
        if not sha:
            marker = directory / ".finn-resource"
            return f"pinned {marker.read_text().strip()[:19]}" if marker.exists() else "unknown"
        return f"{sha} dirty" if git("status", "--porcelain", "--untracked-files=no") else sha

    finnlib = finnlib_root()
    print(f"finn {revision(Path(__file__).resolve().parent)}", flush=True)
    print(f"finnlib {revision(finnlib)} {finnlib}", flush=True)


__all__ = ["finnlib_root", "print_identity"]
