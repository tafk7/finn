# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a simulation runs on: FinnLib as FINN resolves it, the selected Vivado, and the
revisions a run prints first in its log.

The simulation harness's half of the test tree's shared code; ``kernels.helpers``
holds the construction half (kernels, configurations, samples), whose effect an XSim
job's key reads from the captured designs (``scripts/emitted_text.py``). A change
here changes every job that imports it.
"""

import os
import shutil
import subprocess
from pathlib import Path

from finn import resources


def finnlib_root() -> Path:
    """FinnLib as FINN resolves it: FINN_RESOURCES_FINNLIB, a cached copy, or a fetch."""
    return Path(resources.path("finnlib"))


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


def vivado_simulator() -> bool:
    """Whether a selected Vivado provides xvlog, xelab and xsim.

    FINN images put tool shims on PATH, so a command being found does not mean
    a Vivado installation is selected.
    """
    tools = ("xvlog", "xelab", "xsim")
    return bool(os.environ.get("XILINX_VIVADO")) and all(shutil.which(tool) for tool in tools)
