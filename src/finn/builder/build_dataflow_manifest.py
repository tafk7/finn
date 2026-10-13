# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's manifest, ``report/build.json``: the index of one run of
``build_dataflow_cfg`` (``finn.core.reporting.manifest``). A program reads the exit code,
then the manifest, then the reports it lists, never the build's log. The model indexes
the design and what a later step uses; the manifest indexes the run. The build states
three sections of its own before the steps and the outcome:

- ``identity``: what ran it: FINN's revision where it is known (``finn_revision``), the
  toolchain's selection, the target, the model read and the output directory;
- ``plan``: the steps the build runs (finn.builder.build_dataflow_plan), by name, and
  each setting with its value, whether the configuration gave it (else its default), and
  the steps that read it;
- ``configuration_findings``: what checking the configuration found before the first
  step (finn.builder.build_dataflow_checks), each a finding record
  (``Finding.record``): the rejections that refused it, and the
  limitations the build warned of.

A refused configuration's findings are its ``configuration_findings``.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from finn.core.reporting.findings import Finding
from finn.core.reporting.manifest import Manifest

#: The manifest's file, under the output directory.
MANIFEST_FILE = "report/build.json"


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


class BuildManifest(Manifest):
    """The manifest of the build into ``output_dir``; ``identity`` is the build's
    identity so far."""

    def __init__(self, output_dir: str | os.PathLike[str], identity: Mapping[str, Any]) -> None:
        sections = {"identity": dict(identity), "plan": None, "configuration_findings": None}
        super().__init__(Path(output_dir) / MANIFEST_FILE, output_dir, sections)

    def identify(self, **fields: Any) -> None:
        """Add ``fields`` to the build's identity."""
        self.section("identity", {**self.state["identity"], **fields})

    def planned(self, plan: Mapping[str, Any]) -> None:
        """Record the build's plan, as ``Plan.record`` states it."""
        self.section("plan", dict(plan))

    def configured(self, findings: list[Finding]) -> None:
        """Record what checking the configuration found."""
        self.section("configuration_findings", [each.record() for each in findings])


__all__ = ["MANIFEST_FILE", "BuildManifest", "finn_revision"]
