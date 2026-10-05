# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""scripts/benchmark-space.py runs every suite, at its smallest sizes, as a maintainer runs it.

The benchmark is the measuring stick a change to the Space engine's cost is
compared with; nothing imports it, so an API change that breaks it fails only
here. Its timings are not checked. Its semantic work assertions are, by the
script itself; this test checks that every suite reached them and reported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark-space.py"
# Two of everything: the least that gives each comparison the script makes two
# sides (one narrow query against several branches, alternating kernel choices).
SIZES = {
    "flat": 2,
    "children": 2,
    "depth": 2,
    "batch": 2,
    "branches": 2,
    "population": 2,
    "self_depth": 2,
    "fan_in": 2,
    "kernel_trials": 2,
}


def test_every_suite_runs_and_reports(tmp_path: Path) -> None:
    output, markdown = tmp_path / "report.json", tmp_path / "PERFORMANCE.md"
    sizes = [f"--{name.replace('_', '-')}={value}" for name, value in SIZES.items()]
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--suite", "all", "--output", str(output)]
        + ["--markdown", str(markdown), *sizes],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout.splitlines()[-1])["semantic_assertions"] == "passed"

    report = json.loads(output.read_text())
    assert report["suite"] == "all"
    assert report["configuration"] == SIZES
    # The generic suite.
    assert [item["fixture"] for item in report["compilation"]] == [
        "flat",
        "repeated children",
        "guarded scope depth",
    ]
    assert [item["fixture"] for item in report["batch_updates"]] == ["independent", "dependent"]
    assert report["replacement_validation"]["validated"] == SIZES["batch"]
    assert [item["inactive_callbacks"] for item in report["narrow_queries"]] == [0, 0]
    assert [item["leaf_callbacks_per_query"] for item in report["wide_choices"]] == [1, 1]
    assert report["cache_reclamation"]["alive_after_configurations_released"] == 0
    assert report["constant_expressions"]["terms"] == SIZES["batch"]
    # The runtime suite, each workload in its own process.
    assert [item["shape"] for item in report["self_workloads"]] == ["chain", "fan_in", "mixed"]
    # The kernel suite: every configuration reached its module view, and none is retained.
    kernels = report["kernel_workloads"]
    assert [item["kernel"] for item in kernels] == ["fifo", "dotp", "matmul"]
    assert all(item["trials"] == SIZES["kernel_trials"] for item in kernels)
    assert all(item["retained_old_configuration_count"] == 0 for item in kernels)
    assert all(item["retained_population_count"] == 0 for item in kernels)

    sections = [line for line in markdown.read_text().splitlines() if line.startswith("## ")]
    assert sections == [
        "## Preparation",
        "## Admission and query work",
        "## Ordinary self reads",
        "## Repeated kernel configurations",
        "## Constant expressions",
        "## Interpretation and limits",
    ]
