#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Sequential bounded fresh-process matrix; previous code is comparison only."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--evidence", type=Path, required=True)
args = parser.parse_args()
previous = args.evidence / "iteration1-runtime.py"
cases = []
for implementation in ("previous", "current", "explicit"):
    for depth in (128, 1024, 4096, 20000):
        for repetition in range(3 if depth == 20000 else 1):
            cases.append((implementation, "chain", depth, (), repetition))
        cases.append((implementation, "chain", depth, ("--trace",), 0))
    for size in (64, 256, 1024, 4096):
        for repetition in range(3 if size == 4096 else 1):
            cases.append((implementation, "fanin", size, (), repetition))
        cases.append((implementation, "fanin", size, ("--trace",), 0))
    for prefix, depth in ((1024, 4096), (4096, 20000)):
        for trace in ((), ("--trace",)):
            cases.append((implementation, "mixed", depth, ("--prefix", str(prefix), *trace), 0))
    for unrelated in (1, 500):
        cases.append((implementation, "narrow", unrelated, (), 0))
for implementation in ("previous", "current"):
    for depth in (128, 1024, 4096, 20000):
        cases.append((implementation, "chain", depth, ("--profile",), 0))
cases.append(("current", "chain", 20000, ("--rounds", "10"), 0))
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"}
with (args.evidence / "measurements.jsonl").open("w") as output:
    for implementation, fixture, size, extra, repetition in cases:
        command = [
            sys.executable,
            "scripts/benchmark-self-space-iteration2.py",
            "--implementation",
            implementation,
            "--fixture",
            fixture,
            "--size",
            str(size),
            *extra,
        ]
        if implementation == "previous":
            command += ["--previous-source", str(previous)]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, env=env, timeout=120
            )
            if completed.returncode:
                row = {
                    "status": "process-error",
                    "implementation": implementation,
                    "fixture": fixture,
                    "size": size,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                }
            else:
                row = json.loads(completed.stdout)
                row["status"] = "complete"
        except subprocess.TimeoutExpired:
            row = {
                "status": "timeout",
                "implementation": implementation,
                "fixture": fixture,
                "size": size,
                "timeout_seconds": 120,
            }
        row.update(
            {
                "command": command,
                "repetition": repetition,
                "process_seconds": time.perf_counter() - started,
            }
        )
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        print(implementation, fixture, size, extra, repetition, row["status"], flush=True)
