#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Replay bounded benchmark matrix in fresh processes, emitting durable JSONL rows."""

import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
fixtures = [
    ("chain", 20000),
    *(("fanin", n) for n in (64, 256, 1024, 4096)),
    ("narrow", 1),
    ("narrow", 500),
]
cases = [
    (scheduler, kind, size, False)
    for scheduler in ("greenlet", "baseline", "replay", "recursive")
    for kind, size in fixtures
]
cases += [
    (scheduler, kind, size, True)
    for scheduler in ("greenlet", "baseline")
    for kind, size in fixtures
]
cases += [
    (scheduler, kind, size, True)
    for scheduler in ("replay", "recursive")
    for kind, size in (("chain", 20000), ("fanin", 64), ("narrow", 500))
]
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
source_hashes = {
    name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for name in (
        "src/finn/kernels/space/_self_prototype.py",
        "scripts/benchmark-self-space.py",
        "scripts/run-self-space-matrix.py",
    )
}
with args.output.open("w") as output:
    for scheduler, kind, size, traced in cases:
        command = [
            sys.executable,
            "scripts/benchmark-self-space.py",
            "--scheduler",
            scheduler,
            "--fixture",
            kind,
            "--size",
            str(size),
        ]
        if traced:
            command.append("--memory")
        bound = 180 if scheduler == "replay" and kind == "fanin" and size == 4096 else 90
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=bound,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
            )
            if completed.returncode != 0:
                row = {
                    "status": "process-error",
                    "scheduler": scheduler,
                    "fixture": kind,
                    "size": size,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "tracemalloc_enabled": traced,
                }
            else:
                row = json.loads(completed.stdout)
        except subprocess.TimeoutExpired:
            row = {
                "status": "timeout",
                "scheduler": scheduler,
                "fixture": kind,
                "size": size,
                "timeout_seconds": bound,
                "tracemalloc_enabled": traced,
            }
        row["candidate_revision"] = revision
        row["checkout_dirty"] = dirty
        row["source_hashes"] = source_hashes
        row["command"] = command
        row["process_elapsed_seconds"] = time.perf_counter() - started
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        print(scheduler, kind, size, "traced" if traced else "timed", row["status"], flush=True)
