#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Reproducible focused validation with exact expected negative typing lines."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = "src/finn/kernels/space/_self_prototype.py"
tests = "tests/kernels/space/test_self_prototype.py"
positive = "tests/kernels/typing/self_prototype_types.py"
negative = "tests/kernels/typing/self_prototype_types_negative.py"
scripts = [
    "scripts/benchmark-self-space.py",
    "scripts/run-self-space-matrix.py",
    "scripts/validate-self-space.py",
    "scripts/check-self-space-lifecycle.py",
]
files = [source, tests, positive, negative, *scripts]
mypy = [
    "mypy",
    "--strict",
    "--namespace-packages",
    "--explicit-package-bases",
    "--follow-imports=silent",
    "--cache-dir=/tmp/finn-self-mypy-cache",
]
commands = [
    [sys.executable, "-m", "pytest", "--confcutdir=tests/kernels", tests, "-q"],
    [*mypy, source, positive],
    [*mypy, negative],
    ["ruff", "check", *files],
    ["ruff", "format", "--check", *files],
]
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src:tests", "MYPYPATH": "src"}
with args.output.open("w") as output:
    output.write("Environment: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests MYPYPATH=src\n")
    for command in commands:
        output.write("$ " + shlex.join(command) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        output.write(result.stdout + result.stderr)
        output.write(f"exit={result.returncode}\n\n")
        output.flush()
        if command == [*mypy, negative]:
            expected = {
                i
                for i, line in enumerate(Path(negative).read_text().splitlines(), 1)
                if "# BAD" in line
            }
            found = [
                int(line)
                for line in re.findall(rf"{re.escape(negative)}:(\d+): error:", result.stdout)
            ]
            assert (
                result.returncode == 1 and set(found) == expected and len(found) == len(expected)
            ), result.stdout
            output.write(f"Expected negative errors matched exactly: {sorted(expected)}\n\n")
        else:
            assert result.returncode == 0, result.stdout + result.stderr
print("Focused tests, strict typing, negative typing, lint and formatting passed.")
