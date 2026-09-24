#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Focused old/new semantics plus unchanged independent regressions and typing."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--evidence", type=Path, required=True)
args = parser.parse_args()
source = "src/finn/kernels/space/_self_prototype.py"
tests = [
    "tests/kernels/space/test_self_prototype.py",
    "tests/kernels/space/test_self_prototype_iteration2.py",
]
positive = "tests/kernels/typing/self_prototype_types.py"
negative = "tests/kernels/typing/self_prototype_types_negative.py"
scripts = [f"scripts/{name}-self-space-iteration2.py" for name in ("benchmark", "run", "validate")]
files = [source, *tests, positive, negative, *scripts]
mypy = [
    "mypy",
    "--strict",
    "--namespace-packages",
    "--explicit-package-bases",
    "--follow-imports=silent",
    "--cache-dir=/tmp/self-space-iteration2-mypy",
]
review = args.evidence.parent / "review-evidence" / "test_adversarial.py"
commands = [
    [
        sys.executable,
        "-m",
        "pytest",
        "--confcutdir=tests/kernels",
        "-o",
        "cache_dir=/tmp/self-space-iteration2-pytest",
        *tests,
        str(review),
        "-q",
    ],
    [*mypy, source, positive],
    [*mypy, negative],
    ["ruff", "check", *files],
    ["ruff", "format", "--check", *files],
]
env = {
    **os.environ,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "src:tests",
    "MYPYPATH": "src",
    "RUFF_CACHE_DIR": "/tmp/self-space-iteration2-ruff",
}
with (args.evidence / "validation.log").open("w") as output:
    output.write(
        "Environment: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests MYPYPATH=src "
        "RUFF_CACHE_DIR=/tmp/self-space-iteration2-ruff\n"
    )
    for command in commands:
        output.write("$ " + shlex.join(command) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        output.write(result.stdout + result.stderr + f"exit={result.returncode}\n\n")
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
            output.write(f"Exactly expected negative errors: {sorted(expected)}\n\n")
        else:
            assert result.returncode == 0, result.stdout + result.stderr
print("All focused validation passed.")
