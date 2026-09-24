# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Positive and line-specific negative strict-mypy evidence for the public API."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess


def test_strict_authoring_types(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    assert mypy is not None, "the DS1 typing gate requires mypy"
    root = Path(__file__).resolve().parents[3]
    fixtures = Path(__file__).with_name("typing")
    environment = dict(os.environ, MYPYPATH=f"{root / 'src'}:{root / 'tests'}")
    command = [
        mypy,
        "--strict",
        "--explicit-package-bases",
        "--no-incremental",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    positive = subprocess.run(
        [*command, str(fixtures / "positive.py")],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert positive.returncode == 0, positive.stdout + positive.stderr
    source = (fixtures / "negative.py.txt").read_text()
    negative_file = tmp_path / "negative.py"
    negative_file.write_text(source)
    negative = subprocess.run(
        [*command, str(negative_file)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    expected = {index for index, line in enumerate(source.splitlines(), 1) if "# E" in line}
    actual = {int(line) for line in re.findall(r"negative\.py:(\d+): error:", negative.stdout)}
    assert negative.returncode == 1, negative.stdout + negative.stderr
    assert actual == expected, negative.stdout + negative.stderr
