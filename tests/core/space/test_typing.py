# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Line-specific negative strict-mypy evidence for the public API.

Each ``typing/<family>_negative.py.txt`` marks every line strict mypy must
reject with ``# E``; every other line must type-check. The fixtures are text
so that no gate checks them as source. The positive fixtures,
``typing/<family>.py``, are ordinary modules: ``scripts/check-space.sh`` checks
them with the rest of ``tests/core/space`` (``gate_mypy tests/core/space``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).with_name("typing")
NEGATIVE = "_negative.py.txt"
FAMILIES = sorted(path.name.removesuffix(NEGATIVE) for path in FIXTURES.glob(f"*{NEGATIVE}"))


@pytest.fixture(scope="module")
def mypy_errors(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[int]]:
    """The lines strict mypy rejects in each negative fixture, from one mypy run.

    The fixtures are copied to ``<family>_negative.py`` and checked together as
    the gates check (``gate_mypy`` in ``scripts/_gate-common.sh``), uncoloured
    whatever FORCE_COLOR says, so that the output parses.
    """
    directory = tmp_path_factory.mktemp("typing")
    sources = []
    for family in FAMILIES:
        source = directory / f"{family}_negative.py"
        source.write_text((FIXTURES / f"{family}{NEGATIVE}").read_text())
        sources.append(str(source))
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            "--no-incremental",
            "--no-color-output",
            "--cache-dir",
            str(directory / "cache"),
            *sources,
        ],
        cwd=ROOT,
        env=dict(environment, MYPYPATH=f"{ROOT / 'src'}:{ROOT / 'tests'}"),
        capture_output=True,
        text=True,
        check=False,
    )
    report = result.stdout + result.stderr
    assert result.returncode == 1, report
    errors: dict[str, set[int]] = {family: set() for family in FAMILIES}
    for line in report.splitlines():
        if ": error:" not in line:
            continue
        # An error anywhere else (in finn itself, say) fails every family.
        match = re.match(r".*/(\w+)_negative\.py:(\d+): error:", line)
        assert match is not None and match.group(1) in errors, report
        errors[match.group(1)].add(int(match.group(2)))
    return errors


def test_negative_fixtures_are_found() -> None:
    assert FAMILIES, f"no *{NEGATIVE} under {FIXTURES}"


@pytest.mark.parametrize("family", FAMILIES)
def test_strict_mypy_rejects_exactly_the_marked_lines(
    family: str, mypy_errors: dict[str, set[int]]
) -> None:
    source = (FIXTURES / f"{family}{NEGATIVE}").read_text()
    expected = {index for index, line in enumerate(source.splitlines(), 1) if "# E" in line}
    assert mypy_errors[family] == expected
