# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Make the probe modules importable and check each test restores the evaluation context."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (HERE, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finn.core.space import _execution  # noqa: E402


@pytest.fixture(autouse=True)
def evaluation_context_is_reset() -> Iterator[None]:
    assert _execution.current() is None
    yield
    assert _execution.current() is None
