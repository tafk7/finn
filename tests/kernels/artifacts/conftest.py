# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures for the tests of ``finn.kernels.artifacts`` and its isolation.

Nothing here generates stimulus, so nothing here needs the numpy and torch
seeding that ``tests/conftest.py`` does for the tests that do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The repository root, four levels up from ``tests/kernels/artifacts/``.
FINN_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session", params=("dataflow", "kernels"))
def production_source_root(request: pytest.FixtureRequest) -> Path:
    """Each planning package's source root, for the checks that it runs no tool."""

    return FINN_ROOT / "src" / "finn" / str(request.param)
