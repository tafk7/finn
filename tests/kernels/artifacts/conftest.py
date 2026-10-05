# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures for the artifact substrate, and nothing from anywhere else.

This tree has its own ``conftest`` so that it shares no file with the
``Kernel`` migration running in parallel.  It deliberately does not
reuse ``tests/conftest.py``: that one seeds numpy and torch for tests that
generate stimulus, and nothing here does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The repository root, four levels up from ``tests/kernels/artifacts/``.
FINN_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def finn_root() -> Path:
    return FINN_ROOT


@pytest.fixture(scope="session", params=("dataflow", "kernels"))
def production_source_root(request: pytest.FixtureRequest) -> Path:
    """Both planning packages remain independent of tool execution."""

    return FINN_ROOT / "src" / "finn" / str(request.param)
