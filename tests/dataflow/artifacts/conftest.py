# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures for the artifact substrate, and nothing from anywhere else.

This tree has its own ``conftest`` so that it shares no file with the
``DataflowDesign`` migration running in parallel.  It deliberately does not
reuse ``tests/conftest.py``: that one seeds numpy and torch for tests that
generate stimulus, and nothing here does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The repository root, four levels up from ``tests/dataflow/artifacts/``.
FINN_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def finn_root() -> Path:
    return FINN_ROOT


@pytest.fixture(scope="session")
def dataflow_source_root() -> Path:
    """Everything under ``finn.dataflow``, for the whole-package invariants."""

    return FINN_ROOT / "src" / "finn" / "dataflow"


@pytest.fixture(scope="session")
def artifacts_source_root() -> Path:
    """The package this effort owns."""

    return FINN_ROOT / "src" / "finn" / "dataflow" / "artifacts"
