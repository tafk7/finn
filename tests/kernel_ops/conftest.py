# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 (``kernel_ops.tfc``) for the tests that only read it, built once a run and
saved: a test loads its own copy (``ModelWrapper(str(path))``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from qonnx.core.modelwrapper import ModelWrapper

from kernel_ops.tfc import EXPORT, built, exported, kernel_ops, prepared


@pytest.fixture(scope="session")
def tfc_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where the run's TFC builds are kept: the parent of every pytest-xdist worker's base
    temporary directory, the run's own under a single process."""
    base = tmp_path_factory.getbasetemp()
    return (base.parent if "PYTEST_XDIST_WORKER" in os.environ else base) / "tfc"


@pytest.fixture(scope="session")
def tfc_export(tfc_cache: Path) -> Path:
    """TFC_W2A2 as Brevitas exports it, its preprocessing model beside it (``exported``)."""
    return built(tfc_cache / "export" / EXPORT, exported)


@pytest.fixture(scope="session")
def tfc_streamlined(tfc_cache: Path, tfc_export: Path) -> Path:
    """TFC_W2A2 through the builder's graph-preparation phase (``prepared``)."""
    return built(
        tfc_cache / "streamlined.onnx",
        lambda _: prepared(ModelWrapper(str(tfc_export)), tfc_export.parent),
    )


@pytest.fixture(scope="session")
def tfc_kernel_ops(tfc_cache: Path, tfc_streamlined: Path) -> Path:
    """TFC_W2A2's partition of KernelOps for Ultra96 at 5 ns in the Zynq shell, cut
    once: its body, every choice open (``kernel_ops``)."""
    return built(
        tfc_cache / "kernel_ops.onnx",
        lambda directory: kernel_ops(
            ModelWrapper(str(tfc_streamlined)), directory / "kernel_ops_cut"
        ),
    )
