# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pytest side of the XSim testbench (``finn.core.executors.xsim.rtl``): the marker an
XSim test carries, and the skip of one that synthesizes an HLS leaf first."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from finn.harness.toolchain import vivado_simulator
from finn.util.toolchain import machine_toolchain

_Test = TypeVar("_Test", bound=Callable[..., object])


def requires_xsim(test: _Test) -> _Test:
    """Marked ``xsim``, and skipped without a selected Vivado.

    The marker is what keeps the fast gate fast: ``check-kernels.sh`` deselects
    ``xsim`` whether or not Vivado is selected, and ``xsim-sweep.sh`` runs it.
    """
    skip = pytest.mark.skipif(
        not vivado_simulator(), reason="Vivado simulator tools are unavailable"
    )
    marked: _Test = pytest.mark.xsim(skip(test))  # a dynamic mark is typed Any
    return marked


def hls_synthesizer() -> bool:
    """Whether the machine's toolchain selects an HLS frontend and an HLS installation."""
    toolchain = machine_toolchain()
    frontend = toolchain.selection.hls_frontend
    if frontend is None:
        return False
    try:
        toolchain.command(frontend)
        toolchain.hls_installation()
    except (FileNotFoundError, LookupError):
        return False
    return True


def requires_hls(test: _Test) -> _Test:
    """``requires_xsim``, and skipped without an HLS frontend to synthesize with."""
    skip = pytest.mark.skipif(not hls_synthesizer(), reason="no HLS frontend is selected")
    marked: _Test = requires_xsim(skip(test))
    return marked
