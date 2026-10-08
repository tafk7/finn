# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pytest side of the RTL harness (``finn.harness.rtl``): the marker an XSim test carries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from finn.harness.toolchain import vivado_simulator

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
