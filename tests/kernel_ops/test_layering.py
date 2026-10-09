# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The rows of the layer table (``tests/layering.py``) that this tree checks.

They are the layers above the kernels: the platform registry, the KernelOps,
their transformations, the kernel-partition facts, util, the XSim testbench, the
executors, the harness and the flow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from layering import Layer, checked_by, sources, violations


@pytest.mark.parametrize("layer", checked_by(Path(__file__).parent), ids=lambda layer: layer.name)
def test_imports_follow_the_layer_table(layer: Layer) -> None:
    assert sources(layer), layer.name
    assert not violations(layer)
