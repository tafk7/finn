# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The rows of the layer table (``tests/support/layering.py``) that this tree checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from support.layering import BY_NAME, Layer, checked_by, sources, violations


@pytest.mark.parametrize("layer", checked_by(Path(__file__).parent), ids=lambda layer: layer.name)
def test_imports_follow_the_layer_table(layer: Layer) -> None:
    assert sources(layer), layer.name
    assert not violations(layer)


def test_the_values_import_only_the_findings() -> None:
    """``finn.dataflow`` is engine-free: its row names only the findings its refusals
    carry, so an import of ``finn.core.space`` (or any other FINN layer) is a
    violation."""

    assert BY_NAME["dataflow"].imports == ("reporting",)
    assert BY_NAME["tests.dataflow"].imports == (
        "reporting",
        "dataflow",
        "tests.layering",
        "tests.refusals",
        "tests.value_classes",
    )


def test_the_values_and_the_kernels_may_import_numpy_and_nothing_else() -> None:
    """numpy is the one third-party package beside QONNX's datatypes and their ranges (and
    pyslang, for the kernels' pin checks) that the values and the kernels may import."""

    assert BY_NAME["dataflow"].packages == ("qonnx.core.datatype", "qonnx.core.ranges", "numpy")
    assert BY_NAME["kernels"].packages == (
        "qonnx.core.datatype",
        "qonnx.core.ranges",
        "numpy",
        "pyslang",
    )
