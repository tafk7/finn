# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The rows of the layer table (``tests/layering.py``) that this tree checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from layering import BY_NAME, Layer, checked_by, sources, violations


@pytest.mark.parametrize("layer", checked_by(Path(__file__).parent), ids=lambda layer: layer.name)
def test_imports_follow_the_layer_table(layer: Layer) -> None:
    assert sources(layer), layer.name
    assert not violations(layer)


def test_the_values_import_no_other_layer() -> None:
    """``finn.dataflow`` is engine-free: its row names no layer, so an import of
    ``finn.core.space`` (or any other FINN layer) is a violation."""

    assert BY_NAME["dataflow"].imports == ()
    assert BY_NAME["tests.dataflow"].imports == (
        "dataflow",
        "tests.layering",
        "tests.value_classes",
    )


def test_the_values_and_the_kernels_may_import_numpy_and_nothing_else() -> None:
    """numpy is the one third-party package beside QONNX's datatypes (and pyslang, for
    the kernels' pin checks) that the values and the kernels may import."""

    assert BY_NAME["dataflow"].packages == ("qonnx.core.datatype", "numpy")
    assert BY_NAME["kernels"].packages == ("qonnx.core.datatype", "numpy", "pyslang")
