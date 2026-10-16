# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The rows of the layer table (``tests/support/layering.py``) that this tree checks.

The engine's own imports: the standard library, its native dependency and the findings
below it. The table's and the walker's own tests, and the other layers' cases, are in
``tests/layers``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support.layering import BY_NAME, Layer, checked_by, permits, sources, violations


@pytest.mark.parametrize("layer", checked_by(Path(__file__).parent), ids=lambda layer: layer.name)
def test_imports_follow_the_layer_table(layer: Layer) -> None:
    assert sources(layer), layer.name
    assert not violations(layer)


@pytest.mark.parametrize(("layer", "name"), [("space", "finn.dataflow.tensor"), ("space", "numpy")])
def test_the_table_rejects_an_import_across_its_order(layer: str, name: str) -> None:
    assert not permits(BY_NAME[layer], name)


@pytest.mark.parametrize(
    ("layer", "name"),
    [
        ("space", "greenlet"),
        ("space", "collections.abc.Mapping"),
        ("space", "finn.core.reporting.findings.Finding"),
    ],
)
def test_the_table_accepts_an_import_down_its_order(layer: str, name: str) -> None:
    assert permits(BY_NAME[layer], name)
