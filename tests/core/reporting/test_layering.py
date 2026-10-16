# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The rows of the layer table (``tests/support/layering.py``) that this tree checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from support.layering import BY_NAME, Layer, checked_by, permits, sources, violations


@pytest.mark.parametrize("layer", checked_by(Path(__file__).parent), ids=lambda layer: layer.name)
def test_imports_follow_the_layer_table(layer: Layer) -> None:
    assert sources(layer), layer.name
    assert not violations(layer)


def test_reporting_imports_the_standard_library_only() -> None:
    """``finn.core.reporting`` is below every other layer: its row names no layer and no
    third-party package, so the engine and the build can both import it."""

    reporting = BY_NAME["reporting"]
    assert reporting.imports == () and reporting.packages == ()
    assert not permits(reporting, "finn.core.space.results.Rejected")
    assert permits(BY_NAME["space"], "finn.core.reporting.findings.Finding")
