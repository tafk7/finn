# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The initializers' annotations against their values (``finn.analysis.initializers``),
on hand-made graphs: a MatMul's weights within their integer annotation, and weights
their annotation does not hold, a value no integer or out of its range. TFC's prepared
graph is checked in ``test_preparation.py``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.analysis.initializers import annotated_initializers
from finn.core.reporting.findings import FindingKind
from kernel_ops.test_preparation import graph


def weighted(weights: Any, annotation: str) -> ModelWrapper:
    """An INT4 input times stored ``weights`` annotated ``annotation``, then a stored
    float scale, unannotated."""
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["s"], name="MatMul_0"),
        helper.make_node("Mul", ["s", "scale"], ["y"], name="Mul_0"),
    ]
    return graph(
        nodes, {"x": ([1, 2], "INT4")}, ["y"], {"w": weights, "scale": [0.25]}, {"w": annotation}
    )


def test_weights_their_annotation_holds_give_no_finding() -> None:
    held = weighted([[-8.0, 7.0], [0.0, 3.0]], "INT4").analysis(annotated_initializers)
    assert (held["checked"], held["findings"]) == (("w",), ())


def test_a_float_annotation_is_not_checked() -> None:
    held = weighted([[0.5, 1.5], [2.0, 3.0]], "FLOAT32").analysis(annotated_initializers)
    assert (held["checked"], held["findings"]) == ((), ())


@pytest.mark.parametrize(
    ("weights", "annotation", "unheld", "example"),
    [
        pytest.param([[1.0, 0.5], [2.0, 3.0]], "INT4", 1, 0.5, id="no-integer"),
        pytest.param([[1.0, 2.0], [-3.0, 1.0]], "INT2", 2, 2, id="out-of-range"),
        pytest.param([[1.0, -1.0], [0.0, 1.0]], "BIPOLAR", 1, 0, id="bipolar-zero"),
    ],
)
def test_weights_their_annotation_does_not_hold_are_named(
    weights: list[list[float]], annotation: str, unheld: int, example: float
) -> None:
    held = weighted(np.asarray(weights), annotation).analysis(annotated_initializers)
    (found,) = held["findings"]
    assert (found.kind, found.code, found.owner) == (
        FindingKind.REJECTION,
        "annotation-unsound",
        "initializers",
    )
    assert dict(found.details) == {
        "tensor": "w",
        "annotation": annotation,
        "unheld": unheld,
        "example": example,
    }
    assert held["checked"] == ("w",)
