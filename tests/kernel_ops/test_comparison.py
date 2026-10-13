# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph comparison (``finn.analysis.comparison``): two graphs executed on the same
inputs, compared value for value, the differences returned as findings. Hand-made graphs
of one Add each."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.analysis.comparison import Run, compared_with
from finn.core.space import Finding, FindingKind


def adding(constant: float, output: str = "y", rows: int = 1) -> ModelWrapper:
    """``x`` (two INT8) plus ``constant`` (in ``rows`` rows) into ``s``, then doubled into
    ``output``, shapes inferred."""
    nodes = [
        helper.make_node("Add", ["x", "c"], ["s"], name="Add_0"),
        helper.make_node("Add", ["s", "s"], [output], name="Add_1"),
    ]
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                nodes,
                "g",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info(output, TensorProto.FLOAT, None)],
            )
        )
    )
    model.set_initializer("c", np.full((rows, 1), constant, dtype=np.float32))
    model.set_tensor_datatype("x", DataType["INT8"])
    return model.transform(InferShapes())


INPUTS: list[dict[str, Any]] = [{"x": np.array([[1, 2]])}, {"x": np.array([[-3, 0]])}]


def test_equal_graphs_compare_with_nothing_found_on_every_run() -> None:
    assert adding(1.0, "z").analysis(compared_with(adding(1.0), INPUTS)) == {
        "runs": 2,
        "findings": (),
    }


def test_a_difference_is_found_once_naming_the_tensor_and_its_first_element() -> None:
    """By default each graph output against the other's at the same index, whatever its
    name; a difference on every run is one finding, at the first run."""
    compared = adding(2.0, "z").analysis(compared_with(adding(1.0), INPUTS))
    assert compared["runs"] == 2
    (found,) = compared["findings"]
    assert (found.kind, found.code, dict(found.details)) == (
        FindingKind.REJECTION,
        "values-differ",
        {"tensor": "z", "differing": 2},
    )
    assert found.message == (
        "z: 2 of 2 values differ from the reference's y, none explained (the predicates "
        "given: none); first at (0, 0): the reference 4.0, the model 6.0"
    )


def test_named_tensors_are_compared_and_a_predicate_explains_a_difference() -> None:
    """``tensors`` pairs any two tensors, the reference's first; a predicate's elements
    pass."""
    pairs = [("s", "s")]
    (found,) = adding(2.0).analysis(compared_with(adding(1.0), INPUTS, tensors=pairs))["findings"]
    assert dict(found.details)["tensor"] == "s"

    def off_by_one(run: Run, expected: str, computed: str) -> npt.NDArray[np.bool_]:
        return np.asarray(run.computed[computed] - run.expected[expected] == 1)

    explained = compared_with(
        adding(1.0), INPUTS, tensors=pairs, explains={"off-by-one": off_by_one}
    )
    assert adding(2.0).analysis(explained)["findings"] == ()


def test_a_shape_that_differs_is_found_and_a_check_adds_its_findings() -> None:
    wider = adding(1.0, rows=2)
    (found,) = wider.analysis(compared_with(adding(1.0), INPUTS[:1]))["findings"]
    assert found.message == "y: the reference's y has shape (1, 2), the model's (2, 2)"

    def noted(run: Run) -> list[Finding]:
        return [Finding(FindingKind.LIMITATION, "noted", "test", "noted", (("tensor", "x"),))]

    checked = adding(1.0).analysis(compared_with(adding(1.0), INPUTS, checks=(noted,)))
    assert [(each.code, each.kind) for each in checked["findings"]] == [
        ("noted", FindingKind.LIMITATION)
    ]
