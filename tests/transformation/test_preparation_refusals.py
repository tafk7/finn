# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Trunc and weight Quant preparation refuses an input graph it cannot convert as a
``Refusal``, each with its code, its transformation as owner and the node it is about,
so a build counts it as a refusal, not a bug."""

import pytest

import numpy as np
import onnx.numpy_helper as nph
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.reporting.findings import FindingKind, Refusal
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import (
    AvgPoolAndTruncToQuantAvgPool,
    _get_signed_from_upstream,
)

QONNX = "qonnx.custom_op.general"
OWNER = "AvgPoolAndTruncv2ToQuantAvgPool"


def refused(error: pytest.ExceptionInfo) -> tuple:
    """The one finding's kind, code, owner and details."""
    assert type(error.value) is Refusal
    ((kind, code, owner, details),) = [
        (each.kind, each.code, each.owner, dict(each.details)) for each in error.value.findings
    ]
    return kind, code, owner, details


def trunc_after(nodes, source: str) -> ModelWrapper:
    """``nodes`` from the graph input ``x``, then a Trunc of ``source``; no datatype
    is stated, so the Trunc's sign is looked for up the graph."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    trunc = helper.make_node(
        "Trunc", [source, "scale", "zeropt", "in_bits", "out_bits"], ["y"], "Trunc_0", domain=QONNX
    )
    graph = helper.make_graph([*nodes, trunc], "g", [x], [y])
    return ModelWrapper(helper.make_model(graph))


@pytest.mark.transform
@pytest.mark.parametrize(
    ("nodes", "source", "code"),
    (
        ([helper.make_node("Relu", ["x"], ["r"], "Relu_0")], "r", "trunc-sign-not-preserved"),
        ([], "x", "trunc-sign-unstated"),
        (
            [
                helper.make_node("Relu", ["x"], ["a"], "Relu_0"),
                helper.make_node("Relu", ["x"], ["b"], "Relu_1"),
                helper.make_node("Add", ["a", "b"], ["s"], "Add_0"),
            ],
            "s",
            "trunc-sign-join",
        ),
    ),
)
def test_a_trunc_whose_input_sign_is_not_found_is_refused(nodes, source, code):
    model = trunc_after(nodes, source)
    (trunc,) = [node for node in model.graph.node if node.op_type == "Trunc"]
    with pytest.raises(Refusal) as error:
        _get_signed_from_upstream(model, trunc, OWNER)
    assert refused(error) == (FindingKind.REJECTION, code, OWNER, {"node": "Trunc_0"})


@pytest.mark.transform
def test_a_trunc_opset_not_converted_is_refused():
    model = trunc_after([], "x")
    model.model.opset_import.append(helper.make_opsetid(QONNX, 3))
    with pytest.raises(Refusal) as error:
        model.transform(AvgPoolAndTruncToQuantAvgPool())
    kind, code, owner, details = refused(error)
    assert (kind, code, owner, details) == (
        FindingKind.REJECTION,
        "trunc-opset",
        "AvgPoolAndTruncToQuantAvgPool",
        {},
    )
    assert "Trunc opset version 3" in error.value.findings[0].message


@pytest.mark.transform
def test_a_weight_quant_with_no_consumer_is_refused():
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, [2])
    quant = helper.make_node(
        "Quant", ["w", "scale", "zeropt", "bits"], ["q"], "Quant_0", domain=QONNX
    )
    inits = [
        nph.from_array(np.array([0.5, -1.0], dtype=np.float32), "w"),
        nph.from_array(np.array(0.5, dtype=np.float32), "scale"),
        nph.from_array(np.array(0.0, dtype=np.float32), "zeropt"),
        nph.from_array(np.array(4.0, dtype=np.float32), "bits"),
    ]
    model = ModelWrapper(helper.make_model(helper.make_graph([quant], "g", [], [q], inits)))
    with pytest.raises(Refusal) as error:
        model.transform(FoldQuantWeights())
    assert refused(error) == (
        FindingKind.REJECTION,
        "weight-quant-unconsumed",
        "FoldQuantWeights",
        {"node": "Quant_0"},
    )
