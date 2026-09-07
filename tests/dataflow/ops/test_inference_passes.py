# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-pass ownership on the actual wrappers used by QONNX callbacks."""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.analysis.tensor_value_summary import summarize_tensor_values
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes as QonnxInferDataTypes
from qonnx.transformation.infer_shapes import InferShapes as QonnxInferShapes

from finn.analysis.verify_custom_nodes import verify_nodes
from finn.dataflow.ops import reconstruction
from finn.dataflow.ops.base import DATAFLOW_DOMAIN
from finn.dataflow.ops.inference import InferDataTypes, InferShapes
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.ops.replay.op import ActivationReplayOp


def _replay_chain(*, stale_types=False):
    names = ["activation", "first", "second", "output"]
    nodes = [
        helper.make_node(
            "ActivationReplayOp",
            [names[index]],
            [names[index + 1]],
            name=f"replay{index}",
            domain=DATAFLOW_DOMAIN,
            neuron_folds=1,
        )
        for index in range(3)
    ]
    tensors = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 8]) for name in names]
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                nodes, "replay_chain", tensors[:1], tensors[-1:], value_info=tensors[1:-1]
            ),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    for index, name in enumerate(names):
        model.set_tensor_datatype(name, DataType["FLOAT32" if stale_types and index else "INT8"])
    # A real initializer ensures the pass analyzes values, not just an empty graph.
    model.set_initializer("activation", np.ones((2, 8), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _count_analyses(monkeypatch):
    calls = []
    original = reconstruction.initializer_value_summaries

    def counted(model):
        result = original(model)
        calls.append((model, result))
        return result

    monkeypatch.setattr(reconstruction, "initializer_value_summaries", counted)
    return calls


@pytest.mark.parametrize("transformation", [InferDataTypes, InferShapes])
@pytest.mark.parametrize("invoke", ["apply", "transform"])
def test_three_real_replay_callbacks_share_one_analysis(transformation, invoke, monkeypatch):
    model = _replay_chain()
    before = model.model.SerializeToString(deterministic=True)
    calls = _count_analyses(monkeypatch)
    callback_models = []
    callback_name = (
        "infer_node_datatype" if transformation is InferDataTypes else "make_shape_compatible_op"
    )
    original_callback = getattr(ActivationReplayOp, callback_name)

    def callback(operation, actual_model):
        callback_models.append(actual_model)
        return original_callback(operation, actual_model)

    monkeypatch.setattr(ActivationReplayOp, callback_name, callback)
    if invoke == "apply":
        result, changed = transformation().apply(model)
        assert changed is False
    else:
        result = model.transform(transformation())  # default deepcopy and cleanup
        assert model.model.SerializeToString(deterministic=True) == before
    assert len(calls) == 1
    assert len(callback_models) == 3
    assert all(actual is calls[0][0] for actual in callback_models)
    assert (calls[0][0] is model) == (invoke == "apply")
    assert result.get_tensor_datatype("output") == DataType["INT8"]
    assert result.get_tensor_shape("output") == [2, 8]


def test_datatype_fixed_point_iterations_each_own_one_analysis(monkeypatch):
    model = _replay_chain(stale_types=True)
    calls = _count_analyses(monkeypatch)
    applications = []
    original = QonnxInferDataTypes.apply

    def apply(transformation, actual_model):
        applications.append(actual_model)
        return original(transformation, actual_model)

    monkeypatch.setattr(QonnxInferDataTypes, "apply", apply)
    result = model.transform(InferDataTypes())
    assert len(applications) == 2  # changing pass, then unchanged pass: QONNX's existing contract
    assert len(calls) == len(applications)
    assert all(current is result for current in applications)
    assert all(current is result for current, _ in calls)
    assert model.get_tensor_datatype("output") == DataType["FLOAT32"]
    assert result.get_tensor_datatype("output") == DataType["INT8"]


def test_analysis_sees_transform_preprocessing_on_the_copy(monkeypatch):
    model = _replay_chain()
    model.fix_float64 = True
    values = np.ones((2, 8), dtype=np.float64)
    model.set_initializer("activation", values)
    calls = _count_analyses(monkeypatch)
    result = model.transform(InferDataTypes())
    assert len(calls) == 1
    actual_model, summaries = calls[0]
    assert actual_model is result and actual_model is not model
    assert model.get_initializer("activation").dtype == np.float64
    assert result.get_initializer("activation").dtype == np.float32
    assert summaries["activation"] == summarize_tensor_values(values.astype(np.float32))
    assert summaries["activation"] != summarize_tensor_values(values)


def test_later_passes_reanalyze_current_initializer_values(monkeypatch):
    model = _replay_chain()
    calls = _count_analyses(monkeypatch)
    transform = InferDataTypes()
    transform.apply(model)
    model.set_initializer("activation", np.zeros((2, 8), dtype=np.float32))
    transform.apply(model)
    assert len(calls) == 2
    assert calls[0][1]["activation"] != calls[1][1]["activation"]


def test_verification_still_owns_one_analysis_for_three_operations(monkeypatch):
    model = _replay_chain()
    calls = _count_analyses(monkeypatch)
    assert verify_nodes(model) == {"ActivationReplayOp": []}
    assert len(calls) == 1 and calls[0][0] is model


def test_pinned_qonnx_transformation_classes_are_not_patched():
    assert QonnxInferDataTypes.apply.__module__ == "qonnx.transformation.infer_datatypes"
    assert QonnxInferShapes.apply.__module__ == "qonnx.transformation.infer_shapes"
    assert InferDataTypes.apply is not QonnxInferDataTypes.apply
    assert InferShapes.apply is not QonnxInferShapes.apply


@pytest.mark.parametrize("allow_scaled", [False, True])
def test_non_dataflow_inference_retains_qonnx_options_without_source_analysis(
    allow_scaled, monkeypatch
):
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["input"], ["output"])],
                "identity",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
            )
        )
    )
    model.set_tensor_datatype("input", DataType["SCALEDINT<32>"])
    calls = _count_analyses(monkeypatch)
    result = model.transform(InferDataTypes(allow_scaledint_dtypes=allow_scaled))
    assert not calls
    expected = "SCALEDINT<32>" if allow_scaled else "FLOAT32"
    assert result.get_tensor_datatype("output") == DataType[expected]
