# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The QONNX factory returns an adapter with a separate ready-to-query Space."""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp

from finn.kernels._engine import Decided, RequestError
from finn.parked.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.parked.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.parked.dataflow.ops.replay.op import ReplaySpace
from finn.parked.dataflow.ops.space import DataflowSpace
from finn.kernels.space.declarations import Space


def _model():
    node = helper.make_node(
        "ActivationReplayOp", ["x"], ["y"], name="replay", domain=DATAFLOW_DOMAIN, neuron_folds=2
    )

    def tensor(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    model = ModelWrapper(
        helper.make_model(
            helper.make_graph([node], "replay", [tensor("x", [2, 8])], [tensor("y", [4, 8])]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("x", DataType["INT8"])
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def test_factory_immediately_exposes_distinct_model_free_space_without_writes():
    model = _model()
    before = model.model.SerializeToString(deterministic=True)
    op = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(op, DataflowOp) and isinstance(op, CustomOp)
    assert not isinstance(op, Space)
    assert isinstance(op.space, DataflowSpace) and isinstance(op.space, ReplaySpace)
    assert not isinstance(op.space, CustomOp)
    assert op.space is not op
    assert not hasattr(op.space, "_model") and not hasattr(op.space, "onnx_node")
    assert not hasattr(op, "bind") and not hasattr(op, "hydrate")
    assert op.space.operand_type("result") == Decided(DataType["INT8"])
    assert model.model.SerializeToString(deterministic=True) == before


def test_factory_queries_without_creating_a_persisted_scope_id():
    model = _model()
    node = model.graph.node[0]
    kept = [item for item in node.attribute if item.name != "dataflow_scope_id"]
    del node.attribute[:]
    node.attribute.extend(kept)
    before = model.model.SerializeToString(deterministic=True)
    op = model.get_customop_wrapper(node)
    assert op.space.recorded_scope_id() == ""
    assert op.space.operand_type("result") == Decided(DataType["INT8"])
    assert op.space.operand_domain("result").value.extents == (4, 8)
    assert model.model.SerializeToString(deterministic=True) == before


def test_execution_uses_the_frozen_node_slots_with_its_frozen_facts():
    model = _model()
    op = model.get_customop_wrapper(model.graph.node[0])
    model.graph.node[0].input[0] = "renamed_x"
    model.graph.node[0].output[0] = "renamed_y"
    values = {"x": np.arange(16, dtype=np.float32).reshape(2, 8)}
    op.execute_node(values, model.graph)
    np.testing.assert_array_equal(values["y"], np.repeat(values["x"], 2, axis=0))
    assert "renamed_y" not in values


def test_successor_exploration_save_and_new_factory_keep_old_spaces_immutable():
    model = _model()
    op = model.get_customop_wrapper(model.graph.node[0])
    old = op.space
    proposal = old.commit_choices({"kernel__simd": 4})
    assert old.recorded() == {} and op.space is old
    saved = op.save_space(proposal)
    assert op.space is saved and saved.recorded()["kernel.simd"] == 4
    assert old.recorded() == {}
    model.set_tensor_datatype("x", DataType["INT4"])
    fresh = model.get_customop_wrapper(model.graph.node[0])
    assert fresh.space.operand_type("result") == Decided(DataType["INT4"])
    assert saved.operand_type("result") == Decided(DataType["INT8"])


def test_failed_save_preserves_graph_and_adapter_pointer():
    model = _model()
    op = model.get_customop_wrapper(model.graph.node[0])
    proposal = op.space.commit_choices({"kernel__simd": 4})
    old = op.space
    model.set_tensor_shape("x", [2, 6])
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises((DataflowOpError, RequestError, ValueError)):
        op.save_space(proposal)
    assert op.space is old
    assert model.model.SerializeToString(deterministic=True) == before


def test_context_refresh_preserves_unsaved_choices_and_failure_preserves_pointer():
    model = _model()
    op = model.get_customop_wrapper(model.graph.node[0])
    op.space = op.space.commit_choices({"kernel__simd": 4})
    before = model.model.SerializeToString(deterministic=True)
    assert op.set_context() is op
    assert op.space.recorded()["kernel.simd"] == 4
    assert model.model.SerializeToString(deterministic=True) == before
    old = op.space
    model.set_tensor_shape("x", [2, 6])
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises((DataflowOpError, RequestError, ValueError)):
        op.set_context()
    assert op.space is old
    assert model.model.SerializeToString(deterministic=True) == before


def test_standard_callbacks_use_factory_initialized_space():
    model = _model()
    op = model.get_customop_wrapper(model.graph.node[0])
    assert op.make_shape_compatible_op(model).op_type == "RandomNormal"
    op.infer_node_datatype(model)
    assert model.get_tensor_datatype("y") == DataType["INT8"]
    values = {"x": np.arange(16, dtype=np.float32).reshape(2, 8)}
    op.execute_node(values, model.graph)
    np.testing.assert_array_equal(values["y"], np.repeat(values["x"], 2, axis=0))
    assert op.verify_node() == []


def _remove_ids(model):
    for node in model.graph.node:
        kept = [item for item in node.attribute if item.name != "dataflow_scope_id"]
        del node.attribute[:]
        node.attribute.extend(kept)


def test_first_save_identifies_only_the_exact_supplied_node():
    model = _model()
    clone = model.graph.node.add()
    clone.CopyFrom(model.graph.node[0])
    clone.output[0] = "z"
    model.graph.output.append(helper.make_tensor_value_info("z", TensorProto.FLOAT, [4, 8]))
    _remove_ids(model)
    first_before = model.graph.node[0].SerializeToString(deterministic=True)
    op = model.get_customop_wrapper(model.graph.node[1])
    old = op.space
    saved = op.save_space(old.commit_choices({"kernel__simd": 4}))
    assert op.space is saved and saved.recorded_scope_id()
    assert old.recorded_scope_id() == ""
    assert model.graph.node[0].SerializeToString(deterministic=True) == first_before
    assert not any(item.name == "dataflow_scope_id" for item in model.graph.node[0].attribute)
    assert op.onnx_node is model.graph.node[1]


def test_first_save_replaces_an_empty_scope_attribute():
    model = _model()
    _remove_ids(model)
    model.graph.node[0].attribute.append(helper.make_attribute("dataflow_scope_id", ""))
    op = model.get_customop_wrapper(model.graph.node[0])
    saved = op.save_space()
    ids = [
        item.s.decode()
        for item in model.graph.node[0].attribute
        if item.name == "dataflow_scope_id"
    ]
    assert ids == [saved.recorded_scope_id()]
    assert ids[0]


@pytest.mark.parametrize("failure", ("invalid-choice", "unsupported-schema"))
def test_first_save_failure_preserves_scope_space_and_graph(failure):
    model = _model()
    _remove_ids(model)
    op = model.get_customop_wrapper(model.graph.node[0])
    old = op.space
    proposal = old.commit_choices({"kernel__simd": 4})
    if failure == "invalid-choice":
        model.set_tensor_shape("x", [2, 6])
    else:
        model.graph.node[0].attribute.append(helper.make_attribute("dataflow_schema_version", 999))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises((DataflowOpError, RequestError, ValueError)):
        op.save_space(proposal)
    assert op.space is old and op.recorded_scope_id() is None
    assert model.model.SerializeToString(deterministic=True) == before


def test_first_save_final_hydration_failure_rolls_back_and_same_adapter_can_retry(monkeypatch):
    from finn.parked.dataflow.ops import reconstruction  # noqa: PLC0415

    model = _model()
    _remove_ids(model)
    op = model.get_customop_wrapper(model.graph.node[0])
    old = op.space
    proposal = old.commit_choices({"kernel__simd": 4})
    before = model.model.SerializeToString(deterministic=True)
    original = reconstruction.build_space

    def fail_live_finish(space_type, current_model, node, **kwargs):
        if current_model is model and any(
            item.name == "dataflow_scope_id" for item in node.attribute
        ):
            raise DataflowOpError("injected final hydration failure")
        return original(space_type, current_model, node, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(reconstruction, "build_space", fail_live_finish)
        with pytest.raises(DataflowOpError, match="injected final"):
            op.save_space(proposal)
    assert model.model.SerializeToString(deterministic=True) == before
    assert op.space is old and op.recorded_scope_id() is None
    assert op.onnx_node is model.graph.node[0]
    saved = op.save_space(proposal)
    assert op.space is saved and op.recorded_scope_id()
