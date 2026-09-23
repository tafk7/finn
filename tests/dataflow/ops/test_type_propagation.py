# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Producer contracts, rather than cached annotations, authorize downstream types."""

import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType

from dataflow.ops.test_dataflow_op import _mvau_model
from finn.kernels._engine import Decided
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOpError
from finn.dataflow.ops.native import SCHEMA_VERSION_ATTRIBUTE
from finn.dataflow.ops.persistence import apply_graph_effects, assign_dataflow_scope_ids
from finn.dataflow.ops.type_context import producer_type


def _chain():
    model = _mvau_model()
    for index in range(2):
        name = f"consumer{index}"
        model.graph.node.append(
            helper.make_node(
                "ActivationReplayOp",
                ["output"],
                [name],
                domain=DATAFLOW_DOMAIN,
                name=name,
                neuron_folds=1,
            )
        )
        model.graph.value_info.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 4])
        )
        model.set_tensor_datatype(name, DataType["INT32"])
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _precision(model, name):
    node = model.graph.node[0]
    kept = [item for item in node.attribute if item.name not in {"accDataType", "outputDataType"}]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.extend(
        [helper.make_attribute("accDataType", name), helper.make_attribute("outputDataType", name)]
    )


def test_raw_producer_edit_does_not_authorize_stale_annotation_and_queries_are_pure():
    model = _chain()
    old = model.get_customop_wrapper(model.graph.node[1]).space
    assert old.operand_type("result") == Decided(DataType["INT32"])
    _precision(model, "INT16")
    before = model.model.SerializeToString(deterministic=True)
    assert model.get_tensor_datatype("output") == DataType["INT32"]
    for node in model.graph.node[1:]:
        use = model.get_customop_wrapper(node).space
        assert use.operand_type("activation") == Decided(DataType["INT16"])
        assert use.operand_type("result") == Decided(DataType["INT16"])
    assert old.operand_type("result") == Decided(DataType["INT32"])
    assert model.model.SerializeToString(deterministic=True) == before


def test_checked_update_invalidates_both_consumers_and_preserves_choices():
    model = _chain()
    for node in model.graph.node[1:]:
        bound = model.get_customop_wrapper(node).space
        node.attribute.append(helper.make_attribute("kernel__simd", 2))
        node.attribute.append(helper.make_attribute(SCHEMA_VERSION_ATTRIBUTE, bound.schema_version))
    _precision(model, "INT16")
    producer = model.get_customop_wrapper(model.graph.node[0])
    producer.save_space()
    assert model.get_tensor_datatype("output") == DataType["INT16"]
    for index, node in enumerate(model.graph.node[1:]):
        assert any(item.name == "kernel__simd" and item.i == 2 for item in node.attribute)
        assert not any(
            item.tensor_name == f"consumer{index}"
            and any(entry.key == "finn_datatype" for entry in item.quant_parameter_tensor_names)
            for item in model.graph.quantization_annotation
        )
        assert producer_type(model, f"consumer{index}") == Decided(DataType["INT16"])


def test_rejected_producer_type_remains_rejected_through_consumer():
    model = _chain()
    _precision(model, "INT16")
    node = model.graph.node[0]
    for attribute in node.attribute:
        if attribute.name == "outputDataType":
            attribute.s = b"INT8"
    produced = producer_type(model, "output")
    assert not isinstance(produced, Decided)
    use = model.get_customop_wrapper(model.graph.node[1]).space
    assert not isinstance(use.operand_type("result"), Decided)
    assert use.source.inputs[0].datatype is None


def test_producer_change_invalidates_a_consumer_write_plan_atomically():
    model = _chain()
    consumer = model.get_customop_wrapper(model.graph.node[1]).space
    effects = consumer.graph_effects()
    _precision(model, "INT16")
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="different problem|read"):
        apply_graph_effects(model, effects)
    assert model.model.SerializeToString(deterministic=True) == before
