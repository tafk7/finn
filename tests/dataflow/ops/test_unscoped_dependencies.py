# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only producer addresses let a consumer save before upstream nodes do."""

from copy import deepcopy

import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType

from dataflow.ops.test_graph_context import _replay_chain
from dataflow.ops.test_partial_type_graph import DecisionTypeOp, _chain
from finn.custom_op.dataflow import custom_op
from finn.dataflow._engine import Decided
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOpError
from finn.dataflow.ops.model_effects import ModelReadKind, validate_model_read_set
from finn.dataflow.ops.native import SCHEMA_VERSION_ATTRIBUTE, SCOPE_ID_ATTRIBUTE
from finn.dataflow.ops.persistence import apply_graph_effects


def _remove_ids(model):
    for node in model.graph.node:
        kept = [item for item in node.attribute if item.name != SCOPE_ID_ATTRIBUTE]
        del node.attribute[:]
        node.attribute.extend(kept)


def _model(count):
    model, _ = _replay_chain()
    if count == 3:
        value = model.graph.node[-1].output[0]
        model.graph.node.append(
            helper.make_node(
                "ActivationReplayOp",
                [value],
                ["tail"],
                name="third",
                domain=DATAFLOW_DOMAIN,
                neuron_folds=1,
            )
        )
        model.graph.value_info.append(
            helper.make_tensor_value_info(
                "tail",
                TensorProto.FLOAT,
                model.get_tensor_shape(value),
            )
        )
        model.set_tensor_datatype("tail", DataType["INT8"])
    _remove_ids(model)
    return model


@pytest.mark.parametrize("count", [2, 3])
def test_unscoped_chain_queries_and_consumer_first_save_need_no_upstream_identity(count):
    model = _model(count)
    before = model.model.SerializeToString(deterministic=True)
    operation = model.get_customop_wrapper(model.graph.node[-1])
    assert operation.space.operand_type("result") == Decided(DataType["INT8"])
    reads = operation.space.source.inputs[0].producer_reads
    assert any(item.kind is ModelReadKind.NODE_BY_OUTPUT for item in reads.expectations)
    validate_model_read_set(model, reads)
    assert model.model.SerializeToString(deterministic=True) == before
    upstream = [node.SerializeToString(deterministic=True) for node in model.graph.node[:-1]]
    operation.save_space()
    assert [
        node.SerializeToString(deterministic=True) for node in model.graph.node[:-1]
    ] == upstream
    assert [
        any(attr.name == SCOPE_ID_ATTRIBUTE for attr in node.attribute) for node in model.graph.node
    ] == [False] * (count - 1) + [True]


def test_unscoped_upstream_semantic_change_refuses_old_effects_and_allows_fresh_save():
    model = _model(3)
    operation = model.get_customop_wrapper(model.graph.node[-1])
    operation.save_space()
    original_space = operation.space
    effects = original_space.graph_effects()
    next(attr for attr in model.graph.node[0].attribute if attr.name == "neuron_folds").i += 1
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError):
        apply_graph_effects(model, effects)
    assert model.model.SerializeToString(deterministic=True) == before
    assert operation.space is original_space
    operation.save_space()
    assert not any(attr.name == SCOPE_ID_ATTRIBUTE for attr in model.graph.node[0].attribute)


@pytest.mark.parametrize("cached_type", [True, False])
def test_unscoped_duplicate_producer_refuses_read_and_save_without_mutation_then_retries(
    cached_type,
):
    model = _model(2)
    if not cached_type:
        model.set_tensor_datatype(model.graph.node[0].output[0], None)
    operation = model.get_customop_wrapper(model.graph.node[-1])
    reads = operation.space.source.inputs[0].producer_reads
    original_space = operation.space
    model.graph.node.append(deepcopy(model.graph.node[0]))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="exactly one producer"):
        validate_model_read_set(model, reads)
    with pytest.raises(DataflowOpError):
        operation.save_space()
    assert model.model.SerializeToString(deterministic=True) == before
    assert operation.space is original_space and operation.recorded_scope_id() is None
    del model.graph.node[-1]
    operation.save_space()
    assert operation.recorded_scope_id()


def test_unscoped_native_precision_change_refuses_old_consumer_effects(monkeypatch):
    monkeypatch.setitem(custom_op, DecisionTypeOp.__name__, DecisionTypeOp)
    model = _chain(DecisionTypeOp)
    _remove_ids(model)
    node = model.graph.node[0]
    node.attribute.extend(
        [
            helper.make_attribute("precision", 16),
            helper.make_attribute(
                SCHEMA_VERSION_ATTRIBUTE, DecisionTypeOp.space_type.schema_version
            ),
        ]
    )
    operation = model.get_customop_wrapper(model.graph.node[-1])
    assert operation.space.operand_type("result") == Decided(DataType["INT16"])
    operation.save_space()
    effects = operation.space.graph_effects()
    next(attr for attr in model.graph.node[0].attribute if attr.name == "precision").i = 32
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError):
        apply_graph_effects(model, effects)
    assert model.model.SerializeToString(deterministic=True) == before
