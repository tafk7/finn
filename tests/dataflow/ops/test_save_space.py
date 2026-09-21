# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Proposed choices are checked against explicit current target facts on save."""

import numpy as np
import pytest
from onnx import helper
from qonnx.core.datatype import DataType

from dataflow.ops.test_dataflow_op import Build, _configure_mvau_point, _mvau_model, _replay_model
from dataflow.ops.test_partial_type_graph import AlternativeTypeOp, _chain
from dataflow.ops.test_persistence_codecs import _model as _codec_model
from finn.custom_op.dataflow import custom_op
from finn.dataflow._engine import Decided, RequestError, Unresolved
from finn.dataflow.ops.base import DataflowOp, DataflowOpError
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.native import (
    AttributeCodec,
    OBSOLETE_FINGERPRINT_ATTRIBUTE,
    serialize_choices,
)
from finn.dataflow.ops.persistence import apply_graph_effects
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.type_context import producer_type
from finn.dataflow.space.declarations import Decision


def _proposal(model=None, *, simd=2):
    model = _mvau_model() if model is None else model
    return _configure_mvau_point(
        MvauDataflowOp(model.graph.node[0]).bind(model, Build()), simd=simd
    )


def _set_precision(node, datatype):
    kept = [item for item in node.attribute if item.name not in {"accDataType", "outputDataType"}]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.extend(
        (
            helper.make_attribute("accDataType", datatype),
            helper.make_attribute("outputDataType", datatype),
        )
    )


def test_save_rechecks_changed_values_shape_and_semantics_without_copying_old_facts():
    model = _mvau_model()
    proposal = _proposal(model)
    old_weights = proposal.source.operand("weight").initializer_value.array_copy()
    weights = np.ones((8, 6), dtype=np.float32)
    model.set_initializer("weight", weights)
    model.set_tensor_shape("activation", [3, 8])
    _set_precision(model.graph.node[0], "INT24")
    model.graph.node[0].attribute.append(
        helper.make_attribute(OBSOLETE_FINGERPRINT_ATTRIBUTE, "obsolete")
    )
    target = MvauDataflowOp(model.graph.node[0])
    saved = target.save_space(model, proposal, Build())
    assert serialize_choices(saved) == serialize_choices(proposal)
    assert np.array_equal(model.get_initializer("weight"), weights)
    assert np.count_nonzero(old_weights) == 0
    assert proposal.source.operand("weight").shape == (8, 4)
    assert model.get_tensor_shape("activation") == [3, 8]
    assert model.get_tensor_shape("output") == [3, 6]
    assert model.get_tensor_datatype("output") == DataType["INT24"]
    assert saved.operand_type("result") == Decided(DataType["INT24"])
    assert not any(
        item.name == OBSOLETE_FINGERPRINT_ATTRIBUTE for item in model.graph.node[0].attribute
    )
    assert (
        MvauDataflowOp(model.graph.node[0]).hydrate(model, Build()).recorded() == saved.recorded()
    )


def test_compatible_choices_save_to_another_source_origin():
    proposal = _proposal()
    target_model = _mvau_model()
    target_model.set_initializer("weight", np.full((8, 4), 2, dtype=np.float32))
    target = MvauDataflowOp(target_model.graph.node[0])
    assert target.recorded_scope_id() != proposal.recorded_scope_id()
    saved = target.save_space(target_model, proposal, Build())
    assert saved.recorded_scope_id() == target.recorded_scope_id()
    assert saved.recorded() == proposal.recorded()
    assert np.all(target_model.get_initializer("weight") == 2)


def test_normal_hydration_rechecks_native_choices_under_current_source_without_fingerprint():
    model = _mvau_model()
    proposal = _proposal(model, simd=4)
    proposal.commit(model, Build())
    assert not any(
        item.name == OBSOLETE_FINGERPRINT_ATTRIBUTE for item in model.graph.node[0].attribute
    )
    model.set_initializer("weight", np.ones((8, 4), dtype=np.float32))
    restored = MvauDataflowOp(model.graph.node[0]).hydrate(model, Build())
    assert restored.recorded() == proposal.recorded()
    assert np.all(restored.source.operand("weight").initializer_value.array_copy() == 1)
    model.set_initializer("weight", np.ones((6, 4), dtype=np.float32))
    model.set_tensor_shape("activation", [2, 6])
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="cannot replay"):
        MvauDataflowOp(model.graph.node[0]).hydrate(model, Build())
    assert model.model.SerializeToString(deterministic=True) == before


def test_invalid_choice_domain_refuses_save_atomically():
    proposal = _proposal(simd=4)
    model = _mvau_model(matrix_width=6)
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises((DataflowOpError, RequestError, ValueError)):
        MvauDataflowOp(model.graph.node[0]).save_space(model, proposal, Build())
    assert model.model.SerializeToString(deterministic=True) == before


def test_family_and_schema_mismatch_refuse_without_writes():
    proposal = _proposal()
    replay = _replay_model()
    before = replay.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="family"):
        ActivationReplayOp(replay.graph.node[0]).save_space(replay, proposal)
    assert replay.model.SerializeToString(deterministic=True) == before

    class OtherSchema(MvauDataflowOp):
        schema_version = 999

    model = _mvau_model()
    other = OtherSchema(model.graph.node[0]).hydrate(model, Build())
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="schema_version"):
        MvauDataflowOp(model.graph.node[0]).save_space(model, other, Build())
    assert model.model.SerializeToString(deterministic=True) == before


def test_unresolved_saved_producer_clears_its_annotation_and_invalidates_downstream(monkeypatch):
    monkeypatch.setitem(custom_op, "AlternativeTypeOp", AlternativeTypeOp)
    model = _chain(AlternativeTypeOp)
    target = AlternativeTypeOp(model.graph.node[0])
    proposal = target.hydrate(model)
    assert isinstance(proposal.operand_type("result"), Unresolved)
    saved = target.save_space(model, proposal)
    assert isinstance(saved.operand_type("result"), Unresolved)
    for name in ("middle", "downstream0", "downstream1"):
        assert not any(
            annotation.tensor_name == name and entry.key == "finn_datatype"
            for annotation in model.graph.quantization_annotation
            for entry in annotation.quant_parameter_tensor_names
        )
        assert isinstance(producer_type(model, name), Unresolved)


def test_codec_mismatch_refuses_even_with_equal_family_and_schema():
    first_codec = AttributeCodec("test.choice", 1, list, tuple, "ints")
    changed_codec = AttributeCodec("test.changed.choice", 1, list, tuple, "ints")

    class Original(DataflowOp):
        family = "test.save-codec"
        choice = Decision(tuple, values=((1,),), canonical=first_codec)

    class Changed(Original):
        choice = Decision(tuple, values=((1,),), canonical=changed_codec)

    source_model = _codec_model(Original)
    proposal = Original(source_model.graph.node[0]).bind(source_model).assign(Original.choice, (1,))
    model = _codec_model(Changed)
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="codec"):
        Changed(model.graph.node[0]).save_space(model, proposal)
    assert model.model.SerializeToString(deterministic=True) == before


def test_precomputed_effects_still_refuse_stale_target_reads():
    model = _mvau_model()
    effects = _proposal(model).graph_effects()
    model.set_initializer("weight", np.ones((8, 4), dtype=np.float32))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="different problem|read"):
        apply_graph_effects(model, effects)
    assert model.model.SerializeToString(deterministic=True) == before
