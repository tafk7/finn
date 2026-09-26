# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Proposed choices are checked against explicit current target facts on save."""

import numpy as np
import pytest
from onnx import helper
from qonnx.core.datatype import DataType

from parked.dataflow.ops.test_dataflow_op import Build, _configure_mvau_point, _mvau_model, _replay_model
from parked.dataflow.ops.test_partial_type_graph import AlternativeTypeOp, _chain
from parked.dataflow.ops.test_persistence_codecs import _model as _codec_model, _op as _codec_op
from parked.dataflow.ops.factory import make_op, make_space
from finn.parked.dataflow.ops.space import DataflowSpace
from finn.parked.custom_op.dataflow import custom_op
from finn.kernels._engine import Decided, RequestError, Unresolved
from finn.parked.dataflow.ops.base import DataflowOpError
from finn.parked.dataflow.ops.mvau.op import MvauSpace
from finn.parked.dataflow.ops.native import (
    AttributeCodec,
    OBSOLETE_FINGERPRINT_ATTRIBUTE,
    SCHEMA_VERSION_ATTRIBUTE,
    serialize_choices,
)
from finn.parked.dataflow.ops.persistence import apply_graph_effects
from finn.parked.dataflow.ops.type_context import producer_type
from finn.kernels.space.declarations import Decision
from finn.parked.dataflow.ops import reconstruction


def _proposal(model=None, *, simd=2):
    model = _mvau_model() if model is None else model
    return _configure_mvau_point(make_space(model, build=Build()), simd=simd)


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
    target = make_op(model, build=Build())
    saved = target.save_space(proposal)
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
    assert make_space(model, build=Build()).recorded() == saved.recorded()


def test_compatible_choices_save_to_another_source_origin():
    proposal = _proposal()
    target_model = _mvau_model()
    target_model.set_initializer("weight", np.full((8, 4), 2, dtype=np.float32))
    target = make_op(target_model, build=Build())
    assert target.recorded_scope_id() != proposal.recorded_scope_id()
    saved = target.save_space(proposal)
    assert saved.recorded_scope_id() == target.recorded_scope_id()
    assert saved.recorded() == proposal.recorded()
    assert np.all(target_model.get_initializer("weight") == 2)


def test_normal_hydration_rechecks_native_choices_under_current_source_without_fingerprint():
    model = _mvau_model()
    proposal = _proposal(model, simd=4)
    make_op(model, build=Build()).save_space(proposal)
    assert not any(
        item.name == OBSOLETE_FINGERPRINT_ATTRIBUTE for item in model.graph.node[0].attribute
    )
    model.set_initializer("weight", np.ones((8, 4), dtype=np.float32))
    restored = make_space(model, build=Build())
    assert restored.recorded() == proposal.recorded()
    assert np.all(restored.source.operand("weight").initializer_value.array_copy() == 1)
    model.set_initializer("weight", np.ones((6, 4), dtype=np.float32))
    model.set_tensor_shape("activation", [2, 6])
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="cannot replay"):
        make_space(model, build=Build())
    assert model.model.SerializeToString(deterministic=True) == before


def test_invalid_choice_domain_refuses_save_atomically():
    proposal = _proposal(simd=4)
    model = _mvau_model(matrix_width=6)
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises((DataflowOpError, RequestError, ValueError)):
        make_op(model, build=Build()).save_space(proposal)
    assert model.model.SerializeToString(deterministic=True) == before


def test_family_and_schema_mismatch_refuse_without_writes():
    proposal = _proposal()
    replay = _replay_model()
    before = replay.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="family"):
        make_op(replay).save_space(proposal)
    assert replay.model.SerializeToString(deterministic=True) == before

    class OtherSchema(MvauSpace):
        schema_version = 999

    model = _mvau_model()
    other = make_space(model, space_type=OtherSchema, build=Build())
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="schema_version"):
        make_op(model, build=Build()).save_space(other)
    assert model.model.SerializeToString(deterministic=True) == before


def test_unresolved_saved_producer_clears_its_annotation_and_invalidates_downstream(monkeypatch):
    monkeypatch.setitem(custom_op, "AlternativeTypeOp", AlternativeTypeOp)
    model = _chain(AlternativeTypeOp)
    target = make_op(model)
    proposal = target.space
    assert isinstance(proposal.operand_type("result"), Unresolved)
    saved = target.save_space(proposal)
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

    class Original(DataflowSpace):
        family = "test.save-codec"
        choice = Decision(tuple, values=((1,),), canonical=first_codec)

    class Changed(Original):
        choice = Decision(tuple, values=((1,),), canonical=changed_codec)

    source_model = _codec_model(Original)
    proposal = _codec_op(source_model, Original).space.assign(Original.choice, (1,))
    model = _codec_model(Changed)
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="codec"):
        _codec_op(model, Changed).save_space(proposal)
    assert model.model.SerializeToString(deterministic=True) == before


def test_precomputed_effects_still_refuse_stale_target_reads():
    model = _mvau_model()
    effects = _proposal(model).graph_effects()
    model.set_initializer("weight", np.ones((8, 4), dtype=np.float32))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="different problem|read"):
        apply_graph_effects(model, effects)
    assert model.model.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize(
    "attributes",
    (
        ((SCHEMA_VERSION_ATTRIBUTE, 999),),
        ((SCHEMA_VERSION_ATTRIBUTE, "8"),),
        (("kernel__case", "dot_product"),),
        ((SCHEMA_VERSION_ATTRIBUTE, 8), ("kernel__dot_product__simd", "2")),
        ((SCHEMA_VERSION_ATTRIBUTE, 8), (SCHEMA_VERSION_ATTRIBUTE, 8)),
    ),
)
def test_save_refuses_unsupported_or_malformed_current_native_encoding(attributes, monkeypatch):
    proposal = _proposal()
    model = _mvau_model()
    target = make_op(model, build=Build())
    model.graph.node[0].attribute.extend(
        helper.make_attribute(name, value) for name, value in attributes
    )
    before = model.model.SerializeToString(deterministic=True)

    def unexpected_source_hydration(*args, **kwargs):
        raise AssertionError("target encoding must be checked before source hydration")

    monkeypatch.setattr(reconstruction, "build_space", unexpected_source_hydration)
    with pytest.raises(DataflowOpError, match="schema version|cannot decode|duplicate"):
        target.save_space(proposal)
    assert model.model.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize(
    "field,value", (("op_type", "ActivationReplayOp"), ("domain", "wrong.domain"))
)
def test_save_refuses_changed_current_operator_identity_atomically(field, value):
    model = _mvau_model()
    proposal = _proposal()
    target = make_op(model, build=Build())
    setattr(model.graph.node[0], field, value)
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="operator identity"):
        target.save_space(proposal)
    assert model.model.SerializeToString(deterministic=True) == before


def test_save_can_replace_well_encoded_choices_invalid_under_changed_current_domains():
    model = _mvau_model()
    target = make_op(model, build=Build())
    target.save_space(_proposal(model, simd=4))
    model.set_initializer("weight", np.ones((6, 4), dtype=np.float32))
    model.set_tensor_shape("activation", [2, 6])
    saved = target.save_space(_proposal(simd=2))
    assert serialize_choices(saved)["kernel__dot_product__simd"].value == 2
    assert saved.source.operand("weight").shape == (6, 4)
