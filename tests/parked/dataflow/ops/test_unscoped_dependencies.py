# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only producer addresses let a consumer save before upstream nodes do."""

from copy import deepcopy
import json

import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType

from parked.dataflow.ops.test_graph_context import _replay_chain
from parked.dataflow.ops.test_partial_type_graph import (
    DecisionTypeOp,
    DecisionTypeSpace,
    AlternativeTypeOp,
    AlternativeTypeSpace,
    _PartialSpace,
    _PrecisionInterface,
    _chain,
)
from finn.parked.custom_op.dataflow import custom_op
from finn.kernels._engine import Absent, Decided, Unresolved
from finn.parked.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.parked.dataflow.ops.binding import ChoiceBinding
from finn.parked.dataflow.ops.model_effects import ModelReadKind, validate_model_read_set
from finn.parked.dataflow.ops.native import SCHEMA_VERSION_ATTRIBUTE, SCOPE_ID_ATTRIBUTE
from finn.parked.dataflow.ops.persistence import apply_graph_effects
from finn.parked.dataflow.ops.type_context import producer_type_facts
from finn.kernels.space.declarations import (
    ConstraintGroup,
    Decision,
    Projection,
    Readiness,
    Subspace,
    SubspaceChoice,
    constraint,
    derived,
)
from finn.parked.dataflow.model.logical.interface_authoring import PublicOperandDeclaration
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


class ReadChoicesSpace(DecisionTypeSpace):
    required_support = Decision(bool, values=(False, True))
    diagnostic = Decision(bool, values=(False, True))

    @constraint(supported=required_support)
    def source_type_supported(*, supported):
        return supported

    type_source_accepts = ConstraintGroup(source_type_supported)
    choice_bindings = (
        *DecisionTypeSpace.choice_bindings,
        ChoiceBinding("required_support", (), "required_support"),
        ChoiceBinding("diagnostic", (), "diagnostic"),
    )


class ReadChoicesOp(DataflowOp):
    space_type = ReadChoicesSpace


class WhenTypeSpace(_PartialSpace):
    enabled = Decision(bool, values=(False, True))
    kernel = Subspace(
        _PrecisionInterface,
        when=enabled,
        activation_type=_PartialSpace.activation.datatype,
        shape=_PartialSpace.activation.shape,
    )
    choice_bindings = (
        ChoiceBinding("enabled", (), "enabled"),
        ChoiceBinding("precision", ("kernel",), "precision"),
    )


class WhenTypeOp(DataflowOp):
    space_type = WhenTypeSpace


class BranchWhenTypeSpace(_PartialSpace):
    enabled = Decision(bool, values=(False, True))
    kernel = SubspaceChoice(
        {
            "a": Subspace(
                _PrecisionInterface,
                activation_type=_PartialSpace.activation.datatype,
                shape=_PartialSpace.activation.shape,
            ),
            "b": Subspace(
                _PrecisionInterface,
                activation_type=_PartialSpace.activation.datatype,
                shape=_PartialSpace.activation.shape,
            ),
        },
        when=enabled,
    )
    choice_bindings = (
        ChoiceBinding("enabled", (), "enabled"),
        ChoiceBinding("implementation", ("kernel",), "case"),
        ChoiceBinding("precision_a", ("kernel", "a"), "precision"),
        ChoiceBinding("precision_b", ("kernel", "b"), "precision"),
    )


class BranchWhenTypeOp(DataflowOp):
    space_type = BranchWhenTypeSpace


class SameTypeAlternativeSpace(AlternativeTypeSpace):
    @derived(QONNX_DATATYPE_VALUE_SEMANTICS)
    def narrow():
        return DataType["INT16"]


class SameTypeAlternativeOp(DataflowOp):
    space_type = SameTypeAlternativeSpace


class _FacetInterface(_PrecisionInterface):
    ready = Decision(bool, values=(False, True))
    applies = Decision(bool, values=(False, True))
    accepted = Decision(bool, values=(False, True))

    @constraint(flag=accepted)
    def supported(*, flag):
        return flag

    type_support = ConstraintGroup(supported)
    type_ready = Readiness(properties=(_PrecisionInterface.result_type,), decisions=(ready,))
    public_result_type = Projection(
        _PrecisionInterface.result_type,
        readiness=type_ready,
        applicable_if=applies,
        constraints=type_support,
    )
    public_operands = (
        _PrecisionInterface.public_operands[0],
        PublicOperandDeclaration(
            "result", "output", public_result_type, _PrecisionInterface.public_operands[1].domain
        ),
    )


class FacetTypeSpace(_PartialSpace):
    kernel = Subspace(
        _FacetInterface,
        activation_type=_PartialSpace.activation.datatype,
        shape=_PartialSpace.activation.shape,
    )
    choice_bindings = tuple(
        ChoiceBinding(name, ("kernel",), name)
        for name in ("precision", "ready", "applies", "accepted")
    )


class FacetTypeOp(DataflowOp):
    space_type = FacetTypeSpace


def _read_fields(reads):
    return {
        json.loads(item.field)[1] if item.kind is ModelReadKind.NODE_BY_OUTPUT else item.field
        for item in reads.expectations
    }


def _set_native(node, name, value):
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    if value is not None:
        node.attribute.append(helper.make_attribute(name, value))


def _type_model(monkeypatch, operation_type, scoped, attributes):
    monkeypatch.setitem(custom_op, operation_type.__name__, operation_type)
    model = _chain(operation_type)
    if not scoped:
        _remove_ids(model)
    node = model.graph.node[0]
    _set_native(node, SCHEMA_VERSION_ATTRIBUTE, operation_type.space_type.schema_version)
    for name, value in attributes.items():
        _set_native(node, name, value)
    return model


@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize(
    "name,value",
    (("precision", 32), ("precision", None), ("required_support", 0), ("required_support", None)),
)
def test_direct_producer_read_set_tracks_type_choices_and_source_obligations(
    monkeypatch,
    scoped,
    name,
    value,
):
    model = _type_model(
        monkeypatch,
        ReadChoicesOp,
        scoped,
        {"precision": 16, "required_support": 1, "diagnostic": 0},
    )
    consumer = model.get_customop_wrapper(model.graph.node[-1])
    assert consumer.space.operand_type("result") == Decided(DataType["INT16"])
    reads = consumer.space.source.inputs[0].producer_reads
    validate_model_read_set(model, reads)
    _set_native(model.graph.node[0], name, value)
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError):
        validate_model_read_set(model, reads)
    assert model.model.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("scoped", (False, True))
def test_direct_type_read_set_excludes_unrelated_native_choice(monkeypatch, scoped):
    model = _type_model(
        monkeypatch,
        ReadChoicesOp,
        scoped,
        {"precision": 16, "required_support": 1, "diagnostic": 0},
    )
    answer, reads = producer_type_facts(model, "middle")
    assert answer == Decided(DataType["INT16"])
    assert "diagnostic" not in _read_fields(reads)
    _set_native(model.graph.node[0], "diagnostic", 1)
    validate_model_read_set(model, reads)


@pytest.mark.parametrize("scoped", (False, True))
def test_missing_type_choice_is_a_recorded_absence(monkeypatch, scoped):
    model = _type_model(monkeypatch, DecisionTypeOp, scoped, {})
    answer, reads = producer_type_facts(model, "middle")
    assert isinstance(answer, Unresolved)
    _set_native(model.graph.node[0], "precision", 16)
    with pytest.raises(DataflowOpError):
        validate_model_read_set(model, reads)


@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("initial,next_value", ((None, "wide"), ("wide", "narrow"), ("wide", None)))
def test_type_route_selector_read_is_recorded_before_and_after_selection(
    monkeypatch,
    scoped,
    initial,
    next_value,
):
    model = _type_model(monkeypatch, AlternativeTypeOp, scoped, {"implementation": initial})
    answer, reads = producer_type_facts(model, "middle")
    assert isinstance(answer, Unresolved if initial is None else Decided)
    _set_native(model.graph.node[0], "implementation", next_value)
    with pytest.raises(DataflowOpError):
        validate_model_read_set(model, reads)


@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("enabled", (None, 0))
def test_unresolved_and_inactive_when_routes_capture_gate_without_child_choices(
    monkeypatch,
    scoped,
    enabled,
):
    model = _type_model(monkeypatch, WhenTypeOp, scoped, {"enabled": enabled})
    answer, reads = producer_type_facts(model, "middle")
    assert isinstance(answer, Unresolved if enabled is None else Absent)
    assert "precision" not in _read_fields(reads)
    _set_native(model.graph.node[0], "enabled", 1)
    with pytest.raises(DataflowOpError):
        validate_model_read_set(model, reads)


@pytest.mark.parametrize("enabled", (None, 0))
def test_branch_when_route_stops_before_selector_and_inactive_alternatives(monkeypatch, enabled):
    model = _type_model(monkeypatch, BranchWhenTypeOp, False, {"enabled": enabled})
    answer, reads = producer_type_facts(model, "middle")
    assert isinstance(answer, Unresolved if enabled is None else Absent)
    assert not (_read_fields(reads) & {"implementation", "precision_a", "precision_b"})
    _set_native(model.graph.node[0], "enabled", 1)
    with pytest.raises(DataflowOpError):
        validate_model_read_set(model, reads)


def test_selector_is_a_read_even_when_alternatives_have_equal_types(monkeypatch):
    model = _type_model(monkeypatch, SameTypeAlternativeOp, True, {"implementation": "wide"})
    before, reads = producer_type_facts(model, "middle")
    _set_native(model.graph.node[0], "implementation", "narrow")
    after, _ = producer_type_facts(model, "middle")
    assert before == after == Decided(DataType["INT16"])
    with pytest.raises(DataflowOpError):
        validate_model_read_set(model, reads)


@pytest.mark.parametrize("name", ("ready", "applies", "accepted"))
@pytest.mark.parametrize("value", (0, None))
def test_facet_readiness_applicability_and_constraint_choices_are_reads(monkeypatch, name, value):
    model = _type_model(
        monkeypatch, FacetTypeOp, False, {"precision": 16, "ready": 1, "applies": 1, "accepted": 1}
    )
    answer, reads = producer_type_facts(model, "middle")
    assert answer == Decided(DataType["INT16"])
    _set_native(model.graph.node[0], name, value)
    with pytest.raises(DataflowOpError):
        validate_model_read_set(model, reads)


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
