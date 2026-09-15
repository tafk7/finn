# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from onnx import TensorProto, checker, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from dataflow.ops.test_conformance import (
    Build,
    _configure_replay,
    _mvau_model,
    _replay_model,
)
from dataflow.physical_fixture import source_model
from finn.dataflow._engine import Decided
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOpError
from finn.dataflow.ops.graph_context import (
    ContextRead,
    CurrentGraphContext,
    ExternalOperandOrigin,
    GraphInputEntry,
    GraphInputOrigin,
    IncomingGraphContext,
    LogicalBoundaryContract,
    _contract_for_mapping,
    capture_frozen_op_logical,
    checked_context_read,
    validate_frozen_op_logical,
)
from finn.dataflow.ops.inference import InferDataTypes, InferShapes
from finn.dataflow.ops.model_effects import (
    MODEL_READ_PRESENT,
    ModelReadExpectation,
    ModelReadKind,
    ModelReadSet,
    ObservationMutationError,
    merge_model_read_sets,
)
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.ops.reconstruction import bind_operations


def _prepared_replay() -> tuple[Any, Any, Build, LogicalBoundaryContract]:
    build = Build()
    model = _replay_model(repetitions=2, matrix_width=8, folds=4)
    model = model.transform(InferShapes()).transform(InferDataTypes())
    local = _configure_replay(bind_operations(model, build)[0])
    mappings = local.operand_mapping
    assert isinstance(mappings, Decided)
    activation = next(item for item in mappings.value if item.source_operand == "activation")
    contract = _contract_for_mapping(
        local,
        local.network.value,
        activation,
        output=False,
    ).contract
    return model, local, build, contract


def _graph_candidate() -> tuple[Any, Any, Build, CurrentGraphContext]:
    model, local, build, contract = _prepared_replay()
    context = CurrentGraphContext((GraphInputEntry("activation", "entry.activation", contract),))
    bound = bind_operations(model, build, graph_context=context)[0]
    candidate = _configure_replay(bound)
    assert candidate.local_problem_fingerprint == local.local_problem_fingerprint
    return model, candidate, build, context


def _make_initializer_overrideable(model: ModelWrapper) -> str:
    weight = model.graph.node[0].input[1]
    value_info = next(item for item in model.graph.value_info if item.name == weight)
    retained = [item for item in model.graph.value_info if item.name != weight]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    model.graph.input.append(value_info)
    checker.check_model(model.model)
    return weight


def test_graph_problem_changes_only_the_full_occurrence_identity() -> None:
    _model, candidate, _build, _context = _graph_candidate()
    local = candidate.reconstruct()
    values = dict(local.problem_snapshot)
    values.pop(type(candidate).incoming_graph_context)
    source_only = type(candidate)._start_frozen(values, candidate._bound_node())

    assert candidate.problem_fingerprint != source_only.problem_fingerprint
    assert candidate.local_problem_fingerprint == source_only.local_problem_fingerprint


def test_current_context_distinguishes_fixed_initializer_and_true_graph_input() -> None:
    model, build, context = source_model()
    node = model.graph.node[0]
    answer = context.read_inputs(
        model,
        build,
        consumer_scope_id=context.external_operands[0].consumer_scope_id,
    )

    assert isinstance(answer, Decided)
    by_operand = {item.consumer_input.operand_id: item for item in answer.value.incoming.bindings}
    assert isinstance(by_operand["activation"].origin, GraphInputOrigin)
    assert isinstance(by_operand["weight"].origin, ExternalOperandOrigin)
    expectations = answer.value.model_reads.expectations
    assert (
        ModelReadExpectation(
            ModelReadKind.GRAPH_INPUT,
            node.input[1],
            None,
            None,
        )
        in expectations
    )
    assert (
        ModelReadExpectation(
            ModelReadKind.GRAPH_INPUT,
            node.input[0],
            None,
            next(
                item for item in model.graph.input if item.name == node.input[0]
            ).SerializeToString(deterministic=True),
        )
        in expectations
    )


def test_initial_bind_refuses_overrideable_initializer_without_writing() -> None:
    model, build, context = source_model()
    _make_initializer_overrideable(model)
    before = model.model.SerializeToString(deterministic=True)

    with pytest.raises(DataflowOpError, match="graph-context-initializer-overrideable"):
        MvauDataflowOp(model.graph.node[0]).bind(
            model,
            build,
            graph_context=context,
        )

    assert model.model.SerializeToString(deterministic=True) == before


def test_local_problem_fingerprint_keeps_exact_replay_and_mvau_vectors() -> None:
    expected = {
        "ActivationReplayOp": "97bd329c6cc31bd10a7e2616962d136ec9628809ac1d2d4b044a0ab4e513f11e",
        "MvauDataflowOp": "aaa3b8846a09383cb254cbcf84ea4c0ce8b599079688f8868bb9ab45d610974a",
    }
    build = Build()
    for make_model in (_replay_model, _mvau_model):
        model = make_model().transform(InferShapes()).transform(InferDataTypes())
        operation = bind_operations(model, build)[0]
        assert operation.local_problem_fingerprint == expected[type(operation).__name__]


def test_graph_capture_strong_commit_and_rebound_recapture() -> None:
    model, candidate, build, context = _graph_candidate()
    before = model.model.SerializeToString(deterministic=True)
    capture = capture_frozen_op_logical(candidate)

    assert (
        validate_frozen_op_logical(
            candidate,
            capture,
            model=model,
            build=build,
            graph_context=context,
        )
        == ()
    )
    rebound = candidate.commit(
        model,
        build,
        require_graph=True,
        graph_context=context,
    )
    assert model.model.SerializeToString(deterministic=True) != before
    assert isinstance(rebound.graph_dataflow.accepted_answer, Decided)
    assert dict(rebound.recorded()) == {"design.pe": 1, "design.simd": 4}

    fresh = capture_frozen_op_logical(rebound)
    assert (
        validate_frozen_op_logical(
            rebound,
            fresh,
            model=model,
            build=build,
            graph_context=context,
        )
        == ()
    )
    assert validate_frozen_op_logical(
        rebound,
        capture,
        model=model,
        build=build,
        graph_context=context,
    )


def test_graph_commit_can_publish_derived_output_while_repairing_its_annotation() -> None:
    model, _local, build, contract = _prepared_replay()
    model.set_tensor_shape("expanded", [2, 99])
    context = CurrentGraphContext((GraphInputEntry("activation", "entry.activation", contract),))
    candidate = _configure_replay(bind_operations(model, build, graph_context=context)[0])

    outgoing = candidate.outgoing_logical_contracts
    assert isinstance(outgoing, Decided)
    assert outgoing.value[0].contract.source_shape == (8, 8)
    rebound = candidate.commit(
        model,
        build,
        require_graph=True,
        graph_context=context,
    )
    assert model.get_tensor_shape("expanded") == [8, 8]
    assert isinstance(rebound.graph_dataflow.accepted_answer, Decided)


def test_strong_commit_refuses_changed_complete_context_without_writing() -> None:
    model, candidate, build, context = _graph_candidate()
    before = model.model.SerializeToString(deterministic=True)
    original = context.graph_inputs[0].contract
    changed = replace(
        original,
        beat_sequence=BeatSequence(
            8,
            (
                tuple((0, index) for index in range(8)),
                tuple((1, index) for index in range(8)),
            ),
        ),
    )
    changed_context = CurrentGraphContext(
        (GraphInputEntry("activation", "entry.activation", changed),)
    )

    with pytest.raises(DataflowOpError, match="current incoming graph contracts differ"):
        candidate.commit(
            model,
            build,
            require_graph=True,
            graph_context=changed_context,
        )
    assert model.model.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("mutate_live", (False, True))
def test_provider_mutation_is_rejected_and_live_bytes_are_restored(mutate_live: bool) -> None:
    model, candidate, build, _context = _graph_candidate()
    before = model.model.SerializeToString(deterministic=True)

    class MutatingContext:
        def read_inputs(
            self,
            detached: Any,
            _build: object,
            *,
            consumer_scope_id: str,
        ) -> Any:
            del consumer_scope_id
            target = model if mutate_live else detached
            target.graph.name = "provider-write"
            return Decided(ContextRead(IncomingGraphContext(()), ModelReadSet()))

    with pytest.raises(ObservationMutationError, match="mutated"):
        checked_context_read(
            MutatingContext(),
            model,
            build,
            consumer_scope_id=candidate.recorded_scope_id(),
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_provider_mutation_while_raising_is_the_reported_failure() -> None:
    model, candidate, build, _context = _graph_candidate()
    before = model.model.SerializeToString(deterministic=True)

    class MutatingContext:
        def read_inputs(self, detached: Any, _build: object, **_kwargs: object) -> Any:
            detached.graph.name = "provider-write"
            raise RuntimeError("provider failed")

    with pytest.raises(ObservationMutationError, match="while raising"):
        checked_context_read(
            MutatingContext(),
            model,
            build,
            consumer_scope_id=candidate.recorded_scope_id(),
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_failed_return_hydration_rolls_back_a_strong_commit() -> None:
    model, candidate, build, context = _graph_candidate()
    before = model.model.SerializeToString(deterministic=True)

    class FailsDuringFinish:
        calls = 0

        def read_inputs(
            self,
            detached: Any,
            offered_build: object,
            *,
            consumer_scope_id: str,
        ) -> Any:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("finish read failed")
            return context.read_inputs(
                detached,
                offered_build,
                consumer_scope_id=consumer_scope_id,
            )

    provider = FailsDuringFinish()
    with pytest.raises(RuntimeError, match="finish read failed"):
        candidate.commit(
            model,
            build,
            require_graph=True,
            graph_context=provider,
        )
    assert provider.calls == 2
    assert model.model.SerializeToString(deterministic=True) == before


def test_read_merge_is_deterministic_and_never_last_wins() -> None:
    present = ModelReadExpectation(
        ModelReadKind.OPERAND_SLOT,
        "consumer",
        "input:0",
        MODEL_READ_PRESENT,
    )
    exact = replace(present, expected="X")
    metadata = ModelReadExpectation(ModelReadKind.METADATA, "key", None, "value")

    left = merge_model_read_sets(ModelReadSet((present, metadata)), ModelReadSet((exact,)))
    right = merge_model_read_sets(ModelReadSet((exact,)), ModelReadSet((metadata, present)))
    assert left == right
    assert exact in left.expectations
    assert present not in left.expectations

    with pytest.raises(DataflowOpError, match="contradictory model reads"):
        merge_model_read_sets(ModelReadSet((exact,)), ModelReadSet((replace(exact, expected="Y"),)))
    with pytest.raises(DataflowOpError, match="contradictory model reads"):
        merge_model_read_sets(
            ModelReadSet((present,)),
            ModelReadSet((replace(present, expected=None),)),
        )


def _replay_chain() -> tuple[ModelWrapper, Build]:
    nodes = [
        helper.make_node(
            "ActivationReplayOp",
            ["X"],
            ["M"],
            domain=DATAFLOW_DOMAIN,
            name="producer",
            neuron_folds=1,
        ),
        helper.make_node(
            "ActivationReplayOp",
            ["M"],
            ["Y"],
            domain=DATAFLOW_DOMAIN,
            name="consumer",
            neuron_folds=1,
        ),
    ]

    def tensor(name: str) -> Any:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4])

    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(nodes, "replay-chain", [tensor("X")], [tensor("Y")]),
            opset_imports=(
                helper.make_opsetid("", 13),
                helper.make_opsetid(DATAFLOW_DOMAIN, 1),
            ),
        )
    )
    model.set_tensor_shape("M", [1, 4])
    for name in ("X", "M", "Y"):
        model.set_tensor_datatype(name, DataType["INT8"])
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model, Build()


def test_current_provider_proves_transitive_freshness_without_cached_annotations() -> None:
    model, build = _replay_chain()
    producer, consumer = bind_operations(model, build)
    producer = _configure_replay(producer)
    producer = producer.commit(model, build)
    consumer = _configure_replay(consumer.rebind(model, build))
    consumer.commit(model, build)

    mappings = producer.operand_mapping
    assert isinstance(mappings, Decided)
    activation = next(item for item in mappings.value if item.source_operand == "activation")
    entry_contract = _contract_for_mapping(
        producer,
        producer.network.value,
        activation,
        output=False,
    ).contract
    context = CurrentGraphContext((GraphInputEntry("X", "entry.X", entry_contract),))
    current_consumer = consumer.rebind(model, build, graph_context=context)
    assert isinstance(current_consumer.graph_dataflow.accepted_answer, Decided)

    incompatible = replace(
        entry_contract,
        beat_sequence=BeatSequence(2, (((0, 0), (0, 1)), ((0, 2), (0, 3)))),
    )
    changed = CurrentGraphContext((GraphInputEntry("X", "entry.X", incompatible),))
    answer = changed.read_inputs(
        model,
        build,
        consumer_scope_id=consumer.recorded_scope_id(),
    )
    assert not isinstance(answer, Decided)
