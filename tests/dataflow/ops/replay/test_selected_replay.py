from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from dataflow.ops.test_dataflow_op import _configured_replay, _replay_model
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

from finn.dataflow._engine import Absent, Decided, Finding, FindingKind, QualifiedPath, Unresolved
from finn.dataflow.designs.design import SelectedGraph
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.model.maps import CoordinateSet, RectangularDomain
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.ops import native
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.replay.selected import (
    ACTIVATION_KEY,
    EXPANDED_KEY,
    REPLAY_CONSTRUCTION_FAMILY,
    REPLAY_CONSTRUCTION_VERSION,
    REPLAY_SELECTED_CONSTRUCTION,
    ReplaySourceSemantics,
    construct_replay_snapshot,
    derive_replay_facts,
    encode_replay_source_semantics,
)
from finn.dataflow.ops.selected import (
    ConstructionIdentity,
    ConstructionInputs,
    ConstructionRegistry,
    RecordedChoice,
    RelationKind,
    SelectedGraphError,
    SourceProvenance,
    SourceOrigin,
    SourceValueRef,
    build_selected_snapshot,
    construct_selected_graph,
    decode_selected_graph,
)
from finn.dataflow.ops.selected_registry import DEFAULT_SELECTED_CONSTRUCTIONS
from finn.dataflow.space.occurrence import ProjectionAssessment
from finn.dataflow.ops.source import SourceError


def _facts(*, source_shape=(2, 6), folds=3, simd=2):
    rows = int(np.prod(source_shape[:-1]))
    source = SourceProvenance.create(
        family="finn.dataflow.activation_replay",
        family_version="2",
        schema_version=3,
        problem_fingerprint="problem",
        scope_id="scope",
        operands=(
            SourceValueRef(ACTIVATION_KEY, source_shape, TensorProto.FLOAT, "INT8", None),
            SourceValueRef(
                EXPANDED_KEY,
                (rows * folds, source_shape[-1]),
                TensorProto.FLOAT,
                "INT8",
                None,
            ),
        ),
        semantics=encode_replay_source_semantics(ReplaySourceSemantics(folds)),
    )
    return derive_replay_facts(
        ConstructionIdentity(
            REPLAY_CONSTRUCTION_FAMILY,
            REPLAY_CONSTRUCTION_VERSION,
            "canonical",
        ),
        source,
        ReplaySourceSemantics(folds),
        (RecordedChoice("design.pe", 1), RecordedChoice("design.simd", simd)),
    )


def test_selected_replay_computes_distinct_xr_and_decodes() -> None:
    facts = _facts()
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    decoded = decode_selected_graph(snapshot, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)

    activation = np.arange(12, dtype=np.float32).reshape(2, 6)
    (actual,) = ReferenceEvaluator(snapshot.model_copy().model).run(None, {"X": activation})
    assert np.array_equal(actual, np.repeat(activation, 3, axis=0))

    replay = decoded.network.node("replay").region
    assert replay.input("X").operand.shape == (2, 6)
    assert replay.output_interface("activation_out").port.operand.id == "XR"
    assert replay.output_interface("activation_out").port.operand.shape == (6, 6)
    assert replay.output_interface("activation_out").port.beat_sequence.position_at(4, 1) == (
        1,
        3,
    )


@pytest.mark.parametrize("folds", (1, 3))
@pytest.mark.parametrize("simd", (1, 2, 3, 6))
def test_selected_replay_accepts_every_required_fold_and_simd_case(folds, simd) -> None:
    facts = _facts(folds=folds, simd=simd)
    decoded = decode_selected_graph(construct_replay_snapshot(facts, ConstructionInputs()))
    replay = decoded.network.node("replay").region
    assert replay.output_interface("activation_out").port.beat_sequence.beat_count == (
        2 * folds * (6 // simd)
    )


def test_selected_replay_flattens_leading_source_dimensions() -> None:
    facts = _facts(source_shape=(2, 2, 3), folds=2, simd=3)
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    decoded = decode_selected_graph(snapshot, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
    activation = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    (actual,) = ReferenceEvaluator(snapshot.model_copy().model).run(None, {"X_source": activation})
    assert np.array_equal(actual, np.repeat(activation.reshape(4, 3), 2, axis=0))
    assert decoded.network.node("replay").region.input("X").operand.shape == (4, 3)
    activation_binding = next(
        item for item in decoded.declaration.source_bindings if item.source == ACTIVATION_KEY
    )
    assert activation_binding.graph_value == "X"
    assert activation_binding.relation.kind is RelationKind.ROW_MAJOR_RESHAPE


def test_bound_replay_routes_through_the_design_construction_hook() -> None:
    _model, operation = _configured_replay(simd=2)
    answer = operation.selected_snapshot
    assert isinstance(answer, Decided)
    decoded = decode_selected_graph(answer.value, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
    assert decoded.network == operation.network.value


def test_selected_graph_is_a_projection_and_rejects_a_constructor_source_swap() -> None:
    _model, operation = _configured_replay(simd=2)
    declaration = ActivationReplayDesign.selected_graph
    assert declaration is not None
    construction = declaration.construction
    original = construction.construct

    def wrong_source(facts, inputs):
        changed = replace(
            facts,
            source=replace(
                facts.source,
                origin=replace(facts.source.origin, problem_fingerprint="wrong"),
            ),
        )
        return original(changed, inputs)

    with patch.object(
        ActivationReplayDesign,
        "selected_graph",
        SelectedGraph(replace(construction, construct=wrong_source)),
    ):
        assessment = operation.selected_graph
    assert isinstance(assessment, ProjectionAssessment)
    assert isinstance(assessment.accepted_answer, Absent)
    assert "expected_mismatch" in assessment.accepted_answer.findings[0].message


def test_selected_only_unresolved_dependency_blocks_projection_readiness() -> None:
    _model, operation = _configured_replay(simd=2)

    original = native.occurrence_answer_at

    def unresolved(root, reference):
        if str(reference.path).endswith("design.simd"):
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "selected-test-unresolved",
                        QualifiedPath("selected.test"),
                        "selected-only dependency is unresolved",
                    ),
                )
            )
        return original(root, reference)

    with patch.object(native, "occurrence_answer_at", unresolved):
        assessment = operation.selected_graph
    assert assessment.readiness.ready is None
    assert isinstance(assessment.accepted_answer, Unresolved)


def test_detached_construction_seam_cross_checks_its_expected_source() -> None:
    facts = _facts()
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    network = decode_selected_graph(snapshot, constructions=DEFAULT_SELECTED_CONSTRUCTIONS).network
    with pytest.raises(SelectedGraphError) as error:
        construct_selected_graph(
            REPLAY_SELECTED_CONSTRUCTION,
            facts,
            ConstructionInputs(),
            expected_network=network,
            expected_source=replace(
                facts.source,
                origin=replace(facts.source.origin, problem_fingerprint="wrong"),
            ),
        )
    assert error.value.code == "selected.source.fact_mismatch"


def test_detached_construction_seam_cross_checks_decoded_facts() -> None:
    facts = _facts()
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    network = decode_selected_graph(snapshot, constructions=DEFAULT_SELECTED_CONSTRUCTIONS).network

    def changed_facts(identity, source, semantics, choices):
        derived = derive_replay_facts(identity, source, semantics, choices)
        return replace(derived, parameters=replace(derived.parameters, simd=1))

    changed = replace(
        REPLAY_SELECTED_CONSTRUCTION,
        derive_facts=changed_facts,
        verify=lambda _snapshot, _facts: (),
    )
    with pytest.raises(SelectedGraphError) as error:
        construct_selected_graph(
            changed,
            facts,
            ConstructionInputs(),
            expected_network=network,
            expected_source=facts.source,
        )
    assert error.value.code == "selected.projection.network_mismatch"


def test_selected_replay_requires_nominal_pe() -> None:
    facts = _facts()
    with pytest.raises(ValueError, match="PE=1"):
        derive_replay_facts(
            facts.construction,
            facts.source,
            facts.source_semantics,
            (RecordedChoice("design.pe", 2), facts.choices[1]),
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda choices: choices[:-1],
        lambda choices: tuple(reversed(choices)),
        lambda choices: (*choices, RecordedChoice("unexpected", 1)),
        lambda choices: (replace(choices[0], value=True, encoding=True), choices[1]),
    ),
)
def test_selected_replay_choice_subset_is_checked_generically(mutate) -> None:
    snapshot = construct_replay_snapshot(_facts(), ConstructionInputs())
    broken = build_selected_snapshot(
        snapshot.model_copy(),
        replace(snapshot.declaration, choices=mutate(snapshot.declaration.choices)),
    )
    with pytest.raises(SelectedGraphError) as error:
        decode_selected_graph(broken)
    assert error.value.code in {
        "selected.construction.choice_subset",
        "selected.construction.choice_value",
    }


def test_native_schema_origin_does_not_change_replay_semantic_identity() -> None:
    facts = _facts()
    changed_source = replace(facts.source, origin=SourceOrigin(99, "problem", "scope"))
    changed = derive_replay_facts(
        facts.construction,
        changed_source,
        facts.source_semantics,
        facts.choices,
    )
    assert changed.source.semantic_fingerprint == facts.source.semantic_fingerprint
    assert changed.selection_fingerprint == facts.selection_fingerprint
    decoded = decode_selected_graph(construct_replay_snapshot(changed, ConstructionInputs()))
    assert decoded.selection_facts.source.schema_version == 99


def test_selected_replay_refuses_missing_source_logical_annotation() -> None:
    model = _replay_model()
    del model.graph.quantization_annotation[:]
    _model, operation = _configured_replay(model, simd=2)
    answer = operation.selected_snapshot
    assert isinstance(answer, Absent)
    assert "explicit logical datatype" in answer.findings[0].message


def test_duplicate_source_logical_annotations_refuse_at_source_capture() -> None:
    model = _replay_model()
    original = model.graph.quantization_annotation[0]
    model.graph.quantization_annotation.add().CopyFrom(original)
    with pytest.raises(SourceError, match="duplicate logical datatype annotations"):
        _configured_replay(model, simd=2)


def test_initializer_backed_leading_activation_uses_checked_external_boundary() -> None:
    facts = _facts(source_shape=(2, 2, 3), folds=2, simd=3)
    activation = facts.source.operands[0]
    source = SourceProvenance.create(
        family=facts.source.family,
        family_version=facts.source.family_version,
        schema_version=facts.source.schema_version,
        problem_fingerprint=facts.source.problem_fingerprint,
        scope_id=facts.source.scope_id,
        operands=(
            replace(
                activation,
                initializer_content_digest="1" * 64,
            ),
            facts.source.operands[1],
        ),
        semantics=facts.source.semantics,
    )
    rebound = derive_replay_facts(
        facts.construction,
        source,
        facts.source_semantics,
        facts.choices,
    )
    snapshot = construct_replay_snapshot(rebound, ConstructionInputs())
    assert decode_selected_graph(snapshot).selection_facts == rebound


def test_large_selected_replay_never_enters_expansion(monkeypatch) -> None:
    facts = _facts(source_shape=(2, 3), folds=1_048_576, simd=3)

    def refuse(*_args, **_kwargs):
        raise AssertionError("compact selected Replay entered an expansion iterator")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    monkeypatch.setattr(BeatSequence, "iter_beats", refuse)

    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    decoded = decode_selected_graph(snapshot)
    replay = decoded.network.node("replay").region
    output = replay.output_interface("activation_out").port.beat_sequence
    assert output.beat_count == 2 * 1_048_576
    assert output.position_at(output.beat_count - 1, 2) == (2 * 1_048_576 - 1, 2)
    assert len(snapshot.model_bytes) < 16_384


def test_selected_replay_decodes_source_free_in_a_fresh_process(tmp_path) -> None:
    path = tmp_path / "selected-replay.onnx"
    path.write_bytes(construct_replay_snapshot(_facts(), ConstructionInputs()).model_bytes)
    script = f"""
from unittest.mock import patch
from finn.dataflow.ops.selected_registry import DEFAULT_SELECTED_CONSTRUCTIONS
from finn.dataflow.ops.selected import reconstruct_selected_graph

def forbidden(*args, **kwargs):
    raise AssertionError('detached decode touched source, occurrence, engine, or build state')

with patch('finn.dataflow.ops.source.read_source_node', forbidden), \\
     patch('finn.dataflow.space.occurrence.start_occurrence', forbidden), \\
     patch('finn.dataflow._engine.Engine.start', forbidden), \\
     patch('finn.dataflow.ops.base._build_value', forbidden):
    decoded = reconstruct_selected_graph(
        {str(path)!r}, constructions=DEFAULT_SELECTED_CONSTRUCTIONS
    )
assert (
    decoded.network.node('replay').region
    .output_interface('activation_out').port.operand.id == 'XR'
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_replay_verifier_rejects_a_fold_major_expansion_with_refreshed_hashes() -> None:
    facts = _facts()
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    model = snapshot.model_copy()
    model.set_initializer("axes_replay", np.asarray((0,), dtype=np.int64))
    model.set_initializer("shape_XRF", np.asarray((3, 2, 6), dtype=np.int64))
    broken = build_selected_snapshot(model, snapshot.declaration)

    neutral = replace(REPLAY_SELECTED_CONSTRUCTION, verify=lambda _snapshot, _facts: ())
    decode_selected_graph(
        broken,
        constructions=ConstructionRegistry(
            {(neutral.family, neutral.version): neutral},
            {
                (
                    neutral.family,
                    neutral.version,
                ): DEFAULT_SELECTED_CONSTRUCTIONS.resolve_choice_schema(facts.construction)
            },
        ),
    )
    with pytest.raises(SelectedGraphError) as error:
        decode_selected_graph(broken, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
    assert error.value.code == "selected.construction.verification"


def test_replay_verifier_accepts_consistent_node_and_value_renames() -> None:
    snapshot = construct_replay_snapshot(_facts(), ConstructionInputs())
    model = snapshot.model_copy()
    model.rename_tensor("X", "renamed_input")
    model.rename_tensor("XR", "renamed_output")
    for index, node in enumerate(model.graph.node):
        node.name = f"display_{index}"
    declaration = replace(
        snapshot.declaration,
        interface_bindings=tuple(
            replace(
                item,
                graph_value={"X": "renamed_input", "XR": "renamed_output"}.get(
                    item.graph_value, item.graph_value
                ),
            )
            for item in snapshot.declaration.interface_bindings
        ),
        source_bindings=tuple(
            replace(
                item,
                graph_value={"X": "renamed_input", "XR": "renamed_output"}.get(
                    item.graph_value, item.graph_value
                ),
            )
            for item in snapshot.declaration.source_bindings
        ),
    )
    renamed = build_selected_snapshot(model, declaration)
    assert decode_selected_graph(renamed).network == decode_selected_graph(snapshot).network


def test_expanded_coordinates_preserve_the_previous_physical_transfer_values() -> None:
    facts = _facts()
    decoded = decode_selected_graph(construct_replay_snapshot(facts, ConstructionInputs()))
    sequence = (
        decoded.network.node("replay").region.output_interface("activation_out").port.beat_sequence
    )
    activation = np.arange(12).reshape(2, 6)
    expanded = np.repeat(activation, 3, axis=0)
    for beat, positions in enumerate(sequence.materialize_beats(max_fields=36)):
        old_positions = tuple((position[0] // 3, position[1]) for position in positions)
        assert tuple(expanded[position] for position in positions) == tuple(
            activation[position] for position in old_positions
        ), beat


def test_standalone_replay_versions_move_together() -> None:
    assert ActivationReplayOp.family_version == "2"
    assert ActivationReplayOp.schema_version == 3
    assert ActivationReplayDesign.version == "2"
    assert ReplayBufferKernel.version == "2"
    assert ReplayBufferKernel.region.version == "2"
