# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace

import pytest

from finn.dataflow.design import (
    Absent,
    Decided,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.definition import MVAUComputeBinding, MVAUComputeKernelPaths, MVAUDspBlock
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_batch_interleaved_streamed_mvau_region,
)
from finn.dataflow.network_validation import NetworkValidationReport
from finn.dataflow.ops.mvau import (
    CoordinateMappingKind,
    MVAU_DATAFLOW_OP_SPEC,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    MVAUSourceDescription,
    NetworkRef,
    RegionRef,
)
from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterBinding,
    CyclicParameterKernelPaths,
    CyclicParameterRegionDeclaration,
    CyclicRamStyle,
)
from finn.dataflow.region import NumericElementType

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


def _source_description(*, fused: bool = False) -> MVAUSourceDescription:
    return MVAUSourceDescription(
        "node0",
        "source_x",
        "source_w",
        "source_y",
        (2,),
        "source_thresholds" if fused else None,
        ("matmul", "threshold") if fused else ("matmul",),
    )


def _problem(
    *,
    repetitions: int = 2,
    matrix_width: int = 4,
    matrix_height: int = 4,
    source_description: MVAUSourceDescription | None = None,
    external_weight_sequence: object | None = None,
) -> dict[str, object]:
    problem: dict[str, object] = {
        str(MVAUComputeKernelPaths.REPETITIONS): repetitions,
        str(MVAUComputeKernelPaths.MATRIX_WIDTH): matrix_width,
        str(MVAUComputeKernelPaths.MATRIX_HEIGHT): matrix_height,
        str(MVAUComputeKernelPaths.ACTIVATION_ELEMENT_TYPE): INT8,
        str(MVAUComputeKernelPaths.WEIGHT_ELEMENT_TYPE): INT8,
        str(MVAUComputeKernelPaths.ACCUMULATOR_ELEMENT_TYPE): INT16,
        str(MVAUComputeKernelPaths.OUTPUT_ELEMENT_TYPE): INT16,
        str(MVAUComputeKernelPaths.COMPUTATION_PROFILE): (
            MVAUComputationProfile.ACCUMULATOR_INTEGER
        ),
        str(MVAUComputeKernelPaths.WEIGHT_INITIALIZER_AVAILABLE): True,
        str(MVAUComputeKernelPaths.TARGET_DSP_BLOCK): MVAUDspBlock.DSP58,
        str(MVAUComputeKernelPaths.WEIGHTS_NARROW): True,
        str(CyclicParameterKernelPaths.INITIALIZER_AVAILABLE): True,
        str(CyclicParameterKernelPaths.RUNTIME_WRITABLE): False,
        str(MVAUDataflowOpPaths.SOURCE_DESCRIPTION): source_description or _source_description(),
    }
    if external_weight_sequence is not None:
        problem[str(MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE)] = external_weight_sequence
    return problem


def _started(**overrides: object) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(MVAU_DATAFLOW_OP_SPEC)
    problem = _problem()
    override_paths = {
        "repetitions": MVAUComputeKernelPaths.REPETITIONS,
        "matrix_width": MVAUComputeKernelPaths.MATRIX_WIDTH,
        "matrix_height": MVAUComputeKernelPaths.MATRIX_HEIGHT,
        "source_description": MVAUDataflowOpPaths.SOURCE_DESCRIPTION,
        "external_weight_sequence": MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE,
    }
    for name, value in overrides.items():
        problem[str(override_paths[name])] = value
    return engine, engine.start(space, problem)


def _compute_assignments(
    declaration: MVAURegionDeclaration,
    topology: MVAUParameterTopology,
) -> dict[QualifiedPath, object]:
    assignments: dict[QualifiedPath, object] = {
        MVAUComputeKernelPaths.PE: 2,
        MVAUComputeKernelPaths.SIMD: 2,
        MVAUComputeKernelPaths.REGION_DECLARATION: declaration,
        MVAUDataflowOpPaths.PARAMETER_TOPOLOGY: topology,
    }
    if declaration is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED:
        assignments[MVAUComputeKernelPaths.INTERLEAVE] = 2
    return assignments


def _cyclic_assignments(
    declaration: CyclicParameterRegionDeclaration,
) -> dict[QualifiedPath, object]:
    return {
        CyclicParameterKernelPaths.REGION_DECLARATION: declaration,
        CyclicParameterKernelPaths.BINDING: CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        CyclicParameterKernelPaths.RAM_STYLE: CyclicRamStyle.BRAM,
        CyclicParameterKernelPaths.PUMPED_MEMORY: False,
    }


@pytest.mark.parametrize(
    "declaration,topology",
    [
        (MVAURegionDeclaration.STANDARD_EMBEDDED, MVAUParameterTopology.EMBEDDED),
        (MVAURegionDeclaration.STANDARD_STREAMED, MVAUParameterTopology.DIRECT),
    ],
)
def test_embedded_and_direct_topologies_resolve_to_region_refs(
    declaration: MVAURegionDeclaration, topology: MVAUParameterTopology
) -> None:
    engine, point = _started()
    point = engine.commit_assignments(point, _compute_assignments(declaration, topology)).point
    answer = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, RegionRef)
    region_answer = engine.query_property(point, MVAUComputeKernelPaths.REGION)
    assert isinstance(region_answer, Decided)
    assert answer.value.region == region_answer.value
    assert engine.check_readiness(point, "mvau_op_structural").ready is True


@pytest.mark.parametrize(
    "compute_declaration,delivery_declaration",
    [
        (
            MVAURegionDeclaration.STANDARD_STREAMED,
            CyclicParameterRegionDeclaration.FULL_TILE,
        ),
        (
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
            CyclicParameterRegionDeclaration.CHUNKED,
        ),
    ],
)
def test_cyclic_topologies_resolve_to_structurally_valid_network_refs(
    compute_declaration: MVAURegionDeclaration,
    delivery_declaration: CyclicParameterRegionDeclaration,
) -> None:
    engine, point = _started()
    assignments = {
        **_compute_assignments(compute_declaration, MVAUParameterTopology.CYCLIC),
        **_cyclic_assignments(delivery_declaration),
    }
    point = engine.commit_assignments(point, assignments).point
    answer = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, NetworkRef)
    assert tuple(node.id for node in answer.value.network.nodes) == ("compute", "delivery")
    assert engine.query_property(point, MVAUDataflowOpPaths.NETWORK_VALIDATION) == Decided(
        NetworkValidationReport()
    )
    assert engine.check_readiness(point, "mvau_op_structural").ready is True


def test_source_association_records_flattening_transpose_and_fused_provenance() -> None:
    description = _source_description(fused=True)
    engine, point = _started(source_description=description)
    point = engine.commit_assignments(
        point,
        _compute_assignments(
            MVAURegionDeclaration.STANDARD_EMBEDDED, MVAUParameterTopology.EMBEDDED
        ),
    ).point
    answer = engine.query_property(point, MVAUDataflowOpPaths.SOURCE_ASSOCIATION)
    assert isinstance(answer, Decided)
    association = answer.value
    assert isinstance(association, MVAUSourceAssociation)
    assert association.source_node_id == "node0"
    assert association.fused_source_node_ids == ("matmul", "threshold")
    by_role = {item.role: item for item in association.operands}
    assert by_role["activation"].mapping is CoordinateMappingKind.FLATTEN_LEADING
    assert by_role["activation"].map_position((1, 3)) == (1, 3)
    assert by_role["weight"].mapping is CoordinateMappingKind.TRANSPOSE_2D
    assert by_role["weight"].map_position((3, 2)) == (2, 3)
    assert by_role["output"].mapping is CoordinateMappingKind.FLATTEN_LEADING
    assert by_role["threshold"].mapping is CoordinateMappingKind.BINDING_LOCAL_STATE


def test_source_association_rejects_incorrect_flattened_repetition_extent() -> None:
    description = replace(_source_description(), leading_shape=(1, 3))
    engine, point = _started(source_description=description)
    point = engine.commit_assignments(
        point,
        _compute_assignments(
            MVAURegionDeclaration.STANDARD_EMBEDDED, MVAUParameterTopology.EMBEDDED
        ),
    ).point
    assessment = engine.evaluate_constraint_set(point, "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID] == Decided(False)


def test_direct_interleaved_requires_the_exact_external_weight_sequence() -> None:
    compute = construct_batch_interleaved_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2, 2)
    expected = compute.input_interface("weight").port.beat_sequence
    engine, point = _started(external_weight_sequence=expected)
    point = engine.commit_assignments(
        point,
        _compute_assignments(
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
            MVAUParameterTopology.DIRECT,
        ),
    ).point
    assert engine.evaluate_constraint_set(point, "mvau_op_structural").verdict is True

    wrong = replace(expected, beats=tuple(tuple(reversed(beat)) for beat in expected.beats))
    wrong_engine, wrong_point = _started(external_weight_sequence=wrong)
    wrong_point = wrong_engine.commit_assignments(
        wrong_point,
        _compute_assignments(
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
            MVAUParameterTopology.DIRECT,
        ),
    ).point
    assessment = wrong_engine.evaluate_constraint_set(wrong_point, "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE] == Decided(
        False
    )


def test_direct_interleaved_missing_source_is_unresolved_but_region_is_resolved() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        _compute_assignments(
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
            MVAUParameterTopology.DIRECT,
        ),
    ).point
    assert isinstance(engine.query_property(point, MVAUComputeKernelPaths.REGION), Decided)
    answer = engine.evaluate_constraint_set(point, "mvau_op_structural").answers[
        MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE
    ]
    assert isinstance(answer, Unresolved)


def test_topology_and_delivery_declaration_mismatches_are_explicit_constraints() -> None:
    engine, point = _started()
    assignments = {
        **_compute_assignments(
            MVAURegionDeclaration.STANDARD_EMBEDDED, MVAUParameterTopology.CYCLIC
        ),
        **_cyclic_assignments(CyclicParameterRegionDeclaration.CHUNKED),
    }
    point = engine.commit_assignments(point, assignments).point
    assessment = engine.evaluate_constraint_set(point, "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.TOPOLOGY_MATCHES_REGION] == Decided(False)
    assert isinstance(assessment.answers[MVAUDataflowOpPaths.DELIVERY_MATCHES_REGION], Absent)


def test_all_kernel_topology_and_spatialization_choices_can_commit_together() -> None:
    engine, point = _started()
    assignments = {
        **_compute_assignments(
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
            MVAUParameterTopology.CYCLIC,
        ),
        MVAUComputeKernelPaths.BINDING: MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
        MVAUComputeKernelPaths.COMPUTE_PUMPING: False,
        **_cyclic_assignments(CyclicParameterRegionDeclaration.CHUNKED),
    }
    result = engine.commit_assignments(point, assignments)
    assert {item.disposition for item in result.outcomes} == {"committed"}
    assert engine.check_readiness(result.point, "mvau_op_structural").ready is True


def test_dynamic_is_not_smuggled_in_as_a_legacy_topology_value() -> None:
    engine, point = _started()
    with pytest.raises(RequestError):
        engine.commit_assignments(point, {MVAUDataflowOpPaths.PARAMETER_TOPOLOGY: "dynamic"})


def test_declaration_order_does_not_change_a_fully_committed_result() -> None:
    reversed_spec = DesignSpaceSpec(
        MVAU_DATAFLOW_OP_SPEC.problem_schema,
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.decisions)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.properties)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.constraints)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.constraint_sets)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.readiness_profiles)),
    )
    assignments = {
        **_compute_assignments(
            MVAURegionDeclaration.STANDARD_STREAMED, MVAUParameterTopology.CYCLIC
        ),
        **_cyclic_assignments(CyclicParameterRegionDeclaration.FULL_TILE),
    }
    first_engine, first = _started()
    first = first_engine.commit_assignments(first, assignments).point
    second_engine = Engine()
    second = second_engine.start(second_engine.validate(reversed_spec), _problem())
    second = second_engine.commit_assignments(second, assignments).point
    assert first_engine.query_property(first, MVAUDataflowOpPaths.RESULT) == (
        second_engine.query_property(second, MVAUDataflowOpPaths.RESULT)
    )
