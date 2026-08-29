# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5 gate: MVAU composition is selected Kernels, not a choice grid."""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

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
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.compute_kernels import (
    BATCH_INTERLEAVED_PATHS,
    LEGACY_HLS_PATHS,
    PACKED_DSP_PATHS,
    SOFT_VECTOR_PATHS,
    MVAUComputeKernelId,
    MVAUComputeKernelPathSet,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.mvau.regions import construct_batch_interleaved_streamed_mvau_region
from finn.dataflow.mvau.weight_adapter_kernel import FULL_TILE_TO_CHUNKED
from finn.dataflow.network_validation import NetworkValidationReport
from finn.dataflow.ops.mvau import (
    MVAU_COMPUTE_SELECTION,
    MVAU_DATAFLOW_OP_SPEC,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    BindingLocalStateDestination,
    CoordinateMappingKind,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    MVAUSourceDescription,
    NetworkRef,
    RegionRef,
    SemanticOperandDestination,
)
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.parameters.supply_kernels import (
    FINNLIB_MEMSTREAM_PATHS,
    FINN_RTL_MEMSTREAM_PATHS,
    MVAUWeightSupplyKernelId,
    WeightOrganization,
)
from finn.dataflow.region import Port
from finn.dataflow.selection import enumerate_feasible_points
from finn.dataflow.mvau_problem import MVAUDspBlock, MVAUProblemPaths

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]

COMPUTE = MVAU_COMPUTE_SELECTION.paths
SUPPLY = MVAU_WEIGHT_SUPPLY_SELECTION.paths
ADAPTER = MVAU_WEIGHT_ADAPTER_SELECTION.paths

KERNEL_PATHS: dict[MVAUComputeKernelId, MVAUComputeKernelPathSet] = {
    MVAUComputeKernelId.LEGACY_HLS: LEGACY_HLS_PATHS,
    MVAUComputeKernelId.SOFT_VECTOR: SOFT_VECTOR_PATHS,
    MVAUComputeKernelId.PACKED_DSP: PACKED_DSP_PATHS,
    MVAUComputeKernelId.BATCH_INTERLEAVED_DSP: BATCH_INTERLEAVED_PATHS,
}


def _source_description(*, fused: bool = False) -> MVAUSourceDescription:
    return MVAUSourceDescription(
        "node0",
        "source_x",
        "source_w",
        "source_y",
        (2,),
        "source_thresholds" if fused else None,
        ("matmul", "threshold") if fused else ("matmul",),
        (4, 3) if fused else None,
    )


def _problem(
    *,
    repetitions: int = 2,
    matrix_width: int = 4,
    matrix_height: int = 4,
    source_description: MVAUSourceDescription | None = None,
    external_weight_sequence: object | None = None,
    computation_profile: MVAUComputationProfile = MVAUComputationProfile.ACCUMULATOR_INTEGER,
) -> dict[QualifiedPath, object]:
    P = MVAUProblemPaths
    problem: dict[QualifiedPath, object] = {
        P.REPETITIONS: repetitions,
        P.MATRIX_WIDTH: matrix_width,
        P.MATRIX_HEIGHT: matrix_height,
        P.ACTIVATION_ELEMENT_TYPE: INT8,
        P.WEIGHT_ELEMENT_TYPE: INT8,
        P.ACCUMULATOR_ELEMENT_TYPE: INT16,
        P.OUTPUT_ELEMENT_TYPE: INT16,
        P.COMPUTATION_PROFILE: computation_profile,
        P.WEIGHT_INITIALIZER_AVAILABLE: True,
        P.THRESHOLD_INITIALIZER_AVAILABLE: True,
        P.TARGET_DSP_BLOCK: MVAUDspBlock.DSP58,
        P.INITIALIZER_EXCLUDES_MINIMUM: True,
        MVAUProblemPaths.RUNTIME_WRITABLE: False,
        MVAUDataflowOpPaths.SOURCE_DESCRIPTION: source_description or _source_description(),
    }
    if external_weight_sequence is not None:
        problem[MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE] = external_weight_sequence
    return problem


def _started(**overrides: object) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(MVAU_DATAFLOW_OP_SPEC)
    return engine, engine.start(space, _problem(**overrides))  # type: ignore[arg-type]


def _compute(
    kernel: MVAUComputeKernelId,
    *,
    pe: int = 2,
    simd: int = 2,
    interleave: int = 2,
    weight_source: MVAUWeightSource = MVAUWeightSource.STREAMED,
) -> dict[QualifiedPath, object]:
    paths = KERNEL_PATHS[kernel]
    assignments: dict[QualifiedPath, object] = {
        COMPUTE.kernel: kernel.value,
        paths.pe: pe,
        paths.simd: simd,
    }
    if kernel is MVAUComputeKernelId.LEGACY_HLS:
        assignments[paths.resource] = MVAUHlsResource.LUT
        assignments[paths.weight_source] = weight_source
    if kernel is MVAUComputeKernelId.BATCH_INTERLEAVED_DSP:
        assignments[paths.interleave] = interleave
    if kernel in {MVAUComputeKernelId.SOFT_VECTOR, MVAUComputeKernelId.PACKED_DSP}:
        assignments[paths.compute_pumping] = False
    return assignments


def _supply(
    *,
    organization: WeightOrganization = WeightOrganization.AS_DEMANDED,
    pumped_memory: bool = False,
    adapter: str = NO_KERNEL,
) -> dict[QualifiedPath, object]:
    return {
        SUPPLY.kernel: MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value,
        FINN_RTL_MEMSTREAM_PATHS.organization: organization,
        FINN_RTL_MEMSTREAM_PATHS.ram_style: CyclicRamStyle.BRAM,
        FINN_RTL_MEMSTREAM_PATHS.pumped_memory: pumped_memory,
        ADAPTER.kernel: adapter,
    }


def _unsupplied() -> dict[QualifiedPath, object]:
    return {SUPPLY.kernel: NO_KERNEL}


# -- assembly shapes ---------------------------------------------------------


@pytest.mark.parametrize(
    "assignments,topology",
    [
        (
            {**_compute(MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.EMBEDDED)},
            MVAUParameterTopology.EMBEDDED,
        ),
        (
            {**_compute(MVAUComputeKernelId.SOFT_VECTOR), **_unsupplied()},
            MVAUParameterTopology.DIRECT,
        ),
    ],
)
def test_an_unsupplied_weight_path_resolves_to_a_region_ref(
    assignments: dict[QualifiedPath, object], topology: MVAUParameterTopology
) -> None:
    engine, point = _started()
    point = engine.commit_assignments(point, assignments).point
    answer = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, RegionRef)
    assert answer.value.source_association.parameter_topology is topology
    region = engine.query_property(point, COMPUTE.region)
    assert isinstance(region, Decided)
    assert answer.value.region == region.value
    assert engine.check_readiness(point, "mvau_op_structural").ready is True


@pytest.mark.parametrize(
    "kernel",
    [MVAUComputeKernelId.SOFT_VECTOR, MVAUComputeKernelId.BATCH_INTERLEAVED_DSP],
)
def test_a_selected_supplier_resolves_to_a_valid_network_ref(
    kernel: MVAUComputeKernelId,
) -> None:
    engine, point = _started()
    point = engine.commit_assignments(point, {**_compute(kernel), **_supply()}).point
    answer = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, NetworkRef)
    assert answer.value.source_association.parameter_topology is MVAUParameterTopology.CYCLIC
    assert tuple(node.id for node in answer.value.network.nodes) == ("compute", "delivery")
    assert engine.query_property(point, MVAUDataflowOpPaths.NETWORK_VALIDATION) == Decided(
        NetworkValidationReport()
    )
    assert engine.check_readiness(point, "mvau_op_structural").ready is True


def test_the_topology_is_derived_from_the_selected_kernels_not_decided() -> None:
    decisions = {str(item.path) for item in MVAU_DATAFLOW_OP_SPEC.decisions}
    assert str(MVAUDataflowOpPaths.PARAMETER_TOPOLOGY) not in decisions
    assert not any("connection_topology" in path for path in decisions)
    assert not any("delivery" in path and path.endswith((".pe", ".simd")) for path in decisions)
    properties = {str(item.path) for item in MVAU_DATAFLOW_OP_SPEC.properties}
    assert str(MVAUDataflowOpPaths.PARAMETER_TOPOLOGY) in properties


def test_a_direct_connection_has_no_adapter_and_matches_exactly() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point, {**_compute(MVAUComputeKernelId.SOFT_VECTOR), **_supply()}
    ).point
    producer = engine.query_property(point, SUPPLY.export("output_port"))
    consumer = engine.query_property(point, MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT)
    result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(producer, Decided) and isinstance(consumer, Decided)
    assert isinstance(producer.value, Port) and isinstance(consumer.value, Port)
    assert producer.value.beat_sequence == consumer.value.beat_sequence
    assert isinstance(result, Decided) and isinstance(result.value, NetworkRef)
    assert tuple(edge.id for edge in result.value.network.edges) == ("weight",)
    assert result.value.source_association.adapter_kernel_id is None


def test_an_independently_organized_supplier_uses_one_explicit_adapter_region() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        {
            **_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP),
            **_supply(
                organization=WeightOrganization.STANDARD_FULL_TILE,
                adapter=FULL_TILE_TO_CHUNKED,
            ),
        },
    ).point
    result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assessment = engine.evaluate_constraint_set(point, "mvau_op_structural")
    assert isinstance(result, Decided) and isinstance(result.value, NetworkRef)
    assert tuple(node.id for node in result.value.network.nodes) == (
        "compute",
        "delivery",
        "weight_adapter",
    )
    assert tuple(edge.id for edge in result.value.network.edges) == (
        "adapter_to_compute",
        "delivery_to_adapter",
    )
    assert result.value.source_association.adapter_kernel_id == FULL_TILE_TO_CHUNKED
    assert assessment.answers[MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED] == Decided(True)
    assert engine.query_property(point, MVAUDataflowOpPaths.NETWORK_VALIDATION) == Decided(
        NetworkValidationReport()
    )


def test_an_unadapted_mismatch_remains_infeasible() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        {
            **_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP),
            **_supply(organization=WeightOrganization.STANDARD_FULL_TILE),
        },
    ).point
    assessment = engine.evaluate_constraint_set(point, "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED] == Decided(False)
    assert point.assignments[ADAPTER.kernel] == NO_KERNEL


def test_joint_selection_finds_both_direct_and_adapter_compositions() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        {
            **_compute(MVAUComputeKernelId.SOFT_VECTOR),
            SUPPLY.kernel: MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value,
            FINN_RTL_MEMSTREAM_PATHS.ram_style: CyclicRamStyle.BRAM,
            FINN_RTL_MEMSTREAM_PATHS.pumped_memory: False,
        },
    ).point
    coordinated = (FINN_RTL_MEMSTREAM_PATHS.organization, ADAPTER.kernel)

    def signatures(points: tuple[DesignPoint, ...]) -> set[tuple[object, object]]:
        return {
            (
                candidate.assignments[FINN_RTL_MEMSTREAM_PATHS.organization],
                candidate.assignments[ADAPTER.kernel],
            )
            for candidate in points
        }

    forward = enumerate_feasible_points(
        engine,
        point,
        coordinated,
        constraint_set="mvau_op_structural",
        traversal_order=coordinated,
    )
    reverse = enumerate_feasible_points(
        engine,
        point,
        coordinated,
        constraint_set="mvau_op_structural",
        traversal_order=tuple(reversed(coordinated)),
    )
    # Both organizations produce the same sequence for a standard-streamed
    # demand, so both connect directly and neither needs the adapter.
    expected = {
        (WeightOrganization.AS_DEMANDED, NO_KERNEL),
        (WeightOrganization.STANDARD_FULL_TILE, NO_KERNEL),
    }
    assert signatures(forward.points) == expected
    assert signatures(reverse.points) == expected


def test_a_supplier_failure_can_veto_a_point_without_rewriting_compute() -> None:
    engine, point = _started()
    compute = _compute(MVAUComputeKernelId.SOFT_VECTOR)
    region_before = engine.query_property(
        engine.commit_assignments(point, {**compute, **_unsupplied()}).point, COMPUTE.region
    )
    vetoed = engine.commit_assignments(
        point,
        {
            **compute,
            SUPPLY.kernel: MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM.value,
            FINNLIB_MEMSTREAM_PATHS.organization: WeightOrganization.AS_DEMANDED,
            ADAPTER.kernel: NO_KERNEL,
        },
    ).point
    # FinnLib is feasible here; make it infeasible by removing the initializer.
    infeasible_engine, infeasible = _started()
    infeasible = infeasible_engine.start(
        infeasible_engine.validate(MVAU_DATAFLOW_OP_SPEC),
        {
            **_problem(),
            MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: False,
        },
    )
    infeasible = infeasible_engine.commit_assignments(
        infeasible,
        {
            **compute,
            SUPPLY.kernel: MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM.value,
            FINNLIB_MEMSTREAM_PATHS.organization: WeightOrganization.AS_DEMANDED,
            ADAPTER.kernel: NO_KERNEL,
        },
    ).point
    assert (
        infeasible_engine.evaluate_constraint_set(infeasible, "mvau_op_feasibility").verdict
        is False
    )
    # The compute Region is untouched by the supplier's failure.
    assert infeasible_engine.query_property(infeasible, COMPUTE.region) == region_before
    assert engine.query_property(vetoed, COMPUTE.region) == region_before


# -- source association ------------------------------------------------------


def test_source_association_records_flattening_transpose_and_fused_provenance() -> None:
    description = _source_description(fused=True)
    engine, point = _started(
        source_description=description,
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD,
    )
    point = engine.commit_assignments(
        point,
        _compute(MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.EMBEDDED),
    ).point
    answer = engine.query_property(point, MVAUDataflowOpPaths.SOURCE_ASSOCIATION)
    assert isinstance(answer, Decided)
    association = answer.value
    assert isinstance(association, MVAUSourceAssociation)
    assert association.source_node_id == "node0"
    assert association.fused_source_node_ids == ("matmul", "threshold")
    assert association.compute_kernel_id == MVAUComputeKernelId.LEGACY_HLS.value
    assert association.supply_kernel_id is None
    by_role = {item.role: item for item in association.operands}
    assert by_role["activation"].mapping is CoordinateMappingKind.FLATTEN_LEADING
    assert by_role["activation"].destination == SemanticOperandDestination("mvau.compute", "X")
    assert by_role["activation"].map_position((1, 3)) == (1, 3)
    assert by_role["weight"].mapping is CoordinateMappingKind.TRANSPOSE_2D
    assert by_role["weight"].destination == BindingLocalStateDestination("mvau.compute", "weights")
    assert by_role["weight"].map_position((3, 2)) == (2, 3)
    assert by_role["output"].mapping is CoordinateMappingKind.FLATTEN_LEADING
    assert by_role["threshold"].mapping is CoordinateMappingKind.BINDING_LOCAL_STATE
    assert by_role["threshold"].source_shape == (4, 3)
    assert by_role["threshold"].destination == BindingLocalStateDestination(
        "mvau.compute", "thresholds"
    )


@pytest.mark.parametrize(
    "topology,weight_destination,semantic_owner",
    [
        (
            MVAUParameterTopology.EMBEDDED,
            BindingLocalStateDestination("mvau.compute", "weights"),
            "mvau.compute",
        ),
        (
            MVAUParameterTopology.DIRECT,
            SemanticOperandDestination("mvau.compute", "W"),
            "mvau.compute",
        ),
        (
            MVAUParameterTopology.CYCLIC,
            BindingLocalStateDestination("delivery", "weights"),
            "compute",
        ),
    ],
)
def test_source_associations_are_topology_aware_and_qualified(
    topology: MVAUParameterTopology,
    weight_destination: object,
    semantic_owner: str,
) -> None:
    engine, point = _started()
    if topology is MVAUParameterTopology.EMBEDDED:
        assignments = _compute(
            MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.EMBEDDED
        )
    elif topology is MVAUParameterTopology.DIRECT:
        assignments = {**_compute(MVAUComputeKernelId.SOFT_VECTOR), **_unsupplied()}
    else:
        assignments = {**_compute(MVAUComputeKernelId.SOFT_VECTOR), **_supply()}
    point = engine.commit_assignments(point, assignments).point
    association_answer = engine.query_property(point, MVAUDataflowOpPaths.SOURCE_ASSOCIATION)
    result_answer = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(association_answer, Decided)
    assert isinstance(result_answer, Decided)
    association = association_answer.value
    assert isinstance(association, MVAUSourceAssociation)
    associations = {item.role: item for item in association.operands}
    assert associations["weight"].destination == weight_destination
    for role in ("activation", "output"):
        destination = associations[role].destination
        assert isinstance(destination, SemanticOperandDestination)
        assert destination.owner_id == semantic_owner
        if isinstance(result_answer.value, RegionRef):
            assert destination.operand_id in {
                interface.port.operand.id for interface in result_answer.value.region.interfaces
            }
        else:
            assert isinstance(result_answer.value, NetworkRef)
            node = result_answer.value.network.node(destination.owner_id)
            assert destination.operand_id in {
                interface.port.operand.id for interface in node.region.interfaces
            }


def test_source_association_rejects_incorrect_flattened_repetition_extent() -> None:
    description = replace(_source_description(), leading_shape=(1, 3))
    engine, point = _started(source_description=description)
    point = engine.commit_assignments(
        point,
        _compute(MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.EMBEDDED),
    ).point
    assessment = engine.evaluate_constraint_set(point, "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID] == Decided(False)


def test_the_association_records_every_selected_kernel_identity() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        {
            **_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP),
            **_supply(
                organization=WeightOrganization.STANDARD_FULL_TILE,
                adapter=FULL_TILE_TO_CHUNKED,
            ),
        },
    ).point
    answer = engine.query_property(point, MVAUDataflowOpPaths.SOURCE_ASSOCIATION)
    assert isinstance(answer, Decided)
    association = answer.value
    assert isinstance(association, MVAUSourceAssociation)
    assert association.compute_kernel_id == MVAUComputeKernelId.BATCH_INTERLEAVED_DSP.value
    assert association.supply_kernel_id == MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
    assert association.adapter_kernel_id == FULL_TILE_TO_CHUNKED


# -- exposed boundaries ------------------------------------------------------


def test_an_exposed_interleaved_boundary_requires_the_exact_external_sequence() -> None:
    compute = construct_batch_interleaved_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2, 2)
    expected = compute.input_interface("weight").port.beat_sequence
    engine, point = _started(external_weight_sequence=expected)
    point = engine.commit_assignments(
        point,
        {**_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP), **_unsupplied()},
    ).point
    assert engine.evaluate_constraint_set(point, "mvau_op_structural").verdict is True

    wrong = replace(expected, beats=tuple(tuple(reversed(beat)) for beat in expected.beats))
    wrong_engine, wrong_point = _started(external_weight_sequence=wrong)
    wrong_point = wrong_engine.commit_assignments(
        wrong_point,
        {**_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP), **_unsupplied()},
    ).point
    assessment = wrong_engine.evaluate_constraint_set(wrong_point, "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.EXPOSED_WEIGHT_SOURCE_AVAILABLE] == Decided(False)


def test_an_exposed_interleaved_boundary_without_a_source_is_unresolved() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        {**_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP), **_unsupplied()},
    ).point
    assert isinstance(engine.query_property(point, COMPUTE.region), Decided)
    answer = engine.evaluate_constraint_set(point, "mvau_op_structural").answers[
        MVAUDataflowOpPaths.EXPOSED_WEIGHT_SOURCE_AVAILABLE
    ]
    assert isinstance(answer, Unresolved)


def test_an_embedded_region_makes_the_supply_pool_inapplicable() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        _compute(MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.EMBEDDED),
    ).point
    assert isinstance(engine.decision_state(point, SUPPLY.kernel), Absent)
    assert isinstance(engine.query_property(point, SUPPLY.region), Absent)


def test_interleaved_compute_rejects_unvalidated_pumped_supply() -> None:
    engine, point = _started()
    point = engine.commit_assignments(
        point,
        {
            **_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP),
            **_supply(pumped_memory=True),
        },
    ).point
    assessment = engine.evaluate_constraint_set(point, "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.INTERLEAVED_PUMPING_SUPPORTED] == Decided(False)


# -- commitment and order ----------------------------------------------------


def test_every_pool_and_local_choice_can_commit_together() -> None:
    engine, point = _started()
    result = engine.commit_assignments(
        point,
        {**_compute(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP), **_supply()},
    )
    assert {item.disposition for item in result.outcomes} == {"committed"}
    assert engine.check_readiness(result.point, "mvau_op_structural").ready is True


def test_an_unknown_kernel_identity_is_rejected() -> None:
    engine, point = _started()
    outcome = engine.commit_assignments(point, {COMPUTE.kernel: "dynamic"}).outcomes[0]
    assert outcome.disposition not in {"committed", "unchanged"}
    with pytest.raises(RequestError):
        engine.commit_assignments(point, {COMPUTE.kernel: 3})


def test_declaration_order_does_not_change_a_fully_committed_result() -> None:
    reversed_spec = DesignSpaceSpec(
        MVAU_DATAFLOW_OP_SPEC.problem_schema,
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.decisions)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.properties)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.constraints)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.constraint_sets)),
        tuple(reversed(MVAU_DATAFLOW_OP_SPEC.readiness_profiles)),
    )
    assignments = {**_compute(MVAUComputeKernelId.SOFT_VECTOR), **_supply()}
    first_engine, first = _started()
    first = first_engine.commit_assignments(first, assignments).point
    second_engine = Engine()
    second = second_engine.start(second_engine.validate(reversed_spec), _problem())
    second = second_engine.commit_assignments(second, assignments).point
    assert first_engine.query_property(first, MVAUDataflowOpPaths.RESULT) == (
        second_engine.query_property(second, MVAUDataflowOpPaths.RESULT)
    )
