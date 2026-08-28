# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 gate: one compute-Kernel identity axis, no Region x binding grid."""

from __future__ import annotations

import pytest

from finn.dataflow._engine.conformance import MonotonicityHarness
from finn.dataflow.design import (
    Absent,
    Decided,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    FindingKind,
    ProblemSchema,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.kernels import SelectedKernel
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.compute_kernels import (
    BATCH_INTERLEAVED_PATHS,
    FULL_TILE_WEIGHT_EXPORT,
    LEGACY_HLS_PATHS,
    MVAU_COMPUTE_SELECTION,
    PACKED_DSP_PATHS,
    REGION_FORM_EXPORT,
    SOFT_VECTOR_PATHS,
    WEIGHT_INTERFACE,
    MVAUComputeKernelId,
    MVAUComputeKernelPathSet,
    MVAUComputeProblemPaths,
    MVAUDspBlock,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_embedded_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.ops.mvau import MVAU_DATAFLOW_OP_SPEC, MVAUDataflowOpPaths
from finn.dataflow.region import DataflowRegion, NumericElementType
from finn.dataflow.region_validation import RegionValidationReport
from finn.dataflow.spec_algebra import assemble_specs

INT2 = NumericElementType("int", 2)
INT8 = NumericElementType("int", 8)
INT9 = NumericElementType("int", 9)
INT16 = NumericElementType("int", 16)
FLOAT32 = NumericElementType("float", 32)
_MISSING = object()

PATHS = MVAU_COMPUTE_SELECTION.paths
KERNEL_PATHS: dict[MVAUComputeKernelId, MVAUComputeKernelPathSet] = {
    MVAUComputeKernelId.LEGACY_HLS: LEGACY_HLS_PATHS,
    MVAUComputeKernelId.SOFT_VECTOR: SOFT_VECTOR_PATHS,
    MVAUComputeKernelId.PACKED_DSP: PACKED_DSP_PATHS,
    MVAUComputeKernelId.BATCH_INTERLEAVED_DSP: BATCH_INTERLEAVED_PATHS,
}


def _spec() -> DesignSpaceSpec:
    """The compute pool plus exactly the operation-owned facts it reads."""

    return assemble_specs(
        (
            MVAU_COMPUTE_SELECTION.build_spec(),
            DesignSpaceSpec(
                ProblemSchema(
                    tuple(
                        field
                        for field in MVAU_DATAFLOW_OP_SPEC.problem_schema.fields
                        if str(field.path).startswith(("problem.mvau.", "problem.target."))
                        and field.path != MVAUDataflowOpPaths.SOURCE_DESCRIPTION
                    )
                )
            ),
        )
    )


def _problem(
    *,
    repetitions: object = 4,
    matrix_width: object = 4,
    matrix_height: object = 4,
    activation_type: object = INT8,
    weight_type: object = INT8,
    accumulator_type: object = INT16,
    output_type: object = INT16,
    threshold_type: object = _MISSING,
    threshold_initializer_available: object = True,
    computation_profile: object = MVAUComputationProfile.ACCUMULATOR_INTEGER,
    weight_initializer_available: object = True,
    target_dsp: object = MVAUDspBlock.DSP58,
    weights_narrow: object = True,
) -> dict[QualifiedPath, object]:
    P = MVAUComputeProblemPaths
    problem: dict[QualifiedPath, object] = {
        P.REPETITIONS: repetitions,
        P.MATRIX_WIDTH: matrix_width,
        P.MATRIX_HEIGHT: matrix_height,
        P.ACTIVATION_ELEMENT_TYPE: activation_type,
        P.WEIGHT_ELEMENT_TYPE: weight_type,
        P.ACCUMULATOR_ELEMENT_TYPE: accumulator_type,
        P.OUTPUT_ELEMENT_TYPE: output_type,
        P.COMPUTATION_PROFILE: computation_profile,
        P.WEIGHT_INITIALIZER_AVAILABLE: weight_initializer_available,
    }
    if threshold_initializer_available is not _MISSING:
        problem[P.THRESHOLD_INITIALIZER_AVAILABLE] = threshold_initializer_available
    if threshold_type is not _MISSING:
        problem[P.THRESHOLD_ELEMENT_TYPE] = threshold_type
    if target_dsp is not _MISSING:
        problem[P.TARGET_DSP_BLOCK] = target_dsp
    if weights_narrow is not _MISSING:
        problem[P.WEIGHTS_NARROW] = weights_narrow
    return problem


def _started(**overrides: object) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    return engine, engine.start(engine.validate(_spec()), _problem(**overrides))


def _select(
    engine: Engine,
    point: DesignPoint,
    kernel: MVAUComputeKernelId,
    *,
    pe: int = 2,
    simd: int = 2,
    interleave: int = 2,
    weight_source: MVAUWeightSource = MVAUWeightSource.STREAMED,
    resource: MVAUHlsResource = MVAUHlsResource.LUT,
    compute_pumping: bool = False,
) -> DesignPoint:
    paths = KERNEL_PATHS[kernel]
    assignments: dict[QualifiedPath, object] = {
        PATHS.kernel: kernel.value,
        paths.pe: pe,
        paths.simd: simd,
    }
    if kernel is MVAUComputeKernelId.LEGACY_HLS:
        assignments[paths.resource] = resource
        assignments[paths.weight_source] = weight_source
    if kernel is MVAUComputeKernelId.BATCH_INTERLEAVED_DSP:
        assignments[paths.interleave] = interleave
    if kernel in {MVAUComputeKernelId.SOFT_VECTOR, MVAUComputeKernelId.PACKED_DSP}:
        assignments[paths.compute_pumping] = compute_pumping
    return engine.commit_assignments(point, assignments).point


def _region(engine: Engine, point: DesignPoint) -> DataflowRegion:
    answer = engine.query_property(point, PATHS.region)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, DataflowRegion)
    return answer.value


def _feasibility(engine: Engine, point: DesignPoint) -> object:
    return engine.evaluate_constraint_set(point, MVAU_COMPUTE_SELECTION.feasibility_constraint_set)


# -- one identity axis -------------------------------------------------------


def test_the_pool_has_exactly_one_identity_decision() -> None:
    spec = MVAU_COMPUTE_SELECTION.build_spec()
    identity = [item for item in spec.decisions if item.path == PATHS.kernel]
    assert len(identity) == 1
    assert set(MVAU_COMPUTE_SELECTION.candidate_ids) == {
        kernel.value for kernel in MVAUComputeKernelId
    }
    other = {str(item.path) for item in spec.decisions if item.path != PATHS.kernel}
    assert not any("binding" in path for path in other)
    assert not any("weight_interface" in path for path in other)


def test_soft_vector_and_packed_produce_equal_standard_streamed_regions() -> None:
    engine, root = _started()
    left = _region(engine, _select(engine, root, MVAUComputeKernelId.SOFT_VECTOR))
    right = _region(engine, _select(engine, root, MVAUComputeKernelId.PACKED_DSP))
    expected = construct_standard_streamed_mvau_region(4, 4, 4, INT8, INT8, INT16, 2, 2)
    assert left == right == expected


def test_equal_regions_keep_distinct_kernel_identities_and_providers() -> None:
    engine, root = _started()
    left = _select(engine, root, MVAUComputeKernelId.SOFT_VECTOR)
    right = _select(engine, root, MVAUComputeKernelId.PACKED_DSP)
    assert engine.query_property(left, PATHS.selected_kernel) == Decided(
        SelectedKernel(MVAU_COMPUTE_SELECTION.name, MVAUComputeKernelId.SOFT_VECTOR.value, "1")
    )
    assert engine.query_property(right, PATHS.selected_kernel) == Decided(
        SelectedKernel(MVAU_COMPUTE_SELECTION.name, MVAUComputeKernelId.PACKED_DSP.value, "1")
    )
    assert MVAU_COMPUTE_SELECTION.kernel(
        MVAUComputeKernelId.SOFT_VECTOR.value
    ) is not MVAU_COMPUTE_SELECTION.kernel(MVAUComputeKernelId.PACKED_DSP.value)


def test_the_batch_interleaved_kernel_produces_only_its_own_region() -> None:
    engine, point = _started(repetitions=6, matrix_height=6, target_dsp=MVAUDspBlock.DSP58)
    point = _select(
        engine, point, MVAUComputeKernelId.BATCH_INTERLEAVED_DSP, pe=3, simd=2, interleave=3
    )
    expected = construct_batch_interleaved_streamed_mvau_region(6, 4, 6, INT8, INT8, INT16, 3, 2, 3)
    assert _region(engine, point) == expected
    assert engine.query_property(point, PATHS.export(REGION_FORM_EXPORT)) == Decided(
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
    )
    assert engine.query_property(point, PATHS.region_validation) == Decided(
        RegionValidationReport()
    )


@pytest.mark.parametrize(
    "weight_source,constructor,region_form",
    [
        (
            MVAUWeightSource.EMBEDDED,
            construct_standard_embedded_mvau_region,
            MVAURegionDeclaration.STANDARD_EMBEDDED,
        ),
        (
            MVAUWeightSource.STREAMED,
            construct_standard_streamed_mvau_region,
            MVAURegionDeclaration.STANDARD_STREAMED,
        ),
    ],
)
def test_legacy_hls_derives_both_of_its_region_forms(
    weight_source: MVAUWeightSource,
    constructor: object,
    region_form: MVAURegionDeclaration,
) -> None:
    engine, point = _started(target_dsp=_MISSING, weights_narrow=_MISSING)
    point = _select(engine, point, MVAUComputeKernelId.LEGACY_HLS, weight_source=weight_source)
    assert _region(engine, point) == constructor(4, 4, 4, INT8, INT8, INT16, 2, 2)  # type: ignore[operator]
    assert engine.query_property(point, PATHS.export(REGION_FORM_EXPORT)) == Decided(region_form)
    assert (
        engine.check_readiness(point, MVAU_COMPUTE_SELECTION.structural_constraint_set).ready
        is True
    )


def test_an_embedded_legacy_region_publishes_no_weight_demand() -> None:
    engine, point = _started()
    embedded = _select(
        engine, point, MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.EMBEDDED
    )
    assert isinstance(engine.query_property(embedded, PATHS.demand(WEIGHT_INTERFACE)), Absent)
    streamed = _select(
        engine, point, MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.STREAMED
    )
    demand = engine.query_property(streamed, PATHS.demand(WEIGHT_INTERFACE))
    assert isinstance(demand, Decided)
    assert demand.value == _region(engine, streamed).input_interface("weight").port


def test_embedded_weights_are_unavailable_without_an_initializer() -> None:
    engine, point = _started(weight_initializer_available=False)
    candidates = engine.enumerate_candidates(
        engine.commit_assignments(
            point, {PATHS.kernel: MVAUComputeKernelId.LEGACY_HLS.value}
        ).point,
        LEGACY_HLS_PATHS.weight_source,
    )
    assert isinstance(candidates, Decided)
    assert candidates.value == (MVAUWeightSource.STREAMED,)


def test_invalid_former_region_binding_pairs_are_unrepresentable() -> None:
    engine, point = _started()
    # There is no interleave decision on a soft-vector point at all, and no
    # weight-source decision either: the pairing cannot be constructed.
    point = _select(engine, point, MVAUComputeKernelId.SOFT_VECTOR)
    assert isinstance(engine.decision_state(point, BATCH_INTERLEAVED_PATHS.interleave), Absent)
    assert isinstance(engine.decision_state(point, LEGACY_HLS_PATHS.weight_source), Absent)
    assert isinstance(engine.decision_state(point, PACKED_DSP_PATHS.compute_pumping), Absent)


def test_region_derivation_does_not_read_target_or_provider_facts() -> None:
    forbidden = {
        MVAUComputeProblemPaths.TARGET_DSP_BLOCK,
        MVAUComputeProblemPaths.WEIGHTS_NARROW,
    }
    for kernel in MVAU_COMPUTE_SELECTION.kernels:
        region = next(item for item in kernel.spec.properties if item.path == kernel.region_path)
        dependencies = {item.path for item in region.evaluator.dependencies}
        assert dependencies.isdisjoint(forbidden)


def test_region_validation_is_separate_from_kernel_feasibility() -> None:
    engine, point = _started(matrix_width=2048)
    point = _select(engine, point, MVAUComputeKernelId.LEGACY_HLS, simd=1)
    assert (
        engine.evaluate_constraint_set(
            point, MVAU_COMPUTE_SELECTION.structural_constraint_set
        ).verdict
        is True
    )
    assessment = _feasibility(engine, point)
    assert assessment.answers[LEGACY_HLS_PATHS.constraint("partition_supported")] == Decided(False)


# -- Kernel-local feasibility ------------------------------------------------


def test_a_missing_target_fact_leaves_only_the_applicable_constraint_unresolved() -> None:
    engine, point = _started(target_dsp=_MISSING, weights_narrow=_MISSING)
    point = _select(engine, point, MVAUComputeKernelId.SOFT_VECTOR)
    assessment = _feasibility(engine, point)
    target = assessment.answers[SOFT_VECTOR_PATHS.constraint("target_supported")]
    assert isinstance(target, Unresolved)
    assert target.findings[0].kind is FindingKind.LIMITATION

    hls_engine, hls = _started(target_dsp=_MISSING, weights_narrow=_MISSING)
    hls = _select(hls_engine, hls, MVAUComputeKernelId.LEGACY_HLS)
    hls_assessment = _feasibility(hls_engine, hls)
    assert hls_assessment.verdict is True


def test_packed_preserves_the_num_lanes_limit() -> None:
    engine, point = _started(activation_type=INT2, weight_type=INT2)
    point = _select(engine, point, MVAUComputeKernelId.PACKED_DSP)
    assessment = _feasibility(engine, point)
    assert assessment.answers[PACKED_DSP_PATHS.constraint("packing_supported")] == Decided(False)


@pytest.mark.parametrize(
    "activation_type,weight_type",
    [
        (NumericElementType("int", 1), INT8),
        (INT8, NumericElementType("int", 1)),
    ],
)
def test_rtl_minimum_input_width_assertions_are_represented(
    activation_type: NumericElementType, weight_type: NumericElementType
) -> None:
    engine, point = _started(activation_type=activation_type, weight_type=weight_type)
    point = _select(engine, point, MVAUComputeKernelId.SOFT_VECTOR)
    assessment = _feasibility(engine, point)
    assert assessment.answers[SOFT_VECTOR_PATHS.constraint("numeric_supported")] == Decided(False)


def test_the_batch_interleaved_kernel_requires_dsp58_and_narrow_datapaths() -> None:
    engine, point = _started(
        repetitions=6,
        matrix_height=6,
        activation_type=INT9,
        weight_type=INT9,
        target_dsp=MVAUDspBlock.DSP48E2,
    )
    point = _select(
        engine, point, MVAUComputeKernelId.BATCH_INTERLEAVED_DSP, pe=3, simd=2, interleave=3
    )
    assessment = _feasibility(engine, point)
    false_paths = {path for path, answer in assessment.answers.items() if answer == Decided(False)}
    assert {
        BATCH_INTERLEAVED_PATHS.constraint("target_supported"),
        BATCH_INTERLEAVED_PATHS.constraint("tiled_width_supported"),
    } <= false_paths
    assert assessment.verdict is False


def test_rtl_kernels_support_only_the_accumulator_profile() -> None:
    engine, point = _started(
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD, threshold_type=INT16
    )
    point = _select(engine, point, MVAUComputeKernelId.SOFT_VECTOR)
    assessment = _feasibility(engine, point)
    assert assessment.answers[SOFT_VECTOR_PATHS.constraint("computation_supported")] == Decided(
        False
    )


def test_fused_threshold_requires_a_representable_threshold_type() -> None:
    engine, point = _started(
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD, threshold_type=_MISSING
    )
    point = _select(
        engine, point, MVAUComputeKernelId.LEGACY_HLS, weight_source=MVAUWeightSource.EMBEDDED
    )
    answer = _feasibility(engine, point).answers[
        LEGACY_HLS_PATHS.constraint("threshold_representable")
    ]
    assert isinstance(answer, Unresolved)

    narrow_engine, narrow = _started(
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD,
        threshold_type=INT8,
        accumulator_type=INT16,
    )
    narrow = _select(
        narrow_engine,
        narrow,
        MVAUComputeKernelId.LEGACY_HLS,
        weight_source=MVAUWeightSource.EMBEDDED,
    )
    assert _feasibility(narrow_engine, narrow).answers[
        LEGACY_HLS_PATHS.constraint("threshold_representable")
    ] == Decided(False)


def test_accumulator_profiles_require_equal_accumulator_and_output_types() -> None:
    engine, point = _started(accumulator_type=INT16, output_type=INT8)
    point = _select(engine, point, MVAUComputeKernelId.SOFT_VECTOR)
    assert _feasibility(engine, point).answers[
        SOFT_VECTOR_PATHS.constraint("accumulator_output_type_supported")
    ] == Decided(False)


@pytest.mark.parametrize(
    "target,accumulator",
    [
        (MVAUDspBlock.DSP48E1, NumericElementType("int", 49)),
        (MVAUDspBlock.DSP58, NumericElementType("int", 59)),
    ],
)
def test_rtl_width_envelope_rejects_datapath_overflow(
    target: MVAUDspBlock, accumulator: NumericElementType
) -> None:
    engine, point = _started(
        accumulator_type=accumulator, output_type=accumulator, target_dsp=target
    )
    point = _select(engine, point, MVAUComputeKernelId.SOFT_VECTOR)
    assert _feasibility(engine, point).answers[
        SOFT_VECTOR_PATHS.constraint("width_supported")
    ] == Decided(False)


def test_compute_pumping_is_a_soft_vector_and_packed_choice_only() -> None:
    engine, point = _started()
    hls = _select(engine, point, MVAUComputeKernelId.LEGACY_HLS)
    assert isinstance(engine.decision_state(hls, SOFT_VECTOR_PATHS.compute_pumping), Absent)
    rtl = _select(engine, point, MVAUComputeKernelId.SOFT_VECTOR, simd=1, compute_pumping=True)
    assert _feasibility(engine, rtl).answers[
        SOFT_VECTOR_PATHS.constraint("compute_pumping_supported")
    ] == Decided(False)


def test_each_streamed_kernel_exports_its_natural_full_tile_contract() -> None:
    engine, point = _started(repetitions=6, matrix_height=6)
    interleaved = _select(
        engine, point, MVAUComputeKernelId.BATCH_INTERLEAVED_DSP, pe=3, simd=2, interleave=3
    )
    demand = engine.query_property(interleaved, PATHS.demand(WEIGHT_INTERFACE))
    full_tile = engine.query_property(interleaved, PATHS.export(FULL_TILE_WEIGHT_EXPORT))
    assert isinstance(demand, Decided) and isinstance(full_tile, Decided)
    assert demand.value != full_tile.value


def test_wrong_nominal_assignment_types_are_request_errors() -> None:
    engine, point = _started()
    with pytest.raises(RequestError) as caught:
        engine.commit_assignments(point, {SOFT_VECTOR_PATHS.pe: "two"})
    assert caught.value.findings[0].code == "assignment-type"


def test_compute_assignments_pass_the_monotonicity_harness() -> None:
    engine, point = _started()
    result = MonotonicityHarness(engine).verify(
        point,
        {
            PATHS.kernel: tuple(kernel.value for kernel in MVAUComputeKernelId),
            LEGACY_HLS_PATHS.pe: (1, 2, 4),
            LEGACY_HLS_PATHS.simd: (1, 2, 4),
            LEGACY_HLS_PATHS.resource: tuple(MVAUHlsResource),
            LEGACY_HLS_PATHS.weight_source: tuple(MVAUWeightSource),
            SOFT_VECTOR_PATHS.pe: (1, 2, 4),
            SOFT_VECTOR_PATHS.simd: (1, 2, 4),
            SOFT_VECTOR_PATHS.compute_pumping: (False, True),
            PACKED_DSP_PATHS.pe: (1, 2, 4),
            PACKED_DSP_PATHS.simd: (1, 2, 4),
            PACKED_DSP_PATHS.compute_pumping: (False, True),
            BATCH_INTERLEAVED_PATHS.pe: (1, 2, 4),
            BATCH_INTERLEAVED_PATHS.simd: (1, 2, 4),
            BATCH_INTERLEAVED_PATHS.interleave: (2, 4),
        },
    )
    assert result.violations == ()
    assert result.conformant is True
