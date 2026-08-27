# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from finn.dataflow._engine.conformance import MonotonicityHarness
from finn.dataflow.design import (
    Absent,
    Decided,
    DesignPoint,
    Engine,
    FindingKind,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.mvau.computation import MVAUBindingSelection, MVAUComputationProfile
from finn.dataflow.mvau.definition import (
    MVAU_COMPUTE_KERNEL_SPEC,
    MVAUComputeBinding,
    MVAUComputeKernelPaths,
    MVAUDspBlock,
)
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    MVAUWeightInterface,
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_embedded_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.mvau_design import MVAU_DESIGN_SPACE_SPEC, MVAUDesignPaths
from finn.dataflow.region import DataflowRegion, NumericElementType
from finn.dataflow.region_validation import RegionValidationReport

INT2 = NumericElementType("int", 2)
INT8 = NumericElementType("int", 8)
INT9 = NumericElementType("int", 9)
INT16 = NumericElementType("int", 16)
FLOAT32 = NumericElementType("float", 32)
_MISSING = object()


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
) -> dict[str, object]:
    problem = {
        str(MVAUComputeKernelPaths.REPETITIONS): repetitions,
        str(MVAUComputeKernelPaths.MATRIX_WIDTH): matrix_width,
        str(MVAUComputeKernelPaths.MATRIX_HEIGHT): matrix_height,
        str(MVAUComputeKernelPaths.ACTIVATION_ELEMENT_TYPE): activation_type,
        str(MVAUComputeKernelPaths.WEIGHT_ELEMENT_TYPE): weight_type,
        str(MVAUComputeKernelPaths.ACCUMULATOR_ELEMENT_TYPE): accumulator_type,
        str(MVAUComputeKernelPaths.OUTPUT_ELEMENT_TYPE): output_type,
        str(MVAUComputeKernelPaths.COMPUTATION_PROFILE): computation_profile,
        str(MVAUComputeKernelPaths.WEIGHT_INITIALIZER_AVAILABLE): weight_initializer_available,
    }
    if threshold_initializer_available is not _MISSING:
        problem[str(MVAUComputeKernelPaths.THRESHOLD_INITIALIZER_AVAILABLE)] = (
            threshold_initializer_available
        )
    if threshold_type is not _MISSING:
        problem[str(MVAUComputeKernelPaths.THRESHOLD_ELEMENT_TYPE)] = threshold_type
    if target_dsp is not _MISSING:
        problem[str(MVAUComputeKernelPaths.TARGET_DSP_BLOCK)] = target_dsp
    if weights_narrow is not _MISSING:
        problem[str(MVAUComputeKernelPaths.WEIGHTS_NARROW)] = weights_narrow
    return problem


def _started(**overrides: object) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(MVAU_COMPUTE_KERNEL_SPEC)
    return engine, engine.start(space, _problem(**overrides))


def _commit_region(
    engine: Engine,
    point: DesignPoint,
    declaration: MVAURegionDeclaration | MVAUWeightInterface,
    *,
    pe: int = 2,
    simd: int = 2,
    interleave: int = 2,
) -> DesignPoint:
    assignments: dict[QualifiedPath, object] = {
        MVAUComputeKernelPaths.PE: pe,
        MVAUComputeKernelPaths.SIMD: simd,
        MVAUComputeKernelPaths.REGION_DECLARATION: declaration,
    }
    if declaration is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED:
        assignments[MVAUComputeKernelPaths.INTERLEAVE] = interleave
    return engine.commit_assignments(point, assignments).point


def _commit_binding(
    engine: Engine,
    point: DesignPoint,
    binding: MVAUComputeBinding,
    *,
    compute_pumping: bool = False,
) -> DesignPoint:
    return engine.commit_assignments(
        point,
        {
            MVAUComputeKernelPaths.BINDING: binding,
            MVAUComputeKernelPaths.COMPUTE_PUMPING: compute_pumping,
        },
    ).point


def _region(engine: Engine, point: DesignPoint) -> DataflowRegion:
    answer = engine.query_property(point, MVAUComputeKernelPaths.REGION)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, DataflowRegion)
    return answer.value


def test_paths_and_compatibility_names_remain_available() -> None:
    assert str(MVAUComputeKernelPaths.PE) == "mvau.compute.pe"
    assert str(MVAUComputeKernelPaths.REGION_DECLARATION) == "mvau.compute.weight_interface"
    assert MVAUComputeKernelPaths.WEIGHT_INTERFACE is MVAUComputeKernelPaths.REGION_DECLARATION
    assert str(MVAUDesignPaths.PE) == "mvau.pe"
    assert str(MVAUDesignPaths.REGION) == "semantic.mvau.region"
    assert MVAU_DESIGN_SPACE_SPEC is not MVAU_COMPUTE_KERNEL_SPEC


def test_legacy_spec_accepts_its_original_problem_schema_and_paths() -> None:
    engine = Engine()
    space = engine.validate(MVAU_DESIGN_SPACE_SPEC)
    point = engine.start(
        space,
        {
            str(MVAUDesignPaths.REPETITIONS): 2,
            str(MVAUDesignPaths.MATRIX_WIDTH): 4,
            str(MVAUDesignPaths.MATRIX_HEIGHT): 4,
            str(MVAUDesignPaths.ACTIVATION_ELEMENT_TYPE): INT8,
            str(MVAUDesignPaths.WEIGHT_ELEMENT_TYPE): INT8,
            str(MVAUDesignPaths.OUTPUT_ELEMENT_TYPE): INT16,
        },
    )
    point = engine.commit_assignments(point, {MVAUDesignPaths.PE: 2, MVAUDesignPaths.SIMD: 2}).point
    expected = construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    assert engine.query_property(point, MVAUDesignPaths.REGION) == Decided(expected)
    assert engine.check_readiness(point, "model_structural").ready is True


def test_standard_region_properties_are_explicit_and_preserve_old_construction() -> None:
    for declaration, expected_constructor, candidate_path in (
        (
            MVAURegionDeclaration.STANDARD_EMBEDDED,
            construct_standard_embedded_mvau_region,
            MVAUComputeKernelPaths.STANDARD_EMBEDDED_REGION,
        ),
        (
            MVAURegionDeclaration.STANDARD_STREAMED,
            construct_standard_streamed_mvau_region,
            MVAUComputeKernelPaths.STANDARD_STREAMED_REGION,
        ),
    ):
        engine, point = _started(target_dsp=_MISSING, weights_narrow=_MISSING)
        point = _commit_region(engine, point, declaration)
        expected = expected_constructor(4, 4, 4, INT8, INT8, INT16, 2, 2)
        assert _region(engine, point) == expected
        assert engine.query_property(point, candidate_path) == Decided(expected)
        assert engine.check_readiness(point, "model_structural").ready is True
        assert engine.check_readiness(point, "binding_feasibility").ready is None


def test_legacy_weight_interface_values_are_normalized_by_the_compatibility_path() -> None:
    engine, point = _started()
    point = _commit_region(engine, point, MVAUWeightInterface.STREAMED)
    assert point.assignments[MVAUComputeKernelPaths.REGION_DECLARATION] is (
        MVAURegionDeclaration.STANDARD_STREAMED
    )


def test_interleave_is_branch_specific_and_has_exact_candidates() -> None:
    engine, point = _started(repetitions=6, matrix_height=6)
    point = engine.commit_assignments(
        point,
        {
            MVAUComputeKernelPaths.REGION_DECLARATION: (
                MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
            )
        },
    ).point
    assert isinstance(
        engine.enumerate_candidates(point, MVAUComputeKernelPaths.INTERLEAVE), Unresolved
    )
    point = engine.commit_assignments(
        point,
        {MVAUComputeKernelPaths.PE: 3, MVAUComputeKernelPaths.SIMD: 2},
    ).point
    assert engine.enumerate_candidates(point, MVAUComputeKernelPaths.INTERLEAVE) == Decided(
        (2, 3, 6)
    )

    standard_engine, standard = _started()
    standard = _commit_region(standard_engine, standard, MVAURegionDeclaration.STANDARD_STREAMED)
    assert isinstance(
        standard_engine.decision_state(standard, MVAUComputeKernelPaths.INTERLEAVE), Absent
    )


def test_invalid_interleave_reports_each_failed_domain_condition() -> None:
    engine, point = _started(repetitions=5, matrix_height=4)
    result = engine.commit_assignments(
        point,
        {
            MVAUComputeKernelPaths.PE: 2,
            MVAUComputeKernelPaths.SIMD: 2,
            MVAUComputeKernelPaths.REGION_DECLARATION: (
                MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
            ),
            MVAUComputeKernelPaths.INTERLEAVE: 3,
        },
    )
    outcome = next(
        item for item in result.outcomes if item.path == MVAUComputeKernelPaths.INTERLEAVE
    )
    assert outcome.disposition == "rejected"
    assert {finding.code for finding in outcome.findings} == {
        "mvau-interleave-does-not-divide-repetitions",
        "mvau-interleave-does-not-divide-weight-tile",
    }


def test_batch_interleaved_region_is_selected_and_structurally_ready_without_binding() -> None:
    engine, point = _started(repetitions=6, matrix_height=6, target_dsp=_MISSING)
    point = _commit_region(
        engine,
        point,
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
        pe=3,
        simd=2,
        interleave=3,
    )
    expected = construct_batch_interleaved_streamed_mvau_region(6, 4, 6, INT8, INT8, INT16, 3, 2, 3)
    assert _region(engine, point) == expected
    assert engine.query_property(point, MVAUComputeKernelPaths.REGION_VALIDATION) == Decided(
        RegionValidationReport()
    )
    assert engine.check_readiness(point, "model_structural").ready is True


def test_region_declarations_do_not_depend_on_binding_or_target_facts() -> None:
    forbidden = {
        MVAUComputeKernelPaths.BINDING,
        MVAUComputeKernelPaths.COMPUTE_PUMPING,
        MVAUComputeKernelPaths.TARGET_DSP_BLOCK,
        MVAUComputeKernelPaths.WEIGHTS_NARROW,
    }
    region_paths = {
        MVAUComputeKernelPaths.STANDARD_EMBEDDED_REGION,
        MVAUComputeKernelPaths.STANDARD_STREAMED_REGION,
        MVAUComputeKernelPaths.BATCH_INTERLEAVED_REGION,
        MVAUComputeKernelPaths.REGION,
    }
    for declaration in MVAU_COMPUTE_KERNEL_SPEC.properties:
        if declaration.path not in region_paths:
            continue
        dependencies = {dependency.path for dependency in declaration.evaluator.dependencies}
        assert dependencies.isdisjoint(forbidden)


def test_tiled_binding_is_ready_only_after_all_binding_facts_are_present() -> None:
    engine, point = _started(repetitions=6, matrix_height=6, target_dsp=_MISSING)
    point = _commit_region(
        engine,
        point,
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
        pe=3,
        interleave=3,
    )
    point = _commit_binding(engine, point, MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58)
    assert engine.check_readiness(point, "binding_feasibility").ready is None

    resolved_engine, resolved = _started(repetitions=6, matrix_height=6)
    resolved = _commit_region(
        resolved_engine,
        resolved,
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
        pe=3,
        interleave=3,
    )
    resolved = _commit_binding(
        resolved_engine, resolved, MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58
    )
    assert resolved_engine.check_readiness(resolved, "binding_feasibility").ready is True
    assert resolved_engine.evaluate_constraint_set(resolved, "binding_feasibility").verdict is True


@pytest.mark.parametrize(
    "left,right",
    [
        (MVAUComputeBinding.LEGACY_HLS_LUT, MVAUComputeBinding.LEGACY_HLS_DSP),
        (MVAUComputeBinding.RTL_SOFTVEC, MVAUComputeBinding.RTL_PACKED),
    ],
)
def test_binding_alternatives_preserve_the_standard_region(
    left: MVAUComputeBinding, right: MVAUComputeBinding
) -> None:
    engine, root = _started()
    model = _commit_region(engine, root, MVAURegionDeclaration.STANDARD_STREAMED)
    expected = _region(engine, model)
    assert _region(engine, _commit_binding(engine, model, left)) == expected
    assert _region(engine, _commit_binding(engine, model, right)) == expected


@pytest.mark.parametrize(
    "binding,declaration,expected",
    [
        (MVAUComputeBinding.LEGACY_HLS_LUT, MVAURegionDeclaration.STANDARD_EMBEDDED, True),
        (MVAUComputeBinding.LEGACY_HLS_DSP, MVAURegionDeclaration.STANDARD_STREAMED, True),
        (MVAUComputeBinding.RTL_SOFTVEC, MVAURegionDeclaration.STANDARD_STREAMED, True),
        (MVAUComputeBinding.RTL_PACKED, MVAURegionDeclaration.STANDARD_STREAMED, True),
        (
            MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
            True,
        ),
        (MVAUComputeBinding.RTL_SOFTVEC, MVAURegionDeclaration.STANDARD_EMBEDDED, False),
        (
            MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
            MVAURegionDeclaration.STANDARD_STREAMED,
            False,
        ),
        (
            MVAUComputeBinding.LEGACY_HLS_LUT,
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
            False,
        ),
    ],
)
def test_binding_region_compatibility_matrix(
    binding: MVAUComputeBinding,
    declaration: MVAURegionDeclaration,
    expected: bool,
) -> None:
    overrides = {"repetitions": 6, "matrix_height": 6} if "interleaved" in declaration.value else {}
    engine, point = _started(**overrides)
    point = _commit_region(
        engine,
        point,
        declaration,
        pe=3 if overrides else 2,
        interleave=3,
    )
    point = _commit_binding(engine, point, binding)
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.BINDING_REGION_SUPPORTED] == Decided(expected)


@pytest.mark.parametrize(
    "binding,profile,expected",
    [
        (MVAUComputeBinding.LEGACY_HLS_LUT, MVAUComputationProfile.FUSED_THRESHOLD, True),
        (
            MVAUComputeBinding.LEGACY_HLS_DSP,
            MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR,
            True,
        ),
        (MVAUComputeBinding.RTL_SOFTVEC, MVAUComputationProfile.FUSED_THRESHOLD, False),
        (
            MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
            MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR,
            False,
        ),
    ],
)
def test_computation_compatibility_is_separate_from_region_structure(
    binding: MVAUComputeBinding,
    profile: MVAUComputationProfile,
    expected: bool,
) -> None:
    activation_type = INT8
    weight_type = INT8
    if profile is MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR:
        activation_type = NumericElementType("bipolar", 1)
        weight_type = NumericElementType("bipolar", 1)
    engine, point = _started(
        computation_profile=profile,
        activation_type=activation_type,
        weight_type=weight_type,
        threshold_type=INT16,
    )
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(engine, point, binding)
    assert engine.evaluate_constraint_set(point, "model_structural").verdict is True
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.BINDING_COMPUTATION_SUPPORTED] == Decided(
        expected
    )


def test_tiled_binding_constraints_are_binding_only_and_report_all_failures() -> None:
    engine, point = _started(
        activation_type=INT9,
        weight_type=INT9,
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD,
        threshold_type=INT16,
        target_dsp=MVAUDspBlock.DSP48E2,
    )
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(
        engine,
        point,
        MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
        compute_pumping=True,
    )
    assert engine.evaluate_constraint_set(point, "model_structural").verdict is True
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    false_paths = {path for path, answer in assessment.answers.items() if answer == Decided(False)}
    assert {
        MVAUComputeKernelPaths.BINDING_REGION_SUPPORTED,
        MVAUComputeKernelPaths.BINDING_COMPUTATION_SUPPORTED,
        MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED,
        MVAUComputeKernelPaths.BINDING_TILED_WIDTH_SUPPORTED,
    } <= false_paths
    assert isinstance(engine.decision_state(point, MVAUComputeKernelPaths.COMPUTE_PUMPING), Absent)
    assert isinstance(
        assessment.answers[MVAUComputeKernelPaths.BINDING_COMPUTE_PUMPING_SUPPORTED], Absent
    )
    assert assessment.verdict is False


def test_missing_target_blocks_only_applicable_binding_constraints() -> None:
    engine, point = _started(target_dsp=_MISSING, weights_narrow=_MISSING)
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(engine, point, MVAUComputeBinding.RTL_SOFTVEC)
    assert engine.check_readiness(point, "model_structural").ready is True
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    target = assessment.answers[MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED]
    assert isinstance(target, Unresolved)
    assert target.findings[0].kind is FindingKind.LIMITATION
    assert engine.check_readiness(point, "binding_feasibility").ready is None

    hls_engine, hls_point = _started(target_dsp=_MISSING, weights_narrow=_MISSING)
    hls_point = _commit_region(hls_engine, hls_point, MVAURegionDeclaration.STANDARD_STREAMED)
    hls_point = _commit_binding(hls_engine, hls_point, MVAUComputeBinding.LEGACY_HLS_LUT)
    hls_assessment = hls_engine.evaluate_constraint_set(hls_point, "binding_feasibility")
    assert isinstance(
        hls_assessment.answers[MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED], Absent
    )
    assert hls_assessment.verdict is True


def test_packed_binding_preserves_num_lanes_limit() -> None:
    engine, point = _started(activation_type=INT2, weight_type=INT2)
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(engine, point, MVAUComputeBinding.RTL_PACKED)
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.BINDING_PACKED_SUPPORTED] == Decided(False)


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
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(engine, point, MVAUComputeBinding.RTL_SOFTVEC)
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.BINDING_NUMERIC_SUPPORTED] == Decided(False)


def test_fused_threshold_requires_compatible_threshold_representation() -> None:
    engine, point = _started(
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD,
        threshold_type=_MISSING,
    )
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_EMBEDDED)
    point = _commit_binding(engine, point, MVAUComputeBinding.LEGACY_HLS_LUT)
    answer = engine.evaluate_constraint_set(point, "binding_feasibility").answers[
        MVAUComputeKernelPaths.BINDING_HLS_THRESHOLD_SUPPORTED
    ]
    assert isinstance(answer, Unresolved)

    wide_engine, wide_point = _started(
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD,
        threshold_type=INT16,
    )
    wide_point = _commit_region(wide_engine, wide_point, MVAURegionDeclaration.STANDARD_EMBEDDED)
    wide_point = _commit_binding(wide_engine, wide_point, MVAUComputeBinding.LEGACY_HLS_LUT)
    assert wide_engine.evaluate_constraint_set(wide_point, "binding_feasibility").verdict is True

    missing_init_engine, missing_init = _started(
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD,
        threshold_type=INT16,
        threshold_initializer_available=_MISSING,
    )
    missing_init = _commit_region(
        missing_init_engine, missing_init, MVAURegionDeclaration.STANDARD_EMBEDDED
    )
    missing_init = _commit_binding(
        missing_init_engine, missing_init, MVAUComputeBinding.LEGACY_HLS_LUT
    )
    missing_answer = missing_init_engine.evaluate_constraint_set(
        missing_init, "binding_feasibility"
    ).answers[MVAUComputeKernelPaths.FUSED_THRESHOLD_SOURCE_SUPPORTED]
    assert isinstance(missing_answer, Unresolved)

    unavailable_engine, unavailable = _started(
        computation_profile=MVAUComputationProfile.FUSED_THRESHOLD,
        threshold_type=INT16,
        threshold_initializer_available=False,
    )
    unavailable = _commit_region(
        unavailable_engine, unavailable, MVAURegionDeclaration.STANDARD_EMBEDDED
    )
    unavailable = _commit_binding(
        unavailable_engine, unavailable, MVAUComputeBinding.LEGACY_HLS_LUT
    )
    assert unavailable_engine.evaluate_constraint_set(unavailable, "binding_feasibility").answers[
        MVAUComputeKernelPaths.FUSED_THRESHOLD_SOURCE_SUPPORTED
    ] == Decided(False)


def test_accumulator_output_profile_requires_equal_accumulator_and_output_types() -> None:
    engine, point = _started(accumulator_type=INT8, output_type=INT16)
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(engine, point, MVAUComputeBinding.LEGACY_HLS_LUT)
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.ACCUMULATOR_OUTPUT_TYPE_SUPPORTED] == Decided(
        False
    )
    assert assessment.verdict is False


@pytest.mark.parametrize(
    "target,activation_width,weight_width,accumulator_width",
    [
        (MVAUDspBlock.DSP58, 8, 30, 58),
        (MVAUDspBlock.DSP48E1, 18, 26, 48),
        (MVAUDspBlock.DSP48E2, 19, 27, 48),
        (MVAUDspBlock.DSP58, 25, 27, 58),
        (MVAUDspBlock.DSP58, 24, 28, 58),
        (MVAUDspBlock.DSP58, 24, 27, 59),
    ],
)
def test_rtl_width_envelope_rejects_datapath_overflow(
    target: MVAUDspBlock,
    activation_width: int,
    weight_width: int,
    accumulator_width: int,
) -> None:
    accumulator = NumericElementType("int", accumulator_width)
    engine, point = _started(
        activation_type=NumericElementType("int", activation_width),
        weight_type=NumericElementType("int", weight_width),
        accumulator_type=accumulator,
        output_type=accumulator,
        target_dsp=target,
    )
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(engine, point, MVAUComputeBinding.RTL_SOFTVEC)
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.BINDING_RTL_WIDTH_SUPPORTED] == Decided(False)


@pytest.mark.parametrize(
    "target,activation_width,weight_width,accumulator_width",
    [
        (MVAUDspBlock.DSP48E1, 18, 25, 48),
        (MVAUDspBlock.DSP48E2, 18, 27, 48),
        (MVAUDspBlock.DSP58, 24, 27, 58),
    ],
)
def test_rtl_width_envelope_accepts_target_datapath_boundaries(
    target: MVAUDspBlock,
    activation_width: int,
    weight_width: int,
    accumulator_width: int,
) -> None:
    accumulator = NumericElementType("int", accumulator_width)
    engine, point = _started(
        activation_type=NumericElementType("int", activation_width),
        weight_type=NumericElementType("int", weight_width),
        accumulator_type=accumulator,
        output_type=accumulator,
        target_dsp=target,
    )
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = _commit_binding(engine, point, MVAUComputeBinding.RTL_SOFTVEC)
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.BINDING_RTL_WIDTH_SUPPORTED] == Decided(True)


def test_irrelevant_compute_pumping_is_absent_and_does_not_block_hls_readiness() -> None:
    engine, point = _started()
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED)
    point = engine.commit_assignments(
        point, {MVAUComputeKernelPaths.BINDING: MVAUComputeBinding.LEGACY_HLS_LUT}
    ).point
    assert isinstance(engine.decision_state(point, MVAUComputeKernelPaths.COMPUTE_PUMPING), Absent)
    assert engine.check_readiness(point, "binding_feasibility").ready is True


def test_binding_selection_is_separate_from_feasibility_and_region() -> None:
    engine, point = _started(repetitions=6, matrix_height=6)
    point = _commit_region(
        engine,
        point,
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
        pe=3,
        interleave=3,
    )
    region = _region(engine, point)
    point = _commit_binding(engine, point, MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58)
    selection = engine.query_property(point, MVAUComputeKernelPaths.BINDING_SELECTION)
    assert isinstance(selection, Decided)
    assert isinstance(selection.value, MVAUBindingSelection)
    assert selection.value.binding_id == "rtl_batch_interleaved_dsp58"
    assert selection.value.compute_pumping is None
    assert _region(engine, point) == region


def test_hls_partition_limit_is_a_binding_constraint() -> None:
    engine, point = _started(matrix_width=2048)
    point = _commit_region(engine, point, MVAURegionDeclaration.STANDARD_STREAMED, simd=1)
    point = _commit_binding(engine, point, MVAUComputeBinding.LEGACY_HLS_DSP)
    assessment = engine.evaluate_constraint_set(point, "binding_feasibility")
    assert assessment.answers[MVAUComputeKernelPaths.BINDING_HLS_PARTITION_SUPPORTED] == Decided(
        False
    )
    assert engine.evaluate_constraint_set(point, "model_structural").verdict is True


def test_wrong_nominal_assignment_types_are_request_errors() -> None:
    engine, point = _started()
    with pytest.raises(RequestError) as caught:
        engine.commit_assignments(point, {MVAUComputeKernelPaths.BINDING: "rtl_softvec"})
    assert caught.value.findings[0].code == "assignment-type"


def test_mvau_assignments_pass_the_monotonicity_harness() -> None:
    engine, point = _started()
    result = MonotonicityHarness(engine).verify(
        point,
        {
            MVAUComputeKernelPaths.PE: (1, 2, 4),
            MVAUComputeKernelPaths.SIMD: (1, 2, 4),
            MVAUComputeKernelPaths.REGION_DECLARATION: tuple(MVAURegionDeclaration),
            MVAUComputeKernelPaths.INTERLEAVE: (2, 4),
            MVAUComputeKernelPaths.BINDING: tuple(MVAUComputeBinding),
            MVAUComputeKernelPaths.COMPUTE_PUMPING: (False, True),
        },
    )
    assert result.violations == ()
    assert result.conformant is True
