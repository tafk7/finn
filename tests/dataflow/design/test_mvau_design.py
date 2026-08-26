# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

import finn.dataflow.mvau_design as mvau_design
from finn.dataflow._engine.conformance import MonotonicityHarness
from finn.dataflow.design import (
    Decided,
    DecisionState,
    DesignPoint,
    Engine,
    EvaluationError,
    RegionValidationReport,
    RequestError,
    Unresolved,
    validate_region,
)
from finn.dataflow.mvau_design import (
    MVAU_DESIGN_SPACE_SPEC,
    MVAUDesignPaths,
    construct_streamed_weight_mvau_region,
)
from finn.dataflow.region import DataflowRegion, NumericElementType

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


def _problem(
    *,
    repetitions: object = 2,
    matrix_width: object = 4,
    matrix_height: object = 4,
    activation_type: object = INT8,
    weight_type: object = INT8,
    output_type: object = INT16,
) -> dict[str, object]:
    return {
        str(MVAUDesignPaths.REPETITIONS): repetitions,
        str(MVAUDesignPaths.MATRIX_WIDTH): matrix_width,
        str(MVAUDesignPaths.MATRIX_HEIGHT): matrix_height,
        str(MVAUDesignPaths.ACTIVATION_ELEMENT_TYPE): activation_type,
        str(MVAUDesignPaths.WEIGHT_ELEMENT_TYPE): weight_type,
        str(MVAUDesignPaths.OUTPUT_ELEMENT_TYPE): output_type,
    }


def _started() -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(MVAU_DESIGN_SPACE_SPEC)
    return engine, engine.start(space, _problem())


def test_partial_point_progression_blocks_until_pe_and_simd_are_committed() -> None:
    engine, point = _started()

    pe_state = engine.decision_state(point, MVAUDesignPaths.PE)
    simd_state = engine.decision_state(point, MVAUDesignPaths.SIMD)
    assert pe_state == Decided(DecisionState(MVAUDesignPaths.PE, "unassigned"))
    assert simd_state == Decided(DecisionState(MVAUDesignPaths.SIMD, "unassigned"))
    assert isinstance(engine.query_property(point, MVAUDesignPaths.REGION), Unresolved)
    assert engine.check_readiness(point, "model_structural").ready is None

    with_pe = engine.commit_assignments(point, {MVAUDesignPaths.PE: 2})
    assert isinstance(engine.query_property(with_pe.point, MVAUDesignPaths.REGION), Unresolved)

    complete = engine.commit_assignments(with_pe.point, {MVAUDesignPaths.SIMD: 2})
    region_answer = engine.query_property(complete.point, MVAUDesignPaths.REGION)
    report_answer = engine.query_property(complete.point, MVAUDesignPaths.REGION_VALIDATION)
    assessment = engine.evaluate_constraint_set(complete.point, "model_structural")

    assert isinstance(region_answer, Decided)
    assert isinstance(region_answer.value, DataflowRegion)
    assert report_answer == Decided(RegionValidationReport())
    assert assessment.answers[MVAUDesignPaths.REGION_STRUCTURALLY_WELL_FORMED] == Decided(True)
    assert assessment.verdict is True
    assert engine.check_readiness(complete.point, "model_structural").ready is True


def test_no_region_evaluator_runs_before_all_declared_dependencies_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = mvau_design.construct_streamed_weight_mvau_region

    def counted(
        repetitions: int,
        matrix_width: int,
        matrix_height: int,
        activation_element_type: NumericElementType,
        weight_element_type: NumericElementType,
        output_element_type: NumericElementType,
        pe: int,
        simd: int,
    ) -> DataflowRegion:
        nonlocal calls
        calls += 1
        return original(
            repetitions,
            matrix_width,
            matrix_height,
            activation_element_type,
            weight_element_type,
            output_element_type,
            pe,
            simd,
        )

    monkeypatch.setattr(mvau_design, "construct_streamed_weight_mvau_region", counted)
    engine, point = _started()
    assert isinstance(engine.query_property(point, MVAUDesignPaths.REGION), Unresolved)
    with_pe = engine.commit_assignments(point, {MVAUDesignPaths.PE: 2}).point
    assert isinstance(engine.query_property(with_pe, MVAUDesignPaths.REGION), Unresolved)
    assert calls == 0
    complete = engine.commit_assignments(with_pe, {MVAUDesignPaths.SIMD: 2}).point
    assert isinstance(engine.query_property(complete, MVAUDesignPaths.REGION), Decided)
    assert calls == 1


def test_divisor_candidates_and_domain_rejections_are_exact() -> None:
    engine, point = _started()
    assert engine.enumerate_candidates(point, MVAUDesignPaths.PE) == Decided((1, 2, 4))
    assert engine.enumerate_candidates(point, MVAUDesignPaths.SIMD) == Decided((1, 2, 4))

    for path, invalid in (
        (MVAUDesignPaths.PE, 0),
        (MVAUDesignPaths.PE, -1),
        (MVAUDesignPaths.PE, 3),
        (MVAUDesignPaths.PE, 2.0),
        (MVAUDesignPaths.SIMD, 0),
        (MVAUDesignPaths.SIMD, -1),
        (MVAUDesignPaths.SIMD, 3),
        (MVAUDesignPaths.SIMD, 2.0),
    ):
        result = engine.commit_assignments(point, {path: invalid})
        assert result.point is point
        assert result.outcomes[0].disposition == "rejected"
    assert point.assignments == {}


def test_independent_valid_assignments_commit_in_one_batch() -> None:
    engine, point = _started()
    result = engine.commit_assignments(
        point,
        {MVAUDesignPaths.PE: 2, MVAUDesignPaths.SIMD: 4},
    )
    assert dict(result.point.assignments) == {
        MVAUDesignPaths.PE: 2,
        MVAUDesignPaths.SIMD: 4,
    }
    assert {outcome.disposition for outcome in result.outcomes} == {"committed"}


@pytest.mark.parametrize(
    "problem",
    [
        _problem(repetitions=0),
        _problem(matrix_width=-1),
        _problem(matrix_height=2.0),
        _problem(activation_type=NumericElementType("", 8)),
        _problem(weight_type=NumericElementType("int", 0)),
        {
            key: value
            for key, value in _problem().items()
            if key != str(MVAUDesignPaths.OUTPUT_ELEMENT_TYPE)
        },
    ],
)
def test_malformed_problem_input_is_a_request_error(problem: dict[str, object]) -> None:
    engine = Engine()
    space = engine.validate(MVAU_DESIGN_SPACE_SPEC)
    with pytest.raises(RequestError):
        engine.start(space, problem)


def test_evaluator_exception_remains_contextual(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(
        _repetitions: int,
        _matrix_width: int,
        _matrix_height: int,
        _activation_element_type: NumericElementType,
        _weight_element_type: NumericElementType,
        _output_element_type: NumericElementType,
        _pe: int,
        _simd: int,
    ) -> DataflowRegion:
        raise RuntimeError("boom")

    monkeypatch.setattr(mvau_design, "construct_streamed_weight_mvau_region", broken)
    engine, point = _started()
    complete = engine.commit_assignments(
        point,
        {MVAUDesignPaths.PE: 2, MVAUDesignPaths.SIMD: 2},
    ).point
    with pytest.raises(EvaluationError) as caught:
        engine.query_property(complete, MVAUDesignPaths.REGION)
    assert caught.value.owner == MVAUDesignPaths.REGION
    assert caught.value.role == "property"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_queries_never_commit_decisions_or_proposals() -> None:
    engine, point = _started()
    engine.enumerate_candidates(point, MVAUDesignPaths.PE)
    engine.query_property(point, MVAUDesignPaths.REGION)
    engine.evaluate_constraints(point)
    engine.check_readiness(point, "model_structural")
    assert point.assignments == {}


def test_derived_region_equals_direct_construction_and_changes_with_folding() -> None:
    engine, point = _started()
    first = engine.commit_assignments(
        point,
        {MVAUDesignPaths.PE: 2, MVAUDesignPaths.SIMD: 2},
    ).point
    first_answer = engine.query_property(first, MVAUDesignPaths.REGION)
    assert isinstance(first_answer, Decided)
    assert isinstance(first_answer.value, DataflowRegion)
    direct = construct_streamed_weight_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    assert first_answer.value == direct
    assert validate_region(first_answer.value) == validate_region(direct)

    second = engine.commit_assignments(
        point,
        {MVAUDesignPaths.PE: 1, MVAUDesignPaths.SIMD: 2},
    ).point
    second_answer = engine.query_property(second, MVAUDesignPaths.REGION)
    assert isinstance(second_answer, Decided)
    assert isinstance(second_answer.value, DataflowRegion)
    assert second_answer.value != first_answer.value
    assert second_answer.value.schedule.extents != first_answer.value.schedule.extents
    assert (
        second_answer.value.input_interface("weight").port.beat_sequence.elements_per_beat
        != first_answer.value.input_interface("weight").port.beat_sequence.elements_per_beat
    )


def test_validation_is_derived_once_and_reused_by_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = mvau_design.validate_region  # type: ignore[attr-defined]

    def counted(region: DataflowRegion) -> RegionValidationReport:
        nonlocal calls
        calls += 1
        return original(region)

    monkeypatch.setattr(mvau_design, "validate_region", counted)
    engine, point = _started()
    complete = engine.commit_assignments(
        point,
        {MVAUDesignPaths.PE: 2, MVAUDesignPaths.SIMD: 2},
    ).point
    assert engine.evaluate_constraint_set(complete, "model_structural").verdict is True
    assert isinstance(
        engine.query_property(complete, MVAUDesignPaths.REGION_VALIDATION),
        Decided,
    )
    assert calls == 1


def test_mvau_assignments_pass_the_monotonicity_harness() -> None:
    engine, point = _started()
    result = MonotonicityHarness(engine).verify(
        point,
        {
            MVAUDesignPaths.PE: (1, 2, 4),
            MVAUDesignPaths.SIMD: (1, 2, 4),
        },
    )
    assert result.checked_successors == 6
    assert result.violations == ()
    assert result.conformant is True
