# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from dataclasses import FrozenInstanceError

from finn.dataflow._engine import (
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    QualifiedPath,
    ReadinessProfile,
    as_object_semantics,
)
from finn.dataflow.space.dataflow_value_semantics import (
    DATAFLOW_REGION_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS,
)
from finn.dataflow.model.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    Coordinate,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.model.region_validation import (
    RegionValidationReport,
    validate_region,
)
from finn.dataflow.ops.mvau.regions import construct_activation_replay_region
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]


class DataflowRegionSubclass(DataflowRegion):
    pass


class RegionValidationReportSubclass(RegionValidationReport):
    pass


def _region(extent: int = 1) -> DataflowRegion:
    return DataflowRegion(LogicalSchedule((ScheduleLevel("step", extent),)), (), ())


def test_region_value_semantics_are_exact_immutable_and_model_equal() -> None:
    left = _region()
    right = _region()
    different = _region(2)

    assert DATAFLOW_REGION_SEMANTICS.accepts(left)
    assert not DATAFLOW_REGION_SEMANTICS.accepts(object())
    assert not DATAFLOW_REGION_SEMANTICS.accepts(
        DataflowRegionSubclass(left.schedule, left.inputs, left.outputs)
    )
    assert DATAFLOW_REGION_SEMANTICS.values_equal(left, right)
    assert not DATAFLOW_REGION_SEMANTICS.values_equal(left, different)
    assert DATAFLOW_REGION_SEMANTICS.freeze(left) is left
    assert validate_region(left) == validate_region(right)
    with pytest.raises(FrozenInstanceError):
        setattr(left, "schedule", LogicalSchedule(()))


def test_explicit_and_compact_replay_regions_are_engine_equal() -> None:
    compact = construct_activation_replay_region(1, 2, 2, DataType["INT8"], 1, 1)
    schedule = LogicalSchedule(
        (ScheduleLevel("rep", 1), ScheduleLevel("nf", 2), ScheduleLevel("sf", 2))
    )
    operand = Operand("X", DataType["INT8"], (1, 2))
    expanded_operand = Operand("XR", DataType["INT8"], (2, 2))
    requirements: dict[RequirementKey, int] = {
        ((0, neuron_fold, synapse_fold), (0, synapse_fold)): 1
        for neuron_fold in range(2)
        for synapse_fold in range(2)
    }
    availability: dict[Coordinate, Coordinate] = {
        (0, 0): (0, 0, 0),
        (0, 1): (0, 0, 1),
        (1, 0): (0, 1, 0),
        (1, 1): (0, 1, 1),
    }
    explicit = DataflowRegion(
        schedule,
        (
            InputInterface(
                Port(
                    "activation_in",
                    operand,
                    BeatSequence(1, (((0, 0),), ((0, 1),))),
                ),
                ScheduledInputRequirements(requirements),
            ),
        ),
        (
            OutputInterface(
                Port(
                    "activation_out",
                    expanded_operand,
                    BeatSequence(
                        1,
                        (
                            ((0, 0),),
                            ((0, 1),),
                            ((1, 0),),
                            ((1, 1),),
                        ),
                    ),
                ),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )

    assert explicit == compact
    assert hash(explicit) == hash(compact)
    assert DATAFLOW_REGION_SEMANTICS.values_equal(explicit, compact)
    assert DATAFLOW_REGION_SEMANTICS.freeze(compact) is compact


def test_validation_report_semantics_are_nominal_and_immutable() -> None:
    report = validate_region(_region(0))
    twin = RegionValidationReport(report.issues)

    assert report
    assert REGION_VALIDATION_REPORT_SEMANTICS.accepts(report)
    assert not REGION_VALIDATION_REPORT_SEMANTICS.accepts(report.issues)
    assert not REGION_VALIDATION_REPORT_SEMANTICS.accepts(
        RegionValidationReportSubclass(report.issues)
    )
    assert REGION_VALIDATION_REPORT_SEMANTICS.values_equal(report, twin)
    assert REGION_VALIDATION_REPORT_SEMANTICS.freeze(report) is report
    with pytest.raises(FrozenInstanceError):
        setattr(report, "issues", ())


def _synthetic_region_spec(
    region: DataflowRegion,
    validation_calls: list[DataflowRegion],
) -> DesignSpaceSpec:
    region_path = QualifiedPath("semantic.region")
    report_path = QualifiedPath("semantic.region_validation")
    constraint_path = QualifiedPath("constraint.region_structurally_well_formed")
    region_semantics = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
    report_semantics = as_object_semantics(REGION_VALIDATION_REPORT_SEMANTICS)
    region_dependency = DependencyRef.property("region", region_path, region_semantics)
    report_dependency = DependencyRef.property("report", report_path, report_semantics)

    def derive_report(dependencies: DependencyView) -> Answer[object]:
        resolved = dependencies["region"]
        assert isinstance(resolved, DataflowRegion)
        validation_calls.append(resolved)
        return Decided(validate_region(resolved))

    def assess(dependencies: DependencyView) -> Answer[bool]:
        report = dependencies["report"]
        assert isinstance(report, RegionValidationReport)
        return Decided(not report)

    return DesignSpaceSpec(
        properties=(
            DerivedProperty(
                region_path,
                region_semantics,
                EvaluatorSpec((), lambda _dependencies: Decided(region)),
            ),
            DerivedProperty(
                report_path,
                report_semantics,
                EvaluatorSpec((region_dependency,), derive_report),
            ),
        ),
        constraints=(
            Constraint(
                constraint_path,
                EvaluatorSpec((report_dependency,), assess),
            ),
        ),
        constraint_sets=(ConstraintSet("structural", (constraint_path,)),),
        readiness_profiles=(
            ReadinessProfile(
                "structural",
                properties=(region_path, report_path),
                constraints=(constraint_path,),
            ),
        ),
    )


@pytest.mark.parametrize("extent,expected", [(1, True), (0, False)])
def test_structural_report_is_retained_and_reused_by_constraint(
    extent: int,
    expected: bool,
) -> None:
    calls: list[DataflowRegion] = []
    engine = Engine()
    point = engine.start(engine.validate(_synthetic_region_spec(_region(extent), calls)), {})

    assessment = engine.evaluate_constraint_set(point, "structural")
    report_answer = engine.query_property(point, "semantic.region_validation")

    assert assessment.verdict is expected
    assert assessment.answers[
        QualifiedPath("constraint.region_structurally_well_formed")
    ] == Decided(expected)
    assert isinstance(report_answer, Decided)
    assert isinstance(report_answer.value, RegionValidationReport)
    assert bool(report_answer.value) is (not expected)
    assert len(calls) == 1
    assert engine.check_readiness(point, "structural").ready is True
