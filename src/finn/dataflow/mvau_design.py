# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Static dataflow design-space specification for a streamed-weight MVAU."""

from __future__ import annotations

from typing import cast

from finn.dataflow.design import (
    DATAFLOW_REGION_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    ValueSemantics,
    as_object_semantics,
    validate_region,
)
from finn.dataflow.region import (
    BeatSequence,
    Coordinate,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.region_validation import RegionValidationReport


class MVAUDesignPaths:
    """Stable paths used by the MVAU problem, decisions, and derived values."""

    REPETITIONS = QualifiedPath("problem.mvau.r")
    MATRIX_WIDTH = QualifiedPath("problem.mvau.mw")
    MATRIX_HEIGHT = QualifiedPath("problem.mvau.mh")
    ACTIVATION_ELEMENT_TYPE = QualifiedPath("problem.mvau.activation_element_type")
    WEIGHT_ELEMENT_TYPE = QualifiedPath("problem.mvau.weight_element_type")
    OUTPUT_ELEMENT_TYPE = QualifiedPath("problem.mvau.output_element_type")

    PE = QualifiedPath("mvau.pe")
    SIMD = QualifiedPath("mvau.simd")

    REGION = QualifiedPath("semantic.mvau.region")
    REGION_VALIDATION = QualifiedPath("semantic.mvau.region_validation")
    REGION_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.region_structurally_well_formed"
    )


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _complete_numeric_element_type(value: object) -> bool:
    return type(value) is NumericElementType and bool(value.type_id) and value.bit_width > 0


def construct_streamed_weight_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Construct the concrete logical region for one folded streamed-weight MVAU."""

    dimensions = {
        "repetitions": repetitions,
        "matrix_width": matrix_width,
        "matrix_height": matrix_height,
        "pe": pe,
        "simd": simd,
    }
    for dimension_name, dimension_value in dimensions.items():
        if not _positive_integer(dimension_value):
            raise ValueError(f"{dimension_name} must be a positive integer")
    element_types = {
        "activation_element_type": activation_element_type,
        "weight_element_type": weight_element_type,
        "output_element_type": output_element_type,
    }
    for type_name, element_type in element_types.items():
        if not _complete_numeric_element_type(element_type):
            raise ValueError(f"{type_name} must be a complete numeric element type")
    if matrix_width % simd:
        raise ValueError("SIMD must divide matrix_width exactly")
    if matrix_height % pe:
        raise ValueError("PE must divide matrix_height exactly")

    synapse_folds = matrix_width // simd
    neuron_folds = matrix_height // pe
    schedule = LogicalSchedule(
        (
            ScheduleLevel("rep", repetitions),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
        )
    )

    activation = Operand(
        "X",
        activation_element_type,
        (repetitions, matrix_width),
    )
    weight = Operand("W", weight_element_type, (matrix_height, matrix_width))
    output = Operand("Y", output_element_type, (repetitions, matrix_height))

    activation_requirements: dict[RequirementKey, int] = {}
    weight_requirements: dict[RequirementKey, int] = {}
    output_availability: dict[Coordinate, Coordinate] = {}
    for repetition in range(repetitions):
        for neuron_fold in range(neuron_folds):
            for synapse_fold in range(synapse_folds):
                iteration = (repetition, neuron_fold, synapse_fold)
                for lane in range(simd):
                    activation_requirements[
                        (iteration, (repetition, synapse_fold * simd + lane))
                    ] = 1
                for pe_index in range(pe):
                    for lane in range(simd):
                        weight_requirements[
                            (
                                iteration,
                                (
                                    neuron_fold * pe + pe_index,
                                    synapse_fold * simd + lane,
                                ),
                            )
                        ] = 1
            for pe_index in range(pe):
                output_availability[(repetition, neuron_fold * pe + pe_index)] = (
                    repetition,
                    neuron_fold,
                    synapse_folds - 1,
                )

    activation_beats = tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for synapse_fold in range(synapse_folds)
    )
    weight_field_order = tuple((pe_index, lane) for pe_index in range(pe) for lane in range(simd))
    weight_beats = tuple(
        tuple(
            (
                neuron_fold * pe + pe_index,
                synapse_fold * simd + lane,
            )
            for pe_index, lane in weight_field_order
        )
        for _repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )
    output_field_order = tuple(range(pe))
    output_beats = tuple(
        tuple((repetition, neuron_fold * pe + pe_index) for pe_index in output_field_order)
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
    )

    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port(
                    "activation",
                    activation,
                    BeatSequence(simd, activation_beats),
                ),
                ScheduledInputRequirements(activation_requirements),
            ),
            InputInterface(
                Port(
                    "weight",
                    weight,
                    BeatSequence(pe * simd, weight_beats),
                ),
                ScheduledInputRequirements(weight_requirements),
            ),
        ),
        (
            OutputInterface(
                Port("output", output, BeatSequence(pe, output_beats)),
                ScheduledOutputAvailability(output_availability),
            ),
        ),
    )


_INTEGER_SEMANTICS = ValueSemantics.immutable_nominal(int, name="integer")
_DECISION_INTEGER_SEMANTICS: ValueSemantics[object] = ValueSemantics(
    int,
    "integer decision candidate",
    lambda _value: True,
    lambda left, right: type(left) is int and type(right) is int and left == right,
    lambda value: value,
)
_ELEMENT_TYPE_SEMANTICS = ValueSemantics.immutable_nominal(
    NumericElementType,
    name="NumericElementType",
)
_INTEGER_OBJECT_SEMANTICS = as_object_semantics(_INTEGER_SEMANTICS)
_DECISION_INTEGER_OBJECT_SEMANTICS = as_object_semantics(_DECISION_INTEGER_SEMANTICS)
_ELEMENT_TYPE_OBJECT_SEMANTICS = as_object_semantics(_ELEMENT_TYPE_SEMANTICS)
_REGION_OBJECT_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_REPORT_OBJECT_SEMANTICS = as_object_semantics(REGION_VALIDATION_REPORT_SEMANTICS)


def _divisor_domain(dimension: QualifiedPath) -> DecisionDomain:
    dependency = DependencyRef.problem(
        "dimension",
        dimension,
        _INTEGER_OBJECT_SEMANTICS,
    )

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        extent = cast(int, dependencies["dimension"])
        return Decided(type(value) is int and value > 0 and extent % value == 0)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        extent = cast(int, dependencies["dimension"])
        return Decided(tuple(value for value in range(1, extent + 1) if extent % value == 0))

    return DecisionDomain(
        (dependency,),
        accepts,
        EvaluatorSpec((dependency,), candidates),
    )


_REGION_DEPENDENCIES = (
    DependencyRef.problem("repetitions", MVAUDesignPaths.REPETITIONS, _INTEGER_OBJECT_SEMANTICS),
    DependencyRef.problem("matrix_width", MVAUDesignPaths.MATRIX_WIDTH, _INTEGER_OBJECT_SEMANTICS),
    DependencyRef.problem(
        "matrix_height", MVAUDesignPaths.MATRIX_HEIGHT, _INTEGER_OBJECT_SEMANTICS
    ),
    DependencyRef.problem(
        "activation_element_type",
        MVAUDesignPaths.ACTIVATION_ELEMENT_TYPE,
        _ELEMENT_TYPE_OBJECT_SEMANTICS,
    ),
    DependencyRef.problem(
        "weight_element_type",
        MVAUDesignPaths.WEIGHT_ELEMENT_TYPE,
        _ELEMENT_TYPE_OBJECT_SEMANTICS,
    ),
    DependencyRef.problem(
        "output_element_type",
        MVAUDesignPaths.OUTPUT_ELEMENT_TYPE,
        _ELEMENT_TYPE_OBJECT_SEMANTICS,
    ),
    DependencyRef.decision(
        "pe",
        MVAUDesignPaths.PE,
        _DECISION_INTEGER_OBJECT_SEMANTICS,
    ),
    DependencyRef.decision(
        "simd",
        MVAUDesignPaths.SIMD,
        _DECISION_INTEGER_OBJECT_SEMANTICS,
    ),
)


def _derive_region(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        construct_streamed_weight_mvau_region(
            cast(int, dependencies["repetitions"]),
            cast(int, dependencies["matrix_width"]),
            cast(int, dependencies["matrix_height"]),
            cast(NumericElementType, dependencies["activation_element_type"]),
            cast(NumericElementType, dependencies["weight_element_type"]),
            cast(NumericElementType, dependencies["output_element_type"]),
            cast(int, dependencies["pe"]),
            cast(int, dependencies["simd"]),
        )
    )


_REGION_REPORT_DEPENDENCY = DependencyRef.property(
    "region",
    MVAUDesignPaths.REGION,
    _REGION_OBJECT_SEMANTICS,
)


def _derive_region_validation(dependencies: DependencyView) -> Answer[object]:
    return Decided(validate_region(cast(DataflowRegion, dependencies["region"])))


_STRUCTURAL_REPORT_DEPENDENCY = DependencyRef.property(
    "report",
    MVAUDesignPaths.REGION_VALIDATION,
    _REPORT_OBJECT_SEMANTICS,
)


def _region_is_structurally_well_formed(dependencies: DependencyView) -> Answer[bool]:
    report = cast(RegionValidationReport, dependencies["report"])
    return Decided(not report)


MVAU_DESIGN_SPACE_SPEC = DesignSpaceSpec(
    problem_schema=ProblemSchema(
        (
            ProblemField(
                MVAUDesignPaths.REPETITIONS,
                _INTEGER_OBJECT_SEMANTICS,
                constraint=_positive_integer,
                constraint_description="must be a positive integer",
            ),
            ProblemField(
                MVAUDesignPaths.MATRIX_WIDTH,
                _INTEGER_OBJECT_SEMANTICS,
                constraint=_positive_integer,
                constraint_description="must be a positive integer",
            ),
            ProblemField(
                MVAUDesignPaths.MATRIX_HEIGHT,
                _INTEGER_OBJECT_SEMANTICS,
                constraint=_positive_integer,
                constraint_description="must be a positive integer",
            ),
            ProblemField(
                MVAUDesignPaths.ACTIVATION_ELEMENT_TYPE,
                _ELEMENT_TYPE_OBJECT_SEMANTICS,
                constraint=_complete_numeric_element_type,
                constraint_description="must be a complete numeric element type",
            ),
            ProblemField(
                MVAUDesignPaths.WEIGHT_ELEMENT_TYPE,
                _ELEMENT_TYPE_OBJECT_SEMANTICS,
                constraint=_complete_numeric_element_type,
                constraint_description="must be a complete numeric element type",
            ),
            ProblemField(
                MVAUDesignPaths.OUTPUT_ELEMENT_TYPE,
                _ELEMENT_TYPE_OBJECT_SEMANTICS,
                constraint=_complete_numeric_element_type,
                constraint_description="must be a complete numeric element type",
            ),
        )
    ),
    decisions=(
        Decision(
            MVAUDesignPaths.PE,
            _DECISION_INTEGER_OBJECT_SEMANTICS,
            _divisor_domain(MVAUDesignPaths.MATRIX_HEIGHT),
        ),
        Decision(
            MVAUDesignPaths.SIMD,
            _DECISION_INTEGER_OBJECT_SEMANTICS,
            _divisor_domain(MVAUDesignPaths.MATRIX_WIDTH),
        ),
    ),
    properties=(
        DerivedProperty(
            MVAUDesignPaths.REGION,
            _REGION_OBJECT_SEMANTICS,
            EvaluatorSpec(_REGION_DEPENDENCIES, _derive_region),
        ),
        DerivedProperty(
            MVAUDesignPaths.REGION_VALIDATION,
            _REPORT_OBJECT_SEMANTICS,
            EvaluatorSpec(
                (_REGION_REPORT_DEPENDENCY,),
                _derive_region_validation,
            ),
        ),
    ),
    constraints=(
        Constraint(
            MVAUDesignPaths.REGION_STRUCTURALLY_WELL_FORMED,
            EvaluatorSpec(
                (_STRUCTURAL_REPORT_DEPENDENCY,),
                _region_is_structurally_well_formed,
            ),
        ),
    ),
    constraint_sets=(
        ConstraintSet(
            "model_structural",
            (MVAUDesignPaths.REGION_STRUCTURALLY_WELL_FORMED,),
        ),
    ),
    readiness_profiles=(
        ReadinessProfile(
            "model_structural",
            decisions=(MVAUDesignPaths.PE, MVAUDesignPaths.SIMD),
            properties=(MVAUDesignPaths.REGION, MVAUDesignPaths.REGION_VALIDATION),
            constraints=(MVAUDesignPaths.REGION_STRUCTURALLY_WELL_FORMED,),
        ),
    ),
)

__all__ = [
    "MVAU_DESIGN_SPACE_SPEC",
    "MVAUDesignPaths",
    "construct_streamed_weight_mvau_region",
]
