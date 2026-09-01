# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The original streamed-weight MVAU design space, retained as a fixture.

This is not part of the Kernel pool.  It is the first public MVAU design-space
API, kept because it is a small self-contained example of the flat engine
surface and is exercised independently of the operation assembly.
"""

from __future__ import annotations

from typing import cast

from finn.dataflow.design import (
    DATAFLOW_REGION_SEMANTICS,
    Answer,
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
)
from finn.dataflow.kernel import RegionDeclaration, build_kernel_semantic_declarations
from finn.dataflow.ops.mvau.regions import construct_standard_streamed_mvau_region
from finn.dataflow.design.region import QONNX_DATATYPE_SEMANTICS
from finn.dataflow.region import NumericElementType, is_element_type


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _complete_numeric_element_type(value: object) -> bool:
    return is_element_type(value)


_INTEGER_SEMANTICS = ValueSemantics.immutable_nominal(int, name="integer")
_ELEMENT_TYPE_SEMANTICS = QONNX_DATATYPE_SEMANTICS
_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_INTEGER_OBJECT_SEMANTICS = as_object_semantics(_INTEGER_SEMANTICS)
_ELEMENT_TYPE_OBJECT_SEMANTICS = as_object_semantics(_ELEMENT_TYPE_SEMANTICS)


def _divisor_domain(dimension: QualifiedPath) -> DecisionDomain:
    dependency = DependencyRef.problem("dimension", dimension, _INTEGER_OBJECT_SEMANTICS)

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        extent = cast(int, dependencies["dimension"])
        return Decided(type(value) is int and value > 0 and extent % value == 0)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        extent = cast(int, dependencies["dimension"])
        return Decided(tuple(value for value in range(1, extent + 1) if extent % value == 0))

    return DecisionDomain((dependency,), accepts, EvaluatorSpec((dependency,), candidates))


class MVAUDesignPaths:
    """Paths retained by the original streamed-weight compatibility spec."""

    REPETITIONS = QualifiedPath("problem.mvau.r")
    MATRIX_WIDTH = QualifiedPath("problem.mvau.mw")
    MATRIX_HEIGHT = QualifiedPath("problem.mvau.mh")
    ACTIVATION_ELEMENT_TYPE = QualifiedPath("problem.mvau.activation_element_type")
    WEIGHT_ELEMENT_TYPE = QualifiedPath("problem.mvau.weight_element_type")
    OUTPUT_ELEMENT_TYPE = QualifiedPath("problem.mvau.output_element_type")
    PE = QualifiedPath("mvau.pe")
    SIMD = QualifiedPath("mvau.simd")
    STANDARD_STREAMED_REGION = QualifiedPath("semantic.mvau.region_declarations.standard_streamed")
    REGION = QualifiedPath("semantic.mvau.region")
    REGION_VALIDATION = QualifiedPath("semantic.mvau.region_validation")
    REGION_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.region_structurally_well_formed"
    )


def build_legacy_mvau_design_space_spec() -> DesignSpaceSpec:
    """Build the original streamed-weight MVAU design-space API."""
    region_declarations = (
        RegionDeclaration("standard.streamed", MVAUDesignPaths.STANDARD_STREAMED_REGION),
    )
    semantic = build_kernel_semantic_declarations(
        region_declarations,
        selected_region_path=MVAUDesignPaths.REGION,
        validation_report_path=MVAUDesignPaths.REGION_VALIDATION,
        structural_constraint_path=MVAUDesignPaths.REGION_STRUCTURALLY_WELL_FORMED,
    )
    dependencies = (
        DependencyRef.problem(
            "repetitions", MVAUDesignPaths.REPETITIONS, _INTEGER_OBJECT_SEMANTICS
        ),
        DependencyRef.problem(
            "matrix_width", MVAUDesignPaths.MATRIX_WIDTH, _INTEGER_OBJECT_SEMANTICS
        ),
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
        DependencyRef.decision("pe", MVAUDesignPaths.PE, _INTEGER_OBJECT_SEMANTICS),
        DependencyRef.decision("simd", MVAUDesignPaths.SIMD, _INTEGER_OBJECT_SEMANTICS),
    )

    def derive_region(values: DependencyView) -> Answer[object]:
        return Decided(
            construct_standard_streamed_mvau_region(
                cast(int, values["repetitions"]),
                cast(int, values["matrix_width"]),
                cast(int, values["matrix_height"]),
                cast(NumericElementType, values["activation_element_type"]),
                cast(NumericElementType, values["weight_element_type"]),
                cast(NumericElementType, values["output_element_type"]),
                cast(int, values["pe"]),
                cast(int, values["simd"]),
            )
        )

    return DesignSpaceSpec(
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
                _INTEGER_OBJECT_SEMANTICS,
                _divisor_domain(MVAUDesignPaths.MATRIX_HEIGHT),
            ),
            Decision(
                MVAUDesignPaths.SIMD,
                _INTEGER_OBJECT_SEMANTICS,
                _divisor_domain(MVAUDesignPaths.MATRIX_WIDTH),
            ),
        ),
        properties=(
            DerivedProperty(
                MVAUDesignPaths.STANDARD_STREAMED_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec(dependencies, derive_region),
            ),
            semantic.selected_region,
            semantic.validation_report,
        ),
        constraints=(semantic.structural_constraint,),
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


MVAU_DESIGN_SPACE_SPEC = build_legacy_mvau_design_space_spec()

__all__ = [
    "MVAU_DESIGN_SPACE_SPEC",
    "MVAUDesignPaths",
    "build_legacy_mvau_design_space_spec",
]
