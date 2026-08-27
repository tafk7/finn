# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU compute Kernel definition expressed with the public design API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import TypeVar, cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    AbsenceMode,
    Absent,
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
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernel import (
    BindingDefinition,
    KernelDefinition,
    RegionDeclaration,
    build_kernel_semantic_declarations,
)
from finn.dataflow.mvau.computation import MVAUBindingWitness, MVAUComputationProfile
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    MVAUWeightInterface,
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_embedded_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.region import NumericElementType

E = TypeVar("E", bound=Enum)


class MVAUComputeBinding(str, Enum):
    """Concrete implementation identities, not implementation languages."""

    LEGACY_HLS_LUT = "legacy_hls_lut"
    LEGACY_HLS_DSP = "legacy_hls_dsp"
    RTL_SOFTVEC = "rtl_softvec"
    RTL_PACKED = "rtl_packed"
    RTL_BATCH_INTERLEAVED_DSP58 = "rtl_batch_interleaved_dsp58"

    # Compatibility with the first public MVAU slice.  New code should choose
    # one explicit HLS arithmetic implementation.
    HLS = "legacy_hls_lut"


class MVAUDspBlock(str, Enum):
    """Target DSP capability consumed only by binding constraints."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


class MVAUComputeKernelPaths:
    """Stable paths for the MVAU compute Kernel definition."""

    REPETITIONS = QualifiedPath("problem.mvau.r")
    MATRIX_WIDTH = QualifiedPath("problem.mvau.mw")
    MATRIX_HEIGHT = QualifiedPath("problem.mvau.mh")
    ACTIVATION_ELEMENT_TYPE = QualifiedPath("problem.mvau.activation_element_type")
    WEIGHT_ELEMENT_TYPE = QualifiedPath("problem.mvau.weight_element_type")
    ACCUMULATOR_ELEMENT_TYPE = QualifiedPath("problem.mvau.accumulator_element_type")
    OUTPUT_ELEMENT_TYPE = QualifiedPath("problem.mvau.output_element_type")
    THRESHOLD_ELEMENT_TYPE = QualifiedPath("problem.mvau.threshold_element_type")
    COMPUTATION_PROFILE = QualifiedPath("problem.mvau.computation_profile")
    WEIGHT_INITIALIZER_AVAILABLE = QualifiedPath("problem.mvau.weight_initializer_available")
    TARGET_DSP_BLOCK = QualifiedPath("problem.target.dsp_block")
    WEIGHTS_NARROW = QualifiedPath("problem.mvau.weights_narrow")

    PE = QualifiedPath("mvau.compute.pe")
    SIMD = QualifiedPath("mvau.compute.simd")
    # The original public path is retained while its value is generalized from
    # a two-way interface choice to an explicit region-declaration identity.
    REGION_DECLARATION = QualifiedPath("mvau.compute.weight_interface")
    WEIGHT_INTERFACE = REGION_DECLARATION
    INTERLEAVE = QualifiedPath("mvau.compute.interleave")
    BINDING = QualifiedPath("mvau.compute.binding")
    COMPUTE_PUMPING = QualifiedPath("mvau.compute.binding.compute_pumping")

    NEURON_FOLDS = QualifiedPath("semantic.mvau.compute.neuron_folds")
    SYNAPSE_FOLDS = QualifiedPath("semantic.mvau.compute.synapse_folds")
    BATCH_COUNT = QualifiedPath("semantic.mvau.compute.batch_count")
    WEIGHT_CHUNK_ELEMENTS = QualifiedPath("semantic.mvau.compute.weight_chunk_elements")
    STANDARD_EMBEDDED_REGION = QualifiedPath(
        "semantic.mvau.compute.region_declarations.standard_embedded"
    )
    STANDARD_STREAMED_REGION = QualifiedPath(
        "semantic.mvau.compute.region_declarations.standard_streamed"
    )
    BATCH_INTERLEAVED_REGION = QualifiedPath(
        "semantic.mvau.compute.region_declarations.batch_interleaved_streamed"
    )
    REGION = QualifiedPath("semantic.mvau.compute.region")
    REGION_VALIDATION = QualifiedPath("semantic.mvau.compute.region_validation")
    BINDING_WITNESS = QualifiedPath("binding.mvau.compute.witness")

    REGION_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.compute.region_structurally_well_formed"
    )
    BINDING_REGION_SUPPORTED = QualifiedPath("constraint.mvau.compute.binding_region_supported")
    BINDING_COMPUTATION_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.binding_computation_supported"
    )
    COMPUTATION_TYPES_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.computation_types_supported"
    )
    BINDING_NUMERIC_SUPPORTED = QualifiedPath("constraint.mvau.compute.binding_numeric_supported")
    BINDING_TARGET_SUPPORTED = QualifiedPath("constraint.mvau.compute.binding_target_supported")
    BINDING_DSP48E1_NARROW_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.binding_dsp48e1_narrow_supported"
    )
    BINDING_PACKED_SUPPORTED = QualifiedPath("constraint.mvau.compute.binding_packed_supported")
    BINDING_HLS_PARTITION_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.binding_hls_partition_supported"
    )
    BINDING_HLS_WIDTH_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.binding_hls_width_supported"
    )
    BINDING_HLS_THRESHOLD_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.binding_hls_threshold_supported"
    )
    BINDING_TILED_WIDTH_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.binding_tiled_width_supported"
    )
    BINDING_COMPUTE_PUMPING_SUPPORTED = QualifiedPath(
        "constraint.mvau.compute.binding_compute_pumping_supported"
    )


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _complete_numeric_element_type(value: object) -> bool:
    return type(value) is NumericElementType and bool(value.type_id) and value.bit_width > 0


_INTEGER_SEMANTICS = ValueSemantics.immutable_nominal(int, name="integer")
_BOOL_SEMANTICS = ValueSemantics.immutable_nominal(bool, name="boolean")
_ELEMENT_TYPE_SEMANTICS = ValueSemantics.immutable_nominal(
    NumericElementType, name="NumericElementType"
)
_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_INTEGER_OBJECT_SEMANTICS = as_object_semantics(_INTEGER_SEMANTICS)
_BOOL_OBJECT_SEMANTICS = as_object_semantics(_BOOL_SEMANTICS)
_ELEMENT_TYPE_OBJECT_SEMANTICS = as_object_semantics(_ELEMENT_TYPE_SEMANTICS)


def _enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    semantics: ValueSemantics[E] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


def _normalize_region_declaration(value: object) -> MVAURegionDeclaration:
    if type(value) is MVAURegionDeclaration:
        return value
    if value is MVAUWeightInterface.EMBEDDED:
        return MVAURegionDeclaration.STANDARD_EMBEDDED
    if value is MVAUWeightInterface.STREAMED:
        return MVAURegionDeclaration.STANDARD_STREAMED
    raise TypeError("expected an MVAU region declaration")


_REGION_DECLARATION_SEMANTICS: ValueSemantics[object] = ValueSemantics(
    MVAURegionDeclaration,
    "MVAURegionDeclaration",
    lambda value: type(value) in {MVAURegionDeclaration, MVAUWeightInterface},
    lambda left, right: _normalize_region_declaration(left) is _normalize_region_declaration(right),
    lambda value: _normalize_region_declaration(value),
)
_COMPUTATION_SEMANTICS = _enum_semantics(MVAUComputationProfile)
_BINDING_SEMANTICS = _enum_semantics(MVAUComputeBinding)
_DSP_BLOCK_SEMANTICS = _enum_semantics(MVAUDspBlock)
_WITNESS_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUBindingWitness, name="MVAUBindingWitness")
)


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


def _divisor_domain(dimension: QualifiedPath) -> DecisionDomain:
    dependency = DependencyRef.problem("dimension", dimension, _INTEGER_OBJECT_SEMANTICS)

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        extent = cast(int, dependencies["dimension"])
        return Decided(type(value) is int and value > 0 and extent % value == 0)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        extent = cast(int, dependencies["dimension"])
        return Decided(tuple(value for value in range(1, extent + 1) if extent % value == 0))

    return DecisionDomain((dependency,), accepts, EvaluatorSpec((dependency,), candidates))


_REGION_DECLARATION_REF = DependencyRef.decision(
    "region_declaration",
    MVAUComputeKernelPaths.REGION_DECLARATION,
    _REGION_DECLARATION_SEMANTICS,
)
_PE_REF = DependencyRef.decision("pe", MVAUComputeKernelPaths.PE, _INTEGER_OBJECT_SEMANTICS)
_SIMD_REF = DependencyRef.decision("simd", MVAUComputeKernelPaths.SIMD, _INTEGER_OBJECT_SEMANTICS)
_INTERLEAVE_REF = DependencyRef.decision(
    "interleave",
    MVAUComputeKernelPaths.INTERLEAVE,
    _INTEGER_OBJECT_SEMANTICS,
)
_BINDING_REF = DependencyRef.decision("binding", MVAUComputeKernelPaths.BINDING, _BINDING_SEMANTICS)
_COMPUTE_PUMPING_REF = DependencyRef.decision(
    "compute_pumping",
    MVAUComputeKernelPaths.COMPUTE_PUMPING,
    _BOOL_OBJECT_SEMANTICS,
)
_ACTIVATION_TYPE_REF = DependencyRef.problem(
    "activation_element_type",
    MVAUComputeKernelPaths.ACTIVATION_ELEMENT_TYPE,
    _ELEMENT_TYPE_OBJECT_SEMANTICS,
)
_WEIGHT_TYPE_REF = DependencyRef.problem(
    "weight_element_type",
    MVAUComputeKernelPaths.WEIGHT_ELEMENT_TYPE,
    _ELEMENT_TYPE_OBJECT_SEMANTICS,
)
_ACCUMULATOR_TYPE_REF = DependencyRef.problem(
    "accumulator_element_type",
    MVAUComputeKernelPaths.ACCUMULATOR_ELEMENT_TYPE,
    _ELEMENT_TYPE_OBJECT_SEMANTICS,
)
_THRESHOLD_TYPE_REF = DependencyRef.problem(
    "threshold_element_type",
    MVAUComputeKernelPaths.THRESHOLD_ELEMENT_TYPE,
    _ELEMENT_TYPE_OBJECT_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_COMPUTATION_REF = DependencyRef.problem(
    "computation_profile",
    MVAUComputeKernelPaths.COMPUTATION_PROFILE,
    _COMPUTATION_SEMANTICS,
)
_TARGET_DSP_REF = DependencyRef.problem(
    "target_dsp_block",
    MVAUComputeKernelPaths.TARGET_DSP_BLOCK,
    _DSP_BLOCK_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_WEIGHTS_NARROW_REF = DependencyRef.problem(
    "weights_narrow",
    MVAUComputeKernelPaths.WEIGHTS_NARROW,
    _BOOL_OBJECT_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)


def _region_declaration_domain() -> DecisionDomain:
    initializer = DependencyRef.problem(
        "weight_initializer_available",
        MVAUComputeKernelPaths.WEIGHT_INITIALIZER_AVAILABLE,
        _BOOL_OBJECT_SEMANTICS,
    )

    def available(dependencies: DependencyView) -> tuple[object, ...]:
        declarations = list(MVAURegionDeclaration)
        if not cast(bool, dependencies["weight_initializer_available"]):
            declarations.remove(MVAURegionDeclaration.STANDARD_EMBEDDED)
        return tuple(declarations)

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        return Decided(_normalize_region_declaration(value) in available(dependencies))

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(available(dependencies))

    return DecisionDomain((initializer,), accepts, EvaluatorSpec((initializer,), candidates))


def _region_applies(*declarations: MVAURegionDeclaration) -> EvaluatorSpec[Answer[bool]]:
    accepted = frozenset(declarations)

    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(dependencies["region_declaration"] in accepted)

    return EvaluatorSpec((_REGION_DECLARATION_REF,), evaluate)


def _interleave_domain() -> DecisionDomain:
    repetitions = DependencyRef.problem(
        "repetitions", MVAUComputeKernelPaths.REPETITIONS, _INTEGER_OBJECT_SEMANTICS
    )

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        if type(value) is not int:
            return Decided(False)
        candidate = value
        repeated = cast(int, dependencies["repetitions"])
        pe = cast(int, dependencies["pe"])
        simd = cast(int, dependencies["simd"])
        findings = []
        if candidate <= 1:
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "mvau-interleave-not-greater-than-one",
                    MVAUComputeKernelPaths.INTERLEAVE,
                    "batch interleave must be greater than one",
                    (("candidate", candidate),),
                )
            )
        if candidate > 0 and repeated % candidate:
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "mvau-interleave-does-not-divide-repetitions",
                    MVAUComputeKernelPaths.INTERLEAVE,
                    "batch interleave must divide the repetition extent",
                    (("candidate", candidate), ("repetitions", repeated)),
                )
            )
        if candidate > 0 and (pe * simd) % candidate:
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "mvau-interleave-does-not-divide-weight-tile",
                    MVAUComputeKernelPaths.INTERLEAVE,
                    "batch interleave must divide PE * SIMD",
                    (("candidate", candidate), ("pe", pe), ("simd", simd)),
                )
            )
        return Absent(tuple(findings)) if findings else Decided(True)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        repeated = cast(int, dependencies["repetitions"])
        tile_elements = cast(int, dependencies["pe"]) * cast(int, dependencies["simd"])
        limit = gcd(repeated, tile_elements)
        return Decided(tuple(value for value in range(2, limit + 1) if limit % value == 0))

    dependencies = (repetitions, _PE_REF, _SIMD_REF)
    return DecisionDomain(dependencies, accepts, EvaluatorSpec(dependencies, candidates))


def _binding_applies(*bindings: MVAUComputeBinding) -> EvaluatorSpec[Answer[bool]]:
    accepted = frozenset(bindings)

    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(dependencies["binding"] in accepted)

    return EvaluatorSpec((_BINDING_REF,), evaluate)


def _binding_and_computation_apply(
    bindings: tuple[MVAUComputeBinding, ...],
    profiles: tuple[MVAUComputationProfile, ...],
) -> EvaluatorSpec[Answer[bool]]:
    accepted_bindings = frozenset(bindings)
    accepted_profiles = frozenset(profiles)

    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(
            dependencies["binding"] in accepted_bindings
            and dependencies["computation_profile"] in accepted_profiles
        )

    return EvaluatorSpec((_BINDING_REF, _COMPUTATION_REF), evaluate)


def _missing_problem(
    owner: QualifiedPath, field: QualifiedPath, code: str, message: str
) -> Unresolved:
    return Unresolved((Finding(FindingKind.LIMITATION, code, owner, message, trace=(field,)),))


_STANDARD_REGION_DEPENDENCIES = (
    DependencyRef.problem(
        "repetitions", MVAUComputeKernelPaths.REPETITIONS, _INTEGER_OBJECT_SEMANTICS
    ),
    DependencyRef.problem(
        "matrix_width", MVAUComputeKernelPaths.MATRIX_WIDTH, _INTEGER_OBJECT_SEMANTICS
    ),
    DependencyRef.problem(
        "matrix_height", MVAUComputeKernelPaths.MATRIX_HEIGHT, _INTEGER_OBJECT_SEMANTICS
    ),
    _ACTIVATION_TYPE_REF,
    _WEIGHT_TYPE_REF,
    DependencyRef.problem(
        "output_element_type",
        MVAUComputeKernelPaths.OUTPUT_ELEMENT_TYPE,
        _ELEMENT_TYPE_OBJECT_SEMANTICS,
    ),
    _PE_REF,
    _SIMD_REF,
)


@dataclass(frozen=True)
class _MVAURegionArguments:
    repetitions: int
    matrix_width: int
    matrix_height: int
    activation_element_type: NumericElementType
    weight_element_type: NumericElementType
    output_element_type: NumericElementType
    pe: int
    simd: int


def _region_arguments(dependencies: DependencyView) -> _MVAURegionArguments:
    return _MVAURegionArguments(
        repetitions=cast(int, dependencies["repetitions"]),
        matrix_width=cast(int, dependencies["matrix_width"]),
        matrix_height=cast(int, dependencies["matrix_height"]),
        activation_element_type=cast(NumericElementType, dependencies["activation_element_type"]),
        weight_element_type=cast(NumericElementType, dependencies["weight_element_type"]),
        output_element_type=cast(NumericElementType, dependencies["output_element_type"]),
        pe=cast(int, dependencies["pe"]),
        simd=cast(int, dependencies["simd"]),
    )


def _derive_standard_embedded_region(dependencies: DependencyView) -> Answer[object]:
    arguments = _region_arguments(dependencies)
    return Decided(
        construct_standard_embedded_mvau_region(
            arguments.repetitions,
            arguments.matrix_width,
            arguments.matrix_height,
            arguments.activation_element_type,
            arguments.weight_element_type,
            arguments.output_element_type,
            arguments.pe,
            arguments.simd,
        )
    )


def _derive_standard_streamed_region(dependencies: DependencyView) -> Answer[object]:
    arguments = _region_arguments(dependencies)
    return Decided(
        construct_standard_streamed_mvau_region(
            arguments.repetitions,
            arguments.matrix_width,
            arguments.matrix_height,
            arguments.activation_element_type,
            arguments.weight_element_type,
            arguments.output_element_type,
            arguments.pe,
            arguments.simd,
        )
    )


def _derive_batch_interleaved_region(dependencies: DependencyView) -> Answer[object]:
    arguments = _region_arguments(dependencies)
    return Decided(
        construct_batch_interleaved_streamed_mvau_region(
            arguments.repetitions,
            arguments.matrix_width,
            arguments.matrix_height,
            arguments.activation_element_type,
            arguments.weight_element_type,
            arguments.output_element_type,
            arguments.pe,
            arguments.simd,
            cast(int, dependencies["interleave"]),
        )
    )


def _derive_quotient(
    numerator_name: str, denominator_name: str
) -> Callable[[DependencyView], Answer[object]]:
    def evaluate(dependencies: DependencyView) -> Answer[object]:
        return Decided(
            cast(int, dependencies[numerator_name]) // cast(int, dependencies[denominator_name])
        )

    return evaluate


_MVAU_REGION_DECLARATIONS = (
    RegionDeclaration(
        MVAURegionDeclaration.STANDARD_EMBEDDED.value,
        MVAUComputeKernelPaths.STANDARD_EMBEDDED_REGION,
    ),
    RegionDeclaration(
        MVAURegionDeclaration.STANDARD_STREAMED.value,
        MVAUComputeKernelPaths.STANDARD_STREAMED_REGION,
    ),
    RegionDeclaration(
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED.value,
        MVAUComputeKernelPaths.BATCH_INTERLEAVED_REGION,
    ),
)


def _binding_region_supported(dependencies: DependencyView) -> Answer[bool]:
    binding = cast(MVAUComputeBinding, dependencies["binding"])
    declaration = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    supported = {
        MVAUComputeBinding.LEGACY_HLS_LUT: {
            MVAURegionDeclaration.STANDARD_EMBEDDED,
            MVAURegionDeclaration.STANDARD_STREAMED,
        },
        MVAUComputeBinding.LEGACY_HLS_DSP: {
            MVAURegionDeclaration.STANDARD_EMBEDDED,
            MVAURegionDeclaration.STANDARD_STREAMED,
        },
        MVAUComputeBinding.RTL_SOFTVEC: {MVAURegionDeclaration.STANDARD_STREAMED},
        MVAUComputeBinding.RTL_PACKED: {MVAURegionDeclaration.STANDARD_STREAMED},
        MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58: {
            MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
        },
    }
    return Decided(declaration in supported[binding])


def _binding_computation_supported(dependencies: DependencyView) -> Answer[bool]:
    binding = cast(MVAUComputeBinding, dependencies["binding"])
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    if binding in {
        MVAUComputeBinding.LEGACY_HLS_LUT,
        MVAUComputeBinding.LEGACY_HLS_DSP,
    }:
        return Decided(True)
    return Decided(profile is MVAUComputationProfile.ACCUMULATOR_INTEGER)


def _computation_types_supported(dependencies: DependencyView) -> Answer[bool]:
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    if profile is MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR:
        return Decided(activation.type_id == "bipolar" and weight.type_id == "bipolar")
    return Decided(activation.type_id != "binary" and weight.type_id != "binary")


def _binding_numeric_supported(dependencies: DependencyView) -> Answer[bool]:
    binding = cast(MVAUComputeBinding, dependencies["binding"])
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    if binding in {
        MVAUComputeBinding.LEGACY_HLS_LUT,
        MVAUComputeBinding.LEGACY_HLS_DSP,
    }:
        supported = {"int", "uint", "bipolar"}
        return Decided(activation.type_id in supported and weight.type_id in supported)
    return Decided(
        activation.type_id in {"int", "uint"}
        and weight.type_id == "int"
        and activation.bit_width >= 2
        and weight.bit_width >= 2
    )


def _binding_target_supported(dependencies: DependencyView) -> Answer[bool]:
    target = dependencies["target_dsp_block"]
    if target is ABSENT:
        return _missing_problem(
            MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED,
            MVAUComputeKernelPaths.TARGET_DSP_BLOCK,
            "mvau-target-dsp-missing",
            "the selected RTL binding requires a target DSP block",
        )
    binding = cast(MVAUComputeBinding, dependencies["binding"])
    dsp = cast(MVAUDspBlock, target)
    if binding in {
        MVAUComputeBinding.RTL_PACKED,
        MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
    }:
        return Decided(dsp is MVAUDspBlock.DSP58)
    return Decided(dsp in frozenset(MVAUDspBlock))


def _dsp48e1_narrow_supported(dependencies: DependencyView) -> Answer[bool]:
    target = dependencies["target_dsp_block"]
    narrow = dependencies["weights_narrow"]
    if narrow is not ABSENT and cast(bool, narrow):
        return Decided(True)
    if target is ABSENT:
        return _missing_problem(
            MVAUComputeKernelPaths.BINDING_DSP48E1_NARROW_SUPPORTED,
            MVAUComputeKernelPaths.TARGET_DSP_BLOCK,
            "mvau-target-dsp-missing",
            "the selected RTL binding requires a target DSP block",
        )
    if target is not MVAUDspBlock.DSP48E1:
        return Decided(True)
    if narrow is ABSENT:
        return _missing_problem(
            MVAUComputeKernelPaths.BINDING_DSP48E1_NARROW_SUPPORTED,
            MVAUComputeKernelPaths.WEIGHTS_NARROW,
            "mvau-weights-narrow-missing",
            "DSP48E1 RTL feasibility requires a weights-narrow fact",
        )
    return Decided(cast(bool, narrow))


def _packed_supported(dependencies: DependencyView) -> Answer[bool]:
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    if weight.bit_width > 8 or activation.bit_width > 9:
        return Decided(False)
    narrow = dependencies["weights_narrow"]
    if narrow is ABSENT:
        return _missing_problem(
            MVAUComputeKernelPaths.BINDING_PACKED_SUPPORTED,
            MVAUComputeKernelPaths.WEIGHTS_NARROW,
            "mvau-weights-narrow-missing",
            "packed RTL feasibility requires a weights-narrow fact",
        )
    dsp_input_width = 27
    lane_width = weight.bit_width + activation.bit_width - 1
    lanes = (
        1
        if dsp_input_width == weight.bit_width
        else 1
        + (dsp_input_width - (0 if cast(bool, narrow) else 1) - weight.bit_width) // lane_width
    )
    return Decided(lanes <= 3)


def _hls_partition_supported(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        cast(int, dependencies["simd"]) * 1024 >= cast(int, dependencies["matrix_width"])
    )


def _hls_width_supported(dependencies: DependencyView) -> Answer[bool]:
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    accumulator = cast(NumericElementType, dependencies["accumulator_element_type"])
    output = cast(NumericElementType, dependencies["output_element_type"])
    pe = cast(int, dependencies["pe"])
    simd = cast(int, dependencies["simd"])
    widths = (
        activation.bit_width * simd,
        weight.bit_width * pe * simd,
        accumulator.bit_width * pe,
        output.bit_width * pe,
    )
    return Decided(max(widths) <= 8191)


def _hls_threshold_supported(dependencies: DependencyView) -> Answer[bool]:
    threshold = dependencies["threshold_element_type"]
    if threshold is ABSENT:
        return _missing_problem(
            MVAUComputeKernelPaths.BINDING_HLS_THRESHOLD_SUPPORTED,
            MVAUComputeKernelPaths.THRESHOLD_ELEMENT_TYPE,
            "mvau-threshold-type-missing",
            "fused-threshold feasibility requires a threshold element type",
        )
    accumulator = cast(NumericElementType, dependencies["accumulator_element_type"])
    return Decided(cast(NumericElementType, threshold).bit_width >= accumulator.bit_width)


def _tiled_width_supported(dependencies: DependencyView) -> Answer[bool]:
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    return Decided(activation.bit_width <= 8 and weight.bit_width <= 8)


def _compute_pumping_supported(dependencies: DependencyView) -> Answer[bool]:
    if not cast(bool, dependencies["compute_pumping"]):
        return Decided(True)
    binding = cast(MVAUComputeBinding, dependencies["binding"])
    declaration = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    simd = cast(int, dependencies["simd"])
    return Decided(
        declaration is MVAURegionDeclaration.STANDARD_STREAMED
        and binding in {MVAUComputeBinding.RTL_SOFTVEC, MVAUComputeBinding.RTL_PACKED}
        and simd > 1
    )


def _derive_binding_witness(dependencies: DependencyView) -> Answer[object]:
    binding = cast(MVAUComputeBinding, dependencies["binding"])
    declaration = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    mechanisms = ["activation_replay", "dot_product_accumulation"]
    if declaration is MVAURegionDeclaration.STANDARD_EMBEDDED:
        mechanisms.append("local_weight_state")
    elif declaration is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED:
        mechanisms.extend(("weight_chunk_assembly_and_replay", "output_reorder"))
    else:
        mechanisms.append("streamed_full_weight_tiles")
    if profile is MVAUComputationProfile.FUSED_THRESHOLD:
        mechanisms.append("fused_threshold_state")
    return Decided(MVAUBindingWitness(binding.value, declaration.value, profile, tuple(mechanisms)))


def build_mvau_compute_kernel_spec() -> DesignSpaceSpec:
    """Build the MVAU-specific flat engine specification."""
    semantic = build_kernel_semantic_declarations(
        _MVAU_REGION_DECLARATIONS,
        selected_region_path=MVAUComputeKernelPaths.REGION,
        validation_report_path=MVAUComputeKernelPaths.REGION_VALIDATION,
        structural_constraint_path=MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
    )
    hls_bindings = (MVAUComputeBinding.LEGACY_HLS_LUT, MVAUComputeBinding.LEGACY_HLS_DSP)
    rtl_bindings = (
        MVAUComputeBinding.RTL_SOFTVEC,
        MVAUComputeBinding.RTL_PACKED,
        MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
    )
    return DesignSpaceSpec(
        problem_schema=ProblemSchema(
            (
                ProblemField(
                    MVAUComputeKernelPaths.REPETITIONS,
                    _INTEGER_OBJECT_SEMANTICS,
                    constraint=_positive_integer,
                    constraint_description="must be a positive integer",
                ),
                ProblemField(
                    MVAUComputeKernelPaths.MATRIX_WIDTH,
                    _INTEGER_OBJECT_SEMANTICS,
                    constraint=_positive_integer,
                    constraint_description="must be a positive integer",
                ),
                ProblemField(
                    MVAUComputeKernelPaths.MATRIX_HEIGHT,
                    _INTEGER_OBJECT_SEMANTICS,
                    constraint=_positive_integer,
                    constraint_description="must be a positive integer",
                ),
                ProblemField(
                    MVAUComputeKernelPaths.ACTIVATION_ELEMENT_TYPE,
                    _ELEMENT_TYPE_OBJECT_SEMANTICS,
                    constraint=_complete_numeric_element_type,
                    constraint_description="must be a complete numeric element type",
                ),
                ProblemField(
                    MVAUComputeKernelPaths.WEIGHT_ELEMENT_TYPE,
                    _ELEMENT_TYPE_OBJECT_SEMANTICS,
                    constraint=_complete_numeric_element_type,
                    constraint_description="must be a complete numeric element type",
                ),
                ProblemField(
                    MVAUComputeKernelPaths.ACCUMULATOR_ELEMENT_TYPE,
                    _ELEMENT_TYPE_OBJECT_SEMANTICS,
                    constraint=_complete_numeric_element_type,
                    constraint_description="must be a complete numeric element type",
                ),
                ProblemField(
                    MVAUComputeKernelPaths.OUTPUT_ELEMENT_TYPE,
                    _ELEMENT_TYPE_OBJECT_SEMANTICS,
                    constraint=_complete_numeric_element_type,
                    constraint_description="must be a complete numeric element type",
                ),
                ProblemField(
                    MVAUComputeKernelPaths.THRESHOLD_ELEMENT_TYPE,
                    _ELEMENT_TYPE_OBJECT_SEMANTICS,
                    required=False,
                    constraint=_complete_numeric_element_type,
                    constraint_description="must be a complete numeric element type",
                ),
                ProblemField(MVAUComputeKernelPaths.COMPUTATION_PROFILE, _COMPUTATION_SEMANTICS),
                ProblemField(
                    MVAUComputeKernelPaths.WEIGHT_INITIALIZER_AVAILABLE,
                    _BOOL_OBJECT_SEMANTICS,
                ),
                ProblemField(
                    MVAUComputeKernelPaths.TARGET_DSP_BLOCK,
                    _DSP_BLOCK_SEMANTICS,
                    required=False,
                ),
                ProblemField(
                    MVAUComputeKernelPaths.WEIGHTS_NARROW,
                    _BOOL_OBJECT_SEMANTICS,
                    required=False,
                ),
            )
        ),
        decisions=(
            Decision(
                MVAUComputeKernelPaths.PE,
                _INTEGER_OBJECT_SEMANTICS,
                _divisor_domain(MVAUComputeKernelPaths.MATRIX_HEIGHT),
            ),
            Decision(
                MVAUComputeKernelPaths.SIMD,
                _INTEGER_OBJECT_SEMANTICS,
                _divisor_domain(MVAUComputeKernelPaths.MATRIX_WIDTH),
            ),
            Decision(
                MVAUComputeKernelPaths.REGION_DECLARATION,
                _REGION_DECLARATION_SEMANTICS,
                _region_declaration_domain(),
            ),
            Decision(
                MVAUComputeKernelPaths.INTERLEAVE,
                _INTEGER_OBJECT_SEMANTICS,
                _interleave_domain(),
                applies_if=_region_applies(MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED),
            ),
            Decision(
                MVAUComputeKernelPaths.BINDING,
                _BINDING_SEMANTICS,
                _finite_domain(tuple(MVAUComputeBinding)),
            ),
            Decision(
                MVAUComputeKernelPaths.COMPUTE_PUMPING,
                _BOOL_OBJECT_SEMANTICS,
                _finite_domain((False, True)),
            ),
        ),
        properties=(
            DerivedProperty(
                MVAUComputeKernelPaths.NEURON_FOLDS,
                _INTEGER_OBJECT_SEMANTICS,
                EvaluatorSpec(
                    (
                        DependencyRef.problem(
                            "matrix_height",
                            MVAUComputeKernelPaths.MATRIX_HEIGHT,
                            _INTEGER_OBJECT_SEMANTICS,
                        ),
                        _PE_REF,
                    ),
                    _derive_quotient("matrix_height", "pe"),
                ),
            ),
            DerivedProperty(
                MVAUComputeKernelPaths.SYNAPSE_FOLDS,
                _INTEGER_OBJECT_SEMANTICS,
                EvaluatorSpec(
                    (
                        DependencyRef.problem(
                            "matrix_width",
                            MVAUComputeKernelPaths.MATRIX_WIDTH,
                            _INTEGER_OBJECT_SEMANTICS,
                        ),
                        _SIMD_REF,
                    ),
                    _derive_quotient("matrix_width", "simd"),
                ),
            ),
            DerivedProperty(
                MVAUComputeKernelPaths.BATCH_COUNT,
                _INTEGER_OBJECT_SEMANTICS,
                EvaluatorSpec(
                    (
                        DependencyRef.problem(
                            "repetitions",
                            MVAUComputeKernelPaths.REPETITIONS,
                            _INTEGER_OBJECT_SEMANTICS,
                        ),
                        _INTERLEAVE_REF,
                    ),
                    _derive_quotient("repetitions", "interleave"),
                ),
                applies_if=_region_applies(MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED),
            ),
            DerivedProperty(
                MVAUComputeKernelPaths.WEIGHT_CHUNK_ELEMENTS,
                _INTEGER_OBJECT_SEMANTICS,
                EvaluatorSpec(
                    (_PE_REF, _SIMD_REF, _INTERLEAVE_REF),
                    lambda dependencies: Decided(
                        cast(int, dependencies["pe"])
                        * cast(int, dependencies["simd"])
                        // cast(int, dependencies["interleave"])
                    ),
                ),
                applies_if=_region_applies(MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED),
            ),
            DerivedProperty(
                MVAUComputeKernelPaths.STANDARD_EMBEDDED_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec(_STANDARD_REGION_DEPENDENCIES, _derive_standard_embedded_region),
                applies_if=_region_applies(MVAURegionDeclaration.STANDARD_EMBEDDED),
            ),
            DerivedProperty(
                MVAUComputeKernelPaths.STANDARD_STREAMED_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec(_STANDARD_REGION_DEPENDENCIES, _derive_standard_streamed_region),
                applies_if=_region_applies(MVAURegionDeclaration.STANDARD_STREAMED),
            ),
            DerivedProperty(
                MVAUComputeKernelPaths.BATCH_INTERLEAVED_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec(
                    (*_STANDARD_REGION_DEPENDENCIES, _INTERLEAVE_REF),
                    _derive_batch_interleaved_region,
                ),
                applies_if=_region_applies(MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED),
            ),
            semantic.selected_region,
            semantic.validation_report,
            DerivedProperty(
                MVAUComputeKernelPaths.BINDING_WITNESS,
                _WITNESS_SEMANTICS,
                EvaluatorSpec(
                    (_BINDING_REF, _REGION_DECLARATION_REF, _COMPUTATION_REF),
                    _derive_binding_witness,
                ),
            ),
        ),
        constraints=(
            semantic.structural_constraint,
            Constraint(
                MVAUComputeKernelPaths.BINDING_REGION_SUPPORTED,
                EvaluatorSpec((_BINDING_REF, _REGION_DECLARATION_REF), _binding_region_supported),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_COMPUTATION_SUPPORTED,
                EvaluatorSpec((_BINDING_REF, _COMPUTATION_REF), _binding_computation_supported),
            ),
            Constraint(
                MVAUComputeKernelPaths.COMPUTATION_TYPES_SUPPORTED,
                EvaluatorSpec(
                    (_COMPUTATION_REF, _ACTIVATION_TYPE_REF, _WEIGHT_TYPE_REF),
                    _computation_types_supported,
                ),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_NUMERIC_SUPPORTED,
                EvaluatorSpec(
                    (_BINDING_REF, _ACTIVATION_TYPE_REF, _WEIGHT_TYPE_REF),
                    _binding_numeric_supported,
                ),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED,
                EvaluatorSpec((_BINDING_REF, _TARGET_DSP_REF), _binding_target_supported),
                applies_if=_binding_applies(*rtl_bindings),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_DSP48E1_NARROW_SUPPORTED,
                EvaluatorSpec(
                    (_TARGET_DSP_REF, _WEIGHTS_NARROW_REF),
                    _dsp48e1_narrow_supported,
                ),
                applies_if=_binding_applies(MVAUComputeBinding.RTL_SOFTVEC),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_PACKED_SUPPORTED,
                EvaluatorSpec(
                    (_ACTIVATION_TYPE_REF, _WEIGHT_TYPE_REF, _WEIGHTS_NARROW_REF),
                    _packed_supported,
                ),
                applies_if=_binding_applies(MVAUComputeBinding.RTL_PACKED),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_HLS_PARTITION_SUPPORTED,
                EvaluatorSpec(
                    (
                        DependencyRef.problem(
                            "matrix_width",
                            MVAUComputeKernelPaths.MATRIX_WIDTH,
                            _INTEGER_OBJECT_SEMANTICS,
                        ),
                        _SIMD_REF,
                    ),
                    _hls_partition_supported,
                ),
                applies_if=_binding_applies(*hls_bindings),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_HLS_WIDTH_SUPPORTED,
                EvaluatorSpec(
                    (
                        _ACTIVATION_TYPE_REF,
                        _WEIGHT_TYPE_REF,
                        _ACCUMULATOR_TYPE_REF,
                        DependencyRef.problem(
                            "output_element_type",
                            MVAUComputeKernelPaths.OUTPUT_ELEMENT_TYPE,
                            _ELEMENT_TYPE_OBJECT_SEMANTICS,
                        ),
                        _PE_REF,
                        _SIMD_REF,
                    ),
                    _hls_width_supported,
                ),
                applies_if=_binding_applies(*hls_bindings),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_HLS_THRESHOLD_SUPPORTED,
                EvaluatorSpec(
                    (_THRESHOLD_TYPE_REF, _ACCUMULATOR_TYPE_REF),
                    _hls_threshold_supported,
                ),
                applies_if=_binding_and_computation_apply(
                    hls_bindings, (MVAUComputationProfile.FUSED_THRESHOLD,)
                ),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_TILED_WIDTH_SUPPORTED,
                EvaluatorSpec((_ACTIVATION_TYPE_REF, _WEIGHT_TYPE_REF), _tiled_width_supported),
                applies_if=_binding_applies(MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58),
            ),
            Constraint(
                MVAUComputeKernelPaths.BINDING_COMPUTE_PUMPING_SUPPORTED,
                EvaluatorSpec(
                    (
                        _COMPUTE_PUMPING_REF,
                        _BINDING_REF,
                        _REGION_DECLARATION_REF,
                        _SIMD_REF,
                    ),
                    _compute_pumping_supported,
                ),
            ),
        ),
        constraint_sets=(
            ConstraintSet(
                "model_structural",
                (MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,),
            ),
            ConstraintSet(
                "binding_feasibility",
                (
                    MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    MVAUComputeKernelPaths.BINDING_REGION_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_COMPUTATION_SUPPORTED,
                    MVAUComputeKernelPaths.COMPUTATION_TYPES_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_NUMERIC_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_DSP48E1_NARROW_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_PACKED_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_HLS_PARTITION_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_HLS_WIDTH_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_HLS_THRESHOLD_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_TILED_WIDTH_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_COMPUTE_PUMPING_SUPPORTED,
                ),
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(
                "model_structural",
                decisions=(
                    MVAUComputeKernelPaths.PE,
                    MVAUComputeKernelPaths.SIMD,
                    MVAUComputeKernelPaths.REGION_DECLARATION,
                    MVAUComputeKernelPaths.INTERLEAVE,
                ),
                properties=(
                    MVAUComputeKernelPaths.REGION,
                    MVAUComputeKernelPaths.REGION_VALIDATION,
                ),
                constraints=(MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,),
            ),
            ReadinessProfile(
                "binding_feasibility",
                decisions=(
                    MVAUComputeKernelPaths.PE,
                    MVAUComputeKernelPaths.SIMD,
                    MVAUComputeKernelPaths.REGION_DECLARATION,
                    MVAUComputeKernelPaths.INTERLEAVE,
                    MVAUComputeKernelPaths.BINDING,
                    MVAUComputeKernelPaths.COMPUTE_PUMPING,
                ),
                properties=(
                    MVAUComputeKernelPaths.REGION,
                    MVAUComputeKernelPaths.REGION_VALIDATION,
                    MVAUComputeKernelPaths.BINDING_WITNESS,
                ),
                constraints=(
                    MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    MVAUComputeKernelPaths.BINDING_REGION_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_COMPUTATION_SUPPORTED,
                    MVAUComputeKernelPaths.COMPUTATION_TYPES_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_NUMERIC_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_DSP48E1_NARROW_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_PACKED_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_HLS_PARTITION_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_HLS_WIDTH_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_HLS_THRESHOLD_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_TILED_WIDTH_SUPPORTED,
                    MVAUComputeKernelPaths.BINDING_COMPUTE_PUMPING_SUPPORTED,
                ),
            ),
        ),
    )


MVAU_COMPUTE_KERNEL = KernelDefinition(
    id="mvau.compute",
    spec=build_mvau_compute_kernel_spec(),
    region_declarations=_MVAU_REGION_DECLARATIONS,
    binding_definitions=tuple(BindingDefinition(binding.value) for binding in MVAUComputeBinding),
    selected_region_path=MVAUComputeKernelPaths.REGION,
    binding_decision_path=MVAUComputeKernelPaths.BINDING,
    binding_witness_path=MVAUComputeKernelPaths.BINDING_WITNESS,
    structural_readiness_profile="model_structural",
    binding_readiness_profile="binding_feasibility",
)
MVAU_COMPUTE_KERNEL_SPEC = MVAU_COMPUTE_KERNEL.spec

__all__ = [
    "MVAU_COMPUTE_KERNEL",
    "MVAU_COMPUTE_KERNEL_SPEC",
    "MVAUComputeBinding",
    "MVAUComputeKernelPaths",
    "MVAUDspBlock",
    "build_mvau_compute_kernel_spec",
]
