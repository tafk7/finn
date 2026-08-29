# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The four initial MVAU compute Kernels.

Each Kernel names one meaningful microarchitecture, owns its own PE, SIMD, and
implementation-specific choices, derives exactly one Region per complete local
assignment, and derives its own weight demand.  There is no Region-declaration
choice above them and no binding choice below them.

Legacy HLS embedded versus streamed weights is one Kernel with one local
boundary choice, not two Kernels: the generated HLS matrix-vector unit reads
its weight array inside its own schedule either way, so the two forms differ
only in whether that array is exposed at the Region boundary.  See
``scratchpad/open/op-kernel-waterfall-design.md`` §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import TypeVar, cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    Absent,
    Answer,
    Constraint,
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
    QualifiedPath,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.kernels import (
    KernelDeclaration,
    KernelDemand,
    KernelExport,
    KernelProvider,
    KernelSelection,
)
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_embedded_mvau_region,
    construct_standard_mvau_weight_port,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.mvau.compute_pool import (
    FULL_TILE_WEIGHT_EXPORT,
    MVAUComputeKernelId,
    MVAU_COMPUTE_SELECTION_NAME,
    REGION_FORM_EXPORT,
    WEIGHT_INTERFACE,
)
from finn.dataflow.mvau.decomposed import build_decomposed_mvau_kernels
from finn.dataflow.mvau_problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAUDspBlock,
)
from finn.dataflow.region import DataflowRegion, NumericElementType, Port

E = TypeVar("E", bound=Enum)


class MVAUHlsResource(str, Enum):
    """The legacy HLS arithmetic resource mapping, not a microarchitecture."""

    LUT = "lut"
    DSP = "dsp"


class MVAUWeightSource(str, Enum):
    """Where the legacy HLS Kernel's weight array sits on its Region boundary."""

    EMBEDDED = "embedded"
    STREAMED = "streamed"


#: The RTL generator that realizes the soft-vector Kernel.  A provider is
#: declared here, on the Kernel it implements, so the inventory is the single
#: answer to "what can build this".
SOFT_VECTOR_PROVIDER_ID = "finn.rtl.mvu_vvu_axi"


_INTEGER = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
_BOOL = as_object_semantics(ValueSemantics.immutable_nominal(bool, name="boolean"))
_ELEMENT_TYPE = as_object_semantics(
    ValueSemantics.immutable_nominal(NumericElementType, name="NumericElementType")
)
_REGION = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_PORT = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))


def _enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    semantics: ValueSemantics[E] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


_COMPUTATION = _enum_semantics(MVAUComputationProfile)
_DSP_BLOCK = _enum_semantics(MVAUDspBlock)
_REGION_FORM = _enum_semantics(MVAURegionDeclaration)
_HLS_RESOURCE = _enum_semantics(MVAUHlsResource)
_WEIGHT_SOURCE = _enum_semantics(MVAUWeightSource)


# Every problem dependency below comes from the operation's declaration, not
# from a path this module restates.  A Kernel reads a fact through the handle
# the operation owns; the name on the left of each pair is only the evaluator
# parameter it arrives under.
_ACTIVATION_TYPE = MVAU_PROBLEM.activation_element_type.dependency("activation_element_type")
_WEIGHT_TYPE = MVAU_PROBLEM.weight_element_type.dependency("weight_element_type")
_ACCUMULATOR_TYPE = MVAU_PROBLEM.accumulator_element_type.dependency("accumulator_element_type")
_OUTPUT_TYPE = MVAU_PROBLEM.output_element_type.dependency("output_element_type")
_THRESHOLD_TYPE = MVAU_PROBLEM.threshold_element_type.allow_absent().dependency(
    "threshold_element_type"
)
_THRESHOLD_INITIALIZER = MVAU_PROBLEM.threshold_initializer_available.allow_absent().dependency(
    "threshold_initializer_available"
)
_COMPUTATION_PROFILE = MVAU_PROBLEM.computation_profile.dependency("computation_profile")
_TARGET_DSP = MVAU_PROBLEM.target_dsp_block.allow_absent().dependency("target_dsp_block")
_NARROW_WEIGHTS = MVAU_EFFECTIVE_NARROW_WEIGHTS.allow_absent().dependency("narrow_weights")
_REPETITIONS = MVAU_PROBLEM.repetitions.dependency("repetitions")
_MATRIX_WIDTH = MVAU_PROBLEM.matrix_width.dependency("matrix_width")
_MATRIX_HEIGHT = MVAU_PROBLEM.matrix_height.dependency("matrix_height")
_WEIGHT_INITIALIZER = MVAU_PROBLEM.weight_initializer_available.dependency(
    "weight_initializer_available"
)


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


def _divisor_domain(dimension: Ref[int]) -> DecisionDomain:
    dependency = dimension.dependency("dimension")

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        extent = cast(int, dependencies["dimension"])
        return Decided(type(value) is int and value > 0 and extent % value == 0)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        extent = cast(int, dependencies["dimension"])
        return Decided(tuple(value for value in range(1, extent + 1) if extent % value == 0))

    return DecisionDomain((dependency,), accepts, EvaluatorSpec((dependency,), candidates))


def _missing_problem(
    owner: QualifiedPath, field: QualifiedPath, code: str, message: str
) -> Unresolved:
    return Unresolved((Finding(FindingKind.LIMITATION, code, owner, message, trace=(field,)),))


@dataclass(frozen=True)
class MVAUComputeKernelPathSet:
    """Every path one compute Kernel owns locally."""

    kernel_id: str

    def _local(self, suffix: str) -> QualifiedPath:
        return QualifiedPath(f"{MVAU_COMPUTE_SELECTION_NAME}.{self.kernel_id}.{suffix}")

    def _semantic(self, suffix: str) -> QualifiedPath:
        return QualifiedPath(f"semantic.{MVAU_COMPUTE_SELECTION_NAME}.{self.kernel_id}.{suffix}")

    def _constraint(self, suffix: str) -> QualifiedPath:
        return QualifiedPath(f"constraint.{MVAU_COMPUTE_SELECTION_NAME}.{self.kernel_id}.{suffix}")

    @property
    def pe(self) -> QualifiedPath:
        return self._local("pe")

    @property
    def simd(self) -> QualifiedPath:
        return self._local("simd")

    @property
    def resource(self) -> QualifiedPath:
        return self._local("resource")

    @property
    def weight_source(self) -> QualifiedPath:
        return self._local("weight_source")

    @property
    def interleave(self) -> QualifiedPath:
        return self._local("interleave")

    @property
    def compute_pumping(self) -> QualifiedPath:
        return self._local("compute_pumping")

    @property
    def region(self) -> QualifiedPath:
        return self._semantic("region")

    @property
    def region_form(self) -> QualifiedPath:
        return self._semantic("region_form")

    @property
    def weight_demand(self) -> QualifiedPath:
        return self._semantic("weight_demand")

    @property
    def full_tile_weight_port(self) -> QualifiedPath:
        return self._semantic("full_tile_weight_port")

    @property
    def neuron_folds(self) -> QualifiedPath:
        return self._semantic("neuron_folds")

    @property
    def synapse_folds(self) -> QualifiedPath:
        return self._semantic("synapse_folds")

    def constraint(self, name: str) -> QualifiedPath:
        return self._constraint(name)


def _pe_ref(paths: MVAUComputeKernelPathSet) -> DependencyRef:
    return DependencyRef.decision("pe", paths.pe, _INTEGER)


def _simd_ref(paths: MVAUComputeKernelPathSet) -> DependencyRef:
    return DependencyRef.decision("simd", paths.simd, _INTEGER)


def _region_ref(paths: MVAUComputeKernelPathSet) -> DependencyRef:
    return DependencyRef.property("region", paths.region, _REGION)


# -- shared source-admission constraints -------------------------------------


def _computation_types_supported(dependencies: DependencyView) -> Answer[bool]:
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    if profile is MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR:
        return Decided(activation.type_id == "bipolar" and weight.type_id == "bipolar")
    return Decided(activation.type_id != "binary" and weight.type_id != "binary")


def _accumulator_output_type_supported(dependencies: DependencyView) -> Answer[bool]:
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    if profile is MVAUComputationProfile.FUSED_THRESHOLD:
        return Decided(True)
    accumulator = cast(NumericElementType, dependencies["accumulator_element_type"])
    output = cast(NumericElementType, dependencies["output_element_type"])
    return Decided(accumulator == output)


def _fused_threshold_source_supported(
    owner: QualifiedPath,
) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        if (
            cast(MVAUComputationProfile, dependencies["computation_profile"])
            is not MVAUComputationProfile.FUSED_THRESHOLD
        ):
            return Decided(True)
        threshold = dependencies["threshold_element_type"]
        initialized = dependencies["threshold_initializer_available"]
        if threshold is ABSENT:
            return _missing_problem(
                owner,
                MVAU_PROBLEM.threshold_element_type.path,
                "mvau-threshold-type-missing",
                "fused-threshold computation requires a threshold element type",
            )
        if initialized is ABSENT:
            return _missing_problem(
                owner,
                MVAU_PROBLEM.threshold_initializer_available.path,
                "mvau-threshold-initializer-fact-missing",
                "fused-threshold computation requires an initializer-availability fact",
            )
        return Decided(cast(bool, initialized))

    return EvaluatorSpec((_COMPUTATION_PROFILE, _THRESHOLD_TYPE, _THRESHOLD_INITIALIZER), evaluate)


def _shared_source_constraints(
    paths: MVAUComputeKernelPathSet,
) -> tuple[Constraint, ...]:
    return (
        Constraint(
            paths.constraint("computation_types_supported"),
            EvaluatorSpec(
                (_COMPUTATION_PROFILE, _ACTIVATION_TYPE, _WEIGHT_TYPE),
                _computation_types_supported,
            ),
        ),
        Constraint(
            paths.constraint("accumulator_output_type_supported"),
            EvaluatorSpec(
                (_COMPUTATION_PROFILE, _ACCUMULATOR_TYPE, _OUTPUT_TYPE),
                _accumulator_output_type_supported,
            ),
        ),
        Constraint(
            paths.constraint("fused_threshold_source_supported"),
            _fused_threshold_source_supported(paths.constraint("fused_threshold_source_supported")),
        ),
    )


def _shared_source_constraint_paths(
    paths: MVAUComputeKernelPathSet,
) -> tuple[QualifiedPath, ...]:
    return (
        paths.constraint("computation_types_supported"),
        paths.constraint("accumulator_output_type_supported"),
        paths.constraint("fused_threshold_source_supported"),
    )


def _fold_properties(paths: MVAUComputeKernelPathSet) -> tuple[DerivedProperty, ...]:
    def neuron_folds(dependencies: DependencyView) -> Answer[object]:
        return Decided(cast(int, dependencies["matrix_height"]) // cast(int, dependencies["pe"]))

    def synapse_folds(dependencies: DependencyView) -> Answer[object]:
        return Decided(cast(int, dependencies["matrix_width"]) // cast(int, dependencies["simd"]))

    return (
        DerivedProperty(
            paths.neuron_folds,
            _INTEGER,
            EvaluatorSpec((_MATRIX_HEIGHT, _pe_ref(paths)), neuron_folds),
        ),
        DerivedProperty(
            paths.synapse_folds,
            _INTEGER,
            EvaluatorSpec((_MATRIX_WIDTH, _simd_ref(paths)), synapse_folds),
        ),
    )


def _weight_demand_property(
    paths: MVAUComputeKernelPathSet,
    *,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
) -> DerivedProperty:
    """Derive the streamed weight contract straight from the selected Region."""

    def derive(dependencies: DependencyView) -> Answer[object]:
        region = cast(DataflowRegion, dependencies["region"])
        return Decided(region.input_interface(WEIGHT_INTERFACE).port)

    return DerivedProperty(
        paths.weight_demand,
        _PORT,
        EvaluatorSpec((_region_ref(paths),), derive),
        applies_if=applies_if,
    )


def _full_tile_weight_property(
    paths: MVAUComputeKernelPathSet,
    *,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
) -> DerivedProperty:
    def derive(dependencies: DependencyView) -> Answer[object]:
        return Decided(
            construct_standard_mvau_weight_port(
                cast(int, dependencies["repetitions"]),
                cast(int, dependencies["matrix_width"]),
                cast(int, dependencies["matrix_height"]),
                cast(NumericElementType, dependencies["weight_element_type"]),
                cast(int, dependencies["pe"]),
                cast(int, dependencies["simd"]),
            )
        )

    return DerivedProperty(
        paths.full_tile_weight_port,
        _PORT,
        EvaluatorSpec(
            (
                _REPETITIONS,
                _MATRIX_WIDTH,
                _MATRIX_HEIGHT,
                _WEIGHT_TYPE,
                _pe_ref(paths),
                _simd_ref(paths),
            ),
            derive,
        ),
        applies_if=applies_if,
    )


def _region_form_property(
    paths: MVAUComputeKernelPathSet,
    evaluator: EvaluatorSpec[Answer[object]],
) -> DerivedProperty:
    return DerivedProperty(paths.region_form, _REGION_FORM, evaluator)


def _standard_region_dependencies(
    paths: MVAUComputeKernelPathSet,
) -> tuple[DependencyRef, ...]:
    return (
        _REPETITIONS,
        _MATRIX_WIDTH,
        _MATRIX_HEIGHT,
        _ACTIVATION_TYPE,
        _WEIGHT_TYPE,
        _OUTPUT_TYPE,
        _pe_ref(paths),
        _simd_ref(paths),
    )


def _standard_region_arguments(dependencies: DependencyView) -> tuple[object, ...]:
    return (
        cast(int, dependencies["repetitions"]),
        cast(int, dependencies["matrix_width"]),
        cast(int, dependencies["matrix_height"]),
        cast(NumericElementType, dependencies["activation_element_type"]),
        cast(NumericElementType, dependencies["weight_element_type"]),
        cast(NumericElementType, dependencies["output_element_type"]),
        cast(int, dependencies["pe"]),
        cast(int, dependencies["simd"]),
    )


# -- LegacyHLSMVAUKernel -----------------------------------------------------

LEGACY_HLS_PATHS = MVAUComputeKernelPathSet(MVAUComputeKernelId.LEGACY_HLS.value)


def _legacy_hls_region(dependencies: DependencyView) -> Answer[object]:
    arguments = _standard_region_arguments(dependencies)
    source = cast(MVAUWeightSource, dependencies["weight_source"])
    construct = (
        construct_standard_embedded_mvau_region
        if source is MVAUWeightSource.EMBEDDED
        else construct_standard_streamed_mvau_region
    )
    return Decided(construct(*arguments))  # type: ignore[arg-type]


def _legacy_hls_region_form(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        MVAURegionDeclaration.STANDARD_EMBEDDED
        if cast(MVAUWeightSource, dependencies["weight_source"]) is MVAUWeightSource.EMBEDDED
        else MVAURegionDeclaration.STANDARD_STREAMED
    )


def _legacy_hls_weight_source_domain() -> DecisionDomain:
    def available(dependencies: DependencyView) -> tuple[object, ...]:
        if cast(bool, dependencies["weight_initializer_available"]):
            return (MVAUWeightSource.EMBEDDED, MVAUWeightSource.STREAMED)
        return (MVAUWeightSource.STREAMED,)

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in available(dependencies))

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(available(dependencies))

    return DecisionDomain(
        (_WEIGHT_INITIALIZER,), accepts, EvaluatorSpec((_WEIGHT_INITIALIZER,), candidates)
    )


def _hls_numeric_supported(dependencies: DependencyView) -> Answer[bool]:
    supported = {"int", "uint", "bipolar"}
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    return Decided(activation.type_id in supported and weight.type_id in supported)


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
    if (
        cast(MVAUComputationProfile, dependencies["computation_profile"])
        is not MVAUComputationProfile.FUSED_THRESHOLD
    ):
        return Decided(True)
    threshold = dependencies["threshold_element_type"]
    if threshold is ABSENT:
        return _missing_problem(
            LEGACY_HLS_PATHS.constraint("threshold_representable"),
            MVAU_PROBLEM.threshold_element_type.path,
            "mvau-threshold-type-missing",
            "fused-threshold feasibility requires a threshold element type",
        )
    accumulator = cast(NumericElementType, dependencies["accumulator_element_type"])
    return Decided(cast(NumericElementType, threshold).bit_width >= accumulator.bit_width)


def build_legacy_hls_mvau_kernel() -> KernelDeclaration:
    """The existing FINN HLS matrix-vector-activation microarchitecture."""

    paths = LEGACY_HLS_PATHS
    weight_source_ref = DependencyRef.decision("weight_source", paths.weight_source, _WEIGHT_SOURCE)

    def is_streamed(dependencies: DependencyView) -> Answer[bool]:
        return Decided(dependencies["weight_source"] is MVAUWeightSource.STREAMED)

    streamed_only: EvaluatorSpec[Answer[bool]] = EvaluatorSpec((weight_source_ref,), is_streamed)
    spec = DesignSpaceSpec(
        decisions=(
            Decision(paths.pe, _INTEGER, _divisor_domain(MVAU_PROBLEM.matrix_height)),
            Decision(paths.simd, _INTEGER, _divisor_domain(MVAU_PROBLEM.matrix_width)),
            Decision(paths.resource, _HLS_RESOURCE, _finite_domain(tuple(MVAUHlsResource))),
            Decision(paths.weight_source, _WEIGHT_SOURCE, _legacy_hls_weight_source_domain()),
        ),
        properties=(
            *_fold_properties(paths),
            DerivedProperty(
                paths.region,
                _REGION,
                EvaluatorSpec(
                    (*_standard_region_dependencies(paths), weight_source_ref),
                    _legacy_hls_region,
                ),
            ),
            _region_form_property(
                paths, EvaluatorSpec((weight_source_ref,), _legacy_hls_region_form)
            ),
            _weight_demand_property(paths, applies_if=streamed_only),
            _full_tile_weight_property(paths, applies_if=streamed_only),
        ),
        constraints=(
            *_shared_source_constraints(paths),
            Constraint(
                paths.constraint("numeric_supported"),
                EvaluatorSpec((_ACTIVATION_TYPE, _WEIGHT_TYPE), _hls_numeric_supported),
            ),
            Constraint(
                paths.constraint("threshold_representable"),
                EvaluatorSpec(
                    (_COMPUTATION_PROFILE, _THRESHOLD_TYPE, _ACCUMULATOR_TYPE),
                    _hls_threshold_supported,
                ),
            ),
            Constraint(
                paths.constraint("partition_supported"),
                EvaluatorSpec((_MATRIX_WIDTH, _simd_ref(paths)), _hls_partition_supported),
            ),
            Constraint(
                paths.constraint("interface_width_supported"),
                EvaluatorSpec(
                    (
                        _ACTIVATION_TYPE,
                        _WEIGHT_TYPE,
                        _ACCUMULATOR_TYPE,
                        _OUTPUT_TYPE,
                        _pe_ref(paths),
                        _simd_ref(paths),
                    ),
                    _hls_width_supported,
                ),
            ),
        ),
    )
    source_admission = (
        *_shared_source_constraint_paths(paths),
        paths.constraint("numeric_supported"),
        paths.constraint("threshold_representable"),
    )
    return KernelDeclaration(
        MVAUComputeKernelId.LEGACY_HLS.value,
        "1",
        spec,
        paths.region,
        feasibility_constraints=(
            *source_admission,
            paths.constraint("partition_supported"),
            paths.constraint("interface_width_supported"),
        ),
        source_admission_constraints=source_admission,
        demands=(KernelDemand(WEIGHT_INTERFACE, paths.weight_demand),),
        exports=(
            KernelExport(REGION_FORM_EXPORT, paths.region_form, _REGION_FORM),
            KernelExport(FULL_TILE_WEIGHT_EXPORT, paths.full_tile_weight_port, _PORT),
        ),
        providers=(),
    )


# -- shared RTL feasibility --------------------------------------------------


def _rtl_numeric_supported(dependencies: DependencyView) -> Answer[bool]:
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    return Decided(
        activation.type_id in {"int", "uint"}
        and weight.type_id == "int"
        and activation.bit_width >= 2
        and weight.bit_width >= 2
    )


def _rtl_accumulator_profile_supported(dependencies: DependencyView) -> Answer[bool]:
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    return Decided(profile is MVAUComputationProfile.ACCUMULATOR_INTEGER)


def _rtl_width_supported(owner: QualifiedPath) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        target = dependencies["target_dsp_block"]
        if target is ABSENT:
            return _missing_problem(
                owner,
                MVAU_PROBLEM.target_dsp_block.path,
                "mvau-target-dsp-missing",
                "RTL width validation requires a target DSP block",
            )
        a_width, b_width, p_width = {
            MVAUDspBlock.DSP48E1: (25, 18, 48),
            MVAUDspBlock.DSP48E2: (27, 18, 48),
            MVAUDspBlock.DSP58: (27, 24, 58),
        }[cast(MVAUDspBlock, target)]
        activation = cast(NumericElementType, dependencies["activation_element_type"])
        weight = cast(NumericElementType, dependencies["weight_element_type"])
        accumulator = cast(NumericElementType, dependencies["accumulator_element_type"])
        output = cast(NumericElementType, dependencies["output_element_type"])
        return Decided(
            2 <= weight.bit_width <= a_width
            and 2 <= activation.bit_width <= b_width
            and accumulator.bit_width <= p_width
            and output.bit_width <= p_width
        )

    return EvaluatorSpec(
        (_TARGET_DSP, _ACTIVATION_TYPE, _WEIGHT_TYPE, _ACCUMULATOR_TYPE, _OUTPUT_TYPE), evaluate
    )


def _dsp58_required(owner: QualifiedPath) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        target = dependencies["target_dsp_block"]
        if target is ABSENT:
            return _missing_problem(
                owner,
                MVAU_PROBLEM.target_dsp_block.path,
                "mvau-target-dsp-missing",
                "this Kernel requires a target DSP block",
            )
        return Decided(cast(MVAUDspBlock, target) is MVAUDspBlock.DSP58)

    return EvaluatorSpec((_TARGET_DSP,), evaluate)


def _any_dsp_required(owner: QualifiedPath) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        target = dependencies["target_dsp_block"]
        if target is ABSENT:
            return _missing_problem(
                owner,
                MVAU_PROBLEM.target_dsp_block.path,
                "mvau-target-dsp-missing",
                "this Kernel requires a target DSP block",
            )
        return Decided(cast(MVAUDspBlock, target) in frozenset(MVAUDspBlock))

    return EvaluatorSpec((_TARGET_DSP,), evaluate)


def _rtl_shared_constraints(
    paths: MVAUComputeKernelPathSet,
) -> tuple[Constraint, ...]:
    return (
        *_shared_source_constraints(paths),
        Constraint(
            paths.constraint("numeric_supported"),
            EvaluatorSpec((_ACTIVATION_TYPE, _WEIGHT_TYPE), _rtl_numeric_supported),
        ),
        Constraint(
            paths.constraint("computation_supported"),
            EvaluatorSpec((_COMPUTATION_PROFILE,), _rtl_accumulator_profile_supported),
        ),
        Constraint(
            paths.constraint("width_supported"),
            _rtl_width_supported(paths.constraint("width_supported")),
        ),
    )


def _rtl_shared_constraint_paths(
    paths: MVAUComputeKernelPathSet,
) -> tuple[QualifiedPath, ...]:
    return (
        *_rtl_source_admission_paths(paths),
        paths.constraint("width_supported"),
    )


def _rtl_source_admission_paths(
    paths: MVAUComputeKernelPathSet,
) -> tuple[QualifiedPath, ...]:
    """The RTL constraints a source graph alone can answer.

    Datapath width and DSP block are target facts, so they stay out of
    admission and are asked at selection time instead.
    """

    return (
        *_shared_source_constraint_paths(paths),
        paths.constraint("numeric_supported"),
        paths.constraint("computation_supported"),
    )


def _compute_pumping_supported(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        not cast(bool, dependencies["compute_pumping"]) or cast(int, dependencies["simd"]) > 1
    )


def _standard_streamed_region(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        construct_standard_streamed_mvau_region(*_standard_region_arguments(dependencies))  # type: ignore[arg-type]
    )


def _always_standard_streamed(_dependencies: DependencyView) -> Answer[object]:
    return Decided(MVAURegionDeclaration.STANDARD_STREAMED)


def _build_standard_rtl_kernel(
    kernel_id: str,
    *,
    extra_constraints: tuple[Constraint, ...],
    extra_constraint_paths: tuple[QualifiedPath, ...],
    extra_source_admission: tuple[QualifiedPath, ...],
    providers: tuple[KernelProvider, ...],
) -> KernelDeclaration:
    paths = MVAUComputeKernelPathSet(kernel_id)
    pumping_ref = DependencyRef.decision("compute_pumping", paths.compute_pumping, _BOOL)
    spec = DesignSpaceSpec(
        decisions=(
            Decision(paths.pe, _INTEGER, _divisor_domain(MVAU_PROBLEM.matrix_height)),
            Decision(paths.simd, _INTEGER, _divisor_domain(MVAU_PROBLEM.matrix_width)),
            Decision(paths.compute_pumping, _BOOL, _finite_domain((False, True))),
        ),
        properties=(
            *_fold_properties(paths),
            DerivedProperty(
                paths.region,
                _REGION,
                EvaluatorSpec(_standard_region_dependencies(paths), _standard_streamed_region),
            ),
            _region_form_property(paths, EvaluatorSpec((), _always_standard_streamed)),
            _weight_demand_property(paths),
            _full_tile_weight_property(paths),
        ),
        constraints=(
            *_rtl_shared_constraints(paths),
            *extra_constraints,
            Constraint(
                paths.constraint("compute_pumping_supported"),
                EvaluatorSpec((pumping_ref, _simd_ref(paths)), _compute_pumping_supported),
            ),
        ),
    )
    source_admission = (*_rtl_source_admission_paths(paths), *extra_source_admission)
    return KernelDeclaration(
        kernel_id,
        "1",
        spec,
        paths.region,
        feasibility_constraints=(
            *_rtl_shared_constraint_paths(paths),
            *extra_constraint_paths,
            paths.constraint("compute_pumping_supported"),
        ),
        source_admission_constraints=source_admission,
        demands=(KernelDemand(WEIGHT_INTERFACE, paths.weight_demand),),
        exports=(
            KernelExport(REGION_FORM_EXPORT, paths.region_form, _REGION_FORM),
            KernelExport(FULL_TILE_WEIGHT_EXPORT, paths.full_tile_weight_port, _PORT),
        ),
        providers=providers,
    )


# -- SoftVectorMVAUKernel ----------------------------------------------------

SOFT_VECTOR_PATHS = MVAUComputeKernelPathSet(MVAUComputeKernelId.SOFT_VECTOR.value)


def _dsp48e1_narrow_supported(dependencies: DependencyView) -> Answer[bool]:
    owner = SOFT_VECTOR_PATHS.constraint("dsp48e1_narrow_supported")
    narrow = dependencies["narrow_weights"]
    if narrow is not ABSENT and cast(bool, narrow):
        return Decided(True)
    target = dependencies["target_dsp_block"]
    if target is ABSENT:
        return _missing_problem(
            owner,
            MVAU_PROBLEM.target_dsp_block.path,
            "mvau-target-dsp-missing",
            "soft-vector feasibility requires a target DSP block",
        )
    if target is not MVAUDspBlock.DSP48E1:
        return Decided(True)
    if narrow is ABSENT:
        return _missing_problem(
            owner,
            MVAU_EFFECTIVE_NARROW_WEIGHTS.path,
            "mvau-weights-narrow-missing",
            "DSP48E1 soft-vector feasibility requires a weights-narrow fact",
        )
    return Decided(cast(bool, narrow))


def build_soft_vector_mvau_kernel() -> KernelDeclaration:
    """RTL soft-vector accumulation over a standard streamed Region."""

    paths = SOFT_VECTOR_PATHS
    return _build_standard_rtl_kernel(
        paths.kernel_id,
        extra_constraints=(
            Constraint(
                paths.constraint("target_supported"),
                _any_dsp_required(paths.constraint("target_supported")),
            ),
            Constraint(
                paths.constraint("dsp48e1_narrow_supported"),
                EvaluatorSpec((_TARGET_DSP, _NARROW_WEIGHTS), _dsp48e1_narrow_supported),
            ),
        ),
        extra_constraint_paths=(
            paths.constraint("target_supported"),
            paths.constraint("dsp48e1_narrow_supported"),
        ),
        # Both remaining constraints read the target DSP block, so neither is
        # a source-admission question.
        extra_source_admission=(),
        providers=(KernelProvider(SOFT_VECTOR_PROVIDER_ID, paths.kernel_id),),
    )


# -- PackedDSPMVAUKernel -----------------------------------------------------

PACKED_DSP_PATHS = MVAUComputeKernelPathSet(MVAUComputeKernelId.PACKED_DSP.value)


def _packed_supported(dependencies: DependencyView) -> Answer[bool]:
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    if weight.bit_width > 8 or activation.bit_width > 9:
        return Decided(False)
    narrow = dependencies["narrow_weights"]
    if narrow is ABSENT:
        return _missing_problem(
            PACKED_DSP_PATHS.constraint("packing_supported"),
            MVAU_EFFECTIVE_NARROW_WEIGHTS.path,
            "mvau-weights-narrow-missing",
            "packed-DSP feasibility requires a weights-narrow fact",
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


def build_packed_dsp_mvau_kernel() -> KernelDeclaration:
    """Packed DSP accumulation over a standard streamed Region."""

    paths = PACKED_DSP_PATHS
    return _build_standard_rtl_kernel(
        paths.kernel_id,
        extra_constraints=(
            Constraint(
                paths.constraint("target_supported"),
                _dsp58_required(paths.constraint("target_supported")),
            ),
            Constraint(
                paths.constraint("packing_supported"),
                EvaluatorSpec((_ACTIVATION_TYPE, _WEIGHT_TYPE, _NARROW_WEIGHTS), _packed_supported),
            ),
        ),
        extra_constraint_paths=(
            paths.constraint("target_supported"),
            paths.constraint("packing_supported"),
        ),
        # Packing is decidable from the source element types alone; requiring
        # DSP58 is not, so only the former is a source-admission question.
        extra_source_admission=(paths.constraint("packing_supported"),),
        providers=(),
    )


# -- BatchInterleavedDSPMVAUKernel -------------------------------------------

BATCH_INTERLEAVED_PATHS = MVAUComputeKernelPathSet(MVAUComputeKernelId.BATCH_INTERLEAVED_DSP.value)


def _interleave_domain(paths: MVAUComputeKernelPathSet) -> DecisionDomain:
    pe_ref = _pe_ref(paths)
    simd_ref = _simd_ref(paths)

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
                    paths.interleave,
                    "batch interleave must be greater than one",
                    (("candidate", candidate),),
                )
            )
        if candidate > 0 and repeated % candidate:
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "mvau-interleave-does-not-divide-repetitions",
                    paths.interleave,
                    "batch interleave must divide the repetition extent",
                    (("candidate", candidate), ("repetitions", repeated)),
                )
            )
        if candidate > 0 and (pe * simd) % candidate:
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "mvau-interleave-does-not-divide-weight-tile",
                    paths.interleave,
                    "batch interleave must divide PE * SIMD",
                    (("candidate", candidate), ("pe", pe), ("simd", simd)),
                )
            )
        return Absent(tuple(findings)) if findings else Decided(True)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        limit = gcd(
            cast(int, dependencies["repetitions"]),
            cast(int, dependencies["pe"]) * cast(int, dependencies["simd"]),
        )
        return Decided(tuple(value for value in range(2, limit + 1) if limit % value == 0))

    dependencies = (_REPETITIONS, pe_ref, simd_ref)
    return DecisionDomain(dependencies, accepts, EvaluatorSpec(dependencies, candidates))


def _interleave_available(dependencies: DependencyView) -> Answer[bool]:
    """Whether any batch interleave greater than one could ever be chosen.

    The interleave must exceed one and divide both the repetition extent and
    PE * SIMD.  PE * SIMD is at most the whole matrix, so this is the exact
    existence question the source graph can answer on its own.
    """

    return Decided(
        gcd(
            cast(int, dependencies["repetitions"]),
            cast(int, dependencies["matrix_width"]) * cast(int, dependencies["matrix_height"]),
        )
        > 1
    )


def _tiled_width_supported(dependencies: DependencyView) -> Answer[bool]:
    activation = cast(NumericElementType, dependencies["activation_element_type"])
    weight = cast(NumericElementType, dependencies["weight_element_type"])
    return Decided(activation.bit_width <= 8 and weight.bit_width <= 8)


def _batch_interleaved_region(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        construct_batch_interleaved_streamed_mvau_region(
            cast(int, dependencies["repetitions"]),
            cast(int, dependencies["matrix_width"]),
            cast(int, dependencies["matrix_height"]),
            cast(NumericElementType, dependencies["activation_element_type"]),
            cast(NumericElementType, dependencies["weight_element_type"]),
            cast(NumericElementType, dependencies["output_element_type"]),
            cast(int, dependencies["pe"]),
            cast(int, dependencies["simd"]),
            cast(int, dependencies["interleave"]),
        )
    )


def _always_batch_interleaved(_dependencies: DependencyView) -> Answer[object]:
    return Decided(MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED)


def build_batch_interleaved_dsp_mvau_kernel() -> KernelDeclaration:
    """Batch-interleaved DSP58 accumulation and its own chunked weight demand."""

    paths = BATCH_INTERLEAVED_PATHS
    interleave_ref = DependencyRef.decision("interleave", paths.interleave, _INTEGER)
    spec = DesignSpaceSpec(
        decisions=(
            Decision(paths.pe, _INTEGER, _divisor_domain(MVAU_PROBLEM.matrix_height)),
            Decision(paths.simd, _INTEGER, _divisor_domain(MVAU_PROBLEM.matrix_width)),
            Decision(paths.interleave, _INTEGER, _interleave_domain(paths)),
        ),
        properties=(
            *_fold_properties(paths),
            DerivedProperty(
                paths.region,
                _REGION,
                EvaluatorSpec(
                    (*_standard_region_dependencies(paths), interleave_ref),
                    _batch_interleaved_region,
                ),
            ),
            _region_form_property(paths, EvaluatorSpec((), _always_batch_interleaved)),
            _weight_demand_property(paths),
            _full_tile_weight_property(paths),
        ),
        constraints=(
            *_rtl_shared_constraints(paths),
            Constraint(
                paths.constraint("target_supported"),
                _dsp58_required(paths.constraint("target_supported")),
            ),
            Constraint(
                paths.constraint("tiled_width_supported"),
                EvaluatorSpec((_ACTIVATION_TYPE, _WEIGHT_TYPE), _tiled_width_supported),
            ),
            Constraint(
                paths.constraint("interleave_available"),
                EvaluatorSpec((_REPETITIONS, _MATRIX_WIDTH, _MATRIX_HEIGHT), _interleave_available),
            ),
        ),
    )
    source_admission = (
        *_rtl_source_admission_paths(paths),
        paths.constraint("tiled_width_supported"),
        paths.constraint("interleave_available"),
    )
    return KernelDeclaration(
        paths.kernel_id,
        "1",
        spec,
        paths.region,
        feasibility_constraints=(
            *source_admission,
            paths.constraint("width_supported"),
            paths.constraint("target_supported"),
        ),
        source_admission_constraints=source_admission,
        demands=(KernelDemand(WEIGHT_INTERFACE, paths.weight_demand),),
        exports=(
            KernelExport(REGION_FORM_EXPORT, paths.region_form, _REGION_FORM),
            KernelExport(FULL_TILE_WEIGHT_EXPORT, paths.full_tile_weight_port, _PORT),
        ),
        providers=(),
    )


LEGACY_HLS_MVAU_KERNEL = build_legacy_hls_mvau_kernel()
SOFT_VECTOR_MVAU_KERNEL = build_soft_vector_mvau_kernel()
PACKED_DSP_MVAU_KERNEL = build_packed_dsp_mvau_kernel()
BATCH_INTERLEAVED_DSP_MVAU_KERNEL = build_batch_interleaved_dsp_mvau_kernel()

#: The decomposed member and the replay pool that accompanies it.  Declared
#: here rather than elsewhere because the dot product belongs to *this* pool:
#: it is the replacement for the two RTL standard-streamed Kernels, and picking
#: it is the same kind of choice as picking either of them.
DECOMPOSED_MVAU_KERNELS = build_decomposed_mvau_kernels(MVAU_COMPUTE_SELECTION_NAME)

#: The optional second node the decomposed member needs.
MVAU_REPLAY_SELECTION = DECOMPOSED_MVAU_KERNELS.activation_replay

MVAU_COMPUTE_SELECTION = KernelSelection(
    MVAU_COMPUTE_SELECTION_NAME,
    (
        LEGACY_HLS_MVAU_KERNEL,
        # Being replaced by the decomposed member below; they remain only
        # because elaboration still runs through them.
        SOFT_VECTOR_MVAU_KERNEL,
        PACKED_DSP_MVAU_KERNEL,
        BATCH_INTERLEAVED_DSP_MVAU_KERNEL,
        DECOMPOSED_MVAU_KERNELS.dot_product,
    ),
)

MVAU_COMPUTE_KERNEL_PATH_SETS = {
    kernel.id: MVAUComputeKernelPathSet(kernel.id) for kernel in MVAU_COMPUTE_SELECTION.kernels
}

__all__ = [
    "BATCH_INTERLEAVED_DSP_MVAU_KERNEL",
    "BATCH_INTERLEAVED_PATHS",
    "FULL_TILE_WEIGHT_EXPORT",
    "LEGACY_HLS_MVAU_KERNEL",
    "LEGACY_HLS_PATHS",
    "MVAU_COMPUTE_KERNEL_PATH_SETS",
    "MVAU_COMPUTE_SELECTION",
    "MVAU_COMPUTE_SELECTION_NAME",
    "MVAUComputeKernelId",
    "MVAUComputeKernelPathSet",
    "MVAUDspBlock",
    "MVAUHlsResource",
    "MVAUWeightSource",
    "PACKED_DSP_MVAU_KERNEL",
    "PACKED_DSP_PATHS",
    "REGION_FORM_EXPORT",
    "SOFT_VECTOR_MVAU_KERNEL",
    "SOFT_VECTOR_PATHS",
    "SOFT_VECTOR_PROVIDER_ID",
    "WEIGHT_INTERFACE",
    "build_batch_interleaved_dsp_mvau_kernel",
    "build_legacy_hls_mvau_kernel",
    "build_packed_dsp_mvau_kernel",
    "build_soft_vector_mvau_kernel",
]
