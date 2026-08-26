# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone MVAU design-space example built on :mod:`finn.dataflow._engine`.

The model is informed by FINN's MatrixVectorActivation implementation and
``SpecializeLayers`` policy, but imports no FINN domain modules. It intentionally
models only the choice structure and representative feasibility relationships—not
resource or throughput estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast

from finn.dataflow._engine import (
    ABSENT,
    AbsenceMode,
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
)
from finn.dataflow._engine.declarations import (
    ApplicabilityResult,
    CandidateResult,
    ConstraintResult,
    DomainResult,
    ValueResult,
)

E = TypeVar("E", bound=Enum)
T = TypeVar("T")


class Backend(str, Enum):
    HLS = "hls"
    RTL = "rtl"


class ResourceStyle(str, Enum):
    AUTO = "auto"
    LUT = "lut"
    DSP = "dsp"


class WeightDelivery(str, Enum):
    INTERNAL_EMBEDDED = "internal_embedded"
    INTERNAL_DECOUPLED = "internal_decoupled"
    EXTERNAL_STREAM = "external"
    EXTERNAL_MEMORY = "external_mem"
    DYNAMIC = "dynamic"


class RamStyle(str, Enum):
    AUTO = "auto"
    BLOCK = "block"
    DISTRIBUTED = "distributed"
    ULTRA = "ultra"


class ThresholdRamStyle(str, Enum):
    AUTO = "auto"
    BLOCK = "block"
    DISTRIBUTED = "distributed"


class ActivationMode(str, Enum):
    THRESHOLDS = "thresholds"
    ACCUMULATORS = "accumulators"


class BinaryMode(str, Enum):
    MULTIPLY = "multiply"
    XNOR_POPCOUNT = "xnor_popcount"


class NumericEncoding(str, Enum):
    INTEGER = "integer"
    BINARY = "binary"
    BIPOLAR = "bipolar"


class DspBlock(str, Enum):
    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


@dataclass(frozen=True, slots=True)
class NumericType:
    name: str
    bit_width: int
    signed: bool
    encoding: NumericEncoding = NumericEncoding.INTEGER

    @property
    def minimum(self) -> int:
        if self.encoding is NumericEncoding.BIPOLAR:
            return -1
        if not self.signed:
            return 0
        return -(1 << (self.bit_width - 1))


@dataclass(frozen=True, slots=True)
class WeightSummary:
    """Small substitute for inspecting the full FINN weight initializer."""

    minimum: int | None


class Paths:
    MATRIX_WIDTH = QualifiedPath("problem.matrix_width")
    MATRIX_HEIGHT = QualifiedPath("problem.matrix_height")
    INPUT_TYPE = QualifiedPath("problem.input_type")
    WEIGHT_TYPE = QualifiedPath("problem.weight_type")
    OUTPUT_TYPE = QualifiedPath("problem.output_type")
    WEIGHT_SUMMARY = QualifiedPath("problem.weight_summary")
    FPGA_PART = QualifiedPath("problem.fpga_part")
    TARGET_CLOCK_NS = QualifiedPath("problem.target_clock_ns")
    INPUT_VECTORS = QualifiedPath("problem.input_vectors")
    RUNTIME_WEIGHT_UPDATES = QualifiedPath("problem.runtime_weight_updates_required")
    WEIGHT_SET_COUNT = QualifiedPath("problem.weight_set_count")

    PE = QualifiedPath("mvau.pe")
    SIMD = QualifiedPath("mvau.simd")
    BACKEND = QualifiedPath("mvau.backend")
    RESOURCE_STYLE = QualifiedPath("mvau.resource_style")
    WEIGHT_DELIVERY = QualifiedPath("mvau.weight_delivery")
    WEIGHT_RAM_STYLE = QualifiedPath("mvau.weight_ram_style")
    THRESHOLD_RAM_STYLE = QualifiedPath("mvau.threshold_ram_style")
    PUMPED_MEMORY = QualifiedPath("mvau.pumped_memory")
    PUMPED_COMPUTE = QualifiedPath("mvau.pumped_compute")
    TILE_HEIGHT = QualifiedPath("mvau.tile_height")
    ACTIVATION_MODE = QualifiedPath("mvau.activation_mode")
    BINARY_MODE = QualifiedPath("mvau.binary_mode")
    ACTIVATION_BIAS = QualifiedPath("mvau.activation_bias")

    DSP_BLOCK = QualifiedPath("derived.dsp_block")
    DSP_VERSION = QualifiedPath("derived.dsp_version")
    NARROW_WEIGHTS = QualifiedPath("derived.narrow_weights")
    RUNTIME_WRITABLE_WEIGHTS = QualifiedPath("derived.runtime_writeable_weights")
    WEIGHT_MEMORY_DEPTH = QualifiedPath("derived.weight_memory_depth")
    THRESHOLD_MEMORY_DEPTH = QualifiedPath("derived.threshold_memory_depth")
    SEGMENT_LENGTH = QualifiedPath("derived.segment_length")

    BACKEND_SUPPORTED = QualifiedPath("constraint.backend_supported")
    RESOURCE_STYLE_SUPPORTED = QualifiedPath("constraint.resource_style_supported")
    WEIGHT_SETS_SUPPORTED = QualifiedPath("constraint.weight_sets_supported")
    WEIGHT_DATA_SUPPORTED = QualifiedPath("constraint.weight_data_supported")
    RUNTIME_UPDATES_SUPPORTED = QualifiedPath("constraint.runtime_updates_supported")
    PUMPED_MEMORY_SUPPORTED = QualifiedPath("constraint.pumped_memory_supported")
    PUMPED_COMPUTE_SUPPORTED = QualifiedPath("constraint.pumped_compute_supported")
    TILE_HEIGHT_SUPPORTED = QualifiedPath("constraint.tile_height_supported")
    BINARY_MODE_SUPPORTED = QualifiedPath("constraint.binary_mode_supported")
    HLS_PARTITION_SUPPORTED = QualifiedPath("constraint.hls_partition_supported")
    CLOCK_SUPPORTED = QualifiedPath("constraint.clock_supported")


@dataclass(frozen=True, slots=True)
class MVAUProblem:
    matrix_width: int
    matrix_height: int
    input_type: NumericType
    weight_type: NumericType
    output_type: NumericType
    weight_summary: WeightSummary
    fpga_part: str
    target_clock_ns: float
    input_vectors: tuple[int, ...] = (1,)
    runtime_weight_updates_required: bool = False
    weight_set_count: int = 0

    def as_mapping(self) -> dict[QualifiedPath, object]:
        return {
            Paths.MATRIX_WIDTH: self.matrix_width,
            Paths.MATRIX_HEIGHT: self.matrix_height,
            Paths.INPUT_TYPE: self.input_type,
            Paths.WEIGHT_TYPE: self.weight_type,
            Paths.OUTPUT_TYPE: self.output_type,
            Paths.WEIGHT_SUMMARY: self.weight_summary,
            Paths.FPGA_PART: self.fpga_part,
            Paths.TARGET_CLOCK_NS: self.target_clock_ns,
            Paths.INPUT_VECTORS: self.input_vectors,
            Paths.RUNTIME_WEIGHT_UPDATES: self.runtime_weight_updates_required,
            Paths.WEIGHT_SET_COUNT: self.weight_set_count,
        }


INT = as_object_semantics(ValueSemantics.immutable_nominal(int))
FLOAT = as_object_semantics(ValueSemantics.immutable_nominal(float))
BOOL = as_object_semantics(ValueSemantics.immutable_nominal(bool))
STRING = as_object_semantics(ValueSemantics.immutable_nominal(str))
NUMERIC_TYPE = as_object_semantics(ValueSemantics.immutable_nominal(NumericType))
WEIGHT_SUMMARY = as_object_semantics(ValueSemantics.immutable_nominal(WeightSummary))


def _enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    semantics: ValueSemantics[E] = ValueSemantics(
        type_token=enum_type,
        name=enum_type.__name__,
        recognizes=lambda value: type(value) is enum_type,
        equal=lambda left, right: left is right,
        snapshot=lambda value: value,
    )
    return as_object_semantics(semantics)


BACKEND = _enum_semantics(Backend)
RESOURCE_STYLE = _enum_semantics(ResourceStyle)
WEIGHT_DELIVERY = _enum_semantics(WeightDelivery)
RAM_STYLE = _enum_semantics(RamStyle)
THRESHOLD_RAM_STYLE = _enum_semantics(ThresholdRamStyle)
ACTIVATION_MODE = _enum_semantics(ActivationMode)
BINARY_MODE = _enum_semantics(BinaryMode)
DSP_BLOCK = _enum_semantics(DspBlock)

INPUT_VECTORS: ValueSemantics[object] = ValueSemantics(
    type_token=tuple,
    name="tuple[int, ...]",
    recognizes=lambda value: (
        type(value) is tuple and all(type(item) is int for item in cast(tuple[object, ...], value))
    ),
    equal=lambda left, right: left == right,
    snapshot=lambda value: tuple(cast(tuple[object, ...], value)),
)


def _expect(dependencies: DependencyView, name: str, expected_type: type[T]) -> T:
    value = dependencies[name]
    if value is ABSENT or type(value) is not expected_type:
        raise TypeError(f"dependency {name!r} is not {expected_type.__name__}")
    return value


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _positive_float(value: object) -> bool:
    return type(value) is float and value > 0.0


def _positive_vector_shape(value: object) -> bool:
    return (
        type(value) is tuple
        and bool(value)
        and all(type(item) is int and item > 0 for item in cast(tuple[object, ...], value))
    )


def _finite_domain(path: QualifiedPath, allowed: frozenset[object]) -> DecisionDomain:
    def check(value: object, _dependencies: DependencyView) -> DomainResult:
        return Decided(value in allowed)

    return DecisionDomain((), check)


def _positive_integer_domain(path: QualifiedPath) -> DecisionDomain:
    def check(value: object, _dependencies: DependencyView) -> DomainResult:
        return Decided(type(value) is int and value > 0)

    return DecisionDomain((), check)


def _divisor_domain(
    path: QualifiedPath,
    dimension_path: QualifiedPath,
    dimension_name: str,
) -> DecisionDomain:
    dimension_ref = DependencyRef.problem("dimension", dimension_path, INT)

    def check(value: object, dependencies: DependencyView) -> DomainResult:
        dimension = _expect(dependencies, "dimension", int)
        return Decided(type(value) is int and value > 0 and dimension % value == 0)

    def enumerate_divisors(dependencies: DependencyView) -> CandidateResult:
        dimension = _expect(dependencies, "dimension", int)
        divisors = tuple(value for value in range(1, dimension + 1) if dimension % value == 0)
        return Decided(divisors)

    return DecisionDomain(
        (dimension_ref,),
        check,
        candidates=EvaluatorSpec((dimension_ref,), enumerate_divisors),
    )


def _constant_proposal(
    value: object,
    dependencies: tuple[DependencyRef, ...] = (),
) -> EvaluatorSpec[ValueResult]:
    def evaluate(_dependencies: DependencyView) -> ValueResult:
        return Decided(value)

    return EvaluatorSpec(dependencies, evaluate)


def _applicable_when(
    dependency: DependencyRef,
    expected: object,
) -> EvaluatorSpec[ApplicabilityResult]:
    def evaluate(dependencies: DependencyView) -> ApplicabilityResult:
        return Decided(dependencies[dependency.name] == expected)

    return EvaluatorSpec((dependency,), evaluate)


def _dsp_block_for_part(part: str) -> DspBlock:
    if part[:4] in {"xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm"} or part[:5] in {
        "xqrvc",
        "xcv80",
    }:
        return DspBlock.DSP58
    if len(part) > 2 and part[2] == "7":
        return DspBlock.DSP48E1
    return DspBlock.DSP48E2


def _rtl_supported(dependencies: DependencyView, *, automatic: bool) -> tuple[bool, str]:
    activation_mode = _expect(dependencies, "activation_mode", ActivationMode)
    binary_mode = _expect(dependencies, "binary_mode", BinaryMode)
    input_type = _expect(dependencies, "input_type", NumericType)
    weight_type = _expect(dependencies, "weight_type", NumericType)
    narrow_weights = _expect(dependencies, "narrow_weights", bool)
    dsp_block = _expect(dependencies, "dsp_block", DspBlock)

    if activation_mode is not ActivationMode.ACCUMULATORS:
        return False, "the RTL MVAU does not implement embedded thresholds"
    if binary_mode is BinaryMode.XNOR_POPCOUNT:
        return False, "the RTL MVAU does not implement XNOR-popcount mode"
    if not weight_type.signed:
        return False, "the RTL MVAU requires signed weights"
    minimum_width = 4 if automatic else 2
    if input_type.bit_width < minimum_width or weight_type.bit_width < minimum_width:
        return False, f"the selected RTL policy requires at least {minimum_width}-bit operands"
    if dsp_block is DspBlock.DSP48E1 and not narrow_weights:
        return False, "DSP48E1 requires narrow-range weights for the RTL MVAU"
    return True, ""


# --------------------------------------------------------------------------------------
# Dependency declarations
#
# Every evaluator states exactly what it reads. These are constants: a DependencyRef
# binds a local name to a path and its value semantics, and nothing about it varies at
# build time.
# --------------------------------------------------------------------------------------


_backend_ref = DependencyRef.decision("backend", Paths.BACKEND, BACKEND)
_pe_ref = DependencyRef.decision("pe", Paths.PE, INT)
_simd_ref = DependencyRef.decision("simd", Paths.SIMD, INT)
_resource_ref = DependencyRef.decision("resource_style", Paths.RESOURCE_STYLE, RESOURCE_STYLE)
_delivery_ref = DependencyRef.decision("weight_delivery", Paths.WEIGHT_DELIVERY, WEIGHT_DELIVERY)
_weight_ram_ref = DependencyRef.decision(
    "weight_ram_style",
    Paths.WEIGHT_RAM_STYLE,
    RAM_STYLE,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_pumped_memory_ref = DependencyRef.decision(
    "pumped_memory",
    Paths.PUMPED_MEMORY,
    BOOL,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_pumped_compute_ref = DependencyRef.decision(
    "pumped_compute",
    Paths.PUMPED_COMPUTE,
    BOOL,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_tile_height_ref = DependencyRef.decision(
    "tile_height",
    Paths.TILE_HEIGHT,
    INT,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_activation_mode_ref = DependencyRef.decision(
    "activation_mode", Paths.ACTIVATION_MODE, ACTIVATION_MODE
)
_binary_mode_ref = DependencyRef.decision("binary_mode", Paths.BINARY_MODE, BINARY_MODE)
_input_type_ref = DependencyRef.problem("input_type", Paths.INPUT_TYPE, NUMERIC_TYPE)
_weight_type_ref = DependencyRef.problem("weight_type", Paths.WEIGHT_TYPE, NUMERIC_TYPE)
_matrix_width_ref = DependencyRef.problem("matrix_width", Paths.MATRIX_WIDTH, INT)
_matrix_height_ref = DependencyRef.problem("matrix_height", Paths.MATRIX_HEIGHT, INT)
_fpga_part_ref = DependencyRef.problem("fpga_part", Paths.FPGA_PART, STRING)
_target_clock_ref = DependencyRef.problem("target_clock_ns", Paths.TARGET_CLOCK_NS, FLOAT)
_input_vectors_ref = DependencyRef.problem("input_vectors", Paths.INPUT_VECTORS, INPUT_VECTORS)
_runtime_updates_ref = DependencyRef.problem("runtime_updates", Paths.RUNTIME_WEIGHT_UPDATES, BOOL)
_weight_set_count_ref = DependencyRef.problem("weight_set_count", Paths.WEIGHT_SET_COUNT, INT)
_weight_summary_ref = DependencyRef.problem("weight_summary", Paths.WEIGHT_SUMMARY, WEIGHT_SUMMARY)
_dsp_block_ref = DependencyRef.property("dsp_block", Paths.DSP_BLOCK, DSP_BLOCK)
_narrow_weights_ref = DependencyRef.property("narrow_weights", Paths.NARROW_WEIGHTS, BOOL)
_backend_dependencies = (
    _activation_mode_ref,
    _binary_mode_ref,
    _input_type_ref,
    _weight_type_ref,
    _narrow_weights_ref,
    _dsp_block_ref,
)


# --------------------------------------------------------------------------------------
# Evaluators
#
# Pure functions over a DependencyView. None of them closes over build state, which is
# what lets each be read -- and tested -- on its own.
# --------------------------------------------------------------------------------------


def _choose_backend(dependencies: DependencyView) -> ValueResult:
    supported, _message = _rtl_supported(dependencies, automatic=True)
    return Decided(Backend.RTL if supported else Backend.HLS)


def _resolve_dsp_block(dependencies: DependencyView) -> ValueResult:
    return Decided(_dsp_block_for_part(_expect(dependencies, "fpga_part", str)))


def _resolve_narrow_weights(dependencies: DependencyView) -> ValueResult:
    summary = _expect(dependencies, "weight_summary", WeightSummary)
    weight_type = _expect(dependencies, "weight_type", NumericType)
    narrow = summary.minimum is not None and summary.minimum > weight_type.minimum
    return Decided(narrow)


def _resolve_weight_memory_depth(dependencies: DependencyView) -> ValueResult:
    mw = _expect(dependencies, "matrix_width", int)
    mh = _expect(dependencies, "matrix_height", int)
    pe_value = _expect(dependencies, "pe", int)
    simd_value = _expect(dependencies, "simd", int)
    return Decided(mw * mh // (pe_value * simd_value))


def _resolve_threshold_memory_depth(dependencies: DependencyView) -> ValueResult:
    mode = _expect(dependencies, "activation_mode", ActivationMode)
    if mode is ActivationMode.ACCUMULATORS:
        return Decided(0)
    mh = _expect(dependencies, "matrix_height", int)
    pe_value = _expect(dependencies, "pe", int)
    return Decided(mh // pe_value)


def _resolve_segment_length(dependencies: DependencyView) -> ValueResult:
    clock = _expect(dependencies, "target_clock_ns", float)
    simd_value = _expect(dependencies, "simd", int)
    pumped = _expect(dependencies, "pumped_compute", bool)
    reference_clock = clock / 2 if pumped else clock
    simd_factor = 6 if pumped else 3
    critical_path_dsps = math.floor((reference_clock - 0.741) / 0.605 + 1)
    maximum_chain = math.ceil(simd_value / simd_factor)
    return Decided(max(1, min(critical_path_dsps, maximum_chain)))


def _backend_supported(dependencies: DependencyView) -> ConstraintResult:
    backend_value = _expect(dependencies, "backend", Backend)
    if backend_value is Backend.HLS:
        return Decided(True)
    supported, _message = _rtl_supported(dependencies, automatic=False)
    return Decided(supported)


def _resource_style_supported(dependencies: DependencyView) -> ConstraintResult:
    backend_value = _expect(dependencies, "backend", Backend)
    style = _expect(dependencies, "resource_style", ResourceStyle)
    return Decided(not (backend_value is Backend.RTL and style is ResourceStyle.LUT))


def _weight_sets_supported(dependencies: DependencyView) -> ConstraintResult:
    count = _expect(dependencies, "weight_set_count", int)
    if count == 0:
        return Decided(True)
    backend_value = _expect(dependencies, "backend", Backend)
    delivery = _expect(dependencies, "weight_delivery", WeightDelivery)
    return Decided(backend_value is Backend.RTL and delivery is WeightDelivery.EXTERNAL_MEMORY)


def _weight_data_supported(dependencies: DependencyView) -> ConstraintResult:
    summary = _expect(dependencies, "weight_summary", WeightSummary)
    count = _expect(dependencies, "weight_set_count", int)
    delivery = _expect(dependencies, "weight_delivery", WeightDelivery)
    if summary.minimum is not None or count > 0:
        return Decided(True)
    return Decided(
        delivery
        in {
            WeightDelivery.EXTERNAL_STREAM,
            WeightDelivery.EXTERNAL_MEMORY,
            WeightDelivery.DYNAMIC,
        }
    )


def _runtime_updates_supported(dependencies: DependencyView) -> ConstraintResult:
    required = _expect(dependencies, "runtime_updates", bool)
    delivery = _expect(dependencies, "weight_delivery", WeightDelivery)
    return Decided(not required or delivery is WeightDelivery.INTERNAL_DECOUPLED)


def _resolve_runtime_writable_weights(dependencies: DependencyView) -> ValueResult:
    delivery = _expect(dependencies, "weight_delivery", WeightDelivery)
    required = _expect(dependencies, "runtime_updates", bool)
    style = dependencies["weight_ram_style"]
    dsp = _expect(dependencies, "dsp_block", DspBlock)
    required_by_uram = style is RamStyle.ULTRA and dsp is not DspBlock.DSP58
    enabled = delivery is WeightDelivery.INTERNAL_DECOUPLED and (required or required_by_uram)
    return Decided(enabled)


def _pumped_memory_supported(dependencies: DependencyView) -> ConstraintResult:
    pumped = _expect(dependencies, "pumped_memory", bool)
    pe_value = _expect(dependencies, "pe", int)
    simd_value = _expect(dependencies, "simd", int)
    return Decided(not (pumped and pe_value == 1 and simd_value == 1))


def _pumped_compute_supported(dependencies: DependencyView) -> ConstraintResult:
    pumped = _expect(dependencies, "pumped_compute", bool)
    simd_value = _expect(dependencies, "simd", int)
    return Decided(not (pumped and simd_value == 1))


def _tile_height_supported(dependencies: DependencyView) -> ConstraintResult:
    tile_height_value = _expect(dependencies, "tile_height", int)
    if tile_height_value == 1:
        return Decided(True)
    pe_value = _expect(dependencies, "pe", int)
    simd_value = _expect(dependencies, "simd", int)
    input_type = _expect(dependencies, "input_type", NumericType)
    weight_type = _expect(dependencies, "weight_type", NumericType)
    input_vectors = _expect(dependencies, "input_vectors", tuple)
    vector_count = math.prod(cast(tuple[int, ...], input_vectors))
    return Decided(
        (pe_value * simd_value) % tile_height_value == 0
        and input_type.bit_width <= 8
        and weight_type.bit_width <= 8
        and vector_count % tile_height_value == 0
    )


def _binary_mode_supported(dependencies: DependencyView) -> ConstraintResult:
    mode = _expect(dependencies, "binary_mode", BinaryMode)
    input_type = _expect(dependencies, "input_type", NumericType)
    weight_type = _expect(dependencies, "weight_type", NumericType)
    binary_like = {NumericEncoding.BINARY, NumericEncoding.BIPOLAR}
    if mode is BinaryMode.XNOR_POPCOUNT:
        if input_type.encoding in binary_like and weight_type.encoding in binary_like:
            return Decided(True)
        return Decided(False)
    return Decided(
        input_type.encoding is not NumericEncoding.BINARY
        and weight_type.encoding is not NumericEncoding.BINARY
    )


def _hls_partition_supported(dependencies: DependencyView) -> ConstraintResult:
    simd_value = _expect(dependencies, "simd", int)
    mw = _expect(dependencies, "matrix_width", int)
    return Decided(simd_value * 1024 >= mw)


def _clock_supported(dependencies: DependencyView) -> ConstraintResult:
    clock = _expect(dependencies, "target_clock_ns", float)
    pumped = _expect(dependencies, "pumped_compute", bool)
    reference_clock = clock / 2 if pumped else clock
    return Decided(reference_clock > 0.741)


def _mvau_decisions() -> tuple[Decision, ...]:
    """The choices an MVAU implementation exposes."""

    pe = Decision(
        Paths.PE,
        INT,
        _divisor_domain(Paths.PE, Paths.MATRIX_HEIGHT, "matrix_height"),
    )
    simd = Decision(
        Paths.SIMD,
        INT,
        _divisor_domain(Paths.SIMD, Paths.MATRIX_WIDTH, "matrix_width"),
    )
    activation_mode = Decision(
        Paths.ACTIVATION_MODE,
        ACTIVATION_MODE,
        _finite_domain(Paths.ACTIVATION_MODE, frozenset(ActivationMode)),
        proposal=_constant_proposal(ActivationMode.THRESHOLDS),
    )
    binary_mode = Decision(
        Paths.BINARY_MODE,
        BINARY_MODE,
        _finite_domain(Paths.BINARY_MODE, frozenset(BinaryMode)),
        proposal=_constant_proposal(BinaryMode.MULTIPLY),
    )
    backend = Decision(
        Paths.BACKEND,
        BACKEND,
        _finite_domain(Paths.BACKEND, frozenset(Backend)),
        proposal=EvaluatorSpec(_backend_dependencies, _choose_backend),
    )
    resource_style = Decision(
        Paths.RESOURCE_STYLE,
        RESOURCE_STYLE,
        _finite_domain(Paths.RESOURCE_STYLE, frozenset(ResourceStyle)),
        proposal=EvaluatorSpec(
            (_backend_ref,),
            lambda dependencies: Decided(
                ResourceStyle.DSP
                if _expect(dependencies, "backend", Backend) is Backend.RTL
                else ResourceStyle.LUT
            ),
        ),
    )
    weight_delivery = Decision(
        Paths.WEIGHT_DELIVERY,
        WEIGHT_DELIVERY,
        _finite_domain(Paths.WEIGHT_DELIVERY, frozenset(WeightDelivery)),
        proposal=_constant_proposal(WeightDelivery.INTERNAL_DECOUPLED),
    )
    weight_ram_style = Decision(
        Paths.WEIGHT_RAM_STYLE,
        RAM_STYLE,
        _finite_domain(Paths.WEIGHT_RAM_STYLE, frozenset(RamStyle)),
        applies_if=_applicable_when(_delivery_ref, WeightDelivery.INTERNAL_DECOUPLED),
        proposal=_constant_proposal(RamStyle.AUTO, (_delivery_ref,)),
    )
    threshold_ram_style = Decision(
        Paths.THRESHOLD_RAM_STYLE,
        THRESHOLD_RAM_STYLE,
        _finite_domain(Paths.THRESHOLD_RAM_STYLE, frozenset(ThresholdRamStyle)),
        applies_if=_applicable_when(_activation_mode_ref, ActivationMode.THRESHOLDS),
        proposal=_constant_proposal(ThresholdRamStyle.AUTO, (_activation_mode_ref,)),
    )
    pumped_memory = Decision(
        Paths.PUMPED_MEMORY,
        BOOL,
        _finite_domain(Paths.PUMPED_MEMORY, frozenset({False, True})),
        applies_if=_applicable_when(_delivery_ref, WeightDelivery.INTERNAL_DECOUPLED),
        proposal=_constant_proposal(False, (_delivery_ref,)),
    )
    pumped_compute = Decision(
        Paths.PUMPED_COMPUTE,
        BOOL,
        _finite_domain(Paths.PUMPED_COMPUTE, frozenset({False, True})),
        applies_if=_applicable_when(_backend_ref, Backend.RTL),
        proposal=_constant_proposal(False, (_backend_ref,)),
    )
    tile_height = Decision(
        Paths.TILE_HEIGHT,
        INT,
        _positive_integer_domain(Paths.TILE_HEIGHT),
        applies_if=_applicable_when(_backend_ref, Backend.RTL),
        proposal=_constant_proposal(1, (_backend_ref,)),
    )
    activation_bias = Decision(
        Paths.ACTIVATION_BIAS,
        INT,
        DecisionDomain((), lambda _value, _deps: Decided(True)),
        applies_if=_applicable_when(_activation_mode_ref, ActivationMode.THRESHOLDS),
        proposal=_constant_proposal(0, (_activation_mode_ref,)),
    )

    return (
        pe,
        simd,
        activation_mode,
        binary_mode,
        backend,
        resource_style,
        weight_delivery,
        weight_ram_style,
        threshold_ram_style,
        pumped_memory,
        pumped_compute,
        tile_height,
        activation_bias,
    )


def _mvau_properties() -> tuple[DerivedProperty, ...]:
    """Values derived from decisions and the problem."""

    dsp_block = DerivedProperty(
        Paths.DSP_BLOCK,
        DSP_BLOCK,
        EvaluatorSpec((_fpga_part_ref,), _resolve_dsp_block),
    )
    narrow_weights = DerivedProperty(
        Paths.NARROW_WEIGHTS,
        BOOL,
        EvaluatorSpec((_weight_summary_ref, _weight_type_ref), _resolve_narrow_weights),
    )
    dsp_version = DerivedProperty(
        Paths.DSP_VERSION,
        INT,
        EvaluatorSpec(
            (_dsp_block_ref,),
            lambda dependencies: Decided(
                {
                    DspBlock.DSP48E1: 1,
                    DspBlock.DSP48E2: 2,
                    DspBlock.DSP58: 3,
                }[_expect(dependencies, "dsp_block", DspBlock)]
            ),
        ),
        applies_if=_applicable_when(_backend_ref, Backend.RTL),
    )
    weight_memory_depth = DerivedProperty(
        Paths.WEIGHT_MEMORY_DEPTH,
        INT,
        EvaluatorSpec(
            (_matrix_width_ref, _matrix_height_ref, _pe_ref, _simd_ref),
            _resolve_weight_memory_depth,
        ),
    )
    threshold_memory_depth = DerivedProperty(
        Paths.THRESHOLD_MEMORY_DEPTH,
        INT,
        EvaluatorSpec(
            (_activation_mode_ref, _matrix_height_ref, _pe_ref),
            _resolve_threshold_memory_depth,
        ),
    )
    segment_length = DerivedProperty(
        Paths.SEGMENT_LENGTH,
        INT,
        EvaluatorSpec(
            (_target_clock_ref, _simd_ref, _pumped_compute_ref),
            _resolve_segment_length,
        ),
        applies_if=_applicable_when(_backend_ref, Backend.RTL),
    )
    runtime_writable_weights = DerivedProperty(
        Paths.RUNTIME_WRITABLE_WEIGHTS,
        BOOL,
        EvaluatorSpec(
            (_delivery_ref, _weight_ram_ref, _dsp_block_ref, _runtime_updates_ref),
            _resolve_runtime_writable_weights,
        ),
    )

    return (
        dsp_block,
        narrow_weights,
        dsp_version,
        weight_memory_depth,
        threshold_memory_depth,
        segment_length,
        runtime_writable_weights,
    )


def _mvau_constraints() -> tuple[Constraint, ...]:
    """Conditions an MVAU design point must satisfy."""

    backend_constraint = Constraint(
        Paths.BACKEND_SUPPORTED,
        EvaluatorSpec((*_backend_dependencies, _backend_ref), _backend_supported),
    )
    resource_constraint = Constraint(
        Paths.RESOURCE_STYLE_SUPPORTED,
        EvaluatorSpec((_backend_ref, _resource_ref), _resource_style_supported),
    )
    weight_sets_constraint = Constraint(
        Paths.WEIGHT_SETS_SUPPORTED,
        EvaluatorSpec(
            (_weight_set_count_ref, _backend_ref, _delivery_ref),
            _weight_sets_supported,
        ),
    )
    weight_data_constraint = Constraint(
        Paths.WEIGHT_DATA_SUPPORTED,
        EvaluatorSpec(
            (_weight_summary_ref, _weight_set_count_ref, _delivery_ref),
            _weight_data_supported,
        ),
    )
    runtime_updates_constraint = Constraint(
        Paths.RUNTIME_UPDATES_SUPPORTED,
        EvaluatorSpec((_runtime_updates_ref, _delivery_ref), _runtime_updates_supported),
    )
    pumped_memory_constraint = Constraint(
        Paths.PUMPED_MEMORY_SUPPORTED,
        EvaluatorSpec((_pumped_memory_ref, _pe_ref, _simd_ref), _pumped_memory_supported),
        applies_if=_applicable_when(_delivery_ref, WeightDelivery.INTERNAL_DECOUPLED),
    )
    pumped_compute_constraint = Constraint(
        Paths.PUMPED_COMPUTE_SUPPORTED,
        EvaluatorSpec((_pumped_compute_ref, _simd_ref), _pumped_compute_supported),
        applies_if=_applicable_when(_backend_ref, Backend.RTL),
    )
    tile_height_constraint = Constraint(
        Paths.TILE_HEIGHT_SUPPORTED,
        EvaluatorSpec(
            (
                _tile_height_ref,
                _pe_ref,
                _simd_ref,
                _input_type_ref,
                _weight_type_ref,
                _input_vectors_ref,
            ),
            _tile_height_supported,
        ),
        applies_if=_applicable_when(_backend_ref, Backend.RTL),
    )
    binary_constraint = Constraint(
        Paths.BINARY_MODE_SUPPORTED,
        EvaluatorSpec(
            (_binary_mode_ref, _input_type_ref, _weight_type_ref), _binary_mode_supported
        ),
    )
    hls_partition_constraint = Constraint(
        Paths.HLS_PARTITION_SUPPORTED,
        EvaluatorSpec((_simd_ref, _matrix_width_ref), _hls_partition_supported),
        applies_if=_applicable_when(_backend_ref, Backend.HLS),
    )
    clock_constraint = Constraint(
        Paths.CLOCK_SUPPORTED,
        EvaluatorSpec((_target_clock_ref, _pumped_compute_ref), _clock_supported),
        applies_if=_applicable_when(_backend_ref, Backend.RTL),
    )

    return (
        backend_constraint,
        resource_constraint,
        weight_sets_constraint,
        weight_data_constraint,
        runtime_updates_constraint,
        pumped_memory_constraint,
        pumped_compute_constraint,
        tile_height_constraint,
        binary_constraint,
        hls_partition_constraint,
        clock_constraint,
    )


def build_mvau_design_space() -> DesignSpaceSpec:
    """Assemble the MVAU design space.

    Deliberately thin: the declarations live in the three builders above, so this
    reads as a table of contents rather than as the space itself.
    """

    decisions = _mvau_decisions()
    properties = _mvau_properties()
    constraints = _mvau_constraints()
    return DesignSpaceSpec(
        problem_schema=ProblemSchema(
            (
                ProblemField(Paths.MATRIX_WIDTH, INT, constraint=_positive_int),
                ProblemField(Paths.MATRIX_HEIGHT, INT, constraint=_positive_int),
                ProblemField(Paths.INPUT_TYPE, NUMERIC_TYPE),
                ProblemField(Paths.WEIGHT_TYPE, NUMERIC_TYPE),
                ProblemField(Paths.OUTPUT_TYPE, NUMERIC_TYPE),
                ProblemField(Paths.WEIGHT_SUMMARY, WEIGHT_SUMMARY),
                ProblemField(Paths.FPGA_PART, STRING),
                ProblemField(Paths.TARGET_CLOCK_NS, FLOAT, constraint=_positive_float),
                ProblemField(Paths.INPUT_VECTORS, INPUT_VECTORS, constraint=_positive_vector_shape),
                ProblemField(Paths.RUNTIME_WEIGHT_UPDATES, BOOL),
                ProblemField(Paths.WEIGHT_SET_COUNT, INT, constraint=_nonnegative_int),
            )
        ),
        decisions=decisions,
        properties=properties,
        constraints=constraints,
        constraint_sets=(ConstraintSet("all", tuple(item.path for item in constraints)),),
        readiness_profiles=(
            ReadinessProfile(
                "implementation",
                decisions=tuple(item.path for item in decisions),
                properties=tuple(item.path for item in properties),
                constraints=tuple(item.path for item in constraints),
            ),
        ),
    )


def example_problem() -> MVAUProblem:
    return MVAUProblem(
        matrix_width=64,
        matrix_height=32,
        input_type=NumericType("INT4", 4, True),
        weight_type=NumericType("INT4", 4, True),
        output_type=NumericType("INT32", 32, True),
        weight_summary=WeightSummary(minimum=-7),
        fpga_part="xcvu9p-flga2104-2L-e",
        target_clock_ns=5.0,
        input_vectors=(1,),
    )


def main() -> None:
    from finn.dataflow._engine import Engine, ProposalAdoptionMode

    engine = Engine()
    space = engine.validate(build_mvau_design_space())
    started = engine.start(space, example_problem().as_mapping())

    explicit = engine.commit_assignments(
        started,
        {
            Paths.PE: 8,
            Paths.SIMD: 8,
            Paths.ACTIVATION_MODE: ActivationMode.ACCUMULATORS,
        },
    )
    proposed = engine.adopt_profile_proposals(
        explicit.point, "implementation", ProposalAdoptionMode.TO_FIXPOINT
    )

    constraint_evaluation = engine.evaluate_constraint_set(proposed.point, "all")
    readiness = engine.check_readiness(proposed.point, "implementation")
    assert constraint_evaluation.verdict is True
    assert readiness.ready is True

    print("Committed MVAU decisions:")
    for decision_path, value in sorted(proposed.point.assignments.items()):
        rendered = value.value if isinstance(value, Enum) else value
        print(f"  {decision_path}: {rendered}")


if __name__ == "__main__":
    main()
