# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative one-Region model of FinnLib's ``dotp_axi`` Kernel."""

from __future__ import annotations

from enum import Enum
from math import ceil, floor
from typing import cast

from typing_extensions import Self

from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ComponentABI,
    Derived as DerivedClock,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.dataflow.artifacts.contributions import CopiedSource
from finn.dataflow.computation import DOT_PRODUCT_COMPUTATION
from finn.dataflow.design.region import (
    DATAFLOW_REGION_SEMANTICS,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.dataflow.model.declarations import (
    Decision,
    Input,
    constraint,
    derived,
    divisors_of,
    reject,
    unresolved,
)
from finn.dataflow.model.kernel import Kernel, Parameter
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
    element_width,
)

FINNLIB_ROOT = "finnlib"
FINNLIB_SOURCES = (
    "rtl/arith/add_multi_pkg.sv",
    "rtl/arith/add_multi.sv",
    "rtl/linalg/dotp_8sx9_dsp58.sv",
    "rtl/linalg/dotp.sv",
    "rtl/linalg/dotp_axi.sv",
)


class DspBlock(str, Enum):
    """DSP generation selected by the target platform."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


_DSP_VERSION = {
    DspBlock.DSP48E1: 1,
    DspBlock.DSP48E2: 2,
    DspBlock.DSP58: 3,
}
_DSP_WIDTHS = {
    DspBlock.DSP48E1: (25, 18, 48),
    DspBlock.DSP48E2: (27, 18, 48),
    DspBlock.DSP58: (27, 24, 58),
}
_MULTIPLIABLE_FAMILIES = ("INT", "UINT")
_SIGNED_ROLES = frozenset({"weight", "accumulator", "output"})
_SEGMENT_BASE_DELAY_NS = 0.741
_SEGMENT_STAGE_DELAY_NS = 0.605


def _expanded_activation_beats(
    repetitions: int, neuron_folds: int, synapse_folds: int, simd: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for _neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )


def _weight_beats(
    repetitions: int,
    neuron_folds: int,
    synapse_folds: int,
    pe: int,
    simd: int,
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple(
            (neuron_fold * pe + pe_index, synapse_fold * simd + lane)
            for pe_index in range(pe)
            for lane in range(simd)
        )
        for _repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )


def _output_beats(
    repetitions: int, neuron_folds: int, pe: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, neuron_fold * pe + pe_index) for pe_index in range(pe))
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
    )


def construct_dot_product_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_type: NumericElementType,
    weight_type: NumericElementType,
    output_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """The folded stream contract directly implemented by ``dotp_axi``."""

    dimensions = (repetitions, matrix_width, matrix_height, pe, simd)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("dot-product dimensions and folding must be positive integers")
    if matrix_width % simd:
        raise ValueError("SIMD must divide matrix_width exactly")
    if matrix_height % pe:
        raise ValueError("PE must divide matrix_height exactly")

    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    schedule = LogicalSchedule(
        (
            ScheduleLevel("rep", repetitions),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
        )
    )
    activation = Operand("X", activation_type, (repetitions, matrix_width))
    weight = Operand("W", weight_type, (matrix_height, matrix_width))
    output = Operand("Y", output_type, (repetitions, matrix_height))
    activation_requirements: dict[RequirementKey, int] = {
        (
            (repetition, neuron_fold, synapse_fold),
            (repetition, synapse_fold * simd + lane),
        ): 1
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
        for lane in range(simd)
    }
    weight_requirements: dict[RequirementKey, int] = {
        (
            (repetition, neuron_fold, synapse_fold),
            (neuron_fold * pe + pe_index, synapse_fold * simd + lane),
        ): 1
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
        for pe_index in range(pe)
        for lane in range(simd)
    }
    availability_entries: dict[Coordinate, Coordinate] = {
        (repetition, neuron_fold * pe + pe_index): (
            repetition,
            neuron_fold,
            synapse_folds - 1,
        )
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for pe_index in range(pe)
    }
    availability = ScheduledOutputAvailability(availability_entries)
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port(
                    "activation",
                    activation,
                    BeatSequence(
                        simd,
                        _expanded_activation_beats(repetitions, neuron_folds, synapse_folds, simd),
                    ),
                ),
                ScheduledInputRequirements(activation_requirements),
            ),
            InputInterface(
                Port(
                    "weight",
                    weight,
                    BeatSequence(
                        pe * simd,
                        _weight_beats(repetitions, neuron_folds, synapse_folds, pe, simd),
                    ),
                ),
                ScheduledInputRequirements(weight_requirements),
            ),
        ),
        (
            OutputInterface(
                Port(
                    "output",
                    output,
                    BeatSequence(pe, _output_beats(repetitions, neuron_folds, pe)),
                ),
                availability,
            ),
        ),
    )


def _is_twos_complement_integer(datatype: NumericElementType) -> bool:
    name = datatype.name
    return any(
        name.startswith(prefix) and name[len(prefix) :].isdigit()
        for prefix in _MULTIPLIABLE_FAMILIES
    )


def _operand_types_supported(
    activation: NumericElementType,
    weight: NumericElementType,
    accumulator: NumericElementType,
    output: NumericElementType,
) -> object:
    rejected: dict[str, str] = {}
    for role, datatype in (
        ("activation", activation),
        ("weight", weight),
        ("accumulator", accumulator),
        ("output", output),
    ):
        if not _is_twos_complement_integer(datatype):
            rejected[role] = f"{datatype.name} is not a two's-complement integer"
        elif role in _SIGNED_ROLES and not datatype.signed():
            rejected[role] = f"{datatype.name} is unsigned; the core declares this role signed"
    if "accumulator" not in rejected and "output" not in rejected and output != accumulator:
        rejected["output"] = (
            f"{output.name} is not the accumulator {accumulator.name}; "
            "the core drives the accumulator straight out"
        )
    if rejected:
        return reject(
            "dotp-axi-numeric-types-unsupported",
            "this dot-product core multiplies two's-complement integers",
            values=rejected,
        )
    return True


def _operand_widths_supported(activation: NumericElementType, weight: NumericElementType) -> object:
    if element_width(activation) < 2 or element_width(weight) < 2:
        return reject(
            "dotp-axi-operands-too-narrow",
            "the dot-product core needs at least two bits of each operand",
            values={"activation": element_width(activation), "weight": element_width(weight)},
        )
    return True


def _width_supported(
    target: DspBlock,
    activation: NumericElementType,
    weight: NumericElementType,
    accumulator: NumericElementType,
    output: NumericElementType,
) -> bool:
    a_width, b_width, p_width = _DSP_WIDTHS[target]
    return (
        element_width(weight) <= a_width
        and element_width(activation) <= b_width
        and element_width(accumulator) <= p_width
        and element_width(output) <= p_width
    )


def _packing_fits(
    *, a_width: int, weight_width: int, activation_width: int, narrow_weights: bool
) -> tuple[bool, int]:
    sign_bit = 0 if narrow_weights else 1
    minimum_lane_width = weight_width + activation_width - 1
    lanes = (
        1
        if a_width == weight_width
        else 1 + (a_width - sign_bit - weight_width) // minimum_lane_width
    )
    slack = a_width - sign_bit - weight_width - (lanes - 1) * minimum_lane_width
    return slack >= 0, slack


def _narrow_weights_supported(
    target: DspBlock,
    activation: NumericElementType,
    weight: NumericElementType,
    narrow: bool,
) -> object:
    a_width = _DSP_WIDTHS[target][0]
    weight_bits = element_width(weight)
    if weight_bits > a_width:
        return True
    fits, slack = _packing_fits(
        a_width=a_width,
        weight_width=weight_bits,
        activation_width=element_width(activation),
        narrow_weights=narrow,
    )
    if not fits:
        return reject(
            "dotp-axi-weights-do-not-pack",
            "these weights do not fit the DSP A datapath without the narrow-weight promise",
            values={
                "weight_width": weight_bits,
                "a_datapath_width": a_width,
                "narrow_weights": narrow,
                "bit_slack": slack,
            },
        )
    return True


def _segment_length(clock_period_ns: float, pumping: bool, simd: int) -> object:
    reference_clock = clock_period_ns / 2 if pumping else clock_period_ns
    if reference_clock <= _SEGMENT_BASE_DELAY_NS:
        return unresolved(
            "mvau-segment-length-clock-infeasible",
            "the target clock period is below the covered RTL segment-delay bound",
            values={"reference_clock_ns": reference_clock},
        )
    covered = floor((reference_clock - _SEGMENT_BASE_DELAY_NS) / _SEGMENT_STAGE_DELAY_NS + 1)
    longest = ceil(simd / (6 if pumping else 3))
    return min(covered, longest)


def _byte_aligned(width: int) -> int:
    return (width + 7) // 8 * 8


def _rtl_scalar(value: bool | int | float | str) -> str:
    return str(int(value)) if isinstance(value, bool) else str(value)


def _axis(
    name: str,
    *,
    width: int,
    endpoint: Endpoint,
    last: bool = False,
) -> Bus:
    members = [
        Member("tdata", f"{name}_tdata", width),
        Member("tvalid", f"{name}_tvalid"),
        Member("tready", f"{name}_tready"),
    ]
    if last:
        members.append(Member("tlast", f"{name}_tlast"))
    return Bus(
        name,
        StandardProtocol.AXIS,
        members,
        endpoint=endpoint,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


class DotpAxiKernel(Kernel):
    """FinnLib folded dot product as one Region and one physical module."""

    id = "dotp_axi"
    version = "1"
    computation = DOT_PRODUCT_COMPUTATION

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    narrow_weights = Input(bool)
    target_dsp = Input(DspBlock)
    clock_period_ns = Input(float)

    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))
    compute_pumping = Decision(bool, values=(False, True))

    @derived(
        DATAFLOW_REGION_SEMANTICS,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation_type,
        weight_type=weight_type,
        output_type=output_type,
        pe=pe,
        simd=simd,
    )
    def region(
        *,
        repetitions: int,
        matrix_width: int,
        matrix_height: int,
        activation_type: NumericElementType,
        weight_type: NumericElementType,
        output_type: NumericElementType,
        pe: int,
        simd: int,
    ) -> DataflowRegion:
        return construct_dot_product_region(
            repetitions,
            matrix_width,
            matrix_height,
            activation_type,
            weight_type,
            output_type,
            pe,
            simd,
        )

    exports = (pe, simd)

    @derived(int, target=target_dsp)
    def dsp_version(*, target: DspBlock) -> int:
        return _DSP_VERSION[target]

    @derived(bool, activation=activation_type)
    def signed_activations(*, activation: NumericElementType) -> bool:
        return bool(activation.signed())

    @derived(int, clock=clock_period_ns, pumping=compute_pumping, simd=simd)
    def segment_length(*, clock: float, pumping: bool, simd: int) -> object:
        return _segment_length(clock, pumping, simd)

    @derived(int, datatype=activation_type)
    def activation_width(*, datatype: NumericElementType) -> int:
        return element_width(datatype)

    @derived(int, datatype=weight_type)
    def weight_width(*, datatype: NumericElementType) -> int:
        return element_width(datatype)

    @derived(int, datatype=accumulator_type)
    def accumulator_width(*, datatype: NumericElementType) -> int:
        return element_width(datatype)

    @constraint(
        activation=activation_type,
        weight=weight_type,
        accumulator=accumulator_type,
        output=output_type,
    )
    def operand_types_supported(
        *,
        activation: NumericElementType,
        weight: NumericElementType,
        accumulator: NumericElementType,
        output: NumericElementType,
    ) -> object:
        return _operand_types_supported(activation, weight, accumulator, output)

    @constraint(activation=activation_type, weight=weight_type)
    def operand_widths_supported(
        *, activation: NumericElementType, weight: NumericElementType
    ) -> object:
        return _operand_widths_supported(activation, weight)

    @constraint(target=target_dsp)
    def target_supported(*, target: DspBlock) -> bool:
        return target in _DSP_VERSION

    @constraint(
        target=target_dsp,
        activation=activation_type,
        weight=weight_type,
        accumulator=accumulator_type,
        output=output_type,
    )
    def width_supported(
        *,
        target: DspBlock,
        activation: NumericElementType,
        weight: NumericElementType,
        accumulator: NumericElementType,
        output: NumericElementType,
    ) -> bool:
        return _width_supported(target, activation, weight, accumulator, output)

    @constraint(
        target=target_dsp,
        activation=activation_type,
        weight=weight_type,
        narrow=narrow_weights,
    )
    def narrow_weights_supported(
        *,
        target: DspBlock,
        activation: NumericElementType,
        weight: NumericElementType,
        narrow: bool,
    ) -> object:
        return _narrow_weights_supported(target, activation, weight, narrow)

    @constraint(pumping=compute_pumping, simd=simd)
    def pumping_supported(*, pumping: bool, simd: int) -> bool:
        return simd >= 2 if pumping else True

    PE = Parameter(pe)
    SIMD = Parameter(simd)
    PUMPED_COMPUTE = Parameter(compute_pumping)
    ACTIVATION_WIDTH = Parameter(activation_width)
    WEIGHT_WIDTH = Parameter(weight_width)
    ACCU_WIDTH = Parameter(accumulator_width)
    VERSION = Parameter(dsp_version)
    SIGNED_ACTIVATIONS = Parameter(signed_activations)
    SEGMENTLEN = Parameter(segment_length)
    NARROW_WEIGHTS = Parameter(narrow_weights)
    ACTIVATION_BROADCASTING = Parameter.constant(
        1,
        why="this implementation broadcasts one activation vector across its PE lanes",
    )
    FORCE_BEHAVIORAL = Parameter.constant(
        0,
        why="the production implementation uses inferred DSP logic",
    )

    sources = (
        CopiedSource(
            FINNLIB_ROOT,
            FINNLIB_SOURCES[0],
            provides=("package:add_multi_pkg",),
        ),
        CopiedSource(
            FINNLIB_ROOT,
            FINNLIB_SOURCES[1],
            provides=("module:add_multi",),
            requires=("package:add_multi_pkg",),
        ),
        CopiedSource(
            FINNLIB_ROOT,
            FINNLIB_SOURCES[2],
            provides=("module:dotp_8sx9_dsp58",),
        ),
        CopiedSource(
            FINNLIB_ROOT,
            FINNLIB_SOURCES[3],
            provides=("module:dotp",),
            requires=("package:add_multi_pkg", "module:add_multi"),
        ),
        CopiedSource(
            FINNLIB_ROOT,
            FINNLIB_SOURCES[4],
            provides=("module:dotp_axi",),
            requires=("module:dotp", "module:dotp_8sx9_dsp58"),
        ),
    )

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        parameters = configured.parameters
        pe = cast(int, parameters["PE"])
        simd = cast(int, parameters["SIMD"])
        activation_width = cast(int, parameters["ACTIVATION_WIDTH"])
        weight_width = cast(int, parameters["WEIGHT_WIDTH"])
        accumulator_width = cast(int, parameters["ACCU_WIDTH"])
        return ComponentABI(
            "dotp_axi",
            (
                Signal("ap_clk", Direction.IN, 1, Clock(Free())),
                Signal("ap_clk2x", Direction.IN, 1, Clock(DerivedClock("ap_clk", 2))),
                Signal("ap_rst_n", Direction.IN, 1, Reset(active_low=True)),
                _axis(
                    "s_axis_weights",
                    width=_byte_aligned(pe * simd * weight_width),
                    endpoint=Endpoint.TARGET,
                ),
                _axis(
                    "s_axis_input",
                    width=_byte_aligned(simd * activation_width),
                    endpoint=Endpoint.TARGET,
                    last=True,
                ),
                _axis(
                    "m_axis_output",
                    width=_byte_aligned(pe * accumulator_width),
                    endpoint=Endpoint.INITIATOR,
                ),
            ),
            tuple((name, _rtl_scalar(value)) for name, value in parameters.items()),
        )


__all__ = [
    "DOT_PRODUCT_COMPUTATION",
    "DspBlock",
    "DotpAxiKernel",
    "FINNLIB_ROOT",
    "FINNLIB_SOURCES",
    "construct_dot_product_region",
]
