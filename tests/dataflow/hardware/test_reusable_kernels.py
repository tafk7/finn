# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contract-neutral promotion gates for reusable physical Kernels."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.authoring import OpDesign, Ref, finite
from finn.dataflow.authoring.design import (
    DataflowDesign,
    DataflowDesignScope,
)
from finn.dataflow.authoring.inventory import (
    DataflowDesignEntry,
    declare_dataflow_design_inventory,
)
from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.authoring.scope import Unresolvable
from finn.dataflow.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.design import Answer, Decided, DependencyKind, Engine, QualifiedPath, Unresolved
from finn.dataflow.design.region import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.kernels.dotp_axi import (
    DotProductKernelInputs,
    DotpAxiKernel,
    covers_numeric_types,
    covers_operand_types,
)
from finn.dataflow.kernels.dsp import DspBlock, a_datapath_width, pack_lanes
from finn.dataflow.kernels.finn_rtl_memstream import (
    FinnRtlMemstreamInputs,
    FinnRtlMemstreamKernel,
)
from finn.dataflow.kernels.replay_buffer import ReplayBufferInputs, ReplayBufferKernel
from finn.dataflow.kernels import Kernel
from finn.dataflow.kernels.numeric import DotProductNumericTypes
from finn.dataflow.kernels.rtl_parameters import dsp_version, segment_length, signed_activations
from finn.dataflow.parameters.cyclic.computation import CYCLIC_PARAMETER_DELIVERY
from finn.dataflow.parameters.cyclic.definition import (
    CyclicRamStyle,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    element_width,
)

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]
INT20 = DataType["INT20"]
INT25 = DataType["INT25"]
INT27 = DataType["INT27"]
INT32 = DataType["INT32"]
INT64 = DataType["INT64"]
UINT8 = DataType["UINT8"]

REPETITIONS = 1
MATRIX_WIDTH = 4
MATRIX_HEIGHT = 4
PE = 2
SIMD = 2


def _stream_region(extent: int) -> DataflowRegion:
    positions = tuple((index,) for index in range(extent))
    beats = BeatSequence(1, tuple((position,) for position in positions))
    operand = Operand("value", INT8, (extent,))
    requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        ((index,), (index,)): 1 for index in range(extent)
    }
    availability: dict[tuple[int, ...], tuple[int, ...]] = {
        (index,): (index,) for index in range(extent)
    }
    return DataflowRegion(
        LogicalSchedule((ScheduleLevel("element", extent),)),
        (
            InputInterface(
                Port("input", operand, beats),
                ScheduledInputRequirements(requirements),
            ),
        ),
        (
            OutputInterface(
                Port("output", operand, beats),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


def _dot_product_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_type: NumericElementType,
    weight_type: NumericElementType,
    output_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """A truthful folded dot product declared without importing MVAU."""

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
    activation_beats = tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for _neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )
    weight_beats = tuple(
        tuple(
            (neuron_fold * pe + pe_index, synapse_fold * simd + lane)
            for pe_index in range(pe)
            for lane in range(simd)
        )
        for _repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )
    output_beats = tuple(
        tuple((repetition, neuron_fold * pe + pe_index) for pe_index in range(pe))
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
    )
    activation_requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        (
            (repetition, neuron_fold, synapse_fold),
            (repetition, synapse_fold * simd + lane),
        ): 1
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
        for lane in range(simd)
    }
    weight_requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
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
    availability: dict[tuple[int, ...], tuple[int, ...]] = {
        (repetition, neuron_fold * pe + pe_index): (
            repetition,
            neuron_fold,
            synapse_folds - 1,
        )
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for pe_index in range(pe)
    }
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("activation", activation, BeatSequence(simd, activation_beats)),
                ScheduledInputRequirements(activation_requirements),
            ),
            InputInterface(
                Port("weight", weight, BeatSequence(pe * simd, weight_beats)),
                ScheduledInputRequirements(weight_requirements),
            ),
        ),
        (
            OutputInterface(
                Port("output", output, BeatSequence(pe, output_beats)),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


def _activation_replay_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """A truthful compact-to-expanded activation replay contract."""

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
    compact = tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for synapse_fold in range(synapse_folds)
    )
    expanded = tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for _neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )
    requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        (
            (repetition, neuron_fold, synapse_fold),
            (repetition, synapse_fold * simd + lane),
        ): 1
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
        for lane in range(simd)
    }
    availability: dict[tuple[int, ...], tuple[int, ...]] = {
        (repetition, synapse_fold * simd + lane): (repetition, 0, synapse_fold)
        for repetition in range(repetitions)
        for synapse_fold in range(synapse_folds)
        for lane in range(simd)
    }
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("activation_in", activation, BeatSequence(simd, compact)),
                ScheduledInputRequirements(requirements),
            ),
        ),
        (
            OutputInterface(
                Port("activation_out", activation, BeatSequence(simd, expanded)),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


@dataclass(frozen=True)
class NeutralInputs:
    extent: Ref[int]
    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_type: Ref[NumericElementType]
    weight_type: Ref[NumericElementType]
    accumulator_type: Ref[NumericElementType]
    output_type: Ref[NumericElementType]
    narrow: Ref[bool]
    target: Ref[DspBlock]
    clock: Ref[float]
    initializer_available: Ref[bool]
    runtime_writable: Ref[bool]
    memory_capabilities: Ref[CyclicTargetMemoryCapabilities]
    sets: Ref[int]


class NeutralDotProductDesign(DataflowDesign):
    id = "neutral_dot_product"

    @classmethod
    def define(cls, design: DataflowDesignScope[NeutralInputs]) -> None:
        pe = design.choice("pe", int, domain=finite((PE,)))
        simd = design.choice("simd", int, domain=finite((1, SIMD)))
        node = design.region(
            "mac",
            node_id="mac_unit",
            dependencies={
                "repetitions": design.inputs.repetitions,
                "matrix_width": design.inputs.matrix_width,
                "matrix_height": design.inputs.matrix_height,
                "activation_type": design.inputs.activation_type,
                "weight_type": design.inputs.weight_type,
                "output_type": design.inputs.output_type,
                "pe": pe,
                "simd": simd,
            },
            evaluate=_dot_product_region,
            computation=DOT_PRODUCT_COMPUTATION,
        )
        design.singleton_network(node)
        design.kernels(
            "mac_placement",
            covers=(node,),
            candidates=(DotpAxiKernel,),
            inputs=DotProductKernelInputs(
                "mac",
                node.region,
                node.computation,
                pe,
                simd,
                design.inputs.activation_type,
                design.inputs.weight_type,
                design.inputs.output_type,
                design.inputs.accumulator_type,
                design.inputs.narrow,
                design.inputs.target,
                design.inputs.clock,
            ),
        )


class NeutralReplayDesign(DataflowDesign):
    id = "neutral_replay"

    @classmethod
    def define(cls, design: DataflowDesignScope[NeutralInputs]) -> None:
        pe = design.choice("pe", int, domain=finite((PE,)))
        simd = design.choice("simd", int, domain=finite((SIMD,)))
        node = design.region(
            "fanout",
            node_id="activation_fanout",
            dependencies={
                "repetitions": design.inputs.repetitions,
                "matrix_width": design.inputs.matrix_width,
                "matrix_height": design.inputs.matrix_height,
                "activation_type": design.inputs.activation_type,
                "pe": pe,
                "simd": simd,
            },
            evaluate=_activation_replay_region,
            computation=ACTIVATION_REPLAY_COMPUTATION,
        )
        design.singleton_network(node)
        length = design.derived(
            "fanout.length",
            int,
            dependencies={"matrix_width": design.inputs.matrix_width, "simd": simd},
            evaluate=lambda matrix_width, simd: matrix_width // simd,
        )
        repetitions = design.derived(
            "fanout.repetitions",
            int,
            dependencies={"matrix_height": design.inputs.matrix_height, "pe": pe},
            evaluate=lambda matrix_height, pe: matrix_height // pe,
        )
        width = design.derived(
            "fanout.width",
            int,
            dependencies={"activation_type": design.inputs.activation_type, "simd": simd},
            evaluate=lambda activation_type, simd: simd * element_width(activation_type),
        )
        design.kernels(
            "fanout_placement",
            covers=(node,),
            candidates=(ReplayBufferKernel,),
            inputs=ReplayBufferInputs(
                "fanout",
                node.region,
                node.computation,
                length,
                repetitions,
                width,
            ),
        )


class NeutralMemstreamDesign(DataflowDesign):
    id = "neutral_memstream"

    @classmethod
    def define(cls, design: DataflowDesignScope[NeutralInputs]) -> None:
        output_port = design.derived(
            "output_port",
            Port,
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: _stream_region(extent).output_interface("output").port,
        )
        ram_style = design.choice(
            "ram_style", CyclicRamStyle, domain=finite((CyclicRamStyle.BRAM,))
        )
        pumped = design.choice("pumped", bool, domain=finite((False,)))
        node = design.region(
            "parameter_source",
            node_id="parameter_source",
            dependencies={"output_port": output_port},
            evaluate=lambda output_port: construct_cyclic_parameter_region(output_port),
            computation=CYCLIC_PARAMETER_DELIVERY,
        )
        design.singleton_network(node)
        design.kernels(
            "parameter_placement",
            covers=(node,),
            candidates=(FinnRtlMemstreamKernel,),
            inputs=FinnRtlMemstreamInputs(
                "parameter_source",
                node.region,
                node.computation,
                output_port,
                design.inputs.initializer_available,
                design.inputs.runtime_writable,
                design.inputs.memory_capabilities,
                ram_style,
                pumped,
                design.inputs.sets,
            ),
        )


def problem_path(name: str) -> QualifiedPath:
    return QualifiedPath(f"problem.neutral.{name}")


def _neutral_inputs(operation: OpDesign) -> NeutralInputs:
    return NeutralInputs(
        operation.graph_fact("extent", int),
        operation.graph_fact("repetitions", int),
        operation.graph_fact("matrix_width", int),
        operation.graph_fact("matrix_height", int),
        operation.graph_fact("activation_type", QONNX_DATATYPE_VALUE_SEMANTICS),
        operation.graph_fact("weight_type", QONNX_DATATYPE_VALUE_SEMANTICS),
        operation.graph_fact("accumulator_type", QONNX_DATATYPE_VALUE_SEMANTICS),
        operation.graph_fact("output_type", QONNX_DATATYPE_VALUE_SEMANTICS),
        operation.graph_fact("narrow", bool),
        operation.target_fact("dsp_block", DspBlock),
        operation.target_fact("clock", float),
        operation.graph_fact("initializer_available", bool),
        operation.build_fact("runtime_writable", bool),
        operation.target_fact("memory_capabilities", CyclicTargetMemoryCapabilities),
        operation.graph_fact("sets", int),
    )


def _problem_values(
    inputs: NeutralInputs,
    *,
    activation_type: NumericElementType = INT8,
    weight_type: NumericElementType = INT8,
    accumulator_type: NumericElementType = INT16,
    output_type: NumericElementType = INT16,
    narrow: bool = False,
    target: DspBlock = DspBlock.DSP58,
    clock: float = 5.0,
) -> dict[QualifiedPath, object]:
    return {
        inputs.extent.path: 4,
        inputs.repetitions.path: REPETITIONS,
        inputs.matrix_width.path: MATRIX_WIDTH,
        inputs.matrix_height.path: MATRIX_HEIGHT,
        inputs.activation_type.path: activation_type,
        inputs.weight_type.path: weight_type,
        inputs.accumulator_type.path: accumulator_type,
        inputs.output_type.path: output_type,
        inputs.narrow.path: narrow,
        inputs.target.path: target,
        inputs.clock.path: clock,
        inputs.initializer_available.path: True,
        inputs.runtime_writable.path: False,
        inputs.memory_capabilities.path: CyclicTargetMemoryCapabilities(True),
        inputs.sets.path: 3,
    }


def _resolve(
    design_type: type[DataflowDesign],
    assignments: Mapping[str, object],
    **problem: object,
) -> Answer[DesignRealization]:
    operation = OpDesign("neutral", problem_namespace="neutral")
    inputs = _neutral_inputs(operation)
    inventory = declare_dataflow_design_inventory(
        "neutral",
        (DataflowDesignEntry(design_type, inputs),),
        shared_specs=(operation.spec(),),
    )
    engine = Engine()
    point = engine.start(
        engine.validate(inventory.specification),
        _problem_values(inputs, **problem),  # type: ignore[arg-type]
    )
    by_suffix = {
        suffix: next(
            item.path
            for item in inventory.specification.decisions
            if item.path.value.endswith(suffix)
        )
        for suffix in assignments
    }
    point = engine.commit_assignments(
        point,
        {by_suffix[suffix]: value for suffix, value in assignments.items()},
    ).point
    realized = inventory.realize(engine, point)
    return realized


def _realize(
    design_type: type[DataflowDesign],
    assignments: Mapping[str, object],
    **problem: object,
) -> Kernel:
    realized = _resolve(design_type, assignments, **problem)
    assert isinstance(realized, Decided)
    return next(iter(realized.value.kernels.values()))


def test_promoted_kernels_have_no_reverse_mvau_imports() -> None:
    root = Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "kernels"
    for path in root.glob("*.py"):
        imports = {
            node.module
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(module.startswith("finn.dataflow.ops.mvau") for module in imports), path


def test_dot_product_kernel_binds_in_a_non_mvau_design() -> None:
    kernel = _realize(
        NeutralDotProductDesign,
        {".pe": PE, ".simd": SIMD, ".compute_pumping": False},
    )
    assert kernel.id == DotpAxiKernel.id
    assert tuple(kernel.regions) == ("mac",)
    assert {item.port.id for item in kernel.regions["mac"].region.inputs} == {
        "activation",
        "weight",
    }
    assert kernel.regions["mac"].region.outputs[0].port.id == "output"
    assert dict(kernel.parameters)["PE"] == PE
    assert dict(kernel.parameters)["SIMD"] == SIMD
    assert kernel.components()[0].id == "dotp_axi"


def test_replay_buffer_kernel_binds_in_a_non_mvau_design() -> None:
    kernel = _realize(
        NeutralReplayDesign,
        {".pe": PE, ".simd": SIMD},
    )
    assert kernel.id == ReplayBufferKernel.id
    assert tuple(kernel.regions) == ("fanout",)
    region = kernel.regions["fanout"].region
    assert region.inputs[0].port.id == "activation_in"
    assert region.outputs[0].port.id == "activation_out"
    assert len(region.outputs[0].port.beat_sequence.beats) > len(
        region.inputs[0].port.beat_sequence.beats
    )
    assert dict(kernel.parameters) == {"LEN": 2, "REP": 2, "W": 16}
    assert kernel.components()[0].id == "replay_buffer"


def test_memstream_kernel_binds_in_a_non_mvau_design() -> None:
    kernel = _realize(
        NeutralMemstreamDesign,
        {".ram_style": CyclicRamStyle.BRAM, ".pumped": False},
    )
    assert kernel.id == FinnRtlMemstreamKernel.id
    assert tuple(kernel.regions) == ("parameter_source",)
    assert dict(kernel.parameters)["SETS"] == 3
    assert dict(kernel.parameters)["DEPTH"] == 4
    assert kernel.components()[0].id == "memstream"


def test_promoted_kernel_parameter_and_source_ownership_is_complete() -> None:
    dot = _realize(
        NeutralDotProductDesign,
        {".pe": PE, ".simd": SIMD, ".compute_pumping": False},
    )
    replay = _realize(NeutralReplayDesign, {".pe": PE, ".simd": SIMD})
    memstream = _realize(
        NeutralMemstreamDesign,
        {".ram_style": CyclicRamStyle.BRAM, ".pumped": False},
    )
    assert set(dot.declaration.parameter_names) == {
        "VERSION",
        "ACTIVATION_BROADCASTING",
        "PE",
        "SIMD",
        "SEGMENTLEN",
        "ACTIVATION_WIDTH",
        "WEIGHT_WIDTH",
        "ACCU_WIDTH",
        "NARROW_WEIGHTS",
        "SIGNED_ACTIVATIONS",
        "PUMPED_COMPUTE",
        "FORCE_BEHAVIORAL",
    }
    assert set(replay.declaration.parameter_names) == {"LEN", "REP", "W"}
    assert set(memstream.declaration.parameter_names) == {
        "DEPTH",
        "SETS",
        "WIDTH",
        "INIT_FILE",
        "RAM_STYLE",
        "PUMPED_MEMORY",
        "INITIALIZER_AVAILABLE",
        "RUNTIME_WRITABLE",
    }
    for kernel in (dot, replay, memstream):
        for parameter in kernel.declaration.parameters:
            if parameter.is_constant:
                assert parameter.why
            else:
                assert parameter.source is not None
                assert parameter.source.kind in {
                    DependencyKind.PROBLEM,
                    DependencyKind.DECISION,
                    DependencyKind.PROPERTY,
                }
    assert [item.path for item in dot.sources][-1] == "rtl/linalg/dotp_axi.sv"
    assert [item.path for item in replay.sources] == [
        "finn-rtllib/mvu/mvu_pkg.sv",
        "finn-rtllib/mvu/replay_buffer.sv",
    ]
    assert [item.path for item in memstream.sources] == [
        "finn-rtllib/memstream/hdl/memstream_wrapper_template.v",
        "finn-rtllib/memstream/hdl/memstream.sv",
        "finn-rtllib/memstream/hdl/memstream_axi.sv",
        "finn-rtllib/axi/hdl/axilite.sv",
    ]


@pytest.mark.parametrize(
    ("target", "expected"),
    (
        (DspBlock.DSP48E1, 1),
        (DspBlock.DSP48E2, 2),
        (DspBlock.DSP58, 3),
    ),
)
def test_dsp_generation_maps_to_exact_rtl_version(target: DspBlock, expected: int) -> None:
    assert dsp_version(target) == expected


@pytest.mark.parametrize(
    ("clock", "simd", "pumping", "expected"),
    (
        (4.0, 2, False, 1),
        (4.0, 12, False, 4),
        (10.0, 12, False, 4),
        (2.0, 12, False, 3),
        (4.0, 12, True, 2),
    ),
)
def test_segment_length_values_remain_exact(
    clock: float, simd: int, pumping: bool, expected: int
) -> None:
    assert segment_length(clock, pumping, simd) == expected


def test_segment_length_reports_an_infeasible_clock_directly() -> None:
    answer = segment_length(0.5, False, 2)
    assert isinstance(answer, Unresolvable)
    assert answer.finding.code == "mvau-segment-length-clock-infeasible"


@pytest.mark.parametrize(("datatype", "expected"), ((INT8, True), (UINT8, False)))
def test_signed_activation_parameter_uses_the_activation_encoding(
    datatype: NumericElementType, expected: bool
) -> None:
    assert signed_activations(datatype) is expected


def test_role_specific_numeric_coverage_remains_complete() -> None:
    assert covers_operand_types(DotProductNumericTypes(INT8, INT8, INT16, INT16))
    assert not covers_operand_types(
        DotProductNumericTypes(INT8, INT8, DataType["FLOAT16"], DataType["FLOAT16"])
    )
    refused = {
        verdict.role
        for verdict in covers_numeric_types(
            DotProductNumericTypes(INT8, UINT8, DataType["UINT16"], DataType["UINT16"])
        )
        if not verdict.supported
    }
    assert refused == {"weight", "accumulator", "output"}


@pytest.mark.parametrize(
    ("target", "weight"),
    (
        (DspBlock.DSP48E1, INT25),
        (DspBlock.DSP48E2, INT27),
        (DspBlock.DSP58, INT27),
    ),
)
def test_narrow_weight_packing_matches_each_dsp_a_port(
    target: DspBlock, weight: NumericElementType
) -> None:
    width = a_datapath_width(cast(int, dsp_version(target)))
    assert not pack_lanes(
        a_width=width,
        weight_width=weight.bitwidth(),
        activation_width=INT8.bitwidth(),
        narrow_weights=False,
    ).fits
    assert pack_lanes(
        a_width=width,
        weight_width=weight.bitwidth(),
        activation_width=INT8.bitwidth(),
        narrow_weights=True,
    ).fits


def test_target_width_pumping_and_narrow_limits_refuse_the_bound_kernel() -> None:
    common = {".pe": PE, ".simd": SIMD, ".compute_pumping": False}
    too_wide = _resolve(
        NeutralDotProductDesign,
        common,
        activation_type=INT20,
        target=DspBlock.DSP48E2,
    )
    assert isinstance(too_wide, Unresolved)
    assert "hardware-coverage-refused" in {item.code for item in too_wide.findings}
    assert isinstance(
        _resolve(
            NeutralDotProductDesign,
            common,
            activation_type=INT20,
            target=DspBlock.DSP58,
        ),
        Decided,
    )

    accumulator = _resolve(
        NeutralDotProductDesign,
        common,
        accumulator_type=INT64,
        output_type=INT64,
        target=DspBlock.DSP48E2,
    )
    assert isinstance(accumulator, Unresolved)

    pumping = _resolve(
        NeutralDotProductDesign,
        {".pe": PE, ".simd": 1, ".compute_pumping": True},
    )
    assert isinstance(pumping, Unresolved)

    wide_weight = _resolve(
        NeutralDotProductDesign,
        common,
        weight_type=INT27,
        accumulator_type=INT32,
        output_type=INT32,
        target=DspBlock.DSP58,
        narrow=False,
    )
    assert isinstance(wide_weight, Unresolved)
    assert isinstance(
        _resolve(
            NeutralDotProductDesign,
            common,
            weight_type=INT27,
            accumulator_type=INT32,
            output_type=INT32,
            target=DspBlock.DSP58,
            narrow=True,
        ),
        Decided,
    )
