# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contract-neutral promotion gates for reusable physical Kernels."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.authoring import OpDesign, Ref, finite
from finn.dataflow.authoring.design import (
    DataflowDesign,
    DataflowDesignEntry,
    DataflowDesignScope,
    declare_dataflow_design_inventory,
)
from finn.dataflow.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.design import Decided, Engine, QualifiedPath
from finn.dataflow.design.region import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.kernels.dotp_axi import DotProductKernelInputs, DotpAxiKernel
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.kernels.finn_rtl_memstream import (
    FinnRtlMemstreamInputs,
    FinnRtlMemstreamKernel,
)
from finn.dataflow.kernels.replay_buffer import ReplayBufferInputs, ReplayBufferKernel
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
)

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]


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


@dataclass(frozen=True)
class NeutralInputs:
    extent: Ref[int]
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


class NeutralDotProductDesign(DataflowDesign):
    id = "neutral_dot_product"

    @classmethod
    def define(cls, design: DataflowDesignScope[NeutralInputs]) -> None:
        pe = design.choice("pe", int, domain=finite((2,)))
        simd = design.choice("simd", int, domain=finite((2,)))
        node = design.region(
            "compute",
            node_id="compute",
            dependencies={"extent": design.inputs.extent},
            evaluate=_stream_region,
            computation=DOT_PRODUCT_COMPUTATION,
        )
        design.singleton_network(node)
        design.kernels(
            "compute",
            covers=(node,),
            candidates=(DotpAxiKernel,),
            inputs=DotProductKernelInputs(
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
        pe = design.choice("pe", int, domain=finite((2,)))
        simd = design.choice("simd", int, domain=finite((2,)))
        node = design.region(
            "replay",
            node_id="replay",
            dependencies={"extent": design.inputs.extent},
            evaluate=_stream_region,
            computation=ACTIVATION_REPLAY_COMPUTATION,
        )
        design.singleton_network(node)
        design.kernels(
            "replay",
            covers=(node,),
            candidates=(ReplayBufferKernel,),
            inputs=ReplayBufferInputs(
                node.region,
                node.computation,
                design.inputs.extent,
                design.inputs.extent,
                pe,
                simd,
                design.inputs.activation_type,
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
            "delivery",
            node_id="delivery",
            dependencies={"output_port": output_port},
            evaluate=lambda output_port: construct_cyclic_parameter_region(output_port),
            computation=CYCLIC_PARAMETER_DELIVERY,
        )
        design.singleton_network(node)
        design.kernels(
            "delivery",
            covers=(node,),
            candidates=(FinnRtlMemstreamKernel,),
            inputs=FinnRtlMemstreamInputs(
                "delivery",
                node.region,
                node.computation,
                output_port,
                design.inputs.initializer_available,
                design.inputs.runtime_writable,
                design.inputs.memory_capabilities,
                ram_style,
                pumped,
            ),
        )


def problem_path(name: str) -> QualifiedPath:
    return QualifiedPath(f"problem.neutral.{name}")


def _neutral_inputs(operation: OpDesign) -> NeutralInputs:
    return NeutralInputs(
        operation.graph_fact("extent", int),
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
    )


def _problem_values(inputs: NeutralInputs) -> dict[QualifiedPath, object]:
    return {
        inputs.extent.path: 4,
        inputs.activation_type.path: INT8,
        inputs.weight_type.path: INT8,
        inputs.accumulator_type.path: INT16,
        inputs.output_type.path: INT16,
        inputs.narrow.path: False,
        inputs.target.path: DspBlock.DSP58,
        inputs.clock.path: 5.0,
        inputs.initializer_available.path: True,
        inputs.runtime_writable.path: False,
        inputs.memory_capabilities.path: CyclicTargetMemoryCapabilities(True),
    }


def _realize(
    design_type: type[DataflowDesign],
    assignments: dict[str, object],
) -> str:
    operation = OpDesign("neutral", problem_namespace="neutral")
    inputs = _neutral_inputs(operation)
    inventory = declare_dataflow_design_inventory(
        "neutral",
        (DataflowDesignEntry(design_type, inputs),),
        shared_specs=(operation.spec(),),
    )
    engine = Engine()
    point = engine.start(engine.validate(inventory.specification), _problem_values(inputs))
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
    assert isinstance(realized, Decided)
    return next(iter(realized.value.kernels.values())).id


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
    assert (
        _realize(
            NeutralDotProductDesign,
            {".pe": 2, ".simd": 2, ".compute_pumping": False},
        )
        == DotpAxiKernel.id
    )


def test_replay_buffer_kernel_binds_in_a_non_mvau_design() -> None:
    assert (
        _realize(
            NeutralReplayDesign,
            {".pe": 2, ".simd": 2},
        )
        == ReplayBufferKernel.id
    )


def test_memstream_kernel_binds_in_a_non_mvau_design() -> None:
    assert (
        _realize(
            NeutralMemstreamDesign,
            {".ram_style": CyclicRamStyle.BRAM, ".pumped": False},
        )
        == FinnRtlMemstreamKernel.id
    )
