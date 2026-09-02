# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Non-MVAU forcing operation for the complete class-authored Op frontend."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]
import numpy.typing as npt  # type: ignore[import-not-found]
from onnx import GraphProto, NodeProto  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.authoring import (
    Attribute,
    BuildFlag,
    Choice,
    ClosedDesigns,
    Covers,
    DataflowBuildConfigView,
    DataflowOp,
    InputTensor,
    Imported,
    Kernels,
    Network as NetworkDeclaration,
    NoInitializer,
    OptionalInitializer,
    OutputTensor,
    Parameter,
    Persist,
    RequiredInitializer,
    Region as RegionDeclaration,
    RegionClaim,
    SourceInput,
    SourceScope,
    TargetClockPeriod,
    TargetFpgaPart,
    TensorShape,
    UsesDesign,
    class_divisors_of,
    constraint,
    derived,
)
from finn.dataflow.authoring.design import DataflowDesign
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design import ABSENT, QONNX_DATATYPE_VALUE_SEMANTICS, QualifiedPath
from finn.dataflow.design import DATAFLOW_REGION_SEMANTICS
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.kernels import Kernel, PhysicalComponent, scalar_parameters
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)

AFFINE_COMPUTATION = ComputationContract("channelwise_affine")


@dataclass(frozen=True)
class AffineSourceAssociation:
    """Logical source tensors and their selected semantic destinations."""

    source_scope_id: str
    data: tuple[str, tuple[int, ...], str]
    scale: tuple[str, tuple[int, ...], str]
    bias: tuple[str, tuple[int, ...], str] | None
    output: tuple[str, tuple[int, ...], str]


@dataclass(frozen=True)
class AffineInputs:
    source_scope_id: Any
    repetitions: Any
    channels: Any
    data_id: Any
    data_shape: Any
    data_type: Any
    scale_id: Any
    scale_shape: Any
    bias_present: Any
    bias_id: Any
    bias_shape: Any
    output_id: Any
    output_shape: Any


@dataclass(frozen=True)
class AffineKernelInputs:
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    lanes: Ref[int]


def _affine_region(
    repetitions: int,
    channels: int,
    datatype: QONNXDataType,
    bias_present: bool,
    lanes: int,
    *,
    reuse: bool,
) -> DataflowRegion:
    if channels % lanes:
        raise ValueError("lanes must divide channels")
    schedule = (
        LogicalSchedule(
            (
                ScheduleLevel("channel_tile", channels // lanes),
                ScheduleLevel("repetition", repetitions),
                ScheduleLevel("lane", lanes),
            )
        )
        if reuse
        else LogicalSchedule(
            (
                ScheduleLevel("repetition", repetitions),
                ScheduleLevel("channel_tile", channels // lanes),
                ScheduleLevel("lane", lanes),
            )
        )
    )
    iterations = schedule.iteration_points
    output_positions = tuple(
        (iteration[1], iteration[0] * lanes + iteration[2])
        if reuse
        else (iteration[0], iteration[1] * lanes + iteration[2])
        for iteration in iterations
    )
    data = Operand("data", datatype, (repetitions, channels))
    scale = Operand("scale", datatype, (channels,))
    output = Operand("output", datatype, (repetitions, channels))
    data_requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        (iteration, position): 1 for iteration, position in zip(iterations, output_positions)
    }
    scale_requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        (iteration, (position[1],)): 1 for iteration, position in zip(iterations, output_positions)
    }
    interfaces = [
        InputInterface(
            Port("data", data, BeatSequence(1, tuple((item,) for item in output_positions))),
            ScheduledInputRequirements(data_requirements),
        ),
        InputInterface(
            Port(
                "scale",
                scale,
                BeatSequence(1, tuple(((item[1],),) for item in output_positions)),
            ),
            ScheduledInputRequirements(scale_requirements),
        ),
    ]
    if bias_present:
        bias = Operand("bias", datatype, (channels,))
        interfaces.append(
            InputInterface(
                Port(
                    "bias",
                    bias,
                    BeatSequence(1, tuple(((item[1],),) for item in output_positions)),
                ),
                ScheduledInputRequirements(scale_requirements),
            )
        )
    return DataflowRegion(
        schedule,
        tuple(interfaces),
        (
            OutputInterface(
                Port(
                    "output",
                    output,
                    BeatSequence(1, tuple((item,) for item in output_positions)),
                ),
                ScheduledOutputAvailability(
                    cast(
                        "dict[tuple[int, ...], tuple[int, ...]]",
                        dict(zip(output_positions, iterations)),
                    )
                ),
            ),
        ),
    )


def _association(
    source_scope_id: str,
    data_id: str,
    data_shape: tuple[int, ...],
    scale_id: str,
    scale_shape: tuple[int, ...],
    bias_present: bool,
    bias_id: object,
    bias_shape: object,
    output_id: str,
    output_shape: tuple[int, ...],
) -> AffineSourceAssociation:
    bias = None
    if bias_present:
        if bias_id is ABSENT or bias_shape is ABSENT:
            raise ValueError("active bias facts are unresolved")
        bias = (cast(str, bias_id), cast("tuple[int, ...]", bias_shape), "compute.bias")
    return AffineSourceAssociation(
        source_scope_id,
        (data_id, data_shape, "compute.data"),
        (scale_id, scale_shape, "compute.scale"),
        bias,
        (output_id, output_shape, "compute.output"),
    )


class AffineStreamKernel(Kernel):
    id = "affine_stream"
    version = "1"
    uses_class_authoring = True

    covered_region = Imported(DATAFLOW_REGION_SEMANTICS, stable_name="region")
    computation = Imported(ComputationContract)
    lanes = Imported(int)
    coverage = Covers(
        RegionClaim("compute", covered_region, computation, AFFINE_COMPUTATION),
    )
    pipeline = Choice(bool, domain=(False, True))
    lanes_parameter = Parameter("LANES", lanes)
    pipeline_parameter = Parameter("PIPELINE", pipeline)

    @classmethod
    def elaborate(cls, kernel: Kernel) -> tuple[PhysicalComponent, ...]:
        return (
            PhysicalComponent(
                "affine",
                "test.channelwise_affine",
                scalar_parameters(dict(kernel.parameters)),
            ),
        )


class DirectAffineDesign(DataflowDesign):
    id = "direct"
    version = "1"
    uses_class_authoring = True

    source_scope_id = Imported(str)
    repetitions = Imported(int)
    channels = Imported(int)
    data_id = Imported(str)
    data_shape = Imported(tuple)
    data_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    scale_id = Imported(str)
    scale_shape = Imported(tuple)
    bias_present = Imported(bool)
    bias_id = Imported(str)
    bias_shape = Imported(tuple)
    output_id = Imported(str)
    output_shape = Imported(tuple)

    lanes = Choice(int, domain=class_divisors_of(channels))
    compute = RegionDeclaration(
        node_id="compute",
        construct=lambda repetitions, channels, datatype, bias_present, lanes: _affine_region(
            repetitions,
            channels,
            datatype,
            bias_present,
            lanes,
            reuse=False,
        ),
        dependencies=(repetitions, channels, data_type, bias_present, lanes),
        computation=AFFINE_COMPUTATION,
    )
    network = NetworkDeclaration(compute)
    data_mapping = SourceInput("data", compute.input("data"), "input.data")
    scale_mapping = SourceInput("scale", compute.input("scale"), "input.scale")
    compute_placement = Kernels(
        name="compute",
        covers=(compute,),
        candidates=(AffineStreamKernel,),
        inputs=AffineKernelInputs(
            cast("Ref[DataflowRegion]", compute.region),
            cast("Ref[ComputationContract]", compute.computation),
            cast("Ref[int]", lanes),
        ),
    )
    source_association = derived(
        source_scope_id,
        data_id,
        data_shape,
        scale_id,
        scale_shape,
        bias_present,
        bias_id.allow_absent(),
        bias_shape.allow_absent(),
        output_id,
        output_shape,
        value_type=AffineSourceAssociation,
    )(_association)


class ReuseAffineDesign(DataflowDesign):
    id = "reuse"
    version = "1"
    uses_class_authoring = True

    source_scope_id = Imported(str)
    repetitions = Imported(int)
    channels = Imported(int)
    data_id = Imported(str)
    data_shape = Imported(tuple)
    data_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    scale_id = Imported(str)
    scale_shape = Imported(tuple)
    bias_present = Imported(bool)
    bias_id = Imported(str)
    bias_shape = Imported(tuple)
    output_id = Imported(str)
    output_shape = Imported(tuple)

    lanes = Choice(int, domain=class_divisors_of(channels))
    channel_tile = Choice(int, domain=class_divisors_of(channels))
    compute = RegionDeclaration(
        node_id="compute",
        construct=lambda repetitions, channels, datatype, bias_present, lanes, channel_tile: (
            _affine_region(
                repetitions,
                channels,
                datatype,
                bias_present,
                lanes,
                reuse=bool(channel_tile),
            )
        ),
        dependencies=(
            repetitions,
            channels,
            data_type,
            bias_present,
            lanes,
            channel_tile,
        ),
        computation=AFFINE_COMPUTATION,
    )
    network = NetworkDeclaration(compute)
    data_mapping = SourceInput("data", compute.input("data"), "input.data")
    scale_mapping = SourceInput("scale", compute.input("scale"), "input.scale")
    compute_placement = Kernels(
        name="compute",
        covers=(compute,),
        candidates=(AffineStreamKernel,),
        inputs=AffineKernelInputs(
            cast("Ref[DataflowRegion]", compute.region),
            cast("Ref[ComputationContract]", compute.computation),
            cast("Ref[int]", lanes),
        ),
    )
    source_association = derived(
        source_scope_id,
        data_id,
        data_shape,
        scale_id,
        scale_shape,
        bias_present,
        bias_id.allow_absent(),
        bias_shape.allow_absent(),
        output_id,
        output_shape,
        value_type=AffineSourceAssociation,
    )(_association)


class ChannelwiseAffineDataflowOp(DataflowOp):
    family_id = "finn.dataflow.channelwise_affine"
    family_version = "channelwise-affine-v1"
    declaration_namespace = "channelwise_affine"
    uses_class_authoring = True

    source_scope_id = SourceScope()
    use_bias = Attribute("use_bias", bool, default=True)
    saturate = Attribute("saturate", bool, default=False)
    data = InputTensor(
        "data",
        index=0,
        shape=TensorShape(min_rank=2),
        initializer=NoInitializer(),
    )
    scale = InputTensor(
        "scale",
        index=1,
        shape=TensorShape(rank=1),
        initializer=OptionalInitializer(fingerprint=True),
    )
    bias = InputTensor(
        "bias",
        index=2,
        when=use_bias,
        shape=TensorShape(rank=1),
        initializer=RequiredInitializer(fingerprint=True),
    )
    output = OutputTensor("output", index=0, shape=TensorShape(min_rank=2))

    @derived(data.shape, value_type=int)
    def channels(shape: tuple[int, ...]) -> int:
        return shape[-1]

    @derived(data.shape, value_type=int)
    def repetitions(shape: tuple[int, ...]) -> int:
        return prod(shape[:-1])

    @constraint(scale.shape, channels, sets=("channelwise_affine.structural",))
    def scale_shape_supported(shape: tuple[int, ...], channels: int) -> bool:
        return shape == (channels,)

    @constraint(
        bias.shape,
        channels,
        when=bias.present,
        sets=("channelwise_affine.structural",),
    )
    def bias_shape_supported(shape: tuple[int, ...], channels: int) -> bool:
        return shape == (channels,)

    @constraint(data.shape, output.shape, sets=("channelwise_affine.structural",))
    def output_shape_supported(data_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> bool:
        return data_shape == output_shape

    target_part = TargetFpgaPart()
    target_clock = TargetClockPeriod()
    runtime_coefficients = BuildFlag("runtime_coefficients", default=False)

    design_inputs = AffineInputs(
        source_scope_id,
        repetitions,
        channels,
        data.tensor_id,
        data.shape,
        data.datatype,
        scale.tensor_id,
        scale.shape,
        bias.present,
        bias.tensor_id,
        bias.shape,
        output.tensor_id,
        output.shape,
    )
    direct = UsesDesign(DirectAffineDesign, design_inputs)
    reuse = UsesDesign(ReuseAffineDesign, design_inputs)
    designs = ClosedDesigns(direct, reuse)

    persistence = (
        Persist(designs.choice, "dataflow_affine_design"),
        Persist(direct.lanes, "dataflow_affine_direct_lanes"),
        Persist(reuse.lanes, "dataflow_affine_reuse_lanes"),
        Persist(reuse.channel_tile, "dataflow_affine_channel_tile"),
        Persist(
            direct.compute.affine_stream.pipeline,
            "dataflow_affine_direct_pipeline",
        ),
        Persist(
            reuse.compute.affine_stream.pipeline,
            "dataflow_affine_reuse_pipeline",
        ),
    )

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        del model
        return self.make_const_shape_op(self.declared_tensor_shape("data"))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        model.set_tensor_datatype(self.onnx_node.output[0], self.declared_tensor_datatype("data"))

    def execute_node(self, context: dict[str, npt.NDArray], graph: GraphProto) -> None:
        del graph
        data = np.asarray(context[self.onnx_node.input[0]])
        scale = np.asarray(context[self.onnx_node.input[1]])
        result = data * scale
        if bool(self.get_nodeattr("use_bias")):
            result = result + np.asarray(context[self.onnx_node.input[2]])
        context[self.onnx_node.output[0]] = result

    def verify_node(self) -> None:
        self.validate_declared_source()


@dataclass(frozen=True)
class AffineBuildConfig(DataflowBuildConfigView):
    synth_clk_period_ns: float = 5.0
    fpga_part: str = "xczu3eg-sbva484-1-e"

    def _resolve_fpga_part(self) -> str:
        return self.fpga_part


DIRECT_DESIGN = QualifiedPath("channelwise_affine.design")
DIRECT_LANES = QualifiedPath("channelwise_affine.design.direct.lanes")
DIRECT_PIPELINE = QualifiedPath("channelwise_affine.design.direct.compute.affine_stream.pipeline")


__all__ = [
    "AffineBuildConfig",
    "AffineSourceAssociation",
    "ChannelwiseAffineDataflowOp",
    "DIRECT_DESIGN",
    "DIRECT_LANES",
    "DIRECT_PIPELINE",
]
