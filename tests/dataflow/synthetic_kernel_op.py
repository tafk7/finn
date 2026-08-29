# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A synthetic Op whose two Kernels derive equal Regions with distinct identity.

The fixture exists to hold the Phase 1 authoring contract still:  ``Op ->
Kernel pool -> selected Kernel -> Region`` with no separate binding selection
layer, and Region equality that does not collapse Kernel identity.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from collections.abc import Mapping
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import numpy.typing as npt  # type: ignore[import-not-found]
from onnx import GraphProto, NodeProto  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.authoring import DataflowBuildConfigView, DataflowOp, NodeAttrCodec
from finn.dataflow.design import (
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
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import (
    SELECTED_KERNEL_SEMANTICS,
    KernelDeclaration,
    KernelDemand,
    KernelProvider,
    KernelSelection,
    SelectedKernel,
)
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
from finn.dataflow.resolution import DATAFLOW_OP_RESULT_SEMANTICS, RegionRef
from finn.dataflow.spec_algebra import assemble_specs

__all__ = [
    "PAIRED_SELECTION",
    "PairedKernelDataflowOp",
    "PairedPaths",
    "build_paired_kernel_op_spec",
]

_INT = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
_STR = as_object_semantics(ValueSemantics.immutable_nominal(str, name="string"))
_FLOAT = as_object_semantics(ValueSemantics.immutable_nominal(float, name="float"))
_REGION = as_object_semantics(
    ValueSemantics.immutable_nominal(DataflowRegion, name="DataflowRegion")
)
_PORT = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))


class PairedPaths:
    """Paths owned by the synthetic Op outside its Kernel pool."""

    SOURCE_ID = QualifiedPath("problem.paired.source_id")
    EXTENT = QualifiedPath("problem.paired.extent")
    CLOCK = QualifiedPath("problem.target.clock_period_ns")
    ASSOCIATION = QualifiedPath("semantic.paired.source_association")
    RESULT = QualifiedPath("semantic.paired.result")


def _finite(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    return DecisionDomain((), accepts, EvaluatorSpec((), lambda _deps: Decided(values)))


def _region(extent: int) -> DataflowRegion:
    element_type = DataType["INT8"]
    source = Operand("x", element_type, (extent,))
    result = Operand("y", element_type, (extent,))
    beats = BeatSequence(1, tuple(((index,),) for index in range(extent)))
    schedule = LogicalSchedule((ScheduleLevel("element", extent),))
    requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        ((index,), (index,)): 1 for index in range(extent)
    }
    availability: dict[tuple[int, ...], tuple[int, ...]] = {
        (index,): (index,) for index in range(extent)
    }
    return DataflowRegion(
        schedule,
        (InputInterface(Port("input", source, beats), ScheduledInputRequirements(requirements)),),
        (
            OutputInterface(
                Port("output", result, beats), ScheduledOutputAvailability(availability)
            ),
        ),
    )


def _build_kernel(
    kernel_id: str,
    *,
    lane_values: tuple[object, ...],
    demands_parameter: bool,
    providers: tuple[str, ...],
) -> KernelDeclaration:
    lanes = QualifiedPath(f"paired.{kernel_id}.lanes")
    region = QualifiedPath(f"semantic.paired.{kernel_id}.region")
    demand = QualifiedPath(f"semantic.paired.{kernel_id}.parameter_demand")
    lanes_fit = QualifiedPath(f"constraint.paired.{kernel_id}.lanes_fit")
    extent_admitted = QualifiedPath(f"constraint.paired.{kernel_id}.extent_admitted")
    extent_ref = DependencyRef.problem("extent", PairedPaths.EXTENT, _INT)
    lanes_ref = DependencyRef.decision("lanes", lanes, _INT)

    def derive_region(dependencies: DependencyView) -> Answer[object]:
        return Decided(_region(cast(int, dependencies["extent"])))

    def derive_demand(dependencies: DependencyView) -> Answer[object]:
        extent = cast(int, dependencies["extent"])
        element_type = DataType["INT8"]
        return Decided(
            Port(
                "parameter",
                Operand("w", element_type, (extent,)),
                BeatSequence(1, tuple(((index,),) for index in range(extent))),
            )
        )

    def lanes_fit_extent(dependencies: DependencyView) -> Answer[bool]:
        return Decided(cast(int, dependencies["extent"]) % cast(int, dependencies["lanes"]) == 0)

    def extent_is_admitted(dependencies: DependencyView) -> Answer[bool]:
        # A Kernel-owned source-admission rule, not a transform-owned switch.
        extent = cast(int, dependencies["extent"])
        return Decided(extent % 2 == 0 if kernel_id == "even" else True)

    properties = [
        DerivedProperty(region, _REGION, EvaluatorSpec((extent_ref,), derive_region)),
    ]
    demands: tuple[KernelDemand, ...] = ()
    if demands_parameter:
        properties.append(
            DerivedProperty(demand, _PORT, EvaluatorSpec((extent_ref,), derive_demand))
        )
        demands = (KernelDemand("parameter", demand),)
    spec = DesignSpaceSpec(
        decisions=(Decision(lanes, _INT, _finite(lane_values)),),
        properties=tuple(properties),
        constraints=(
            Constraint(lanes_fit, EvaluatorSpec((extent_ref, lanes_ref), lanes_fit_extent)),
            Constraint(extent_admitted, EvaluatorSpec((extent_ref,), extent_is_admitted)),
        ),
    )
    return KernelDeclaration(
        kernel_id,
        "1",
        spec,
        region,
        feasibility_constraints=(lanes_fit, extent_admitted),
        source_admission_constraints=(extent_admitted,),
        demands=demands,
        providers=tuple(KernelProvider(name, kernel_id) for name in providers),
    )


PAIRED_SELECTION = KernelSelection(
    "paired.compute",
    (
        _build_kernel(
            "even",
            lane_values=(1, 2),
            demands_parameter=True,
            providers=("even.rtl", "even.hls"),
        ),
        _build_kernel(
            "any",
            lane_values=(1, 3),
            demands_parameter=False,
            providers=("any.rtl",),
        ),
    ),
)


def _derive_association(dependencies: DependencyView) -> Answer[object]:
    return Decided(cast(str, dependencies["source_id"]))


def _derive_result(dependencies: DependencyView) -> Answer[object]:
    selected = cast(SelectedKernel, dependencies["selected_kernel"])
    return Decided(
        RegionRef(
            f"paired.{selected.kernel_id}",
            cast(DataflowRegion, dependencies["region"]),
            cast(str, dependencies["association"]),
        )
    )


def build_paired_kernel_op_spec() -> DesignSpaceSpec:
    """Assemble the synthetic Op from its static Kernel pool."""

    paths = PAIRED_SELECTION.paths
    additions = DesignSpaceSpec(
        ProblemSchema(
            (
                ProblemField(PairedPaths.SOURCE_ID, _STR),
                ProblemField(PairedPaths.EXTENT, _INT),
                ProblemField(PairedPaths.CLOCK, _FLOAT),
            )
        ),
        properties=(
            DerivedProperty(
                PairedPaths.ASSOCIATION,
                _STR,
                EvaluatorSpec(
                    (DependencyRef.problem("source_id", PairedPaths.SOURCE_ID, _STR),),
                    _derive_association,
                ),
            ),
            DerivedProperty(
                PairedPaths.RESULT,
                DATAFLOW_OP_RESULT_SEMANTICS,
                EvaluatorSpec(
                    (
                        DependencyRef.property("region", paths.region, _REGION),
                        DependencyRef.property(
                            "selected_kernel",
                            paths.selected_kernel,
                            SELECTED_KERNEL_SEMANTICS,
                        ),
                        DependencyRef.property("association", PairedPaths.ASSOCIATION, _STR),
                    ),
                    _derive_result,
                ),
            ),
        ),
    )
    return assemble_specs((PAIRED_SELECTION.build_spec(), additions))


class PairedKernelDataflowOp(DataflowOp):
    """Identity operation selecting one of two equal-Region Kernels."""

    @classmethod
    def dataflow_family_id(cls) -> str:
        return "test.paired_kernel"

    @classmethod
    def dataflow_family_version(cls) -> str:
        return "1"

    @classmethod
    def build_design_space_spec(cls) -> DesignSpaceSpec:
        return build_paired_kernel_op_spec()

    @classmethod
    def result_path(cls) -> QualifiedPath:
        return PairedPaths.RESULT

    @classmethod
    def source_association_path(cls) -> QualifiedPath:
        return PairedPaths.ASSOCIATION

    @classmethod
    def decision_nodeattrs(cls) -> Mapping[QualifiedPath, NodeAttrCodec]:
        return {
            PAIRED_SELECTION.paths.kernel: NodeAttrCodec.string("dataflow_paired_kernel"),
            QualifiedPath("paired.even.lanes"): NodeAttrCodec.integer("dataflow_even_lanes"),
            QualifiedPath("paired.any.lanes"): NodeAttrCodec.integer("dataflow_any_lanes"),
        }

    def project_graph_problem(self) -> dict[QualifiedPath, object]:
        model = self._attached_model()
        shape = model.get_tensor_shape(self.onnx_node.input[0])
        if shape is None or len(shape) != 1 or shape[0] <= 0:
            raise ValueError("paired input requires one positive dimension")
        return {
            PairedPaths.SOURCE_ID: self.dataflow_scope_id(),
            PairedPaths.EXTENT: int(shape[0]),
        }

    def project_build_problem(self, config: DataflowBuildConfigView) -> dict[QualifiedPath, object]:
        return {PairedPaths.CLOCK: float(config.synth_clk_period_ns)}

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        shape = model.get_tensor_shape(self.onnx_node.input[0])
        if shape is None:
            raise ValueError("paired input shape is unavailable")
        return self.make_const_shape_op(shape)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        model.set_tensor_datatype(
            self.onnx_node.output[0], model.get_tensor_datatype(self.onnx_node.input[0])
        )

    def execute_node(self, context: dict[str, npt.NDArray], graph: GraphProto) -> None:
        del graph
        context[self.onnx_node.output[0]] = np.asarray(context[self.onnx_node.input[0]]).copy()

    def verify_node(self) -> None:
        if len(self.onnx_node.input) != 1 or len(self.onnx_node.output) != 1:
            raise ValueError("PairedKernelDataflowOp requires one input and one output")
