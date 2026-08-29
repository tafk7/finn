# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Small non-MVAU logical operation used by DataflowOp conformance tests."""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from collections.abc import Mapping
from enum import Enum
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import numpy.typing as npt  # type: ignore[import-not-found]
from onnx import GraphProto, NodeProto  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.design import (
    Answer,
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
from finn.dataflow.authoring import DataflowBuildConfigView, DataflowOp, NodeAttrCodec
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

__all__ = ["SyntheticDataflowOp", "ZeroDecisionDataflowOp"]


class SyntheticMode(str, Enum):
    FIRST = "first"
    SECOND = "second"


class SyntheticPaths:
    SOURCE_ID = QualifiedPath("problem.synthetic.source_id")
    EXTENT = QualifiedPath("problem.synthetic.extent")
    CLOCK = QualifiedPath("problem.target.clock_period_ns")
    LANES = QualifiedPath("synthetic.lanes")
    ENABLED = QualifiedPath("synthetic.enabled")
    LABEL = QualifiedPath("synthetic.label")
    MODE = QualifiedPath("synthetic.mode")
    ASSOCIATION = QualifiedPath("semantic.synthetic.source_association")
    REGION = QualifiedPath("semantic.synthetic.region")
    RESULT = QualifiedPath("semantic.synthetic.result")


_INT = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
_BOOL = as_object_semantics(ValueSemantics.immutable_nominal(bool, name="boolean"))
_STR = as_object_semantics(ValueSemantics.immutable_nominal(str, name="string"))
_FLOAT = as_object_semantics(ValueSemantics.immutable_nominal(float, name="float"))
_MODE = as_object_semantics(ValueSemantics.immutable_nominal(SyntheticMode, name="SyntheticMode"))
_REGION = as_object_semantics(
    ValueSemantics.immutable_nominal(DataflowRegion, name="DataflowRegion")
)


def _finite(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    return DecisionDomain((), accepts, EvaluatorSpec((), lambda _deps: Decided(values)))


def _derive_association(dependencies: DependencyView) -> Answer[object]:
    return Decided(cast(str, dependencies["source_id"]))


def _region(extent: int) -> DataflowRegion:
    element_type = DataType["INT8"]
    source = Operand("x", element_type, (extent,))
    result = Operand("y", element_type, (extent,))
    beats = BeatSequence(1, tuple((((index,),)) for index in range(extent)))
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


def _derive_region(dependencies: DependencyView) -> Answer[object]:
    extent = cast(int, dependencies["extent"])
    # Read every decision so the synthetic result exercises every codec.
    cast(int, dependencies["lanes"])
    cast(bool, dependencies["enabled"])
    cast(str, dependencies["label"])
    cast(SyntheticMode, dependencies["mode"])
    return Decided(_region(extent))


def _derive_fixed_region(dependencies: DependencyView) -> Answer[object]:
    return Decided(_region(cast(int, dependencies["extent"])))


def _derive_result(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        RegionRef(
            "synthetic",
            cast(DataflowRegion, dependencies["region"]),
            cast(str, dependencies["association"]),
        )
    )


def build_synthetic_spec() -> DesignSpaceSpec:
    source_ref = DependencyRef.problem("source_id", SyntheticPaths.SOURCE_ID, _STR)
    extent_ref = DependencyRef.problem("extent", SyntheticPaths.EXTENT, _INT)
    lanes_ref = DependencyRef.decision("lanes", SyntheticPaths.LANES, _INT)
    enabled_ref = DependencyRef.decision("enabled", SyntheticPaths.ENABLED, _BOOL)
    label_ref = DependencyRef.decision("label", SyntheticPaths.LABEL, _STR)
    mode_ref = DependencyRef.decision("mode", SyntheticPaths.MODE, _MODE)
    association_ref = DependencyRef.property("association", SyntheticPaths.ASSOCIATION, _STR)
    region_ref = DependencyRef.property("region", SyntheticPaths.REGION, _REGION)
    return DesignSpaceSpec(
        ProblemSchema(
            (
                ProblemField(SyntheticPaths.SOURCE_ID, _STR),
                ProblemField(SyntheticPaths.EXTENT, _INT),
                ProblemField(SyntheticPaths.CLOCK, _FLOAT),
            )
        ),
        decisions=(
            Decision(SyntheticPaths.LANES, _INT, _finite((1, 2, 4))),
            Decision(SyntheticPaths.ENABLED, _BOOL, _finite((False, True))),
            Decision(SyntheticPaths.LABEL, _STR, _finite(("alpha", "beta"))),
            Decision(SyntheticPaths.MODE, _MODE, _finite(tuple(SyntheticMode))),
        ),
        properties=(
            DerivedProperty(
                SyntheticPaths.ASSOCIATION,
                _STR,
                EvaluatorSpec((source_ref,), _derive_association),
            ),
            DerivedProperty(
                SyntheticPaths.REGION,
                _REGION,
                EvaluatorSpec(
                    (extent_ref, lanes_ref, enabled_ref, label_ref, mode_ref),
                    _derive_region,
                ),
            ),
            DerivedProperty(
                SyntheticPaths.RESULT,
                DATAFLOW_OP_RESULT_SEMANTICS,
                EvaluatorSpec((region_ref, association_ref), _derive_result),
            ),
        ),
    )


def build_zero_decision_spec() -> DesignSpaceSpec:
    source_ref = DependencyRef.problem("source_id", SyntheticPaths.SOURCE_ID, _STR)
    extent_ref = DependencyRef.problem("extent", SyntheticPaths.EXTENT, _INT)
    association_ref = DependencyRef.property("association", SyntheticPaths.ASSOCIATION, _STR)
    region_ref = DependencyRef.property("region", SyntheticPaths.REGION, _REGION)
    return DesignSpaceSpec(
        ProblemSchema(
            (
                ProblemField(SyntheticPaths.SOURCE_ID, _STR),
                ProblemField(SyntheticPaths.EXTENT, _INT),
                ProblemField(SyntheticPaths.CLOCK, _FLOAT),
            )
        ),
        properties=(
            DerivedProperty(
                SyntheticPaths.ASSOCIATION,
                _STR,
                EvaluatorSpec((source_ref,), _derive_association),
            ),
            DerivedProperty(
                SyntheticPaths.REGION,
                _REGION,
                EvaluatorSpec((extent_ref,), _derive_fixed_region),
            ),
            DerivedProperty(
                SyntheticPaths.RESULT,
                DATAFLOW_OP_RESULT_SEMANTICS,
                EvaluatorSpec((region_ref, association_ref), _derive_result),
            ),
        ),
    )


class SyntheticDataflowOp(DataflowOp):
    """Identity operation used to exercise the generic node lifecycle."""

    spec_build_count = 0

    @classmethod
    def dataflow_family_id(cls) -> str:
        return "test.synthetic"

    @classmethod
    def dataflow_family_version(cls) -> str:
        return "1"

    @classmethod
    def build_design_space_spec(cls) -> DesignSpaceSpec:
        cls.spec_build_count += 1
        return build_synthetic_spec()

    @classmethod
    def result_path(cls) -> QualifiedPath:
        return SyntheticPaths.RESULT

    @classmethod
    def source_association_path(cls) -> QualifiedPath:
        return SyntheticPaths.ASSOCIATION

    @classmethod
    def decision_nodeattrs(cls) -> Mapping[QualifiedPath, NodeAttrCodec]:
        return {
            SyntheticPaths.LANES: NodeAttrCodec.integer("dataflow_lanes"),
            SyntheticPaths.ENABLED: NodeAttrCodec.boolean("dataflow_enabled"),
            SyntheticPaths.LABEL: NodeAttrCodec.string("dataflow_label"),
            SyntheticPaths.MODE: NodeAttrCodec.finite_enum(
                "dataflow_mode",
                SyntheticMode,
                {"first": SyntheticMode.FIRST, "second": SyntheticMode.SECOND},
            ),
        }

    def project_graph_problem(self) -> dict[QualifiedPath, object]:
        model = self._attached_model()
        shape = model.get_tensor_shape(self.onnx_node.input[0])
        if shape is None or len(shape) != 1 or shape[0] <= 0:
            raise ValueError("synthetic input requires one positive dimension")
        return {
            SyntheticPaths.SOURCE_ID: self.dataflow_scope_id(),
            SyntheticPaths.EXTENT: int(shape[0]),
        }

    def project_build_problem(self, config: DataflowBuildConfigView) -> dict[QualifiedPath, object]:
        return {SyntheticPaths.CLOCK: float(config.synth_clk_period_ns)}

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        shape = model.get_tensor_shape(self.onnx_node.input[0])
        if shape is None:
            raise ValueError("synthetic input shape is unavailable")
        return self.make_const_shape_op(shape)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        datatype = model.get_tensor_datatype(self.onnx_node.input[0])
        model.set_tensor_datatype(self.onnx_node.output[0], datatype)

    def execute_node(self, context: dict[str, npt.NDArray], graph: GraphProto) -> None:
        del graph
        context[self.onnx_node.output[0]] = np.asarray(context[self.onnx_node.input[0]]).copy()

    def verify_node(self) -> None:
        if len(self.onnx_node.input) != 1 or len(self.onnx_node.output) != 1:
            raise ValueError("SyntheticDataflowOp requires one input and one output")


class ZeroDecisionDataflowOp(SyntheticDataflowOp):
    """Complete logical operation whose static family has no decisions."""

    @classmethod
    def dataflow_family_id(cls) -> str:
        return "test.synthetic.zero_decision"

    @classmethod
    def build_design_space_spec(cls) -> DesignSpaceSpec:
        return build_zero_decision_spec()

    @classmethod
    def decision_nodeattrs(cls) -> Mapping[QualifiedPath, NodeAttrCodec]:
        return {}
