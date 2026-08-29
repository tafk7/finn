# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical model-aware MVAU custom operation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import numpy.typing as npt  # type: ignore[import-not-found]
from onnx import GraphProto, NodeProto  # type: ignore[import-not-found]
import qonnx.custom_op.general.xnorpopcount as xp  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.general.multithreshold import (  # type: ignore[import-not-found]
    multithreshold,
)

from finn.dataflow.authoring import (
    DataflowBuildConfigView,
    DataflowOp,
    NodeAttrCodec,
    NodeAttributeType,
)
from finn.dataflow.design import DesignSpaceSpec, Engine, Finding, FindingKind, QualifiedPath
from finn.dataflow.mvau.assignments import MVAU_DECISION_NODEATTRS
from finn.dataflow.mvau.source import (
    MVAU_LOGICAL_SOURCE_NODEATTRS,
    MVAUProjectionContext,
    MVAUResolvedDesign,
    MVAUSourceAdapterError,
    MVAUSourceProjection,
    classify_mvau_dsp_block,
    project_mvau_build_problem,
    project_mvau_graph_source,
    resolve_mvau_point,
)
from finn.dataflow.kernels import KernelSelection
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAU_SELECTIONS,
    MVAUDataflowOpPaths,
    MVAUSourceDescription,
)
from finn.dataflow.mvau_problem import MVAU_PROBLEM_PROVENANCE, MVAUProblemPaths
from finn.dataflow.region import BeatSequence

#: v3 adds the replay Kernel decision and its ``dataflow_replay_kernel``
#: attribute to the persisted set.
MVAU_DATAFLOW_OP_FAMILY_VERSION = "mvau-dataflow-op-v3"


@dataclass(frozen=True)
class MVAUDataflowBuildContext:
    """Narrow MVAU invocation view around a real build configuration."""

    build_config: DataflowBuildConfigView
    accumulator_type_analysis_owner: str = "finn.MinimizeAccumulatorWidth"
    runtime_writable_weights: bool = False
    external_weight_sequence: BeatSequence | None = None
    supports_initialized_uram: bool | None = None

    @property
    def synth_clk_period_ns(self) -> float:
        return self.build_config.synth_clk_period_ns


class MvauDataflowOp(DataflowOp):
    """Logical MVAU source operation backed by the current static superspace."""

    @classmethod
    def dataflow_family_id(cls) -> str:
        return "finn.dataflow.mvau"

    @classmethod
    def dataflow_family_version(cls) -> str:
        return MVAU_DATAFLOW_OP_FAMILY_VERSION

    @classmethod
    def build_design_space_spec(cls) -> DesignSpaceSpec:
        return MVAU_DATAFLOW_OP_SPEC

    @classmethod
    def result_path(cls) -> QualifiedPath:
        return MVAUDataflowOpPaths.RESULT

    @classmethod
    def source_association_path(cls) -> QualifiedPath:
        return MVAUDataflowOpPaths.SOURCE_ASSOCIATION

    @classmethod
    def kernel_selections(cls) -> tuple[KernelSelection, ...]:
        return MVAU_SELECTIONS

    @classmethod
    def selection_constraint_set(cls) -> str | None:
        return "mvau_op_feasibility"

    @classmethod
    def structural_readiness_profile(cls) -> str | None:
        return "mvau_op_structural"

    @classmethod
    def artifact_readiness_profile(cls) -> str | None:
        return "artifact_inputs"

    @classmethod
    def source_nodeattr_types(cls) -> Mapping[str, NodeAttributeType]:
        return MVAU_LOGICAL_SOURCE_NODEATTRS

    @classmethod
    def decision_nodeattrs(cls) -> Mapping[QualifiedPath, NodeAttrCodec]:
        return MVAU_DECISION_NODEATTRS

    def _graph_projection(self) -> MVAUSourceProjection:
        projection = project_mvau_graph_source(
            self._attached_model(),
            self.onnx_node.name,
            source_scope_id=self.dataflow_scope_id(),
        )
        if projection.blocking_findings:
            raise MVAUSourceAdapterError(projection.blocking_findings)
        return projection

    def project_graph_problem(self) -> Mapping[QualifiedPath, object]:
        problem = self._graph_projection().problem_data
        MVAU_PROBLEM_PROVENANCE.check_graph_projection(problem)
        return problem

    @staticmethod
    def _context_parts(
        config: DataflowBuildConfigView,
    ) -> tuple[DataflowBuildConfigView, str, bool, BeatSequence | None, bool | None]:
        if isinstance(config, MVAUDataflowBuildContext):
            return (
                config.build_config,
                config.accumulator_type_analysis_owner,
                config.runtime_writable_weights,
                config.external_weight_sequence,
                config.supports_initialized_uram,
            )
        return config, "finn.MinimizeAccumulatorWidth", False, None, None

    @staticmethod
    def _target_part(config: DataflowBuildConfigView) -> str | None:
        resolver = getattr(config, "_resolve_fpga_part", None)
        if callable(resolver):
            try:
                value = resolver()
            except (KeyError, TypeError):
                value = None
            return value if isinstance(value, str) and value else None
        value = getattr(config, "fpga_part", None)
        return value if isinstance(value, str) and value else None

    def project_build_problem(
        self, config: DataflowBuildConfigView
    ) -> Mapping[QualifiedPath, object]:
        base, owner, runtime_writable, external_sequence, supports_uram = self._context_parts(
            config
        )
        context = MVAUProjectionContext(
            owner,
            fpga_part=self._target_part(base),
            clock_period_ns=float(base.synth_clk_period_ns),
            supports_initialized_uram=supports_uram,
            external_weight_sequence=external_sequence,
            runtime_writable_weights=runtime_writable,
        )
        if context.fpga_part is not None and classify_mvau_dsp_block(context.fpga_part) is None:
            raise MVAUSourceAdapterError(
                (
                    Finding(
                        FindingKind.LIMITATION,
                        "mvau-target-part-unknown",
                        MVAUProblemPaths.TARGET_DSP_BLOCK,
                        "target FPGA part cannot be classified into a supported DSP family",
                        values=(("fpga_part", context.fpga_part),),
                    ),
                )
            )
        problem = dict(
            project_mvau_build_problem(
                context,
                runtime_writable_weights=runtime_writable,
            )
        )
        problem[MVAUDataflowOpPaths.ACCUMULATOR_TYPE_ANALYSIS_OWNER] = owner
        MVAU_PROBLEM_PROVENANCE.check_build_projection(problem)
        return problem

    def resolve_dataflow(self, config: DataflowBuildConfigView) -> MVAUResolvedDesign:
        point = self.hydrate_dataflow_point(config)
        description = cast(
            MVAUSourceDescription,
            point.problem[MVAUDataflowOpPaths.SOURCE_DESCRIPTION],
        )
        projection = MVAUSourceProjection(description, point.problem, {}, ())
        return resolve_mvau_point(
            Engine(),
            point,
            projection,
            source_scope_id=self.dataflow_scope_id(),
        )

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        activation_shape = model.get_tensor_shape(self.onnx_node.input[0])
        weight_shape = model.get_tensor_shape(self.onnx_node.input[1])
        if activation_shape is None or weight_shape is None or len(weight_shape) != 2:
            raise ValueError("logical MVAU requires concrete activation and weight shapes")
        return self.make_const_shape_op((*activation_shape[:-1], weight_shape[1]))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        if bool(self.get_nodeattr("noActivation")):
            model.set_tensor_datatype(
                self.onnx_node.output[0], DataType[cast(str, self.get_nodeattr("accDataType"))]
            )
            return
        output_type = model.get_tensor_datatype(self.onnx_node.output[0])
        model.set_tensor_datatype(self.onnx_node.output[0], output_type)

    def execute_node(self, context: dict[str, npt.NDArray], graph: GraphProto) -> None:
        del graph
        model = self._attached_model()
        activation = np.asarray(context[self.onnx_node.input[0]])
        weight = np.asarray(context[self.onnx_node.input[1]])
        if bool(self.get_nodeattr("binaryXnorMode")):
            result = xp.xnorpopcountmatmul(activation, weight)
        else:
            activation_type = model.get_tensor_datatype(self.onnx_node.input[0])
            weight_type = model.get_tensor_datatype(self.onnx_node.input[1])
            if activation_type == DataType["BIPOLAR"] and weight_type == DataType["BIPOLAR"]:
                result = xp.xnorpopcountmatmul((activation + 1) / 2, (weight + 1) / 2)
            else:
                result = np.matmul(activation, weight)
        if not bool(self.get_nodeattr("noActivation")):
            thresholds = np.asarray(context[self.onnx_node.input[2]])
            output_type = model.get_tensor_datatype(self.onnx_node.output[0])
            output_scale = 2 if output_type == DataType["BIPOLAR"] else 1
            output_bias = -1 if output_type == DataType["BIPOLAR"] else self.get_nodeattr("ActVal")
            if result.ndim == 4:
                result = result.transpose((0, 3, 1, 2))
            result = multithreshold(result, thresholds, output_scale, output_bias)
            if result.ndim == 4:
                result = result.transpose((0, 2, 3, 1))
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError("logical MVAU output shape is unavailable")
        context[self.onnx_node.output[0]] = result.reshape(output_shape)

    def verify_node(self) -> None:
        no_activation = bool(self.get_nodeattr("noActivation"))
        expected_inputs = 2 if no_activation else 3
        if len(self.onnx_node.input) != expected_inputs or len(self.onnx_node.output) != 1:
            raise ValueError(f"MvauDataflowOp requires {expected_inputs} inputs and one output")
        # Reuse the source adapter's complete graph/type/shape checks.
        self._graph_projection()


__all__ = [
    "MVAU_DATAFLOW_OP_FAMILY_VERSION",
    "MVAUDataflowBuildContext",
    "MvauDataflowOp",
]
