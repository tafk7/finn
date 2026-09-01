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
    Attribute,
    BuildFact,
    DataflowBuildConfigView,
    DataflowOp,
    DatatypeAttribute,
    InitializerAnalysis,
    InputTensor,
    NoInitializer,
    NodeAttrCodec,
    NodeAttributeType,
    OptionalInitializer,
    OutputTensor,
    Persist,
    RequiredInitializer,
    SourceScope,
    TensorShape,
    constraint,
    derived,
    not_,
)
from finn.dataflow.authoring.op_design import Provenance
from finn.dataflow.authoring.inventory import DataflowOpAuthoring
from finn.dataflow.design import ABSENT, Engine, Finding, FindingKind, QualifiedPath
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.ops.mvau._adapter import MVAU_DESIGN_ADAPTER
from finn.dataflow.ops.mvau.assignments import MVAU_DECISION_NODEATTRS
from finn.dataflow.ops.mvau.projection import (
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
from finn.dataflow.ops.mvau.inventory import (
    MVAU_DESIGN_INVENTORY,
    MVAUDataflowOpPaths,
)
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAU_PROBLEM_PROVENANCE,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterKernelPaths,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.region import BeatSequence

#: v6 is the deliberate pre-release cutover from semantic Kernel pools to the
#: closed ``dot_product | batch_interleaved`` DataflowDesign inventory.  Its
#: decision paths and node attributes are new; v5 nodes are rejected rather
#: than interpreted through a migration codec.
MVAU_DATAFLOW_OP_FAMILY_VERSION = "mvau-dataflow-op-v6"


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


def _initializer_excludes_minimum(initializer: object, datatype: object) -> bool | None:
    try:
        return float(np.asarray(initializer).min()) != float(getattr(datatype, "min")())
    except (AttributeError, TypeError, ValueError):
        return None


def _mvau_context(config: object) -> MVAUDataflowBuildContext | None:
    return config if isinstance(config, MVAUDataflowBuildContext) else None


def _mvau_base_config(config: object) -> object:
    context = _mvau_context(config)
    return context.build_config if context is not None else config


def _mvau_target_part(config: object) -> str | None:
    base = _mvau_base_config(config)
    resolver = getattr(base, "_resolve_fpga_part", None)
    if callable(resolver):
        try:
            value = resolver()
        except (KeyError, TypeError, ValueError):
            return None
        return value if isinstance(value, str) and value else None
    value = getattr(base, "fpga_part", None)
    return value if isinstance(value, str) and value else None


def _mvau_target_dsp(config: object) -> DspBlock | None:
    part = _mvau_target_part(config)
    return None if part is None else classify_mvau_dsp_block(part)


def _mvau_memory_capabilities(config: object) -> CyclicTargetMemoryCapabilities | None:
    context = _mvau_context(config)
    explicit = None if context is None else context.supports_initialized_uram
    if explicit is not None:
        return CyclicTargetMemoryCapabilities(explicit)
    target = _mvau_target_dsp(config)
    return None if target is None else CyclicTargetMemoryCapabilities(target is DspBlock.DSP58)


def _mvau_runtime_writable(config: object) -> bool:
    context = _mvau_context(config)
    return False if context is None else context.runtime_writable_weights


def _mvau_runtime_range(config: object) -> bool | None:
    return getattr(config, "runtime_weight_range_contract", None)


def _mvau_external_sequence(config: object) -> BeatSequence | None:
    context = _mvau_context(config)
    return None if context is None else context.external_weight_sequence


def _mvau_analysis_owner(config: object) -> str:
    context = _mvau_context(config)
    return (
        "finn.MinimizeAccumulatorWidth"
        if context is None
        else context.accumulator_type_analysis_owner
    )


def _mvau_clock_period(config: object) -> float | None:
    value = getattr(_mvau_base_config(config), "synth_clk_period_ns", None)
    return None if value is None else float(value)


def _mvau_source_description(
    source_scope_id: str,
    activation_id: str,
    activation_shape: tuple[int, ...],
    weight_id: str,
    threshold_present: bool,
    threshold_id: object,
    threshold_shape: object,
    output_id: str,
    source_nodes: str,
) -> MVAUSourceDescription:
    selected_threshold_id = None
    selected_threshold_shape = None
    if threshold_present:
        if threshold_id is ABSENT or threshold_shape is ABSENT:
            raise ValueError("active threshold facts are unavailable")
        selected_threshold_id = cast(str, threshold_id)
        selected_threshold_shape = cast("tuple[int, ...]", threshold_shape)
    return MVAUSourceDescription(
        source_scope_id,
        activation_id,
        weight_id,
        output_id,
        activation_shape[:-1],
        selected_threshold_id,
        tuple(item for item in source_nodes.split(",") if item),
        selected_threshold_shape,
    )


class MvauDataflowOp(DataflowOp):
    """Logical MVAU source operation backed by the reviewed design inventory."""

    declaration_namespace = "mvau"
    design_adapter = MVAU_DESIGN_ADAPTER

    # Proposed v7 source/build schema. AC3 compiles this in shadow while the
    # v6 hooks below remain authoritative; AC4 switches ``uses_class_authoring``.
    source_scope_id = SourceScope()
    no_activation = Attribute("noActivation", bool, default=True)
    binary_xnor = Attribute("binaryXnorMode", bool, default=False)
    accumulator_element_type = DatatypeAttribute("accDataType", default="INT32")
    activation_bias = Attribute("ActVal", int, default=0)
    source_nodes = Attribute("dataflow_source_nodes", str, default="")

    activation = InputTensor(
        "activation",
        index=0,
        shape=TensorShape(min_rank=2),
        initializer=NoInitializer(),
    )
    weight = InputTensor(
        "weight",
        index=1,
        shape=TensorShape(rank=2),
        initializer=OptionalInitializer(fingerprint=True),
    )
    threshold = InputTensor(
        "threshold",
        index=2,
        when=not_(no_activation),
        shape=TensorShape(rank=2),
        initializer=RequiredInitializer(fingerprint=True),
    )
    output = OutputTensor("output", index=0, shape=TensorShape(min_rank=2))

    @derived(weight.shape, value_type=int)
    def matrix_width(weight_shape: tuple[int, ...]) -> int:
        return weight_shape[0]

    @derived(weight.shape, value_type=int)
    def matrix_height(weight_shape: tuple[int, ...]) -> int:
        return weight_shape[1]

    @derived(activation.shape, value_type=int)
    def repetitions(activation_shape: tuple[int, ...]) -> int:
        return int(np.prod(activation_shape[:-1]))

    @derived(no_activation, binary_xnor, value_type=MVAUComputationProfile)
    def computation_profile(no_activation: bool, binary_xnor: bool) -> MVAUComputationProfile:
        if not no_activation:
            return MVAUComputationProfile.FUSED_THRESHOLD
        if binary_xnor:
            return MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR
        return MVAUComputationProfile.ACCUMULATOR_INTEGER

    @derived(
        source_scope_id,
        activation.tensor_id,
        activation.shape,
        weight.tensor_id,
        threshold.present,
        threshold.tensor_id.allow_absent(),
        threshold.shape.allow_absent(),
        output.tensor_id,
        source_nodes,
        value_type=MVAUSourceDescription,
    )
    def source_description(
        source_scope_id: str,
        activation_id: str,
        activation_shape: tuple[int, ...],
        weight_id: str,
        threshold_present: bool,
        threshold_id: object,
        threshold_shape: object,
        output_id: str,
        source_nodes: str,
    ) -> MVAUSourceDescription:
        return _mvau_source_description(
            source_scope_id,
            activation_id,
            activation_shape,
            weight_id,
            threshold_present,
            threshold_id,
            threshold_shape,
            output_id,
            source_nodes,
        )

    initializer_excludes_minimum = InitializerAnalysis(
        weight,
        bool,
        evaluate=_initializer_excludes_minimum,
    )
    runtime_weight_range_contract = BuildFact(
        "runtime_weight_range_contract",
        bool,
        required=False,
        accessor=_mvau_runtime_range,
    )
    runtime_writable = BuildFact(
        "runtime_writable_weights",
        bool,
        accessor=_mvau_runtime_writable,
        path=CyclicParameterKernelPaths.RUNTIME_WRITABLE,
    )
    external_weight_sequence = BuildFact(
        "external_weight_sequence",
        BeatSequence,
        required=False,
        accessor=_mvau_external_sequence,
    )
    accumulator_type_analysis_owner = BuildFact(
        "accumulator_type_analysis_owner",
        str,
        required=False,
        accessor=_mvau_analysis_owner,
    )
    target_dsp_block = BuildFact(
        "dsp_block",
        DspBlock,
        provenance=Provenance.TARGET,
        required=False,
        accessor=_mvau_target_dsp,
        path=MVAUProblemPaths.TARGET_DSP_BLOCK,
    )
    target_fpga_part = BuildFact(
        "fpga_part",
        str,
        provenance=Provenance.TARGET,
        required=False,
        accessor=_mvau_target_part,
        path=MVAUProblemPaths.TARGET_FPGA_PART,
    )
    target_clock_period_ns = BuildFact(
        "clock_period_ns",
        float,
        provenance=Provenance.TARGET,
        required=False,
        accessor=_mvau_clock_period,
        path=MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS,
    )
    target_memory_capabilities = BuildFact(
        "memory_capabilities",
        CyclicTargetMemoryCapabilities,
        provenance=Provenance.TARGET,
        required=False,
        accessor=_mvau_memory_capabilities,
        path=CyclicParameterKernelPaths.TARGET_MEMORY_CAPABILITIES,
    )

    @derived(
        initializer_excludes_minimum.allow_absent(),
        runtime_weight_range_contract.allow_absent(),
        runtime_writable,
        value_type=bool,
    )
    def effective_narrow_weights(
        initializer_excludes_minimum: object,
        runtime_weight_range_contract: object,
        runtime_writable: bool,
    ) -> bool:
        selected = (
            runtime_weight_range_contract if runtime_writable else initializer_excludes_minimum
        )
        return selected is not ABSENT and bool(selected)

    @constraint(activation.shape, matrix_width)
    def activation_width_supported(shape: tuple[int, ...], matrix_width: int) -> bool:
        return shape[-1] == matrix_width

    @constraint(output.shape, activation.shape, matrix_height)
    def output_shape_supported(
        output_shape: tuple[int, ...],
        activation_shape: tuple[int, ...],
        matrix_height: int,
    ) -> bool:
        return output_shape == (*activation_shape[:-1], matrix_height)

    @constraint(threshold.shape, matrix_height, when=threshold.present)
    def threshold_shape_supported(shape: tuple[int, ...], matrix_height: int) -> bool:
        return shape[0] == matrix_height

    persistence = (
        Persist(design_adapter.refs.design, "dataflow_design"),
        Persist(design_adapter.refs.dot_product_pe, "dataflow_dot_product_pe"),
        Persist(design_adapter.refs.dot_product_simd, "dataflow_dot_product_simd"),
        Persist(design_adapter.refs.batch_interleaved_pe, "dataflow_interleaved_pe"),
        Persist(design_adapter.refs.batch_interleaved_simd, "dataflow_interleaved_simd"),
        Persist(
            design_adapter.refs.batch_interleaved_interleave,
            "dataflow_interleaved_batch",
        ),
        Persist(design_adapter.refs.weight_supply, "dataflow_weight_supply"),
        Persist(design_adapter.refs.compute_pumping, "dataflow_dotp_axi_pumping"),
        Persist(
            design_adapter.refs.ram_style,
            "dataflow_finn_rtl_memstream_ram_style",
        ),
        Persist(
            design_adapter.refs.pumped_memory,
            "dataflow_finn_rtl_memstream_pumping",
        ),
    )

    @classmethod
    def dataflow_family_id(cls) -> str:
        return "finn.dataflow.mvau"

    @classmethod
    def dataflow_family_version(cls) -> str:
        return MVAU_DATAFLOW_OP_FAMILY_VERSION

    @classmethod
    def dataflow_authoring(cls) -> DataflowOpAuthoring:
        return MVAU_DESIGN_INVENTORY.authoring

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
            except (KeyError, TypeError, ValueError):
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
