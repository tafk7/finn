# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical model-aware MVAU custom operation."""

from __future__ import annotations

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
    ClosedDesigns,
    DataflowBuildConfigView,
    DataflowOp,
    DatatypeAttribute,
    InitializerAnalysis,
    InputTensor,
    NoInitializer,
    OptionalInitializer,
    OutputTensor,
    Persist,
    RequiredInitializer,
    SourceScope,
    TensorShape,
    UsesDesign,
    UsesInputSupply,
    constraint,
    derived,
    not_,
)
from finn.dataflow.authoring.op_design import Provenance
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.design import ABSENT
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.ops.mvau.associations import MVAUResolvedDataflowOp
from finn.dataflow.ops.mvau.designs.batch_interleaved import (
    BatchInterleavedDesign,
    BatchInterleavedDesignInputs,
)
from finn.dataflow.ops.mvau.designs.dot_product import (
    DotProductDesign,
    DotProductDesignInputs,
)
from finn.dataflow.ops.mvau.input_supply import declare_mvau_input_supply
from finn.dataflow.ops.mvau.contracts import (
    MVAU_DATAFLOW_OP_FAMILY_ID,
    MVAU_DATAFLOW_OP_FAMILY_VERSION,
)
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAUProblem,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterKernelPaths,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.region import BeatSequence


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
    if part is None or len(part) < 4 or not part.startswith(("xc", "xq")):
        return None
    if part.startswith(("xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm", "xqrvc", "xcv80")):
        return DspBlock.DSP58
    if len(part) > 2 and part[2] == "7":
        return DspBlock.DSP48E1
    return DspBlock.DSP48E2


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
    uses_class_authoring = True
    selection_constraints = "mvau_op_feasibility"
    structural_constraint_set = "mvau_op_structural"
    structural_readiness = "mvau_op_structural"
    artifact_readiness = "artifact_inputs"
    feasibility_constraints = ("mvau_op_feasibility",)

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

    problem = MVAUProblem(
        repetitions=cast("Ref[int]", repetitions),
        matrix_width=cast("Ref[int]", matrix_width),
        matrix_height=cast("Ref[int]", matrix_height),
        activation_element_type=cast("Ref[QONNXDataType]", activation.datatype),
        weight_element_type=cast("Ref[QONNXDataType]", weight.datatype),
        accumulator_element_type=cast("Ref[QONNXDataType]", accumulator_element_type),
        output_element_type=cast("Ref[QONNXDataType]", output.datatype),
        threshold_element_type=cast("Ref[QONNXDataType]", threshold.datatype),
        threshold_initializer_available=cast("Ref[bool]", threshold.initializer_present),
        computation_profile=cast("Ref[MVAUComputationProfile]", computation_profile),
        weight_initializer_available=cast("Ref[bool]", weight.initializer_present),
        weight_initializer_fingerprint=cast("Ref[str]", weight.initializer_fingerprint),
        threshold_initializer_fingerprint=cast("Ref[str]", threshold.initializer_fingerprint),
        source_description=cast("Ref[MVAUSourceDescription]", source_description),
        initializer_excludes_minimum=cast("Ref[bool]", initializer_excludes_minimum),
        runtime_weight_range_contract=cast("Ref[bool]", runtime_weight_range_contract),
        runtime_writable=cast("Ref[bool]", runtime_writable),
        external_weight_sequence=cast("Ref[BeatSequence]", external_weight_sequence),
        accumulator_type_analysis_owner=cast("Ref[str]", accumulator_type_analysis_owner),
        target_dsp_block=cast("Ref[DspBlock]", target_dsp_block),
        target_fpga_part=cast("Ref[str]", target_fpga_part),
        target_clock_period_ns=cast("Ref[float]", target_clock_period_ns),
        target_memory_capabilities=cast(
            "Ref[CyclicTargetMemoryCapabilities]", target_memory_capabilities
        ),
    )
    weight_supply = UsesInputSupply(declare_mvau_input_supply, problem)
    dot_product = UsesDesign(
        DotProductDesign,
        DotProductDesignInputs(
            cast("Ref[int]", repetitions),
            cast("Ref[int]", matrix_width),
            cast("Ref[int]", matrix_height),
            cast("Ref[QONNXDataType]", activation.datatype),
            cast("Ref[QONNXDataType]", weight.datatype),
            cast("Ref[QONNXDataType]", accumulator_element_type),
            cast("Ref[QONNXDataType]", output.datatype),
            cast("Ref[MVAUComputationProfile]", computation_profile),
            cast("Ref[MVAUSourceDescription]", source_description),
            cast("Ref[bool]", effective_narrow_weights),
            cast("Ref[DspBlock]", target_dsp_block),
            cast("Ref[float]", target_clock_period_ns),
            cast("Ref[str]", weight_supply.choice),
        ),
    )
    batch_interleaved = UsesDesign(
        BatchInterleavedDesign,
        BatchInterleavedDesignInputs(
            cast("Ref[int]", repetitions),
            cast("Ref[int]", matrix_width),
            cast("Ref[int]", matrix_height),
            cast("Ref[object]", activation.datatype),
            cast("Ref[object]", weight.datatype),
            cast("Ref[object]", accumulator_element_type),
            cast("Ref[object]", output.datatype),
            cast("Ref[MVAUComputationProfile]", computation_profile),
            cast("Ref[MVAUSourceDescription]", source_description),
            cast("Ref[str]", weight_supply.choice),
        ),
    )
    designs = ClosedDesigns(dot_product, batch_interleaved)

    persistence = (
        Persist(designs.choice, "dataflow_design"),
        Persist(dot_product.pe, "dataflow_dot_product_pe"),
        Persist(dot_product.simd, "dataflow_dot_product_simd"),
        Persist(batch_interleaved.pe, "dataflow_interleaved_pe"),
        Persist(batch_interleaved.simd, "dataflow_interleaved_simd"),
        Persist(
            batch_interleaved.interleave,
            "dataflow_interleaved_batch",
        ),
        Persist(weight_supply.choice, "dataflow_weight_supply"),
        Persist(dot_product.compute.dotp_axi.compute_pumping, "dataflow_dotp_axi_pumping"),
        Persist(
            weight_supply.settings.ram_style,
            "dataflow_finn_rtl_memstream_ram_style",
        ),
        Persist(
            weight_supply.settings.pumped_memory,
            "dataflow_finn_rtl_memstream_pumping",
        ),
    )

    @classmethod
    def dataflow_family_id(cls) -> str:
        return MVAU_DATAFLOW_OP_FAMILY_ID

    @classmethod
    def dataflow_family_version(cls) -> str:
        return MVAU_DATAFLOW_OP_FAMILY_VERSION

    def resolve_dataflow(self, config: DataflowBuildConfigView) -> MVAUResolvedDataflowOp:
        """Return the generic resolved record with MVAU association typing."""

        return cast(MVAUResolvedDataflowOp, super().resolve_dataflow(config))

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
        self.validate_declared_source()


__all__ = [
    "MVAU_DATAFLOW_OP_FAMILY_VERSION",
    "MVAUDataflowBuildContext",
    "MvauDataflowOp",
]
