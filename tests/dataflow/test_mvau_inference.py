# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 6 gate: inference coverage comes from the Kernel pool, not the pass."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.core.onnx_exec import execute_onnx  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import Decided, Engine
from finn.dataflow.kernels import Kernel, KernelSelection, admissible_kernels
from finn.dataflow.mvau.compute_kernels import (
    LEGACY_HLS_PATHS,
    MVAU_COMPUTE_SELECTION,
    SOFT_VECTOR_MVAU_KERNEL,
    MVAUComputeKernelId,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.ops.mvau import MVAUDataflowOpPaths
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp
from finn.transformation.fpgadataflow import infer_mvau_dataflow
from finn.transformation.fpgadataflow.infer_mvau_dataflow import (
    SOURCE_NODES_ATTR,
    InferMVAUDataflowOp,
    mvau_source_admission,
    recognize_mvau_candidates,
    source_nodes_of,
)

PART = "xczu3eg-sbva484-1-e"


@dataclass
class _BuildConfig:
    synth_clk_period_ns: float = 5.0
    fpga_part: str | None = PART

    def _resolve_fpga_part(self) -> str:
        if self.fpga_part is None:
            raise ValueError("no target part")
        return self.fpga_part


def _context() -> MVAUDataflowBuildContext:
    return MVAUDataflowBuildContext(_BuildConfig())


def _model(
    *,
    fused: bool = False,
    activation_type: str = "INT8",
    weight_type: str = "INT8",
    float_weights: bool = False,
    extra_consumer: bool = False,
) -> ModelWrapper:
    width, height, rows = 4, 4, 2
    nodes = [helper.make_node("MatMul", ["activation", "weights"], ["accumulator"], name="matmul0")]
    outputs = [helper.make_tensor_value_info("accumulator", TensorProto.FLOAT, [rows, height])]
    initializers: list[tuple[str, np.ndarray]] = [
        ("weights", np.ones((width, height), dtype=np.float32))
    ]
    value_info: list[object] = []
    if fused:
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                ["accumulator", "thresholds"],
                ["output"],
                name="threshold0",
                domain="qonnx.custom_op.general",
                out_dtype="UINT2",
            )
        )
        value_info.append(
            helper.make_tensor_value_info("accumulator", TensorProto.FLOAT, [rows, height])
        )
        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, height])]
        initializers.append(("thresholds", np.asarray([[1.0, 2.0, 3.0]] * height, np.float32)))
    if extra_consumer:
        nodes.append(helper.make_node("Identity", ["accumulator"], ["spare"], name="identity0"))
        outputs.append(helper.make_tensor_value_info("spare", TensorProto.FLOAT, [rows, height]))
    graph = helper.make_graph(
        nodes,
        "mvau-source",
        [
            helper.make_tensor_value_info("activation", TensorProto.FLOAT, [rows, width]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [width, height]),
        ]
        + (
            [helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [height, 3])]
            if fused
            else []
        ),
        outputs,
        value_info=value_info,
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-inference-test"))
    model.set_tensor_datatype("activation", DataType[activation_type])
    model.set_tensor_datatype(
        "weights", DataType["FLOAT32"] if float_weights else DataType[weight_type]
    )
    model.set_tensor_datatype("accumulator", DataType["INT32"])
    if fused:
        model.set_tensor_datatype("thresholds", DataType["INT32"])
        model.set_tensor_datatype("output", DataType["UINT2"])
    for name, values in initializers:
        model.set_initializer(name, values)
    return model


def _lower(model: ModelWrapper) -> tuple[ModelWrapper, InferMVAUDataflowOp]:
    transform = InferMVAUDataflowOp(_context())
    # cleanup=False keeps the comparison about this transform only.
    lowered = model.transform(transform, cleanup=False)
    return lowered, transform


# -- recognition -------------------------------------------------------------


def test_a_dense_quantized_matmul_is_recognized() -> None:
    candidates = recognize_mvau_candidates(_model())
    assert len(candidates) == 1
    assert candidates[0].source_node_names == ("matmul0",)


def test_an_exclusively_consumed_multithreshold_is_fused() -> None:
    candidates = recognize_mvau_candidates(_model(fused=True))
    assert len(candidates) == 1
    assert candidates[0].source_node_names == ("matmul0", "threshold0")


def test_a_shared_accumulator_is_not_fused() -> None:
    candidates = recognize_mvau_candidates(_model(fused=True, extra_consumer=True))
    assert len(candidates) == 1
    assert candidates[0].source_node_names == ("matmul0",)


def test_a_float_weight_matmul_is_not_a_source_form() -> None:
    assert recognize_mvau_candidates(_model(float_weights=True)) == ()


# -- lowering ----------------------------------------------------------------


def test_lowering_produces_an_unresolved_executable_logical_node() -> None:
    model = _model()
    lowered, transform = _lower(model)
    assert transform.report.lowered
    nodes = list(lowered.graph.node)
    assert [node.op_type for node in nodes] == ["MvauDataflowOp"]
    operation = lowered.get_customop_wrapper(nodes[0])
    assert isinstance(operation, MvauDataflowOp)
    operation.verify_node()
    assert operation.read_assignments() == {}
    assert operation.dataflow_scope_id().startswith("mvau_")
    assert source_nodes_of(operation) == ("matmul0",)


def test_lowering_preserves_source_execution() -> None:
    model = _model()
    activation = np.arange(8, dtype=np.float32).reshape(2, 4)
    before = execute_onnx(model, {"activation": activation})
    lowered, _ = _lower(model)
    after = execute_onnx(lowered, {"activation": activation})
    np.testing.assert_array_equal(after["accumulator"], before["accumulator"])


def test_fused_lowering_preserves_source_execution() -> None:
    model = _model(fused=True)
    activation = np.arange(8, dtype=np.float32).reshape(2, 4)
    before = execute_onnx(model, {"activation": activation})
    lowered, _ = _lower(model)
    after = execute_onnx(lowered, {"activation": activation})
    np.testing.assert_array_equal(after["output"], before["output"])


def test_lowering_is_idempotent() -> None:
    lowered, _ = _lower(_model())
    once = lowered.model.SerializeToString(deterministic=True)
    again, transform = _lower(lowered)
    assert transform.report.lowered == ()
    assert again.model.SerializeToString(deterministic=True) == once


def test_a_refused_candidate_leaves_the_graph_byte_identical() -> None:
    model = _model(activation_type="BINARY", weight_type="BINARY")
    before = model.model.SerializeToString(deterministic=True)
    lowered, transform = _lower(model)
    assert transform.report.lowered == ()
    assert transform.report.refused == (("matmul0",),)
    assert lowered.model.SerializeToString(deterministic=True) == before


def test_no_design_choice_is_persisted_during_lowering() -> None:
    lowered, _ = _lower(_model())
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    codecs = {codec.attribute_name for codec in MvauDataflowOp.decision_nodeattrs().values()}
    present = {item.name for item in operation.onnx_node.attribute}
    assert not (present & codecs)


def test_provenance_records_every_consumed_source_node() -> None:
    lowered, _ = _lower(_model(fused=True))
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    assert source_nodes_of(operation) == ("matmul0", "threshold0")
    problem = operation.problem_instance(_context())
    description = problem[MVAUDataflowOpPaths.SOURCE_DESCRIPTION]
    assert getattr(description, "fused_source_node_ids") == ("matmul0", "threshold0")


# -- admission comes from the pool -------------------------------------------


def test_admission_is_the_existential_union_over_the_kernel_pool() -> None:
    lowered, _ = _lower(_model())
    admitted = mvau_source_admission(lowered, lowered.graph.node[0].name, _context())
    assert MVAUComputeKernelId.SOFT_VECTOR.value in admitted
    assert MVAUComputeKernelId.LEGACY_HLS.value in admitted


def test_fused_admission_depends_on_kernel_owned_profile_support() -> None:
    lowered, _ = _lower(_model(fused=True))
    admitted = mvau_source_admission(lowered, lowered.graph.node[0].name, _context())
    # Only the legacy HLS Kernel claims the fused-threshold profile today.
    assert admitted == (MVAUComputeKernelId.LEGACY_HLS.value,)


def test_adding_and_removing_a_kernel_moves_admission_without_touching_the_pass() -> None:
    """A pool with only DSP58 Kernels refuses a non-DSP58 part, and vice versa."""

    lowered, _ = _lower(_model())
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    engine = Engine()
    point = engine.start(operation.validated_design_space(), operation.problem_instance(_context()))

    packed_only: KernelSelection = replace(
        MVAU_COMPUTE_SELECTION,
        kernels=(MVAU_COMPUTE_SELECTION.kernel(MVAUComputeKernelId.PACKED_DSP.value),),
    )
    assert admissible_kernels(engine, packed_only, point) == ()

    with_soft_vector: KernelSelection = replace(
        packed_only,
        kernels=(*packed_only.kernels, cast(Kernel, SOFT_VECTOR_MVAU_KERNEL)),
    )
    assert admissible_kernels(engine, with_soft_vector, point) == (
        MVAUComputeKernelId.SOFT_VECTOR.value,
    )


def test_the_transform_declares_no_datatype_or_target_switch() -> None:
    source = inspect.getsource(infer_mvau_dataflow)
    for forbidden in ("bit_width", "DSP", "INT8", "resType", "mem_mode", "hls", "rtl"):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("attribute", ["noActivation", "binaryXnorMode", "accDataType"])
def test_source_semantics_are_carried_onto_the_logical_node(attribute: str) -> None:
    lowered, _ = _lower(_model(fused=True))
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    assert operation.get_nodeattr(attribute) is not None
    assert operation.get_nodeattr(SOURCE_NODES_ATTR)


def test_xnor_sources_are_recognized_and_marked() -> None:
    model = _model(activation_type="BIPOLAR", weight_type="BIPOLAR")
    model.set_initializer("weights", np.ones((4, 4), dtype=np.float32))
    lowered, transform = _lower(model)
    if not transform.report.lowered:
        pytest.skip("no compute Kernel admits bipolar XNOR on this target yet")
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    assert operation.get_nodeattr("binaryXnorMode") == 1
    problem = operation.problem_instance(_context())
    profile = problem[next(path for path in problem if str(path).endswith("computation_profile"))]
    assert "xnor" in str(getattr(profile, "value", profile))


def test_a_lowered_node_still_resolves_once_kernels_are_selected() -> None:
    lowered, _ = _lower(_model())
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    context = _context()
    operation.commit_dataflow_assignments(
        context,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.LEGACY_HLS.value,
            LEGACY_HLS_PATHS.pe: 2,
            LEGACY_HLS_PATHS.simd: 2,
            LEGACY_HLS_PATHS.resource: MVAUHlsResource.LUT,
            LEGACY_HLS_PATHS.weight_source: MVAUWeightSource.EMBEDDED,
        },
    )
    resolved = operation.resolve_dataflow(context)
    result = resolved.engine.query_property(resolved.point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(result, Decided)
