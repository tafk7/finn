# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D7 gate: inference admission comes from the MVAU design inventory."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.core.onnx_exec import execute_onnx  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import Decided, Engine
from finn.dataflow.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau.input_supply import EXTERNAL_SUPPLY
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
    with_weight_initializer: bool = True,
    rows: int = 2,
    out_scale: float | None = None,
    out_bias: float | None = None,
    out_dtype: str = "UINT2",
) -> ModelWrapper:
    width, height = 4, 4
    nodes = [helper.make_node("MatMul", ["activation", "weights"], ["accumulator"], name="matmul0")]
    outputs = [helper.make_tensor_value_info("accumulator", TensorProto.FLOAT, [rows, height])]
    initializers: list[tuple[str, np.ndarray]] = (
        [("weights", np.ones((width, height), dtype=np.float32))] if with_weight_initializer else []
    )
    value_info: list[object] = []
    if fused:
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                ["accumulator", "thresholds"],
                ["output"],
                name="threshold0",
                domain="qonnx.custom_op.general",
                out_dtype=out_dtype,
                **({} if out_scale is None else {"out_scale": out_scale}),
                **({} if out_bias is None else {"out_bias": out_bias}),
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
        model.set_tensor_datatype("output", DataType[out_dtype])
    for name, values in initializers:
        model.set_initializer(name, values)
    return model


def _xnor_model() -> ModelWrapper:
    """A real XNOR-popcount source, not a MatMul annotated as bipolar."""

    width = height = 4
    rows = 2
    node = helper.make_node(
        "XnorPopcountMatMul",
        ["activation", "weights"],
        ["output"],
        name="matmul0",
        domain="qonnx.custom_op.general",
    )
    graph = helper.make_graph(
        [node],
        "xnor-source",
        [
            helper.make_tensor_value_info("activation", TensorProto.FLOAT, [rows, width]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [width, height]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, height])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-inference-test"))
    model.set_tensor_datatype("activation", DataType["BIPOLAR"])
    model.set_tensor_datatype("weights", DataType["BIPOLAR"])
    model.set_tensor_datatype("output", DataType["INT32"])
    model.set_initializer("weights", np.ones((width, height), dtype=np.float32))
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


def test_a_constant_activation_is_not_a_source_form() -> None:
    """A constant left operand is a foldable product, not a matrix-vector unit."""

    model = _model()
    model.set_initializer("activation", np.ones((2, 4), dtype=np.float32))
    assert recognize_mvau_candidates(model) == ()


def test_a_dynamic_weight_matmul_is_still_a_source_form() -> None:
    """How the weight is supplied is the pool's question, not the matcher's."""

    model = _model(with_weight_initializer=False)
    candidates = recognize_mvau_candidates(model)
    assert len(candidates) == 1
    assert candidates[0].source_node_names == ("matmul0",)


def test_a_float_matmul_is_recognized_by_the_semantic_only_design() -> None:
    """Logical admission does not falsely claim a physical implementation."""

    model = _model(float_weights=True)
    assert len(recognize_mvau_candidates(model)) == 1
    lowered, transform = _lower(model)
    assert transform.report.lowered
    assert transform.report.refused == ()
    assert mvau_source_admission(lowered, lowered.graph.node[0].name, _context()) == (
        BatchInterleavedDesign.id,
    )


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
    before = model.model.SerializeToString(deterministic=True)
    lowered, transform = _lower(model)
    assert transform.report.lowered == ()
    assert transform.report.refused == (("matmul0", "threshold0"),)
    assert lowered.model.SerializeToString(deterministic=True) == before


def test_lowering_is_idempotent() -> None:
    lowered, _ = _lower(_model())
    once = lowered.model.SerializeToString(deterministic=True)
    again, transform = _lower(lowered)
    assert transform.report.lowered == ()
    assert again.model.SerializeToString(deterministic=True) == once


def test_a_refused_candidate_leaves_the_graph_byte_identical() -> None:
    model = _model(fused=True)
    before = model.model.SerializeToString(deterministic=True)
    lowered, transform = _lower(model)
    assert transform.report.lowered == ()
    assert transform.report.refused == (("matmul0", "threshold0"),)
    assert lowered.model.SerializeToString(deterministic=True) == before


def test_no_design_choice_is_persisted_during_lowering() -> None:
    lowered, _ = _lower(_model())
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    codecs = {codec.attribute_name for codec in MvauDataflowOp.decision_nodeattrs().values()}
    present = {item.name for item in operation.onnx_node.attribute}
    assert not (present & codecs)


def test_provenance_records_every_consumed_source_node() -> None:
    lowered, _ = _lower(_model())
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    assert source_nodes_of(operation) == ("matmul0",)
    problem = operation.problem_instance(_context())
    description = problem[MVAUDataflowOpPaths.SOURCE_DESCRIPTION]
    assert getattr(description, "fused_source_node_ids") == ("matmul0",)


# -- admission comes from the design inventory -------------------------------


def test_admission_is_the_existential_union_over_the_design_inventory() -> None:
    lowered, _ = _lower(_model())
    admitted = mvau_source_admission(lowered, lowered.graph.node[0].name, _context())
    assert admitted == (DotProductDesign.id, BatchInterleavedDesign.id)


def test_fused_threshold_is_deferred_from_the_new_design_inventory() -> None:
    _, transform = _lower(_model(fused=True))
    assert transform.report.lowered == ()
    assert {finding.code for finding in transform.report.findings} == {
        "mvau-inference-no-admitting-design"
    }


def test_semantic_only_admission_does_not_make_a_build_point_feasible() -> None:
    lowered, _ = _lower(_model(float_weights=True))
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    engine = Engine()
    point = engine.start(operation.validated_design_space(), operation.problem_instance(_context()))
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    point = engine.commit_assignments(
        point,
        {
            MVAU_DESIGN_INVENTORY.inventory.design_path: BatchInterleavedDesign.id,
            MVAU_DESIGN_INVENTORY.batch_interleaved.pe.path: 2,
            MVAU_DESIGN_INVENTORY.batch_interleaved.simd.path: 2,
            MVAU_DESIGN_INVENTORY.batch_interleaved.interleave.path: 2,
            MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
        },
    ).point
    assert engine.evaluate_constraint_set(point, "mvau_op_feasibility").verdict is not True


def test_admission_does_not_depend_on_the_fpga_target() -> None:
    """Target feasibility is a selection-time query, not an inference one."""

    lowered, _ = _lower(_model())
    admitted = {
        part: mvau_source_admission(
            lowered,
            lowered.graph.node[0].name,
            MVAUDataflowBuildContext(_BuildConfig(fpga_part=part)),
        )
        for part in (PART, "xcvc1902-vsva2197-2MP-e-S")
    }
    assert len(set(admitted.values())) == 1


def test_an_interleave_that_can_never_be_chosen_is_not_admitted() -> None:
    """One repetition leaves the interleave domain empty, so the Kernel is out."""

    lowered, _ = _lower(_model(rows=1))
    admitted = mvau_source_admission(lowered, lowered.graph.node[0].name, _context())
    assert admitted == (DotProductDesign.id,)


def test_source_admission_never_reads_a_kernel_local_decision() -> None:
    physical = {
        MVAU_DESIGN_INVENTORY.compute_pumping.path,
        MVAU_DESIGN_INVENTORY.input_supply.settings.ram_style.path,
        MVAU_DESIGN_INVENTORY.input_supply.settings.pumped_memory.path,
    }
    for semantics in (
        MVAU_DESIGN_INVENTORY.dot_product,
        MVAU_DESIGN_INVENTORY.batch_interleaved,
    ):
        by_path = {item.path: item for item in semantics.spec.constraints}
        for reference in semantics.source_constraints:
            constraint = by_path[reference.path]
            dependencies = list(constraint.evaluator.dependencies)
            if constraint.applies_if is not None:
                dependencies.extend(constraint.applies_if.dependencies)
            assert not ({item.path for item in dependencies} & physical), reference.path


def test_the_transform_declares_no_datatype_or_target_switch() -> None:
    source = inspect.getsource(infer_mvau_dataflow)
    for forbidden in ("bit_width", "DSP", "INT8", "resType", "mem_mode", "hls", "rtl"):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("attribute", ["noActivation", "binaryXnorMode", "accDataType"])
def test_source_semantics_are_carried_onto_the_logical_node(attribute: str) -> None:
    lowered, _ = _lower(_model())
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    assert operation.get_nodeattr(attribute) is not None
    assert operation.get_nodeattr(SOURCE_NODES_ATTR)


@pytest.mark.parametrize("bias", [0, -1, -2, 3])
def test_a_representable_integer_bias_is_carried_and_preserved(bias: int) -> None:
    """Fused activation remains recognized but is deferred by the inventory."""

    model = _model(fused=True, out_bias=float(bias), out_dtype="INT4")
    lowered, transform = _lower(model)
    assert transform.report.lowered == ()
    assert [node.op_type for node in lowered.graph.node] == ["MatMul", "MultiThreshold"]


def test_a_bipolar_scale_and_bias_pair_is_carried_and_preserved() -> None:
    model = _model(fused=True, out_scale=2.0, out_bias=-1.0, out_dtype="BIPOLAR")
    lowered, transform = _lower(model)
    assert transform.report.lowered == ()
    assert [node.op_type for node in lowered.graph.node] == ["MatMul", "MultiThreshold"]


@pytest.mark.parametrize(
    "out_scale,out_bias,out_dtype",
    [
        (2.0, 0.0, "INT4"),  # a scale the operation cannot apply
        (0.5, 0.0, "INT4"),
        (1.0, -0.5, "INT4"),  # a bias ActVal cannot hold
        (1.0, 0.0, "BIPOLAR"),  # bipolar without its scale/bias pair
    ],
)
def test_an_unrepresentable_activation_is_not_fused(
    out_scale: float, out_bias: float, out_dtype: str
) -> None:
    """The matrix product is still a source form; only the fusion is refused."""

    model = _model(fused=True, out_scale=out_scale, out_bias=out_bias, out_dtype=out_dtype)
    candidates = recognize_mvau_candidates(model)
    assert len(candidates) == 1
    assert candidates[0].source_node_names == ("matmul0",)


def test_an_unrepresentable_activation_leaves_execution_unchanged() -> None:
    model = _model(fused=True, out_bias=-1.0, out_scale=2.0, out_dtype="INT4")
    activation = np.arange(8, dtype=np.float32).reshape(2, 4)
    before = execute_onnx(model, {"activation": activation})["output"]
    lowered, _ = _lower(model)
    assert [node.op_type for node in lowered.graph.node] == [
        "MvauDataflowOp",
        "MultiThreshold",
    ]
    after = execute_onnx(lowered, {"activation": activation})["output"]
    np.testing.assert_array_equal(after, before)


def test_an_xnor_popcount_source_is_recognized_and_marked() -> None:
    model = _xnor_model()
    candidates = recognize_mvau_candidates(model)
    assert len(candidates) == 1
    assert candidates[0].xnor_popcount is True
    lowered, transform = _lower(model)
    assert transform.report.lowered == ()
    assert transform.report.refused == (("matmul0",),)


def test_a_lowered_node_still_resolves_once_kernels_are_selected() -> None:
    lowered, _ = _lower(_model())
    operation = lowered.get_customop_wrapper(lowered.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    context = _context()
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    operation.commit_dataflow_assignments(
        context,
        {
            MVAU_DESIGN_INVENTORY.inventory.design_path: DotProductDesign.id,
            MVAU_DESIGN_INVENTORY.dot_product.pe.path: 2,
            MVAU_DESIGN_INVENTORY.dot_product.simd.path: 2,
            MVAU_DESIGN_INVENTORY.compute_pumping.path: False,
            MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
        },
    )
    resolved = operation.resolve_dataflow(context)
    result = resolved.engine.query_property(resolved.point, MVAUDataflowOpPaths.RESULT)
    assert isinstance(result, Decided)
