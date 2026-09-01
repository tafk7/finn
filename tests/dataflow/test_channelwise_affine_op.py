# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from dataflow.channelwise_affine_op import (
    AffineBuildConfig,
    AffineSourceAssociation,
    ChannelwiseAffineDataflowOp,
    DIRECT_DESIGN,
    DIRECT_LANES,
    DIRECT_PIPELINE,
)
from finn.dataflow.authoring import DataflowOpError
from finn.dataflow.design import QualifiedPath
from finn.dataflow.testing import DataflowOpConformanceCase, assert_dataflow_op_conforms


def _model(*, use_bias: bool = True, initialized_scale: bool = True) -> ModelWrapper:
    inputs = ["data", "scale", *(("bias",) if use_bias else ())]
    graph_inputs = [
        helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 4]),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4]),
    ]
    if use_bias:
        graph_inputs.append(helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]))
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "ChannelwiseAffineDataflowOp",
        inputs,
        ["output"],
        name="affine0",
        domain="dataflow.channelwise_affine_op",
        dataflow_scope_id="affine-scope",
        use_bias=int(use_bias),
        saturate=0,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], "affine", graph_inputs, [output]),
            producer_name="channelwise-affine-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("dataflow.channelwise_affine_op", 1),
            ],
        )
    )
    for tensor in ("data", "scale", "bias", "output"):
        if tensor == "bias" and not use_bias:
            continue
        model.set_tensor_datatype(tensor, DataType["INT8"])
    if initialized_scale:
        model.set_initializer("scale", np.arange(1, 5, dtype=np.float32))
    if use_bias:
        model.set_initializer("bias", np.arange(4, dtype=np.float32))
    return model


def _assignments() -> dict[QualifiedPath, object]:
    return {
        DIRECT_DESIGN: "direct",
        DIRECT_LANES: 2,
        DIRECT_PIPELINE: True,
    }


def _change_shape(model: ModelWrapper) -> None:
    model.set_tensor_shape("data", [3, 4])
    model.set_tensor_shape("output", [3, 4])


def test_channelwise_affine_uses_the_complete_generic_op_lifecycle(tmp_path: Path) -> None:
    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=_model(),
            node_name="affine0",
            operation_type=ChannelwiseAffineDataflowOp,
            config=AffineBuildConfig(),
            complete_assignments=_assignments(),
            rejected_assignments={DIRECT_LANES: 3},
            reload_path=tmp_path / "channelwise-affine.onnx",
            stale_config=AffineBuildConfig(synth_clk_period_ns=3.0),
            mutate_graph_problem=_change_shape,
        )
    )
    association = result.original.source_association
    assert isinstance(association, AffineSourceAssociation)
    assert association.source_scope_id == "affine-scope"
    assert association.data == ("data", (2, 4), "compute.data")
    assert association.scale == ("scale", (4,), "compute.scale")
    assert association.bias == ("bias", (4,), "compute.bias")
    realization = ModelWrapper(str(tmp_path / "channelwise-affine.onnx"))
    operation = realization.get_customop_wrapper(realization.graph.node[0])
    assert isinstance(operation, ChannelwiseAffineDataflowOp)
    bound = operation.realize_dataflow(AffineBuildConfig())
    assert tuple(bound.kernels) == ("compute",)
    assert bound.kernels["compute"].parameters == {"LANES": 2, "PIPELINE": True}


def test_optional_bias_and_initializer_fingerprints_are_projected_generically() -> None:
    model = _model(use_bias=False)
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, ChannelwiseAffineDataflowOp)
    problem = operation.problem_instance(AffineBuildConfig())
    assert problem[QualifiedPath("problem.channelwise_affine.bias.present")] is False
    assert QualifiedPath("problem.channelwise_affine.bias.tensor_id") not in problem
    assert problem[QualifiedPath("problem.channelwise_affine.scale.initializer.present")] is True
    fingerprint = problem[QualifiedPath("problem.channelwise_affine.scale.initializer.fingerprint")]
    assert isinstance(fingerprint, str) and len(fingerprint) == 64


def test_required_bias_and_shape_findings_are_structured_and_aggregated() -> None:
    model = _model()
    model.del_initializer("bias")
    model.set_tensor_shape("data", [2, 0])
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, ChannelwiseAffineDataflowOp)
    with pytest.raises(DataflowOpError) as invalid:
        operation.problem_instance(AffineBuildConfig())
    assert {
        "dataflow-source-initializer-required",
        "dataflow-source-shape-invalid",
    } <= {finding.code for finding in invalid.value.findings}


def test_channelwise_operation_contains_no_manual_projection_or_engine_lifecycle() -> None:
    path = Path(__file__).with_name("channelwise_affine_op.py")
    tree = ast.parse(path.read_text())
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"get_tensor_shape", "get_tensor_datatype", "get_initializer"} & calls
    assert not {
        "build_design_space_spec",
        "project_graph_problem",
        "project_build_problem",
        "decision_nodeattrs",
    } & set(ChannelwiseAffineDataflowOp.__dict__)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.startswith("finn.dataflow.ops.mvau") for module in imports)
