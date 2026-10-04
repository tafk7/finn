# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""CreateDataflowPartition over KernelOps: one StreamingDataflowPartition, its body a model
of KernelOps, the graph's other ops left on the host.

test_design's Chain as KernelOps, its choices saved, between two host ops (an
Identity in front, as TFC's flatten, and one behind, as its label select).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.core.onnx_exec import execute_onnx
from finn.custom_op.kernels.partition import partition_root
from finn.kernels.configure import settle
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.kernels import PackagePartition
from kernel_ops.test_partition import configured, kernel_model
from kernels import test_design as chain


def on_the_host(model: ModelWrapper, *, between: bool = False) -> ModelWrapper:
    """The Chain behind an Identity ``flatten`` and in front of an Identity ``select``;
    ``between``: another Identity between its first two nodes."""
    first = model.graph.node[0]
    first.input[0] = "x_flat"
    nodes = list(model.graph.node)
    if between:
        hidden = nodes[1].input[0]
        nodes[1].input[0] = "hidden_copy"
        nodes.insert(1, helper.make_node("Identity", [hidden], ["hidden_copy"], name="copy"))
        model.set_tensor_shape("hidden_copy", model.get_tensor_shape(hidden))
        model.set_tensor_datatype("hidden_copy", model.get_tensor_datatype(hidden))
    nodes[-1].output[0] = "y_kernel"
    nodes.insert(0, helper.make_node("Identity", ["x"], ["x_flat"], name="flatten"))
    nodes.append(helper.make_node("Identity", ["y_kernel"], ["y"], name="select"))
    while model.graph.node:
        model.graph.node.pop()
    model.graph.node.extend(nodes)
    for name, source in (("x_flat", "x"), ("y_kernel", "y")):
        model.set_tensor_shape(name, model.get_tensor_shape(source))
        model.set_tensor_datatype(name, model.get_tensor_datatype(source))
    return model


def partitioned(tmp_path: object) -> tuple[ModelWrapper, ModelWrapper]:
    source = kernel_model()
    configured(source)
    source = on_the_host(source)
    parent = source.transform(CreateDataflowPartition(partition_model_dir=str(tmp_path)))
    return source, parent


def test_the_kernel_ops_become_one_partition_between_the_host_ops(tmp_path: object) -> None:
    source, parent = partitioned(tmp_path)
    assert [(node.op_type, node.name) for node in parent.graph.node] == [
        ("Identity", "flatten"),
        ("StreamingDataflowPartition", "GenericPartition_kernels"),
        ("Identity", "select"),
    ]
    sdp = getCustomOp(parent.graph.node[1])
    # Placement is the partition's, unset: KernelOps carry none.
    assert (sdp.get_nodeattr("slr"), sdp.get_nodeattr("mem_port")) == (-1, "")
    body = ModelWrapper(sdp.get_nodeattr("model"))
    assert [node.name for node in body.graph.node] == ["first", "activate", "second"]
    assert [item.name for item in body.graph.input] == ["x_flat"]
    assert [item.name for item in body.graph.output] == ["y_kernel"]
    # The body computes the Chain, through the partition node.
    x = np.array(chain.X, dtype=np.float32)
    assert np.array_equal(execute_onnx(parent, {"x": x})["y"], execute_onnx(source, {"x": x})["y"])


def test_the_body_is_the_partition_packaging_takes(tmp_path: object) -> None:
    source, parent = partitioned(tmp_path)
    body = ModelWrapper(getCustomOp(parent.graph.node[1]).get_nodeattr("model"))
    module = PackagePartition("xczu3eg-sbva484-1-e", 5.0, parent.graph.node[1].name).module(body)
    # The same module as the root of the KernelOps where they stood.
    kernel_ops = [node for node in source.graph.node if node.domain == "finn.custom_op.kernels"]
    reference = settle(partition_root(source, kernel_ops).point).point.module
    assert (module.fragment, module.pins) == (reference.fragment, reference.pins)
    assert [port.name for port in module.pins.ports] == [
        "ap_clk",
        "ap_rst_n",
        "s_axis_0",
        "m_axis_0",
    ]


def test_a_host_op_between_kernel_ops_refuses(tmp_path: object) -> None:
    source = kernel_model()
    configured(source)
    source = on_the_host(source, between=True)
    with pytest.raises(AssertionError, match="partition depends on itself"):
        source.transform(CreateDataflowPartition(partition_model_dir=str(tmp_path)))


def test_a_graph_without_kernel_ops_is_left_alone(tmp_path: object) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "host", [x], [y])
    model = ModelWrapper(helper.make_model(graph))
    parent = model.transform(CreateDataflowPartition(partition_model_dir=str(tmp_path)))
    assert [node.op_type for node in parent.graph.node] == ["Identity"]
