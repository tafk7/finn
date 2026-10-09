# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's partitioning (``partition_kernel_ops``): its KernelOps one
StreamingDataflowPartition of the partition domain, its body a model of KernelOps, the
graph's other ops left on the host.

The Chain (``kernels.chain``) as KernelOps, its kernels' choices saved on its nodes,
between two host ops (an Identity in front, as TFC's flatten, and one behind, as its
label select). Its channels' choices are the partition body's: a graph with nodes on the
host states none, so they are stated on the body the cut makes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from kernels import chain
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.custom_op.partition as partition_domain
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.kernels.base import CHANNEL, KernelOpError, channel_choices
from finn.custom_op.kernels.shell import configured_root, member, save_channels, shell_root
from finn.kernels.configure import commit
from finn.transformation.kernels.cut import partition_kernel_ops
from kernel_ops.models import configure_partition, kernel_model


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


#: The Chain's boundary tensors and their names behind and in front of the host ops.
BOUNDARY = {"x": "x_flat", "y": "y_kernel"}


def moved_out(model: ModelWrapper) -> dict[str, dict[str, object]]:
    """The channel choices ``model`` states, by tensor (a boundary tensor's under the name
    ``on_the_host`` gives it), cleared from ``model``: a graph with nodes on the host
    states none."""
    stated = model.tensors_stating(CHANNEL)
    found = {BOUNDARY.get(tensor, tensor): channel_choices(model, tensor) for tensor in stated}
    for tensor in stated:
        model.clear(CHANNEL, tensor=tensor)
    return found


def partitioned(tmp_path: Path) -> tuple[ModelWrapper, ModelWrapper, dict[str, dict[str, object]]]:
    """The configured Chain on the host and its parent graph, its channels' choices stated
    on the body the cut made (and returned, by tensor)."""
    source = kernel_model()
    configure_partition(source)
    channels = moved_out(source)
    source = on_the_host(source)
    parent = partition_kernel_ops(source, tmp_path)
    body_file = getCustomOp(parent.graph.node[1]).get_nodeattr("model")
    body = ModelWrapper(body_file)
    save_channels(body, channels)
    body.save(body_file)
    return source, parent, channels


def test_the_kernel_ops_become_one_partition_between_the_host_ops(tmp_path: Path) -> None:
    source, parent, _ = partitioned(tmp_path)
    assert [(node.op_type, node.name) for node in parent.graph.node] == [
        ("Identity", "flatten"),
        ("StreamingDataflowPartition", "GenericPartition_kernels"),
        ("Identity", "select"),
    ]
    sdp = getCustomOp(parent.graph.node[1])
    # The node states its body and nothing else: no placement, no estimates.
    assert [item.name for item in parent.graph.node[1].attribute] == ["model"]
    body = ModelWrapper(sdp.get_nodeattr("model"))
    assert [node.name for node in body.graph.node] == ["first", "activate", "second"]
    assert [item.name for item in body.graph.input] == ["x_flat"]
    assert [item.name for item in body.graph.output] == ["y_kernel"]
    # The body computes the Chain, through the partition node.
    x = np.array(chain.X, dtype=np.float32)
    assert np.array_equal(execute_onnx(parent, {"x": x})["y"], execute_onnx(source, {"x": x})["y"])


def test_the_parent_graph_imports_its_partition_nodes_domain(tmp_path: Path) -> None:
    """The parent graph states the domain of its StreamingDataflowPartition, so reading
    the node warns of no fallback version (TFC's one build-log warning, observation 37).
    The domain is the kernel path's own, the package that registers the node."""
    _, parent, _ = partitioned(tmp_path)
    imported = {opset.domain: opset.version for opset in parent.model.opset_import}
    assert imported["finn.custom_op.partition"] == 1
    assert parent.graph.node[1].domain == partition_domain.__name__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (node,) = parent.get_nodes_by_op_type("StreamingDataflowPartition")
        getCustomOp(node).get_nodeattr("model")


def test_the_body_is_the_partition_packaging_takes(tmp_path: Path) -> None:
    source, parent, channels = partitioned(tmp_path)
    body = ModelWrapper(getCustomOp(parent.graph.node[1]).get_nodeattr("model"))
    module = configured_root(body, parent.graph.node[1].name)[0].module
    # The same module as the root of the KernelOps where they stood, with the channel
    # choices the body states.
    kernel_ops = [node for node in source.graph.node if node.domain == "finn.custom_op.kernels"]
    stated = {
        f"{member(tensor)}.{key}": value
        for tensor, held in channels.items()
        for key, value in held.items()
    }
    reference = commit(shell_root(source, kernel_ops).point, stated).module
    assert (module.fragment, module.abi) == (reference.fragment, reference.abi)
    assert [port.name for port in module.abi.pins] == [
        "ap_clk",
        "ap_rst_n",
        "s_axis_0",
        "m_axis_0",
    ]


def test_a_host_op_between_kernel_ops_refuses(tmp_path: Path) -> None:
    source = kernel_model()
    configure_partition(source)
    moved_out(source)
    source = on_the_host(source, between=True)
    with pytest.raises(KernelOpError, match="1 host nodes sit between KernelOps") as refused:
        partition_kernel_ops(source, tmp_path)
    assert str(refused.value).endswith(": copy")


def test_a_graph_without_kernel_ops_is_left_alone(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "host", [x], [y])
    model = ModelWrapper(helper.make_model(graph))
    parent = partition_kernel_ops(model, tmp_path)
    assert [node.op_type for node in parent.graph.node] == ["Identity"]
