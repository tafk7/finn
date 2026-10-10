# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``finn.util.graph.between``: the nodes outside a set on a path that leaves it and
re-enters it, by the graph's tensors, in graph order. Its two readers' cases: the
KernelOps' (``test_outcomes``) and the checkpoint's prediction (``test_preparation``).
``upstream``: the nodes outside a set it depends on (nested patterns,
``test_nested_patterns``)."""

from __future__ import annotations

from onnx import NodeProto, TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.util.graph import between, upstream


def graph(nodes: list[NodeProto], outputs: list[str]) -> ModelWrapper:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    ys = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4]) for name in outputs]
    return ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "between", [x], ys)))


def matmul(node: NodeProto) -> bool:
    return node.op_type == "MatMul"


def names(model: ModelWrapper) -> list[str]:
    return [node.name for node in between(model, matmul)]


def test_a_node_that_leaves_and_reenters_is_between_one_that_only_leaves_is_not() -> None:
    """``flip`` leaves ``a`` and re-enters at ``b``; ``negate``, between them in node
    order, only leaves (to an output), and ``scale`` only enters."""
    model = graph(
        [
            helper.make_node("Relu", ["x"], ["r"], name="scale"),
            helper.make_node("MatMul", ["r", "w"], ["h"], name="a"),
            helper.make_node("Neg", ["h"], ["z"], name="negate"),
            helper.make_node("Transpose", ["h"], ["t"], name="flip"),
            helper.make_node("MatMul", ["t", "w"], ["y"], name="b"),
        ],
        ["y", "z"],
    )
    assert names(model) == ["flip"]


def test_every_node_on_a_path_is_named_in_graph_order() -> None:
    model = graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
            helper.make_node("Relu", ["h"], ["r"], name="first"),
            helper.make_node("MatMul", ["h", "w"], ["k"], name="b"),
            helper.make_node("Neg", ["r"], ["n"], name="second"),
            helper.make_node("Add", ["n", "k"], ["s"], name="join"),
            helper.make_node("MatMul", ["s", "w"], ["y"], name="c"),
        ],
        ["y"],
    )
    assert names(model) == ["first", "second", "join"]


def test_nodes_at_the_edges_or_with_none_inside_are_not_between() -> None:
    edges = graph(
        [
            helper.make_node("Relu", ["x"], ["r"], name="before"),
            helper.make_node("MatMul", ["r", "w"], ["h"], name="a"),
            helper.make_node("MatMul", ["h", "w"], ["m"], name="b"),
            helper.make_node("Neg", ["m"], ["y"], name="after"),
        ],
        ["y"],
    )
    assert names(edges) == []
    host = graph([helper.make_node("Relu", ["x"], ["y"], name="only")], ["y"])
    assert names(host) == []


def test_the_nodes_themselves_are_returned_named_or_not() -> None:
    """The walk follows nodes, not their names: of two unnamed nodes, the one on the
    path is returned and the one after the set is not."""
    model = graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
            helper.make_node("Relu", ["h"], ["r"]),
            helper.make_node("MatMul", ["r", "w"], ["m"], name="b"),
            helper.make_node("Neg", ["m"], ["y"]),
        ],
        ["y"],
    )
    found = between(model, matmul)
    assert [(node.op_type, list(node.output)) for node in found] == [("Relu", ["r"])]
    assert found[0] is model.graph.node[1]


def test_upstream_is_every_node_outside_the_set_it_depends_on_in_graph_order() -> None:
    """``scale`` feeds ``a`` and ``bias`` feeds ``b`` through ``shift``; ``after`` and
    ``aside`` feed neither."""
    model = graph(
        [
            helper.make_node("Relu", ["x"], ["r"], name="scale"),
            helper.make_node("MatMul", ["r", "w"], ["h"], name="a"),
            helper.make_node("Neg", ["x"], ["n"], name="bias"),
            helper.make_node("Neg", ["x"], ["z"], name="aside"),
            helper.make_node("Relu", ["n"], ["s"], name="shift"),
            helper.make_node("MatMul", ["h", "s"], ["y"], name="b"),
            helper.make_node("Neg", ["y"], ["q"], name="after"),
        ],
        ["q", "z"],
    )
    assert [node.name for node in upstream(model, matmul)] == ["scale", "bias", "shift"]
