# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""No silent outcomes: what ``ToKernelOps`` made of each node, and why a node stays on
the host; the host nodes between KernelOps, by the graph's paths. Each pattern's
findings: ``test_patterns``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.core.space import FindingKind
from finn.custom_op.kernels.base import KernelOpError
from finn.transformation.kernels import (
    Outcome,
    ToKernelOps,
    between_kernel_ops,
    kernel_ops_report,
    kernel_ops_summary,
    refuse_host_between,
)
from kernel_ops.models import TARGET, chain_source


def convert(model: ModelWrapper) -> tuple[ModelWrapper, ToKernelOps]:
    conversion = ToKernelOps(TARGET)
    return model.transform(conversion), conversion


def codes(outcome: Outcome) -> list[tuple[str, str, str]]:
    return [(f.kind.value, f.owner, f.code) for f in outcome.findings]


def test_every_node_of_the_chain_converts_with_no_finding() -> None:
    expected = (
        Outcome(("first",), "MatMul"),
        Outcome(("activate",), "Thresholding"),
        Outcome(("second",), "MatMul"),
    )
    assert convert(chain_source().transform(InferShapes()))[1].outcomes == expected
    # A fresh graph, only x's shape known: conversion infers as it goes, so the
    # MultiThreshold's channel axis is known when it is visited.
    model, conversion = convert(chain_source())
    assert conversion.outcomes == expected
    assert [node.op_type for node in model.graph.node] == ["MatMul", "Thresholding", "MatMul"]


def graph(nodes: list[Any], outputs: list[str]) -> ModelWrapper:
    """x (4, 4) INT4 and the nodes, with w (4, 4) INT4 for every MatMul. qonnx types
    the host nodes' integer results by their ranges, so the MatMuls after them read
    integers the kernels admit."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])
    made = [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs]
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "g", [x], made),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_initializer("w", np.eye(4, dtype=np.float32))
    for name in ("x", "w"):
        model.set_tensor_datatype(name, DataType["INT4"])
    return model


def test_a_host_node_between_kernel_ops_is_named_one_beside_them_is_not() -> None:
    """``flip`` leaves ``a`` and re-enters at ``b``; ``negate``, between them in node
    order, only leaves (to an output), and ``scale`` only enters: graph convexity,
    not node order."""
    model, conversion = convert(
        graph(
            [
                helper.make_node("Relu", ["x"], ["r"], name="scale"),
                helper.make_node("MatMul", ["r", "w"], ["h"], name="a"),
                helper.make_node("Neg", ["h"], ["z"], name="negate"),
                helper.make_node("Transpose", ["h"], ["t"], name="flip"),
                helper.make_node("MatMul", ["t", "w"], ["y"], name="b"),
            ],
            ["y", "z"],
        )
    )
    assert between_kernel_ops(model) == ("flip",)
    report = kernel_ops_report(model, conversion.outcomes)
    assert report["converted"] == {"MatMul": 2}
    assert report["on_host"] == ["scale", "negate", "flip"]
    assert report["between_kernel_ops"] == ["flip"]
    flip = report["outcomes"][3]
    assert flip["nodes"] == ["flip"] and flip["op"] is None
    assert flip["findings"] == [
        {
            "kind": FindingKind.LIMITATION.value,
            "code": "no-kernel-op",
            "owner": "ToKernelOps",
            "message": "no KernelOp binds onnx.Transpose",
            "details": {"domain": "", "op_type": "Transpose"},
        }
    ]
    assert kernel_ops_summary(report) == [
        "ToKernelOps: 2 converted (MatMul 2); 3 on the host; report/kernel_ops.json",
        "ToKernelOps:   no-kernel-op (limitation) 3: Relu 1, Neg 1, Transpose 1",
    ]
    with pytest.raises(KernelOpError) as refused:
        refuse_host_between(model, conversion.outcomes)
    assert str(refused.value) == (
        "1 host nodes sit between KernelOps, so a partition of the KernelOps would depend "
        "on itself: flip (ToKernelOps: no-kernel-op: no KernelOp binds onnx.Transpose)"
    )


def test_a_path_through_several_host_nodes_names_each() -> None:
    model, _ = convert(
        graph(
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
    )
    assert between_kernel_ops(model) == ("first", "second", "join")


def test_host_nodes_at_the_edges_are_not_between() -> None:
    model, conversion = convert(
        graph(
            [
                helper.make_node("Relu", ["x"], ["r"], name="before"),
                helper.make_node("MatMul", ["r", "w"], ["h"], name="a"),
                helper.make_node("MatMul", ["h", "w"], ["m"], name="b"),
                helper.make_node("Neg", ["m"], ["y"], name="after"),
            ],
            ["y"],
        )
    )
    assert between_kernel_ops(model) == ()
    refuse_host_between(model, conversion.outcomes)
