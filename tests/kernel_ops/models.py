# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Small ONNX models of KernelOps, and of the source graphs conversion reads.

Also the Chain (``kernels.chain``) as KernelOps with its choices saved, its partition
root rebuilt, and a KernelOp's schema digest."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from kernels import chain
from kernels.helpers import ADAPTER_RAM_STYLES
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import KernelOp, write_target
from finn.custom_op.kernels.partition import PartitionRoot, partition_root, save_partition_choices
from finn.kernels.configure import undecided
from finn.transformation.kernels import InferKernelTensors, ToKernelOps, resolve_target

DOMAIN = "finn.custom_op.kernels"
INT3 = DataType["INT3"]
ROWS, K, N = 3, 4, 4
WEIGHTS = np.array([[(3 * n + 2 * k) % 7 - 3 for n in range(N)] for k in range(K)])
X = np.array([[[(5 * r + 3 * k) % 8 - 4 for k in range(K)] for r in range(ROWS)]])
TARGET = resolve_target("xczu3eg-sbva484-1-e", 5.0)  # Ultra96: DSP48E2, no shell


def matmul_model(
    *,
    stored: bool = True,
    weights: Any = WEIGHTS,
    annotate: tuple[str, ...] = ("x", "w"),
    x_shape: list[int] | None = None,
    target: bool = True,
) -> ModelWrapper:
    """x (1, 3, 4) -> MatMul ``first`` (domain ``finn.custom_op.kernels``) with w -> y.

    ``stored``: w an initializer (the node's own), else a graph input.
    """
    weights = np.asarray(weights, dtype=np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape or [1, ROWS, K])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, list(weights.shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="first", domain=DOMAIN)
    inputs = [x] if stored else [x, w]
    graph = helper.make_graph([node], "matmul", inputs, [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-ops-test",
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
        )
    )
    if stored:
        model.set_initializer("w", weights)
    for name in annotate:
        model.set_tensor_datatype(name, INT3)
    if target:
        write_target(model, TARGET)
    return model


THRESHOLDS = np.array([[-9 + c, 1 - c, 8 + 2 * c] for c in range(N)])
H = DataType["INT8"]


def thresholding_model(
    *,
    thresholds: Any = THRESHOLDS,
    stored: bool = True,
    bias: int = 0,
    annotate: tuple[str, ...] = ("x", "t"),
) -> ModelWrapper:
    """x (3, 4) INT8 -> Thresholding ``activate`` with t (C, N) -> y."""
    thresholds = np.asarray(thresholds, dtype=np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [ROWS, N])
    t = helper.make_tensor_value_info("t", TensorProto.FLOAT, list(thresholds.shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    node = helper.make_node(
        "Thresholding", ["x", "t"], ["y"], name="activate", domain=DOMAIN, bias=bias
    )
    graph = helper.make_graph([node], "thresholding", [x] if stored else [x, t], [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-ops-test",
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
        )
    )
    if stored:
        model.set_initializer("t", thresholds)
    for name in annotate:
        model.set_tensor_datatype(name, H)
    write_target(model, TARGET)
    return model


def chain_source(*, annotate_input: bool = True, second_weights: bool = True) -> ModelWrapper:
    """The Chain as an ONNX model, before conversion: x -> MatMul ``first`` (w1)
    -> hidden -> MultiThreshold ``activate`` -> levels -> MatMul ``second`` (w2) -> y.
    Only x's shape is known (a fresh graph); w2 is a graph input unless ``second_weights``.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [chain.ROWS, chain.INPUTS])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["hidden"], name="first"),
        helper.make_node(
            "MultiThreshold",
            ["hidden", "thresholds"],
            ["levels"],
            name="activate",
            domain="qonnx.custom_op.general",
            out_dtype="UINT2",
            out_bias=0.0,
        ),
        helper.make_node("MatMul", ["levels", "w2"], ["y"], name="second"),
    ]
    w2 = helper.make_tensor_value_info("w2", TensorProto.FLOAT, [chain.HIDDEN, chain.OUTPUTS])
    inputs = [x] if second_weights else [x, w2]
    graph = helper.make_graph(nodes, "chain", inputs, [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-ops-test",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.set_initializer("w1", np.array(chain.W1, dtype=np.float32))
    if second_weights:
        model.set_initializer("w2", np.array(chain.W2, dtype=np.float32))
    model.set_initializer("thresholds", np.array(chain.THRESHOLDS[0], dtype=np.float32))
    if annotate_input:
        model.set_tensor_datatype("x", chain.A)
    for name in ("w1", "w2"):
        model.set_tensor_datatype(name, chain.W)
    model.set_tensor_datatype("thresholds", chain.H)
    return model


def lift(model: ModelWrapper, tensor: str) -> None:
    """Make an initializer a graph input (a stored node's weights become an edge)."""
    values = model.get_initializer(tensor)
    model.graph.initializer.remove(next(i for i in model.graph.initializer if i.name == tensor))
    info = model.get_tensor_valueinfo(tensor)
    if info is not None:
        model.graph.value_info.remove(info)
    model.graph.input.append(
        helper.make_tensor_value_info(tensor, TensorProto.FLOAT, list(values.shape))
    )


def schema_digest(cls: type[KernelOp]) -> str:
    rows = sorted((name, kind, cases) for name, (kind, cases) in cls.schema().items())
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


MATMUL = {
    "compute": "packed",
    "compute.packed.pe": chain.PE,
    "compute.packed.simd": chain.SIMD,
    "compute.packed.compute_pumping": False,
    "compute.packed.reducer": "tree",
    "w.source.memstream.ram_style": "auto",
    "w.source.memstream.pumped_memory": False,
    "w.transport": "direct",
    "x.transport": "direct",
}
THRESHOLDING = {
    "pe": chain.PE,
    "use_axilite": False,
    "deep_pipeline": False,
    "ram_style": "auto",
    "ultra_stages": 0,
    "x.transport": "direct",
}


def kernel_model(**options: bool) -> ModelWrapper:
    """The Chain as KernelOps, each node's choices saved as ``kernels.chain`` configures them."""
    model = (
        chain_source(**options)
        .transform(InferShapes())
        .transform(ToKernelOps(TARGET))
        .transform(InferKernelTensors())
    )
    for node in model.graph.node:
        choices = MATMUL if node.op_type == "MatMul" else THRESHOLDING
        if node.op_type == "MatMul" and model.get_initializer(node.input[1]) is None:
            # Streamed weights: no value, so no source; the weight edge's transport is
            # the root's.
            choices = {k: v for k, v in choices.items() if not k.startswith("w.source.")}
        if node.output[0] in {output.name for output in model.graph.output}:
            # A graph output: no KernelOp consumes it, so its producer owns its transport.
            choices = {**choices, "y.transport": "direct"}
        model.get_customop_wrapper(node).save(choices)
    return model


def open_memories(root: PartitionRoot) -> tuple[Any, list[str]]:
    """The root's point and its open adapter memories."""
    return root.point, undecided(root.point, ADAPTER_RAM_STYLES)


def configure_partition(model: ModelWrapper) -> tuple[PartitionRoot, Any]:
    """The root, its open adapter memories chosen and saved on their owners, rebuilt."""
    root = partition_root(model, model.graph.node, name="chain")
    _, styles = open_memories(root)
    save_partition_choices(model, root, dict.fromkeys(styles, "auto"))
    root = partition_root(model, model.graph.node, name="chain")
    point, open_styles = open_memories(root)
    assert open_styles == [] and root.dropped == ()
    return root, point


__all__ = [
    "DOMAIN",
    "chain_source",
    "configure_partition",
    "kernel_model",
    "open_memories",
    "schema_digest",
    "H",
    "INT3",
    "THRESHOLDS",
    "WEIGHTS",
    "X",
    "lift",
    "matmul_model",
    "thresholding_model",
]
