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

from finn.custom_op.kernels.base import KernelOp, kernel_op, write_target
from finn.custom_op.kernels.shell import ShellRoot, persist, shell_root
from finn.kernels.configure import commit, undecided
from finn.kernels.target import Target
from finn.platform import resolve_target
from finn.transformation.kernels import InferKernelTensors, ToKernelOps

DOMAIN = "finn.custom_op.kernels"
INT3 = DataType["INT3"]
ROWS, K, N = 3, 4, 4
WEIGHTS = np.array([[(3 * n + 2 * k) % 7 - 3 for n in range(N)] for k in range(K)])
X = np.array([[[(5 * r + 3 * k) % 8 - 4 for k in range(K)] for r in range(ROWS)]])
TARGET = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)  # Ultra96's part, ip


def matmul_model(
    *,
    stored: bool = True,
    weights: Any = WEIGHTS,
    annotate: tuple[str, ...] = ("x", "w"),
    x_shape: list[int] | None = None,
    target: bool = True,
    infer: bool = True,
) -> ModelWrapper:
    """x (1, 3, 4) -> MatMul ``first`` (domain ``finn.custom_op.kernels``) with w -> y.

    ``stored``: w an initializer (the node's own), else a graph input. ``infer``: y
    stated by ``InferKernelTensors``, as the node root reads it; without it, only
    the node's facts are read.
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
    return model.transform(InferKernelTensors()) if infer else model


THRESHOLDS = np.array([[-9 + c, 1 - c, 8 + 2 * c] for c in range(N)])
H = DataType["INT8"]


def thresholding_model(
    *,
    thresholds: Any = THRESHOLDS,
    stored: bool = True,
    bias: int = 0,
    annotate: tuple[str, ...] = ("x", "t"),
    infer: bool = True,
    target: Target = TARGET,
) -> ModelWrapper:
    """x (3, 4) INT8 -> Thresholding ``activate`` with t (C, N) -> y, built for ``target``;
    ``infer`` as for ``matmul_model``."""
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
    write_target(model, target)
    return model.transform(InferKernelTensors()) if infer else model


def chain_source(
    *, annotate_input: bool = True, second_weights: bool = True, rows: int = chain.ROWS
) -> ModelWrapper:
    """The Chain as an ONNX model, before conversion: x -> MatMul ``first`` (w1)
    -> hidden -> MultiThreshold ``activate`` -> levels -> MatMul ``second`` (w2) -> y.
    Only x's shape is known (a fresh graph), ``rows`` rows (the Chain's 3 by default);
    w2 is a graph input unless ``second_weights``.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, chain.INPUTS])
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


def host_between_source() -> ModelWrapper:
    """x (4, 4) INT3 -> MatMul ``a`` (w) -> h -> Transpose ``flip`` -> t -> MatMul ``b``
    (w) -> y: a host node between two nodes a KernelOp binds."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
        helper.make_node("Transpose", ["h"], ["t"], name="flip"),
        helper.make_node("MatMul", ["t", "w"], ["y"], name="b"),
    ]
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "between", [x], [y]),
            producer_name="kernel-ops-test",
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_initializer("w", np.eye(4, dtype=np.float32))
    for name in ("x", "w"):
        model.set_tensor_datatype(name, INT3)
    return model


def fan_out_source(*, output: str | None = None) -> ModelWrapper:
    """x -> MatMul ``first`` (w1) -> hidden, read more than once: by MultiThreshold ``a``
    -> ya and ``b`` -> yb; or, with ``output`` (``"first"`` or ``"last"``), by ``a`` only
    and a graph output too, listed before or after ya."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [chain.ROWS, chain.INPUTS])
    nodes = [helper.make_node("MatMul", ["x", "w1"], ["hidden"], name="first")]
    outputs = []
    for reader in ("a",) if output else ("a", "b"):
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                ["hidden", "thresholds"],
                [f"y{reader}"],
                name=reader,
                domain="qonnx.custom_op.general",
                out_dtype="UINT2",
                out_bias=0.0,
            )
        )
        outputs.append(helper.make_tensor_value_info(f"y{reader}", TensorProto.FLOAT, None))
    if output:
        hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, None)
        outputs.insert(0 if output == "first" else len(outputs), hidden)
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "fan", [x], outputs),
            producer_name="kernel-ops-test",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.set_initializer("w1", np.array(chain.W1, dtype=np.float32))
    model.set_initializer("thresholds", np.array(chain.THRESHOLDS[0], dtype=np.float32))
    model.set_tensor_datatype("x", chain.A)
    model.set_tensor_datatype("w1", chain.W)
    model.set_tensor_datatype("thresholds", chain.H)
    return model.transform(InferShapes())


def shared_weights_source() -> ModelWrapper:
    """x (4, 4) INT3 -> MatMul ``a`` (w) -> h -> MatMul ``b`` (w) -> y: one initializer,
    each MatMul's own weights."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
        helper.make_node("MatMul", ["h", "w"], ["y"], name="b"),
    ]
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "shared", [x], [y]),
            producer_name="kernel-ops-test",
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_initializer("w", np.eye(4, dtype=np.float32))
    for name in ("x", "w"):
        model.set_tensor_datatype(name, INT3)
    return model.transform(InferShapes())


def lift(model: ModelWrapper, tensor: str) -> None:
    """Make an initializer a graph input (a stored node's weights become an edge)."""
    values = model.get_initializer(tensor)
    assert values is not None, f"{tensor} is not an initializer"
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


def kernel_model(**options: Any) -> ModelWrapper:
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
        kernel_op(model, node).save(choices)
    return model


#: The second MatMul's folding under which the Chain's streamed w2 presents each pass
#: row-major at its free side, the order an IODMA end moves (``IodmaEnd.row_major``): a
#: row of PE lanes a beat. At the Chain's own (2 x 2) it is tiled, and the pynq shell
#: refuses it (``end-order``).
ROW_MAJOR_W2 = {"compute.packed.pe": 4, "compute.packed.simd": 1}


def streamed_w2_model() -> ModelWrapper:
    """The Chain with x and w2 streamed as IODMA ends move them, each one pass a frame
    (``IodmaEnd.single_pass``): one row of x, so w2 presents its pass once (at the
    Chain's three rows it presents it three times, which the pynq shell refuses,
    ``end-repetition``), and the first MatMul folded at PE 4, all of its outputs, so
    that x is read once (at PE 2 it is read twice: over one row, the replay of each
    row is a repetition of the whole pass); the second folded ``ROW_MAJOR_W2``."""
    model = kernel_model(second_weights=False, rows=1)
    (first,) = [node for node in model.graph.node if node.name == "first"]
    kernel_op(model, first).save({"compute.packed.pe": 4})
    (second,) = [node for node in model.graph.node if node.name == "second"]
    kernel_op(model, second).save(ROW_MAJOR_W2)
    return model


def open_memories(root: ShellRoot) -> tuple[Any, list[str]]:
    """The root's point and its open adapter memories."""
    return root.point, undecided(root.point, ADAPTER_RAM_STYLES)


def configure_partition(model: ModelWrapper) -> tuple[ShellRoot, Any]:
    """The root, its open adapter memories chosen and saved on their owners, rebuilt."""
    root = shell_root(model, model.graph.node, name="chain")
    _, styles = open_memories(root)
    persist(model, root, commit(root.point, dict.fromkeys(styles, "auto")))
    root = shell_root(model, model.graph.node, name="chain")
    point, open_styles = open_memories(root)
    assert open_styles == [] and not root.dropped
    return root, point


__all__ = [
    "DOMAIN",
    "chain_source",
    "configure_partition",
    "kernel_model",
    "open_memories",
    "streamed_w2_model",
    "ROW_MAJOR_W2",
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
