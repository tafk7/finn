# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's design space, read as FINN's graph layer reads it.

``declaration`` reads a model of KernelOps whose channel choices are stored (the cut's, or
``make_cut_choices``'), as the graph layer declares it (``finn.custom_op.kernels.network
.declaration``, in the shell of the model's target: its row's movers offered on the
boundary), and the values conversion and the cut stored, by key, as the partition binds
them (``in_shell``). ``one_space`` composes it (``compose``): the class with its facts, and
the stored values by path. ``mlp`` is stage 7's synthetic chain, and ``chain_space`` that
chain composed.

A test helper, imported as ``support.one_space``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.core.graph.annotations.platform import read_target
from finn.custom_op.kernels.network import declaration as declared
from finn.custom_op.kernels.shell import in_shell, make_cut_choices
from finn.kernels.composer import Composed, Key, PartitionDeclaration, compose
from finn.platform.resolve import resolve_target
from finn.platform.shells import shell_row
from finn.transformation.kernels import InferKernelTensors, ToKernelOps


def mlp(layers: int) -> ModelWrapper:
    """Stage 7's synthetic chain: ``layers`` of a MatMul (16 by 16, INT3 weights) and a
    MultiThreshold to UINT2, on 4 rows of UINT2."""
    width = 16
    nodes: list[Any] = []
    current = "x"
    for layer in range(layers):
        levels = "y" if layer == layers - 1 else f"a{layer}"
        nodes.append(
            helper.make_node("MatMul", [current, f"w{layer}"], [f"h{layer}"], name=f"mm{layer}")
        )
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                [f"h{layer}", f"t{layer}"],
                [levels],
                name=f"mt{layer}",
                domain="qonnx.custom_op.general",
                out_dtype="UINT2",
                out_bias=0.0,
            )
        )
        current = levels
    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, width])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.set_tensor_datatype("x", DataType["UINT2"])
    for layer in range(layers):
        weights = [[(3 * n + 2 * k + layer) % 7 - 3 for n in range(width)] for k in range(width)]
        model.set_initializer(f"w{layer}", np.array(weights, dtype=np.float32))
        model.set_tensor_datatype(f"w{layer}", DataType["INT3"])
        steps = [[-9 + c % 5, 1 - c % 3, 8 + c % 4] for c in range(width)]
        model.set_initializer(f"t{layer}", np.array(steps, dtype=np.float32))
        model.set_tensor_datatype(f"t{layer}", DataType["INT8"])
    return model


def declaration(model: ModelWrapper) -> tuple[PartitionDeclaration, dict[Key, object]]:
    """``model``'s KernelOps, their channel choices stored, as a partition's declaration
    (each node's kernel by its facts, in graph order; each channel as the graph layer
    declares it, in its shell), and the values conversion and the cut stored, by (owner,
    key)."""
    nodes = list(model.graph.node)
    target = read_target(model)
    row = shell_row(target.shell, target.board)
    return declared(model, nodes, movers=row.movers), dict(in_shell(model, nodes).chosen())


def one_space(model: ModelWrapper) -> tuple[Composed, dict[str, object]]:
    """``model``'s KernelOps, their channel choices stored, composed as one Space
    (``declaration``, ``compose``: its class, names and facts), and the values conversion
    and the cut stored, by path."""
    declared, stored = declaration(model)
    composed = compose(declared)
    return composed, {composed.names.path(key): value for key, value in stored.items()}


def chain_model(layers: int) -> ModelWrapper:
    """Stage 7's chain of ``2 * layers`` kernels (``mlp``) for the ZU3EG at 5 ns, converted
    and its channel choices made."""
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = mlp(layers).transform(ToKernelOps(target)).transform(InferKernelTensors())
    make_cut_choices(model)
    return model


def chain_space(layers: int) -> tuple[Composed, dict[str, object]]:
    """Stage 7's chain of ``2 * layers`` kernels (``chain_model``), composed as one Space
    (``one_space``), and its stored values."""
    return one_space(chain_model(layers))


__all__ = ["chain_model", "chain_space", "declaration", "mlp", "one_space"]
