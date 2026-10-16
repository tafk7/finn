# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A successor answers as a recomputation, on one whole Space of eight kernels.

Stage 7's synthetic chain (four MatMuls, each followed by a Thresholding, and the
channels between them) joined as one Space, kernels first, its baseline committed. The
engine's walk (``support.space_walk``) applies seeded changes to it: a kernel's
core, its folding, a channel's adapter or depth, a clear, a pair of changes, and ones
the kernels refuse; each successor is compared with a configuration of the same
assignments built from nothing. Here a channel's verdicts read its kernel's forced
``compute``, as the engine's own examples do in ``core.space.test_bounded_change``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from support.space_walk import check, walk

from finn.core.space import Space, _forcing, composite, design_space, occurrence
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import in_shell
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.explore import Seam
from finn.kernels.network import Fact, Interface
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import InferKernelTensors, ToKernelOps

D, ROWS = 16, 4


def mlp(layers: int) -> ModelWrapper:
    """``layers`` of a MatMul and a MultiThreshold, as the stage 7 chain."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [ROWS, D])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes, current = [], "x"
    for i in range(layers):
        hidden, levels = f"h{i}", ("y" if i == layers - 1 else f"a{i}")
        nodes.append(helper.make_node("MatMul", [current, f"w{i}"], [hidden], name=f"mm{i}"))
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                [hidden, f"t{i}"],
                [levels],
                name=f"mt{i}",
                domain="qonnx.custom_op.general",
                out_dtype="UINT2",
                out_bias=0.0,
            )
        )
        current = levels
    graph = helper.make_graph(nodes, "mlp", [x], [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="bounded-change",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.set_tensor_datatype("x", DataType["UINT2"])
    for i in range(layers):
        w = np.array(
            [[(3 * n + 2 * k + i) % 7 - 3 for n in range(D)] for k in range(D)], dtype=np.float32
        )
        model.set_initializer(f"w{i}", w)
        model.set_tensor_datatype(f"w{i}", DataType["INT3"])
        t = np.array([[-9 + c % 5, 1 - c % 3, 8 + c % 4] for c in range(D)], dtype=np.float32)
        model.set_initializer(f"t{i}", t)
        model.set_tensor_datatype(f"t{i}", DataType["INT8"])
    return model


def whole_space(layers: int) -> Space:
    """The chain's kernels and channels as one Space, kernels first, the baseline's
    stored choices committed, its forced Decisions found."""
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = mlp(layers).transform(ToKernelOps(target)).transform(InferKernelTensors())
    nodes = list(model.graph.node)
    net = in_shell(model, nodes)
    baseline = Seam(target.platform).baseline(net)
    declared: dict[str, Any] = {}
    for node in nodes:
        facts = kernel_op(model, node).facts()
        declared[node.name] = facts.space_type(**cast("dict[str, Any]", dict(facts.arguments)))
    channels: dict[str, object] = {}
    for name in net.channels:
        given: dict[str, Any] = {}
        for fact, value in net.wiring(name).facts.items():
            if isinstance(value, (Interface, Fact)):
                value = getattr(declared[value.member], value.view)
            given[fact] = value
        channels[name] = Channel(**given)
    whole = composite(f"chain_{2 * layers}", {**declared, **channels})
    stored = {f"{owner}.{key}": value for (owner, key), value in baseline.chosen().items()}
    point = commit(design_space(whole()), stored)
    _forcing.forced(occurrence.state(point))
    return point


def test_eight_kernels_answer_as_a_recomputation() -> None:
    # Few steps: each recomputes the base and the successor from nothing, ~0.7 s on this
    # chain. Seed 3 accepts, refuses, clears, pairs and moves a forced value within 12.
    check(walk(whole_space(4), seed=3, steps=20, everything=False))
