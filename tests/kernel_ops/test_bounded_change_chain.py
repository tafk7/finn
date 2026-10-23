# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A successor answers as a recomputation, on one whole Space of eight kernels.

Stage 7's synthetic chain (four MatMuls, each followed by a Thresholding, and the
channels between them) joined as one Space, kernels first, its choices committed as
conversion and the cut set them (each kernel's by its default policies, each channel's
by the interim channel rule). The engine's walk (``support.space_walk``) applies seeded
changes to it: a kernel's core, its folding, a channel's memory or depth, a clear, a
pair of changes, and ones the kernels refuse; each successor is compared with a
configuration of the same assignments built from nothing. Here a channel's facts read
its kernel's ``compute``, as the engine's own examples do in
``core.space.test_bounded_change``.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from support.one_space import one_space
from support.space_walk import check, walk

from finn.core.space import Space
from finn.custom_op.kernels.shell import make_cut_choices
from finn.kernels.configure import commit
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
    """The chain's kernels and channels composed as one Space, kernels first
    (``support.one_space``), with the choices conversion and the cut store committed
    (``make_cut_choices``: what the cut does to a body)."""
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = mlp(layers).transform(ToKernelOps(target)).transform(InferKernelTensors())
    made = make_cut_choices(model)
    assert not (made.stale or made.late or made.unset), made
    composed, stored = one_space(model)
    return commit(composed.design_space(), stored)


def test_eight_kernels_answer_as_a_recomputation() -> None:
    # Few steps: each recomputes the base and the successor from nothing, ~0.7 s on this
    # chain. Seed 1 accepts, refuses, clears, pairs and puts a stored choice dormant.
    report = walk(whole_space(4), seed=1, steps=20, everything=False)
    check(report)
    assert report.tallies.dormant, report.tallies
