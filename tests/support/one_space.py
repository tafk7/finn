# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's network as one Space: its kernels' declarations and its channels, as one
composite class.

``one_space`` reads a model of KernelOps whose channel choices are stored (the cut's, or
``make_cut_choices``'): each node's kernel declared from its facts
(``KernelOp.facts``), each channel from the shell's wiring (``in_shell``), its facts read
from the kernels' declarations (``channel_facts``: a port's stream and word width,
``Interface``; a value a kernel states, ``Fact``), and the values conversion and the cut
stored, by key. ``mlp`` is stage 7's
synthetic chain, and ``chain_space`` that chain as one Space.

A test helper, imported as ``support.one_space``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.core.space import Space, composite
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import in_shell, make_cut_choices
from finn.kernels.channels import Channel
from finn.kernels.network import Fact, Interface, Network
from finn.platform.resolve import resolve_target
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


def channel_facts(net: Network, channel: str, members: dict[str, Any]) -> dict[str, Any]:
    """Channel ``channel``'s facts as ``net`` wires them, each bound to the member of
    ``members`` (the declarations of one Space) it is read from, never a view: a port's
    stream (``Interface``: ``x_stream``) and, on the producer's side, its word width
    (``x_data_width``); a value a kernel states (``Fact``). A port's label is a constant:
    its contract's own (``compute.packed.x``), as ``net``'s bound member presents it."""
    given: dict[str, Any] = {}
    for fact, value in net.wiring(channel).facts.items():
        if isinstance(value, Interface):
            member = members[value.member]
            given[fact] = getattr(member, f"{value.port}_stream")
            given[f"{fact}_label"] = getattr(net.member(value.member), value.view).label
            if fact == "producer":
                given["producer_width"] = getattr(member, f"{value.port}_data_width")
        elif isinstance(value, Fact):
            given[fact] = getattr(members[value.member], value.name)
        else:
            given[fact] = value
    return given


def one_space(model: ModelWrapper, name: str) -> tuple[type[Space], dict[str, object]]:
    """``model``'s KernelOps, their channel choices stored, as one Space named ``name``
    (its kernels' declarations and the channels from the shell's wiring), and the values
    conversion and the cut stored, by key."""
    nodes = list(model.graph.node)
    net = in_shell(model, nodes)
    members: dict[str, Any] = {}
    for node in nodes:
        facts = kernel_op(model, node).facts()
        members[node.name] = facts.space_type(**cast("dict[str, Any]", dict(facts.arguments)))
    for channel in net.channels:
        members[channel] = Channel(**channel_facts(net, channel, members))
    stored = {f"{member}.{key}": value for (member, key), value in net.chosen().items()}
    return composite(name, members), stored


def chain_space(layers: int) -> tuple[type[Space], dict[str, object]]:
    """Stage 7's chain of ``2 * layers`` kernels (``mlp``) for the ZU3EG at 5 ns, converted
    and its channel choices made, as one Space (``one_space``), and its stored values."""
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = mlp(layers).transform(ToKernelOps(target)).transform(InferKernelTensors())
    make_cut_choices(model)
    return one_space(model, f"templates_chain_{2 * layers}")


__all__ = ["chain_space", "channel_facts", "mlp", "one_space"]
