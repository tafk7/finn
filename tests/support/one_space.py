# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's network as one Space: the composer's (``finn.kernels.composer``) class,
from the declaration this module reads off the shell's network.

``declaration`` reads a model of KernelOps whose channel choices are stored (the cut's, or
``make_cut_choices``'), as FINN's graph layer will: each node's kernel by its facts
(``KernelOp.facts``), each channel from the shell's wiring (``in_shell``: a port's side,
``Interface``, as the kernel's ``Port``; a value a kernel states, ``Fact``, as ``Stated``;
a side no port is wired to as its source where a kernel states its value, else as the
boundary, ``Side.boundary`` of the wire's ``port`` and ``mover_offer``), and the values
conversion and the cut stored, by key. ``one_space`` composes it (``compose``): the class
with its facts, and the stored values by path. ``mlp`` is stage 7's synthetic chain, and
``chain_space`` that chain composed.

A test helper, imported as ``support.one_space``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import in_shell, make_cut_choices
from finn.kernels.composer import (
    ChannelDeclaration,
    Composed,
    Key,
    PartitionDeclaration,
    Port,
    Stated,
    compose,
)
from finn.kernels.movers import MoverOffer
from finn.kernels.network import Fact, Interface, Wire
from finn.kernels.sides import Side
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


def channel_declaration(wired: Wire) -> ChannelDeclaration:
    """A channel as the network's ``wired`` declares it: a port (``Interface``) as its
    kernel's ``Port``, a value a kernel states (``Fact``) as ``Stated``; a side no port is
    wired to is the channel's source where a kernel states its value (``contents``), else
    the boundary (``Side.boundary``: the wire's ``port`` and ``mover_offer``)."""
    facts = dict(wired.facts)
    port, offers = cast(str, facts.pop("port", None)), facts.pop("mover_offer", ())
    sides: dict[str, Port | Side] = {}
    given: dict[str, object] = {}
    for fact, value in facts.items():
        if isinstance(value, Interface):
            sides[fact] = Port(value.member, value.port)
        elif isinstance(value, Fact):
            given[fact] = Stated(value.member, value.name)
        else:
            given[fact] = value
    if "contents" in given and "producer" not in sides:
        sides["producer"] = Side.source()
    for side in ("producer", "consumer"):
        if side not in sides:
            sides[side] = Side.boundary(port, cast("tuple[MoverOffer, ...]", offers))
    return ChannelDeclaration(sides["producer"], sides["consumer"], given)


def declaration(model: ModelWrapper) -> tuple[PartitionDeclaration, dict[Key, object]]:
    """``model``'s KernelOps, their channel choices stored, as a partition's declaration
    (each node's kernel by its facts, in graph order; each channel as the shell's network
    wires it, ranked), and the values conversion and the cut stored, by (owner, key)."""
    nodes = list(model.graph.node)
    net = in_shell(model, nodes)
    kernels = {node.name: kernel_op(model, node).facts() for node in nodes}
    channels = {name: channel_declaration(net.wiring(name)) for name in net.channels}
    return PartitionDeclaration(kernels, channels), net.chosen()


def one_space(model: ModelWrapper) -> tuple[Composed, dict[str, object]]:
    """``model``'s KernelOps, their channel choices stored, composed as one Space
    (``declaration``, ``compose``: its class, names and facts), and the values conversion
    and the cut stored, by path."""
    declared, stored = declaration(model)
    composed = compose(declared)
    return composed, {composed.names.path(key): value for key, value in stored.items()}


def chain_space(layers: int) -> tuple[Composed, dict[str, object]]:
    """Stage 7's chain of ``2 * layers`` kernels (``mlp``) for the ZU3EG at 5 ns, converted
    and its channel choices made, composed as one Space (``one_space``), and its stored
    values."""
    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = mlp(layers).transform(ToKernelOps(target)).transform(InferKernelTensors())
    make_cut_choices(model)
    return one_space(model)


__all__ = ["chain_space", "channel_declaration", "declaration", "mlp", "one_space"]
