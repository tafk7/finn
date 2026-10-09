# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition body's choices, by graph name: the form ``Pinned`` reads back.

FINN's ``extract_model_config_to_json`` reads every attribute through
``get_nodeattr``, which answers an absent choice with its default, so an open
choice would come back as a value. This export is sparse: each KernelOp node's
choices as its node holds them, each channel's as its tensor states them
(``finn.channel``), absent ones absent, in graph order: each node's input channels
not yet written (a parameter it owns among them), the node, its output channels. A
node and a tensor never share a name in a partition's body (its shell root refuses
it, and this export too, by raw name, on a body no root has seen), so
``{graph name: {key: value}}`` names each once. The nodes' entries are also
the form ``ApplyConfig`` applies; it applies no tensor's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finn.custom_op.kernels.base import CHANNEL, KernelOpError, channel_choices, kernel_op
from finn.custom_op.partition.kernel_partitions import KERNEL_OPS_DOMAIN

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def kernel_choices_config(model: ModelWrapper) -> dict[str, dict[str, object]]:
    """Each KernelOp node's choices by node name and each tensor's channel choices by
    tensor name (``channel_choices``: in a partition's body only), in graph order: for
    each node, the tensors it reads that state choices and are not yet written, then the
    node, then the tensors it writes that state choices; a tensor stating choices that
    no node reads or writes comes last, by name. A node or tensor with none has no
    entry. A tensor stating choices that is named like a node is refused, by name."""
    stating = set(model.tensors_stating(CHANNEL))
    if clash := sorted(stating & {node.name for node in model.graph.node}):
        raise KernelOpError(f"{clash[0]}: a node and a tensor are both named {clash[0]}")
    config: dict[str, dict[str, object]] = {}
    written: set[str] = set()

    def write(tensor: str) -> None:
        if tensor in stating and tensor not in written:
            written.add(tensor)
            if choices := channel_choices(model, tensor):
                config[tensor] = choices

    for node in model.graph.node:
        for tensor in node.input:
            write(tensor)
        if node.domain == KERNEL_OPS_DOMAIN and (choices := kernel_op(model, node).choices()):
            config[node.name] = choices
        for tensor in node.output:
            write(tensor)
    for tensor in sorted(stating - written):
        write(tensor)
    return config


__all__ = ["kernel_choices_config"]
