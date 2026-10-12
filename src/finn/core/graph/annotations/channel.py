# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's choices in graph metadata: the ``finn.channel`` namespace, on the tensor
the channel carries.

The keys are generated from the Decisions ``Channel`` declares, each typed by its
value semantics (``value_type``) and a selector's values its cases. The choices are
stated in a partition's body only: a graph with nodes on the host states none.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from qonnx.core.metadata import Key, MetadataError, Namespace

from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import KERNEL_OPS_DOMAIN, is_kernel_partition
from finn.core.space import inspection
from finn.kernels.channels import Channel

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

CHANNEL = Namespace("finn.channel", version=1, follow=True)
"""A channel's stated choices, typed metadata of the tensor it carries (qonnx's
``qonnx.core.metadata``; entries ``finn.channel/transport``), each at its baseline from
the cut on (``BaselineChoices``) until a strategy replaces it; absent only on a body no
cut created or for a required choice (a FIFO's depth) no strategy has set. One key per
Decision ``Channel`` declares (``CHANNEL_KEYS``), generated from them
as a KernelOp's schema is from its node root, its type the Decision's value semantics'
and a selector's values its cases. Stated in a partition's body only
(``channel_choices``). It follows its tensor into the body the cut makes, and the cut
clears the parent's copy, so a channel has one home. A tensor stating another shape of
it (a key ``Channel`` does not declare, or another version) is refused, by name: the
partition is explored again."""

CHANNEL_TYPES: dict[str, type] = {"int": int, "bool": bool, "str": str}


def value_type(item: inspection.DecisionInfo[object]) -> str:
    """A decision's value type by its semantics' name (``int``, ``bool``, ``str``)."""
    semantics = item.reference.semantics
    if semantics is None:
        raise KernelOpError(f"{item.key} has no value semantics")
    return semantics.name


def _channel_key(item: inspection.DecisionInfo[object]) -> Key[Any]:
    kind = CHANNEL_TYPES[value_type(item)]
    if not item.selector:
        return CHANNEL.key(item.key, kind)
    cases = item.cases
    return CHANNEL.key(
        item.key, kind, check=lambda value: value in cases, expect=f"one of {list(cases)}"
    )


CHANNEL_KEYS: dict[str, Key[Any]] = {
    item.key: _channel_key(item) for item in inspection.decisions(Channel)
}
"""``CHANNEL``'s keys by name: ``Channel``'s decision keys (``transport``,
``transport.fifo.buffer.depth``, ``source.memstream.ram_style``, ...)."""


def refuse_outside_body(model: ModelWrapper, tensors: Iterable[str]) -> None:
    """Refuse the channel choices ``tensors`` state, by name, unless ``model`` is a
    partition's body (its nodes all KernelOps): a graph with nodes on the host (the
    whole graph, before the cut) states none; the build's choices come in through
    ``kernel_choices.json`` (``Pinned``), which exploration applies to the body."""
    named = list(tensors)
    if not named or is_kernel_partition(model):
        return
    hosts = [
        node.name or node.op_type for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN
    ]
    raise KernelOpError(
        f"{', '.join(named)}: channel choices ({CHANNEL.name}) are stated in a partition's "
        f"body only, not on a graph with nodes on the host ({', '.join(hosts) or 'none'}): "
        "state them through kernel_choices.json (Pinned), which exploration applies to the "
        "body",
        tuple(named),
    )


def channel_choices(model: ModelWrapper, tensor: str) -> dict[str, object]:
    """The choices ``tensor``'s channel states (``CHANNEL``), by key: only those present.
    Stated outside a partition's body (``refuse_outside_body``), or in a shape
    ``CHANNEL`` does not declare, they are refused, by name."""
    try:
        stated: dict[str, object] = model.namespace(CHANNEL, tensor=tensor)
    except MetadataError as error:
        raise KernelOpError(
            f"{tensor}: its channel choices are not {CHANNEL.name}'s: {error}; explore the "
            "partition again",
            (tensor,),
        ) from error
    if stated:
        refuse_outside_body(model, [tensor])
    return stated


def clear_channel_choices(model: ModelWrapper) -> None:
    """Clear the choices every tensor's channel states (``CHANNEL``)."""
    for tensor in model.tensors_stating(CHANNEL):
        model.clear(CHANNEL, tensor=tensor)


__all__ = [
    "CHANNEL",
    "CHANNEL_KEYS",
    "CHANNEL_TYPES",
    "channel_choices",
    "clear_channel_choices",
    "refuse_outside_body",
    "value_type",
]
