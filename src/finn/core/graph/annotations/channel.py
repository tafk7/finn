# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's choices, stored as metadata on the tensor the channel carries (the
``finn.channel`` namespace).

A channel belongs to its tensor, not to the nodes at its ends, so its choices are
stored there, and only in a partition's body.
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
"""Where a channel's choices are stored: metadata on the tensor the channel carries
(``finn.channel/<key>``), one key per decision of ``Channel`` (``CHANNEL_KEYS``). Only a
partition's body holds them: the cut moves them there and clears the parent's copy.
Each is set at the cut to its first valid value (the interim channel rule, until the
ends pass); a required one (a FIFO's depth) stays unset until a strategy sets it. A
tensor with a key ``Channel`` does not declare, or another version of the namespace, is
refused by name."""

CHANNEL_TYPES: dict[str, type] = {"int": int, "bool": bool, "str": str}


def value_type(item: inspection.DecisionInfo[object]) -> str:
    """A decision's value type by its semantics' name (``int``, ``bool``, ``str``)."""
    semantics = item.reference.semantics
    if semantics is None:
        raise KernelOpError.refused(
            "channel-key-untyped", "value_type", f"{item.key} has no value semantics"
        )
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
    """Refuse channel choices on ``tensors``, by name, unless ``model`` is a partition's
    body. A build sets them in a step after the cut, with ``Hold``."""
    named = list(tensors)
    if not named or is_kernel_partition(model):
        return
    hosts = [
        node.name or node.op_type for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN
    ]
    raise KernelOpError.refused(
        "channel-choices-outside-body",
        "refuse_outside_body",
        f"{', '.join(named)}: channel choices ({CHANNEL.name}) are stated in a partition's "
        f"body only, not on a graph with nodes on the host ({', '.join(hosts) or 'none'}): "
        "fix them in a step after the cut, with Hold, which sets them in the body",
        tuple(named),
    )


def channel_choices(model: ModelWrapper, tensor: str) -> dict[str, object]:
    """The choices stored on ``tensor``'s channel, by key: only those present. Refused
    outside a partition's body, or under keys ``CHANNEL`` does not declare."""
    try:
        stated: dict[str, object] = model.namespace(CHANNEL, tensor=tensor)
    except MetadataError as error:
        raise KernelOpError.refused(
            "channel-choices-unreadable",
            "channel_choices",
            f"{tensor}: its channel choices are not {CHANNEL.name}'s: {error}; explore the "
            "partition again",
            (tensor,),
            tensor=tensor,
        ) from error
    if stated:
        refuse_outside_body(model, [tensor])
    return stated


def clear_channel_choices(model: ModelWrapper) -> None:
    """Clear the channel choices stored on every tensor."""
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
