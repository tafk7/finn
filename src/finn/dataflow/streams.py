# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as values: what a port presents of a tensor, and the rule that a channel's
two streams traverse its tensor.

A ``Stream`` is what one port presents of the tensor it reads or writes: the encoding of every
element
and its ``BeatSequence``. It holds no owner: where a stream sits (the kernel whose port
presents it, or a network's boundary) is the channel's to say. A channel
(``finn.kernels.channels``) carries one tensor from one producer to one consumer;
``Streams`` holds its two, and ``misfit`` is the logical rule between them and the
tensor: each traverses the tensor's shape, the producer's values fit the tensor's
encoding, and the tensor's fit the consumer's. What joins the two sequences is
``finn.dataflow.steps``. How a channel finds its streams (the kernels' ports, the
network's boundary) is the channel's, not this layer's.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence


@dataclass(frozen=True)
class Stream:
    """What a port presents of a tensor: the encoding of every element and its beat sequence."""

    element: ScalarEncoding
    sequence: BeatSequence


@dataclass(frozen=True)
class Streams:
    """A channel's producing and consuming streams."""

    source: Stream
    sink: Stream


def misfit(
    tensor: Tensor, streams: Streams, owners: tuple[str | None, str | None] = (None, None)
) -> str | None:
    """Why the streams do not traverse ``tensor``, or None when they do: each traverses its
    shape; the source's values fit its element, and its element the sink's. ``owners``
    names where each stream sits, the source's first (None: the boundary), for the
    reason."""
    for side, (stream, owner) in enumerate(zip((streams.source, streams.sink), owners)):
        where = owner or "the boundary"
        if stream.sequence.form.shape != tensor.shape:
            return (
                f"{where} traverses a {stream.sequence.form.shape} tensor; "
                f"the channel carries {tensor.shape}"
            )
        inner, outer = (
            (stream.element, tensor.element) if side == 0 else (tensor.element, stream.element)
        )
        if not inner.fits(outer):
            return f"{where} carries {stream.element}; the channel carries {tensor.element}"
    return None


__all__ = ["Stream", "Streams", "misfit"]
