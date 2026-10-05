# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's two ends as values, and the rule that they traverse its tensor.

A channel (``finn.kernels.channels``) carries one tensor from one producer to
one consumer. Each end presents its own ``BeatSequence`` of that tensor with
its own element; ``End`` and ``Ends`` hold what the two ends present, and
``misfit`` is the logical rule between them and the tensor: each end
traverses the tensor's shape, the source's values fit the tensor's element,
and the tensor's fit the sink's. What joins the two sequences is
``finn.dataflow.plan``. How the ends are found (the kernels' ports, the root's
boundary) is the channel's, not this layer's.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence


@dataclass(frozen=True)
class End:
    """What one end of a channel presents: its element and beat sequence.

    ``owner`` names the kernel presenting it, or None for the root's boundary.
    """

    owner: str | None
    element: ScalarEncoding
    sequence: BeatSequence


@dataclass(frozen=True)
class Ends:
    """A channel's producing and consuming ends."""

    source: End
    sink: End


def misfit(tensor: Tensor, ends: Ends) -> str | None:
    """Why the ends do not traverse ``tensor``, or None when they do: each end traverses
    its shape; the source's values fit its element, and its element the sink's."""
    for end in (ends.source, ends.sink):
        where = end.owner or "the boundary"
        if end.sequence.form.shape != tensor.shape:
            return (
                f"{where} traverses a {end.sequence.form.shape} tensor; "
                f"the channel carries {tensor.shape}"
            )
        inner, outer = (
            (end.element, tensor.element) if end is ends.source else (tensor.element, end.element)
        )
        if not inner.fits(outer):
            return f"{where} carries {end.element}; the channel carries {tensor.element}"
    return None


__all__ = ["End", "Ends", "misfit"]
