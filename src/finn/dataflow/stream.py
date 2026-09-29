# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stream: one tensor from one producer to one consumer, and the plan between them.

A ``Stream`` is a Space of its own, placed beside the kernels it joins. It
carries one ``tensor``, supplied by its composite; each end presents its own
``BeatSequence`` of that tensor. The stream reads its two ends (``ends``: the
source and the sink, each with its element and sequence) and derives the
``plan`` between them (``finn.dataflow.plan``): nothing, when they connect
directly or differ only in field order, otherwise reorders, width
conversions and marker synthesis. A plan no chain of steps carries out is
refused (``stream-plan``), as is any plan on a stream whose ``adaptable``
input is False.

How the ends are found is not logical: a physical stream finds them among the
kernels that reference it, and presents a missing side as its composite's
boundary. So ``ends`` is ``required``: this family cannot be placed, and the
physical stream (``finn.kernels.streams``) defines it.

The tensor must not depend on the stream's users: kernels read it to build
their ends, so a tensor derived from an end is a dependency cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.core.space import (
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    reject,
    required,
)
from finn.dataflow.plan import Plan, Unrealizable, plan
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence


@dataclass(frozen=True)
class End:
    """What one end of a stream presents: its element and beat sequence.

    ``owner`` names the kernel presenting it, or None for the composite's
    boundary.
    """

    owner: str | None
    element: ScalarEncoding
    sequence: BeatSequence


@dataclass(frozen=True)
class Ends:
    """A stream's producing and consuming ends."""

    source: End
    sink: End


class Stream(Space):
    """One tensor, its two ends, and the plan that joins them."""

    tensor: Tensor = Param()
    # False admits no adapter: the ends must connect directly.
    adaptable: bool = Param(default=True)
    ends = required(Ends)

    @constraint
    def well_formed(self) -> bool | Rejected:
        """Each end traverses this stream's tensor, in its element encoding."""
        tensor, ends = self.tensor, self.ends
        for end in (ends.source, ends.sink):
            where = end.owner or "the boundary"
            if end.sequence.form.shape != tensor.shape:
                return reject(
                    "stream-tensor",
                    f"{where} traverses a {end.sequence.form.shape} tensor; "
                    f"the stream carries {tensor.shape}",
                )
            if end.element != tensor.element:
                return reject(
                    "stream-tensor",
                    f"{where} carries {end.element.datatype_name}; "
                    f"the stream carries {tensor.element.datatype_name}",
                )
        return True

    @derived
    def plan(self) -> Plan | Rejected:
        """What must happen between the source's beat sequence and the sink's."""
        ends = self.ends
        try:
            return plan(ends.source.sequence, ends.sink.sequence)
        except Unrealizable as error:
            return reject("stream-plan", f"no adapter can join the ends: {error}")

    @derived
    def adapting(self) -> bool:
        return self.adaptable and bool(self.plan)

    @constraint
    def realizable(self) -> bool | Rejected:
        found = self.plan
        if found and not self.adaptable:
            return reject(
                "stream-plan",
                f"the ends need {found.describe()}, and this stream admits no adapter",
            )
        return True


__all__ = ["End", "Ends", "Stream"]
