# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's side: what one end of a channel is, as one value.

A ``Channel`` (``finn.kernels.channels``) takes two sides, its ``producer`` and its
``consumer``, each a ``Side`` of one of four kinds (``SideKind``):

- a **kernel**'s port (``Side.kernel``): the stream it presents (``x_stream``), its label
  (``compute.packed.x``, the label its contract carries) and, on an output port, its word
  width (``x_data_width``). A kernel states it for each of its ports (``x_side``,
  ``finn.kernels.port``);
- a **frozen neighbour**'s (``Side.neighbour``): the same three, given as a fact read from a
  kernel outside the selection at its accepted point;
- the **boundary**'s (``Side.boundary``): the module's AXIS port, its name as its label
  (``s_axis_0``, ``m_axis_0``), and the movers a shell offers on it (``offers``). Its
  ``stream`` is None: the boundary is unstated, and the channel presents there what its
  other side presents (``Channel.boundary``);
- a valued channel's **source** (``Side.source``): the channel's own memory, which produces
  its known value.

Each kind is made by its own constructor, so a side is whole or not made: a stream comes
with its label, a boundary with its port's name, a source with nothing. On a stated side
(a kernel's or a frozen neighbour's), a word width says it is an output port's: a
channel's producer side carries one, its consumer side none.

A side is a value class: frozen, compared and hashed field by field.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

from finn.core.space import default_semantics
from finn.dataflow.streams import Stream
from finn.kernels.movers import MoverOffer


class SideKind(Enum):
    """What a channel's side is: a kernel's port, a frozen neighbour's, the boundary's, or
    the channel's own source."""

    KERNEL = "kernel"
    NEIGHBOUR = "neighbour"
    BOUNDARY = "boundary"
    SOURCE = "source"


STATED = (SideKind.KERNEL, SideKind.NEIGHBOUR)
"""The kinds that state their stream."""


@dataclass(frozen=True)
class Side:
    """One side of a channel (the module docstring): its ``kind``, its ``label`` (a port's
    label, or the boundary's AXIS port's name; a source's is empty), its ``stream`` (a
    stated side's; None on the boundary while it is unstated, and for a source), its word
    width (``width``: an output port's), and the movers a shell offers on the boundary
    (``offers``). Made by ``kernel``, ``neighbour``, ``boundary`` or ``source``."""

    kind: SideKind
    label: str = ""
    stream: Stream | None = None
    width: int | None = None
    offers: tuple[MoverOffer, ...] = ()

    def __post_init__(self) -> None:
        stated = self.kind in STATED
        if stated and (self.stream is None or not self.label):
            raise ValueError(f"a {self.kind.value}'s side states its stream and its label")
        if not stated and self.width is not None:
            raise ValueError(f"a {self.kind.value}'s side states no word width")
        if self.kind is SideKind.BOUNDARY and (self.stream is not None or not self.label):
            raise ValueError("the boundary's side is its AXIS port's name, its stream unstated")
        if self.kind is not SideKind.BOUNDARY and self.offers:
            raise ValueError("only the boundary's side is offered movers")
        if self.kind is SideKind.SOURCE and (self.stream is not None or self.label):
            raise ValueError("a source's side states nothing: the channel's source is its own")

    @classmethod
    def kernel(cls, stream: Stream, label: str, width: int | None = None) -> Self:
        """A kernel's port: its ``stream``, its ``label``, and on an output port its word
        ``width``."""
        return cls(SideKind.KERNEL, label, stream, width)

    @classmethod
    def neighbour(cls, stream: Stream, label: str, width: int | None = None) -> Self:
        """A frozen neighbour's port, as a fact: its ``stream``, its ``label``, and on an
        output port (an upstream neighbour's) its word ``width``."""
        return cls(SideKind.NEIGHBOUR, label, stream, width)

    @classmethod
    def boundary(cls, port: str, offers: tuple[MoverOffer, ...] = ()) -> Self:
        """The boundary, unstated: its AXIS port's name (``port``) and the movers a shell
        offers on it (``offers``)."""
        return cls(SideKind.BOUNDARY, port, offers=tuple(offers))

    @classmethod
    def source(cls) -> Self:
        """A valued channel's source: its producer is its own memory."""
        return cls(SideKind.SOURCE)

    @property
    def stated(self) -> bool:
        """Whether it states its stream: a kernel's port, or a frozen neighbour's."""
        return self.kind in STATED

    def under(self, member: str) -> Self:
        """A stated side as seen from where ``member``, which states it, is held: its label
        below ``member`` (``first.compute.packed.x``), as ``PortContract.under``."""
        if not self.stated:
            raise ValueError(f"a {self.kind.value}'s side is labelled by no member")
        return replace(self, label=f"{member}.{self.label}")


SIDE = default_semantics(Side)
"""A ``Side``'s value semantics: a frozen value, compared by value."""


__all__ = ["SIDE", "STATED", "Side", "SideKind"]
