# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's side: what one end of a channel is, as one value.

A ``Channel`` (``finn.kernels.channels``) takes two sides, its ``producer`` and its
``consumer``, each a ``Side`` of one of four kinds (``SideKind``):

- a **kernel**'s port (``Side.kernel``): the stream it presents (``x.stream``), its label
  (``compute.packed.x``, the label its stream pins carry) and, on an output port, its word
  width (``x.width``), read from its stream. Each port states it (``x.side``; a served
  port's, ``x_side``, is its child's port's, labelled below the child: ``port_side`` in
  ``finn.kernels.port`` finds either);
- a **frozen neighbour**'s (``Side.neighbour``): the same three, given as a fact read from a
  kernel outside the selection at its accepted point;
- the **boundary**'s (``Side.boundary``): the module's AXIS port, its name as its label
  (``s_axis_0``, ``m_axis_0``), the movers a shell offers on it (``offers``), and what the
  outside states of it (``outside``). Its ``stream`` is None: the channel derives what the
  port presents (``Channel.boundary``). **Unstated** (``outside`` None, the default), that
  is what its other side presents; **stated** (an ``Outside``), the stated order, passes
  and lanes, each one left None as the other side presents it. A side offered a mover
  carries that mover's statement (``MoverOffer.states``), and a mover's statement is made
  only on a side offered that mover: a mover is never placed on a side it does not state,
  and the build states only a side offered none;
- a valued channel's **source** (``Side.source``): the channel's own memory, which produces
  its known value.

Each kind is made by its own constructor, so a side is whole or not made: a stream comes
with its label, a boundary with its port's name, a source with nothing. On a stated side
(a kernel's or a frozen neighbour's), a word width says it is an output port's: a
channel's producer side carries one, its consumer side none.

A side is a value class: frozen, compared and hashed field by field. So are the values the
outside states (``Outside``) and its pace (``BeatTimes``, or the steady shorthand
``Steady``).

**The outside's statement** (``Outside``) is made by its ``source``: a mover a shell offers
on the boundary (a ``MOVERS`` kind, ``iodma_hls``), or the build (``BUILD``). It states
the order of each pass (``ROW_MAJOR``, or None: as the inside presents it), the passes a
frame (1, or None: as presented), the lanes a beat (the edge's word, or None: the inside's),
the pace (the outside's beat times over a frame, or None when timing is not stated; a
mover's pace comes from its contract) and the host's buffer layout (``ROW_MAJOR_HOST``). The pace is
kept as given: ``Steady`` is expanded where the edge's beats are known, in one place
(``beat_times``), which FIFO sizing (``finn.kernels.fifo_sizing.stated_pattern``) and a
boundary channel's cycles (``finn.kernels.channels.Channel.cycles``: no fewer than its
span) read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

from finn.dataflow.streams import Stream
from finn.kernels.movers import MOVERS, MoverOffer


class SideKind(Enum):
    """What a channel's side is: a kernel's port, a frozen neighbour's, the boundary's, or
    the channel's own source."""

    KERNEL = "kernel"
    NEIGHBOUR = "neighbour"
    BOUNDARY = "boundary"
    SOURCE = "source"


STATED = (SideKind.KERNEL, SideKind.NEIGHBOUR)
"""The kinds that state their stream."""


BUILD = "build"
"""The source of a statement the build makes for a port (an ``ip`` shell's)."""

ROW_MAJOR = "row_major"
"""An order the outside states: each pass presents each position once, in the tensor's
row-major order, beat after beat and lane zero first (``Traversal.row_major``)."""

ROW_MAJOR_HOST = "row_major_host"
"""The host buffer's layout: the stream's beats in order, each padded to whole bytes (the
PYNQ driver's packing). The one layout yet."""


def _count(value: object) -> bool:
    """Whether ``value`` is a count: an ``int``, not a ``bool``."""
    return type(value) is int


@dataclass(frozen=True)
class BeatTimes:
    """The outside's beat times over a frame: each beat's cycle (``times``, from the
    frame's start) and the frame's span (``span``, cycles). At least one beat, at most one
    a cycle (``times`` strictly ascending), the first at cycle 0 or later, each within the
    span (``times[-1] < span``). One time a beat at the edge: FIFO sizing reads it where
    the edge presents as many beats a frame (``finn.kernels.fifo_sizing.stated_pattern``)."""

    times: tuple[int, ...]
    span: int

    def __post_init__(self) -> None:
        times = self.times
        if type(times) is not tuple or not times or not all(_count(time) for time in times):
            raise ValueError("a pace's beat times are a tuple of at least one cycle")
        if not _count(self.span) or self.span < 1:
            raise ValueError(f"a frame spans at least one cycle, not {self.span!r}")
        if times[0] < 0 or any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("a pace's beat times start at cycle 0 or later, one beat a cycle")
        if times[-1] >= self.span:
            raise ValueError(
                f"a pace's beat times lie within its span: a beat at {times[-1]} of {self.span}"
            )


@dataclass(frozen=True)
class Steady:
    """The steady shorthand: a beat every ``every`` cycles (at least 1). Kept as given; it
    is expanded to ``BeatTimes((0, k, ..., (B - 1) k), B k)`` at the edge's beats ``B``,
    in one place (``beat_times``)."""

    every: int

    def __post_init__(self) -> None:
        if not _count(self.every) or self.every < 1:
            raise ValueError(f"a steady beat comes every cycle or more, not every {self.every!r}")


def beat_times(pace: BeatTimes | Steady, beats: int) -> BeatTimes:
    """The beat times of a pace the outside states (``Outside.pace``) at an edge of
    ``beats`` beats a frame: beat times as stated; a steady beat every ``k`` cycles
    expanded to ``(0, k, ..., (beats - 1) k)`` over ``beats * k``. This is the one place a
    ``Steady`` is expanded: ``Side.outside``, ``interface.json`` and any report keep the
    shorthand as given. FIFO sizing reads it (``finn.kernels.fifo_sizing.stated_pattern``),
    and a boundary channel's cycles its span (``finn.kernels.channels.Channel.cycles``)."""
    if isinstance(pace, Steady):
        every = pace.every
        return BeatTimes(tuple(range(0, beats * every, every)), beats * every)
    return pace


@dataclass(frozen=True)
class Outside:
    """What the outside states of a boundary side (the module docstring): its ``source``
    (a ``MOVERS`` kind, or ``BUILD``), the ``order`` of each pass (``ROW_MAJOR``), the
    ``passes`` a frame (1), the ``lanes`` a beat (at least 1), the ``pace``
    (``BeatTimes`` or ``Steady``) and the host buffer's ``layout`` (``ROW_MAJOR_HOST``).
    ``order``, ``passes`` and ``lanes`` left None are as the inside presents them; the
    ``pace`` left None leaves timing unstated; a mover's pace comes from its contract."""

    source: str
    order: str | None = None
    passes: int | None = None
    lanes: int | None = None
    pace: BeatTimes | Steady | None = None
    layout: str = ROW_MAJOR_HOST

    def __post_init__(self) -> None:
        if self.source != BUILD and self.source not in MOVERS:
            raise ValueError(
                f"a boundary is stated by a mover ({', '.join(sorted(MOVERS))}) or the "
                f"build, not {self.source!r}"
            )
        if self.order not in (ROW_MAJOR, None):
            raise ValueError(f"the outside states a {ROW_MAJOR} order, not {self.order!r}")
        if self.passes is not None and (not _count(self.passes) or self.passes != 1):
            raise ValueError(f"the outside states one pass a frame, not {self.passes!r}")
        if self.lanes is not None and (not _count(self.lanes) or self.lanes < 1):
            raise ValueError(f"the outside states one lane a beat or more, not {self.lanes!r}")
        if self.pace is not None and not isinstance(self.pace, (BeatTimes, Steady)):
            raise ValueError("the outside's pace is its beat times or a steady beat")
        if self.layout != ROW_MAJOR_HOST:
            raise ValueError(f"the host's buffer is laid out {ROW_MAJOR_HOST}, not {self.layout!r}")


@dataclass(frozen=True)
class Side:
    """One side of a channel (the module docstring): its ``kind``, its ``label`` (a port's
    label, or the boundary's AXIS port's name; a source's is empty), its ``stream`` (a
    stated side's; None on the boundary, which the channel derives, and for a source), its
    word width (``width``: an output port's), the movers a shell offers on the boundary
    (``offers``), and what the outside states of the boundary (``outside``: None, unstated,
    the boundary presents what its other side presents). Made by ``kernel``,
    ``neighbour``, ``boundary`` or ``source``. A boundary offered movers carries the
    statement of one of them (``MoverOffer.states``)."""

    kind: SideKind
    label: str = ""
    stream: Stream | None = None
    width: int | None = None
    offers: tuple[MoverOffer, ...] = ()
    outside: Outside | None = None

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
        if self.kind is not SideKind.BOUNDARY and self.outside is not None:
            raise ValueError("only the boundary's side is stated by the outside")
        outside = self.outside
        if outside is not None and outside.source != BUILD:
            if not any(offer.kind == outside.source for offer in self.offers):
                raise ValueError(
                    f"{self.label} is stated by an {outside.source} mover, and none is offered"
                )
        if self.offers and outside not in [offer.states() for offer in self.offers]:
            kinds = " or ".join(offer.kind for offer in self.offers)
            said = (
                "left unstated"
                if outside is None
                else "stated by the build"
                if outside.source == BUILD
                else f"stated otherwise than its {outside.source} mover states it"
            )
            raise ValueError(
                f"{self.label} is offered an {kinds} mover and {said}: a side offered a "
                "mover carries that mover's statement"
            )
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
    def boundary(
        cls, port: str, offers: tuple[MoverOffer, ...] = (), outside: Outside | None = None
    ) -> Self:
        """The boundary: its AXIS port's name (``port``), the movers a shell offers on it
        (``offers``), and what the outside states of it (``outside``; None, unstated by
        default: the boundary presents what its other side presents). Offered a mover, it
        is stated by that mover (``offer.states()``)."""
        return cls(SideKind.BOUNDARY, port, offers=tuple(offers), outside=outside)

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
        below ``member`` (``first.compute.packed.x``), as ``PortPins.under``."""
        if not self.stated:
            raise ValueError(f"a {self.kind.value}'s side is labelled by no member")
        return replace(self, label=f"{member}.{self.label}")


__all__ = [
    "BUILD",
    "ROW_MAJOR",
    "ROW_MAJOR_HOST",
    "STATED",
    "BeatTimes",
    "Outside",
    "Side",
    "SideKind",
    "Steady",
    "beat_times",
]
