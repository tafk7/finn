# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's FIFO depth, analytically from both sides' beat patterns.

A channel runs ``source -> output adapter -> transport -> input adapter ->
consumer`` (``finn.kernels.channels``); its ``transport`` is ``direct`` or a
FIFO of ``depth`` words of what arrives at it, the pre-replay stream. The depth
a channel needs is read from two patterns over a frame, repeated at the
partition's bottleneck period ``P`` (its ``CYCLES``):

- the **supply**: when each word arrives at the transport, unthrottled. The
  producer's beat times, which its port states with its contract (``Pace``: one
  schedule beat a cycle), carried through the output adapter's width
  conversion (a word leaves once its last element arrived, one a cycle), and the
  producer's span, the beats its kernel takes a frame;
- the **acceptance**: when the input side takes a word. With no input adapter,
  the consumer's beat times (it takes a word on the beat that reads it, no sooner
  than its own spacing). With an ``input_gen``, its buffer (``nest_buffer``, read
  from the RTL): it takes a word while it holds fewer than its capacity unfreed,
  frees words as its nest passes them, and presents each beat once the word it
  reads arrived, at the consumer's spacing.

``simulate`` runs both over ``frames`` frames (``FRAMES``, the model's) with a FIFO of
``depth`` words between them (``0``: direct, a handshake) and gives the producer's
lateness each frame. FinnLib's FIFO, as XSim measures it: a word is presented two cycles
after it is accepted (``FIFO_LATENCY``), and a slot taken is free the cycle after. The
lateness is how far its last word of the frame left after its unthrottled time. A
producer may be late by its idle time a period (``P`` minus its span) without
starting its next frame late; ``keeps`` says whether a FIFO of so many words keeps it
within that, and ``least_depth`` is the least depth that does, by bisection. A kernel's
pipeline latency is a constant shift of a side's times, and the acceptance follows the
supply's data, so they do not change the depth.

``size`` reads a channel's sides, its producer's and its consumer's paces from the
contracts of the ports beside it (its network's ``neighbours``; at a slice's edge, a frozen
neighbour's port's, at its stored choices, as a kernel's), and gives its ``Sized``: a
depth (``0``: direct) and why. The least depth counts the words the FIFO must
hold; the DEPTH proposed is the least whose native storage holds them
(``least_depth_holding``: FinnLib's FIFO is a shift register of five words up to
33, so a DEPTH of two holds up to five). A boundary's free side is paced, in order, by
the pace the outside states of it (``Side.outside``, ``finn.kernels.sides``: its beat
times, or a steady beat expanded at the edge's beats, ``stated_pattern``), else by the
mover the shell places there (``Channel.mover``, ``finn.kernels.movers``): its beat
times over a frame, and its cycles a frame as its span. An unstated boundary without a
mover (the ``ip`` shell: its integrator's pacing), a memory source (paced by its
consumer) and a side no schedule paces are not modelled: direct, with that
reason. So is an adapter
chain other than one ``vpc`` before the transport and one ``input_gen`` after it.
The strategy proposing the depths is ``finn.kernels.explore.SizeFifos``; an rtlsim
check is evidence outside it.

``holds`` checks a channel's transport, a FIFO of so many words or ``direct``, against
its stream again (``keeps``, one simulation): a depth is sized at one bottleneck period,
and a fold anywhere in the network may change the period, or the sides, after it. Nothing
records what it was sized against; the seam checks it where a strategy asks
(``Seam.check_fifos``) and the build once, before emission
(``finn.custom_op.kernels.shell.verify_partition``).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from math import ceil

from finn.kernels.adapters import Convert
from finn.kernels.base import PortContract
from finn.kernels.channels import Channel, Neighbours
from finn.kernels.fifo import least_depth_holding
from finn.kernels.input_generator import Generate, NestBuffer, nest_buffer
from finn.kernels.movers import MoverContract
from finn.kernels.sides import BeatTimes, Side, SideKind, Steady
from finn.kernels.transport import StreamContract


@dataclass(frozen=True)
class Pattern:
    """One side's beat times over a frame (cycles from the frame's first beat) and its
    kernel's span, the beats it takes a frame."""

    times: tuple[int, ...]
    beats: int

    def __post_init__(self) -> None:
        if not self.times or any(b < a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("a side presents at least one beat, in time order")
        if self.beats <= self.times[-1] - self.times[0]:
            raise ValueError("a frame's beats lie within its kernel's span")

    def gaps(self) -> tuple[int, ...]:
        """The least spacing before each beat: to the beat before it, the first to the
        last of the frame before."""
        times = self.times
        first = times[0] + self.beats - times[-1]
        return (first, *(b - a for a, b in zip(times, times[1:])))


@dataclass(frozen=True)
class Buffered:
    """An input adapter's ``input_gen`` (any chain of one ``input_gen``, a replay or
    not): its buffer, and the pattern its consumer reads its output at."""

    buffer: NestBuffer
    reads: Pattern


Acceptance = Pattern | Buffered


def converted(times: Sequence[int], lanes_in: int, lanes_out: int) -> tuple[int, ...]:
    """Word times after a width conversion: a word leaves the cycle after its last
    element arrived, one a cycle."""
    found: list[int] = []
    for word in range(len(times) * lanes_in // lanes_out):
        last = ceil((word + 1) * lanes_out / lanes_in) - 1
        found.append(max(times[last] + 1, found[-1] + 1 if found else 0))
    return tuple(found)


#: The cycles a word takes through FinnLib's FIFO (written, then out through its output
#: register): XSim-measured for its shift-register storage, which every FIFO of 33
#: words or fewer is.
FIFO_LATENCY = 2


@dataclass(frozen=True)
class Flow:
    """Each word's cycle leaving the producer and taken by the input side, and the
    producer's lateness each frame: its last word's leaving after its unthrottled time."""

    left: tuple[int, ...]
    taken: tuple[int, ...]
    late: tuple[int, ...]


def _events(
    buffer: NestBuffer, read_gaps: Sequence[int]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """An input_gen's frame by the events its presentation can wait on: the frame's first
    read of each word (its beat and the word), in beat order; for each freed count ``1 …``
    the most it frees, the first beat by which the frame has freed that many; and the
    reader's gaps summed to each beat (the gaps indexed by the buffer's beat in its frame)."""
    seen: set[int] = set()
    first: list[tuple[int, int]] = []
    for beat, word in enumerate(buffer.reads):
        if word not in seen:
            seen.add(word)
            first.append((beat, word))
    reached: list[int] = []
    for beat, freed in enumerate(buffer.freed):
        while len(reached) < freed:
            reached.append(beat)
    return tuple(first), tuple(reached), tuple(accumulate(read_gaps[: len(buffer.reads)]))


def simulate(
    supply: Pattern, accept: Acceptance, period: int, depth: int, frames: int
) -> Flow | None:
    """``frames`` frames from ``supply``, its frame ``k`` due from ``k * period``,
    through a FIFO of ``depth`` words (0: direct, a handshake) into ``accept``;
    ``None`` when the two sides deadlock.

    An input_gen presents its beats in order, each once the word it reads was taken plus
    two cycles (written, then read through its output register) and no sooner than the
    reader's gap after the beat before. A beat re-reading a word its frame already read is
    never held by that word: the earlier read already waited for it, and no gap is negative.
    So only a frame's first reads can wait on the supply, or deadlock, and between them the
    beats advance by the reader's gaps alone, a prefix sum: the presentation is simulated by
    its first reads and the beats where its freed count rises, not beat by beat."""
    # Each pattern's gaps are read once: a frame at one lane a beat is tens of
    # thousands of words, and a model of 16 frames reads them for every word.
    words, gaps = len(supply.times), supply.gaps()
    accept_gaps = accept.gaps() if isinstance(accept, Pattern) else accept.reads.gaps()
    left: list[int] = []  # when each word left the producer
    taken: list[int] = []  # when the input side took it
    late: list[int] = []
    buffered = accept if isinstance(accept, Buffered) else None
    freed_at: list[int] = []  # the cycle at which the freed count reached each value
    if buffered is not None:
        buffer = buffered.buffer
        frame, frame_words = len(buffer.reads), buffer.freed[-1]
        first, reached, summed = _events(buffer, accept_gaps)
    shown, shown_at = -1, 0  # the last first read presented: its beat and cycle
    reading = 0  # the next first read to present, counted over every frame

    def gaps_to(beat: int) -> int:
        """The reader's gaps summed from the first beat to ``beat``."""
        k, i = divmod(beat, frame)
        return k * summed[-1] + summed[i]

    def present_until(count: int) -> bool:
        """Present beats until ``count`` words are freed; False when a beat waits on a
        word not yet taken."""
        nonlocal shown, shown_at, reading
        while len(freed_at) < count:
            # The beat at which the freed count reaches its next value: frame k's beats free
            # k * frame_words words and the frame's own; the first frame whose most freed
            # reaches the value, at its first beat that does.
            value = len(freed_at) + 1
            k = max(0, -((len(reached) - value) // frame_words))
            until = k * frame + reached[value - k * frame_words - 1]
            while True:
                k, f = divmod(reading, len(first))
                beat, word = first[f]
                beat += k * frame
                if beat > until:
                    break
                word += k * frame_words
                if word >= len(taken):
                    return False
                at = taken[word] + 2  # written, then read through its output register
                if shown >= 0:
                    at = max(at, shown_at + gaps_to(beat) - gaps_to(shown))
                shown, shown_at = beat, at
                reading += 1
            freed_at.append(shown_at + gaps_to(until) - gaps_to(shown))
        return True

    for n in range(frames * words):
        k, j = divmod(n, words)
        ready = k * period + supply.times[j]
        if left:
            ready = max(ready, left[-1] + gaps[j])
        if depth and n >= depth:
            # A slot frees the cycle after its word is taken (the FIFO's ready is a register).
            ready = max(ready, taken[n - depth] + 1)
        at = ready + (FIFO_LATENCY if depth else 0)
        if taken:
            at = max(at, taken[-1] + 1)
        if buffered is None:
            assert isinstance(accept, Pattern)
            reads = accept_gaps[n % len(accept.times)]
            if taken:
                at = max(at, taken[-1] + reads)
        else:
            room = n - buffered.buffer.capacity + 1  # words that must be freed first
            if room > 0:
                if not present_until(room):
                    return None
                at = max(at, freed_at[room - 1] + 1)
        taken.append(at)
        left.append(at if not depth else ready)
        if j == words - 1:
            late.append(left[-1] - (k * period + supply.times[-1]))
    return Flow(tuple(left), tuple(taken), tuple(late))


#: The frames the model runs to reach its periodic state: sizing's and the check's, so a
#: depth sizing proposes is one the check keeps.
FRAMES = 16


def within(late: Sequence[int], supply: Pattern, period: int) -> bool:
    """The producer starts every frame on time: late by no more than its idle time."""
    return all(each <= period - supply.beats for each in late)


def keeps(
    supply: Pattern, accept: Acceptance, period: int, words: int, frames: int = FRAMES
) -> bool:
    """A FIFO of ``words`` words (0: direct) keeps the producer, throttled only by a full
    FIFO, within its idle time a period over ``frames`` frames (``within``); the two
    sides deadlocking do not."""
    flow = simulate(supply, accept, period, words, frames)
    return flow is not None and within(flow.late, supply, period)


def least_depth(
    supply: Pattern, accept: Acceptance, period: int, frames: int = FRAMES
) -> int | None:
    """The least FIFO depth (0: direct) for which the producer, throttled only by a full
    FIFO, is never later than its idle time a period (``keeps``); ``None`` when none up to
    ``frames`` frames of words is."""
    if supply.beats > period:
        raise ValueError(f"a producer of {supply.beats} beats cannot keep a period of {period}")

    def meets(depth: int) -> bool:
        return keeps(supply, accept, period, depth, frames)

    if meets(0):
        return 0
    upper = frames * len(supply.times)
    if not meets(upper):
        return None
    depths = range(1, upper + 1)
    return depths[bisect_left(depths, True, key=meets)]


# -- a channel's sides ---------------------------------------------------------------------


class NotModelled(ValueError):
    """A channel whose sides the model does not read, and why."""


def _pattern(port: PortContract | None, side: str) -> Pattern:
    """The pace the port beside the channel on ``side`` states with its contract."""
    if port is None:
        raise ValueError(f"the channel's {side} is a port, and none is given beside it")
    pace = port.contract.pace
    if pace is None:
        raise NotModelled(f"the {side} presents a given sequence, which no schedule paces")
    return Pattern(pace.times, pace.span)


def stated_pattern(pace: BeatTimes | Steady, beats: int) -> Pattern:
    """The ``Pattern`` of a pace the outside states (``Outside.pace``) at an edge of
    ``beats`` beats a frame: its beat times and its span; a steady beat every ``k`` cycles
    expanded to ``(0, k, ..., (beats - 1) k)`` over ``beats * k``. This is the one place a
    ``Steady`` is expanded: everything else keeps the shorthand as given (``Side.outside``,
    ``interface.json`` and any report). ``NotModelled`` where the stated beat times are not
    one a beat of the edge's."""
    if isinstance(pace, Steady):
        every = pace.every
        return Pattern(tuple(range(0, beats * every, every)), beats * every)
    if len(pace.times) != beats:
        raise NotModelled(
            f"a boundary's stated pace times {len(pace.times)} beats a frame, and its edge "
            f"presents {beats}"
        )
    return Pattern(pace.times, pace.span)


def _free(channel: Channel, side: Side) -> Pattern:
    """A boundary's free side (``side``), paced, in order: by the pace the outside states
    (``stated_pattern``, at the beats the edge presents a frame); else by its mover, its
    beat times and its cycles a frame; else, an unstated side without a mover, not
    modelled ("a boundary: not modelled")."""
    outside = side.outside
    if outside is not None and outside.pace is not None:
        edge: StreamContract | None = channel.boundary
        assert edge is not None  # a channel with a boundary side presents its port
        return stated_pattern(outside.pace, edge.form.beats)
    if not channel.offered:
        raise NotModelled("a boundary: not modelled")
    contract: MoverContract = channel.mover_contract
    return Pattern(contract.times, contract.cycles)


def sides(channel: Channel, neighbours: Neighbours) -> tuple[Pattern, Acceptance]:
    """A channel's supply at its transport and its input side's acceptance, from the
    paces the ports beside it state (``neighbours``, as its network reads them); a
    boundary side's from the pace the outside states, else from its mover (``_free``; a
    stated output's pace is the acceptance, as a consumer's port's); ``NotModelled``
    naming why the model does not read them."""
    if channel.valued:
        raise NotModelled("a memory source: paced by its consumer")
    found = channel.sides
    producer = (
        _free(channel, found.producer)
        if found.producer.kind is SideKind.BOUNDARY
        else _pattern(neighbours.producer, "producer")
    )
    consumer = (
        _free(channel, found.consumer)
        if found.consumer.kind is SideKind.BOUNDARY
        else _pattern(neighbours.consumer, "consumer")
    )
    times = producer.times
    if channel.output_adapting:
        stages = channel.output_adapter.realization
        modules = [stage.module for stage in stages]
        if len(modules) != 1 or not isinstance(modules[0], Convert):
            kinds = " -> ".join(stage.kind for stage in stages)
            raise NotModelled(f"an output adapter of {kinds}: not modelled")
        times = converted(times, modules[0].lanes_in, modules[0].lanes_out)
    # A width conversion sending more words than its producer has beats spans them: the
    # producer and its conversion are one source, as slow as the slower of the two.
    supply = Pattern(times, max(producer.beats, times[-1] - times[0] + 1))
    adapter = channel.adapter
    if adapter is None:
        return supply, consumer
    stages = adapter.realization
    if len(stages) != 1 or not isinstance(stages[0].module, Generate):
        kinds = " -> ".join(stage.kind for stage in stages)
        raise NotModelled(f"an input adapter of {kinds}: not modelled")
    module = stages[0].module
    return supply, Buffered(nest_buffer(module.frame, module.dims, module.coefs), consumer)


@dataclass(frozen=True)
class Sized:
    """One channel's proposal and why: ``depth`` 0 is ``direct``. ``least`` is the
    least number of words a FIFO must hold that the model found (``None`` where it
    found none or did not read the sides); ``word_bits`` the bits of a word at the
    transport."""

    depth: int
    why: str
    word_bits: int
    least: int | None = None

    @property
    def bits(self) -> int:
        """The FIFO's bits: its depth of words."""
        return self.depth * self.word_bits


def size(
    channel: Channel,
    neighbours: Neighbours,
    period: int,
    *,
    margin: int = 0,
    ram_style: str = "auto",
) -> Sized:
    """``channel``'s FIFO at the bottleneck ``period``, between the ports beside it
    (``neighbours``): the least DEPTH whose storage holds the least words plus ``margin``
    (``least_depth_holding``: a FIFO of two words or more), or direct where none is
    needed, none helps, or the sides are not modelled."""
    word_bits = channel.arriving.form.lanes * channel.tensor.element.bits
    try:
        supply, accept = sides(channel, neighbours)
    except NotModelled as error:
        return Sized(0, str(error), word_bits)
    least = least_depth(supply, accept, period)
    if least is None:
        return Sized(0, f"no depth up to {FRAMES} frames keeps the producer's period", word_bits)
    if least == 0:
        return Sized(0, "direct absorbs it", word_bits, 0)
    depth = least_depth_holding(least + margin, word_bits, ram_style, channel.platform.uram)
    return Sized(depth, f"least {least} words", word_bits, least)


def holds(channel: Channel, neighbours: Neighbours, period: int, words: int) -> bool:
    """Whether ``channel``'s transport, a FIFO of ``words`` words (its storage's capacity;
    0: direct), between the ports beside it (``neighbours``), keeps its producer within
    its idle time at the bottleneck ``period`` (``keeps``, one simulation of its sides):
    the test ``size`` sizes it by, at the words it holds. ``NotModelled`` where the model
    does not read its sides (``sides``)."""
    supply, accept = sides(channel, neighbours)
    return keeps(supply, accept, period, words)


__all__ = [
    "FRAMES",
    "Acceptance",
    "Flow",
    "NotModelled",
    "Pattern",
    "Buffered",
    "Sized",
    "converted",
    "holds",
    "keeps",
    "least_depth",
    "sides",
    "simulate",
    "size",
    "stated_pattern",
]
