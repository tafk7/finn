# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's FIFO depth, analytically from both ends' beat patterns (K12, DSE13, FS1).

A channel runs ``source -> output adapter -> transport -> input adapter ->
consumer`` (``finn.kernels.channels``); its ``transport`` is ``direct`` or a
FIFO of ``depth`` words of what arrives at it, the pre-replay stream. The depth
a channel needs is read from two patterns over a frame, repeated at the
partition's bottleneck period ``P`` (its ``CYCLES``):

- the **supply**: when each word arrives at the transport, unthrottled. The
  producer's beat times, which its port states with its contract (``Pace``: one
  schedule beat a cycle, K10), carried through the output adapter's width
  conversion (a word leaves once its last element arrived, one a cycle), and the
  producer's span, the beats its kernel takes a frame;
- the **acceptance**: when the input side takes a word. With no input adapter,
  the consumer's beat times (it takes a word on the beat that reads it, no sooner
  than its own spacing). With an ``input_gen``, its buffer (``nest_buffer``, read
  from the RTL): it takes a word while it holds fewer than its capacity unfreed,
  frees words as its nest passes them, and presents each beat once the word it
  reads arrived, at the consumer's spacing.

``simulate`` runs both over ``frames`` frames with a FIFO of ``depth`` words
between them (``0``: direct, a handshake) and gives the producer's lateness each
frame. FinnLib's FIFO, as XSim measures it: a word is presented two cycles after
it is accepted (``FIFO_LATENCY``), and a slot taken is free the cycle after. The
lateness is how far its last word of the frame left after its unthrottled time. A
producer may be late by its idle time a period (``P`` minus its span) without
starting its next frame late; ``least_depth`` is the least depth for which it
never is later than that (FS1), by bisection. A kernel's pipeline latency is a constant
shift of an end's times, and the acceptance follows the supply's data, so they do
not change the depth.

``size`` reads a channel's ends from its contracts and gives its ``Sized``: a
depth (``0``: direct) and why. The least depth counts the words the FIFO must
hold; the DEPTH proposed is the least whose native storage holds them
(``least_depth_holding``: FinnLib's FIFO is a shift register of five words up to
33, so a DEPTH of two holds up to five). A boundary's free side is paced by the
end the shell places there (``Channel.end``, ``finn.kernels.ends``): its beat
times over a frame, and its cycles a frame as its span. A boundary without an
end (the ``ip`` shell: its integrator's pacing), a memory source (paced by its
consumer) and an end no schedule paces are not modelled: direct, with that
reason. So is an adapter
chain other than one ``vpc`` before the transport and one ``input_gen`` after it.
The strategy proposing the depths is ``finn.kernels.explore.SizeFifos``; the rtlsim
check is evidence outside it (K12).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from finn.kernels.adapters import Convert, Generate
from finn.kernels.channels import Channel
from finn.kernels.ends import EndContract
from finn.kernels.fifo import least_depth_holding
from finn.kernels.input_generator import NestBuffer, nest_buffer
from finn.kernels.transport import StreamContract


@dataclass(frozen=True)
class Pattern:
    """One end's beat times over a frame (cycles from the frame's first beat) and its
    kernel's span, the beats it takes a frame."""

    times: tuple[int, ...]
    beats: int

    def __post_init__(self) -> None:
        if not self.times or any(b < a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("an end presents at least one beat, in time order")
        if self.beats <= self.times[-1] - self.times[0]:
            raise ValueError("a frame's beats lie within its kernel's span")

    def gaps(self) -> tuple[int, ...]:
        """The least spacing before each beat: to the beat before it, the first to the
        last of the frame before."""
        times = self.times
        first = times[0] + self.beats - times[-1]
        return (first, *(b - a for a, b in zip(times, times[1:])))


@dataclass(frozen=True)
class Replay:
    """An input adapter's ``input_gen``: its buffer, and the pattern its consumer reads
    its output at."""

    buffer: NestBuffer
    reads: Pattern


Acceptance = Pattern | Replay


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


def simulate(
    supply: Pattern, accept: Acceptance, period: int, depth: int, frames: int
) -> Flow | None:
    """``frames`` frames from ``supply``, its frame ``k`` due from ``k * period``,
    through a FIFO of ``depth`` words (0: direct, a handshake) into ``accept``;
    ``None`` when the ends deadlock."""
    # Each pattern's gaps are read once: a frame at one lane a beat is tens of
    # thousands of words, and a model of 16 frames reads them for every word.
    words, gaps = len(supply.times), supply.gaps()
    accept_gaps = accept.gaps() if isinstance(accept, Pattern) else accept.reads.gaps()
    left: list[int] = []  # when each word left the producer
    taken: list[int] = []  # when the input side took it
    late: list[int] = []
    # An input_gen's presented beats, when each was presented, and its freed count.
    replay = accept if isinstance(accept, Replay) else None
    shown: list[int] = []
    freed_at: list[int] = []  # the cycle at which the freed count reached each value
    freed_count = 0

    def present_until(count: int) -> bool:
        """Present beats until ``count`` words are freed; False when a beat waits on a
        word not yet taken."""
        nonlocal freed_count
        assert replay is not None
        buffer, frame = replay.buffer, len(replay.buffer.reads)
        frame_words, read_gaps = buffer.freed[-1], accept_gaps
        while freed_count < count:
            beat = len(shown)
            k, i = divmod(beat, frame)
            word = k * frame_words + buffer.reads[i]
            if word >= len(taken):
                return False
            at = taken[word] + 2  # written, then read through its output register
            if shown:
                at = max(at, shown[-1] + read_gaps[i])
            shown.append(at)
            now = k * frame_words + buffer.freed[i]
            while freed_count < now:
                freed_at.append(at)
                freed_count += 1
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
        if replay is None:
            assert isinstance(accept, Pattern)
            reads = accept_gaps[n % len(accept.times)]
            if taken:
                at = max(at, taken[-1] + reads)
        else:
            room = n - replay.buffer.capacity + 1  # words that must be freed first
            if room > 0:
                if not present_until(room):
                    return None
                at = max(at, freed_at[room - 1] + 1)
        taken.append(at)
        left.append(at if not depth else ready)
        if j == words - 1:
            late.append(left[-1] - (k * period + supply.times[-1]))
    return Flow(tuple(left), tuple(taken), tuple(late))


def within(late: Sequence[int], supply: Pattern, period: int) -> bool:
    """The producer starts every frame on time: late by no more than its idle time."""
    return all(each <= period - supply.beats for each in late)


def least_depth(supply: Pattern, accept: Acceptance, period: int, frames: int = 16) -> int | None:
    """The least FIFO depth (0: direct) for which the producer, throttled only by a full
    FIFO, is never later than its idle time a period; ``None`` when none up to
    ``frames`` frames of words is."""
    if supply.beats > period:
        raise ValueError(f"a producer of {supply.beats} beats cannot keep a period of {period}")

    def meets(depth: int) -> bool:
        flow = simulate(supply, accept, period, depth, frames)
        return flow is not None and within(flow.late, supply, period)

    if meets(0):
        return 0
    upper = frames * len(supply.times)
    if not meets(upper):
        return None
    depths = range(1, upper + 1)
    return depths[bisect_left(depths, True, key=meets)]


# -- a channel's ends ----------------------------------------------------------------------


class NotModelled(ValueError):
    """A channel whose ends the model does not read, and why."""


def _pattern(contract: StreamContract, side: str) -> Pattern:
    pace = contract.pace
    if pace is None:
        raise NotModelled(f"the {side} presents a given sequence, which no schedule paces")
    return Pattern(pace.times, pace.span)


def _free(channel: Channel) -> Pattern:
    """A boundary's free side, paced by its end: its beat times and its cycles a frame."""
    if not channel.ended:
        raise NotModelled("a boundary: not modelled")
    contract: EndContract = channel.end_contract
    return Pattern(contract.times, contract.cycles)


def ends(channel: Channel) -> tuple[Pattern, Acceptance]:
    """A channel's supply at its transport and its input side's acceptance, from the
    paces its ends state; ``NotModelled`` naming why the model does not read them."""
    if channel.valued:
        raise NotModelled("a memory source: paced by its consumer")
    found = channel.endpoints
    producer = _free(channel) if found.source_owner is None else _pattern(found.source, "producer")
    consumer = _free(channel) if found.sink_owner is None else _pattern(found.sink, "consumer")
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
    if not channel.adapting:
        return supply, consumer
    stages = channel.adapter.realization
    if len(stages) != 1 or not isinstance(stages[0].module, Generate):
        kinds = " -> ".join(stage.kind for stage in stages)
        raise NotModelled(f"an input adapter of {kinds}: not modelled")
    module = stages[0].module
    return supply, Replay(nest_buffer(module.frame, module.dims, module.coefs), consumer)


@dataclass(frozen=True)
class Sized:
    """One channel's proposal and why: ``depth`` 0 is ``direct``. ``least`` is the
    least number of words a FIFO must hold that the model found (``None`` where it
    found none or did not read the ends); ``word_bits`` the bits of a word at the
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
    period: int,
    *,
    margin: int = 0,
    ram_style: str = "auto",
    frames: int = 16,
) -> Sized:
    """``channel``'s FIFO at the bottleneck ``period``: the least DEPTH whose storage
    holds the least words plus ``margin`` (``least_depth_holding``: a FIFO of two words
    or more), or direct where none is needed, none helps, or the ends are not
    modelled."""
    word_bits = channel.arriving.form.lanes * channel.tensor.element.bits
    try:
        supply, accept = ends(channel)
    except NotModelled as error:
        return Sized(0, str(error), word_bits)
    least = least_depth(supply, accept, period, frames)
    if least is None:
        return Sized(0, f"no depth up to {frames} frames keeps the producer's period", word_bits)
    if least == 0:
        return Sized(0, "direct absorbs it", word_bits, 0)
    depth = least_depth_holding(least + margin, word_bits, ram_style, channel.platform.uram)
    return Sized(depth, f"least {least} words", word_bits, least)


__all__ = [
    "Acceptance",
    "Flow",
    "NotModelled",
    "Pattern",
    "Replay",
    "Sized",
    "converted",
    "ends",
    "least_depth",
    "simulate",
    "size",
    "within",
]
