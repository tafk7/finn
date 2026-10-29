# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The time of a whole network of stages: when each word moves, and the channel
capacities that keep the network's period.

**The network** (``Network``) is a value: its stages and the channels between them, each
channel from one stage's output port to another's input port, of a ``capacity`` in words
(``None``: unbounded). The stages:

- ``Source``: its beat times over a frame, its span (the cycles it takes a frame) and the
  ``period`` its frames are released at: frame ``k``'s word ``j`` is due at
  ``k * period + times[j]``, and a source held back resumes at its own spacing, as a
  producer does in the line's per-channel model (``finn.kernels.fifo_sizing.simulate``).
- ``Memory``: a memory source, paced by its consumer: it presents its next word whenever
  its channel has room, one a cycle. It has no period and is never late.
- ``Kernel``: its iterations a frame, the iterations at which each input port takes a word and each
  output port writes one, and a stated ``latency``. It takes one iteration a cycle, from cycle
  0, and an iteration that moves a word moves only when every word it reads is there and every
  channel it writes has room (``kernel`` builds one from its ports' ``Pace``s, so its pace
  is its schedule's ``beat_times``). A write is visible ``latency`` cycles later than the
  channel alone would make it. On a FIFO or a register the word holds its slot from its
  iteration: the pipeline's words in flight count in the channel it writes. A handshake holds
  no word, so there the pipeline's own ``latency`` registers hold them: the kernel issues
  a word while fewer than ``latency`` of its words on it are untaken, a register free in
  the cycle its word is taken.
- ``Fork``: each word to every branch at once, one a cycle (FINN's DuplicateStreams); it
  holds no buffer.
- ``Sink``: takes a word a cycle, whenever one is there.
- The channel's steps (``steps.Steps``, ``reindex.Route``), each a stage of its own beat
  pattern (``step_stages``, ``route_stages``):
  - ``WidthStage``: the same elements at another number of lanes a beat. A word leaves the
    cycle after its last element arrived, one a cycle (the line's ``converted``); it takes
    input word ``i`` once every output word made only of input words before ``i - 1`` has
    left, so it holds at most what one output word spans and the input word after it.
  - ``ReorderStage``: an input generator. It takes its frame's words in order while fewer
    than its ``capacity`` taken words are unfreed, presents beat ``b`` (frame word
    ``reads[b]``) once that word was taken (``Rules.reorder_latency`` cycles before), and
    has freed ``freed[b]`` words of the frame after beat ``b``. ``reorder`` derives the
    frees from the reads and takes ``buffers.live_window`` as the default capacity; a
    caller's own buffer (the RTL's) is passed as it is.
  - ``InsertionStage`` and ``DropStage``: an in-order insertion of fills, or a drop of
    positions, one beat a cycle, holding no buffer.
  - ``MergeStage``: a merge, reading its sources on the beats ``reindex.pattern`` names, at
    ``reindex.merge_lanes`` lanes (``merge_stage``), one beat a cycle.

**The channel's rules are parameters** (``Rules``): a FIFO's latency (a word visible so
many cycles after it is written); what a direct connection is (``Direct``: a handshake,
the producer's write and the consumer's take one event, or a one-word register, visible
the cycle after it is written); the cycles after its word is taken that a slot is free
(also the cycles after a reorder's beat that the words it frees are free); and the cycles
after taking a word that a reorder presents it. ``FINNLIB`` is FinnLib's, as the line's
per-channel model states them; ``ONE_CYCLE`` is every word visible the cycle after it is
written and a direct connection a one-word register. Capacity ``Rules.direct_capacity``
(``0`` for a handshake, ``1`` for a register) is the direct connection; any larger one, or
``None``, is a FIFO.

**The run** (``run``) is an event model, per word, not cycle by cycle: each stage's moves
(a firing: the words it takes and writes in one cycle) are timed in order, each at the
least cycle every rule allows: its own spacing; the words it reads visible; room in the
channels it writes; and its own buffer's rules (a reorder's frees, a width step's
elements). A handshake joins the writer's and the reader's moves into one event, at the
latest of both sides' bounds: a fork whose branches rejoin through handshakes alone moves
only when every branch can (a deadlock where a join waits on its other branch). Every rule
is a lower bound on one move by an earlier move, so the times are the least solution of
those bounds, whatever order the moves are timed in. A run gives each stage's frame
completions (its last move of each frame), each source's lateness a frame, every word's
write and read cycle on every channel, and each channel's peak: the most words it held
before a write, plus one (the least capacity that would have refused no write of the
run). A word is held until its slot is free: ``Rules.slot_free`` cycles after it is read
(none for a pipeline's register on a handshake). It **deadlocks** when no stage can move
while one has frames left.

**Keeps the period.** A run keeps the period ``P`` (``period_kept``, the rule ``least`` and
``minimal`` apply unless given another as ``keeps``) when both halves hold:

1. *The sources' half* (``Run.on_time``): it does not deadlock and every source starts every
   frame on time: over the frames run (``FRAMES``, 16, a parameter), each source's frame ``k``
   ends no later than ``k * P + times[-1]`` (its unthrottled end) plus its idle time
   ``P - span``. No frame is set aside as warm-up: frame ``k`` is due ``k * P`` after the
   first, and a late frame is late however early it falls. A memory source is never late.
2. *The ends' half* (``Due``): every end (a stage with no output port: a ``Sink``, a kernel
   that writes nothing) completes every frame ``k`` of the frames run by ``k * P + by[end]``,
   where ``by[end]`` is the most cycles any frame of that end completes after ``k * P`` in the
   network's unbounded run (every channel unbounded, the same frames, the same rules), plus a
   slack: by default the most idle time of the sources the end is reached from (0 where it is
   reached from none, a memory branch). The due cycles are read once from the unbounded run,
   a property of the network; a run at the capacities judged is held to them, never to a run
   with every channel unbounded. The worst frame of the unbounded run is the bound, not its
   first: an end's lateness there is not always constant.

The sources' half alone does not see a stage slower than the period behind a channel or a
reorder with room for the backlog: it pushes back to no source, so the network runs slower
than ``P`` and every source stays on time. The ends' half reads that loss at the end.

The **bottleneck period** is the period of the run with every channel unbounded, read at the
network's ends (the cycles between their last two frame completions). ``minimal`` and
``least`` require it to be the sources' ``P`` whatever ``keeps`` is: a network slower than its
sources is refused before any criterion runs (release them at its bottleneck), and so is one
that deadlocks unbounded, where there are no due cycles to read. It judges no sized set: it
always runs with every channel unbounded.

*Against the line's ``within``, on a chain of two stages* (a source, one channel, a
kernel; or a source, one channel, a reorder and its kernel): the run's bounds are the
line's ``simulate``'s, word for word. The source's word ``n`` leaves at the latest of its
due cycle, its previous word plus its gap, and (a FIFO of ``d`` words) the take of word
``n - d`` plus one; the consumer takes it at the latest of its leaving plus the FIFO's
latency (``0``, the same event, for a handshake) and its previous take plus its own gap;
a reorder takes word ``n`` once its beats have freed ``n - capacity + 1`` words, plus one,
and presents a beat ``reorder_latency`` cycles after its word was taken, at its reader's
gaps. Under ``FINNLIB`` those are the line's numbers, and ``Run.on_time`` is ``within`` on
the same lateness over the same frames, so the two agree at every depth. (The line's
consumer has no phase: a kernel here starts at cycle 0, so the two agree where its first
read is at iteration 0.) On a source, a FIFO and a kernel whose bottleneck is ``P``, the default
criterion is ``within`` too: with a slack at least the source's idle time, the sources' half
implies the ends'. The default differs from ``within`` in two places, on purpose:

- a stage slower than the period: one word a frame at a period of 1 cycle, into a consumer
  taking a word every 2 cycles. The line gives 9 words, ``within`` passes the run at 9 (the
  FIFO takes the backlog for the 16 frames, the consumer a cycle further behind each frame),
  and the default refuses the network, its bottleneck being 2 against 1;
- a reorder whose output channel is direct, as the network states it: ``bottleneck`` reads
  the network with that channel unbounded too and may see ``P`` where, with the output direct,
  the reorder and its reader run slower. The default refuses every set (``least`` is None) where
  the line and the sources' half accept some depth. This is the model's claim about a
  generator whose output is direct; the hardware settles it.

On a longer network they differ also because only the sources and the ends are held to the
period: a kernel's own frame may slip within the slack its neighbours leave, where per-channel
sizing releases every producer at ``k * P`` and reads every consumer unthrottled.

What the criterion does not see, here as in ``within``: a steady period ``P' > P`` shows only
once a lateness growing by ``P' - P`` a frame passes its allowance within the frames run. A
source's allowance is its idle time. An end that runs at ``P + d`` from its frame 0 completes
frame ``k`` about ``L0 + k * d`` after ``k * P`` (``L0``: its first frame's lateness), and is
refused when ``L0 + (F - 1) * d`` passes ``by[end]`` (``M + s``: the unbounded run's latest
lateness and the slack), over ``F`` frames; with ``L0 = M`` that is ``s // (F - 1) + 1`` cycles a
frame, the sources' limit under the default slack. A smaller loss is not seen; the frames are a
parameter for it. The ends' allowance is also a latency one: a constant lateness within
``by[end]`` is accepted, one beyond it is refused as a loss (no criterion over a finite run can
accept every constant latency and refuse every loss). ``slack`` is the parameter for the other
trade.

**Sizing.** ``least`` is the least capacity of one channel keeping the period with every
other channel as the network states it; ``minimal`` finds a minimal set jointly:

1. the unbounded run's peaks: at a peak above direct a channel refuses no write of the
   run, so with every channel at its peak the run is the unbounded run itself. A peak at
   direct is a direct connection, whose rules differ from a FIFO's, so the run at it is
   its own run, refused by name where it does not keep the period;
2. each channel in a fixed order (``order``: from the sources down, each stage's output
   channels once the stage's inputs are placed) lowered to the least capacity keeping the
   period with the others as they stand;
3. passes repeated until none moves.

Each channel is lowered by trying direct first, then bisecting the FIFO capacities above
it, up to its current capacity. Bisection rests on **monotonicity**: with the others
fixed, more words in one channel never lose the period. Among FIFO capacities it holds by
construction, for both halves of the criterion: a capacity is one bound, "write ``w`` no
sooner than the take of word ``w - C`` plus the slot's delay", and a larger ``C`` bounds by
an earlier take; every other rule stays as it was, and the least solution of such bounds is
monotone in each of them, so no move is later, no source later, no end's frame completion
later, and a run that ends still ends. The due cycles do not move: they are read from the
network's unbounded run, never from the capacities judged. (A reorder's capacity is its
stage's, part of the network: changing it moves the due cycles.) It fails between direct
and the least FIFO, whose rules differ: a handshake moves a word in the cycles a FIFO's
latency spends (and a kernel's pipeline holds its own words there), a register's latency is
one cycle whatever the FIFO's. Under a handshake, direct against one word is ordered neither
way: a direct connection can keep the period where one word does not, and one word can keep
it where a direct connection does not. Direct is therefore tried apart, a scan of that one
value, as the line's ``least_depth`` tries ``0`` apart, and the FIFO capacities above it are
bisected among themselves, where they are monotone. (Under ``ONE_CYCLE``, direct is the
one-word register, which with a FIFO latency of 1 is the channel of a FIFO of one word: the
separate scan is redundant there, and harmless.)

*Termination:* every lowering only lowers a capacity, and every capacity it lowers to keeps
the period (it was run: the capacity it lowers from first, refused if it does not keep
it, then the one it lowers to), so the capacities fall in every pass that does not end
the search, and a pass that moves none ends it. The result is **minimal**, not least: lowering
any one channel by one word loses the period or deadlocks, but another order can end at
another set (a reorder's capacity and the FIFO before it are one budget), and the total is
not minimized. ``Sizing`` states its order.
"""

from __future__ import annotations

import numpy as np
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from math import ceil

from finn.dataflow_next.buffers import frame_slots, live_window
from finn.dataflow_next.maps import Drop, Insertion
from finn.dataflow_next.reindex import (
    Forked,
    Merge,
    Reindex,
    joined,
    merge_lanes,
    merge_stage_cost,
    pattern,
    route,
    stage_beats,
    stage_lanes,
)
from finn.dataflow_next.schedule import Pace
from finn.dataflow_next.steps import Step, Steps

#: The frames a run takes by default: the line's (``finn.kernels.fifo_sizing.FRAMES``),
#: so that a depth the line sizes and the network checks are read over the same frames.
FRAMES = 16


class Direct(Enum):
    """What a direct connection (a channel at its least capacity) is."""

    HANDSHAKE = "handshake"
    """No storage: the producer's write and the consumer's take are one event."""
    REGISTER = "register"
    """A one-word register: a word visible the cycle after it is written, its slot free
    the cycle after it is taken (``Rules.slot_free``)."""


@dataclass(frozen=True)
class Rules:
    """A channel's rules, in cycles.

    - ``fifo_latency``: a word is visible to the reader so many cycles after a FIFO
      takes it;
    - ``direct``: what a direct connection is (``Direct``);
    - ``slot_free``: a slot is free so many cycles after its word is taken (a FIFO's, a
      register's, and a reorder's buffer's after the beat that frees it);
    - ``reorder_latency``: a reorder presents a word no sooner than so many cycles after
      it took it.
    """

    fifo_latency: int
    direct: Direct
    slot_free: int
    reorder_latency: int

    def __post_init__(self) -> None:
        for name in ("fifo_latency", "slot_free", "reorder_latency"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} is a nonnegative number of cycles")
        if not isinstance(self.direct, Direct):
            raise TypeError("direct is a Direct")

    @property
    def direct_capacity(self) -> int:
        """The capacity of a direct connection: 0 words for a handshake, 1 for a register.
        No channel is smaller."""
        return 0 if self.direct is Direct.HANDSHAKE else 1

    def latency(self, capacity: int | None) -> int:
        """The cycles from a write to the word being visible, in a channel of
        ``capacity``: none through a handshake, one through a register, the FIFO's
        latency through any FIFO (an unbounded channel is one)."""
        if capacity is not None and capacity < self.direct_capacity:
            raise ValueError(
                f"a channel holds at least {self.direct_capacity} words under {self.direct.value}"
            )
        if capacity == self.direct_capacity:
            return 0 if self.direct is Direct.HANDSHAKE else 1
        return self.fifo_latency


FINNLIB = Rules(fifo_latency=2, direct=Direct.HANDSHAKE, slot_free=1, reorder_latency=2)
"""FinnLib's rules, as the line's per-channel model states them: its FIFO presents a word
two cycles after it accepts it, a slot is free the cycle after its word is taken, direct
is a handshake, and an input generator presents a word two cycles after it took it."""

ONE_CYCLE = Rules(fifo_latency=1, direct=Direct.REGISTER, slot_free=1, reorder_latency=1)
"""Every word visible the cycle after it is written, a direct connection a one-word
register, an input generator presenting a word the cycle after it took it (and its
channel the cycle after that): the simplest synchronous reading, kept to compare with."""


def _ints(values: Sequence[int], what: str) -> tuple[int, ...]:
    found = tuple(values)
    if any(type(value) is not int for value in found):
        raise TypeError(f"{what} are integers")
    return found


def _increasing(values: tuple[int, ...], what: str) -> None:
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{what} are strictly increasing: one word a cycle")


def _named(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("a stage or a channel has a nonempty name")


# --- the stages ------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A frame of ``len(times)`` words, word ``j`` due ``times[j]`` cycles after the
    frame's release; ``span`` the cycles the producer takes a frame; frames released every
    ``period`` cycles."""

    name: str
    times: tuple[int, ...]
    span: int
    period: int

    def __post_init__(self) -> None:
        _named(self.name)
        times = _ints(self.times, "a source's times")
        object.__setattr__(self, "times", times)
        if not times or times[0] < 0:
            raise ValueError("a source presents at least one word, from cycle 0")
        _increasing(times, "a source's times")
        if type(self.span) is not int or self.span <= times[-1] - times[0]:
            raise ValueError("a frame's words lie within the source's span")
        if type(self.period) is not int or self.period < self.span:
            raise ValueError(f"a source of span {self.span} cannot keep a period of {self.period}")

    @property
    def idle(self) -> int:
        """The cycles a period the source may be late by without delaying its next frame."""
        return self.period - self.span


@dataclass(frozen=True)
class Memory:
    """A memory source: it presents its next word whenever its channel has room, one a
    cycle (its words a frame are its channel's)."""

    name: str

    def __post_init__(self) -> None:
        _named(self.name)


@dataclass(frozen=True)
class Kernel:
    """A kernel of ``iterations`` iterations a frame: input port ``i`` takes a word at each
    iteration of
    ``reads[i]``, output port ``j`` writes one at each iteration of ``writes[j]``; a write is
    visible ``latency`` cycles later than its channel alone makes it."""

    name: str
    iterations: int
    reads: tuple[tuple[int, ...], ...]
    writes: tuple[tuple[int, ...], ...]
    latency: int = 0

    def __post_init__(self) -> None:
        _named(self.name)
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("a kernel takes at least one iteration a frame")
        if type(self.latency) is not int or self.latency < 0:
            raise ValueError("a kernel's latency is a nonnegative number of cycles")
        for field in ("reads", "writes"):
            ports = tuple(_ints(port, "a port's iterations") for port in getattr(self, field))
            for port in ports:
                if not port or port[0] < 0 or port[-1] >= self.iterations:
                    raise ValueError(
                        f"a port moves at least one word, at iterations 0..{self.iterations - 1}"
                    )
                _increasing(port, "a port's iterations")
            object.__setattr__(self, field, ports)
        if not self.reads and not self.writes:
            raise ValueError("a kernel has a port")


@dataclass(frozen=True)
class Fork:
    """Each word to every one of ``branches`` output ports at once, one a cycle."""

    name: str
    branches: int

    def __post_init__(self) -> None:
        _named(self.name)
        if type(self.branches) is not int or self.branches < 2:
            raise ValueError("a fork has two branches or more")


@dataclass(frozen=True)
class Sink:
    """Takes a word a cycle, whenever one is there."""

    name: str

    def __post_init__(self) -> None:
        _named(self.name)


@dataclass(frozen=True)
class WidthStage:
    """The same elements, ``lanes_in`` a word in, ``lanes_out`` a word out."""

    name: str
    lanes_in: int
    lanes_out: int

    def __post_init__(self) -> None:
        _named(self.name)
        for lanes in (self.lanes_in, self.lanes_out):
            if type(lanes) is not int or lanes < 1:
                raise ValueError("a width step's lanes are positive")


@dataclass(frozen=True)
class ReorderStage:
    """An input generator over a frame of ``freed[-1]`` words: beat ``b`` presents frame
    word ``reads[b]``, and ``freed[b]`` words of the frame are free after it; it takes a
    word while fewer than ``capacity`` taken words are unfreed."""

    name: str
    reads: tuple[int, ...]
    freed: tuple[int, ...]
    capacity: int

    def __post_init__(self) -> None:
        _named(self.name)
        reads, freed = _ints(self.reads, "a reorder's reads"), _ints(self.freed, "its frees")
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "freed", freed)
        if not reads or len(freed) != len(reads):
            raise ValueError("a reorder presents a beat at least, and frees after each")
        if any(b < a for a, b in zip(freed, freed[1:])) or freed[0] < 0:
            raise ValueError("a reorder's freed count never falls")
        words = freed[-1]
        if min(reads) < 0 or max(reads) >= words:
            raise ValueError(f"a reorder's beats read words of its frame of {words}")
        if any(word < done for word, done in zip(reads[1:], freed)):
            raise ValueError("a reorder never reads a word it freed")
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("a reorder holds a word at least")

    @property
    def words(self) -> int:
        """The words it takes a frame."""
        return self.freed[-1]


@dataclass(frozen=True)
class InsertionStage:
    """An in-order insertion: output beat ``b`` takes an input word where ``takes[b]``,
    else presents a fill."""

    name: str
    takes: tuple[bool, ...]

    def __post_init__(self) -> None:
        _named(self.name)
        object.__setattr__(self, "takes", tuple(bool(take) for take in self.takes))
        if not any(self.takes):
            raise ValueError("an insertion takes a word a frame at least")


@dataclass(frozen=True)
class DropStage:
    """An in-order drop: input beat ``b`` is passed on where ``passes[b]``, else dropped."""

    name: str
    passes: tuple[bool, ...]

    def __post_init__(self) -> None:
        _named(self.name)
        object.__setattr__(self, "passes", tuple(bool(kept) for kept in self.passes))
        if not any(self.passes):
            raise ValueError("a drop passes a word a frame at least")


@dataclass(frozen=True)
class MergeStage:
    """A merge of ``beats`` beats a frame: beat ``b`` is taken from input port
    ``pattern[b % len(pattern)]``."""

    name: str
    pattern: tuple[int, ...]
    beats: int

    def __post_init__(self) -> None:
        _named(self.name)
        found = _ints(self.pattern, "a merge's pattern")
        object.__setattr__(self, "pattern", found)
        if len(set(found)) < 2 or set(found) != set(range(len(set(found)))):
            raise ValueError("a merge's pattern names each of its two sources or more")
        if type(self.beats) is not int or self.beats < 1 or self.beats % len(found):
            raise ValueError("a merge's beats are whole repetitions of its pattern")

    @property
    def sources(self) -> int:
        return len(set(self.pattern))


Stage = (
    Source
    | Memory
    | Kernel
    | Fork
    | Sink
    | WidthStage
    | ReorderStage
    | InsertionStage
    | DropStage
    | MergeStage
)


def _ports(stage: Stage) -> tuple[int, int]:
    """A stage's input and output port counts."""
    if isinstance(stage, (Source, Memory)):
        return 0, 1
    if isinstance(stage, Kernel):
        return len(stage.reads), len(stage.writes)
    if isinstance(stage, Fork):
        return 1, stage.branches
    if isinstance(stage, Sink):
        return 1, 0
    if isinstance(stage, MergeStage):
        return stage.sources, 1
    return 1, 1


@dataclass(frozen=True)
class Channel:
    """A channel from ``producer``'s output port ``output`` to ``consumer``'s input port
    ``input``, of ``capacity`` words (``None``: unbounded)."""

    name: str
    producer: str
    consumer: str
    capacity: int | None = None
    output: int = 0
    input: int = 0

    def __post_init__(self) -> None:
        for name in (self.name, self.producer, self.consumer):
            _named(name)
        if self.capacity is not None and (type(self.capacity) is not int or self.capacity < 0):
            raise ValueError("a capacity is a nonnegative number of words, or None")
        for port in (self.output, self.input):
            if type(port) is not int or port < 0:
                raise ValueError("a port is a nonnegative index")


@dataclass(frozen=True)
class Network:
    """Stages and the channels between them: every port of every stage on exactly one
    channel, no stage on a channel to itself, and the words a frame each channel carries
    the same at both its ends (``words``)."""

    stages: tuple[Stage, ...]
    channels: tuple[Channel, ...]

    def __post_init__(self) -> None:
        stages, channels = tuple(self.stages), tuple(self.channels)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "channels", channels)
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("each stage has its own name")
        if len({channel.name for channel in channels}) != len(channels):
            raise ValueError("each channel has its own name")
        by_name = {stage.name: stage for stage in stages}
        wired: set[tuple[str, str, int]] = set()
        for channel in channels:
            for end, stage_name, port in (
                ("out", channel.producer, channel.output),
                ("in", channel.consumer, channel.input),
            ):
                if stage_name not in by_name:
                    raise ValueError(f"channel {channel.name}: no stage {stage_name}")
                ports = _ports(by_name[stage_name])[0 if end == "in" else 1]
                if port >= ports or (stage_name, end, port) in wired:
                    raise ValueError(f"channel {channel.name}: {stage_name}'s {end} port {port}")
                wired.add((stage_name, end, port))
            if channel.producer == channel.consumer:
                raise ValueError(f"channel {channel.name} joins a stage to itself")
        for stage in stages:
            inputs, outputs = _ports(stage)
            if len([w for w in wired if w[0] == stage.name]) != inputs + outputs:
                raise ValueError(f"every port of {stage.name} is on a channel")
        self.words  # noqa: B018 - checks that each channel's two ends agree

    def stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"no stage {name}")

    def channel(self, name: str) -> Channel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise ValueError(f"no channel {name}")

    @property
    def words(self) -> tuple[tuple[str, int], ...]:
        """Each channel's words a frame, by its name: as its producer states them, else
        its consumer; a fork's are its input's, and a width step's out are its in's times
        its lanes in over its lanes out."""
        return tuple(_words(self).items())

    @property
    def period(self) -> int:
        """The period the network's sources are released at: one for all."""
        periods = {stage.period for stage in self.stages if isinstance(stage, Source)}
        if len(periods) != 1:
            raise ValueError(f"a network's sources share one period: {sorted(periods)}")
        return periods.pop()

    def at(self, capacities: Mapping[str, int | None]) -> Network:
        """The same network, the channels named in ``capacities`` at those capacities."""
        unknown = set(capacities) - {channel.name for channel in self.channels}
        if unknown:
            raise ValueError(f"no channels {sorted(unknown)}")
        return Network(
            self.stages,
            tuple(
                replace(channel, capacity=capacities[channel.name])
                if channel.name in capacities
                else channel
                for channel in self.channels
            ),
        )

    def unbounded(self) -> Network:
        """The same network with every channel unbounded."""
        return self.at({channel.name: None for channel in self.channels})


def _stated(stage: Stage, end: str, port: int) -> int | None:
    """The words a frame a stage states on one port, where it states them."""
    if isinstance(stage, Source):
        return len(stage.times)
    if isinstance(stage, Kernel):
        return len((stage.reads if end == "in" else stage.writes)[port])
    if isinstance(stage, ReorderStage):
        return stage.words if end == "in" else len(stage.reads)
    if isinstance(stage, InsertionStage):
        return sum(stage.takes) if end == "in" else len(stage.takes)
    if isinstance(stage, DropStage):
        return len(stage.passes) if end == "in" else sum(stage.passes)
    if isinstance(stage, MergeStage):
        if end == "out":
            return stage.beats
        return stage.pattern.count(port) * (stage.beats // len(stage.pattern))
    return None


def _words(network: Network) -> dict[str, int]:
    found: dict[str, int] = {}
    stages = {stage.name: stage for stage in network.stages}
    for channel in network.channels:
        out = _stated(stages[channel.producer], "out", channel.output)
        read = _stated(stages[channel.consumer], "in", channel.input)
        if out is not None and read is not None and out != read:
            raise ValueError(
                f"channel {channel.name}: {channel.producer} writes {out} words a frame, "
                f"{channel.consumer} reads {read}"
            )
        stated = out if out is not None else read
        if stated is not None:
            found[channel.name] = stated
    # A fork passes its input's count, a width step scales it; memory and sinks take theirs.
    inputs = {(c.consumer, c.input): c.name for c in network.channels}
    outputs = {(c.producer, c.output): c.name for c in network.channels}
    moved = True
    while moved:
        moved = False
        for stage in network.stages:
            if not isinstance(stage, (Fork, WidthStage)):
                continue
            into = inputs[(stage.name, 0)]
            lanes_in, lanes_out = (
                (1, 1) if isinstance(stage, Fork) else (stage.lanes_in, stage.lanes_out)
            )
            for port in range(_ports(stage)[1]):
                out_name = outputs[(stage.name, port)]
                if into in found and out_name in found:
                    if found[out_name] * lanes_out != found[into] * lanes_in:
                        raise ValueError(f"{stage.name}: its ports' words a frame disagree")
                    continue
                if into in found:
                    known, name, scale = found[into], out_name, (lanes_in, lanes_out)
                elif out_name in found:
                    known, name, scale = found[out_name], into, (lanes_out, lanes_in)
                else:
                    continue
                words, rest = divmod(known * scale[0], scale[1])
                if rest:
                    raise ValueError(f"{stage.name}: a frame's elements are no whole words")
                found[name], moved = words, True
    missing = [channel.name for channel in network.channels if channel.name not in found]
    if missing:
        raise ValueError(f"no stage states the words a frame of {missing}")
    return found


# --- building stages -------------------------------------------------------------------


def kernel(name: str, reads: Sequence[Pace], writes: Sequence[Pace], latency: int = 0) -> Kernel:
    """A kernel paced by its schedule: each port's iterations its ``Pace``'s times
    (``Schedule.beat_times``), its iterations a frame its schedule's; ``latency`` the
    pipeline's, stated (0 unless stated)."""
    paces = (*reads, *writes)
    schedules = {pace.schedule for pace in paces}
    if len(schedules) != 1:
        raise ValueError("a kernel's ports share one schedule")
    return Kernel(
        name,
        paces[0].span,
        tuple(pace.times for pace in reads),
        tuple(pace.times for pace in writes),
        latency,
    )


def frees(reads: Sequence[int], words: int) -> tuple[int, ...]:
    """The words of a frame of ``words`` free after each beat reading ``reads``: each word
    freed after its last read, in order (a word no beat reads is free once every word
    before it is)."""
    last: dict[int, int] = {}
    for beat, word in enumerate(reads):
        last[word] = beat
    found: list[int] = []
    freed = 0
    for beat in range(len(reads)):
        while freed < words and last.get(freed, -1) <= beat:
            freed += 1
        found.append(freed)
    return tuple(found)


def reorder(
    name: str, reads: Sequence[int], words: int, capacity: int | None = None
) -> ReorderStage:
    """An input generator over a frame of ``words`` words whose beats read ``reads``, each
    word freed after its last read (``frees``); its capacity ``buffers.live_window`` of
    the reads unless given. This default preserves order, not necessarily rate;
    sustaining a frame period may require a larger capacity under the stated rules."""
    found = tuple(int(word) for word in reads)
    if capacity is None:
        capacity = live_window(np.asarray(found, dtype=np.int64))
    return ReorderStage(name, found, frees(found, words), capacity)


def step_stages(chain: Steps, name: str) -> tuple[WidthStage | ReorderStage, ...]:
    """Each step of ``chain`` as a stage, named ``name`` and the step's place: a width
    step at its two lane counts; a reorder at its slots over its source's frame
    (``reorder``: its frame's slots, ``buffers.frame_slots``, repeated for each reorder frame of
    the source's frame; its live window the default capacity, not a rate guarantee). Markers move
    no word and are no
    stage."""
    found: list[WidthStage | ReorderStage] = []
    for place, hop in enumerate(chain.hops):
        called = f"{name}.{place}"
        if hop.step is Step.WIDTH:
            found.append(WidthStage(called, hop.source.form.lanes, hop.sink.form.lanes))
        elif hop.step is Step.REORDER:
            assert hop.reorder is not None  # a reorder step carries its reorder
            frame = hop.reorder.frame_beats
            slots = [int(s) for s in frame_slots(hop.reorder)]
            frames = hop.source.form.beats // frame
            reads = [k * frame + s for k in range(frames) for s in slots]
            found.append(reorder(called, reads, frames * frame))
    return tuple(found)


def route_stages(
    reindex: Reindex, name: str
) -> tuple[WidthStage | ReorderStage | InsertionStage | DropStage, ...]:
    """The stages of a one-reader reindex's route (``reindex.route``), producer first:
    its ``before`` steps, its insertion or drop at the lanes it runs at
    (``reindex.stage_lanes``), its ``after`` steps."""
    found = route(reindex)
    if isinstance(found, Forked):
        raise ValueError("a fork's routes are staged one branch at a time")
    staged: list[WidthStage | ReorderStage | InsertionStage | DropStage] = [
        *step_stages(found.before, f"{name}.before")
    ]
    if found.stage is not None:
        lanes = stage_lanes(found.stage, reindex.producer.form.lanes)
        staged.append(in_order(f"{name}.stage", found.stage, lanes))
    staged.extend(step_stages(found.after, f"{name}.after"))
    return tuple(staged)


def in_order(name: str, stage: Insertion | Drop, lanes: int) -> InsertionStage | DropStage:
    """An insertion or a drop at ``lanes`` lanes as a stage: per beat, whether it moves a
    word of its input (``reindex.stage_beats``)."""
    moves = stage_beats(stage, lanes)
    if isinstance(stage, Insertion):
        return InsertionStage(name, moves)
    return DropStage(name, moves)


def merge_stage(name: str, merge: Merge) -> MergeStage:
    """A merge as a stage: its sources' pattern (``reindex.pattern``) over the beats it
    merges a frame of the reader, at ``reindex.merge_lanes``."""
    merge_lanes(joined(merge))  # refuses a merge that takes no whole beats
    return MergeStage(name, pattern(merge), merge_stage_cost(merge).reads)


def chained(stages: Sequence[Stage], capacity: int | None = None) -> Network:
    """``stages`` one after the other, each output port 0 to the next's input port 0
    through a channel named ``"<producer>><consumer>"`` of ``capacity`` words."""
    links = tuple(
        Channel(f"{a.name}>{b.name}", a.name, b.name, capacity) for a, b in zip(stages, stages[1:])
    )
    return Network(tuple(stages), links)


def order(network: Network) -> tuple[str, ...]:
    """The channels from the sources down: a stage is placed once every channel into it
    is, then its output channels in port order; stages are taken in the network's order
    among those ready. ``ValueError`` for a loop."""
    into: dict[str, set[str]] = {stage.name: set() for stage in network.stages}
    for channel in network.channels:
        into[channel.consumer].add(channel.name)
    outs = sorted(network.channels, key=lambda channel: channel.output)
    placed: list[str] = []
    done: set[str] = set()
    while len(done) < len(network.stages):
        ready = [s for s in network.stages if s.name not in done and into[s.name] <= set(placed)]
        if not ready:
            raise ValueError("a network's channels form a loop, which the model does not time")
        stage = ready[0]
        done.add(stage.name)
        placed.extend(channel.name for channel in outs if channel.producer == stage.name)
    return tuple(placed)


# --- the run ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """What a run of ``frames`` frames gave.

    - ``deadlocked``: no stage could move while one had frames left;
    - ``completions``: each stage's frame completions, the cycle of its last move of each
      frame it finished (a reorder's and a width step's, of their output);
    - ``late`` and ``idle``: each source's lateness each frame it finished (its last
      word's cycle after its unthrottled one) and its idle time a period;
    - ``writes`` and ``reads``: each channel's words' write and read cycles, in order;
    - ``peaks``: each channel's peak, the most words it held before a write, plus one.

    Each is by name, in the network's order.
    """

    frames: int
    deadlocked: bool
    completions: tuple[tuple[str, tuple[int, ...]], ...]
    late: tuple[tuple[str, tuple[int, ...]], ...]
    idle: tuple[tuple[str, int], ...]
    writes: tuple[tuple[str, tuple[int, ...]], ...]
    reads: tuple[tuple[str, tuple[int, ...]], ...]
    peaks: tuple[tuple[str, int], ...]

    @property
    def on_time(self) -> bool:
        """Whether every source starts every frame on time: no deadlock, and every source
        late by no more than its idle time each frame. The sources' half of keeping the
        period (``period_kept``): alone it does not see a stage slower than the period behind
        a channel or a reorder with room for the backlog. On a chain of two stages it is the
        line's ``within``. The unbounded stage rate is judged separately by
        ``bottleneck(network, rules) == network.period`` over the frames run."""
        idle = dict(self.idle)
        return not self.deadlocked and all(
            each <= idle[name] for name, late in self.late for each in late
        )

    def completed(self, stage: str) -> tuple[int, ...]:
        return dict(self.completions)[stage]

    def period(self, stage: str) -> int | None:
        """The cycles between ``stage``'s last two frame completions; None before two."""
        done = self.completed(stage)
        return None if len(done) < 2 else done[-1] - done[-2]

    def peak(self, channel: str) -> int:
        return dict(self.peaks)[channel]

    def written(self, channel: str) -> tuple[int, ...]:
        return dict(self.writes)[channel]

    def taken(self, channel: str) -> tuple[int, ...]:
        return dict(self.reads)[channel]


class _Link:
    """A channel during a run: its write and read cycles, and its rule: the words it holds
    (``capacity``; a handshake none), the cycles from a write to the word being visible
    (``delay``), and from a take to its slot being free (``free``)."""

    __slots__ = (
        "name",
        "capacity",
        "handshake",
        "delay",
        "free",
        "written",
        "read",
        "writer",
        "reader",
    )

    def __init__(
        self, name: str, capacity: int | None, handshake: bool, delay: int, free: int
    ) -> None:
        self.name, self.capacity, self.handshake = name, capacity, handshake
        self.delay, self.free = delay, free
        self.written: list[int] = []
        self.read: list[int] = []
        self.writer: _Actor
        self.reader: _Actor


class _Move:
    """One move of an actor's frame: its spacing after the move before (the first: after
    the last of the frame before), the channels it reads and writes, a source's due cycle
    within the frame, and whether a handshake joins it to another actor's move."""

    __slots__ = ("gap", "reads", "writes", "due", "joined")

    def __init__(
        self, gap: int, reads: tuple[_Link, ...], writes: tuple[_Link, ...], due: int | None
    ) -> None:
        self.gap, self.reads, self.writes, self.due = gap, reads, writes, due
        self.joined = any(link.handshake for link in (*reads, *writes))


_Bound = Callable[[int], int | None]
"""An actor's own buffer's bound on its move ``i``: the least cycle it allows, or None
while the move it waits on has not happened."""


class _Actor:
    """A sequence of moves, ``frame`` a frame, one a cycle at most, its first no sooner
    than ``start``; a source's frames released every ``period`` cycles."""

    __slots__ = ("name", "frame", "size", "total", "start", "period", "times", "bound")

    def __init__(
        self, name: str, frame: list[_Move], frames: int, start: int, period: int = 0
    ) -> None:
        self.name, self.frame, self.start, self.period = name, frame, start, period
        self.size = len(frame)
        self.total = frames * len(frame)
        self.times: list[int] = []
        self.bound: _Bound | None = None

    def next(self) -> _Move | None:
        done = len(self.times)
        return None if done == self.total else self.frame[done % self.size]

    def own(self, move: _Move) -> int | None:
        """The least cycle its spacing, its due cycle and its buffer allow its next move;
        None while its buffer waits on a move not made."""
        times = self.times
        low = times[-1] + move.gap if times else self.start
        if move.due is not None:
            due = len(times) // self.size * self.period + move.due
            if due > low:
                low = due
        if self.bound is not None:
            bound = self.bound(len(times))
            if bound is None:
                return None
            if bound > low:
                low = bound
        return low


def _spaced(times: Sequence[int], span: int) -> list[int]:
    """The gap before each move at ``times`` of a frame of ``span``: the first's from the
    last of the frame before."""
    return [times[0] + span - times[-1], *(b - a for a, b in zip(times, times[1:]))]


class _Engine:
    """The run's state: actors, links, and the moves made."""

    def __init__(self, network: Network, rules: Rules, frames: int) -> None:
        self.rules, self.frames = rules, frames
        latency = {
            stage.name: stage.latency if isinstance(stage, Kernel) else 0
            for stage in network.stages
        }
        self.links: dict[str, _Link] = {}
        self.into: dict[tuple[str, int], _Link] = {}
        self.out: dict[tuple[str, int], _Link] = {}
        for channel in network.channels:
            pipeline = latency[channel.producer]
            delay = rules.latency(channel.capacity) + pipeline
            if channel.capacity == 0 and pipeline:
                # A handshake holds no word: the pipeline's own registers hold its words in
                # flight, each free as its word is taken (the pipeline advances with it).
                link = _Link(channel.name, pipeline, False, delay, 0)
            else:
                handshake = channel.capacity == 0 and rules.direct is Direct.HANDSHAKE
                link = _Link(channel.name, channel.capacity, handshake, delay, rules.slot_free)
            self.links[channel.name] = link
            self.into[(channel.consumer, channel.input)] = link
            self.out[(channel.producer, channel.output)] = link
        self.words = dict(network.words)
        self.actors: list[_Actor] = []
        self.main: dict[str, _Actor] = {}
        self.sources: list[tuple[Source, _Actor]] = []
        for stage in network.stages:
            self._add(stage)

    def _actor(
        self, name: str, frame: list[_Move], start: int = 0, period: int = 0, main: bool = True
    ) -> _Actor:
        actor = _Actor(name, frame, self.frames, start, period)
        for move in frame:
            for link in move.reads:
                link.reader = actor
            for link in move.writes:
                link.writer = actor
        self.actors.append(actor)
        if main:
            self.main[name] = actor
        return actor

    def _even(self, count: int, reads: tuple[_Link, ...], writes: tuple[_Link, ...]) -> list[_Move]:
        return [_Move(1, reads, writes, None) for _ in range(count)]

    def _add(self, stage: Stage) -> None:
        name = stage.name
        if isinstance(stage, Source):
            out = self.out[(name, 0)]
            gaps = _spaced(stage.times, stage.span)
            frame = [_Move(gap, (), (out,), due) for gap, due in zip(gaps, stage.times)]
            actor = self._actor(name, frame, stage.times[0], stage.period)
            self.sources.append((stage, actor))
        elif isinstance(stage, Memory):
            out = self.out[(name, 0)]
            self._actor(name, self._even(self.words[out.name], (), (out,)))
        elif isinstance(stage, Kernel):
            moves: dict[int, tuple[list[_Link], list[_Link]]] = {}
            for port, at in enumerate(stage.reads):
                for iteration in at:
                    moves.setdefault(iteration, ([], []))[0].append(self.into[(name, port)])
            for port, at in enumerate(stage.writes):
                for iteration in at:
                    moves.setdefault(iteration, ([], []))[1].append(self.out[(name, port)])
            iterations = sorted(moves)
            gaps = _spaced(iterations, stage.iterations)
            frame = [
                _Move(gap, tuple(moves[iteration][0]), tuple(moves[iteration][1]), None)
                for gap, iteration in zip(gaps, iterations)
            ]
            self._actor(name, frame, iterations[0])
        elif isinstance(stage, Fork):
            into = self.into[(name, 0)]
            outs = tuple(self.out[(name, port)] for port in range(stage.branches))
            self._actor(name, self._even(self.words[into.name], (into,), outs))
        elif isinstance(stage, Sink):
            into = self.into[(name, 0)]
            self._actor(name, self._even(self.words[into.name], (into,), ()))
        elif isinstance(stage, MergeStage):
            ins = [self.into[(name, port)] for port in range(stage.sources)]
            out = self.out[(name, 0)]
            frame = [
                _Move(1, (ins[stage.pattern[beat % len(stage.pattern)]],), (out,), None)
                for beat in range(stage.beats)
            ]
            self._actor(name, frame)
        elif isinstance(stage, InsertionStage):
            into, out = self.into[(name, 0)], self.out[(name, 0)]
            frame = [_Move(1, (into,) if take else (), (out,), None) for take in stage.takes]
            self._actor(name, frame)
        elif isinstance(stage, DropStage):
            into, out = self.into[(name, 0)], self.out[(name, 0)]
            frame = [_Move(1, (into,), (out,) if kept else (), None) for kept in stage.passes]
            self._actor(name, frame)
        elif isinstance(stage, WidthStage):
            self._width(stage)
        else:
            self._reorder(stage)

    def _width(self, stage: WidthStage) -> None:
        into, out = self.into[(stage.name, 0)], self.out[(stage.name, 0)]
        lanes_in, lanes_out = stage.lanes_in, stage.lanes_out
        taker = self._actor(
            stage.name + ":in", self._even(self.words[into.name], (into,), ()), main=False
        )
        writer = self._actor(stage.name, self._even(self.words[out.name], (), (out,)))

        def take(i: int) -> int | None:
            # Input word i waits for the last output word made only of words before i - 1.
            last = (i - 1) * lanes_in // lanes_out - 1
            if last < 0:
                return 0
            return writer.times[last] if last < len(writer.times) else None

        def write(w: int) -> int | None:
            # Output word w leaves the cycle after its last element arrived.
            last = ceil((w + 1) * lanes_out / lanes_in) - 1
            return taker.times[last] + 1 if last < len(taker.times) else None

        taker.bound, writer.bound = take, write

    def _reorder(self, stage: ReorderStage) -> None:
        into, out = self.into[(stage.name, 0)], self.out[(stage.name, 0)]
        words, beats = stage.words, len(stage.reads)
        taker = self._actor(stage.name + ":in", self._even(words, (into,), ()), main=False)
        presenter = self._actor(stage.name, self._even(beats, (), (out,)))
        # The first beat of a frame by which it has freed each count 1 .. words.
        reached: list[int] = []
        for beat, freed in enumerate(stage.freed):
            while len(reached) < freed:
                reached.append(beat)
        capacity, slot_free, latency = (
            stage.capacity,
            self.rules.slot_free,
            self.rules.reorder_latency,
        )

        def take(n: int) -> int | None:
            # Word n waits until the beats have freed n - capacity + 1 words.
            count = n - capacity + 1
            if count <= 0:
                return 0
            frame, place = divmod(count - 1, words)
            beat = frame * beats + reached[place]
            return presenter.times[beat] + slot_free if beat < len(presenter.times) else None

        def present(b: int) -> int | None:
            frame, place = divmod(b, beats)
            word = frame * words + stage.reads[place]
            return taker.times[word] + latency if word < len(taker.times) else None

        taker.bound, presenter.bound = take, present

    def _try(self, actor: _Actor) -> bool:
        """Make ``actor``'s next move, with every move a handshake joins to it, at the
        least cycle all their bounds allow; False where one cannot move yet."""
        move = actor.next()
        if move is None:
            return False
        if move.joined:
            return self._joined(actor, move)
        low = actor.own(move)
        if low is None or (low := self._channels(move, low)) is None:
            return False
        for link in move.reads:
            link.read.append(low)
        for link in move.writes:
            link.written.append(low)
        actor.times.append(low)
        return True

    def _channels(self, move: _Move, low: int) -> int | None:
        """``low`` raised to the cycle the move's FIFO channels allow: each word it reads
        visible, and room for each word it writes; None while one waits on a move not
        made. A handshake's are its joined move's."""
        for link in move.reads:
            if link.handshake:
                continue
            word = len(link.read)
            if word >= len(link.written):
                return None
            visible = link.written[word] + link.delay
            if visible > low:
                low = visible
        for link in move.writes:
            if link.handshake or link.capacity is None:
                continue
            word = len(link.written) - link.capacity
            if word >= 0:
                if word >= len(link.read):
                    return None
                free = link.read[word] + link.free
                if free > low:
                    low = free
        return low

    def _joined(self, actor: _Actor, first: _Move) -> bool:
        """A move a handshake joins to others: every joined actor's next move must be the
        one on that handshake; they move together, at the least cycle all allow."""
        moves = {actor: first}
        pending = [actor]
        while pending:
            move = moves[pending.pop()]
            joined = [(link, link.writer, True) for link in move.reads if link.handshake]
            joined += [(link, link.reader, False) for link in move.writes if link.handshake]
            for link, other, writes in joined:
                found = moves.get(other) or other.next()
                if found is None or link not in (found.writes if writes else found.reads):
                    return False
                if other not in moves:
                    moves[other] = found
                    pending.append(other)
        at = 0
        for member, move in moves.items():
            low = member.own(move)
            if low is None or (low := self._channels(move, low)) is None:
                return False
            at = max(at, low)
        for member, move in moves.items():
            for link in move.reads:
                link.read.append(at)
            for link in move.writes:
                link.written.append(at)
            member.times.append(at)
        return True

    def run(self) -> bool:
        """Every move that can be made; True where one stage has frames left (a deadlock)."""
        moved = True
        while moved:
            moved = False
            for actor in self.actors:
                while self._try(actor):
                    moved = True
        return any(len(actor.times) < actor.total for actor in self.actors)


def _peak(link: _Link) -> int:
    """The most words held before a write, plus one: held is the words written before it
    less those whose slot is free by then, ``link.free`` cycles after the word was read
    (a slot is held until then, as the channel's capacity counts it)."""
    best, taken = 0, 0
    for word, at in enumerate(link.written):
        while taken < len(link.read) and link.read[taken] + link.free <= at:
            taken += 1
        best = max(best, word - taken + 1)
    return best


def run(network: Network, rules: Rules, frames: int = FRAMES) -> Run:
    """``frames`` frames through ``network`` at its channels' capacities, under
    ``rules`` (the module's run)."""
    if type(frames) is not int or frames < 1:
        raise ValueError("a run takes a frame at least")
    engine = _Engine(network, rules, frames)
    deadlocked = engine.run()
    completions = []
    for stage in network.stages:
        actor = engine.main[stage.name]
        size = len(actor.frame)
        done = len(actor.times) // size
        completions.append(
            (stage.name, tuple(actor.times[(k + 1) * size - 1] for k in range(done)))
        )
    late = []
    for source, actor in engine.sources:
        size = len(source.times)
        done = len(actor.times) // size
        late.append(
            (
                source.name,
                tuple(
                    actor.times[(k + 1) * size - 1] - (k * source.period + source.times[-1])
                    for k in range(done)
                ),
            )
        )
    links = [engine.links[channel.name] for channel in network.channels]
    return Run(
        frames,
        deadlocked,
        tuple(completions),
        tuple(late),
        tuple((source.name, source.idle) for source, _ in engine.sources),
        tuple((link.name, tuple(link.written)) for link in links),
        tuple((link.name, tuple(link.read)) for link in links),
        tuple((link.name, _peak(link)) for link in links),
    )


# --- sizing ----------------------------------------------------------------------------


def ends(network: Network) -> tuple[str, ...]:
    """The stages with no output: where a network's period is read."""
    return tuple(stage.name for stage in network.stages if _ports(stage)[1] == 0)


def _period(network: Network, found: Run) -> int:
    """The most cycles between any end's last two frame completions in ``found``."""
    if found.deadlocked:
        raise ValueError("the network deadlocks with every channel unbounded")
    periods = [found.period(name) for name in ends(network)]
    return max(period for period in periods if period is not None)


def bottleneck(network: Network, rules: Rules, frames: int = FRAMES) -> int:
    """The bottleneck period: a network's own rate, from the run with every channel
    unbounded: the most cycles between any end's last two frame completions (``ends``).
    It judges no sized set (``period_kept`` does): with channels at capacities, the network
    can be slower than this. ``ValueError`` for a loop: a run times one, but a period is
    read down a network from its sources."""
    order(network)  # refuses a loop
    if frames < 2:
        raise ValueError("a period is read over two frames at least")
    return _period(network, run(network.unbounded(), rules, frames))


def _checked(network: Network, rules: Rules, frames: int) -> Run:
    """The unbounded run, refused where its period is not the sources'."""
    order(network)  # refuses a loop
    if frames < 2:
        raise ValueError("a period is read over two frames at least")
    free = run(network.unbounded(), rules, frames)
    found, period = _period(network, free), network.period
    if found != period:
        raise ValueError(
            f"the network's bottleneck period is {found} cycles over {frames} frames, its "
            f"sources' {period}: release them at its bottleneck"
        )
    return free


@dataclass(frozen=True)
class Due:
    """When each end must complete each frame: end ``e``'s frame ``k`` by
    ``k * period + by[e]`` cycles, over ``frames`` frames. ``by[e]`` is the most cycles any
    frame of ``e`` completes after ``k * period`` in the network's unbounded run, plus a
    slack: by default the most idle time of the sources ``e`` is reached from (0 where none
    is). It is read once for a network, from its unbounded run. A run at the capacities
    judged is held to it (``met``), never re-read with every channel unbounded."""

    period: int
    frames: int
    by: tuple[tuple[str, int], ...]

    def met(self, found: Run) -> bool:
        """Whether every end of ``found`` completes every frame by its due cycle: no
        deadlock, every frame run, none later. ``ValueError`` for a run of other frames."""
        if found.frames != self.frames:
            raise ValueError(
                f"due cycles over {self.frames} frames judge no run of {found.frames} frames"
            )
        if found.deadlocked:
            return False
        for end, by in self.by:
            done = found.completed(end)
            if len(done) != self.frames or any(
                at > frame * self.period + by for frame, at in enumerate(done)
            ):
                return False
        return True


def _reached_from(network: Network) -> dict[str, set[str]]:
    """The names of the sources each stage is reached from, through the channels."""
    producers: dict[str, list[str]] = {stage.name: [] for stage in network.stages}
    for channel in network.channels:
        producers[channel.consumer].append(channel.producer)
    sources = {stage.name for stage in network.stages if isinstance(stage, Source)}
    found: dict[str, set[str]] = {}
    for stage in network.stages:
        seen: set[str] = set()
        stack = [stage.name]
        while stack:
            for producer in producers[stack.pop()]:
                if producer not in seen:
                    seen.add(producer)
                    stack.append(producer)
        found[stage.name] = seen & sources
    return found


def _due(network: Network, free: Run, slack: int | None) -> Due:
    """The due cycles from ``free``, the network's unbounded run (``_checked``)."""
    if slack is not None and (type(slack) is not int or slack < 0):
        raise ValueError("a slack is a nonnegative number of cycles")
    period = network.period
    idle = {stage.name: stage.idle for stage in network.stages if isinstance(stage, Source)}
    reached = _reached_from(network)
    by = []
    for end in ends(network):
        late = max(at - frame * period for frame, at in enumerate(free.completed(end)))
        allowed = max((idle[name] for name in reached[end]), default=0) if slack is None else slack
        by.append((end, late + allowed))
    return Due(period, free.frames, tuple(by))


def _period_kept(due_cycles: Due) -> Callable[[Run], bool]:
    def period_kept(found: Run) -> bool:
        return found.on_time and due_cycles.met(found)

    period_kept.__qualname__ = "period_kept"
    return period_kept


def due(network: Network, rules: Rules, frames: int = FRAMES, slack: int | None = None) -> Due:
    """Each end's due cycles (``Due``), from the run with every channel unbounded; ``slack``,
    where given, in place of each end's sources' idle time. ``ValueError`` where that run
    deadlocks or its bottleneck period is not its sources' (``bottleneck``): a network
    slower than its sources has no due cycles at their period."""
    return _due(network, _checked(network, rules, frames), slack)


def period_kept(
    network: Network, rules: Rules, frames: int = FRAMES, slack: int | None = None
) -> Callable[[Run], bool]:
    """The default criterion of ``least`` and ``minimal``: a run of ``network`` keeps the
    period when its sources start every frame on time (``Run.on_time``) and its ends
    complete every frame by their due cycles (``due``, ``Due.met``). It is monotone as
    capacities of one word or more grow: the due cycles are fixed by the network, and with
    more room no move is later."""
    return _period_kept(due(network, rules, frames, slack))


def _lowered(
    network: Network,
    name: str,
    upper: int,
    rules: Rules,
    frames: int,
    keeps: Callable[[Run], bool],
) -> int:
    """The least capacity of channel ``name`` up to ``upper`` keeping the period (as
    ``keeps`` reads a run) with the others as ``network`` states them: ``upper`` run
    first and refused where it does not keep it, then direct tried apart, then the FIFO
    capacities above it by bisection (monotone among FIFOs).

    Direct is tried apart because it is ordered against a one-word FIFO neither way under
    handshake rules: direct can keep the period where one word does not, and one word can
    keep it where direct does not (a handshake moves a word in the cycles a FIFO's latency
    spends; a FIFO's slot rule and latency are its own). Only among FIFO capacities does a
    larger one never lose the period, so only those are bisected."""
    if not keeps(run(network.at({name: upper}), rules, frames)):
        raise ValueError(
            f"channel {name} does not keep the period at {upper} words, the capacity it is "
            f"lowered from"
        )
    direct = rules.direct_capacity
    if upper == direct or keeps(run(network.at({name: direct}), rules, frames)):
        return direct
    capacities = range(direct + 1, upper + 1)
    # ``upper``, the last, is known to keep it: the search ends there where no other does.
    return capacities[
        bisect_left(
            capacities,
            True,
            hi=len(capacities) - 1,
            key=lambda capacity: keeps(run(network.at({name: capacity}), rules, frames)),
        )
    ]


def least(
    network: Network,
    channel: str,
    rules: Rules,
    frames: int = FRAMES,
    keeps: Callable[[Run], bool] | None = None,
) -> int | None:
    """The least capacity of ``channel`` keeping the period with every other channel at
    the capacity ``network`` states; None where none does (with ``channel`` unbounded the
    run does not keep it: where the channels the caller fixed make an end late, among
    others).

    The network's bottleneck must be its sources' period, whatever ``keeps`` is: the
    unbounded run is checked first, and a network slower than its sources (or one that
    deadlocks unbounded) is refused with ``ValueError`` before ``keeps`` runs.

    ``keeps`` decides, from a run, whether it keeps the period. By default it is
    ``period_kept(network, rules, frames)``, built from that unbounded run: the sources on
    time and every end by its due cycles. The reference of the due cycles is the run with
    every channel unbounded, not the one with only ``channel`` unbounded, so a channel the
    caller fixed (a direct generator output) is held to the network's own due cycles, and
    ``least`` agrees with ``minimal``. ``keeps`` must be monotone as the capacity grows, as
    that rule is (the bisection rests on it)."""
    free = _checked(network, rules, frames)
    if keeps is None:
        keeps = _period_kept(_due(network, free, None))
    free = run(network.at({channel: None}), rules, frames)
    if not keeps(free):
        return None
    upper = max(free.peak(channel), rules.direct_capacity)
    return _lowered(network, channel, upper, rules, frames, keeps)


@dataclass(frozen=True)
class Sizing:
    """A minimal set of capacities keeping the period ``period`` over ``frames`` frames:
    the channels in the order they were lowered (``order``), the unbounded run's peaks
    and the capacities found, in that order, the passes the search took, and the criterion
    used (``criterion``: the ``__qualname__`` of the ``keeps`` callable, ``period_kept``
    for the default)."""

    period: int
    frames: int
    order: tuple[str, ...]
    peaks: tuple[int, ...]
    capacities: tuple[int, ...]
    passes: int
    criterion: str

    def capacity(self, channel: str) -> int:
        return self.capacities[self.order.index(channel)]

    @property
    def found(self) -> dict[str, int]:
        """The capacities by channel."""
        return dict(zip(self.order, self.capacities))

    @property
    def total(self) -> int:
        return sum(self.capacities)


def minimal(
    network: Network,
    rules: Rules,
    frames: int = FRAMES,
    keeps: Callable[[Run], bool] | None = None,
    channels: Sequence[str] | None = None,
) -> Sizing:
    """A minimal set of capacities keeping the bottleneck period, found jointly from the
    unbounded run's peaks, the channels lowered in ``order`` pass after pass until none
    moves (the module's "Sizing": its monotonicity, termination and minimality).

    The network's bottleneck must be its sources' period, whatever ``keeps`` is: the
    unbounded run is checked first, and a network slower than its sources (or one that
    deadlocks unbounded) is refused with ``ValueError`` before ``keeps`` runs.

    ``keeps`` decides, from a run, whether it keeps the period. By default it is
    ``period_kept(network, rules, frames)``, built from that unbounded run: the sources on
    time and every end by its due cycles. It must be monotone as the capacities grow, as
    that rule is. ``channels`` is the order the channels are lowered in, each of the
    network's once (by default ``order``: from the sources down); ``Sizing.order`` states
    the one used, and ``Sizing.criterion`` the criterion."""
    free = _checked(network, rules, frames)
    if keeps is None:
        keeps = _period_kept(_due(network, free, None))
    if channels is None:
        lowering = order(network)
    else:
        lowering = tuple(channels)
        if sorted(lowering) != sorted(channel.name for channel in network.channels):
            raise ValueError(
                f"the search order names each of the network's channels once: {lowering}"
            )
    peaks = tuple(max(free.peak(name), rules.direct_capacity) for name in lowering)
    found = dict(zip(lowering, peaks))
    passes, moved = 0, True
    while moved:
        moved, passes = False, passes + 1
        for name in lowering:
            lowered = _lowered(network.at(found), name, found[name], rules, frames, keeps)
            if lowered < found[name]:
                found[name], moved = lowered, True
    return Sizing(
        network.period,
        frames,
        lowering,
        peaks,
        tuple(found[name] for name in lowering),
        passes,
        getattr(keeps, "__qualname__", type(keeps).__qualname__),
    )


__all__ = [
    "FINNLIB",
    "FRAMES",
    "ONE_CYCLE",
    "Channel",
    "Direct",
    "DropStage",
    "Due",
    "Fork",
    "InsertionStage",
    "Kernel",
    "Memory",
    "MergeStage",
    "Network",
    "ReorderStage",
    "Rules",
    "Run",
    "Sink",
    "Sizing",
    "Source",
    "Stage",
    "WidthStage",
    "bottleneck",
    "chained",
    "due",
    "ends",
    "frees",
    "in_order",
    "kernel",
    "least",
    "merge_stage",
    "minimal",
    "order",
    "period_kept",
    "reorder",
    "route_stages",
    "run",
    "step_stages",
]
