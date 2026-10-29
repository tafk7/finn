# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The line's FIFO depths against the package's network model.

Each sizing FINN asked of ``finn.kernels.fifo_sizing`` (``fifo_record``) is translated into a
network of the package's stages (``finn.dataflow.timing``) and its depth for the channel asked
again of that network, under FinnLib's rules (``timing.FINNLIB``) at the frames the line
runs (``fifo_sizing.FRAMES``).

**The translation.** The line's consumer has no phase, so an acceptance's first read is
shifted to step 0 (its span is kept: the wrap-around gap is the line's). A supply already
width-converted enters as a ``Source`` of the line's converted pattern, not a source and a
``WidthStage``: the model's width step holds its producer back, which the line does not model.

- *A call with sides only* (no partition) is two stages and a channel ``fifo``: the supply as
  a ``Source``; a ``Kernel`` reading at the acceptance's beat times, or for a buffered
  acceptance a ``ReorderStage`` over the line's buffer (its reads, frees and capacity) and a
  ``Kernel`` reading at its reader's beat times (the generator's output direct).
  The line counts a buffer's words across frames, so a consumer's frame may hold several of
  the buffer's (the reader takes a whole number of times the buffer's beats): the generator's
  frame is then the buffer's run that many times, its words and frees following on (``repeated``).
- *A partition a strategy sized* (``SizeFifos``) is one network: each kernel from the
  paces of its ports in the partition (``timing.kernel``); for each channel, its output
  adapter's width conversion as a ``WidthStage`` after a kernel (a ``Source`` of the converted
  pattern after a boundary), the channel itself (named as the line names it, capacity the
  line's depth), its input adapter's buffer as a ``ReorderStage`` (its output direct), a
  memory source as a ``Memory`` stage (paced by its consumer: it presents a word whenever its
  channel has room), a boundary's free side from the line's own pace for it (the pace the
  outside states, else its mover's): a ``Source`` released at the partition's bottleneck
  period on the input; on the output a ``Kernel`` reading at that pace, which is the sink.
  Every channel the line does not size (a width step's input, a generator's output) is direct.
  A channel the line sized with ``None``, or did not model, is direct, as the strategy
  proposes it. A part that cannot be translated (a port that states no pace, an adapter chain
  the line does not model, a channel the strategy did not size, a network whose channels'
  word counts disagree) is named and the partition's calls are ``untranslatable``.

**The classes**, for each call whose line gave a depth or ``None``:

1. *The network at the line's depths.* It keeps the period (``timing.period_kept``: the sources
   on time and every end by its due cycles), or it does not: then the call
   has a finding about the line (``network_not_kept``), with the run (the deadlock, the source
   late and by how much, the periods at the ends) and the channels each of which alone, raised
   to unbounded, restores the period. A period lost only through the memory channels (restored
   by raising only those) is said so.
2. *Per channel.* The network's ``least`` for that channel with every other channel at the
   line's depth, against the line's depth: ``same``; ``smaller`` (the run at the package's depth
   keeps the period); ``larger_latency_counted`` (larger, with a stated latency in the network,
   named; none in the harness, where every latency is 0); else ``differs``. A line answer of
   ``None`` (no depth up to the frames' words keeps the period) is ``same`` if the network with
   that channel at that many words also loses the period or deadlocks, else ``differs``. Each
   ``smaller``, ``larger`` and ``differs`` is listed by key with its run: the period of the
   network's ends and the source's lateness at the line's depth and at the package's.
3. *The joint minimal set* (``timing.minimal``) of a partition's network, reported apart,
   with its order and total, not classed.

**What fails the run: a broken tool, and only that** (``Fault``). A difference, a finding or a
count apart does not. The faults:

- a part that cannot be translated (``untranslatable``: one the package's model cannot take,
  as a source that presents two words in one cycle);
- a call with sides only (a two-stage network) where, at the line's depth, the run's word
  times or its sources' verdict (``Run.on_time``, the line's ``within``) differ from
  ``fifo_sizing.simulate`` and ``within`` on the same sides, word by word (when each word left
  the producer and was taken, against ``Flow``); or where the package's ``least`` differs from
  ``least_depth`` while the network is not slower than the period (below);
- a crash in the comparison (``crashed``);
- a strategy's own result that does not keep the period by its own criterion: a ``smaller``
  whose run at the package's depth does not keep it, or a joint minimal set whose run loses it.

**Findings**, counted and listed, never failing: ``network_not_kept`` (the network at the
line's depths does not keep the period), ``period_other_than_p`` (a partition's bottleneck is
not the period the line sized it at), ``not_modelled``, and ``slower_than_period``: a call with
sides only that is slower than the period: its bottleneck is above the period, or the run with
the FIFO unbounded and the generator's output as the network states it loses the rate at its end
(``period_kept``), as a generator whose direct output makes it slower than its all-unbounded
bottleneck (the line gives such a call a depth that only delays the loss). That call is classed
``slower_than_period`` and listed with the period, the line's depth, the network's bottleneck
and its period at the line's depth; it gets no ``network_not_kept`` finding for the same run.

**Counted apart, never compared**: a call the line refused (``raised``: its producer slower
than the period, by its reason), a channel the line does not model (``not_modelled``: by its
reason), and a call made while ``least_depth`` or ``sides`` was a test's stand-in
(``stubbed``: the answer is the stand-in's). A partition with a part the line does not pace
(a boundary it does not model, a port stating no pace, an adapter chain it does not read) has no
whole network: it is counted apart too, as one ``not_modelled`` entry named ``partition`` with
the part's reason, and its channels are compared by their own sides. A partition with a channel
sized under a stand-in has no depths of the line's: its channels are ``stubbed``.

**Coverage** counts the rows actually compared, by depth (0, above 0) and acceptance
(``Pattern``, ``Buffered``): a row classed ``same``, ``smaller``, ``larger_latency_counted`` or
``differs`` against a real line depth that the package sized. Not a refused, stubbed or not
modelled row, not a finding, not a line answer of ``None``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

from dataflow.harness.fifo_record import (
    DEPTH,
    NONE,
    NOT_MODELLED,
    RAISED,
    STUBBED,
    Acceptance,
    Beats,
    Edge,
    FifoCall,
    Line,
    Nest,
    Partition,
    Port,
    Stored,
    Unpaced,
)
from finn.dataflow import timing
from finn.dataflow.schedule import Pace
from finn.dataflow.timing import (
    FINNLIB,
    Channel,
    Kernel,
    Memory,
    Network,
    ReorderStage,
    Run,
    Source,
    Stage,
    WidthStage,
)
from finn.kernels.fifo_sizing import (
    FRAMES,
    Buffered,
    Pattern,
    converted,
    simulate,
    within,
)
from finn.kernels.input_generator import NestBuffer

SAME, SMALLER, LATENCY = "same", "smaller", "larger_latency_counted"
DIFFERS, SLOWER = "differs", "slower_than_period"
UNTRANSLATABLE, CRASHED = "untranslatable", "crashed"
CLASSES = (
    SAME,
    SMALLER,
    LATENCY,
    DIFFERS,
    SLOWER,
    UNTRANSLATABLE,
    CRASHED,
    NOT_MODELLED,
    RAISED,
    STUBBED,
)
"""Every class a call can get: the compared ones first, the finding, the two that come of a
broken tool, then the three counted apart."""

COMPARED = (SAME, SMALLER, LATENCY, DIFFERS)
"""The classes of a row the package compared against a depth of the line's."""

APART = (NOT_MODELLED, RAISED, STUBBED)
REFUSED = "refused"
"""What the package answers where it will not size a network (its bottleneck is not the period)."""

NOT_KEPT, OTHER_PERIOD = "network_not_kept", "period_other_than_p"
"""The findings besides ``SLOWER`` and ``NOT_MODELLED``."""

SUPPLY, FIFO, ACCEPT = "supply", "fifo", "accept"
"""The names of the two-stage network's stages and its channel."""


class Untranslatable(ValueError):
    """A part of a call with no counterpart in the package's network model, and why."""


class Unmodelled(Untranslatable):
    """A part of a partition the line itself does not model (an end it does not pace, a port
    that states no pace, an adapter chain beside one ``vpc`` and one ``input_gen``): no
    network of the whole partition is defined, though each channel's own sides may be."""


# -- the translation -----------------------------------------------------------------------


def _source(name: str, beats: Beats, period: int) -> Source:
    try:
        return Source(name, beats.times, beats.span, period)
    except (ValueError, TypeError) as error:
        raise Untranslatable(f"{name}: {error}") from error


def _reads(name: str, beats: Beats) -> Kernel:
    """A kernel that reads at ``beats``, its first read at step 0 (the line's consumer has no
    phase) and its steps the span: a gap from its last read of a frame to its first of the
    next is the line's."""
    first = beats.times[0]
    try:
        return Kernel(name, beats.span, (tuple(each - first for each in beats.times),), ())
    except (ValueError, TypeError) as error:
        raise Untranslatable(f"{name}: {error}") from error


def _generator(
    name: str, capacity: int, reads: tuple[int, ...], freed: tuple[int, ...]
) -> ReorderStage:
    try:
        return ReorderStage(name, reads, freed, capacity)
    except (ValueError, TypeError) as error:
        raise Untranslatable(f"{name}: {error}") from error


def _network(stages: Sequence[Stage], channels: Sequence[Channel]) -> Network:
    try:
        return Network(tuple(stages), tuple(channels))
    except (ValueError, TypeError) as error:
        raise Untranslatable(str(error)) from error


def repeated(
    reads: tuple[int, ...], freed: tuple[int, ...], taken: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A buffer's frame run ``taken // len(reads)`` times in a frame of its consumer: the
    line counts its words across frames, so each run's words and frees follow the one before's
    (its consumer's frame holds several of the buffer's, the buffer's reads over and over)."""
    if taken % len(reads):
        raise Untranslatable(
            f"the buffer presents {len(reads)} beats a frame and its reader takes {taken}"
        )
    words = freed[-1]
    times = range(taken // len(reads))
    return (
        tuple(read + each * words for each in times for read in reads),
        tuple(free + each * words for each in times for free in freed),
    )


def sides_network(
    supply: Beats, accept: Acceptance, period: int, capacity: int | None = None
) -> Network:
    """A call with sides only as a network: the supply, channel ``fifo`` of ``capacity``
    words (``None``: unbounded), the acceptance."""
    source = _source(SUPPLY, supply, period)
    if isinstance(accept, Beats):
        return _network(
            (source, _reads(ACCEPT, accept)), (Channel(FIFO, SUPPLY, ACCEPT, capacity),)
        )
    reads, freed = repeated(accept.reads, accept.freed, len(accept.reader.times))
    generator = _generator("generator", accept.capacity, reads, freed)
    return _network(
        (source, generator, _reads(ACCEPT, accept.reader)),
        (
            Channel(FIFO, SUPPLY, "generator", capacity),
            Channel("generated", "generator", ACCEPT, 0),
        ),
    )


def partition_network(partition: Partition, depths: Mapping[str, int | None]) -> Network:
    """The whole of ``partition`` as one network, each channel of ``depths`` at its capacity
    (a channel not named is direct); ``Untranslatable`` naming the part that cannot be."""
    if partition.refused:
        raise Untranslatable(partition.refused)
    reads: dict[str, list[Any]] = {name: [] for name in partition.kernels}
    writes: dict[str, list[Any]] = {name: [] for name in partition.kernels}
    ends: list[Stage] = []  # sources, memories, and what stands between a kernel and its channel
    adapters: list[Stage] = []
    sinks: list[Stage] = []
    channels: list[Channel] = []

    def pace(port: Port) -> Pace:
        if port.pace is None:
            raise Unmodelled(f"kernel {port.kernel}'s port {port.port} states no pace")
        if port.kernel not in partition.kernels:
            raise Untranslatable(f"a port of {port.kernel}, which is no kernel of the partition")
        return port.pace

    for wire in partition.wires:
        name = wire.channel
        if wire.refused:
            raise Unmodelled(f"channel {name}: {wire.refused}")
        capacity = depths.get(name, 0)
        # The producer's end: the stage the channel comes out of, and its output port.
        origin, output = "", 0
        match wire.producer:
            case Port() as port:
                origin, output = port.kernel, len(writes.get(port.kernel, ()))
                writes.setdefault(port.kernel, []).append(pace(port))
                if wire.convert is not None:
                    width = WidthStage(f"{name}.width", *wire.convert)
                    adapters.append(width)
                    channels.append(Channel(f"{name}.in", origin, width.name, 0, output))
                    origin, output = width.name, 0
            case Edge() as edge:
                beats = edge.beats
                if wire.convert is not None:
                    times = converted(beats.times, *wire.convert)
                    beats = Beats(times, max(beats.span, times[-1] - times[0] + 1))
                feed = _source(f"{name}.source", beats, partition.period)
                ends.append(feed)
                origin = feed.name
            case Stored():
                ends.append(Memory(f"{name}.memory"))
                origin = f"{name}.memory"
            case Unpaced() as unpaced:
                raise Unmodelled(f"channel {name}'s producer: {unpaced.reason}")
        # The consumer's end: the stage the channel goes into, and its input port.
        target, entering = "", 0
        match wire.consumer:
            case Port() as port:
                target, entering = port.kernel, len(reads.get(port.kernel, ()))
                reads.setdefault(port.kernel, []).append(pace(port))
            case Edge() as edge:
                sink = _reads(f"{name}.sink", edge.beats)
                sinks.append(sink)
                target = sink.name
            case Unpaced() as unpaced:
                raise Unmodelled(f"channel {name}'s consumer: {unpaced.reason}")
            case _:
                raise Untranslatable(f"channel {name}'s consumer is no kernel's port or edge")
        if wire.buffer is None:
            channels.append(Channel(name, origin, target, capacity, output, entering))
            continue
        buffer = wire.buffer
        taken = len(wire.consumer.beats.times) if isinstance(wire.consumer, Edge) else None
        if isinstance(wire.consumer, Port):
            assert wire.consumer.pace is not None
            taken = len(wire.consumer.pace.times)
        assert taken is not None
        words, freed = repeated(buffer.reads, buffer.freed, taken)
        generator = _generator(f"{name}.generate", buffer.capacity, words, freed)
        adapters.append(generator)
        channels.append(Channel(name, origin, generator.name, capacity, output))
        channels.append(Channel(f"{name}.generated", generator.name, target, 0, 0, entering))
    kernels: list[Stage] = []
    for name in partition.kernels:
        if reads[name] or writes[name]:
            try:
                kernels.append(timing.kernel(name, reads[name], writes[name]))
            except (ValueError, TypeError) as error:
                raise Untranslatable(f"kernel {name}: {error}") from error
    return _network((*ends, *kernels, *adapters, *sinks), channels)


# -- the classes ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One call's class: ``key``, ``period`` and the line's answer, the package's answer
    (``least``, or why it has none), ``verdict`` (one of ``CLASSES``) and ``detail`` (for a
    difference, its run). ``depth`` and ``acceptance`` are where it lies in the coverage;
    ``count`` is how many times it was made."""

    key: str
    period: int
    line: str
    package: str
    verdict: str
    detail: str
    depth: str = ""
    acceptance: str = ""
    count: int = 1


@dataclass(frozen=True)
class Finding:
    """A finding about the line, never a failure. ``kind`` is ``network_not_kept`` (a network
    at the line's depths that does not keep the period: ``run`` says how, ``restored_by`` the
    channels that each, raised to unbounded alone, restore it, ``memory_only`` that raising
    only the memory channels does), ``period_other_than_p`` (a partition whose bottleneck is
    not the period the line sized it at) or ``slower_than_period`` (a call with sides only
    whose bottleneck is above the period: ``period``, the line's ``depth``, the network's
    ``bottleneck`` and its period ``at_depth``, the line's depth)."""

    key: str
    run: str
    restored_by: tuple[str, ...] = ()
    memory_only: bool = False
    kind: str = NOT_KEPT
    period: int = 0
    depth: int | None = None
    bottleneck: int | None = None
    at_depth: int | None = None


@dataclass(frozen=True)
class Fault:
    """A broken tool: what fails the run (the module's list), by the call's ``key``."""

    key: str
    reason: str


@dataclass(frozen=True)
class Joint:
    """A partition's joint minimal set (``timing.minimal``), not classed: its channels in the
    order they were lowered with their capacities, the total and the passes, and whether its run
    keeps the period (``kept``); or ``refused``, why the network cannot be sized."""

    key: str
    period: int
    order: tuple[str, ...] = ()
    capacities: tuple[int, ...] = ()
    total: int = 0
    passes: int = 0
    refused: str = ""
    kept: bool | None = None


class Judged(NamedTuple):
    """What judging a call, or a partition's calls, gave: the entries, the findings, the faults
    and, for a partition, its joint minimal set."""

    entries: list[Entry]
    findings: list[Finding]
    faults: list[Fault]
    joint: Joint | None = None


def shown(network: Network, frames: int) -> tuple[Run, str]:
    """The network's run and a sentence on it: the deadlock; or each source's worst lateness
    against its idle time and the period at each end."""
    found = timing.run(network, FINNLIB, frames)
    if found.deadlocked:
        return found, "deadlocks"
    idle = dict(found.idle)
    sources = ", ".join(
        f"{name} late by {max(late, default=0)} (idle {idle[name]})" for name, late in found.late
    )
    periods = ", ".join(f"{end} every {found.period(end)}" for end in timing.ends(network))
    return found, f"{sources}; {periods}"


def _keeping(network: Network, frames: int) -> Callable[[Run], bool]:
    """The network's criterion (``timing.period_kept``, under FinnLib's rules). Where it has
    no due cycles (its bottleneck is not its sources', or it deadlocks unbounded), no run
    keeps the period; a network with no source is held to nothing but being on time."""
    try:
        return timing.period_kept(network, FINNLIB, frames)
    except ValueError:
        if not any(isinstance(stage, Source) for stage in network.stages):
            return lambda found: found.on_time
        return lambda found: False


def _ending(network: Network, found: Run) -> str:
    return ", ".join(
        f"{end} completes a frame every {found.period(end)}" for end in timing.ends(network)
    )


def _latency(network: Network) -> int:
    """The latency any kernel states (0 in the harness)."""
    return max((s.latency for s in network.stages if isinstance(s, Kernel)), default=0)


def _runs(network: Network, channel: str, line: int, found: int, frames: int) -> tuple[str, bool]:
    """A difference's runs, the network at the line's ``line`` words and at the package's
    ``found`` (each as ``shown``), and whether the second keeps the period."""
    _, at_line = shown(network.at({channel: line}), frames)
    held, at_found = shown(network.at({channel: found}), frames)
    return f"at the line's {line} words: {at_line}; at the package's {found} words: {at_found}", (
        _keeping(network, frames)(held)
    )


def judge(network: Network, channel: str, line: Line, frames: int) -> tuple[str, str, str, str]:
    """A channel's class (the module's step 2) in ``network``, its other channels at the line's
    depths: the package's answer, the class, the detail and the fault (empty, or why a result
    does not keep the period by its own criterion). ``line`` is a depth or ``none``."""
    words = dict(network.words)[channel]
    bound = frames * words
    if line.kind == NONE:
        upper, text = shown(network.at({channel: bound}), frames)
        if _keeping(network, frames)(upper):
            return (
                f"{bound} words keep the period",
                DIFFERS,
                f"the line finds no depth up to {bound} words; the network keeps the period "
                f"with {bound}: {_ending(network, upper)}",
                "",
            )
        return f"none up to {bound} words", SAME, text, ""
    assert line.depth is not None
    try:
        found = timing.least(network, channel, FINNLIB, frames)
    except ValueError as error:
        return REFUSED, DIFFERS, str(error), ""
    if found is None:
        _, text = shown(network.at({channel: None}), frames)
        return "none", DIFFERS, f"no capacity keeps the period: {text}", ""
    if found == line.depth:
        return f"{found} words", SAME, "", ""
    runs, kept = _runs(network, channel, line.depth, found, frames)
    if found < line.depth:
        if not kept:
            return (
                f"{found} words",
                DIFFERS,
                runs,
                f"its least depth, {found} words, does not keep the period",
            )
        return f"{found} words", SMALLER, runs, ""
    latency = _latency(network)
    if latency:
        return (
            f"{found} words",
            LATENCY,
            f"the line's {line.depth} words, the network's {found}: a latency of {latency} "
            f"cycles; {runs}",
            "",
        )
    return f"{found} words", DIFFERS, f"larger than the line's {line.depth} words; {runs}", ""


def _kinds(call: FifoCall) -> tuple[str, str]:
    """The call's depth and acceptance, for the coverage."""
    depth = "none" if call.line.kind == NONE else ""
    if call.line.kind == DEPTH:
        depth = "0" if not call.line.depth else "above 0"
    kind = (
        "" if call.accept is None else ("Buffered" if isinstance(call.accept, Nest) else "Pattern")
    )
    return depth, kind


def _entry(call: FifoCall, count: int, package: str, verdict: str, detail: str) -> Entry:
    depth, kind = _kinds(call)
    return Entry(
        call.key, call.period, call.line.text(), package, verdict, detail, depth, kind, count
    )


def _apart(call: FifoCall, count: int) -> Entry | None:
    """A call counted apart (the line refused it, does not model the channel, or answered
    through a stand-in)."""
    if call.line.kind in APART:
        return _entry(call, count, "", call.line.kind, call.line.reason)
    return None


def _bottleneck(network: Network, frames: int) -> int | None:
    """The network's bottleneck period; None where it deadlocks with every channel unbounded."""
    try:
        return timing.bottleneck(network, FINNLIB, frames)
    except ValueError:
        return None


def _first(name: str, package: Sequence[int], line: Sequence[int]) -> str:
    """The first of ``name``'s words at which the two disagree, said."""
    at = next((i for i, (a, b) in enumerate(zip(package, line)) if a != b), None)
    if at is None:
        return f"{name}: the package has {len(package)} words, the line {len(line)}"
    return f"{name}: word {at} is at cycle {package[at]} in the package, {line[at]} in the line"


def line_acceptance(accept: Acceptance) -> Pattern | Buffered:
    """The line's acceptance (``Pattern`` or ``Buffered``) of a recorded one."""
    if isinstance(accept, Beats):
        return Pattern(accept.times, accept.span)
    return Buffered(
        NestBuffer(accept.capacity, accept.reads, accept.freed),
        Pattern(accept.reader.times, accept.reader.span),
    )


def against_the_line(call: FifoCall, found: Run) -> list[Fault]:
    """The two-stage run at the line's depth against ``fifo_sizing.simulate`` and ``within``
    on the same sides: each word's cycle leaving the producer and taken, and the kept verdict."""
    assert call.supply is not None and call.accept is not None and call.line.depth is not None
    supply = Pattern(call.supply.times, call.supply.span)
    flow = simulate(supply, line_acceptance(call.accept), call.period, call.line.depth, call.frames)
    faults: list[str] = []
    if flow is None or found.deadlocked:
        if (flow is None) != found.deadlocked:
            faults.append(
                f"the line {'deadlocks' if flow is None else 'runs'}, "
                f"the package {'deadlocks' if found.deadlocked else 'runs'}"
            )
    else:
        if found.written(FIFO) != flow.left:
            faults.append(_first("words leaving", found.written(FIFO), flow.left))
        if found.taken(FIFO) != flow.taken:
            faults.append(_first("words taken", found.taken(FIFO), flow.taken))
    kept = flow is not None and within(flow.late, supply, call.period)
    if kept != found.on_time:
        faults.append(f"the line {'keeps' if kept else 'loses'} the period, the package the other")
    return [Fault(call.key, f"at the line's depth: {text}") for text in faults]


def judge_sides(call: FifoCall, count: int) -> Judged:
    """A call with sides only: the two-stage network, the module's steps 1 and 2, against the
    line's ``simulate`` at its depth (a fault where they differ), and where the package's
    ``least`` differs from the line's (a fault, unless the network's bottleneck is above the
    period: ``slower_than_period``)."""
    apart = _apart(call, count)
    if apart is not None:
        return Judged([apart], [], [])
    try:
        assert call.supply is not None and call.accept is not None
        capacity = call.line.depth if call.line.kind == DEPTH else None
        network = sides_network(call.supply, call.accept, call.period, capacity)
    except Untranslatable as error:
        entry = _entry(call, count, "", UNTRANSLATABLE, str(error))
        return Judged([entry], [], [Fault(call.key, f"untranslatable: {error}")])
    package, verdict, detail, broken = judge(network, FIFO, call.line, call.frames)
    findings: list[Finding] = []
    faults: list[Fault] = [Fault(call.key, broken)] if broken else []
    found: Run | None = None
    text = ""
    if call.line.kind == DEPTH:
        found, text = shown(network, call.frames)
        faults.extend(against_the_line(call, found))
    bottleneck = _bottleneck(network, call.frames)
    above = bottleneck is not None and bottleneck > call.period
    # The run with the FIFO unbounded and the generator's output as the network states it: a
    # generator whose direct output is slower than the period loses the rate here though its
    # bottleneck, which runs that output unbounded too, is the period.
    stated, stated_text = shown(network.at({FIFO: None}), call.frames)
    loses = bottleneck is not None and not _keeping(network, call.frames)(stated)
    slower = above or loses
    # ``least`` refuses a network whose bottleneck is above the period, and is None where the
    # run with the FIFO unbounded loses it (a ``refused`` or a ``differs`` row), so a
    # ``smaller`` or latency row is never slower than the period: no row falls between the
    # finding and the fault below.
    assert not (slower and verdict in (SMALLER, LATENCY)), (call.key, verdict, bottleneck)
    if verdict == DIFFERS and slower and not broken:
        at_depth = None
        if found is not None and not found.deadlocked:
            periods = [found.period(end) for end in timing.ends(network)]
            at_depth = max((each for each in periods if each is not None), default=None)
        said = "deadlocks with every channel unbounded" if bottleneck is None else str(bottleneck)
        if above:
            run_text = f"the bottleneck is {said}, above the period {call.period}"
            detail = (
                f"the network's bottleneck is {bottleneck} cycles over {call.frames} frames, "
                f"above the period {call.period}"
            )
        else:
            run_text = (
                f"with the FIFO unbounded and the generator's output as stated the run loses "
                f"the rate at its end ({stated_text}); the bottleneck is {said}, the period "
                f"{call.period}"
            )
            detail = (
                f"with the FIFO unbounded and the generator's output as stated the network "
                f"loses the rate at its end ({stated_text}) over {call.frames} frames, the "
                f"period {call.period}; its bottleneck with every channel unbounded is {said}"
            )
        findings.append(
            Finding(
                call.key,
                f"{run_text}; the line answers {call.line.text()}",
                kind=SLOWER,
                period=call.period,
                depth=call.line.depth,
                bottleneck=bottleneck,
                at_depth=at_depth,
            )
        )
        verdict = SLOWER
        detail += f"; at the line's {call.line.depth} words it is {at_depth}" if at_depth else ""
    elif verdict != SAME and not broken and not slower:
        faults.append(
            Fault(
                call.key,
                f"least differs from the line's with the bottleneck {bottleneck} not above the "
                f"period {call.period}: the package {package}, the line {call.line.text()} "
                f"({verdict}: {detail})",
            )
        )
    # A call slower than the period gets no second finding for the same run.
    if found is not None and verdict != SLOWER and not _keeping(network, call.frames)(found):
        findings.append(_lost(call.key, network, text, call.frames))
    return Judged([_entry(call, count, package, verdict, detail)], findings, faults)


def judge_partition(group: Sequence[tuple[FifoCall, int]]) -> Judged:
    """The calls a strategy made over one partition, together: the whole network at the
    line's depths (step 1), each channel against its ``least`` (step 2), the joint minimal set
    (step 3)."""
    first = group[0][0]
    partition, frames = first.partition, first.frames
    assert partition is not None
    key = first.station
    depths = {call.channel: call.line.depth or 0 for call, _ in group if call.line.kind == DEPTH}
    sized = {call.channel for call, _ in group}
    entries: list[Entry] = []
    pending: list[tuple[FifoCall, int]] = []
    for call, count in group:
        apart = _apart(call, count)
        if apart is not None:
            entries.append(apart)
        else:
            pending.append((call, count))
    if any(call.line.kind == STUBBED for call, _ in group):
        # A channel sized under a stand-in has no depth of the line's: no network at its depths.
        why = "a channel of its partition was sized under a stand-in"
        entries.extend(_entry(call, count, "", STUBBED, why) for call, count in pending)
        return Judged(entries, [], [])
    try:
        network = partition_network(partition, depths)
        for wire in partition.wires:
            if wire.channel not in sized:
                raise Untranslatable(f"channel {wire.channel} is not sized by these calls")
    except Unmodelled as error:
        # The line does not pace a part of the partition: the whole is not defined, each
        # channel's own sides are, and are compared alone.
        entries.append(
            Entry(f"{key} :: partition", partition.period, "", "", NOT_MODELLED, str(error))
        )
        alone = Judged(entries, [], [])
        for call, count in pending:
            judged = judge_sides(call, count)
            alone.entries.extend(judged.entries)
            alone.findings.extend(judged.findings)
            alone.faults.extend(judged.faults)
        return alone
    except Untranslatable as error:
        entries.extend(
            _entry(call, count, "", UNTRANSLATABLE, str(error)) for call, count in pending
        )
        return Judged(entries, [], [Fault(key, f"untranslatable: {error}")])
    findings: list[Finding] = []
    faults: list[Fault] = []
    held, text = shown(network, frames)
    if not _keeping(network, frames)(held):
        findings.append(_lost(key, network, text, frames))
    bottleneck = _bottleneck(network, frames)
    if bottleneck != partition.period:
        said = "deadlocks with every channel unbounded" if bottleneck is None else str(bottleneck)
        findings.append(
            Finding(
                key,
                f"the network's bottleneck is {said}, the period the line sized it at is "
                f"{partition.period}",
                kind=OTHER_PERIOD,
                period=partition.period,
                bottleneck=bottleneck,
            )
        )
    for call, count in pending:
        package, verdict, detail, broken = judge(network, call.channel, call.line, frames)
        entries.append(_entry(call, count, package, verdict, detail))
        if broken:
            faults.append(Fault(call.key, broken))
    joint, lost = _joint(key, partition.period, network, frames)
    return Judged(entries, findings, [*faults, *lost], joint)


def _lost(key: str, network: Network, text: str, frames: int) -> Finding:
    """What raising a channel to unbounded restores, for a network that loses the period."""
    memory = [
        channel.name
        for channel in network.channels
        if isinstance(network.stage(channel.producer), Memory)
    ]
    keeping = _keeping(network, frames)
    restored = tuple(
        channel.name
        for channel in network.channels
        if channel.capacity is not None
        and keeping(timing.run(network.at({channel.name: None}), FINNLIB, frames))
    )
    only = bool(memory) and keeping(timing.run(network.at(dict.fromkeys(memory)), FINNLIB, frames))
    return Finding(key, text, restored, only)


def _joint(key: str, period: int, network: Network, frames: int) -> tuple[Joint, list[Fault]]:
    """The joint minimal set and, as a fault, its run losing the period."""
    try:
        sizing = timing.minimal(network, FINNLIB, frames)
    except ValueError as error:
        return Joint(key, period, refused=str(error)), []
    held, text = shown(network.at(sizing.found), frames)
    kept = _keeping(network, frames)(held)
    joint = Joint(
        key, period, sizing.order, sizing.capacities, sizing.total, sizing.passes, kept=kept
    )
    if kept:
        return joint, []
    return joint, [Fault(f"{key} :: joint", f"the minimal set loses the period: {text}")]


# -- the table -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FifoTable:
    """One source's FIFO comparison: its entries, the findings about the line, the faults (a
    broken tool: they fail the run), the joint minimal sets and the calls made."""

    calls: int
    calls_made: int
    entries: tuple[Entry, ...]
    findings: tuple[Finding, ...]
    joint: tuple[Joint, ...]
    faults: tuple[Fault, ...] = ()
    frames: int = FRAMES

    @property
    def counts(self) -> dict[str, int]:
        """Each class's count over the distinct calls, every class listed, zeros included."""
        found = Counter(entry.verdict for entry in self.entries)
        return {name: found.get(name, 0) for name in CLASSES}

    @property
    def apart(self) -> dict[str, dict[str, int]]:
        """What is counted apart or untranslatable, by class and reason."""
        found: dict[str, Counter[str]] = {name: Counter() for name in (*APART, UNTRANSLATABLE)}
        for entry in self.entries:
            if entry.verdict in found:
                found[entry.verdict][entry.detail] += 1
        return {name: dict(reasons) for name, reasons in found.items()}

    @property
    def finding_counts(self) -> dict[str, int]:
        """The findings by kind, zeros included (``not_modelled``: the entries so classed)."""
        found = Counter(finding.kind for finding in self.findings)
        return {
            NOT_KEPT: found.get(NOT_KEPT, 0),
            OTHER_PERIOD: found.get(OTHER_PERIOD, 0),
            SLOWER: found.get(SLOWER, 0),
            NOT_MODELLED: self.counts[NOT_MODELLED],
        }

    @property
    def coverage(self) -> dict[str, int]:
        """The rows compared, by depth (0, above 0) and acceptance (``Pattern``,
        ``Buffered``): a row classed ``same``, ``smaller``, ``larger_latency_counted`` or
        ``differs`` against a real depth of the line's that the package sized. Not a refused,
        stubbed, not modelled or ``slower_than_period`` row, nor a line answer of ``None``."""
        found = Counter(
            (entry.depth, entry.acceptance)
            for entry in self.entries
            if entry.verdict in COMPARED
            and entry.package != REFUSED
            and entry.depth in ("0", "above 0")
            and entry.acceptance
        )
        return {
            f"{depth} {kind}": found.get((depth, kind), 0)
            for depth in ("0", "above 0")
            for kind in ("Pattern", "Buffered")
        }

    @property
    def gap(self) -> str:
        """The coverage's gap, said, or empty."""
        if self.coverage["above 0 Buffered"]:
            return ""
        return "no compared row has a Buffered acceptance above depth 0"

    @property
    def differences(self) -> tuple[Entry, ...]:
        """Every call the package answers otherwise than the line, or finds slower than the
        period, in the order met: each with its run in ``detail``."""
        return tuple(
            entry for entry in self.entries if entry.verdict in (SMALLER, LATENCY, DIFFERS, SLOWER)
        )

    @property
    def passed(self) -> bool:
        return not self.faults

    def summary(self) -> str:
        counts = ", ".join(f"{name} {count}" for name, count in self.counts.items())
        text = f"fifo: {self.calls} distinct calls ({self.calls_made} made), {counts}"
        kinds = ", ".join(f"{name} {count}" for name, count in self.finding_counts.items())
        text += f"; findings: {kinds}"
        if self.faults:
            text += f"; {len(self.faults)} faults"
        return f"{text}; {self.gap}" if self.gap else text

    def lines(self) -> list[str]:
        """The differences and the faults, one a line, for a log."""
        return [
            *(
                f"fifo difference ({entry.verdict}) {entry.key}: line {entry.line}, package "
                f"{entry.package}; {entry.detail}"
                for entry in self.differences
            ),
            *(f"fifo FAULT {fault.key}: {fault.reason}" for fault in self.faults),
        ]

    def report(self) -> dict[str, Any]:
        """The table's section as plain values (the harness's JSON)."""
        return {
            "rules": "FINNLIB",
            "frames": self.frames,
            "distinct_calls": self.calls,
            "calls_made": self.calls_made,
            "counts": self.counts,
            "apart": self.apart,
            "coverage": self.coverage,
            "gap": self.gap,
            "finding_counts": self.finding_counts,
            "findings": [asdict(finding) for finding in self.findings],
            "differences": [asdict(entry) for entry in self.differences],
            "faults": [asdict(fault) for fault in self.faults],
            "joint": [asdict(each) for each in self.joint],
            "entries": [asdict(entry) for entry in self.entries],
            "passed": self.passed,
        }


def table(calls: Sequence[tuple[FifoCall, int]]) -> FifoTable:
    """Every recorded call classed (the module's classes). A crash in judging a call, or a
    partition's calls, is a fault of the tool: it is named, and the calls are ``crashed``."""
    entries: list[Entry] = []
    findings: list[Finding] = []
    faults: list[Fault] = []
    joint: list[Joint] = []
    groups: dict[tuple[str, Partition], list[tuple[FifoCall, int]]] = {}
    for call, count in calls:
        if call.partition is not None:
            groups.setdefault((call.station, call.partition), []).append((call, count))
            continue
        try:
            judged = judge_sides(call, count)
        except Exception as error:  # noqa: BLE001 - a crash of the comparison is a fault
            judged = _crashed(call, count, error)
        entries.extend(judged.entries)
        findings.extend(judged.findings)
        faults.extend(judged.faults)
    for group in groups.values():
        try:
            judged = judge_partition(group)
        except Exception as error:  # noqa: BLE001
            judged = Judged(
                [
                    _entry(call, count, "", CRASHED, f"{type(error).__name__}: {error}")
                    for call, count in group
                ],
                [],
                [Fault(group[0][0].station, f"crash: {type(error).__name__}: {error}")],
            )
        entries.extend(judged.entries)
        findings.extend(judged.findings)
        faults.extend(judged.faults)
        joint.extend([] if judged.joint is None else [judged.joint])
    return FifoTable(
        len(calls),
        sum(count for _, count in calls),
        tuple(entries),
        tuple(findings),
        tuple(joint),
        tuple(faults),
    )


def _crashed(call: FifoCall, count: int, error: Exception) -> Judged:
    said = f"{type(error).__name__}: {error}"
    return Judged([_entry(call, count, "", CRASHED, said)], [], [Fault(call.key, f"crash: {said}")])
