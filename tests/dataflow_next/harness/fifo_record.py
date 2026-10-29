# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every FIFO sizing FINN asks of ``finn.kernels.fifo_sizing`` while a source runs, recorded.

``FifoRecorder`` wraps ``size``, ``least_depth`` and ``sides`` of that module, in it and in
every loaded module that imported them by name. A call of ``size`` or ``least_depth`` (a
``least_depth`` inside a ``size`` is that ``size``'s own) is kept as one ``FifoCall``, all
of it plain values (tuples, integers and strings, no FINN object but the ``Pace`` of a
kernel's port, a frozen value):

- where it was met (``station``: a sequence's partition, or the test that made the call) and
  its channel's name (``channel``: the partition's, else the sides' owners);
- its two sides (``supply``: ``Beats``; ``accept``: ``Beats``, or ``Nest``, an input
  generator's buffer and its reader's beats), the period and the frames;
- the line's answer (``Line``): the least depth in words (not the DEPTH ``size`` proposes,
  which is storage, not timing), ``None`` where no depth keeps the period, the reason a
  channel is not modelled, the exception raised and its message, or that the call was made
  while ``least_depth``, ``sides``, or what ``least_depth`` calls (``keeps``, ``simulate``,
  ``within``) of that module was a test's stand-in (``stubbed``: the answer is the
  stand-in's, not the line's, so the call is recorded and never compared);
- for a ``size`` a strategy makes (``SizeFifos.search``), the whole partition it sizes
  (``Partition``): each kernel's ports' paces and each channel's ends, adapters and
  buffer, read once as the search first asks, so that a network of the whole partition can
  be built from it (``fifo``).

Each distinct call is kept once, with the number of times it was made. Recording changes no
answer: the wrapper returns what the function returns and raises what it raises.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType
from typing import Any

from finn.dataflow.schedule import Pace

DEPTH = "depth"
NONE = "none"
NOT_MODELLED = "not_modelled"
RAISED = "raised"
STUBBED = "stubbed"


@dataclass(frozen=True)
class Beats:
    """One side's beat times over a frame and its span (the line's ``Pattern``)."""

    times: tuple[int, ...]
    span: int


@dataclass(frozen=True)
class Nest:
    """An input generator's buffer (the line's ``NestBuffer``: its capacity, the frame's
    word each beat reads, the words freed after each) and the beats its reader takes
    (the line's ``Buffered``)."""

    capacity: int
    reads: tuple[int, ...]
    freed: tuple[int, ...]
    reader: Beats


Acceptance = Beats | Nest


@dataclass(frozen=True)
class Line:
    """What the line answered: ``kind`` ``depth`` (``depth`` words), ``none`` (no depth up
    to the frames run keeps the period), ``not_modelled``, ``raised`` or ``stubbed``
    (``reason``: the reason, the exception's class and message, or the stand-ins' names)."""

    kind: str
    depth: int | None = None
    reason: str = ""

    def text(self) -> str:
        if self.kind == DEPTH:
            return f"{self.depth} words"
        return self.kind if not self.reason else f"{self.kind}: {self.reason}"


@dataclass(frozen=True)
class Port:
    """A kernel's port at a channel's end, and the pace its contract states (``None``:
    none: a sequence no schedule paces)."""

    kernel: str
    port: str
    pace: Pace | None


@dataclass(frozen=True)
class Edge:
    """A boundary's free side, paced as the line paces it (the pace the outside states, else
    its mover's)."""

    beats: Beats


@dataclass(frozen=True)
class Stored:
    """A memory source, paced by its consumer."""


@dataclass(frozen=True)
class Unpaced:
    """An end the line does not pace, and why."""

    reason: str


End = Port | Edge | Stored | Unpaced


@dataclass(frozen=True)
class Buffer:
    """An input adapter's ``input_gen``: the line's ``NestBuffer`` as tuples."""

    capacity: int
    reads: tuple[int, ...]
    freed: tuple[int, ...]


@dataclass(frozen=True)
class Wire:
    """One channel of a partition: its two ends, its output adapter's width conversion
    (lanes in and out) and its input adapter's buffer, where it has them; ``refused`` says
    why an adapter chain cannot be read (the line models one ``vpc`` and one ``input_gen``)."""

    channel: str
    producer: End
    consumer: End
    convert: tuple[int, int] | None = None
    buffer: Buffer | None = None
    refused: str = ""


@dataclass(frozen=True)
class Partition:
    """The partition a ``SizeFifos`` search sized: its kernels, its channels (``wires``) in
    the partition's order and the bottleneck period it sized them at. ``refused`` says why
    it could not be read."""

    kernels: tuple[str, ...] = ()
    wires: tuple[Wire, ...] = ()
    period: int = 0
    refused: str = ""


@dataclass(frozen=True)
class FifoCall:
    """One distinct call of ``size`` or ``least_depth``: see the module docstring.
    ``supply`` and ``accept`` are None where the channel's sides were not read."""

    station: str
    channel: str
    function: str
    period: int
    frames: int
    supply: Beats | None
    accept: Acceptance | None
    line: Line
    partition: Partition | None = None

    @property
    def key(self) -> str:
        """Where the call was met and its channel."""
        return f"{self.station} :: {self.channel}" if self.station else self.channel


class _Open:
    """A call being made: what is known of it so far."""

    def __init__(self, function: str, period: int, station: str, channel: str) -> None:
        self.function, self.period, self.station, self.channel = function, period, station, channel
        self.frames = 0
        self.supply: Beats | None = None
        self.accept: Acceptance | None = None
        self.line: Line | None = None
        self.partition: Partition | None = None
        self.stubbed = ""


def beats_of(pattern: Any) -> Beats:
    """The plain beats of a line ``Pattern``."""
    return Beats(tuple(pattern.times), pattern.beats)


def acceptance_of(accept: Any) -> Acceptance:
    """The plain acceptance of the line's ``Pattern`` or ``Buffered``."""
    if hasattr(accept, "buffer"):
        buffer = accept.buffer
        return Nest(
            buffer.capacity, tuple(buffer.reads), tuple(buffer.freed), beats_of(accept.reads)
        )
    return beats_of(accept)


def _message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _end(side_name: str, declared: Any, port: Any, channel: Any, side: Any) -> End:
    """One end of a channel: a kernel's port, a boundary's free side, a memory source, or an
    end the line does not pace."""
    from finn.kernels.composer import Port as Declared  # noqa: PLC0415
    from finn.kernels.fifo_sizing import NotModelled, _free  # noqa: PLC0415
    from finn.kernels.sides import SideKind  # noqa: PLC0415

    if channel.valued and side_name == "producer":
        return Stored()
    if isinstance(declared, Declared):
        pace = None if port is None else port.contract.pace
        return Port(declared.member, declared.port, pace)
    if side.kind is SideKind.BOUNDARY:
        try:
            return Edge(beats_of(_free(channel, side)))
        except NotModelled as error:
            return Unpaced(str(error))
    return Unpaced(f"a {side.kind.value} side")


def _wire(name: str, channel: Any, neighbours: Any, declared: tuple[Any, Any]) -> Wire:
    from finn.kernels.adapters import Convert  # noqa: PLC0415
    from finn.kernels.input_generator import Generate, nest_buffer  # noqa: PLC0415

    found = channel.sides
    producer = _end("producer", declared[0], neighbours.producer, channel, found.producer)
    consumer = _end("consumer", declared[1], neighbours.consumer, channel, found.consumer)
    convert: tuple[int, int] | None = None
    buffer: Buffer | None = None
    refused = ""
    if channel.output_adapting:
        stages = channel.output_adapter.realization
        if len(stages) == 1 and isinstance(stages[0].module, Convert):
            convert = (stages[0].module.lanes_in, stages[0].module.lanes_out)
        else:
            kinds = " -> ".join(stage.kind for stage in stages)
            refused = f"an output adapter of {kinds}: not modelled"
    if channel.adapter is not None:
        stages = channel.adapter.realization
        if len(stages) == 1 and isinstance(stages[0].module, Generate):
            module = stages[0].module
            nest = nest_buffer(module.frame, module.dims, module.coefs)
            buffer = Buffer(nest.capacity, tuple(nest.reads), tuple(nest.freed))
        else:
            kinds = " -> ".join(stage.kind for stage in stages)
            refused = refused or f"an input adapter of {kinds}: not modelled"
    return Wire(name, producer, consumer, convert, buffer, refused)


def read_partition(network: Any, period: int) -> Partition:
    """The plain partition of ``network``, a ``finn.kernels.partition.Partition``, sized at
    ``period``: a read the network refuses is ``Partition.refused``, with the error."""
    try:
        wires = tuple(
            _wire(name, network.member(name), network.neighbours(name), network._sides[name])
            for name in network.channels
        )
        return Partition(tuple(network.kernels), wires, period)
    except Exception as error:  # noqa: BLE001 - a partition the harness cannot read is named
        return Partition(refused=f"the partition cannot be read: {_message(error)}")


class FifoRecorder:
    """A context that records FINN's sizing calls (the module docstring)."""

    NAMES = ("size", "least_depth", "sides")
    CHECKED = ("least_depth", "sides", "keeps", "simulate", "within")
    """What a ``size`` rests on in its module: ``sides``, and ``least_depth`` with the
    ``keeps`` it calls and the ``simulate`` and ``within`` that calls. A stand-in under any
    of these names makes the answer a stand-in's."""

    def __init__(self) -> None:
        self.calls: dict[FifoCall, int] = {}
        self._known: dict[str, Callable[..., Any]] = {}
        self._originals: dict[str, Callable[..., Any]] = {}
        self._wrappers: dict[str, Callable[..., Any]] = {}
        self._stack: list[_Open] = []
        self._undo: list[Callable[[], None]] = []
        self._partitions: dict[int, tuple[Any, Partition]] = {}

    def __enter__(self) -> FifoRecorder:
        import finn.kernels.fifo_sizing as sizing  # noqa: PLC0415

        # Its users import the line's functions by name: load them, so each name is rebound.
        importlib.import_module("finn.kernels.explore")
        originals = {name: getattr(sizing, name) for name in self.NAMES}
        wrappers = {
            "size": self._size(originals["size"]),
            "least_depth": self._least_depth(originals["least_depth"]),
            "sides": self._sides(originals["sides"]),
        }
        self._originals, self._wrappers = originals, wrappers
        self._known = {name: getattr(sizing, name) for name in self.CHECKED}
        for module in list(sys.modules.values()):
            for key, value in list(getattr(module, "__dict__", {}).items()):
                for name, original in originals.items():
                    if value is original:
                        self._set(module, key, wrappers[name])
        return self

    def __exit__(self, *exc: object) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    def _set(self, module: Any, name: str, value: Any) -> None:
        old = module.__dict__[name]
        setattr(module, name, value)
        self._undo.append(lambda: setattr(module, name, old))

    def recorded(self) -> list[tuple[FifoCall, int]]:
        """Each distinct call and the number of times it was made, in the order first made."""
        return list(self.calls.items())

    # -- where a call was made ------------------------------------------------------------

    @staticmethod
    def _station(frame: FrameType | None) -> str:
        """The station and the sequence the stack names (a harness source that builds its
        partitions), else the test being run (``PYTEST_CURRENT_TEST``), else empty."""
        found: dict[str, str] = {}
        while frame is not None:
            called = frame.f_code.co_name
            if called == "dumped" and hasattr(frame.f_locals.get("station"), "label"):
                found.setdefault("station", str(frame.f_locals["station"].label))
            if called == "tfc" and isinstance(frame.f_locals.get("name"), str):
                found.setdefault("sequence", frame.f_locals["name"])
            frame = frame.f_back
        if found:
            return " ".join(found[key] for key in ("station", "sequence") if key in found)
        test = os.environ.get("PYTEST_CURRENT_TEST", "")
        return test.rsplit(" ", 1)[0] if test else ""

    def _partition(self, frame: FrameType | None) -> tuple[str, Partition | None]:
        """The channel's name and its partition, where the caller is ``SizeFifos.search``."""
        if frame is None or frame.f_code.co_name != "search":
            return "", None
        names = frame.f_locals
        if not {"network", "transport", "bottleneck"} <= set(names):
            return "", None
        network = names["network"]
        if id(network) not in self._partitions:
            # The network is kept, so that its id names it while the recorder lives.
            read = read_partition(network, names["bottleneck"].cycles)
            self._partitions[id(network)] = (network, read)
        return str(names["transport"].channel), self._partitions[id(network)][1]

    @staticmethod
    def _named(channel: Any) -> str:
        """A channel's name where the caller does not give it: its sides' owners."""
        try:
            return f"{channel.sides.source_owner} -> {channel.sides.sink_owner}"
        except Exception:  # noqa: BLE001
            return ""

    def _stubbed(self) -> str:
        """The names, among what ``size`` rests on in its module (``CHECKED``), that are now
        bound there to something other than the module's own function or the recorder's
        wrapper of it: a test's stand-in. Empty where the line's own are in place."""
        import finn.kernels.fifo_sizing as sizing  # noqa: PLC0415

        return ", ".join(
            name
            for name in self.CHECKED
            if getattr(sizing, name) not in (self._known[name], self._wrappers.get(name))
        )

    def _close(self, opened: _Open) -> None:
        real = opened.line is not None and (
            # A channel the line's own ``sides`` does not read never reaches ``least_depth``.
            opened.line.kind == NOT_MODELLED
            and "sides" not in opened.stubbed
        )
        if opened.stubbed and not real:
            # The answer is a stand-in's, not the line's: kept as such, never compared.
            opened.line = Line(STUBBED, reason=f"{opened.stubbed} stubbed in the line's test")
        assert opened.line is not None
        call = FifoCall(
            opened.station,
            opened.channel,
            opened.function,
            opened.period,
            opened.frames,
            opened.supply,
            opened.accept,
            opened.line,
            opened.partition,
        )
        self.calls[call] = self.calls.get(call, 0) + 1

    # -- the wrappers ---------------------------------------------------------------------

    def _size(self, original: Callable[..., Any]) -> Callable[..., Any]:
        from finn.kernels.fifo_sizing import FRAMES  # noqa: PLC0415

        @functools.wraps(original)
        def size(channel: Any, neighbours: Any, period: int, **kwargs: Any) -> Any:
            caller = sys._getframe(1)
            name, partition = self._partition(caller)
            opened = _Open("size", period, self._station(caller), name or self._named(channel))
            opened.partition, opened.frames = partition, FRAMES
            opened.stubbed = self._stubbed()
            self._stack.append(opened)
            try:
                result = original(channel, neighbours, period, **kwargs)
            except Exception as error:
                opened.line = opened.line or Line(RAISED, reason=_message(error))
                raise
            finally:
                self._stack.pop()
                if opened.line is not None:
                    self._close(opened)
            if opened.line is None:  # the line's least_depth was not the one called
                if not opened.stubbed:  # (a stand-in's answer is marked by ``_close``)
                    if result.least is not None:
                        opened.line = Line(DEPTH, result.least)
                    elif result.why.startswith("no depth"):
                        opened.line = Line(NONE)
                    else:
                        opened.line = Line(NOT_MODELLED, reason=result.why)
                self._close(opened)
            return result

        return size

    def _least_depth(self, original: Callable[..., Any]) -> Callable[..., Any]:
        from finn.kernels.fifo_sizing import FRAMES  # noqa: PLC0415

        @functools.wraps(original)
        def least_depth(supply: Any, accept: Any, period: int, *args: Any, **kwargs: Any) -> Any:
            frames = args[0] if args else kwargs.get("frames", FRAMES)
            top = self._stack[-1] if self._stack else None
            inside = top is not None and top.function == "size" and top.line is None
            opened = top if inside and top is not None else None
            if opened is None:
                station = self._station(sys._getframe(1))
                opened = _Open("least_depth", period, station, "")
                opened.stubbed = self._stubbed()
            opened.frames = frames
            opened.supply, opened.accept = beats_of(supply), acceptance_of(accept)
            try:
                result = original(supply, accept, period, *args, **kwargs)
            except Exception as error:
                opened.line = Line(RAISED, reason=_message(error))
                if not inside:
                    self._close(opened)
                raise
            opened.line = Line(NONE) if result is None else Line(DEPTH, result)
            if not inside:
                self._close(opened)
            return result

        return least_depth

    def _sides(self, original: Callable[..., Any]) -> Callable[..., Any]:
        from finn.kernels.fifo_sizing import NotModelled  # noqa: PLC0415

        @functools.wraps(original)
        def sides(channel: Any, neighbours: Any) -> Any:
            top = self._stack[-1] if self._stack else None
            try:
                found = original(channel, neighbours)
            except NotModelled as error:
                if top is not None and top.line is None:
                    top.line = Line(NOT_MODELLED, reason=str(error))
                raise
            if top is not None and top.line is None:
                top.supply, top.accept = beats_of(found[0]), acceptance_of(found[1])
            return found

        return sides
