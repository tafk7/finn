# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The XSim testbench: build a module's sources, run a testbench in XSim, stream words
through a module, and measure its cycles. The XSim executor
(``finn.core.rtlsim_exec``) runs a partition's hardware with it.

``materialize`` builds a ``module`` into a directory and places each
memory's INIT_FILE where ``$readmemh`` reads it; an HLS leaf's request is
synthesized first, for ``HLS_PART``, or taken from the HLS cache
(``finn.transformation.kernels.hls.built_hls``). ``simulate`` elaborates a
testbench module ``check`` against sources, with the tools of the toolchain it is
given, and requires it to display ``PASS``;
otherwise it raises ``SimulationFailed``, a verdict. A compiler or elaborator that
fails before the testbench runs is ``finn.toolchain.launcher.ToolFailed``, and a
toolchain that selects no Vivado ``ToolUnavailable``.

``stream_through`` streams words through a module and checks each AXIS output's
words, compared on their payload bits (an AXIS word is padded to bytes).
``stream_bench`` writes what it simulates, and nothing simulates there. The
testbench drives every input pin of the module from what the module declares,
never by hand per test:

- **clocks and resets, by role**: the free clock (``PERIOD_NS``), and a clock
  derived from it at twice its rate (``ap_clk2x``, a pumped instance's) aligned
  with it: half its period, every rising edge of the clock a rising edge of the
  doubled clock. Each reset is asserted at its level for ``RESET_CYCLES``. The
  stimulus changes just after a rising edge of the clock, where the doubled
  clock has no edge;
- **each AXI-Lite control bus** the root presents is written, after reset and
  before any stream starts, with the writes its kernel's configuration declares
  (``BusExport.registers``, read by ``module.declared_registers``), one at a time:
  address and data, then the response, which must be OKAY. ``registers``
  replaces a bus's declared writes, for a test that writes something its
  kernel does not declare;
- **held inputs** of a leaf module (``Leaf.held``) at their declared values;
- **each AXIS stream** given words: an input's words presented in order, an
  output's ready, each paced by its ``Pace`` in ``pacing``
  (``finn.core.rtlsim_pacing``: by default ``STALLED``, ``FREE`` never
  stalls).

Any other input is refused before anything is written (``Undriven``, naming
every one), as is a port name the module does not present on that side (a
``ValueError`` naming the ports it has): a stale name would otherwise leave the
real port idle until the watchdog. A ``repeating`` design (fed by a cyclic
source) never stops producing: each output then takes exactly its words and
holds its ready low after them; otherwise an output word beyond its words fails.

An output word that differs from its expected word does not stop the run: the
first is displayed, and once every output has presented its words the run fails
(``WORDS_DIFFER``) with each output's words as they arrived written beside the
testbench (``<port>.received.mem``), which ``stream_through`` raises as
``WordsDiffer.received``. A caller that knows the model decodes them as an order
(``finn.harness.orders``): a declared order the RTL does not walk is reported as
which beat carries which index tuple, not as a word that differs.

``stream_out`` streams words through a module and returns each output's words as
they arrived, none expected (``Receive``: how many, of how many bits): what an
executor that only executes takes. The testbench then writes every such output's
words beside it (``<port>.received.mem``) once all have arrived, and compares
nothing. It also taps, when asked (``Tap``), a link between the module's instances
at its producer's output pin: each handshake's word there, as many as the tap takes,
written beside the testbench (``<tap>.tapped.mem``) at the end. A tap reads the
netlist's nets by hierarchical name (``instance_net``) from the testbench: the module
is the same, and a testbench without taps is the same text.

The stimulus and the expected words, and each bus's writes, are ``$readmemh``
files beside the testbench (``<port>.input.mem``, ``<port>.output.mem``,
``<bus>.writes.mem``), so a large tensor does not bloat its text.

**The watchdog** is derived from the run's beats: each stream's beats under its
pace (``Pace.cycles``), summed over the streams, plus the ``cycles`` the model
states beyond them, times ``WATCHDOG_MARGIN``; plus ``RESET_CYCLES``,
``WRITE_CYCLES`` a write and a ``LATENCY_ALLOWANCE`` for the latency no beat
count states. When it fires, it names each stream that stopped short, at which
beat and the cycle of its last handshake, and each bus at which write.

``measure`` streams identical frames back to back, never stalled, and records
the cycle each handshake completes in, on the root's streams and on each link
between the module's instances, as ``Measured``. Every frame presents the same
beats, so a stream's handshakes split evenly into frames. From them, with every
count inclusive of both ends (one beat in and out in the same cycle spans one
cycle):

- a frame's **latency**: from its first input beat accepted to its last output
  beat, over the root's streams, or between any two observed streams (``span``);
- the **interval** between frames: their last beats apart on a stream, the root's
  over its outputs; the last interval is the steady state's (II);
- the **total**: from the first input beat to the last output beat of all frames;
- a stream's **busy** span in a frame: its first to its last beat;
- a stream's **stalls** and **starves** in a frame: the cycles its producer offered a
  word and was refused (valid without ready), and the cycles its consumer waited for
  one (ready without valid), each counted against the handshake that ends it;
- a FIFO's **occupancy**: the words it holds, its stream in minus its stream out, and
  its peak in each frame.

A FIFO that in one frame both held its producer back and left its consumer waiting
lost throughput a deeper FIFO would have kept: ``Measured.findings`` names it
(``fifo-held-producer``). A FIFO that only fills, as one behind a slower consumer
does by design, and a ``direct`` link that stalls are in the table, not findings.

``Measured`` is pure arithmetic on the recorded cycles: the fast gate tests it
without Vivado.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finn.core.reporting.findings import Finding, FindingKind, ordered_findings
from finn.core.rtlsim_pacing import FREE, STALLED, Pacing
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Direction,
    Endpoint,
    Free,
    PinInfo,
    Reset,
    StandardProtocol,
    abi_pins,
)
from finn.kernels.artifacts.build import emit_module, instance_net
from finn.kernels.artifacts.module import (
    Composed,
    Leaf,
    Link,
    LinkEnd,
    Module,
    RegisterMap,
    declared_registers,
)
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.kernels.fifo import FifoKernel
from finn.resources import finnlib_root
from finn.toolchain import xsim
from finn.toolchain.launcher import Toolchain
from finn.transformation.kernels.hls import built_hls

_LOG = logging.getLogger(__name__)

Words = tuple[Sequence[int], int]
"""A port's words, and the payload bits of each."""


@dataclass(frozen=True)
class Receive:
    """An output's words taken as they arrive, none expected: how many, and the payload
    bits of each (``stream_out``)."""

    count: int
    bits: int


@dataclass(frozen=True)
class Tap:
    """A link inside a composed module, observed at one instance's end (``end``, its
    producer's output pins): the low ``bits`` of its data word at each of the first
    ``count`` handshakes there (``stream_out``)."""

    end: LinkEnd
    count: int
    bits: int

    def __post_init__(self) -> None:
        if self.end.instance is None:
            raise ValueError(f"{self.end.data}: a root's port is streamed, not tapped")
        if not 0 < self.bits <= self.end.data_bits:
            raise ValueError(
                f"{self.end.instance}.{self.end.data}: {self.bits} bits of a "
                f"{self.end.data_bits}-bit word"
            )

    def net(self, pin: str) -> str:
        """The testbench's hierarchical name of the net of the end's ``pin``."""
        assert self.end.instance is not None
        return f"dut.{instance_net(self.end.instance, pin)}"


def pack_lanes(values: Sequence[int], bits: int) -> int:
    """One beat's lanes as a word: lanes low first, each ``bits`` wide (two's
    complement, as Python integers: a numpy lane does not overflow the shift)."""
    mask = (1 << bits) - 1
    return sum((int(value) & mask) << (index * bits) for index, value in enumerate(values))


#: The part an HLS leaf is synthesized for to be simulated: the one FinnLib's own HLS
#: recipe synthesizes for (``hls/hbc.tcl``), an UltraScale+ device, the fabric the
#: bare-kernel tests' platforms state. The testbench itself reads no part.
HLS_PART = "xczu3eg-sbva484-1-i"


def materialize(
    module: Module,
    directory: Path,
    *,
    part: str = HLS_PART,
    toolchain: Toolchain,
    cache: Path | None = None,
) -> tuple[str, list[str], dict[str, str]]:
    """The top module, its HDL sources, and each INIT_FILE's name and contents.

    FinnLib is the ``finnlib`` resource (``FINN_RESOURCES_FINNLIB`` overrides it). An HLS
    leaf's request is synthesized for ``part`` first by ``toolchain``, unless the HLS
    cache holds it (``built_hls``: ``cache``, ``$FINN_HOME/hls``
    by default).
    """
    root = finnlib_root()
    built = built_hls(module, part, toolchain=toolchain, cache=cache, roots={"finnlib": root})
    emitted = emit_module(module, directory / "module", roots={"finnlib": root}, built=built)
    sources = [str(emitted.directory / path) for path in emitted.sources]
    data = {path: (emitted.directory / path).read_text() for path in emitted.data}
    return emitted.entry_point, sources, data


class SimulationFailed(AssertionError):
    """The testbench ran and did not display PASS, or ended the simulation with an error
    (``$fatal``); the message is what the simulator printed. An ``AssertionError``, a
    verdict: a check that the hardware does what its testbench expects failed. A tool
    that fails before the testbench runs is ``ToolFailed``."""


class WordsDiffer(SimulationFailed):
    """Every output presented all its words, and some differ from the expected ones:
    ``received`` holds each output's words as they arrived (payload bits), for a caller
    that knows the model to decode (``finn.harness.orders``)."""

    def __init__(self, printed: str, received: Mapping[str, Sequence[int]]) -> None:
        super().__init__(printed)
        self.received = {port: tuple(words) for port, words in received.items()}


#: How long each of the simulator's tools may run, in seconds.
TOOL_TIMEOUT = 300


def simulate(
    sources: Sequence[str | Path],
    testbench: str,
    directory: Path,
    *,
    toolchain: Toolchain,
) -> str:
    """Elaborate the testbench module ``check`` over ``sources``; it must display PASS.

    XSim's driver (``finn.toolchain.xsim``) runs each tool through ``toolchain``, whose
    Vivado provides ``glbl.v`` (``ToolUnavailable`` without one), within
    ``TOOL_TIMEOUT``. The simulator's output, which the testbench may have
    displayed more in. ``xvlog`` or ``xelab`` failing is ``ToolFailed``, naming its log;
    the testbench failing is ``SimulationFailed``."""
    bench = directory / "check.sv"
    bench.write_text("`timescale 1ns/1ps\n" + testbench)
    xsim.elaborate(
        toolchain,
        directory,
        [source for source in sources if not is_header(source)],
        "check",
        includes=include_directories(sources),
        snapshot="check",
        testbench=bench,
        timeout=TOOL_TIMEOUT,
    )
    result = xsim.simulate(toolchain, directory, "check", timeout=TOOL_TIMEOUT)
    output = result.stdout.decode(errors="replace")
    printed = output + result.stderr.decode(errors="replace")
    if result.returncode != 0 or "PASS" not in output:
        raise SimulationFailed(printed)
    return output


def stream_through(
    module: Module,
    directory: Path,
    *,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words],
    pacing: Pacing = STALLED,
    repeating: bool = False,
    cycles: int = 0,
    registers: Mapping[str, RegisterMap] | None = None,
    toolchain: Toolchain,
) -> None:
    """Stream ``inputs`` through ``module``, paced by ``pacing``, and check that each output
    presents its words; see the module docstring for every pin it drives. Raises
    ``WordsDiffer`` when every output presented its words and some differ,
    ``SimulationFailed`` on any other failure. ``toolchain`` runs the simulator and HLS
    (``materialize``).

    ``cycles`` is what the model states the run takes beyond its boundary's beats (a root
    kernel's ``cycles``, when its members' work exceeds its streams'), for the watchdog.
    ``registers`` writes a control bus something its kernel does not declare, in place of
    its declared writes."""
    _stream(
        module,
        directory,
        inputs,
        outputs,
        pacing=pacing,
        repeating=repeating,
        cycles=cycles,
        registers=registers,
        toolchain=toolchain,
    )


def stream_out(
    module: Module,
    directory: Path,
    *,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Receive],
    pacing: Pacing = STALLED,
    cycles: int = 0,
    toolchain: Toolchain,
    cache: Path | None = None,
    taps: Mapping[str, Tap] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Stream ``inputs`` through ``module``, paced by ``pacing``, and return each output's
    words as they arrived (payload bits), as many as its ``Receive`` states, by port, and
    each tap's words by its name (``taps``: the words its link carried, up to its count,
    fewer if the link carried fewer); see the module docstring for every pin it drives.
    Nothing is compared: an output word beyond its count, the watchdog or a tool's error
    raises ``SimulationFailed``.

    ``cycles`` as ``stream_through`` takes them. ``toolchain`` runs the simulator and HLS;
    ``cache`` is the HLS cache (``materialize``)."""
    taps = taps or {}
    bench = stream_bench(
        module,
        directory,
        inputs=inputs,
        outputs=outputs,
        pacing=pacing,
        cycles=cycles,
        toolchain=toolchain,
        cache=cache,
        taps=taps,
    )
    for port in outputs:
        (directory / f"{port}.{RECEIVED}").unlink(missing_ok=True)
    for name in taps:
        (directory / f"{name}.{TAPPED}").unlink(missing_ok=True)
    simulate(bench.sources, bench.text, directory, toolchain=toolchain)
    found = {port: tuple(_read_memory(directory / f"{port}.{RECEIVED}")) for port in outputs}
    for name in taps:
        found[name] = tuple(_read_taken(directory / f"{name}.{TAPPED}"))
    return found


def measure(
    module: Module,
    directory: Path,
    *,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words],
    frames: int = 4,
    cycles: int = 0,
    toolchain: Toolchain,
) -> Measured:
    """The cycles of ``frames`` frames streamed back to back, never stalled; its table
    logged too (INFO).

    Each input presents its words ``frames`` times over and each output must
    present its words as many times, checked as ``stream_through`` checks them.
    Every handshake is recorded by the cycle it completes in, with the cycles its
    producer was refused and its consumer waited before it: on the root's
    streams and on each link between the module's instances, observed at the
    sink's nets (``instance_net``), so a kernel's own ports are measured where the
    module composes several. Each FIFO instance (``FifoKernel``'s) whose streams in
    and out are both recorded is kept with them (``FifoStreams``). ``cycles``, a
    frame's, and ``toolchain``, as ``stream_through`` takes them.
    """
    if frames < 1:
        raise ValueError(f"at least one frame, not {frames}")
    _check_streams(module, inputs, outputs)
    observed = {}
    by_data: dict[str, str] = {}  # the root's streams by their data pin
    for name, bus in _streams(module).items():
        if name in inputs or name in outputs:
            member = _members(bus)
            observed[name] = (member["tvalid"], member["tready"])
            by_data[member["tdata"]] = name
    for link in module.fragment.links if isinstance(module, Composed) else ():
        stream = link_stream(link)
        if stream is not None:
            sink = link.sink.instance
            assert sink is not None
            observed[stream] = (
                f"dut.{instance_net(sink, link.sink.valid)}",
                f"dut.{instance_net(sink, link.sink.ready)}",
            )
    fifos = _fifos(module, by_data) if isinstance(module, Composed) else {}
    log = _stream(
        module,
        directory,
        {port: (list(words) * frames, bits) for port, (words, bits) in inputs.items()},
        {port: (list(words) * frames, bits) for port, (words, bits) in outputs.items()},
        pacing=FREE,
        cycles=cycles * frames,
        observed=observed,
        toolchain=toolchain,
    )
    beats: dict[str, list[int]] = {name: [] for name in observed}
    stalls: dict[str, list[int]] = {name: [] for name in observed}
    starves: dict[str, list[int]] = {name: [] for name in observed}
    for line in log.splitlines():
        if line.startswith("BEAT "):
            _, name, cycle, stalled, starved = line.split()
            beats[name].append(int(cycle))
            stalls[name].append(int(stalled))
            starves[name].append(int(starved))
    measured = Measured(
        frames=frames,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        beats={name: tuple(cycles) for name, cycles in beats.items()},
        stalls={name: tuple(counts) for name, counts in stalls.items()},
        starves={name: tuple(counts) for name, counts in starves.items()},
        fifos=fifos,
    )
    _LOG.info(measured.table())
    return measured


def _fifos(module: Composed, by_data: Mapping[str, str]) -> dict[str, FifoStreams]:
    """Each FIFO instance of ``module`` by its label, with its streams as ``measure`` records
    them: a link to the root by the root's output (``by_data``, the given streams by their
    data pins); a FIFO into an output not given is left out, its words out not recorded."""
    links = module.fragment.links
    found = {}
    for label, leaf in module.fragment.instances:
        if leaf.implementation_id != FifoKernel.id:
            continue
        (into,) = (link for link in links if link.sink.instance == label)
        (out,) = (link for link in links if link.source.instance == label)
        into_stream = link_stream(into)
        out_stream = link_stream(out) or by_data.get(out.sink.data)
        assert into_stream is not None
        if out_stream is None:
            continue
        depth = dict(leaf.parameters).get("DEPTH")
        found[label] = FifoStreams(
            into=into_stream,
            out=out_stream,
            producer=into.source.instance or by_data.get(into.source.data, into.source.data),
            depth=depth if isinstance(depth, int) else None,
        )
    return found


def link_stream(link: Link) -> str | None:
    """The name ``measure`` records a link's handshakes under: its sink's data pin,
    ``instance.pin``; None for a link into the root, recorded under the root's output
    port. A link from the root is recorded under both: its sink's pin and the root's input
    port, one handshake."""
    sink = link.sink
    return None if sink.instance is None else f"{sink.instance}.{sink.data}"


# -- the stream testbench ------------------------------------------------------------------

#: The watchdog's margin over the cycles a run's beats take, and its allowance for the
#: latency no beat count states (a module's pipeline, its adapters' fill).
WATCHDOG_MARGIN = 4
LATENCY_ALLOWANCE = 1000
#: Cycles held in reset, past the DSP models' startup recovery (GSR).
RESET_CYCLES = 16
#: An upper bound on one AXI-Lite write: address and data accepted, then the response.
WRITE_CYCLES = 8

#: The testbench's clock period, in ns (its ``timescale`` is 1ns/1ps).
PERIOD_NS = 10

#: What the testbench reports when outputs presented all their words and some differ, and
#: the file beside it each output's received words are written to then (``<port>.``).
WORDS_DIFFER = "output words differ"
RECEIVED = "received.mem"
#: The file beside the testbench each tap's words are written to (``<tap>.``).
TAPPED = "tapped.mem"


class Undriven(ValueError):
    """The module has inputs the testbench has no declared value for; it names them."""


@dataclass(frozen=True)
class _Clocks:
    """The root's clock, its aligned doubled clock (or None), and its resets with the level
    each is asserted at."""

    clock: str
    doubled: str | None
    resets: tuple[tuple[str, int], ...]


def _clocks(pins: Mapping[str, PinInfo]) -> _Clocks:
    free = [name for name, info in pins.items() if _loose(info) and _is_clock(info, Free)]
    if len(free) != 1:
        raise Undriven(f"the testbench drives one free clock; the module has {free or 'none'}")
    (clock,) = free
    doubled: list[str] = []
    for name, info in pins.items():
        if _loose(info) and _is_clock(info, Derived):
            assert isinstance(info.role, Clock) and isinstance(info.role.rate, Derived)
            rate = info.role.rate
            if rate.of != clock or rate.ratio != 2:
                raise Undriven(
                    f"{name} runs at {rate.ratio}x {rate.of}; the testbench drives a 2x {clock}"
                )
            doubled.append(name)
    if len(doubled) > 1:
        raise Undriven(f"the testbench drives one doubled clock; the module has {doubled}")
    resets = tuple(
        (name, int(info.role.active_low is False))
        for name, info in pins.items()
        if _loose(info) and isinstance(info.role, Reset)
    )
    return _Clocks(clock, doubled[0] if doubled else None, resets)


def _loose(info: PinInfo) -> bool:
    return info.bus is None and info.direction is Direction.IN


def _is_clock(info: PinInfo, rate: type) -> bool:
    return isinstance(info.role, Clock) and isinstance(info.role.rate, rate)


def _streams(module: Module) -> dict[str, Bus]:
    return {
        pin.name: pin
        for pin in module.abi.pins
        if isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXIS
    }


def _check_streams(
    module: Module, inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> None:
    buses = _streams(module).values()
    for side, names, endpoint in (
        ("input", inputs, Endpoint.TARGET),
        ("output", outputs, Endpoint.INITIATOR),
    ):
        has = [bus.name for bus in buses if bus.endpoint is endpoint]
        unknown = sorted(set(names) - set(has))
        if unknown:
            raise ValueError(
                f"the module has no {side} stream {', '.join(unknown)}; its {side}s: "
                + (", ".join(has) or "none")
            )


def _members(bus: Bus) -> dict[str, str]:
    """A bus's physical pins by logical member."""
    return {member.logical: member.physical for member in bus.signals}


def _configured(
    module: Module, held: Mapping[str, int], registers: Mapping[str, RegisterMap] | None
) -> dict[str, Bus]:
    """The root's AXI-Lite targets, each driven by the testbench's writes, but for a bus
    the module holds."""
    found = {
        pin.name: pin
        for pin in module.abi.pins
        if isinstance(pin, Bus)
        and pin.protocol is StandardProtocol.AXILITE
        and pin.endpoint is Endpoint.TARGET
        and not any(member.physical in held for member in pin.signals)
    }
    unknown = sorted(set(registers or {}) - set(found))
    if unknown:
        raise ValueError(
            f"the module has no control bus {', '.join(unknown)}; its control buses: "
            + (", ".join(found) or "none")
        )
    return found


def _undriven(
    pins: Mapping[str, PinInfo],
    streams: Mapping[str, Bus],
    driven: set[str],
    held: Mapping[str, int],
    configured: Mapping[str, Bus],
) -> None:
    """Refuse an input the testbench has no value for, naming every one."""
    covered = set(held)
    for bus in (*(streams[name] for name in driven), *configured.values()):
        covered |= {member.physical for member in bus.signals}
    missing = []
    for name, info in pins.items():
        if info.direction is not Direction.IN or name in covered:
            continue
        if _loose(info) and isinstance(info.role, (Clock, Reset)):
            continue
        if info.bus in streams:
            missing.append(f"{name} (stream {info.bus}, given no words)")
        else:
            missing.append(name)
    if missing:
        raise Undriven(
            "the testbench has no value for the inputs "
            + ", ".join(missing)
            + ": the module declares none (neither held, nor a clock or reset by role, "
            "nor a control bus), and no stream drives them"
        )
    for name in driven:
        last = _members(streams[name]).get("tlast")
        if last is not None and pins[last].direction is Direction.IN:
            raise Undriven(f"{last}: the testbench streams no tlast")


def _memory(directory: Path, name: str, bits: int, words: Sequence[int], taken: set[str]) -> str:
    """Write ``words`` as a ``$readmemh`` file beside the testbench; its file name."""
    if name in taken:
        raise ValueError(f"{name} is already a file of the simulation")
    taken.add(name)
    digits = (bits + 3) // 4
    (directory / name).write_text("".join(f"{word:0{digits}x}\n" for word in words))
    return name


@dataclass(frozen=True)
class StreamBench:
    """A stream testbench: the module's sources, and the testbench module ``check``."""

    sources: tuple[str, ...]
    text: str
    budget: int


def stream_bench(
    module: Module,
    directory: Path,
    *,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words | Receive],
    pacing: Pacing = STALLED,
    repeating: bool = False,
    cycles: int = 0,
    registers: Mapping[str, RegisterMap] | None = None,
    observed: Mapping[str, tuple[str, str]] | None = None,
    toolchain: Toolchain,
    cache: Path | None = None,
    taps: Mapping[str, Tap] | None = None,
) -> StreamBench:
    """Write the module's sources, its data files and the stimulus and expected words into
    ``directory``, and the testbench that drives every pin of the module from what it
    declares (the module docstring); nothing simulates. An output given as ``Receive``
    is compared with nothing: its words are written beside the testbench once every
    output has presented its words, and so are each tap's (``Tap``, by its name, an
    identifier). ``toolchain`` and ``cache`` as ``materialize`` takes them.

    Refuses, before writing anything, a stream the module does not present
    (``ValueError``), an input the testbench has no value for (``Undriven``) and a tap
    of an instance the module does not place, or named as one of its streams
    (``ValueError``).
    """
    _check_streams(module, inputs, outputs)
    _check_taps(module, taps or {})
    pins = abi_pins(module.abi.pins)
    clocks = _clocks(pins)
    streams = _streams(module)
    held = dict(module.held.inputs) if isinstance(module, Leaf) else {}
    holding = sorted(
        name
        for name in (*inputs, *outputs)
        if any(member.physical in held for member in streams[name].signals)
    )
    if holding:
        raise ValueError(f"the module holds the streams {', '.join(holding)}: none is driven")
    configured = _configured(module, held, registers)
    _undriven(pins, streams, {*inputs, *outputs}, held, configured)
    writes = {**{port: RegisterMap() for port in configured}, **declared_registers(module)}
    writes.update(registers or {})
    for bus_name, bus in configured.items():
        widths = {member.logical: member.width for member in bus.signals}
        found = writes[bus_name]
        if widths["wdata"] != found.word_bits:
            raise ValueError(f"{bus_name} takes {widths['wdata']}-bit words, not {found.word_bits}")
        if any(at >> widths["awaddr"] for at, _ in found.writes):
            raise ValueError(f"{bus_name}: a write's address exceeds its {widths['awaddr']} bits")

    top, sources, data = materialize(module, directory, toolchain=toolchain, cache=cache)
    taken = set(data)
    for name, text in data.items():  # $readmemh reads an INIT_FILE from the simulator's directory
        (directory / name).write_text(text)

    clock = clocks.clock
    lines: list[str] = []  # declarations
    count: list[str] = []  # at each rising edge, while running
    drive: list[str] = []  # just after each rising edge
    done: list[str] = []
    stopped: list[str] = []  # the watchdog's report
    received: list[str] = []  # each output's words, written out when any differs
    returned: list[str] = []  # each Receive output's words, written out at the end
    covered = {clock, *(name for name, _ in clocks.resets)}
    if clocks.doubled is not None:
        covered.add(clocks.doubled)
    for name, value in held.items():
        width = pins[name].width
        lines.append(f"logic [{width - 1}:0] {name} = {width}'h{value:x};")
        covered.add(name)

    for bus_name, bus in configured.items():
        member = _members(bus)
        found = writes[bus_name]
        word = found.word_bits
        address = next(m.width for m in bus.signals if m.logical == "awaddr")
        total = len(found.writes)
        for logical, physical in member.items():
            width = pins[physical].width
            vector = "" if width == 1 else f"[{width - 1}:0] "
            if pins[physical].direction is Direction.IN:
                lines.append(f"logic {vector}{physical} = 0;")
            else:
                lines.append(f"wire {vector}{physical};")
            covered.add(physical)
        lines.append(f"int {bus_name}_written = 0; logic {bus_name}_aw = 0, {bus_name}_w = 0;")
        done.append(f"{bus_name}_written == {total}")
        stopped.append(
            f'if ({bus_name}_written < {total}) $display("watchdog: {bus_name} stopped at '
            f'write %0d of {total}", {bus_name}_written);'
        )
        if not total:
            continue
        table = _memory(
            directory,
            f"{bus_name}.writes.mem",
            address + word,
            [(a << word) | w for a, w in found.writes],
            taken,
        )
        lines += [
            f"logic [{address + word - 1}:0] {bus_name}_writes [{total}];",
            f'initial $readmemh("{table}", {bus_name}_writes);',
        ]
        awvalid, awready = member["awvalid"], member["awready"]
        wvalid, wready = member["wvalid"], member["wready"]
        bvalid, bready = member["bvalid"], member["bready"]
        response = member.get("bresp")
        complain = (
            f'if ({response} != 0) $fatal(1, "{bus_name} write %0d: response %0d", '
            f"{bus_name}_written, {response});"
            if response
            else ""
        )
        count.append(
            f"""if ({awvalid} && {awready}) {bus_name}_aw <= 1;
            if ({wvalid} && {wready}) {bus_name}_w <= 1;
            if ({bvalid} && {bready}) begin
                {complain}
                {bus_name}_written <= {bus_name}_written + 1;
                {bus_name}_aw <= 0;
                {bus_name}_w <= 0;
            end"""
        )
        current = f"{bus_name}_writes[{bus_name}_written < {total} ? {bus_name}_written : 0]"
        pending = f"running && {bus_name}_written < {total}"
        drive += [
            f"{awvalid} = {pending} && !{bus_name}_aw;",
            f"{member['awaddr']} = {current}[{address + word - 1}:{word}];",
            f"{wvalid} = {pending} && !{bus_name}_w;",
            f"{member['wdata']} = {current}[{word - 1}:0];",
            f"{member['wstrb']} = '1;",
            f"{bready} = 1;",
        ]
    configured_now = " && ".join(
        f"{name}_written == {len(writes[name].writes)}" for name in configured
    )

    paced = 0
    for side, given in (("input", inputs), ("output", outputs)):
        for index, (port, spec) in enumerate(given.items()):
            words: Sequence[int] | None = None  # none expected: a Receive output's
            if isinstance(spec, Receive):
                total, bits = spec.count, spec.bits
            else:
                words, bits = spec
                total = len(words)
            pace = pacing.input(index) if side == "input" else pacing.output(index)
            paced += pace.cycles(total)
            member = _members(streams[port])
            data_pin, valid, ready = member["tdata"], member["tvalid"], member["tready"]
            carrier = pins[data_pin].width
            covered |= {data_pin, valid, ready}
            table_bits = carrier if side == "input" else bits
            if words is not None:
                table = _memory(directory, f"{port}.{side}.mem", table_bits, words, taken)
                lines += [
                    f"logic [{table_bits - 1}:0] {port}_words [{total}];",
                    f'initial $readmemh("{table}", {port}_words);',
                ]
            lines.append(
                f"int {port}_beats = 0, {port}_burst = 0, {port}_idle = 0, {port}_last = -1;"
            )
            handshake = f"{valid} && {ready}"
            paced_count = f"""{port}_beats <= {port}_beats + 1;
                {port}_last <= cycle;
                if ({port}_burst + 1 == {pace.burst}) begin
                    {port}_burst <= 0;
                    {port}_idle <= {pace.pause};
                end else {port}_burst <= {port}_burst + 1;"""
            if side == "input":
                lines.append(
                    f"logic [{carrier - 1}:0] {data_pin}; logic {valid} = 0; wire {ready};"
                )
                count.append(
                    f"""if ({handshake}) begin
                {paced_count}
            end else if ({port}_idle) {port}_idle <= {port}_idle - 1;"""
                )
                drive += [
                    f"{valid} = streaming && {port}_beats < {total} && !{port}_idle;",
                    f"{data_pin} = {port}_words[{port}_beats < {total} ? {port}_beats : 0];",
                ]
            else:
                lines += [
                    f"wire [{carrier - 1}:0] {data_pin}; wire {valid}; logic {ready} = 0;",
                    f"logic [{bits - 1}:0] {port}_got [{total}];",
                ]
                compare = ""  # a Receive output's words: written out at the end, compared never
                if words is None:
                    returned.append(f'$writememh("{port}.{RECEIVED}", {port}_got);')
                else:
                    received.append(f'$writememh("{port}.{RECEIVED}", {port}_got);')
                    compare = f"""
                if ({data_pin}[{bits - 1}:0] !== {port}_words[{port}_beats]) begin
                    if (!differing) $display("{port} word %0d: %h != %h", {port}_beats,
                        {data_pin}, {port}_words[{port}_beats]);
                    differing <= differing + 1;
                end"""
                count.append(
                    f"""if ({handshake}) begin
                if ({port}_beats >= {total})
                    $fatal(1, "{port} word %0d: beyond its {total} words", {port}_beats);
                {port}_got[{port}_beats] <= {data_pin}[{bits - 1}:0];{compare}
                {paced_count}
            end else if ({port}_idle) {port}_idle <= {port}_idle - 1;"""
                )
                taken_all = f"{port}_beats < {total} && " if repeating else ""
                drive.append(f"{ready} = streaming && {taken_all}!{port}_idle;")
                done.append(f"{port}_beats == {total}")
            stopped.append(
                f'if ({port}_beats < {total}) $display("watchdog: {port} stopped at beat %0d '
                f'of {total}, its last handshake at cycle %0d", {port}_beats, {port}_last);'
            )

    for name, info in pins.items():
        if name in covered:
            continue
        vector = "" if info.width == 1 else f"[{info.width - 1}:0] "
        lines.append(f"wire {vector}{name};")  # an output nothing reads
    # Each observed handshake, with the cycles before it its producer was refused (a stall)
    # and its consumer waited (a starve).
    for index, (name, (valid, ready)) in enumerate((observed or {}).items()):
        stall, starve = f"observed{index}_stall", f"observed{index}_starve"
        lines.append(f"int {stall} = 0, {starve} = 0;")
        count.append(
            f"""if ({valid} && {ready}) begin
                $display("BEAT {name} %0d %0d %0d", cycle, {stall}, {starve});
                {stall} <= 0;
                {starve} <= 0;
            end else if ({valid}) {stall} <= {stall} + 1;
            else if ({ready}) {starve} <= {starve} + 1;"""
        )
    for name, tap in (taps or {}).items():
        end, total = tap.end, tap.count
        lines += [f"logic [{tap.bits - 1}:0] {name}_tapped [{total}];", f"int {name}_taps = 0;"]
        count.append(
            f"""if ({tap.net(end.valid)} && {tap.net(end.ready)}) begin
                if ({name}_taps < {total})
                    {name}_tapped[{name}_taps] <= {tap.net(end.data)}[{tap.bits - 1}:0];
                {name}_taps <= {name}_taps + 1;
            end"""
        )
        returned.append(f'$writememh("{name}.{TAPPED}", {name}_tapped);')

    total_writes = sum(len(found.writes) for found in writes.values())
    budget = (
        RESET_CYCLES
        + WRITE_CYCLES * total_writes
        + WATCHDOG_MARGIN * (paced + cycles)
        + LATENCY_ALLOWANCE
    )
    reasons = (
        f"{RESET_CYCLES} in reset, {WRITE_CYCLES} a write for {total_writes}, "
        f"{WATCHDOG_MARGIN} x ({paced} paced beats + {cycles} model cycles), "
        f"{LATENCY_ALLOWANCE} for latency"
    )
    if clocks.doubled is None:
        clocking = f"""logic {clock} = 0;
    always #{PERIOD_NS / 2:g} {clock} = !{clock};"""
    else:
        clocking = f"""logic {clock} = 0, {clocks.doubled} = 0;
    // {clocks.doubled} at twice {clock}, rising with each rising edge of {clock}
    always #{PERIOD_NS / 4:g} begin
        {clocks.doubled} = !{clocks.doubled};
        if ({clocks.doubled}) {clock} = !{clock};
    end"""
    resets = "\n    ".join(f"logic {name} = {level};" for name, level in clocks.resets)
    release = "\n        ".join(f"{name} = {1 - level};" for name, level in clocks.resets)
    newline = "\n    "
    text = f"""module check;
    {clocking}
    {resets}
    logic running = 0;
    int differing = 0;  // output words that differ from the expected
    {newline.join(lines)}
    wire configured = {configured_now or "1"};
    wire streaming = running && configured;
    {top} dut (.*);
    int cycle = 0;
    always @(posedge {clock}) begin
        cycle <= cycle + 1;
        if (running) begin
            {(newline + "        ").join(count)}
        end
        if (cycle == {budget}) begin  // the watchdog: {reasons}
            {(newline + "        ").join(stopped)}
            $fatal(1, "watchdog: the run did not complete within {budget} cycles");
        end
    end
    // Stimulus changes just after a rising edge of {clock}, on no edge of a doubled clock.
    always @(posedge {clock}) #1 begin
        {(newline + "    ").join(drive)}
    end
    initial begin
        repeat ({RESET_CYCLES}) @(posedge {clock});  // past the DSP models' startup recovery (GSR)
        #1;
        {release}
        running = 1;
        wait ({" && ".join(done) or "1"});
        repeat (4) @(posedge {clock});
        if (differing) begin
            {(newline + "        ").join(received)}
            $fatal(1, "{WORDS_DIFFER}: %0d output words", differing);
        end
        {"".join(line + newline + "    " for line in returned)}$display("STREAM_PASS");
        $finish;
    end
endmodule
"""
    return StreamBench(tuple(sources), text, budget)


def _stream(
    module: Module,
    directory: Path,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words],
    *,
    toolchain: Toolchain,
    **options: Any,
) -> str:
    """Stream ``inputs`` through ``module`` checking ``outputs``; the simulator's output.
    Raises ``WordsDiffer`` with the words each output presented when some differ."""
    bench = stream_bench(
        module, directory, inputs=inputs, outputs=outputs, toolchain=toolchain, **options
    )
    for port in outputs:
        (directory / f"{port}.{RECEIVED}").unlink(missing_ok=True)
    try:
        return simulate(bench.sources, bench.text, directory, toolchain=toolchain)
    except SimulationFailed as failed:
        if WORDS_DIFFER not in str(failed):
            raise
        received = {port: _read_memory(directory / f"{port}.{RECEIVED}") for port in outputs}
        raise WordsDiffer(str(failed), received) from None


def _check_taps(module: Module, taps: Mapping[str, Tap]) -> None:
    placed = dict(module.fragment.instances) if isinstance(module, Composed) else {}
    streams = _streams(module)
    for name, tap in taps.items():
        if not name.isidentifier() or name in streams:
            raise ValueError(f"{name!r}: a tap is named by an identifier no stream has")
        if tap.end.instance not in placed:
            raise ValueError(f"tap {name}: the module places no instance {tap.end.instance}")


def _read_taken(path: Path) -> list[int]:
    """A tap's words as ``$writememh`` wrote them, up to the first it did not take (an
    unknown word, ``x``)."""
    words: list[int] = []
    for line in path.read_text().splitlines():
        for token in line.split("//", 1)[0].split():
            if token.startswith("@"):
                continue
            if not all(digit in "0123456789abcdefABCDEF" for digit in token):
                return words
            words.append(int(token, 16))
    return words


def _read_memory(path: Path) -> list[int]:
    """The words of a ``$writememh`` file, its comments and addresses skipped."""
    words = []
    for line in path.read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        words += [int(token, 16) for token in line.split() if not token.startswith("@")]
    return words


# -- measurement ------------------------------------------------------------------------

Frame = tuple[int, ...]
"""The cycles of one frame's beats on one stream, in order."""


@dataclass(frozen=True)
class FifoStreams:
    """A FIFO instance as ``Measured`` reads it: the stream into it and the stream out of
    it, by their names there; the producer that feeds it (an instance, or the root's
    input); and its depth, where its module states one."""

    into: str
    out: str
    producer: str
    depth: int | None = None


#: The code of the finding that names a FIFO that held its producer back
#: (``Measured.findings``).
FIFO_HELD_PRODUCER = "fifo-held-producer"


@dataclass(frozen=True)
class Measured:
    """``frames`` frames' handshakes, by stream name: the root's ``inputs`` and
    ``outputs`` by port, each link by its sink (``instance.pin``). For a stream in
    ``stalls`` and ``starves``, each handshake's cycles before it its producer was
    refused and its consumer waited, in the order of its beats; and each FIFO instance
    between two of the streams, by its label (``fifos``)."""

    frames: int
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    beats: Mapping[str, tuple[int, ...]]
    stalls: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    starves: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    fifos: Mapping[str, FifoStreams] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (*self.inputs, *self.outputs):
            if name not in self.beats:
                raise ValueError(f"no beats recorded on the root's stream {name}")
        for name in self.beats:
            self.of(name)  # refuses uneven frames now
        for counts in (self.stalls, self.starves):
            for name, waits in counts.items():
                beats = len(self.beats.get(name, ()))
                if len(waits) != beats:
                    raise ValueError(f"{name}: {len(waits)} counts of waits for {beats} beats")
        for label, fifo in self.fifos.items():
            words = [len(self.beats.get(name, ())) for name in (fifo.into, fifo.out)]
            if not words[0] or words[0] != words[1]:
                raise ValueError(f"{label}: {words[0]} words in and {words[1]} out")

    def of(self, name: str) -> tuple[Frame, ...]:
        """``name``'s beats, frame by frame."""
        cycles = self.beats[name]
        if not cycles or len(cycles) % self.frames:
            raise ValueError(f"{name}: {len(cycles)} beats do not split into {self.frames} frames")
        count = len(cycles) // self.frames
        return tuple(tuple(cycles[f * count : (f + 1) * count]) for f in range(self.frames))

    def per_frame(self, name: str) -> int:
        """The beats a frame presents on ``name``."""
        return len(self.beats[name]) // self.frames

    def busy(self, name: str) -> tuple[int, ...]:
        """Each frame's first to last beat on ``name``."""
        return tuple(frame[-1] - frame[0] + 1 for frame in self.of(name))

    def _first(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(min(self.of(name)[f][0] for name in names) for f in range(self.frames))

    def _last(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(max(self.of(name)[f][-1] for name in names) for f in range(self.frames))

    def span(self, source: str, sink: str) -> tuple[int, ...]:
        """Each frame's first beat on ``source`` to its last beat on ``sink``."""
        return tuple(
            last - first + 1
            for first, last in zip(self._first([source]), self._last([sink]), strict=True)
        )

    @property
    def latencies(self) -> tuple[int, ...]:
        """Each frame's first input beat to its last output beat."""
        return tuple(
            last - first + 1
            for first, last in zip(self._first(self.inputs), self._last(self.outputs), strict=True)
        )

    def intervals(self, name: str | None = None) -> tuple[int, ...]:
        """Successive frames' last beats apart on ``name``; on the root's outputs by default."""
        ends = self._last(self.outputs if name is None else [name])
        return tuple(later - earlier for earlier, later in zip(ends, ends[1:]))

    def interval(self, name: str | None = None) -> int:
        """The steady state's interval (the last), on ``name`` or the root's outputs."""
        if self.frames < 2:
            raise ValueError("an interval needs at least two frames")
        return self.intervals(name)[-1]

    @property
    def total(self) -> int:
        """The first input beat to the last output beat of the last frame."""
        return self._last(self.outputs)[-1] - self._first(self.inputs)[0] + 1

    def stalled(self, name: str) -> tuple[int, ...]:
        """Each frame's cycles on ``name`` its producer offered a word and was refused."""
        return self._waits(self.stalls, name)

    def starved(self, name: str) -> tuple[int, ...]:
        """Each frame's cycles on ``name`` its consumer waited for a word."""
        return self._waits(self.starves, name)

    def _waits(self, counts: Mapping[str, tuple[int, ...]], name: str) -> tuple[int, ...]:
        if name not in counts:
            raise ValueError(f"{name}: no waits recorded")
        waits, count = counts[name], self.per_frame(name)
        return tuple(sum(waits[f * count : (f + 1) * count]) for f in range(self.frames))

    def occupancy(self, fifo: str) -> tuple[tuple[int, int], ...]:
        """The FIFO ``fifo``'s words over time: at each cycle a word entered or left it,
        the cycle and the words it holds after that cycle's handshakes."""
        streams = self.fifos[fifo]
        change = Counter(self.beats[streams.into])
        change.subtract(self.beats[streams.out])
        held = 0
        found: list[tuple[int, int]] = []
        for cycle in sorted(change):
            held += change[cycle]
            found.append((cycle, held))
        return tuple(found)

    def peaks(self, fifo: str) -> tuple[int, ...]:
        """Each frame's most words the FIFO ``fifo`` holds, just after one of the frame's
        words entered it (only a word entering raises the count)."""
        streams = self.fifos[fifo]
        out = self.beats[streams.out]
        held = [
            entered + 1 - bisect_right(out, cycle)
            for entered, cycle in enumerate(self.beats[streams.into])
        ]
        count = self.per_frame(streams.into)
        return tuple(max(held[f * count : (f + 1) * count]) for f in range(self.frames))

    @property
    def findings(self) -> tuple[Finding, ...]:
        """Each FIFO that in a frame both held its producer back (stalls on its stream in)
        and left its consumer waiting (starves on its stream out): it lost throughput a
        deeper FIFO would have kept (``fifo-held-producer``, a limitation). The consumer's
        wait for the run's first word is left out, the pipeline's fill, which no depth
        shortens; every frame is judged, the first too, so a single frame's burst is.

        A FIFO that only fills, as one behind a slower consumer does, and a stream with no
        FIFO that stalls are not findings; nor is a FIFO whose streams carry no counts."""
        found: list[Finding] = []
        for label, fifo in self.fifos.items():
            if fifo.into not in self.stalls or fifo.out not in self.starves:
                continue
            stalls = self.stalled(fifo.into)
            starves = list(self.starved(fifo.out))
            starves[0] -= self.starves[fifo.out][0]  # the fill
            frames = [f for f in range(self.frames) if stalls[f] and starves[f]]
            if not frames:
                continue
            peaks = self.peaks(label)
            depth = "" if fifo.depth is None else f" of its depth {fifo.depth}"
            found.append(
                Finding(
                    FindingKind.LIMITATION,
                    FIFO_HELD_PRODUCER,
                    "measure",
                    f"{label}: in frames {frames} it held its producer {fifo.producer} back "
                    f"{sum(stalls[f] for f in frames)} cycles and left its consumer waiting "
                    f"{sum(starves[f] for f in frames)} cycles, holding at most "
                    f"{max(peaks[f] for f in frames)} words{depth}: a deeper FIFO would have "
                    "kept that throughput",
                    (
                        ("instance", label),
                        ("producer", fifo.producer),
                        ("depth", fifo.depth),
                        ("frames", frames),
                        ("stalls", stalls),
                        ("starves", starves),
                        ("peaks", peaks),
                    ),
                )
            )
        return ordered_findings(found)

    def table(self) -> str:
        """The measurement as text: the root's latency, interval and total; per stream its
        beats per frame, busy span, interval, first beat, and the last frame's stalls and
        starves where they were counted; per FIFO its depth and each frame's peak, stalls
        in and starves out; then each finding."""
        start = self._first(self.inputs)[0]
        lines = [
            f"frames {self.frames}; latency per frame {list(self.latencies)}; "
            f"intervals {list(self.intervals()) if self.frames > 1 else []}; "
            f"total {self.total}",
            f"{'stream':<56} {'beats':>6} {'busy':>6} {'II':>6} {'first':>6} "
            f"{'stall':>6} {'starve':>6}",
        ]
        for name in self.beats:
            interval = self.interval(name) if self.frames > 1 else "-"
            stall = self.stalled(name)[-1] if name in self.stalls else "-"
            starve = self.starved(name)[-1] if name in self.starves else "-"
            lines.append(
                f"{name:<56} {self.per_frame(name):>6} {self.busy(name)[-1]:>6} "
                f"{interval:>6} {self.of(name)[0][0] - start:>6} {stall:>6} {starve:>6}"
            )
        if self.fifos:
            lines.append(f"{'fifo':<56} {'depth':>6}  peak, stalls in, starves out per frame")
        for label, fifo in self.fifos.items():
            depth = "-" if fifo.depth is None else fifo.depth
            stalls = list(self.stalled(fifo.into)) if fifo.into in self.stalls else "-"
            starves = list(self.starved(fifo.out)) if fifo.out in self.starves else "-"
            lines.append(f"{label:<56} {depth:>6}  {list(self.peaks(label))}, {stalls}, {starves}")
        lines += [str(finding) for finding in self.findings]
        return "\n".join(lines)
