# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The movers a shell offers on a core's edge: where a partition's stream meets memory.

A boundary channel of a partition's core has a side without a port, its **free side**: the
AXIS port the core presents (``Channel.free_side``). A shell with movers offers each such
port the mover it places there (``MoverOffer``, of a mover kind a platform declaration
names, ``finn.platform.declarations``), and the shell places it beside the core as a
member of its own (``finn.kernels.shell``), a kernel (``finn.kernels.iodma``): no choice.
A mover binds no RTL in the core's module: it is a unit of its own, which the shell
integrates beside the core's IP, and its facts are its contract (``MoverContract``).

**``IODMA_hls``** (kind ``iodma_hls``), the Zynq shell's memory mover, modelled at its
stream side, the free side's contract:

- it moves a frame between a flat buffer and the stream in order, a memory word after
  another, so the host's buffer holds the stream's beats as they come; the PYNQ driver
  fills and reads that buffer as the tensor's row-major array, one pass of it a
  frame. So the mover states its free side (``MoverOffer.states``, the boundary side's
  ``outside``): each pass row-major (``Traversal.row_major``), one pass a frame, the
  word (lanes a beat) left the inside's. The edge channel presents that side and
  places what joins it to the inside (``finn.kernels.channels``): an ``input_gen``
  after the transport where the inside writes or reads the pass in another order, or
  reads it more than once a frame (a streamed weight read once per row of its
  activations), its buffer the pass; what no chain of modules joins it refuses by its
  own code (``channel-steps``), naming the mover. The free side the mover moves is
  always one row-major pass a frame;

- its memory port is ``gcd(frame bits, cap)``, a frame's bits the stream's padded
  ``tdata`` times its beats, as ``InsertIODMA`` sizes ``intfWidth``: the offer's cap
  (``width_cap``) bounds the memory port, not the stream, whose ``tdata`` is the edge's
  word; a frame is ``words`` memory words, and a width converter sits between memory
  and stream when the two widths differ;
- its cycles a frame are ``max(beats, words) + ceil(c / frames a call)``: a beat or a
  word a cycle, and each call's constant ``c``, which the offer states with and
  without a converter (``iodma_hls``: measured in simulation against a memory model, 4
  cycles a call with a converter, 13 without; the call ends two converter words
  before the stream runs dry, or once its last word left);
- it presents one AXI-Lite bus to the shell's processor (its ``s_axi_control``),
  whose window the shell's rules place beside the partition's
  (``MoverContract.control_buses``), and on it the registers a runtime drives it by
  (``IODMA_REGISTERS``: its HLS block control at 0x00, the buffer's address at 0x10, a
  64-bit pointer, and the frames a call moves at 0x1C);
- the memory latency ``L`` is **excluded**: the shell does not know it, and each call
  adds it (``MEMORY_LATENCY``, what the exploration report states). A frame a call is
  the bound that holds for any batch, since the batch is the driver's runtime argument;
- its beat times over a frame (``MoverContract.times``), one a cycle, or once the memory
  word that completes the beat arrived where the memory is the narrower; FIFO sizing
  reads them, and its cycles as its span, as a kernel's port's pace, beside the boundary
  channel in its shell (``finn.kernels.fifo_sizing.sides``);
- it initiates one AXI memory port into the shell's memory interconnect
  (``MoverContract.memory_ports``), which the shell's static region scales by;
- its resources (``iodma_hls_resources``), by its widths: its memory port
  (``intfWidth``), its stream (``streamWidth``, the free side's ``tdata``) and the
  converters between them (``converter_kind``: none where the two are equal, one
  where either divides the other, two through their least common multiple
  otherwise). They are the shell's, not the core's: out of context
  (``SHELL_CHARACTERISED``), which overstates the placed mover.

The mover kernel states its contract (``IodmaKernel.contract``), its cycles a frame and its
resources as the shell's member, beside the core's, whose cycles no longer count it.
A mover is not a channel's stream (``finn.dataflow.streams``: the two beat sequences a
channel's producer and consumer present, ``Stream``, ``Streams``).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, gcd, lcm

from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import Registers
from finn.kernels.sides import ROW_MAJOR, Outside
from finn.kernels.transport import StreamPins
from finn.kernels.utilization import Fit
from finn.platform.capabilities import Resources

IODMA_HLS = "iodma_hls"
"""The kind of the ``IODMA_hls`` mover."""

PLACED_KINDS = (IODMA_HLS,)
"""The mover kinds a shell places (``finn.kernels.iodma``): FINN's own alone, until a mover
kind is placed by its declared code."""

MEMORY_LATENCY = "memory latency: unmeasured; each call adds L"
"""What a mover's cycles leave out, as the exploration report states it."""

IODMA_REGISTERS = Registers((("control", 0x00, 32), ("address", 0x10, 64), ("count", 0x1C, 32)))
"""The registers of ``IODMA_hls``'s control bus a runtime drives a call by: the HLS top's
block control (``control``), the buffer's address (``address``) and the frames it moves
(``count``), at the offsets its HLS export gives them."""

NO_CONVERTER, DIVISIBLE, LCM = "none", "divisible", "lcm"
"""The converters between a mover's memory port and its stream (``converter_kind``)."""


def converter_kind(memory_width: int, stream_width: int) -> str:
    """The converters ``IODMA_hls`` places between a memory port and a stream of these
    widths: none where they are equal; one where either divides the other; otherwise
    two, through their least common multiple."""
    if memory_width == stream_width:
        return NO_CONVERTER
    wide, narrow = max(memory_width, stream_width), min(memory_width, stream_width)
    return DIVISIBLE if wide % narrow == 0 else LCM


# ``IODMA_hls`` out of context (``SHELL_CHARACTERISED``; xczu3eg-sbva484-1-e at 5 ns,
# memory ports of 64 to 512 bits, streams of 8, 32, 80 and 128 bits). With no converter
# or one: features the memory port and the converter's wider side (none without one);
# within 7.3 % of every configuration. Through the least common multiple: features the
# memory port and that multiple, which dominates (up to 2.2 times the cost of a
# divisible stream); within 3.1 %, characterised on 80-bit streams only.
IODMA_LEAST_PORT = 64
"""The narrowest memory port characterised; a narrower one is stated at it (a 16-bit
output mover measured above the 64-bit one)."""
IODMA_IN_LUT = Fit(985.5, (0.854, 1.644))
IODMA_IN_FF = Fit(1376.4, (4.191, 2.860))
IODMA_IN_LCM_LUT = Fit(1247.9, (0.270, 1.351))
IODMA_IN_LCM_FF = Fit(1671.2, (0.963, 4.814))
IODMA_OUT_LUT = Fit(1218.4, (0.451, 2.227))
IODMA_OUT_FF = Fit(1606.8, (4.047, 3.179))
IODMA_OUT_LCM_LUT = Fit(1433.3, (0.296, 1.481))
IODMA_OUT_LCM_FF = Fit(1879.5, (1.010, 5.048))
IODMA_IN_BUFFER_BITS = 36
"""The input mover's read buffer: a RAMB18 for each 36 bits of its memory port (1, 2, 4
and 7.5 RAMB36 at 64, 128, 256 and 512 bits, at every stream width); the output mover
has none."""


def iodma_hls_resources(*, direction: str, intf_width: int, stream_width: int) -> Resources:
    """What ``IODMA_hls`` uses, moving memory to a stream (``direction`` ``in``) or a
    stream to memory (``out``), by its memory port (``intf_width``) and its stream
    (``stream_width``), bits: out of context (``SHELL_CHARACTERISED``); a model, its
    converters' kind (``converter_kind``) chosen as the HLS source chooses them."""
    if direction not in ("in", "out"):
        raise ValueError(f"an IODMA moves in or out, not {direction!r}")
    kind = converter_kind(intf_width, stream_width)
    port = max(intf_width, IODMA_LEAST_PORT)
    if kind == LCM:
        lut, ff = (
            (IODMA_IN_LCM_LUT, IODMA_IN_LCM_FF)
            if direction == "in"
            else (IODMA_OUT_LCM_LUT, IODMA_OUT_LCM_FF)
        )
        features = (port, lcm(intf_width, stream_width))
    else:
        lut, ff = (
            (IODMA_IN_LUT, IODMA_IN_FF) if direction == "in" else (IODMA_OUT_LUT, IODMA_OUT_FF)
        )
        features = (port, 0 if kind == NO_CONVERTER else max(port, stream_width))
    buffer = ceil(port / IODMA_IN_BUFFER_BITS) if direction == "in" else 0
    return Resources(lut=lut.at(*features), ff=ff.at(*features), bram18=buffer)


@dataclass(frozen=True, kw_only=True)
class MoverOffer:
    """What a shell offers a boundary channel's free side: a mover's ``kind`` (a declared
    mover kind's name), its memory port's widest (``width_cap``, bits: it bounds the mover's
    memory port, never the stream, whose word is the edge's), the frames each call moves
    (``frames_per_call``), and the cycles each call adds beside the memory latency, with
    a width converter (``call_converted``) and without (``call_direct``). What the mover
    states of the side it is offered on is ``states``."""

    kind: str
    width_cap: int
    frames_per_call: int
    call_converted: int
    call_direct: int

    def __post_init__(self) -> None:
        if self.width_cap < 8 or self.width_cap % 8:
            raise ValueError(f"a memory port of {self.width_cap} bits is no whole bytes")
        if self.frames_per_call < 1:
            raise ValueError(f"{self.frames_per_call} frames a call is none")
        if self.call_converted < 0 or self.call_direct < 0:
            raise ValueError("a call adds no fewer than zero cycles")

    def states(self) -> Outside:
        """What the mover states of the boundary side it is offered on, the one place it
        is stated (``finn.kernels.sides.Outside``, its source the mover's kind): each pass
        row-major, one pass a frame, as the host's buffer holds the tensor. It states no
        word: the edge's lanes stay the inside's, and the mover, generated for that word,
        converts to its memory port itself. It states no pace either: its pace is its
        contract at the edge's word (``finn.kernels.iodma.IodmaKernel.contract``)."""
        return Outside(self.kind, order=ROW_MAJOR, passes=1)


def iodma_hls(width_cap: int, *, frames_per_call: int = 1) -> MoverOffer:
    """The ``IODMA_hls`` mover, its memory port at most ``width_cap`` bits, with its call
    constants as measured: 4 cycles a call with a width converter, 13 without."""
    return MoverOffer(
        kind=IODMA_HLS,
        width_cap=width_cap,
        frames_per_call=frames_per_call,
        call_converted=4,
        call_direct=13,
    )


@dataclass(frozen=True, kw_only=True)
class MoverContract:
    """A mover's facts, as the shell integrates it and an exploration costs it:

    - ``kind``; ``direction``: ``in`` (memory to the module's input) or ``out``;
    - the stream side, the free side's contract: ``port`` (its AXIS port), ``tdata``
      (bits), ``beats`` a frame, ``lanes`` a beat and ``element`` (the channel's);
    - the memory side: ``memory_width`` (bits), ``words`` a frame, and whether a width
      ``converter`` sits between the two;
    - the rate: ``call_cycles`` (the constant each call adds), ``frames_per_call`` and
      ``cycles`` a frame, the memory latency excluded (``MEMORY_LATENCY``);
    - ``control_buses``: the AXI-Lite buses it presents to the shell's processor, and
      ``registers``, those a runtime drives it by on its own (``Registers``);
    - ``memory_ports``: the AXI memory ports it initiates into the shell's memory;
    - ``resources``: what it uses of the device, the shell's (``SHELL_CHARACTERISED``)."""

    kind: str
    direction: str
    port: str
    tdata: int
    beats: int
    lanes: int
    element: ScalarEncoding
    memory_width: int
    words: int
    converter: bool
    call_cycles: int
    frames_per_call: int
    control_buses: int
    registers: Registers
    memory_ports: int
    resources: Resources

    @property
    def converter_kind(self) -> str:
        """The converters between its memory port and its stream (``converter_kind``)."""
        return converter_kind(self.memory_width, self.tdata)

    def cycles_at(self, frames_per_call: int) -> int:
        """Its cycles a frame were each call to move ``frames_per_call`` frames."""
        return max(self.beats, self.words) + ceil(self.call_cycles / frames_per_call)

    @property
    def cycles(self) -> int:
        """Its cycles a frame at its offer's frames a call."""
        return self.cycles_at(self.frames_per_call)

    @property
    def span(self) -> int:
        """The cycles its frame spans: its ``cycles``, the span its ``times`` lie within,
        as FIFO sizing reads them (``finn.kernels.fifo_sizing.pattern``)."""
        return self.cycles

    @property
    def times(self) -> tuple[int, ...]:
        """Each stream beat's cycle from the frame's first: one a cycle, or once the
        memory word that completes it arrived (a word a cycle). Its own: FIFO sizing reads
        them with its ``span`` (``finn.kernels.fifo_sizing.pattern``)."""
        beats, words = self.beats, self.words
        return tuple(max(beat, ceil((beat + 1) * words / beats) - 1) for beat in range(beats))


def iodma_contract(offer: MoverOffer, side: StreamPins) -> MoverContract:
    """``IODMA_hls``'s facts on a free side (``side``, the module's port it serves), sized by
    ``offer`` (see the module docstring)."""
    tdata, beats = side.transport.data_width, side.form.beats
    memory = gcd(tdata * beats, offer.width_cap)
    converter = memory != tdata
    # Seen from inside the module, its input is the AXIS target.
    direction = "in" if side.transport.endpoint is Endpoint.TARGET else "out"
    return MoverContract(
        kind=IODMA_HLS,
        direction=direction,
        port=side.transport.name,
        tdata=tdata,
        beats=beats,
        lanes=side.lanes,
        element=side.element,
        memory_width=memory,
        words=tdata * beats // memory,
        converter=converter,
        call_cycles=offer.call_converted if converter else offer.call_direct,
        frames_per_call=offer.frames_per_call,
        control_buses=1,
        registers=IODMA_REGISTERS,
        memory_ports=1,
        resources=iodma_hls_resources(direction=direction, intf_width=memory, stream_width=tdata),
    )


__all__ = [
    "DIVISIBLE",
    "IODMA_HLS",
    "IODMA_REGISTERS",
    "LCM",
    "MEMORY_LATENCY",
    "NO_CONVERTER",
    "PLACED_KINDS",
    "MoverContract",
    "MoverOffer",
    "converter_kind",
    "iodma_contract",
    "iodma_hls",
    "iodma_hls_resources",
]
