# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``IODMA_hls`` as a kernel: the memory mover on an edge of a partition's core, a leaf
whose module is an HLS top, exported into the IP catalog as FINN's IP.

``IodmaKernel`` is the mover the shell places on each edge of the core: ``IodmaIn``
supplies a core's input from memory, ``IodmaOut`` delivers its output to memory. Its facts
are its contract on that edge (``finn.kernels.movers.iodma_contract``): its rate a frame,
its resources, its control bus and its memory port. The shell places it
(``finn.kernels.shell``), where its row offers the ``iodma_hls`` mover kind, whose
declaration's code (``IodmaHls``) makes the offer at the row's width cap. Its module is the
``IODMA_hls`` top the HWCustomOp flow generated (``iodma_top``), named by its request's
digest (``finn_iodma_<key16>``), synthesized over finn-hlslib (``IODMA_HEADERS``, the
header root ``hlslib``) and exported under FINN's vendor and library.

Its stream port presents the edge's stream as it is, so that it joins the core's port
without a channel: its word the edge's, its tensor and beat sequence the edge's.
"""

from __future__ import annotations

from math import lcm
from typing import ClassVar

from finn.core.space import Param, Rejected, constraint, derived, reject, required, view
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import (
    Bus,
    Clocking,
    Direction,
    Endpoint,
    Member,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.contributions import Contribution, CopiedSource, HlsSource
from finn.kernels.artifacts.module import Held, RegisterMap
from finn.kernels.base import Kernel
from finn.kernels.control import ControlModel
from finn.kernels.laws import Axis, AxisSpelling, Standing, StatedRate
from finn.kernels.library import ModuleSpec
from finn.kernels.movers import (
    DIVISIBLE,
    IODMA_HLS,
    NO_CONVERTER,
    MoverContract,
    MoverOffer,
    converter_kind,
    iodma_contract,
    iodma_hls,
)
from finn.kernels.port_model import InPort, MemoryPort, OtherPins, OutPort
from finn.kernels.transport import StreamPins
from finn.platform.capabilities import Platform, Resources

#: The stem of its HLS top: the module is ``finn_iodma_<key16>``.
IODMA_STEM = "finn_iodma"

#: finn-hlslib's headers its top includes, ``bnn-library.h``'s quoted includes closed over.
IODMA_HEADERS = tuple(
    CopiedSource("hlslib", path)
    for path in (
        "bnn-library.h",
        "convlayer.h",
        "dma.h",
        "interpret.hpp",
        "mac.hpp",
        "maxpool.h",
        "mmv.hpp",
        "mvau.hpp",
        "slidingwindow.h",
        "streamtools.h",
        "tmrcheck.hpp",
        "upsample.hpp",
        "utils.hpp",
        "vvau.hpp",
        "weights.hpp",
    )
)

#: Its synthesis's directives: those the HWCustomOp flow's ``IODMA_hls`` ran.
IODMA_DIRECTIVES = (
    "set_param hls.enable_hidden_option_error false",
    "config_compile -disable_unroll_code_size_check -pipeline_style flp",
    "config_interface -m_axi_addr64",
    "config_rtl -module_auto_prefix",
    "config_rtl -deadlock_detection none",
)

#: Its control bus's address bits: its registers (control, interrupts, ``numReps`` and the
#: buffer's 64-bit address) span 64 bytes.
IODMA_CONTROL_ADDRESS = 6

_IODMA_RATE = (
    "IODMA_hls simulated against a memory model: a beat or a memory word a cycle, and "
    "each call's constant (4 cycles with a width converter, 13 without)"
)
_IODMA_REASON = (
    "its memory side's words and each call's constant take cycles its stream does not show"
)


def iodma_top(
    name: str, *, direction: str, memory_width: int, stream_width: int, frame_bits: int
) -> str:
    """The text of the HLS top ``name``: ``IODMA_hls`` moving a frame of ``frame_bits``
    between a memory port of ``memory_width`` bits and a stream of ``stream_width``, from
    memory to the stream (``direction`` ``in``) or back (``out``), the call's frame count its
    ``numReps``, through the converters ``converter_kind`` names. The HWCustomOp flow's
    ``IODMA_hls`` generated this text, byte for byte, its top named for its node."""
    if direction not in ("in", "out"):
        raise ValueError(f"an IODMA moves in or out, not {direction!r}")
    if frame_bits % 8:
        raise ValueError(f"a frame of {frame_bits} bits is no whole bytes")
    memory, stream = memory_width, stream_width
    wide = lcm(memory, stream)

    def dma(source: str, sink: str) -> str:
        function = "Mem2Stream_Batch" if direction == "in" else "Stream2Mem_Batch"
        return f"{function}<DataWidth1, NumBytes1>({source}, {sink}, numReps);"

    def dwc(into: int, out: int, source: str, sink: str) -> str:
        words = frame_bits // into
        return (
            f"StreamingDataWidthConverter_Batch<{into}, {out}, {words}>({source}, {sink}, numReps);"
        )

    kind = converter_kind(memory, stream)
    if direction == "in":
        signature = f"ap_uint<{memory}> *in0_V, hls::stream<ap_uint<{stream}> > &out0_V"
        ports = (
            "#pragma HLS INTERFACE m_axi offset=slave port=in0_V",
            "#pragma HLS INTERFACE s_axilite port=in0_V bundle=control",
            "#pragma HLS INTERFACE axis port=out0_V",
        )
        if kind == NO_CONVERTER:
            body: tuple[str, ...] = (dma("in0_V", "out0_V"),)
        elif kind == DIVISIBLE:
            body = (
                f"hls::stream<ap_uint<{memory}> > dma2dwc;",
                dma("in0_V", "dma2dwc"),
                dwc(memory, stream, "dma2dwc", "out0_V"),
            )
        else:
            body = (
                f"hls::stream<ap_uint<{memory}> > dma2lcm;",
                f"hls::stream<ap_uint<{wide}> > lcm2out;",
                dma("in0_V", "dma2lcm"),
                dwc(memory, wide, "dma2lcm", "lcm2out"),
                dwc(wide, stream, "lcm2out", "out0_V"),
            )
    else:
        signature = f"hls::stream<ap_uint<{stream}> > &in0_V, ap_uint<{memory}> *out0_V"
        ports = (
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE m_axi offset=slave port=out0_V",
            "#pragma HLS INTERFACE s_axilite port=out0_V bundle=control",
        )
        if kind == NO_CONVERTER:
            body = (dma("in0_V", "out0_V"),)
        elif kind == DIVISIBLE:
            body = (
                f"hls::stream<ap_uint<{memory}> > dwc2dma;",
                dwc(stream, memory, "in0_V", "dwc2dma"),
                dma("dwc2dma", "out0_V"),
            )
        else:
            body = (
                f"hls::stream<ap_uint<{wide}> > in2lcm;",
                f"hls::stream<ap_uint<{memory}> > lcm2dma;",
                dwc(stream, wide, "in0_V", "in2lcm"),
                dwc(wide, memory, "in2lcm", "lcm2dma"),
                dma("lcm2dma", "out0_V"),
            )
    lines = (
        f"void {name}({signature}, unsigned int numReps)",
        "{",
        "#pragma HLS INTERFACE s_axilite port=numReps bundle=control",
        "#pragma HLS INTERFACE s_axilite port=return bundle=control",
        *ports,
        "#pragma HLS DATAFLOW",
        *body,
        "}",
    )
    return (
        f"\n#define AP_INT_MAX_W {wide}\n\n"
        '#include "bnn-library.h"\n\n'
        '#include "dma.h"\n#include "streamtools.h"\n\n'
        f"#define NumBytes1 {frame_bits // 8}\n#define DataWidth1 {memory}\n\n\n"
        + "\n".join(lines)
        + "\n"
    )


def _axi_lite(name: str, address_bits: int, clocking: Clocking) -> Bus:
    """An HLS top's AXI-Lite target ``name``: its upper-case member pins, 32-bit data."""
    return Bus(
        name,
        StandardProtocol.AXILITE,
        tuple(
            Member(member.lower(), f"{name}_{member}", width)
            for member, width in (
                ("AWVALID", 1),
                ("AWREADY", 1),
                ("AWADDR", address_bits),
                ("WVALID", 1),
                ("WREADY", 1),
                ("WDATA", 32),
                ("WSTRB", 4),
                ("ARVALID", 1),
                ("ARREADY", 1),
                ("ARADDR", address_bits),
                ("RVALID", 1),
                ("RREADY", 1),
                ("RDATA", 32),
                ("RRESP", 2),
                ("BVALID", 1),
                ("BREADY", 1),
                ("BRESP", 2),
            )
        ),
        associated_clock=clocking.clock,
        associated_reset=clocking.reset,
    )


def _axi_memory(name: str, data_bits: int, clocking: Clocking) -> Bus:
    """An HLS top's AXI4 initiator ``name`` (``m_axi``, 64-bit addresses): its upper-case
    member pins, ``data_bits`` of data, one-bit IDs and user signals."""
    channel = (
        ("ADDR", 64),
        ("ID", 1),
        ("LEN", 8),
        ("SIZE", 3),
        ("BURST", 2),
        ("LOCK", 2),
        ("CACHE", 4),
        ("PROT", 3),
        ("QOS", 4),
        ("REGION", 4),
        ("USER", 1),
        ("VALID", 1),
        ("READY", 1),
    )
    members = (
        *((prefix + member, width) for prefix in ("AW", "AR") for member, width in channel),
        ("WDATA", data_bits),
        ("WSTRB", data_bits // 8),
        ("WLAST", 1),
        ("WID", 1),
        ("WUSER", 1),
        ("WVALID", 1),
        ("WREADY", 1),
        ("RDATA", data_bits),
        ("RRESP", 2),
        ("RLAST", 1),
        ("RID", 1),
        ("RUSER", 1),
        ("RVALID", 1),
        ("RREADY", 1),
        ("BRESP", 2),
        ("BID", 1),
        ("BUSER", 1),
        ("BVALID", 1),
        ("BREADY", 1),
    )
    return Bus(
        name,
        StandardProtocol.AXIMM,
        tuple(Member(member.lower(), f"{name}_{member}", width) for member, width in members),
        Endpoint.INITIATOR,
        associated_clock=clocking.clock,
        associated_reset=clocking.reset,
    )


class IodmaKernel(Kernel):
    """``IODMA_hls`` as a kernel: the mover on one edge of a partition's core (``edge``, the
    core's port it serves, its stream pins), where ``offer`` offers it, its HLS top
    generated for that edge's stream. Its two directions are its two kinds: ``IodmaIn``
    supplies the core's input from memory, ``IodmaOut`` delivers its output to memory.

    - Its one stream port presents the edge's stream (its tensor, its beat sequence, its
      word), so that it joins the core's port as it is; an edge with transport markers is
      refused (``iodma-markers``): its top sends and reads no ``TLAST``; so is an edge moving
      the other way (``iodma-direction``).
    - Its facts are its contract on the edge (``iodma_contract``): its rate a
      frame (``stated_rate``: its beat times within its cycles, measured), its resources
      (out of context, ``SHELL_CHARACTERISED``) and the one memory port it initiates
      (``memory_initiators``).
    - Its module is its HLS top (``iodma_top``, ``finn_iodma_<key16>``, exported into the IP
      catalog under FINN's identity) over finn-hlslib (the header root ``hlslib``): its
      control bus ``s_axi_control`` (AXI-Lite, presented, its writes the driver's at run
      time, none its configuration's), its memory port ``m_axi_gmem`` (AXI4, its data at
      least 32 bits, as HLS widens it), its ``interrupt`` left unconnected, and its stream,
      ``in0_V`` or ``out0_V``."""

    id = "finn.iodma_hls"

    offer: MoverOffer = Param()
    edge: StreamPins = Param()
    platform: Platform = Param()
    # Its kind's: ``in`` or ``out``, and its port's name.
    direction = required(str)
    port = required(str)

    @constraint
    def offered(self) -> bool | Rejected:
        """The offer is this kind's."""
        if self.offer.kind != IODMA_HLS:
            return reject("mover-not-offered", f"an {IODMA_HLS} mover, offered {self.offer.kind}")
        return True

    @constraint
    def unmarked(self) -> bool | Rejected:
        """The edge carries no transport marker: the top sends and reads no ``TLAST``."""
        if self.edge.markers:
            markers = ", ".join(signal for signal, _ in self.edge.markers)
            return reject(
                "iodma-markers",
                f"{IODMA_HLS} moves no transport marker; the edge carries {markers}",
            )
        return True

    @derived
    def contract(self) -> MoverContract:
        """Its facts on the edge (``iodma_contract``)."""
        return iodma_contract(self.offer, self.edge)

    @constraint
    def facing(self) -> bool | Rejected:
        """It moves the way its edge does: into the core's input (``IodmaIn``), out of its
        output (``IodmaOut``)."""
        contract: MoverContract = self.contract
        if contract.direction != self.direction:
            return reject(
                "iodma-direction",
                f"an IODMA moving {self.direction} serves no {contract.port}, which moves "
                f"{contract.direction}",
            )
        return True

    @derived
    def tensor(self) -> Tensor:
        """The tensor its port moves: the edge's."""
        edge: StreamPins = self.edge
        return Tensor(edge.form.shape, edge.element)

    @derived
    def sequence(self) -> BeatSequence:
        """What its port presents: the edge's beat sequence."""
        edge: StreamPins = self.edge
        return edge.sequence

    @derived
    def stated_rate(self) -> StatedRate:
        """Its port's beat times over a frame within its cycles (``MoverContract.times``,
        ``cycles``), as measured; the memory latency excluded (``MEMORY_LATENCY``). Its
        schedule and ports cannot express it (CONTRACT.md L6): its memory side's words and
        each call's constant take cycles its stream does not show."""
        contract: MoverContract = self.contract
        return StatedRate(
            self.port,
            contract.times,
            contract.cycles,
            Standing.MEASURED,
            _IODMA_RATE,
            _IODMA_REASON,
        )

    @derived
    def memory_initiators(self) -> int:
        """Its memory port."""
        return 1

    @derived
    def control(self) -> str:
        """The port its control bus is presented at: its own name."""
        return "s_axi_control"

    @view
    def register_map(self) -> RegisterMap:
        """None: its registers are written at run time, a call's frames and buffer."""
        return RegisterMap()

    @derived
    def control_bus(self) -> Bus:
        return _axi_lite("s_axi_control", IODMA_CONTROL_ADDRESS, self.clocking)

    @derived
    def memory_bus(self) -> Bus:
        contract: MoverContract = self.contract
        return _axi_memory("m_axi_gmem", max(contract.memory_width, 32), self.clocking)

    config = ControlModel(bus=control_bus, presented=True)
    memory = MemoryPort(bus=memory_bus)
    other = OtherPins(
        signals=(Signal("interrupt", Direction.OUT, 1),), holding=Held((), ("interrupt",))
    )

    @view
    def resource_use(self) -> Resources:
        """Its contract's, out of context (``SHELL_CHARACTERISED``)."""
        contract: MoverContract = self.contract
        return contract.resources

    @view
    def parameters(self) -> dict[str, int | str]:
        """None: the HLS top is written from the point's values."""
        return {}

    @view
    def module_sources(self) -> tuple[Contribution, ...]:
        """Its HLS request, exported as FINN's IP: its top written for its contract."""
        contract: MoverContract = self.contract
        direction, memory, stream = self.direction, contract.memory_width, contract.tdata
        frame_bits = stream * contract.beats
        return (
            HlsSource.request(
                IODMA_STEM,
                lambda name: iodma_top(
                    name,
                    direction=direction,
                    memory_width=memory,
                    stream_width=stream,
                    frame_bits=frame_bits,
                ),
                IODMA_HEADERS,
                self.platform.period_ns,
                IODMA_DIRECTIVES,
                export=True,
            ),
        )

    module_spec: ClassVar[ModuleSpec] = ModuleSpec(IODMA_STEM)


class IodmaIn(IodmaKernel):
    """``IODMA_hls`` from memory to the core's input (``edge``): its stream port, ``out0_V``,
    the edge's stream (``IodmaKernel``)."""

    id = "finn.iodma_hls.in"

    @derived
    def direction(self) -> str:
        return "in"

    @derived
    def port(self) -> str:
        return "output"

    output = OutPort(
        transport=Axis(bus="out0_V", spelling=AxisSpelling.UPPER),
        tensor=IodmaKernel.tensor,
        sequence=IodmaKernel.sequence,
    )


class IodmaOut(IodmaKernel):
    """``IODMA_hls`` from the core's output (``edge``) to memory: its stream port, ``in0_V``,
    the edge's stream (``IodmaKernel``)."""

    id = "finn.iodma_hls.out"

    @derived
    def direction(self) -> str:
        return "out"

    @derived
    def port(self) -> str:
        return "input"

    input = InPort(
        transport=Axis(bus="in0_V", spelling=AxisSpelling.UPPER),
        tensor=IodmaKernel.tensor,
        sequence=IodmaKernel.sequence,
    )


class IodmaHls:
    """The ``iodma_hls`` mover kind, as its declaration's code names it
    (``finn.platform.declarations``): its offer at a shell's width cap. The shell places
    ``IodmaIn`` and ``IodmaOut`` where its row offers it."""

    @staticmethod
    def offer(width_cap: int) -> MoverOffer:
        """Its offer on a shell whose memory port is at most ``width_cap`` bits wide
        (``iodma_hls``)."""
        return iodma_hls(width_cap)


__all__ = [
    "IODMA_CONTROL_ADDRESS",
    "IODMA_DIRECTIVES",
    "IODMA_HEADERS",
    "IODMA_STEM",
    "IodmaHls",
    "IodmaIn",
    "IodmaKernel",
    "IodmaOut",
    "iodma_top",
]
