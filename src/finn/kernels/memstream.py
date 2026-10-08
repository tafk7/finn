# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``memstream_axi``: a stored integer operand streamed in its consumer's order.

The consumer supplies the operand ``contents`` and the beat ``form`` it reads
them in; the kernel packs one image per set in that form into the memory, so
the consumer's order needs no adapter. Initial contents go through INIT_FILE, a generated data file
named by its contents, so they are part of the build identity. It owns the
values it streams, so it states their ``value_range`` on its output: its element is
``dtype`` over the minimum and maximum of every set's contents.

- With one set, the image streams cyclically.
- With ``sets`` > 1, ``contents`` holds one operand per set, and each index
  accepted on ``set_channel`` streams one whole set. The output presents one
  pass per index; the set channel is an ordinary channel reference input.
- Its AXI-Lite port is tied off: the contents are fixed at build time.

It is a channel's ``source`` candidate (``finn.kernels.channels``): placed by the
channel it drives, ``staged``, its output presents into that channel without a
reference to it, and its set port references the channel's ``index``.

``ram_style`` and ``pumped_memory`` are its choices. A pumped memory runs at
``ap_clk2x`` on half-width words and doubles the depth; its 2x clock pin is
driven by role, and tied low when unpumped. Each case states what it needs of
the ``platform``: ``ultra`` UltraRAM that takes initial contents (on Zynq
UltraScale+ an initialized UltraRAM is built as block RAM), a pumped memory the
doubled clock.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from math import ceil, prod

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    requires,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    ordinary_integer_bounds,
)
from finn.dataflow.schedule import Index
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import (
    BeatSequence,
    Repetition,
    Traversal,
    pack,
    vector_major,
)
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, Pin, StandardProtocol
from finn.kernels.artifacts.contributions import Contribution, CopiedSource, GeneratedData
from finn.kernels.artifacts.module import Held
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.control import held_bus
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import Platform
from finn.kernels.utilization import RESOURCES_SEMANTICS, Fit, Resources, memory
from finn.kernels.values.domains import Integer, admit_element, set_index_dtype
from finn.kernels.values.semantics import (
    INTEGER_TENSOR,
    INTEGER_VECTOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    IntegerVector,
    integer_range,
    integer_shape,
    row_major,
)

LANE = Index("lane")
"""The lanes of a stored word, one per lane of the consumer's form."""

_MEMSTREAM_RAM_STYLES = ("auto", "distributed", "block", "ultra")


def memstream_resources(
    *, sets: int, depth: int, width: int, ram_style: str, pumped_memory: bool
) -> Resources:
    """FinnLib ``memstream`` inside ``memstream_axi``: its table, SETS * DEPTH words of
    WIDTH bits in RAM_STYLE, one port (the configuration port shares the read address);
    pumped, twice the words of half the bits (``DEPTH_EFF``, ``WIDTH_EFF``). Its output
    stream (a seven-deep shift register and the output register) is a ``Fit`` over
    WIDTH, plus the read register a bit where LUTRAM leaves it in the fabric (block RAM
    and UltraRAM absorb it); characterised unpumped, a pumped memory's gearbox is not
    in it."""
    words, bits = (2 * depth, (width + 1) // 2) if pumped_memory else (depth, width)
    table = memory(sets * words, bits, ram_style, single_port=True)
    read_register = 0 if table.bram18 or table.uram else bits
    return table + Resources(
        lut=_MEMSTREAM_LUT.at(width), ff=_MEMSTREAM_FF.at(width) + read_register
    )


# Feature: WIDTH.
_MEMSTREAM_LUT = Fit(4.2, (1.026,))
_MEMSTREAM_FF = Fit(4.4, (2.028,))


class MemStreamKernel(Kernel):
    id = "finnlib.memstream_axi"
    version = 1
    rtl_module = "memstream_axi"

    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    form: Traversal = Param()
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR)
    sets: int = Param(default=1)
    # Where a parent places it: the channel it drives and the set-index channel
    # (several sets only); or, as a channel's source, placed by the channel (staged).
    # channels imports this module (a memory is a channel's source), so it is imported last;
    # the engine resolves these annotations when it collects the Space class.
    output_channel: channels.Channel = Param(required=False)
    set_channel: channels.Channel = Param(required=False)
    staged: bool = Param(default=False)
    platform: Platform = Param()
    ram_style: str = Decision(
        values=_MEMSTREAM_RAM_STYLES,
        requires=(
            requires(platform.uram, "uram-absent: the platform has no UltraRAM", cases=("ultra",)),
            requires(
                platform.uram_init,
                "uram-init: the platform's UltraRAM takes no initial contents",
                cases=("ultra",),
            ),
        ),
    )
    pumped_memory: bool = Decision(
        values=(False, True),
        requires=(
            requires(
                platform.clk2x, "clk2x-absent: the platform has no doubled clock", cases=(True,)
            ),
        ),
    )

    @derived
    def value_range(self) -> tuple[int, ...] | Rejected:
        """The minimum and maximum of its contents, every set."""
        least, greatest = integer_range(self.contents)
        low, high = ordinary_integer_bounds(self.dtype)
        if not low <= least <= greatest <= high:
            return reject(
                "memstream-values", f"every value must be an integer admitted by {self.dtype.name}"
            )
        return (least, greatest)

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        """The stored element: an ordinary integer encoding over its contents' range."""
        admitted = Integer().check(self.dtype)
        if isinstance(admitted, Rejected):
            return admitted
        low, high = self.value_range
        return admit_element(self.dtype, (low, high))

    @derived
    def word_bits(self) -> int:
        return self.form.lanes * self.element.bits

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def set_dtype(self) -> QONNXDataType:
        return set_index_dtype(self.sets)

    @derived
    def address_bits(self) -> int:
        # $clog2(SETS * DEPTH * 2**$clog2(ceil(WIDTH/32))) + 2
        segments = 1 << (ceil(self.word_bits / 32) - 1).bit_length()
        return (self.sets * self.form.beats * segments - 1).bit_length() + 2

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        if not 1 <= self.sets <= 0xFFFFFFFF:
            return reject("memstream-sets", "SETS must be a positive native unsigned integer")
        if self.pumped_memory and self.word_bits < 2:
            return reject("memstream-pumping", "a pumped memory splits words of at least 2 bits")
        return True

    @constraint
    def selected(self) -> bool | Rejected:
        """Several sets take a set channel of indices, one a beat; a single set none."""
        placed = self.present(MemStreamKernel.set_channel)
        if self.sets < 2:
            if placed:
                return reject("memstream-set-channel", "a single set takes no set channel")
            return True
        if not placed:
            return reject("memstream-set-channel", "several sets take a set channel")
        if len(self.set_channel.tensor.shape) != 1:
            return reject("memstream-set-channel", "the set channel carries a vector of indices")
        return True

    admission = ConstraintGroup(geometry_supported, selected)

    @derived(semantics=INTEGER_VECTOR)
    def image(self) -> IntegerVector | Rejected:
        """Packed words, set after set, each set in the consumer's ``form``."""
        encoding, sets, form = self.element, self.sets, self.form
        shape = (sets, *form.shape) if sets > 1 else form.shape
        if integer_shape(self.contents) != shape:
            return reject(
                "memstream-values",
                f"contents must be {shape}: "
                + (f"one operand per set ({sets}), each " if sets > 1 else "")
                + f"of the form's shape {form.shape}",
            )
        flat, size = row_major(self.contents), prod(form.shape)
        return tuple(
            word
            for start in range(0, sets * size, size)
            for word in pack(form, flat[start : start + size], encoding.bits)
        )

    @derived
    def init_file(self) -> GeneratedData:
        """``$readmemh`` contents: one word per line; a pumped memory stores each word
        as its low half, then its high half."""
        bits, image = self.word_bits, self.image
        if self.pumped_memory:
            half = (bits + 1) // 2
            words = tuple(
                part for word in image for part in (word & ((1 << half) - 1), word >> half)
            )
            bits = half
        else:
            words = tuple(image)
        digits = (bits + 3) // 4
        data = "".join(f"{word:0{digits}x}\n" for word in words).encode()
        name = f"memstream_{hashlib.sha256(data).hexdigest()[:16]}.dat"
        return GeneratedData(name, data)

    @derived
    def config_bus(self) -> Bus:
        """The AXI-Lite configuration port, present in every configuration."""
        address = self.address_bits
        return Bus(
            "s_axilite",
            StandardProtocol.AXILITE,
            tuple(
                Member(name, name, width)
                for name, width in (
                    ("awready", 1),
                    ("awvalid", 1),
                    ("awprot", 3),
                    ("awaddr", address),
                    ("wready", 1),
                    ("wvalid", 1),
                    ("wdata", 32),
                    ("wstrb", 4),
                    ("bready", 1),
                    ("bvalid", 1),
                    ("bresp", 2),
                    ("arready", 1),
                    ("arvalid", 1),
                    ("arprot", 3),
                    ("araddr", address),
                    ("rready", 1),
                    ("rvalid", 1),
                    ("rresp", 2),
                    ("rdata", 32),
                )
            ),
            associated_clock=NATIVE_CLOCKING.clock,
            associated_reset=NATIVE_CLOCKING.reset,
        )

    @derived
    def set_sequence(self) -> BeatSequence:
        """One index a beat, one beat per pass of the weights."""
        return BeatSequence(vector_major(self.set_channel.tensor.shape, 1))

    @derived
    def output_sequence(self) -> BeatSequence:
        """One set streams cyclically; several stream one pass per accepted index."""
        if self.sets > 1:
            return BeatSequence(self.form.repeated(self.set_channel.tensor.size))
        return BeatSequence(self.form, Repetition.CYCLIC)

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        return memstream_resources(
            sets=self.sets,
            depth=self.form.beats,
            width=self.word_bits,
            ram_style=self.ram_style,
            pumped_memory=self.pumped_memory,
        )

    @derived
    def frame_cycles(self) -> int:
        """Its output's beats a pass (a set's, with several), one a cycle at best: a
        memory states no schedule, its stream is its form."""
        return self.form.beats

    @derived
    def word_factors(self) -> dict[Index, int]:
        """The lanes of a word: the form's lanes, carried by an idle output too."""
        return {LANE: self.form.lanes}

    set = AxiStreamPort(
        name="set",
        endpoint=Endpoint.TARGET,
        channel=set_channel,
        sequence=set_sequence,
        dtype=set_dtype,
        signals=("s_axis_0_tdata", "s_axis_0_tvalid", "s_axis_0_tready"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    output = AxiStreamPort(
        name="m_axis_0",
        endpoint=Endpoint.INITIATOR,
        channel=output_channel,
        staged=staged,
        sequence=output_sequence,
        dtype=dtype,
        value_range=value_range,
        lanes=(LANE,),
        factors=word_factors,
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )

    @derived
    def clocking(self) -> Clocking:
        """``clk2x`` runs a pumped memory, and is held low otherwise."""
        pumped = self.pumped_memory
        return replace(NATIVE_CLOCKING, doubled="clk2x", doubling=pumped)

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "DEPTH": self.form.beats,
            "INIT_FILE": f'"{self.init_file.path}"',
            "PUMPED_MEMORY": int(self.pumped_memory),
            "RAM_STYLE": f'"{self.ram_style}"',
            "SETS": self.sets,
            "WIDTH": self.word_bits,
        }

    def sources(self) -> tuple[Contribution, ...]:
        return (
            CopiedSource("finnlib", "rtl/infra/axilite.sv", provides=("module:axilite",)),
            CopiedSource("finnlib", "rtl/infra/memstream.sv", provides=("module:memstream",)),
            CopiedSource(
                "finnlib",
                "rtl/infra/memstream_axi.sv",
                provides=("module:memstream_axi",),
                requires=("module:axilite", "module:memstream"),
            ),
            self.init_file,
        )

    def other_pins(self) -> tuple[Pin, ...]:
        return (self.config_bus,)

    def held(self) -> Held:
        """AXI-Lite, always: the contents are fixed at build time."""
        return held_bus(self.config_bus)


# Last: channels imports this module (see the annotations above).
from finn.kernels import channels  # noqa: E402

__all__ = ["MemStreamKernel", "memstream_resources"]
