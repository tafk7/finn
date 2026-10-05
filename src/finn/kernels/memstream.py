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
  accepted on ``set_stream`` streams one whole set. The output presents one
  pass per index; the set stream is an ordinary stream reference input.
- Its AXI-Lite port is tied off: the contents are fixed at build time.

It is a stream's ``source`` candidate (``finn.kernels.channels``): placed by the
stream it drives, ``staged``, its output presents into that stream without a
reference to it, and its set port references the stream's ``index``.

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
from math import ceil

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
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, Signal, StandardProtocol
from finn.kernels.artifacts.contributions import Contribution, CopiedSource, GeneratedData
from finn.kernels.artifacts.module import Held
from finn.kernels.base import Clocking, Kernel
from finn.kernels.control import held_bus
from finn.kernels.datatypes.domains import Integer, set_index_dtype
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    INTEGER_VECTOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    IntegerVector,
    integer_range,
)
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import Platform

LANE = Index("lane")
"""The lanes of a stored word, one per lane of the consumer's form."""

_MEMSTREAM_RAM_STYLES = ("auto", "distributed", "block", "ultra")


class MemStreamKernel(Kernel):
    id = "finnlib.memstream_axi"
    version = 1
    rtl_module = "memstream_axi"

    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    form: Traversal = Param()
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR)
    sets: int = Param(default=1)
    # Where a parent places it: the stream it drives and the set-index stream
    # (several sets only); or, as a stream's source, placed by the stream (staged).
    # channels imports this module (a memory is a channel's source), so it is imported last;
    # the engine resolves these annotations when it collects the family.
    output_stream: channels.Channel = Param(required=False)
    set_stream: channels.Channel = Param(required=False)
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
        return ScalarEncoding.admit(self.dtype, (low, high))

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
        """Several sets take a set stream of indices, one a beat; a single set none."""
        placed = self.present(MemStreamKernel.set_stream)
        if self.sets < 2:
            if placed:
                return reject("memstream-set-stream", "a single set takes no set stream")
            return True
        if not placed:
            return reject("memstream-set-stream", "several sets take a set stream")
        if len(self.set_stream.tensor.shape) != 1:
            return reject("memstream-set-stream", "the set stream carries a vector of indices")
        return True

    admission = ConstraintGroup(geometry_supported, selected)

    @derived(semantics=INTEGER_VECTOR)
    def image(self) -> IntegerVector | Rejected:
        """Packed words, set after set, each set in the consumer's ``form``."""
        encoding = self.element
        values, sets = self.contents, self.sets
        groups = values if sets > 1 else (values,)
        if sets > 1 and (not isinstance(values, tuple) or len(values) != sets):
            return reject("memstream-values", f"contents must hold one operand per set ({sets})")
        try:
            words = tuple(
                word for group in groups for word in pack(self.form, group, encoding.bits)
            )
        except ValueError as error:
            return reject("memstream-values", str(error))
        return words

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
            associated_clock="clk",
            associated_reset="rst",
        )

    @derived
    def set_sequence(self) -> BeatSequence:
        """One index a beat, one beat per pass of the weights."""
        return BeatSequence(vector_major(self.set_stream.tensor.shape, 1))

    @derived
    def output_sequence(self) -> BeatSequence:
        """One set streams cyclically; several stream one pass per accepted index."""
        if self.sets > 1:
            return BeatSequence(self.form.repeated(self.set_stream.tensor.size))
        return BeatSequence(self.form, Repetition.CYCLIC)

    @derived
    def word_factors(self) -> dict[Index, int]:
        """The lanes of a word: the form's lanes, carried by an idle output too."""
        return {LANE: self.form.lanes}

    set = AxiStreamPort(
        name="set",
        endpoint=Endpoint.TARGET,
        stream=set_stream,
        sequence=set_sequence,
        dtype=set_dtype,
        signals=("s_axis_0_tdata", "s_axis_0_tvalid", "s_axis_0_tready"),
        clock="clk",
        reset="rst",
    )
    output = AxiStreamPort(
        name="m_axis_0",
        endpoint=Endpoint.INITIATOR,
        stream=output_stream,
        staged=staged,
        sequence=output_sequence,
        dtype=dtype,
        value_range=value_range,
        lanes=(LANE,),
        factors=word_factors,
        clock="clk",
        reset="rst",
    )

    @derived
    def clocking(self) -> Clocking:
        """``clk2x`` runs a pumped memory, and is held low otherwise."""
        pumped = self.pumped_memory
        return Clocking("clk", "rst", doubled="clk2x", doubling=pumped, active_low=False)

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

    def other_pins(self) -> tuple[Signal | Bus, ...]:
        return (self.config_bus,)

    def held(self) -> Held:
        """AXI-Lite, always: the contents are fixed at build time."""
        return held_bus(self.config_bus)


# Last: channels imports this module (see the annotations above).
from finn.kernels import channels  # noqa: E402

__all__ = ["MemStreamKernel"]
