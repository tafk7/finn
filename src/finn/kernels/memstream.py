# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``memstream_axi``: a stored integer operand streamed in its consumer's order.

The consumer supplies the operand ``contents`` and the beat ``form`` it reads
them in; the kernel packs one image per set in that form into the memory, so
the consumer's order needs no adapter. Initial contents go through INIT_FILE, a generated data file
named by its contents, so they are part of the build identity.

- With one set, the image streams cyclically.
- With ``sets`` > 1, ``contents`` holds one operand per set, and each index
  accepted on ``set_stream`` streams one whole set. The output presents one
  pass per index; the set stream is an ordinary stream reference input.
- ``writable`` exports the AXI-Lite port through ``control``, so software can
  rewrite the memory at run time; otherwise the port is tied off.

``ram_style`` and ``pumped_memory`` are its choices. A pumped memory runs at
``ap_clk2x`` on half-width words and doubles the depth; its 2x clock pin is
driven by role, and tied low when unpumped.
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
    default_semantics,
    derived,
    reject,
    view,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import (
    BEAT_SEQUENCE,
    TRAVERSAL,
    BeatSequence,
    Repetition,
    Traversal,
    pack,
    vector_major,
)
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, Signal, StandardProtocol
from finn.kernels.artifacts.contribution_types import CopiedSource, GeneratedData
from finn.kernels.artifacts.requirements import RequirementContribution
from finn.kernels.base import CLOCKING, Clocking, Kernel, Tieoffs
from finn.kernels.control import CONTROL, CONTROL_SEMANTICS, Control, ControlBus, held_bus
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    INTEGER_VECTOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    IntegerVector,
)
from finn.kernels.port import GivenPort
from finn.kernels.streams import Stream

MEMSTREAM_RAM_STYLES = ("auto", "distributed", "block", "ultra")


def stored_element(carried: ScalarEncoding, stored: ScalarEncoding) -> bool | Rejected:
    """A memory's output stream carries the element the memory stores."""
    if carried != stored:
        return reject(
            "memory-element",
            f"the stream carries {carried.datatype_name}, the memory stores {stored.datatype_name}",
        )
    return True


class MemStreamKernel(Kernel):
    id = "finnlib.memstream_axi"
    version = "1"
    module = "memstream_axi"

    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    element = integer_scalar(dtype, Integer())
    form: Traversal = Param(semantics=TRAVERSAL)
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR)
    sets: int = Param(default=1)
    writable: bool = Param(default=False)
    # Where a parent places it: the stream it drives, the set-index stream
    # (several sets only), and the control bus that exports a writable memory.
    output_stream: Stream = Param(required=False)
    set_stream: Stream = Param(required=False)
    control: ControlBus = Param(required=False)
    ram_style: str = Decision(values=MEMSTREAM_RAM_STYLES)
    pumped_memory: bool = Decision(values=(False, True))

    @derived
    def word_bits(self) -> int:
        return self.form.lanes * self.element.encoding.bits

    @derived
    def set_bits(self) -> int:
        # SET_BITS = SETS > 2 ? $clog2(SETS) : 1
        sets = self.sets
        return (sets - 1).bit_length() if sets > 2 else 1

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def set_dtype(self) -> QONNXDataType:
        return resolve_qonnx_datatype_name(f"UINT{self.set_bits}")

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
    def carried(self) -> bool | Rejected:
        """The stream it drives, when placed, carries the element it stores."""
        if not self.present(MemStreamKernel.output_stream):
            return True
        return stored_element(self.output_stream.tensor.element, self.element.encoding)

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
        tensor = self.set_stream.tensor
        index = self.set_dtype
        if tensor.element.datatype_name != index.name or len(tensor.shape) != 1:
            return reject(
                "memstream-set-stream", f"the set stream carries a vector of {index.name} indices"
            )
        return True

    admission = ConstraintGroup(geometry_supported, carried, selected)

    @derived(semantics=INTEGER_VECTOR)
    def image(self) -> IntegerVector | Rejected:
        """Packed words, set after set, each set in the consumer's ``form``."""
        encoding = self.element.encoding
        low, high = ordinary_integer_bounds(encoding.dtype)
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
        if any(not low <= value <= high for value in _leaves(values)):
            return reject(
                "memstream-values",
                f"every value must be an integer admitted by {encoding.datatype_name}",
            )
        return words

    @derived(semantics=default_semantics(GeneratedData))
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

    @derived(semantics=default_semantics(Bus))
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

    @derived(semantics=BEAT_SEQUENCE)
    def set_sequence(self) -> BeatSequence:
        """One index a beat, one beat per pass of the weights."""
        return BeatSequence(vector_major(self.set_stream.tensor.shape, 1))

    @derived(semantics=BEAT_SEQUENCE)
    def output_sequence(self) -> BeatSequence:
        """One set streams cyclically; several stream one pass per accepted index."""
        if self.sets > 1:
            return BeatSequence(self.form.repeated(self.set_stream.tensor.size))
        return BeatSequence(self.form, Repetition.CYCLIC)

    set = GivenPort(
        name="set",
        endpoint=Endpoint.TARGET,
        stream=set_stream,
        sequence=set_sequence,
        idle_dtype=set_dtype,
        signals=("s_axis_0_tdata", "s_axis_0_tvalid", "s_axis_0_tready"),
        clock="clk",
        reset="rst",
    )
    output = GivenPort(
        name="m_axis_0",
        endpoint=Endpoint.INITIATOR,
        stream=output_stream,
        sequence=output_sequence,
        idle_dtype=dtype,
        idle_lanes=output_sequence.form.lanes,
        clock="clk",
        reset="rst",
    )

    @derived(semantics=CLOCKING)
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

    def sources(self) -> tuple[RequirementContribution, ...]:
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

    def held(self) -> Tieoffs | Rejected:
        """AXI-Lite, unless writable."""
        if not self.writable:
            return held_bus(self.config_bus)
        if not self.present(MemStreamKernel.control):
            return reject("memstream-control", "a runtime-writable memory needs a control bus")
        return Tieoffs()

    @view(semantics=CONTROL_SEMANTICS)
    def control_bus(self) -> Control:
        return Control(self.config_bus if self.writable else None)

    exports = {**Kernel.exports, CONTROL: {control: control_bus}}


def _leaves(values: object) -> tuple[int, ...]:
    if type(values) is int:
        return (values,)
    assert isinstance(values, tuple)
    return tuple(leaf for item in values for leaf in _leaves(item))


__all__ = ["MEMSTREAM_RAM_STYLES", "MemStreamKernel", "stored_element"]
