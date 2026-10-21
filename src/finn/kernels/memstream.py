# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``memstream_axi``: a stored integer operand streamed in its consumer's order.

The consumer supplies the operand ``contents`` and the beat ``form`` it reads
them in; the kernel packs one image per set in that form into the memory, so
the consumer's order needs no adapter. Initial contents go through INIT_FILE, a generated data file
named by its contents, so they are part of the build identity. It owns the
values it streams, so it states their ``value_range`` in its output's tensor
(``output_tensor``, one set of the operand in the consumer's form): its element is
``dtype`` over the minimum and maximum of every set's contents.

- With one set, the image streams cyclically.
- With ``sets`` > 1, ``contents`` holds one operand per set, and each index
  accepted on its ``set`` port streams one whole set. The output presents one
  pass per index; the indices are a tensor (``set_tensor``), a fact like any
  other.
- Its AXI-Lite port is tied off: the contents are fixed at build time.

It is a channel's ``source`` candidate (``finn.kernels.channels``): placed by the
channel it drives, its output presents its given sequence, and its set port reads
the channel's ``index_tensor``. Placed by a parent, it states its ports' contracts
(``output_contract``, ``set_contract``), which the channels on them take. Its cycles are
the beats its output carries a frame, as every leaf's: one pass with one set, one pass
an index with several.

``ram_style`` and ``pumped_memory`` are its choices. A pumped memory runs at
``ap_clk2x`` on half-width words and doubles the depth; its 2x clock pin is
driven by role, and tied low when unpumped. The style's cases are the explicit
ones, ordered by the bits the table holds (``table_bits``, which no folding moves,
so the style is chosen before, with or after the folding), then ``auto``, which
states no resources (``finn.kernels.utilization.memory_styles``). ``ultra`` needs
UltraRAM of the ``platform`` that takes initial contents, two constraints of the kernel
(``uram-absent``, ``uram-init``; on Zynq UltraScale+ an initialized UltraRAM is built as
block RAM). A pumped memory takes the
doubled clock, which its module states and the shell admits (``clock-unavailable``).
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil, prod
from typing import TYPE_CHECKING, ClassVar

from qonnx.core.ranges import TensorRange

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    view,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    ordinary_integer_bounds,
    qonnx_datatype_width,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    BeatSequence,
    Repetition,
    Traversal,
    pack,
    vector_major,
)
from finn.kernels.artifacts.abi import Bus, Clocking, Endpoint, Member, StandardProtocol
from finn.kernels.artifacts.contributions import CopiedSource, GeneratedData, hex_image
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace, PortContract
from finn.kernels.control import ControlBus
from finn.kernels.library import LibraryModule
from finn.kernels.port import AxiStreamPort
from finn.kernels.utilization import (
    RESOURCES_SEMANTICS,
    Fit,
    auto_unstated,
    memory,
    memory_styles,
    uram_constraint,
    uram_init_constraint,
)
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
from finn.platform.capabilities import Fabric, Platform, Resources


def memstream_resources(
    *, sets: int, depth: int, width: int, ram_style: str, pumped_memory: bool, fabric: Fabric
) -> Resources:
    """FinnLib ``memstream`` inside ``memstream_axi``: its table, SETS * DEPTH words of
    WIDTH bits in RAM_STYLE on ``fabric``, one port (the configuration port shares the
    read address) and never written (the kernel holds its configuration port); pumped,
    twice the words of half the bits (``DEPTH_EFF``, ``WIDTH_EFF``). Its output stream
    (a seven-deep shift register and the output register) is a ``Fit`` over WIDTH, plus
    the read register a bit where LUTRAM leaves it in the fabric (block RAM and UltraRAM
    absorb it); characterised unpumped, a pumped memory's gearbox is not in it."""
    words, bits = (2 * depth, (width + 1) // 2) if pumped_memory else (depth, width)
    table = memory(sets * words, bits, ram_style, fabric=fabric, single_port=True, written=False)
    read_register = 0 if table.bram18 or table.uram else bits
    return table + Resources(
        lut=_MEMSTREAM_LUT.at(width), ff=_MEMSTREAM_FF.at(width) + read_register
    )


# Feature: WIDTH.
_MEMSTREAM_LUT = Fit(4.2, (1.026,))
_MEMSTREAM_FF = Fit(4.4, (2.028,))


class MemStreamKernel(KernelSpace):
    """A stored operand streamed in its consumer's order; see the module docstring."""

    id = "finnlib.memstream_axi"
    version = 1
    clock_domain = NATIVE_CLOCKING

    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    form: Traversal = Param()
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR)
    sets: int = Param(default=1)
    # The set indices its set port reads (several sets only): a fact it is given.
    set_tensor: Tensor = Param(required=False)
    platform: Platform = Param()

    @derived
    def table_bits(self) -> int | Rejected:
        """The bits the table holds, every set: its contents' elements at the datatype's
        bits (the stored element's), whatever the folding (which shapes the words, not
        their sum) and whether the contents are admitted (the image's to refuse)."""
        shape = integer_shape(self.contents)
        if shape is None:
            return reject("memstream-values", "contents must be an integer tensor")
        return prod(shape) * qonnx_datatype_width(self.dtype)

    ram_style: str = Decision(domain=memory_styles(table_bits))
    uram_present = uram_constraint(platform, ram_style, ("ultra",))
    uram_init_taken = uram_init_constraint(platform, ram_style, ("ultra",))
    # A pumped memory takes a doubled clock, which the shell admits.
    pumped_memory: bool = Decision(values=(False, True))

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
        return admit_element(self.dtype, TensorRange(low, high))

    @derived
    def output_tensor(self) -> Tensor:
        """What it streams: one set of the operand, in the consumer's form, over its
        contents' range."""
        return Tensor(self.form.shape, self.element)

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
        """Several sets take a tensor of set indices, one a beat; a single set none."""
        placed = self.present(MemStreamKernel.set_tensor)
        if self.sets < 2:
            if placed:
                return reject("memstream-set-channel", "a single set takes no set channel")
            return True
        if not placed:
            return reject("memstream-set-channel", "several sets take a set channel")
        if len(self.set_tensor.shape) != 1:
            return reject("memstream-set-channel", "the set channel carries a vector of indices")
        return True

    @view(semantics=INTEGER_VECTOR)
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

    @view
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
        return hex_image("memstream", words, bits)

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
        return BeatSequence(vector_major(self.set_tensor.shape, 1))

    @derived
    def output_sequence(self) -> BeatSequence:
        """One set streams cyclically; several stream one pass per accepted index."""
        if self.sets > 1:
            return BeatSequence(self.form.repeated(self.set_tensor.size))
        return BeatSequence(self.form, Repetition.CYCLIC)

    @view(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        if self.ram_style == "auto":
            return auto_unstated("the memstream's table")
        return memstream_resources(
            sets=self.sets,
            depth=self.form.beats,
            width=self.word_bits,
            ram_style=self.ram_style,
            pumped_memory=self.pumped_memory,
            fabric=self.platform.fabric,
        )

    set = AxiStreamPort(
        name="set",
        endpoint=Endpoint.TARGET,
        tensor=set_tensor,
        sequence=set_sequence,
        dtype=set_dtype,
        signals=("s_axis_0_tdata", "s_axis_0_tvalid", "s_axis_0_tready"),
    )
    output = AxiStreamPort(
        name="m_axis_0",
        endpoint=Endpoint.INITIATOR,
        tensor=output_tensor,
        sequence=output_sequence,
    )
    if TYPE_CHECKING:  # members its ports lower to, which its channel reads
        output_contract: ClassVar[PortContract]

    @derived
    def clocking(self) -> Clocking:
        """``clk2x`` runs a pumped memory, and is held low otherwise."""
        pumped = self.pumped_memory
        return replace(type(self).clock_domain, doubled="clk2x", doubling=pumped)

    @view
    def parameters(self) -> dict[str, int | str]:
        return {
            "DEPTH": self.form.beats,
            "INIT_FILE": f'"{self.init_file.path}"',
            "PUMPED_MEMORY": int(self.pumped_memory),
            "RAM_STYLE": f'"{self.ram_style}"',
            "SETS": self.sets,
            "WIDTH": self.word_bits,
        }

    # AXI-Lite, always held: the contents are fixed at build time.
    config = ControlBus(config_bus)

    # Named ``module`` once the transitional ``module`` view (``MODULE``) goes.
    library = LibraryModule(
        "memstream_axi",
        parameters=parameters,
        sources=(
            CopiedSource("finnlib", "rtl/infra/axilite.sv", provides=("module:axilite",)),
            CopiedSource("finnlib", "rtl/infra/memstream.sv", provides=("module:memstream",)),
            CopiedSource(
                "finnlib",
                "rtl/infra/memstream_axi.sv",
                provides=("module:memstream_axi",),
                requires=("module:axilite", "module:memstream"),
            ),
        ),
        data=init_file,
    )


__all__ = ["MemStreamKernel", "memstream_resources"]
