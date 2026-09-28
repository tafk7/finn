# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib ``memstream_axi``: a stored integer operand streamed in its consumer's order.

As for ``CyclicDelivery``, the consumer supplies the operand ``values`` and the
beat ``form`` it reads them in; the kernel packs one image per set in that form
into the memory. Initial contents go through INIT_FILE, a generated data file
named by its contents, so they are part of the build identity.

- With one set, the image streams cyclically, like the ROM.
- With ``sets`` > 1, ``values`` holds one operand per set, and each index
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
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Data,
    Derived as DerivedRate,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.contribution_types import CopiedSource, GeneratedData
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.base import Kernel
from finn.kernels.control import CONTROL, CONTROL_SEMANTICS, Control, ControlBus
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    INTEGER_VECTOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    IntegerVector,
)
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.dataflow.traversal import TRAVERSAL, Repetition, Traversal, pack, vector_major
from finn.kernels.physical.stream import ReadyValidStream
from finn.kernels.streams import MODULE, PORT, TIEOFFS, TIEOFFS_SEMANTICS, Stream, Tieoffs

MEMSTREAM_RAM_STYLES = ("auto", "distributed", "block", "ultra")


class MemStreamKernel(Kernel):
    id = "finnlib.memstream_axi"
    version = "1"

    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    element = integer_scalar(dtype, Integer())
    form: Traversal = Param(semantics=TRAVERSAL)
    values: IntegerTensor = Param(semantics=INTEGER_TENSOR)
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

    @derived(semantics=INTEGER_VECTOR)
    def image(self) -> IntegerVector | Rejected:
        """Packed words, set after set, each set in the consumer's ``form``."""
        encoding = self.element.encoding
        low, high = ordinary_integer_bounds(encoding.dtype)
        values, sets = self.values, self.sets
        groups = values if sets > 1 else (values,)
        if sets > 1 and (not isinstance(values, tuple) or len(values) != sets):
            return reject("memstream-values", f"values must hold one operand per set ({sets})")
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

    @derived(semantics=default_semantics(AxiStream))
    def output_interface(self) -> AxiStream:
        return AxiStream(
            "m_axis_0", self.element.encoding.dtype, self.form.lanes, endpoint=Endpoint.INITIATOR
        )

    @derived(semantics=default_semantics(ReadyValidStream))
    def set_interface(self) -> ReadyValidStream:
        return ReadyValidStream(
            "set",
            self.set_bits,
            Endpoint.TARGET,
            "s_axis_0_tdata",
            "s_axis_0_tvalid",
            "s_axis_0_tready",
            "clk",
            "rst",
        )

    support = ConstraintGroup(geometry_supported)

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(support,))
    def build_requirements(self) -> ModuleBuildRequirements:
        init = self.init_file
        pumped = self.pumped_memory
        parameters = (
            ("DEPTH", self.form.beats),
            ("INIT_FILE", f'"{init.path}"'),
            ("PUMPED_MEMORY", int(pumped)),
            ("RAM_STYLE", f'"{self.ram_style}"'),
            ("SETS", self.sets),
            ("WIDTH", self.word_bits),
        )
        abi = ModuleABIRequirements(
            FixedModuleName("memstream_axi"),
            (
                Signal("clk", Direction.IN, 1, Clock(Free())),
                Signal(
                    "clk2x", Direction.IN, 1, Clock(DerivedRate("clk", 2)) if pumped else Data()
                ),
                Signal(
                    "rst",
                    Direction.IN,
                    1,
                    Reset(
                        active_low=False,
                        synchronous=True,
                        synchronous_to=("clk", "clk2x") if pumped else ("clk",),
                    ),
                ),
                self.config_bus,
                *self.set_interface.pins(),
                *self.output_interface.native(clock="clk", reset="rst").pins(),
            ),
            tuple((name, str(value)) for name, value in parameters),
            (ClockAlignment("clk", "clk2x"),) if pumped else (),
        )
        sources = (
            CopiedSource("finnlib", "rtl/infra/axilite.sv", provides=("module:axilite",)),
            CopiedSource("finnlib", "rtl/infra/memstream.sv", provides=("module:memstream",)),
            CopiedSource(
                "finnlib",
                "rtl/infra/memstream_axi.sv",
                provides=("module:memstream_axi",),
                requires=("module:axilite", "module:memstream"),
            ),
            init,
        )
        return ModuleBuildRequirements(
            MemStreamKernel.id, MemStreamKernel.version, parameters, abi, sources
        )

    @view(semantics=STREAM_CONTRACT)
    def output_port(self) -> StreamContract:
        """One set streams cyclically; several stream one pass per accepted index."""
        encoding = self.element.encoding
        transport = self.output_interface.native(clock="clk", reset="rst")
        if self.sets > 1:
            passes = self.set_stream.tensor.size
            return StreamContract(transport, encoding, self.form.repeated(passes))
        return StreamContract(transport, encoding, self.form, Repetition.CYCLIC)

    @view(semantics=STREAM_CONTRACT)
    def set_port(self) -> StreamContract | Rejected:
        """One index a beat, one beat per pass of the weights."""
        tensor = self.set_stream.tensor
        if self.sets < 2:
            return reject("memstream-set-stream", "a single set takes no set stream")
        index = resolve_qonnx_datatype_name(f"UINT{self.set_bits}")
        if tensor.element.datatype_name != index.name or len(tensor.shape) != 1:
            return reject(
                "memstream-set-stream", f"the set stream carries a vector of {index.name} indices"
            )
        return StreamContract(self.set_interface, tensor.element, vector_major(tensor.shape, 1))

    @view(semantics=CONTROL_SEMANTICS)
    def control_bus(self) -> Control:
        return Control(self.config_bus if self.writable else None)

    @view(semantics=TIEOFFS_SEMANTICS)
    def tieoffs(self) -> Tieoffs | Rejected:
        """Hold the unused interfaces idle: AXI-Lite unless writable, the set selector
        with a single set, and the 2x clock unless pumped."""
        inputs: list[tuple[str, int]] = []
        unused: list[str] = []
        if self.writable and not self.present(MemStreamKernel.control):
            return reject("memstream-control", "a runtime-writable memory needs a control bus")
        if not self.writable:
            directions = dict(self.config_bus.member_directions())
            for member in self.config_bus.signals:
                if directions[member.physical] is Direction.IN:
                    inputs.append((member.physical, 0))
                else:
                    unused.append(member.physical)
        if self.sets < 2:
            selector = self.set_interface
            inputs += [(selector.data, 0), (selector.valid, 0)]
            unused.append(selector.ready)
        if not self.pumped_memory:
            inputs.append(("clk2x", 0))
        return Tieoffs(tuple(inputs), tuple(unused))

    exports = {
        MODULE: build_requirements,
        PORT: {output_stream: output_port, set_stream: set_port},
        CONTROL: {control: control_bus},
        TIEOFFS: tieoffs,
    }


def _leaves(values: object) -> tuple[int, ...]:
    if type(values) is int:
        return (values,)
    assert isinstance(values, tuple)
    return tuple(leaf for item in values for leaf in _leaves(item))


__all__ = ["MEMSTREAM_RAM_STYLES", "MemStreamKernel"]
